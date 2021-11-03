from __future__ import annotations

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Any, Callable, Dict, Union, Set, List, Hashable, Tuple, TypeVar
from dataclasses import dataclass


from pytato.transform import EdgeCachedMapper, CachedWalkMapper
from pytato.array import (
        Array, AbstractResultWithNamedArrays, Placeholder,
        DictOfNamedArrays, make_placeholder)

from pytato.target import BoundProgram


__doc__ = """
.. autoclass:: CodePartitions
.. autoexception:: PartitionInducedCycleError

.. autofunction:: find_partitions
.. autofunction:: execute_partitions
"""


ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]
T = TypeVar("T", Array, AbstractResultWithNamedArrays)
PartitionId = Hashable


# {{{ graph partitioner

class _GraphPartitioner(EdgeCachedMapper):
    """Given a function *get_partition_id*, produces subgraphs representing
    the computation. Users should not use this class directly, but use
    :meth:`find_partitions` instead.
    """

    # {{{ infrastructure

    def __init__(self, get_partition_id:
                                   Callable[[ArrayOrNames], PartitionId]) -> None:
        super().__init__()

        # Function to determine the Partition ID
        self._get_partition_id: Callable[[ArrayOrNames], PartitionId] = \
                get_partition_id

        # Naming for newly created PlaceHolders at partition edges
        from pytools import UniqueNameGenerator
        self.name_generator = UniqueNameGenerator(forced_prefix="_part_ph_")

        # "edges" of the partitioned graph, maps an edge between two partitions,
        # represented by a tuple of partition identifiers, to a set of placeholder
        # names "conveying" information across the edge.
        self.partition_pair_to_edges: Dict[Tuple[PartitionId, PartitionId],
                Set[str]] = {}

        self.var_name_to_result: Dict[str, Array] = {}

        self._seen_node_to_placeholder: Dict[ArrayOrNames, Placeholder] = {}

        # Reading the seen partition IDs out of partition_pair_to_edges is incorrect:
        # e.g. if each partition is self-contained, no edges would appear. Instead,
        # we remember each partition ID we see below, to guarantee that we don't
        # miss any of them.
        self.seen_partition_ids: Set[PartitionId] = set()

    def get_partition_id(self, expr: ArrayOrNames) -> PartitionId:
        part_id = self._get_partition_id(expr)
        self.seen_partition_ids.add(part_id)
        return part_id

    def does_edge_cross_partition_boundary(self,
            node1: ArrayOrNames, node2: ArrayOrNames) -> bool:
        return self.get_partition_id(node1) != self.get_partition_id(node2)

    def make_new_placeholder_name(self) -> str:
        return self.name_generator()

    def add_interpartition_edge(self, target: ArrayOrNames, dependency: ArrayOrNames,
                                placeholder_name: str) -> None:
        pid_target = self.get_partition_id(target)
        pid_dependency = self.get_partition_id(dependency)

        self.partition_pair_to_edges.setdefault(
                (pid_target, pid_dependency), set()).add(placeholder_name)

    def handle_edge(self, expr: ArrayOrNames, child: ArrayOrNames) -> Any:
        if self.does_edge_cross_partition_boundary(expr, child):
            try:
                ph = self._seen_node_to_placeholder[child]
            except KeyError:
                ph_name = self.make_new_placeholder_name()
                # If an edge crosses a partition boundary, replace the
                # depended-upon node (that nominally lives in the other partition)
                # with a Placeholder that lives in the current partition. For each
                # partition, collect the placeholder names that it’s supposed to
                # compute.

                if not isinstance(child, Array):
                    raise NotImplementedError("not currently supporting "
                            "DictOfNamedArrays in the middle of graph "
                            "partitioning")

                ph = make_placeholder(ph_name,
                    shape=child.shape,
                    dtype=child.dtype,
                    tags=child.tags)

                self.var_name_to_result[ph_name] = self.rec(child)

                self._seen_node_to_placeholder[child] = ph

            assert ph.name
            self.add_interpartition_edge(expr, child, ph.name)
            return ph

        else:
            return self.rec(child)

    def __call__(self, expr: T, *args: Any, **kwargs: Any) -> Any:
        # Need to make sure the first node's partition is 'seen'
        self.get_partition_id(expr)

        return super().__call__(expr, *args, **kwargs)

    # }}}

# }}}


# {{{ code partitions

@dataclass
class CodePartitions:
    """Store information about generated partitions.

    .. attribute:: toposorted_partitions

       List of topologically sorted partitions, represented by their
       identifiers.

    .. attribute:: partition_id_to_input_names

       Mapping of partition identifiers to names of placeholders
       the partition requires as input.

    .. attribute:: partition_id_to_output_names

       Mapping of partition IDs to the names of placeholders
       they provide as output.

    .. attribute:: var_name_to_result

       Mapping of placeholder names to their respective :class:`pytato.array.Array`
       they represent.
    """
    toposorted_partitions: List[PartitionId]
    partition_id_to_input_names: Dict[PartitionId, Set[str]]
    partition_id_to_output_names: Dict[PartitionId, Set[str]]
    var_name_to_result: Dict[str, Array]

# }}}


class PartitionInducedCycleError(Exception):
    """Raised by :func:`find_partitions` if the partitioning induced a
    cycle in the graph of partitions.
    """


# {{{ find_partitions

def find_partitions(outputs: DictOfNamedArrays,
        part_func: Callable[[ArrayOrNames], PartitionId]) ->\
        CodePartitions:
    """Partitions the *expr* according to *part_func* and generates code for
    each partition. Raises :exc:`PartitionInducedCycleError` if the partitioning
    induces a cycle, e.g. for a graph like the following::

           ┌───┐
        ┌──┤ A ├──┐
        │  └───┘  │
        │       ┌─▼─┐
        │       │ B │
        │       └─┬─┘
        │  ┌───┐  │
        └─►│ C │◄─┘
           └───┘

    where ``A`` and ``C`` are in partition 1, and ``B`` is in partition 2.

    :param expr: The expression to partition.
    :param part_func: A callable that returns an instance of
        :class:`Hashable` for a node.
    :returns: An instance of :class:`CodePartitions` that contains the partitions.
    """

    pf = _GraphPartitioner(part_func)
    rewritten_outputs = {name: pf(expr) for name, expr in outputs._data.items()}

    partition_id_to_output_names: Dict[PartitionId, Set[str]] = {
        pid: set() for pid in pf.seen_partition_ids}
    partition_id_to_input_names: Dict[PartitionId, Set[str]] = {
        pid: set() for pid in pf.seen_partition_ids}

    partitions = set()

    var_name_to_result = pf.var_name_to_result.copy()

    for out_name, rewritten_output in rewritten_outputs.items():
        out_part_id = part_func(outputs._data[out_name])
        partition_id_to_output_names.setdefault(out_part_id, set()).add(out_name)
        var_name_to_result[out_name] = rewritten_output

    # Mapping of nodes to their successors; used to compute the topological order
    partition_nodes_to_targets: Dict[PartitionId, List[PartitionId]] = {
            pid: [] for pid in pf.seen_partition_ids}

    for (pid_target, pid_dependency), var_names in \
            pf.partition_pair_to_edges.items():
        partitions.add(pid_target)
        partitions.add(pid_dependency)

        partition_nodes_to_targets[pid_dependency].append(pid_target)

        for var_name in var_names:
            partition_id_to_output_names[pid_dependency].add(var_name)
            partition_id_to_input_names[pid_target].add(var_name)

    from pytools.graph import compute_topological_order, CycleError
    try:
        toposorted_partitions = compute_topological_order(partition_nodes_to_targets)
    except CycleError:
        raise PartitionInducedCycleError

    result = CodePartitions(toposorted_partitions, partition_id_to_input_names,
                          partition_id_to_output_names, var_name_to_result)

    if __debug__:
        _check_partition_disjointness(result)

    return result


class _SeenNodesWalkMapper(CachedWalkMapper):
    def __init__(self) -> None:
        super().__init__()
        self.seen_nodes: Set[ArrayOrNames] = set()

    def visit(self, expr: ArrayOrNames) -> bool:
        super().visit(expr)
        self.seen_nodes.add(expr)
        return True


def _check_partition_disjointness(parts: CodePartitions) -> None:
    part_id_to_nodes: Dict[PartitionId, Set[ArrayOrNames]] = {}

    for part_id, out_names in parts.partition_id_to_output_names.items():

        mapper = _SeenNodesWalkMapper()
        for out_name in out_names:
            mapper(parts.var_name_to_result[out_name])

        # FIXME This check won't do much unless we successfully visit
        # all the nodes, but we're not currently checking that.
        for my_node in mapper.seen_nodes:
            for other_part_id, other_node_set in part_id_to_nodes.items():
                # Placeholders represent values computed in one partition
                # and used in one or more other ones. As a result, the
                # same placeholder may occur in more than one partition.
                assert (
                    isinstance(my_node, Placeholder)
                    or my_node not in other_node_set), (
                        "partitions not disjoint: "
                        f"{my_node.__class__.__name__} (id={id(my_node)}) "
                        f"in both '{part_id}' and '{other_part_id}'")

        part_id_to_nodes[part_id] = mapper.seen_nodes

# }}}


# {{{ generate_code_for_partitions

def generate_code_for_partitions(parts: CodePartitions) \
        -> Dict[PartitionId, BoundProgram]:
    """Return a mapping of partition identifiers to their
       :class:`pytato.target.BoundProgram`."""
    from pytato import generate_loopy
    prg_per_partition = {}

    from pytato.distributed import _DistributedCommReplacer
    comm_replacer = _DistributedCommReplacer()

    for pid in parts.toposorted_partitions:
        d = DictOfNamedArrays(
                    {var_name: comm_replacer(parts.var_name_to_result[var_name])
                        for var_name in parts.partition_id_to_output_names[pid]
                     })
        prg_per_partition[pid] = generate_loopy(d)

    return prg_per_partition

# }}}


# {{{ execute_partitions

def execute_partitions(parts: CodePartitions, prg_per_partition:
        Dict[PartitionId, BoundProgram], queue: Any) -> Dict[str, Any]:
    """Executes a set of partitions on a :class:`pyopencl.CommandQueue`.

    :param parts: An instance of :class:`CodePartitions` representing the
        partitioned code.
    :param queue: An instance of :class:`pyopencl.CommandQueue` to execute the
        code on.
    :returns: A dictionary of variable names mapped to their values.
    """
    context: Dict[str, Any] = {}
    for pid in parts.toposorted_partitions:
        inputs = {
            k: context[k] for k in parts.partition_id_to_input_names[pid]
            if k in context}

        _evt, result_dict = prg_per_partition[pid](queue=queue, **inputs)
        context.update(result_dict)

    return context

# }}}


# vim: foldmethod=marker
