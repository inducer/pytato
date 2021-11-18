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

from typing import (Any, Callable, Dict, Union, Set, List, Hashable, Tuple, TypeVar,
        FrozenSet, Mapping, TYPE_CHECKING)
from dataclasses import dataclass


from pytato.transform import EdgeCachedMapper, CachedWalkMapper
from pytato.array import (
        Array, AbstractResultWithNamedArrays, Placeholder,
        DictOfNamedArrays, make_placeholder)

from pytato.target import BoundProgram

if TYPE_CHECKING:
    from pytato.distributed import DistributedSend, DistributedSendRefHolder


__doc__ = """
.. autoclass:: GraphPart
.. autoclass:: GraphPartition
.. autoexception:: PartitionInducedCycleError

.. autofunction:: find_partition
.. autofunction:: execute_partition
"""


ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]
T = TypeVar("T", Array, AbstractResultWithNamedArrays)
PartId = Hashable


# {{{ graph partitioner

class _GraphPartitioner(EdgeCachedMapper):
    """Given a function *get_part_id*, produces subgraphs representing
    the computation. Users should not use this class directly, but use
    :meth:`find_partition` instead.
    """

    def __init__(self, get_part_id: Callable[[ArrayOrNames], PartId]) -> None:
        super().__init__()

        # Function to determine the part ID
        self._get_part_id: Callable[[ArrayOrNames], PartId] = \
                get_part_id

        # Naming for newly created PlaceHolders at part edges
        from pytools import UniqueNameGenerator
        self.name_generator = UniqueNameGenerator(forced_prefix="_part_ph_")

        # "edges" of the partitioned graph, maps an edge between two parts,
        # represented by a tuple of part identifiers, to a set of placeholder
        # names "conveying" information across the edge.
        self.part_pair_to_edges: Dict[Tuple[PartId, PartId],
                Set[str]] = {}

        self.var_name_to_result: Dict[str, Array] = {}

        self._seen_node_to_placeholder: Dict[ArrayOrNames, Placeholder] = {}

        # Reading the seen part IDs out of part_pair_to_edges is incorrect:
        # e.g. if each part is self-contained, no edges would appear. Instead,
        # we remember each part ID we see below, to guarantee that we don't
        # miss any of them.
        self.seen_part_ids: Set[PartId] = set()

        self.pid_to_dist_sends: Dict[PartId, List[DistributedSend]] = {}

        self.pid_to_user_input_names: Dict[PartId, Set[str]] = {}

    def get_part_id(self, expr: ArrayOrNames) -> PartId:
        part_id = self._get_part_id(expr)
        self.seen_part_ids.add(part_id)
        return part_id

    def does_edge_cross_part_boundary(self,
            node1: ArrayOrNames, node2: ArrayOrNames) -> bool:
        return self.get_part_id(node1) != self.get_part_id(node2)

    def make_new_placeholder_name(self) -> str:
        return self.name_generator()

    def add_inter_part_edge(self, target: ArrayOrNames, dependency: ArrayOrNames,
                                placeholder_name: str) -> None:
        pid_target = self.get_part_id(target)
        pid_dependency = self.get_part_id(dependency)

        self.part_pair_to_edges.setdefault(
                (pid_target, pid_dependency), set()).add(placeholder_name)

    def handle_edge(self, expr: ArrayOrNames, child: ArrayOrNames) -> Any:
        if self.does_edge_cross_part_boundary(expr, child):
            try:
                ph = self._seen_node_to_placeholder[child]
            except KeyError:
                ph_name = self.make_new_placeholder_name()
                # If an edge crosses a part boundary, replace the
                # depended-upon node (that nominally lives in the other part)
                # with a Placeholder that lives in the current part. For each
                # part, collect the placeholder names that it’s supposed to
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
            self.add_inter_part_edge(expr, child, ph.name)
            return ph

        else:
            return self.rec(child)

    def __call__(self, expr: T, *args: Any, **kwargs: Any) -> Any:
        # Need to make sure the first node's part is 'seen'
        self.get_part_id(expr)

        return super().__call__(expr, *args, **kwargs)

    def map_placeholder(self, expr: Placeholder, *args: Any) -> Any:
        pid = self.get_part_id(expr)
        self.pid_to_user_input_names.setdefault(pid, set()).add(expr.name)
        return super().map_placeholder(expr)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder, *args: Any) -> Any:
        send_part_id = self.get_part_id(expr.send.data)

        from pytato.distributed import DistributedSend
        self.pid_to_dist_sends.setdefault(send_part_id, []).append(
                DistributedSend(
                    data=self.rec(expr.send.data),
                    dest_rank=expr.send.dest_rank,
                    comm_tag=expr.send.comm_tag,
                    tags=expr.send.tags))

        return self.rec(expr.passthrough_data)

# }}}


# {{{ graph partition

@dataclass(frozen=True)
class GraphPart:
    """
    .. attribute:: pid

        An identifier for this part of the graph.

    .. attribute:: needed_pids

        The IDs of parts that are required to be evaluated before this
        part can be evaluated.

    .. attribute:: input_names

        Names of placeholders the part requires as input.

    .. attribute:: output_names

        Names of placeholders this part provides as output.

    .. attribute:: distributed_sends

        List of :class:`pytato.distributed.DistributedSend` instances whose
        :attr:`DistributedSend.data` are in this part.

    .. attribute:: user_input_names

        A :class:`dict` mapping names to :class:`Placeholder` instances that
        represent input to the computational graph, i.e. were *not* introduced
        by partitioning.
    """
    pid: PartId
    needed_pids: FrozenSet[PartId]
    input_names: FrozenSet[str]
    output_names: FrozenSet[str]
    distributed_sends: List[DistributedSend]
    user_input_names: FrozenSet[str]

    # FIXME: Refactor _GraphPartitioner/find_partition so that this does not
    # have to know about distributed_sends. It will disappear from the data
    # structure when find_partition and gather_distributed_comm_info become
    # a single function aimed at the distributed use case.


@dataclass(frozen=True)
class GraphPartition:
    """Store information about a partitioning of an expression graph.

    .. attribute:: parts

        Mapping from part IDs to instances of :class:`GraphPart`.

    .. attribute:: var_name_to_result

       Mapping of placeholder names to the respective :class:`pytato.array.Array`
       they represent.

    .. attribute:: toposorted_part_ids

       One possible topologically sorted ordering of part IDs that is
       admissible under :attr:`GraphPart.needed_pids`.

       .. note::

           This attribute could be recomputed for those dependencies. Since it
           is computed as part of :func:`find_partition` anyway, it is
           preserved here.
    """
    parts: Mapping[PartId, GraphPart]
    var_name_to_result: Mapping[str, Array]
    toposorted_part_ids: List[PartId]

# }}}


class PartitionInducedCycleError(Exception):
    """Raised by :func:`find_partition` if the partitioning induced a
    cycle in the graph of partitions.
    """


# {{{ find_partitions

def find_partition(outputs: DictOfNamedArrays,
        part_func: Callable[[ArrayOrNames], PartId]) ->\
        GraphPartition:
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
    :returns: An instance of :class:`GraphPartition` that contains the partition.
    """

    gp = _GraphPartitioner(part_func)
    rewritten_outputs = {name: gp(expr) for name, expr in outputs._data.items()}

    pid_to_output_names: Dict[PartId, Set[str]] = {
        pid: set() for pid in gp.seen_part_ids}
    pid_to_input_names: Dict[PartId, Set[str]] = {
        pid: set() for pid in gp.seen_part_ids}

    var_name_to_result = gp.var_name_to_result.copy()

    for out_name, rewritten_output in rewritten_outputs.items():
        out_part_id = part_func(outputs._data[out_name])
        pid_to_output_names.setdefault(out_part_id, set()).add(out_name)
        var_name_to_result[out_name] = rewritten_output

    # Mapping of nodes to their successors; used to compute the topological order
    pid_to_needing_pids: Dict[PartId, Set[PartId]] = {
            pid: set() for pid in gp.seen_part_ids}
    pid_to_needed_pids: Dict[PartId, Set[PartId]] = {
            pid: set() for pid in gp.seen_part_ids}

    for (pid_target, pid_dependency), var_names in \
            gp.part_pair_to_edges.items():
        pid_to_needing_pids[pid_dependency].add(pid_target)
        pid_to_needed_pids[pid_target].add(pid_dependency)

        for var_name in var_names:
            pid_to_output_names[pid_dependency].add(var_name)
            pid_to_input_names[pid_target].add(var_name)

    from pytools.graph import compute_topological_order, CycleError
    try:
        toposorted_part_ids = compute_topological_order(pid_to_needing_pids)
    except CycleError:
        raise PartitionInducedCycleError

    result = GraphPartition(
            parts={
                pid: GraphPart(
                    pid=pid,
                    needed_pids=frozenset(pid_to_needed_pids[pid]),
                    input_names=frozenset(pid_to_input_names[pid]),
                    output_names=frozenset(pid_to_output_names[pid]),
                    distributed_sends=gp.pid_to_dist_sends.get(pid, []),
                    user_input_names=frozenset(
                        gp.pid_to_user_input_names.get(pid, set())),
                    )
                for pid in gp.seen_part_ids},
            var_name_to_result=var_name_to_result,
            toposorted_part_ids=toposorted_part_ids)

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


def _check_partition_disjointness(partition: GraphPartition) -> None:
    part_id_to_nodes: Dict[PartId, Set[ArrayOrNames]] = {}

    for part in partition.parts.values():
        mapper = _SeenNodesWalkMapper()
        for out_name in part.output_names:
            mapper(partition.var_name_to_result[out_name])

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
                        f"in both '{part.pid}' and '{other_part_id}'")

        part_id_to_nodes[part.pid] = mapper.seen_nodes

# }}}


# {{{ generate_code_for_partitions

def generate_code_for_partition(partition: GraphPartition) \
        -> Dict[PartId, BoundProgram]:
    """Return a mapping of partition identifiers to their
       :class:`pytato.target.BoundProgram`."""
    from pytato import generate_loopy
    part_id_to_prg = {}

    for part in partition.parts.values():
        d = DictOfNamedArrays(
                    {var_name: partition.var_name_to_result[var_name]
                        for var_name in part.output_names
                     })
        part_id_to_prg[part.pid] = generate_loopy(d)

    return part_id_to_prg

# }}}


# {{{ execute_partitions

def execute_partition(partition: GraphPartition, prg_per_partition:
        Dict[PartId, BoundProgram], queue: Any) -> Dict[str, Any]:
    """Executes a set of partitions on a :class:`pyopencl.CommandQueue`.

    :param parts: An instance of :class:`GraphPartition` representing the
        partitioned code.
    :param queue: An instance of :class:`pyopencl.CommandQueue` to execute the
        code on.
    :returns: A dictionary of variable names mapped to their values.
    """
    context: Dict[str, Any] = {}
    for pid in partition.toposorted_part_ids:
        part = partition.parts[pid]
        inputs = {
            k: context[k] for k in part.input_names
            if k in context}

        _evt, result_dict = prg_per_partition[pid](queue=queue, **inputs)
        context.update(result_dict)

    return context

# }}}


# vim: foldmethod=marker
