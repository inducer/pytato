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

from typing import Dict, Union, Set, Hashable, TypeVar, FrozenSet, Mapping
import attrs

import logging
logger = logging.getLogger(__name__)

from pytools import memoize_method
from pytato.transform import CachedWalkMapper
from pytato.array import (
        Array, AbstractResultWithNamedArrays, Placeholder,
        DictOfNamedArrays, make_dict_of_named_arrays)

from pytato.target import BoundProgram
from pymbolic.mapper.optimize import optimize_mapper


__doc__ = """
Partitioning of graphs in :mod:`pytato` currently mainly serves to enable
:ref:`distributed computation <distributed>`, i.e. sending and receiving data
as part of graph evaluation.

However, as implemented, it is completely general and not specific to this use
case.  Partitioning of expression graphs is based on a few assumptions:

- We must be able to execute parts in any dependency-respecting order.
- Parts are compiled at partitioning time, so what inputs they take from memory
  vs. what they compute is decided at that time.
- No part may depend on its own outputs as inputs.
  (cf. :exc:`PartitionInducedCycleError`)

.. autoclass:: GraphPart
.. autoclass:: GraphPartition
.. autoexception:: PartitionInducedCycleError

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: T

    A type variable for :class:`~pytato.array.AbstractResultWithNamedArrays`.
"""


ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]
T = TypeVar("T", bound=ArrayOrNames)
PartId = Hashable


# {{{ graph partition

@attrs.define(frozen=True, slots=False)
class GraphPart:
    """
    .. attribute:: pid

        An identifier for this part of the graph.

    .. attribute:: needed_pids

        The IDs of parts that are required to be evaluated before this
        part can be evaluated.

    .. attribute:: user_input_names

        A :class:`frozenset` of names representing input to the computational
        graph, i.e. which were *not* introduced by partitioning.

    .. attribute:: partition_input_names

        A :class:`frozenset` of names of placeholders the part requires as
        input from other parts in the partition.

    .. attribute:: output_names

        Names of placeholders this part provides as output.

    .. automethod:: all_input_names
    """
    pid: PartId
    needed_pids: FrozenSet[PartId]
    user_input_names: FrozenSet[str]
    partition_input_names: FrozenSet[str]
    output_names: FrozenSet[str]

    @memoize_method
    def all_input_names(self) -> FrozenSet[str]:
        return self.user_input_names | self. partition_input_names


@attrs.define(frozen=True, slots=False)
class GraphPartition:
    """Store information about a partitioning of an expression graph.

    .. attribute:: parts

        Mapping from part IDs to instances of :class:`GraphPart`.

    .. attribute:: var_name_to_result

       Mapping of placeholder names to the respective :class:`pytato.array.Array`
       they represent.
    """
    parts: Mapping[PartId, GraphPart]
    var_name_to_result: Mapping[str, Array]

# }}}


class PartitionInducedCycleError(Exception):
    """Raised by :func:`find_partition` if the partitioning induced a
    cycle in the graph of partitions.
    """


# {{{ _run_partition_diagnostics

def _run_partition_diagnostics(
        outputs: DictOfNamedArrays, gp: GraphPartition) -> None:
    if __debug__:
        _check_partition_disjointness(gp)

    from pytato.analysis import get_num_nodes
    num_nodes_per_part = [get_num_nodes(make_dict_of_named_arrays(
            {x: gp.var_name_to_result[x] for x in part.output_names}))
            for part in gp.parts.values()]

    logger.info(f"find_partition: Split {get_num_nodes(outputs)} nodes into "
        f"{len(gp.parts)} parts, with {num_nodes_per_part} nodes in each "
        "partition.")

# }}}


@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class _SeenNodesWalkMapper(CachedWalkMapper):
    def __init__(self) -> None:
        super().__init__()
        self.seen_nodes: Set[ArrayOrNames] = set()

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self, expr: ArrayOrNames) -> int:  # type: ignore[override]
        return id(expr)

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def visit(self, expr: ArrayOrNames) -> bool:  # type: ignore[override]
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
                if not (isinstance(my_node, Placeholder)
                       or my_node not in other_node_set):
                    raise RuntimeError(
                        "Partitions not disjoint: "
                        f"{my_node.__class__.__name__} (id={hex(id(my_node))}) "
                        f"in both '{part.pid}' and '{other_part_id}'"
                        f"{part.output_names=} "
                        f"{partition.parts[other_part_id].output_names=} ")

        part_id_to_nodes[part.pid] = mapper.seen_nodes

# }}}


# {{{ generate_code_for_partition

def generate_code_for_partition(partition: GraphPartition) \
        -> Mapping[PartId, BoundProgram]:
    """Return a mapping of partition identifiers to their
       :class:`pytato.target.BoundProgram`."""
    from pytato import generate_loopy
    part_id_to_prg = {}

    for part in sorted(partition.parts.values(),
                       key=lambda part_: sorted(part_.output_names)):
        d = make_dict_of_named_arrays(
                    {var_name: partition.var_name_to_result[var_name]
                        for var_name in part.output_names
                     })
        part_id_to_prg[part.pid] = generate_loopy(d)

    return part_id_to_prg

# }}}


# vim: foldmethod=marker
