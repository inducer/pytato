"""
Partitioning
------------

Partitioning of graphs in :mod:`pytato` serves to enable
:ref:`distributed computation <distributed>`, i.e. sending and receiving data
as part of graph evaluation.

Partitioning of expression graphs is based on a few assumptions:

- We must be able to execute parts in any dependency-respecting order.
- Parts are compiled at partitioning time, so what inputs they take from memory
  vs. what they compute is decided at that time.
- No part may depend on its own outputs as inputs.

.. currentmodule:: pytato

.. autoclass:: DistributedGraphPart
.. autoclass:: DistributedGraphPartition

.. autofunction:: find_distributed_partition

.. currentmodule:: pytato.distributed.partition

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: T

    A type variable for :class:`~pytato.array.AbstractResultWithNamedArrays`.
.. autoclass:: CommunicationOpIdentifier
.. class:: CommunicationDepGraph

    An alias for
    ``Mapping[CommunicationOpIdentifier, AbstractSet[CommunicationOpIdentifier]]``.
"""

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

from functools import reduce
import collections
from typing import (
        Iterator, Iterable, Sequence, Any, Mapping, FrozenSet, Set, Dict, cast,
        List, AbstractSet, TypeVar, TYPE_CHECKING, Hashable, Optional, Tuple)

import attrs
from immutabledict import immutabledict

from pytools.graph import CycleError
from pytools import memoize_method

from pymbolic.mapper.optimize import optimize_mapper
from pytools import UniqueNameGenerator

from pytato.scalar_expr import SCALAR_CLASSES
from pytato.array import (Array, DictOfNamedArrays, Placeholder, make_placeholder)
from pytato.transform import (ArrayOrNames, CopyMapper,
                              CachedWalkMapper,
                              CombineMapper)
from pytato.distributed.nodes import (
        DistributedRecv, DistributedSend, DistributedSendRefHolder)
from pytato.distributed.nodes import CommTagType
from pytato.analysis import DirectPredecessorsGetter

if TYPE_CHECKING:
    import mpi4py.MPI


@attrs.define(frozen=True)
class CommunicationOpIdentifier:
    """Identifies a communication operation (consisting of a pair of
    a send and a receive).

    .. attribute:: src_rank
    .. attribute:: dest_rank
    .. attribute:: comm_tag

    .. note::

        In :func:`~pytato.find_distributed_partition`, we use instances of this
        type as though they identify sends or receives, i.e. just a single end
        of the communication. Realize that this is only true given the
        additional context of which rank is the local rank.
    """
    src_rank: int
    dest_rank: int
    comm_tag: CommTagType


CommunicationDepGraph = Mapping[
        CommunicationOpIdentifier, AbstractSet[CommunicationOpIdentifier]]


_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


# {{{ crude ordered set


class _OrderedSet(collections.abc.MutableSet[_ValueT]):
    def __init__(self, items: Optional[Iterable[_ValueT]] = None):
        # Could probably also use a valueless dictionary; not sure if it matters
        self._items: Set[_ValueT] = set()
        self._items_ordered: List[_ValueT] = []
        if items is not None:
            for item in items:
                self.add(item)

    def add(self, item: _ValueT) -> None:
        if item not in self._items:
            self._items.add(item)
            self._items_ordered.append(item)

    def discard(self, item: _ValueT) -> None:
        # Not currently needed
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[_ValueT]:
        return iter(self._items_ordered)

    def __contains__(self, item: Any) -> bool:
        return item in self._items

    def __and__(self, other: AbstractSet[_ValueT]) -> _OrderedSet[_ValueT]:
        result: _OrderedSet[_ValueT] = _OrderedSet()
        for item in self._items_ordered:
            if item in other:
                result.add(item)
        return result

    # Must be "Any" instead of "_ValueT", otherwise it violates Liskov substitution
    # according to mypy. *shrug*
    def __or__(self, other: AbstractSet[Any]) -> _OrderedSet[_ValueT]:
        result: _OrderedSet[_ValueT] = _OrderedSet(self._items_ordered)
        for item in other:
            result.add(item)
        return result

    def __sub__(self, other: AbstractSet[_ValueT]) -> _OrderedSet[_ValueT]:
        result: _OrderedSet[_ValueT] = _OrderedSet()
        for item in self._items_ordered:
            if item not in other:
                result.add(item)
        return result

# }}}


# {{{ distributed graph part

PartId = Hashable


@attrs.define(frozen=True, slots=False)
class DistributedGraphPart:
    """For one graph part, record send/receive information for input/
    output names.

    Names that occur as keys in :attr:`name_to_recv_node` and
    :attr:`name_to_send_nodes` are usable as input names by other
    parts, or in the result of the computation.

    - Names specified in :attr:`name_to_recv_node` *must not* occur in
      :attr:`output_names`.
    - Names specified in :attr:`name_to_send_nodes` *must* occur in
      :attr:`output_names`.

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

    .. attribute:: name_to_recv_node
    .. attribute:: name_to_send_nodes

    .. automethod:: all_input_names
    """
    pid: PartId
    needed_pids: FrozenSet[PartId]
    user_input_names: FrozenSet[str]
    partition_input_names: FrozenSet[str]
    output_names: FrozenSet[str]

    name_to_recv_node: Mapping[str, DistributedRecv]
    name_to_send_nodes: Mapping[str, Sequence[DistributedSend]]

    @memoize_method
    def all_input_names(self) -> FrozenSet[str]:
        return self.user_input_names | self. partition_input_names

# }}}


# {{{ distributed graph partition

@attrs.define(frozen=True, slots=False)
class DistributedGraphPartition:
    """
    .. attribute:: parts

        Mapping from part IDs to instances of :class:`DistributedGraphPart`.

    .. attribute:: name_to_output

       Mapping of placeholder names to the respective :class:`pytato.array.Array`
       they represent. This is where the actual expressions are stored, for
       all parts. Observe that the :class:`DistributedGraphPart`, for the most
       part, only stores names. These "outputs" may be 'part outputs' (i.e.
       data computed in one part for use by another, effectively tempoarary
       variables), or 'overall outputs' of the comutation.

    .. attribute:: overall_output_names

        The names of the outputs (in :attr:`name_to_output`) that were given to
        :func:`find_distributed_partition` to specify the overall computaiton.

    """
    parts: Mapping[PartId, DistributedGraphPart]
    name_to_output: Mapping[str, Array]
    overall_output_names: Sequence[str]

# }}}


# {{{ _DistributedInputReplacer

class _DistributedInputReplacer(CopyMapper):
    """Replaces part inputs with :class:`~pytato.array.Placeholder`
    instances for their assigned names. Also gathers names for
    user-supplied inputs needed by the part
    """

    def __init__(self,
                 recvd_ary_to_name: Mapping[Array, str],
                 sptpo_ary_to_name: Mapping[Array, str],
                 name_to_output: Mapping[str, Array],
                 ) -> None:
        super().__init__()

        self.recvd_ary_to_name = recvd_ary_to_name
        self.sptpo_ary_to_name = sptpo_ary_to_name
        self.name_to_output = name_to_output
        self.output_arrays = frozenset(name_to_output.values())

        self.user_input_names: Set[str] = set()
        self.partition_input_name_to_placeholder: Dict[str, Placeholder] = {}

    def map_placeholder(self, expr: Placeholder) -> Placeholder:
        self.user_input_names.add(expr.name)
        return expr

    def map_distributed_recv(self, expr: DistributedRecv) -> Placeholder:
        name = self.recvd_ary_to_name[expr]
        return self._get_placeholder_for(name, expr)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> Array:
        result = self.rec(expr.passthrough_data)
        assert isinstance(result, Array)
        return result

    # Note: map_distributed_send() is not called like other mapped methods in a
    # DAG traversal, since a DistributedSend is not an Array and has no
    # '_mapper_method' field. Furthermore, at the point where this mapper is used,
    # the DistributedSendRefHolders have been removed from the DAG, and hence there
    # are no more references to the DistributedSends from within the DAG. This
    # method must therefore be called explicitly.
    def map_distributed_send(self, expr: DistributedSend) -> DistributedSend:
        new_data = self.rec(expr.data)
        assert isinstance(new_data, Array)
        new_send = DistributedSend(
                data=new_data,
                dest_rank=expr.dest_rank,
                comm_tag=expr.comm_tag,
                tags=expr.tags)
        return new_send

    # type ignore because no args, kwargs
    def rec(self, expr: ArrayOrNames) -> ArrayOrNames:  # type: ignore[override]
        assert isinstance(expr, Array)

        key = self.get_cache_key(expr)
        try:
            return self._cache[key]
        except KeyError:
            pass

        # If the array is an output from the current part, it would
        # be counterproductive to turn it into a placeholder: we're
        # the ones who are supposed to compute it!
        if expr not in self.output_arrays:

            name = self.recvd_ary_to_name.get(expr)
            if name is not None:
                return self._get_placeholder_for(name, expr)

            name = self.sptpo_ary_to_name.get(expr)
            if name is not None:
                return self._get_placeholder_for(name, expr)

        return cast(ArrayOrNames, super().rec(expr))

    def _get_placeholder_for(self, name: str, expr: Array) -> Placeholder:
        placeholder = self.partition_input_name_to_placeholder.get(name)
        if placeholder is None:
            placeholder = make_placeholder(
                    name, expr.shape, expr.dtype, expr.tags,
                    expr.axes)
            self.partition_input_name_to_placeholder[name] = placeholder
        return placeholder

# }}}


@attrs.define(frozen=True)
class _PartCommIDs:
    """A *part*, unlike a *batch*, begins with receives and ends with sends.
    """
    recv_ids: FrozenSet[CommunicationOpIdentifier]
    send_ids: FrozenSet[CommunicationOpIdentifier]


# {{{ _make_distributed_partition

def _make_distributed_partition(
        name_to_output_per_part: Sequence[Mapping[str, Array]],
        part_comm_ids: Sequence[_PartCommIDs],
        recvd_ary_to_name: Mapping[Array, str],
        sent_ary_to_name: Mapping[Array, str],
        sptpo_ary_to_name: Mapping[Array, str],
        local_recv_id_to_recv_node: Dict[CommunicationOpIdentifier, DistributedRecv],
        local_send_id_to_send_node: Dict[CommunicationOpIdentifier, DistributedSend],
        overall_output_names: Sequence[str],
        ) -> DistributedGraphPartition:
    name_to_output = {}
    parts: Dict[PartId, DistributedGraphPart] = {}

    for part_id, name_to_ouput in enumerate(name_to_output_per_part):
        comm_replacer = _DistributedInputReplacer(
            recvd_ary_to_name, sptpo_ary_to_name, name_to_ouput)

        for name, val in name_to_ouput.items():
            assert name not in name_to_output
            name_to_output[name] = comm_replacer(val)

        comm_ids = part_comm_ids[part_id]

        name_to_send_nodes: Dict[str, List[DistributedSend]] = {}
        for send_id in comm_ids.send_ids:
            send_node = local_send_id_to_send_node[send_id]
            name = sent_ary_to_name[send_node.data]
            name_to_send_nodes.setdefault(name, []).append(
                comm_replacer.map_distributed_send(send_node))

        parts[part_id] = DistributedGraphPart(
                pid=part_id,
                needed_pids=frozenset({part_id - 1} if part_id else {}),
                user_input_names=frozenset(comm_replacer.user_input_names),
                partition_input_names=frozenset(
                    comm_replacer.partition_input_name_to_placeholder.keys()),
                output_names=frozenset(name_to_ouput.keys()),
                name_to_recv_node=immutabledict({
                    recvd_ary_to_name[local_recv_id_to_recv_node[recv_id]]:
                    local_recv_id_to_recv_node[recv_id]
                    for recv_id in comm_ids.recv_ids}),
                name_to_send_nodes=immutabledict(name_to_send_nodes))

    result = DistributedGraphPartition(
            parts=parts,
            name_to_output=name_to_output,
            overall_output_names=overall_output_names,
            )

    return result

# }}}


# {{{ _LocalSendRecvDepGatherer

def _send_to_comm_id(
        local_rank: int, send: DistributedSend) -> CommunicationOpIdentifier:
    if local_rank == send.dest_rank:
        raise NotImplementedError("Self-sends are not currently allowed. "
                                  f"(tag: '{send.comm_tag}')")

    return CommunicationOpIdentifier(
        src_rank=local_rank,
        dest_rank=send.dest_rank,
        comm_tag=send.comm_tag)


def _recv_to_comm_id(
        local_rank: int, recv: DistributedRecv) -> CommunicationOpIdentifier:
    if local_rank == recv.src_rank:
        raise NotImplementedError("Self-receives are not currently allowed. "
                                  f"(tag: '{recv.comm_tag}')")

    return CommunicationOpIdentifier(
        src_rank=recv.src_rank,
        dest_rank=local_rank,
        comm_tag=recv.comm_tag)


class _LocalSendRecvDepGatherer(
        CombineMapper[FrozenSet[CommunicationOpIdentifier]]):
    def __init__(self, local_rank: int) -> None:
        super().__init__()
        self.local_comm_ids_to_needed_comm_ids: \
                Dict[CommunicationOpIdentifier,
                     FrozenSet[CommunicationOpIdentifier]] = {}

        self.local_recv_id_to_recv_node: \
                Dict[CommunicationOpIdentifier, DistributedRecv] = {}
        self.local_send_id_to_send_node: \
                Dict[CommunicationOpIdentifier, DistributedSend] = {}

        self.local_rank = local_rank

    def combine(
            self, *args: FrozenSet[CommunicationOpIdentifier]
            ) -> FrozenSet[CommunicationOpIdentifier]:
        return reduce(frozenset.union, args, frozenset())

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> FrozenSet[CommunicationOpIdentifier]:
        send_id = _send_to_comm_id(self.local_rank, expr.send)

        if send_id in self.local_send_id_to_send_node:
            from pytato.distributed.verify import DuplicateSendError
            raise DuplicateSendError(f"Multiple sends found for '{send_id}'")

        self.local_comm_ids_to_needed_comm_ids[send_id] = \
                self.rec(expr.send.data)

        self.local_send_id_to_send_node[send_id] = expr.send

        return self.rec(expr.passthrough_data)

    def _map_input_base(self, expr: Array) -> FrozenSet[CommunicationOpIdentifier]:
        return frozenset()

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_distributed_recv(
            self, expr: DistributedRecv
            ) -> FrozenSet[CommunicationOpIdentifier]:
        recv_id = _recv_to_comm_id(self.local_rank, expr)

        if recv_id in self.local_recv_id_to_recv_node:
            from pytato.distributed.verify import DuplicateRecvError
            raise DuplicateRecvError(f"Multiple receives found for '{recv_id}'")

        self.local_comm_ids_to_needed_comm_ids[recv_id] = frozenset()

        self.local_recv_id_to_recv_node[recv_id] = expr

        return frozenset({recv_id})

# }}}

# {{{ schedule_wrapper


def schedule_wrapper(
        comm_ids_to_needed_comm_ids: CommunicationDepGraph, cnts: set[int]
        ) -> Sequence[AbstractSet[CommunicationOpIdentifier]]:
    """ Wrapper to enable testing the scheduler. cnts will hold the total
    nodes searched during the sorting followed by the scheduling.
    """

    return _schedule_comm_batches(comm_ids_to_needed_comm_ids, cnts)

# }}}

# {{{ _schedule_comm_batches


def _schedule_comm_batches(
        comm_ids_to_needed_comm_ids: CommunicationDepGraph,
        cnts: Optional[set[int]] = None) \
        -> Sequence[AbstractSet[CommunicationOpIdentifier]]:
    """For each :class:`CommunicationOpIdentifier`, determine the
    'round'/'batch' during which it will be performed. A 'batch'
    of communication consists of sends and receives. Computation
    occurs between batches. (So, from the perspective of the
    :class:`DistributedGraphPartition`, communication batches
    sit *between* parts.)
    """
    # FIXME: I'm an O(n^2) algorithm.
    comm_batches: List[AbstractSet[CommunicationOpIdentifier]] = []

    scheduled_comm_ids: Set[CommunicationOpIdentifier] = set()

    total_ids = len(comm_ids_to_needed_comm_ids)
    n_visited_in_scheduling = 0
    sorted_ids, n_visited_in_sort = _topo_sort(comm_ids_to_needed_comm_ids)
    while len(scheduled_comm_ids) < total_ids:
        batch_ready = False
        comm_ids_this_batch = set()
        while not batch_ready:
            comm_id = sorted_ids[-1]
            n_visited_in_scheduling += 1
            needed_comm_ids = comm_ids_to_needed_comm_ids[comm_id]
            if (needed_comm_ids > scheduled_comm_ids):
                batch_ready = True  # batch is done.
            else:
                # Append to batch.
                comm_id = sorted_ids.pop()
                comm_ids_this_batch.add(comm_id)
            if len(sorted_ids) == 0:
                batch_ready = True
        scheduled_comm_ids.update(comm_ids_this_batch)
        comm_batches.append(comm_ids_this_batch)
    if cnts:
        cnts.clear()
        cnts.add(sum([n_visited_in_sort, n_visited_in_scheduling]))
    return comm_batches

# }}}

# {{{ _topo_sort


def _topo_sort(
        comm_ids_to_needed_comm_ids: CommunicationDepGraph
        ) -> Tuple[List[CommunicationOpIdentifier], int]:
    """
    Compute a topological sort of the input graph which specifies
    the tasks that need to be completed before task_i can be scheduled
    for every i.
    """
    locations_visited: Set[CommunicationOpIdentifier] = set()
    temp_visit: Set[CommunicationOpIdentifier] = set()
    sorted_list: List[CommunicationOpIdentifier] = []

    def _topo_helper(
            comm_id: CommunicationOpIdentifier
            ) -> Tuple[List[CommunicationOpIdentifier], int]:
        """
        Helper funciton to do depth first search.
        """
        count = 0
        if comm_id in locations_visited:
            return sorted_list, count
        if comm_id in temp_visit:
            raise CycleError("Cycle detected in communication graph")

        temp_visit.add(comm_id)
        for item in comm_ids_to_needed_comm_ids[comm_id]:
            count += 1
            _topo_helper(item)
        temp_visit.remove(comm_id)
        locations_visited.add(comm_id)
        sorted_list.append(comm_id)
        return sorted_list, count

    num_visited = 0
    for comm_id in comm_ids_to_needed_comm_ids:
        sorted_list, num_visited = _topo_helper(comm_id)
    sorted_list.reverse()
    return sorted_list, num_visited
# }}}

# {{{  _MaterializedArrayCollector


@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class _MaterializedArrayCollector(CachedWalkMapper):
    """
    Collects all nodes that have to be materialized during code-generation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.materialized_arrays: _OrderedSet[Array] = _OrderedSet()

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self, expr: ArrayOrNames) -> int:  # type: ignore[override]
        return id(expr)

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def post_visit(self, expr: Any) -> None:  # type: ignore[override]
        from pytato.tags import ImplStored
        from pytato.loopy import LoopyCallResult

        if (isinstance(expr, Array) and expr.tags_of_type(ImplStored)):
            self.materialized_arrays.add(expr)

        if isinstance(expr, LoopyCallResult):
            self.materialized_arrays.add(expr)
            from pytato.loopy import LoopyCall
            assert isinstance(expr._container, LoopyCall)
            for _, subexpr in sorted(expr._container.bindings.items()):
                if isinstance(subexpr, Array):
                    self.materialized_arrays.add(subexpr)
                else:
                    assert isinstance(subexpr, SCALAR_CLASSES)

# }}}


# {{{ _set_dict_union_mpi

def _set_dict_union_mpi(
        dict_a: Mapping[_KeyT, FrozenSet[_ValueT]],
        dict_b: Mapping[_KeyT, FrozenSet[_ValueT]],
        mpi_data_type: mpi4py.MPI.Datatype) -> Mapping[_KeyT, FrozenSet[_ValueT]]:
    assert mpi_data_type is None
    result = dict(dict_a)
    for key, values in dict_b.items():
        result[key] = result.get(key, frozenset()) | values
    return result

# }}}


# {{{ find_distributed_partition

def find_distributed_partition(
        mpi_communicator: mpi4py.MPI.Comm,
        outputs: DictOfNamedArrays
        ) -> DistributedGraphPartition:
    r"""
    Compute a :class:DistributedGraphPartition` (for use with
    :func:`execute_distributed_partition`) that evaluates the
    same result as *outputs*, such that:

    - communication only happens at the beginning and end of
      each :class:`DistributedGraphPart`, and
    - the partition introduces no circular dependencies between parts,
      mediated by either local data flow or off-rank communication.

    .. warning::

        This is an MPI-collective operation.

    The following sections describe the (non-binding, as far as documentation
    is concerned) algorithm behind the partitioner.

    .. rubric:: Preliminaries

    We identify a communication operation (consisting of a pair of a send
    and a receive) by a
    :class:`~pytato.distributed.partition.CommunicationOpIdentifier`. We keep
    graphs of these in
    :class:`~pytato.distributed.partition.CommunicationDepGraph`.

    If ``graph`` is a
    :class:`~pytato.distributed.partition.CommunicationDepGraph`, then ``b in
    graph[a]`` means that, in order to initiate the communication operation
    identified by :class:`~pytato.distributed.partition.CommunicationOpIdentifier`
    ``a``, the communication operation identified by
    :class:`~pytato.distributed.partition.CommunicationOpIdentifier` ``b`` must
    be completed.
    I.e. the nodes are "communication operations", i.e. pairs of
    send/receive. Edges represent (rank-local) data flow between them.

    .. rubric:: Step 1: Build a global graph of data flow between communication
        operations

    As a first step, each rank receives a copy of global
    :class:`~pytato.distributed.partition.CommunicationDepGraph`, as described
    above. This becomes ``comm_ids_to_needed_comm_ids``.

    .. rubric:: Step 2: Obtain a "schedule" of "communication batches"

    On rank 0, compute and broadcast a topological order of
    ``comm_ids_to_needed_comm_ids``. The result of this is
    ``comm_batches``, a sequence of sets of
    :class:`~pytato.distributed.partition.CommunicationOpIdentifier`
    instances, identifying sets of communication operations expected
    to complete *between* parts of the computation. (I.e. computation
    will occur before the first communication batch, then between the
    first and second, and so on.)

    .. note::

        An important restriction of this scheme is that a linear order
        of communication batches is obtained, meaning that, typically,
        no overlap of computation and communication occurs.

    .. rubric:: Step 3: Create rank-local part descriptors

    On each rank, we next rewrite the communication batches into computation
    parts, each identified by a ``_PartCommIDs`` structure, which
    gathers receives that need to complete *before* the computation on a part
    can begin and sends that can begin once computation on a part
    is complete.

    .. rubric:: Step 4: Assign materialized arrays to parts

    "Stored" arrays are those whose value will be computed and stored
    in memory. This includes the following:

    - Arrays tagged :class:`~pytato.tags.ImplStored` by prior processing of the DAG,
    - arrays being sent (because we need to hand a buffer to MPI),
    - arrays being received (because MPI puts the received data
      in memory)
    - Overall outputs of the computation.

    By contrast, the code below uses the word "materialized" only for arrays of
    the first type (tagged :class:`~pytato.tags.ImplStored`), so that 'stored' is a
    superset of 'materialized'.

    In addition, data computed by one *part* (in the above sense) of the
    computation and used by another must be in memory. Evaluating and storing
    temporary arrays is expensive, and so we try to minimize the number of
    times that that this occurs as part of the partitioning.  This is done by
    relying on already-stored arrays as much as possible and recomputing any
    intermediate results needed in, say, an originating and a consuming part.

    We begin this process by assigning each materialized
    array to a part in which it is computed, based on the part in which
    data depending on such arrays is sent. This choice implies that these
    computations occur as late as possible.

    .. rubric:: Step 5: Promote stored arrays to part outputs if needed

    In :class:`DistributedGraphPart`, our description of the partitioned
    computation, each part can declare named 'outputs' that can be used
    by subsequent parts. Stored arrays are promoted to part outputs
    if they have users in parts other than the one in which they
    are computed.

    .. rubric:: Step 6:: Rewrite the DAG into its parts

    In the final step, we traverse the DAG to apply the following changes:

    - Replace :class:`DistributedRecv` nodes with placeholders for names
      assigned in :attr:`DistributedGraphPart.name_to_recv_node`.
    - Replace references to out-of-part stored arrays with
      :class:`~pytato.array.Placeholder` instances.
    - Gather sent arrays into
      assigned in :attr:`DistributedGraphPart.name_to_send_nodes`.
    """
    from pytato.transform import SubsetDependencyMapper

    import mpi4py.MPI as MPI

    local_rank = mpi_communicator.rank

    # {{{ find comm_ids_to_needed_comm_ids

    lsrdg = _LocalSendRecvDepGatherer(local_rank=local_rank)
    lsrdg(outputs)
    local_comm_ids_to_needed_comm_ids = \
            lsrdg.local_comm_ids_to_needed_comm_ids

    set_dict_union_mpi_op = MPI.Op.Create(
            # type ignore reason: mpi4py misdeclares op functions as returning
            # None.
            _set_dict_union_mpi,  # type: ignore[arg-type]
            commute=True)
    try:
        comm_ids_to_needed_comm_ids = mpi_communicator.allreduce(
                local_comm_ids_to_needed_comm_ids, set_dict_union_mpi_op)
    finally:
        set_dict_union_mpi_op.Free()

    # }}}

    # {{{ make batches out of comm_ids_to_needed_comm_ids

    if mpi_communicator.rank == 0:
        # The comm_batches correspond one-to-one to DistributedGraphParts
        # in the output.
        try:
            comm_batches = _schedule_comm_batches(comm_ids_to_needed_comm_ids)
        except Exception as exc:
            mpi_communicator.bcast(exc)
            raise
        else:
            mpi_communicator.bcast(comm_batches)
    else:
        comm_batches_or_exc = mpi_communicator.bcast(None)
        if isinstance(comm_batches_or_exc, Exception):
            raise comm_batches_or_exc

        comm_batches = cast(
                Sequence[AbstractSet[CommunicationOpIdentifier]],
                comm_batches_or_exc)

    # }}}

    # {{{ create (local) parts out of batch ids

    part_comm_ids: List[_PartCommIDs] = []
    if comm_batches:
        recv_ids: FrozenSet[CommunicationOpIdentifier] = frozenset()
        for batch in comm_batches:
            send_ids = frozenset(
                comm_id for comm_id in batch
                if comm_id.src_rank == local_rank)
            if recv_ids or send_ids:
                part_comm_ids.append(
                    _PartCommIDs(
                        recv_ids=recv_ids,
                        send_ids=send_ids))
            # These go into the next part
            recv_ids = frozenset(
                comm_id for comm_id in batch
                if comm_id.dest_rank == local_rank)
        if recv_ids:
            part_comm_ids.append(
                _PartCommIDs(
                    recv_ids=recv_ids,
                    send_ids=frozenset()))
    else:
        part_comm_ids.append(
            _PartCommIDs(
                recv_ids=frozenset(),
                send_ids=frozenset()))

    nparts = len(part_comm_ids)

    if __debug__:
        from pytato.distributed.verify import MissingRecvError, MissingSendError

        for part in part_comm_ids:
            for recv_id in part.recv_ids:
                if recv_id not in lsrdg.local_recv_id_to_recv_node:
                    raise MissingRecvError(f"no receive for '{recv_id}'")
            for send_id in part.send_ids:
                if send_id not in lsrdg.local_send_id_to_send_node:
                    raise MissingSendError(f"no send for '{send_id}'")

    comm_id_to_part_id = {
        comm_id: ipart
        for ipart, comm_ids in enumerate(part_comm_ids)
        for comm_id in comm_ids.send_ids | comm_ids.recv_ids}

    # }}}

    # {{{ assign each compulsorily materialized array to a part

    materialized_arrays_collector = _MaterializedArrayCollector()
    materialized_arrays_collector(outputs)

    # The sets of arrays below must have a deterministic order in order to ensure
    # that the resulting partition is also deterministic

    sent_arrays = _OrderedSet(
        send_node.data for send_node in lsrdg.local_send_id_to_send_node.values())

    received_arrays = _OrderedSet(lsrdg.local_recv_id_to_recv_node.values())

    # While receive nodes may be marked as materialized, we shouldn't be
    # including them here because we're using them (along with the send nodes)
    # as anchors to place *other* materialized data into the batches.
    # We could allow sent *arrays* to be included here because they are distinct
    # from send *nodes*, but we choose to exclude them in order to simplify the
    # processing below.
    materialized_arrays = (
        materialized_arrays_collector.materialized_arrays
        - received_arrays
        - sent_arrays)

    # "mso" for "materialized/sent/output"
    output_arrays = _OrderedSet(outputs._data.values())
    mso_arrays = materialized_arrays | sent_arrays | output_arrays

    # FIXME: This gathers up materialized_arrays recursively, leading to
    # result sizes potentially quadratic in the number of materialized arrays.
    mso_array_dep_mapper = SubsetDependencyMapper(frozenset(mso_arrays))

    mso_ary_to_first_dep_send_part_id: Dict[Array, int] = {
        ary: nparts
        for ary in mso_arrays}
    for send_id, send_node in lsrdg.local_send_id_to_send_node.items():
        for ary in mso_array_dep_mapper(send_node.data):
            mso_ary_to_first_dep_send_part_id[ary] = min(
                mso_ary_to_first_dep_send_part_id[ary],
                comm_id_to_part_id[send_id])

    if __debug__:
        recvd_array_dep_mapper = SubsetDependencyMapper(frozenset(received_arrays))

        mso_ary_to_last_dep_recv_part_id: Dict[Array, int] = {
                ary: max(
                        (comm_id_to_part_id[
                            _recv_to_comm_id(local_rank,
                                            cast(DistributedRecv, recvd_ary))]
                        for recvd_ary in recvd_array_dep_mapper(ary)),
                        default=-1)
                for ary in mso_arrays
                }

        assert all(
                (
                    mso_ary_to_last_dep_recv_part_id[ary]
                    <= mso_ary_to_first_dep_send_part_id[ary])
                for ary in mso_arrays), \
            "unable to find suitable part for materialized or output array"

    # FIXME: (Seemingly) arbitrary decision, subject to future investigation.
    # Evaluation of materialized arrays is pushed as late as possible,
    # in order to minimize the amount of computation that might prevent
    # data from being sent.
    mso_ary_to_part_id: Dict[Array, int] = {
            ary: min(
                mso_ary_to_first_dep_send_part_id[ary],
                nparts-1)
            for ary in mso_arrays}

    # }}}

    recvd_ary_to_part_id: Dict[Array, int] = {
            recvd_ary: (
                comm_id_to_part_id[
                    _recv_to_comm_id(local_rank, recvd_ary)])
            for recvd_ary in received_arrays}

    # "Materialized" arrays are arrays that are tagged with ImplStored,
    # i.e. "the outside world" (from the perspective of the partitioner)
    # has decided that these arrays will live in memory.
    #
    # In addition, arrays that are sent and received must also live in memory.
    # So, "stored" = "materialized" ∪ "overall outputs" ∪ "communicated"
    stored_ary_to_part_id = mso_ary_to_part_id.copy()
    stored_ary_to_part_id.update(recvd_ary_to_part_id)

    assert all(0 <= part_id < nparts
               for part_id in stored_ary_to_part_id.values())

    stored_arrays = _OrderedSet(stored_ary_to_part_id)

    # {{{ find which stored arrays should become part outputs
    # (because they are used in not just their local part, but also others)

    direct_preds_getter = DirectPredecessorsGetter()

    def get_materialized_predecessors(ary: Array) -> _OrderedSet[Array]:
        materialized_preds: _OrderedSet[Array] = _OrderedSet()
        for pred in direct_preds_getter(ary):
            if pred in materialized_arrays:
                materialized_preds.add(pred)
            else:
                materialized_preds |= get_materialized_predecessors(pred)
        return materialized_preds

    stored_arrays_promoted_to_part_outputs = {
                stored_pred
                for stored_ary in stored_arrays
                for stored_pred in get_materialized_predecessors(stored_ary)
                if (stored_ary_to_part_id[stored_ary]
                    != stored_ary_to_part_id[stored_pred])
                }

    # }}}

    # Don't be tempted to put outputs in _array_names; the mapping from output array
    # to name may not be unique
    _array_name_gen = UniqueNameGenerator(forced_prefix="_pt_dist_")
    _array_names: Dict[Array, str] = {}

    def gen_array_name(ary: Array) -> str:
        name = _array_names.get(ary)
        if name is not None:
            return name
        else:
            name = _array_name_gen()
            _array_names[ary] = name
            return name

    recvd_ary_to_name: Dict[Array, str] = {
        ary: gen_array_name(ary)
        for ary in received_arrays}

    name_to_output_per_part: List[Dict[str, Array]] = [{} for _pid in range(nparts)]

    for name, ary in outputs._data.items():
        pid = stored_ary_to_part_id[ary]
        name_to_output_per_part[pid][name] = ary

    sent_ary_to_name: Dict[Array, str] = {}
    for ary in sent_arrays:
        pid = stored_ary_to_part_id[ary]
        name = gen_array_name(ary)
        sent_ary_to_name[ary] = name
        name_to_output_per_part[pid][name] = ary

    sptpo_ary_to_name: Dict[Array, str] = {}
    for ary in stored_arrays_promoted_to_part_outputs:
        pid = stored_ary_to_part_id[ary]
        name = gen_array_name(ary)
        sptpo_ary_to_name[ary] = name
        name_to_output_per_part[pid][name] = ary

    partition = _make_distributed_partition(
            name_to_output_per_part,
            part_comm_ids,
            recvd_ary_to_name,
            sent_ary_to_name,
            sptpo_ary_to_name,
            lsrdg.local_recv_id_to_recv_node,
            lsrdg.local_send_id_to_send_node,
            tuple(outputs))

    from pytato.distributed.verify import _run_partition_diagnostics
    _run_partition_diagnostics(outputs, partition)

    if __debug__:
        # Avoid potentially confusing errors if one rank manages to continue
        # when another is not able.
        mpi_communicator.barrier()

    return partition

# }}}

# vim: foldmethod=marker
