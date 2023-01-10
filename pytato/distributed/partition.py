"""
.. autoclass:: CommunicationOpIdentifier
.. class:: CommunicationDepGraph

    An alias for
    ``Mapping[CommunicationOpIdentifier, AbstractSet[CommunicationOpIdentifier]]``.

.. currentmodule:: pytato

.. autoclass:: DistributedGraphPart
.. autoclass:: DistributedGraphPartition

.. autofunction:: find_distributed_partition
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
from typing import (
        Sequence, Any, Mapping, FrozenSet, Set, Dict, cast,
        List, AbstractSet, TypeVar, TYPE_CHECKING)

import attrs
from immutables import Map

from pytools.graph import CycleError

from pymbolic.mapper.optimize import optimize_mapper
from pytools import UniqueNameGenerator

from pytato.scalar_expr import SCALAR_CLASSES
from pytato.array import (Array, DictOfNamedArrays, Placeholder, make_placeholder)
from pytato.transform import (ArrayOrNames, CopyMapper,
                              CachedWalkMapper,
                              CombineMapper)
from pytato.partition import GraphPart, GraphPartition, PartId
from pytato.distributed.nodes import (
        DistributedRecv, DistributedSend, DistributedSendRefHolder)
from pytato.distributed.nodes import CommTagType
from pytato.analysis import DirectPredecessorsGetter

if TYPE_CHECKING:
    import mpi4py.MPI as MPI


@attrs.define(frozen=True)
class CommunicationOpIdentifier:
    """Identifies a communication operation (consisting of a pair of
    a send and a receive).

    .. attribute:: src_rank
    .. attribute:: dest_rank
    .. attribute:: comm_tag

    .. note::

        In :func:`find_distributed_partition`, we use instances of this type as
        though they identify sends or receives, i.e. just a single end of the
        communication. Realize that this is only true given the additional
        context of which rank is the local rank.
    """
    src_rank: int
    dest_rank: int
    comm_tag: CommTagType


CommunicationDepGraph = Mapping[
        CommunicationOpIdentifier, AbstractSet[CommunicationOpIdentifier]]


_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


# {{{ distributed graph partition

@attrs.define(frozen=True, slots=False)
class DistributedGraphPart(GraphPart):
    """For one graph part, record send/receive information for input/
    output names.

    Names that occur as keys in :attr:`name_to_recv_node` and
    :attr:`name_to_send_node` are usable as input names by other
    parts, or in the result of the computation.

    .. attribute:: name_to_recv_node
    .. attribute:: name_to_send_node
    """
    name_to_recv_node: Mapping[str, DistributedRecv]
    name_to_send_node: Mapping[str, DistributedSend]


@attrs.define(frozen=True, slots=False)
class DistributedGraphPartition(GraphPartition):
    """Store information about distributed graph partitions. This
    has the same attributes as :class:`~pytato.partition.GraphPartition`,
    however :attr:`~pytato.partition.GraphPartition.parts` now maps to
    instances of :class:`DistributedGraphPart`.
    """
    parts: Dict[PartId, DistributedGraphPart]

# }}}


# {{{ _DistributedInputReplacer

class _DistributedInputReplacer(CopyMapper):
    """Replaces part inputs with :class:`~pytato.array.Placeholder`
    instances for their assigned names. Also gathers names for
    user-supplied inputs needed by the part
    """

    def __init__(self,
                 roo_ary_to_name: Mapping[Array, str],
                 part_outputs: AbstractSet[Array],
                 ) -> None:
        super().__init__()

        self.roo_ary_to_name = roo_ary_to_name
        self.part_outputs = part_outputs

        self.user_input_names: Set[str] = set()
        self.partition_input_names: Set[str] = set()

    def map_placeholder(self, expr: Placeholder) -> Array:
        self.user_input_names.add(expr.name)
        return expr

    def map_distributed_recv(self, expr: DistributedRecv) -> Placeholder:
        name = self.roo_ary_to_name[expr]
        self.partition_input_names.add(name)
        return make_placeholder(
                name, expr.shape, expr.dtype, expr.tags,
                expr.axes)

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
            return cast(ArrayOrNames, self._cache[key])
        except KeyError:
            pass

        # If the array is an output from the current part, it would
        # be counterproductive to turn it into a placeholder: we're
        # the ones who are supposed to compute it!
        if expr not in self.part_outputs:

            name = self.roo_ary_to_name.get(expr)

            if name is not None:
                self.partition_input_names.add(name)
                return make_placeholder(
                        name, expr.shape, expr.dtype, expr.tags,
                        expr.axes)

        return cast(ArrayOrNames, super().rec(expr))

# }}}


@attrs.define(frozen=True)
class _PartCommIDs:
    """A *part*, unlike a *batch*, begins with receives and ends with sends.
    """
    recv_ids: FrozenSet[CommunicationOpIdentifier]
    send_ids: FrozenSet[CommunicationOpIdentifier]


# {{{ _make_distributed_partition

def _make_distributed_partition(
        outputs_per_part: Sequence[AbstractSet[Array]],
        part_comm_ids: Sequence[_PartCommIDs],
        roo_ary_to_name: Mapping[Array, str],
        local_recv_id_to_recv_node: Dict[CommunicationOpIdentifier, DistributedRecv],
        local_send_id_to_send_node: Dict[CommunicationOpIdentifier, DistributedSend],
        ) -> DistributedGraphPartition:
    var_name_to_result = {}
    parts: Dict[PartId, DistributedGraphPart] = {}

    for part_id, part_outputs in enumerate(outputs_per_part):
        comm_replacer = _DistributedInputReplacer(roo_ary_to_name, part_outputs)

        for val in part_outputs:
            name = roo_ary_to_name[val]
            assert name not in var_name_to_result
            var_name_to_result[name] = comm_replacer(val)

        comm_ids = part_comm_ids[part_id]

        name_to_send_node = Map({
            roo_ary_to_name[local_send_id_to_send_node[send_id].data]:
            comm_replacer.map_distributed_send(local_send_id_to_send_node[send_id])
            for send_id in comm_ids.send_ids})

        parts[part_id] = DistributedGraphPart(
                pid=part_id,
                needed_pids=frozenset({part_id - 1} if part_id else {}),
                user_input_names=frozenset(comm_replacer.user_input_names),
                partition_input_names=frozenset(comm_replacer.partition_input_names),
                output_names=frozenset({
                    roo_ary_to_name[output] for output in part_outputs}),
                name_to_recv_node=Map({
                    roo_ary_to_name[local_recv_id_to_recv_node[recv_id]]:
                    local_recv_id_to_recv_node[recv_id]
                    for recv_id in comm_ids.recv_ids}),
                name_to_send_node=name_to_send_node)

    result = DistributedGraphPartition(
            parts=parts,
            var_name_to_result=var_name_to_result,
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


# {{{ _schedule_comm_batches

def _schedule_comm_batches(
        comm_ids_to_needed_comm_ids: CommunicationDepGraph
        ) -> Sequence[AbstractSet[CommunicationOpIdentifier]]:
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
    comms_to_schedule = set(comm_ids_to_needed_comm_ids)

    all_comm_ids = frozenset(comm_ids_to_needed_comm_ids)

    # FIXME In order for this to work, comm tags must be unique
    while len(scheduled_comm_ids) < len(all_comm_ids):
        comm_ids_this_batch = {
                comm_id for comm_id in comms_to_schedule
                if comm_ids_to_needed_comm_ids[comm_id] <= scheduled_comm_ids}

        if not comm_ids_this_batch:
            raise CycleError("cycle detected in communication graph")

        scheduled_comm_ids.update(comm_ids_this_batch)
        comms_to_schedule = comms_to_schedule - comm_ids_this_batch

        comm_batches.append(comm_ids_this_batch)

    return comm_batches

# }}}


# {{{  _MaterializedArrayCollector

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class _MaterializedArrayCollector(CachedWalkMapper):
    """
    Collects all nodes that have to be materialized during code-generation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.materialized_arrays: Set[Array] = set()

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
        mpi_data_type: MPI.Datatype) -> Mapping[_KeyT, FrozenSet[_ValueT]]:
    assert mpi_data_type is None
    result = dict(dict_a)
    for key, values in dict_b.items():
        result[key] = result.get(key, frozenset()) | values
    return result

# }}}


# {{{ find_distributed_partition

def find_distributed_partition(
        mpi_communicator: MPI.Comm,
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

    On rank ``i``, collect the receives that have a data flow to that send
    in a :class:`~pytato.distributed.partition.CommunicationDepGraph`
    ``local_comm_ids_to_needed_comm_ids data structure``

    Let ``gathered_local_send_to_needed_local_recvs[i]`` be the
    ``local_comm_ids_to_needed_comm_ids`` gathered on rank ``i``. Since each
    send is carried out by exactly one rank, :math:`\bigcup_i`
    ``gathered_local_send_to_needed_local_recvs[i].keys()`` is disjoint, and
    thus simply combining all the dictionaries by key yields the rank-global
    graph of data flow between communication operations.

    Using allreduce (with disjoint union of :class:`dict` as the operation) of
    the local-pieces, this global graph will be conveyed to each rank. Each
    rank will then have the same global
    :class:`~pytato.distributed.partition.CommunicationDepGraph`
    ``comm_ids_to_needed_comm_ids``.

    .. rubric:: Step 2: Collect rank-local sends needed by each receive

    On rank ``i``, do the following:

    For each communicaton operation with destination rank ``i`` (i.e., from
    the point of view of rank ``i``, a receive), (recursively) find all
    needed communications with source rank ``i`` (i.e., from the point of
    view of rank ``i``, a send). Record these in
    :class:`~pytato.distributed.partition.CommunicationDepGraph`
    ``local_recv_id_to_needed_local_send_ids``.

    .. rubric:: Step 3: Obtain a rank-local communication dependency graph

    On rank ``i``, do the following:

    Define :class:`~pytato.distributed.partition.CommunicationDepGraph`
    ``local_comm_to_needed_local_comms`` as the key-wise disjoint union of
    ``local_comm_ids_to_needed_comm_ids`` and
    ``local_recv_id_to_needed_local_send_ids``.

    This graph now carries globally valid information on dependencies
    between communication operations taking place on the local rank.

    .. rubric:: Step 4: Partition the graph of local communication ops

    On rank ``i``, do the following:

    Compute a topological order of ``local_comm_to_needed_local_comms``.

    Starting from a single communication operation, greedily pack additional
    communication operations into the current part if:

    -  the additional operation is a send
    -  the additional operation is a receive that does not “need” (according
        to ``local_recv_id_to_needed_local_send_ids``) a send in the current part.
        Partitions *outputs* into parts. Between two parts communication
        statements (sends/receives) are scheduled.

    """
    # FIXME: Massage salvageable bits from old docstring into new.
    """
    -------

    .. note::

        The partitioning of a DAG generally does not have a unique solution.
        The heuristic employed by this partitioner is as follows:

        1. The data contained in :class:`~pytato.DistributedSend` are marked as
           *mandatory part outputs*.
        2. Based on the dependencies in *outputs*, a DAG is constructed with
           only the mandatory part outputs as the nodes.
        3. Using a topological sort the mandatory part outputs are assigned a
           "time" (an integer) such that evaluating these outputs at that time
           would preserve dependencies. We maximize the number of part outputs
           scheduled at a each "time". This requirement ensures our topological
           sort is deterministic.
        4. We then turn our attention to the other arrays that are allocated to a
           buffer. These are the materialized arrays and belong to one of the
           following classes:
           - An :class:`~pytato.Array` tagged with :class:`pytato.tags.ImplStored`.
           - The expressions in a :class:`~pytato.DictOfNamedArrays`.
        5. Based on *outputs*, we compute the predecessors of a materialized
           that were a part of the mandatory part outputs. A materialized array
           is scheduled to be evaluated in a part as soon as all of its inputs
           are available. Note that certain inputs (like
           :class:`~pytato.DistributedRecv`) might not be available until
           certain mandatory part outputs have been evaluated.
        6. From *outputs*, we can construct a DAG comprising only of mandatory
           part outputs and materialized arrays. We mark all materialized
           arrays that are being used by nodes in a part that's not the one in
           which the materialized array itself was evaluated. Such materialized
           arrays are also realized as part outputs. This is done to avoid
           recomputations.

        Knobs to tweak the partition:

        1. By removing dependencies between the mandatory part outputs, the
           resulting DAG would lead to fewer number of parts and parts with
           more number of nodes in them. Similarly, adding dependencies between
           the part outputs would lead to smaller parts.
        2. Tagging nodes with :class:~pytato.tags.ImplStored` would help in
           avoiding re-computations.
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

    # {{{ assign each materialized array to a part

    materialized_arrays_collector = _MaterializedArrayCollector()
    materialized_arrays_collector(outputs)

    sent_array_to_send_node = {
            send.data: send
            for send in lsrdg.local_send_id_to_send_node.values()}
    sent_arrays = frozenset(sent_array_to_send_node)

    received_arrays = frozenset(lsrdg.local_recv_id_to_recv_node.values())

    # While sent/received arrays may be marked as materialized, we shouldn't be
    # including them here because we're using sent/received arrays as anchors
    # to place *other* materialized data into the batches.
    materialized_arrays = (
        frozenset(materialized_arrays_collector.materialized_arrays)
        - sent_arrays
        - received_arrays)

    # "moo" for "materialized or output"
    output_arrays = set(outputs._data.values())
    moo_arrays = materialized_arrays | output_arrays

    # FIXME: This gathers up materialized_arrays recursively, leading to
    # result sizes potentially quadratic in the number of materialized arrays.
    moo_array_dep_mapper = SubsetDependencyMapper(moo_arrays)

    moo_ary_to_first_dep_send_part_id: Dict[Array, int] = {
        ary: nparts
        for ary in moo_arrays}
    for sent_ary in sent_arrays:
        for ary in moo_array_dep_mapper(sent_ary):
            moo_ary_to_first_dep_send_part_id[ary] = min(
                moo_ary_to_first_dep_send_part_id[ary],
                comm_id_to_part_id[
                    _send_to_comm_id(local_rank,
                                     sent_array_to_send_node[sent_ary])])

    if __debug__:
        recvd_array_dep_mapper = SubsetDependencyMapper(received_arrays)

        moo_ary_to_last_dep_recv_part_id: Dict[Array, int] = {
                ary: max(
                        (comm_id_to_part_id[
                            _recv_to_comm_id(local_rank,
                                            cast(DistributedRecv, recvd_ary))]
                        for recvd_ary in recvd_array_dep_mapper(ary)),
                        default=-1)
                for ary in moo_arrays
                }

        assert all(
                (
                    moo_ary_to_last_dep_recv_part_id[ary]
                    <= moo_ary_to_first_dep_send_part_id[ary])
                for ary in moo_arrays), \
            "unable to find suitable part for materialized or output array"

    # FIXME: (Seemingly) arbitrary decision, subject to future investigation.
    # Evaluation of materialized arrays is pushed as late as possible,
    # in order to minimize the amount of computation that might prevent
    # data from being sent.
    moo_ary_to_part_id: Dict[Array, int] = {
            ary: min(
                moo_ary_to_first_dep_send_part_id[ary],
                nparts-1)
            for ary in moo_arrays}

    # }}}

    sent_ary_to_part_id = {
            sent_ary: (
                comm_id_to_part_id[
                    _send_to_comm_id(local_rank, send_node)])
            for sent_ary, send_node in sent_array_to_send_node.items()}

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
    stored_ary_to_part_id = moo_ary_to_part_id.copy()
    stored_ary_to_part_id.update(sent_ary_to_part_id)
    stored_ary_to_part_id.update(recvd_ary_to_part_id)

    assert all(0 <= part_id < nparts
               for part_id in stored_ary_to_part_id.values())

    stored_arrays = set(stored_ary_to_part_id)

    # {{{ find which materialized arrays should become part outputs
    # (because they are used in not just their local part, but also others)

    direct_preds_getter = DirectPredecessorsGetter()

    def get_stored_predecessors(ary: Array) -> FrozenSet[Array]:
        if ary in stored_arrays:
            return frozenset({ary})
        else:
            return reduce(frozenset.union,
                        [get_stored_predecessors(pred)
                         for pred in direct_preds_getter(ary)],
                        frozenset())

    materialized_arrays_promoted_to_part_outputs = {
                stored_pred
                for stored_ary in stored_arrays
                for stored_pred in get_stored_predecessors(stored_ary)
                if (stored_ary_to_part_id[stored_ary]
                    != stored_ary_to_part_id[stored_pred])
                }

    stored_ary_part_outputs = (
        output_arrays
        | sent_arrays
        | materialized_arrays_promoted_to_part_outputs)

    # }}}

    out_name_gen = UniqueNameGenerator(forced_prefix="_pt_out_")
    recv_name_gen = UniqueNameGenerator(forced_prefix="_pt_recv_")
    # "roo" for "received or output"
    roo_ary_to_name = {
        ary: out_name_gen()
        for ary in materialized_arrays_promoted_to_part_outputs}
    roo_ary_to_name.update({ary: out_name_gen() for ary in sent_arrays})
    roo_ary_to_name.update({ary: name for name, ary in outputs._data.items()})
    roo_ary_to_name.update({recv: recv_name_gen() for recv in received_arrays})

    outputs_per_part: List[Set[Array]] = [set() for _pid in range(nparts)]
    for output_ary in stored_ary_part_outputs:
        outputs_per_part[stored_ary_to_part_id[output_ary]].add(output_ary)

    partition = _make_distributed_partition(
            outputs_per_part,
            part_comm_ids,
            roo_ary_to_name,
            lsrdg.local_recv_id_to_recv_node,
            lsrdg.local_send_id_to_send_node)

    from pytato.partition import _run_partition_diagnostics
    _run_partition_diagnostics(outputs, partition)

    if __debug__:
        # Avoid potentially confusing errors if one rank manages to continue
        # when another is not able.
        mpi_communicator.barrier()

    return partition

# }}}

# vim: foldmethod=marker
