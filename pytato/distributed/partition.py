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
        Sequence, Tuple, Any, Mapping, FrozenSet, Set, Dict, cast, Iterable,
        Callable, List, AbstractSet, TypeVar, TYPE_CHECKING)
from functools import cached_property

import attrs
from immutables import Map

from pymbolic.mapper.optimize import optimize_mapper
from pytools import UniqueNameGenerator
from pytools.tag import UniqueTag

from pytato.scalar_expr import SCALAR_CLASSES
from pytato.array import (Array,
                          DictOfNamedArrays, Placeholder, make_placeholder,
                          NamedArray)
from pytato.transform import (ArrayOrNames, CopyMapper, Mapper,
                              CachedWalkMapper, CopyMapperWithExtraArgs,
                              CombineMapper)
from pytato.partition import GraphPart, GraphPartition, PartId, GraphPartitioner
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
    """For one graph partition, record send/receive information for input/
    output names.

    .. attribute:: input_name_to_recv_node
    .. attribute:: output_name_to_send_node
    .. attribute:: distributed_sends
    """
    input_name_to_recv_node: Dict[str, DistributedRecv]
    output_name_to_send_node: Dict[str, DistributedSend]
    distributed_sends: List[DistributedSend]


@attrs.define(frozen=True, slots=False)
class DistributedGraphPartition(GraphPartition):
    """Store information about distributed graph partitions. This
    has the same attributes as :class:`~pytato.partition.GraphPartition`,
    however :attr:`~pytato.partition.GraphPartition.parts` now maps to
    instances of :class:`DistributedGraphPart`.
    """
    parts: Dict[PartId, DistributedGraphPart]

# }}}


# {{{ _partition_to_distributed_partition

def _map_distributed_graph_partition_nodes(
        map_array: Callable[[Array], Array],
        map_send: Callable[[DistributedSend], DistributedSend],
        gp: DistributedGraphPartition) -> DistributedGraphPartition:
    """Return a new copy of *gp* with all :class:`~pytato.Array` instances
    mapped by *map_array* and all :class:`DistributedSend` instances mapped
    by *map_send*.
    """
    from attrs import evolve as replace

    return replace(
            gp,
            var_name_to_result={name: map_array(ary)
                for name, ary in gp.var_name_to_result.items()},
            parts={
                pid: replace(part,
                    input_name_to_recv_node={
                        in_name: map_array(recv)
                        for in_name, recv in part.input_name_to_recv_node.items()},
                    output_name_to_send_node={
                        out_name: map_send(send)
                        for out_name, send in part.output_name_to_send_node.items()},
                    distributed_sends=[
                        map_send(send) for send in part.distributed_sends]
                    )
                for pid, part in gp.parts.items()
                })


class _DistributedCommReplacer(CopyMapper):
    """Mapper to process a DAG for realization of :class:`DistributedSend`
    and :class:`DistributedRecv` outside of normal code generation.

    -   Replaces :class:`DistributedRecv` with :class`~pytato.Placeholder`
        so that received data can be externally supplied, making a note
        in :attr:`input_name_to_recv_node`.

    -   Eliminates :class:`DistributedSendRefHolder` and
        :class:`DistributedSend` from the DAG, making a note of data
        to be send in :attr:`output_name_to_send_node`.
    """

    def __init__(self, dist_name_generator: UniqueNameGenerator) -> None:
        super().__init__()

        self.name_generator = dist_name_generator

        self.input_name_to_recv_node: Dict[str, DistributedRecv] = {}
        self.output_name_to_send_node: Dict[str, DistributedSend] = {}

    def map_distributed_recv(self, expr: DistributedRecv) -> Placeholder:
        # no children, no need to recurse

        new_name = self.name_generator()
        self.input_name_to_recv_node[new_name] = expr
        return make_placeholder(new_name, self.rec_idx_or_size_tuple(expr.shape),
                                expr.dtype, tags=expr.tags, axes=expr.axes)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> Array:
        raise ValueError("DistributedSendRefHolder should not occur in partitioned "
                "graphs")

    def map_distributed_send(self, expr: DistributedSend) -> DistributedSend:
        new_send = DistributedSend(
                data=self.rec(expr.data),
                dest_rank=expr.dest_rank,
                comm_tag=expr.comm_tag,
                tags=expr.tags)

        new_name = self.name_generator()
        self.output_name_to_send_node[new_name] = new_send

        return new_send


def _partition_to_distributed_partition(partition: GraphPartition,
        pid_to_distributed_sends: Dict[PartId, List[DistributedSend]]) -> \
            DistributedGraphPartition:
    var_name_to_result = {}
    parts: Dict[PartId, DistributedGraphPart] = {}

    dist_name_generator = UniqueNameGenerator(forced_prefix="_pt_dist_")

    for part in sorted(partition.parts.values(),
                       key=lambda k: sorted(k.output_names)):
        comm_replacer = _DistributedCommReplacer(dist_name_generator)
        part_results = {
                var_name: comm_replacer(partition.var_name_to_result[var_name])
                for var_name in sorted(part.output_names)}

        dist_sends = [
                comm_replacer.map_distributed_send(send)
                for send in pid_to_distributed_sends.get(part.pid, [])]

        part_results.update({
            name: send_node.data
            for name, send_node in
            comm_replacer.output_name_to_send_node.items()})

        parts[part.pid] = DistributedGraphPart(
                pid=part.pid,
                needed_pids=part.needed_pids,
                user_input_names=part.user_input_names,
                partition_input_names=(part.partition_input_names
                    | frozenset(comm_replacer.input_name_to_recv_node)),
                output_names=(part.output_names
                    | frozenset(comm_replacer.output_name_to_send_node)),
                distributed_sends=dist_sends,

                input_name_to_recv_node=comm_replacer.input_name_to_recv_node,
                output_name_to_send_node=comm_replacer.output_name_to_send_node)

        for name, val in part_results.items():
            assert name not in var_name_to_result
            var_name_to_result[name] = val

    result = DistributedGraphPartition(
            parts=parts,
            var_name_to_result=var_name_to_result,
            toposorted_part_ids=partition.toposorted_part_ids)

    if __debug__:
        # Check disjointness again since we replaced a few nodes.
        from pytato.partition import _check_partition_disjointness
        _check_partition_disjointness(result)

    return result

# }}}


# {{{ helpers for find_distributed_partition

class _DistributedGraphPartitioner(GraphPartitioner):

    def __init__(self, get_part_id: Callable[[ArrayOrNames], PartId]) -> None:
        super().__init__(get_part_id)
        self.pid_to_dist_sends: Dict[PartId, List[DistributedSend]] = {}

    def map_distributed_send_ref_holder(
                self, expr: DistributedSendRefHolder, *args: Any) -> Any:
        send_part_id = self.get_part_id(expr.send.data)

        self.pid_to_dist_sends.setdefault(send_part_id, []).append(
                DistributedSend(
                    data=self.rec(expr.send.data),
                    dest_rank=expr.send.dest_rank,
                    comm_tag=expr.send.comm_tag,
                    tags=expr.send.tags))

        return self.rec(expr.passthrough_data)

    def make_partition(self, outputs: DictOfNamedArrays) \
            -> DistributedGraphPartition:

        partition = super().make_partition(outputs)
        return _partition_to_distributed_partition(partition, self.pid_to_dist_sends)

# }}}


# {{{ _LocalSendRecvDepGatherer

def _send_to_comm_id(
        local_rank: int, send: DistributedSend) -> CommunicationOpIdentifier:
    return CommunicationOpIdentifier(
        src_rank=local_rank,
        dest_rank=send.dest_rank,
        comm_tag=send.comm_tag)


def _recv_to_comm_id(
        local_rank: int, recv: DistributedRecv) -> CommunicationOpIdentifier:
    return CommunicationOpIdentifier(
        src_rank=recv.src_rank,
        dest_rank=local_rank,
        comm_tag=recv.comm_tag)


class _LocalSendRecvDepGatherer(
        CombineMapper[FrozenSet[CommunicationOpIdentifier]]):
    def __init__(self, local_rank: int) -> None:
        super().__init__()
        self.local_send_id_to_needed_local_recv_ids: \
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

        if send_id in self.local_send_id_to_needed_local_recv_ids:
            raise ValueError(f"Multiple sends found for '{send_id}'")

        self.local_send_id_to_needed_local_recv_ids[send_id] = \
                self.rec(expr.send.data)

        assert send_id not in self.local_send_id_to_send_node
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
            raise ValueError(f"Multiple receives found for '{recv_id}'")

        self.local_recv_id_to_recv_node[recv_id] = expr

        return frozenset({recv_id})

# }}}


# {{{ _make_local_comm_batches

@attrs.define(frozen=True)
class _CommunicationBatch:
    receives: FrozenSet[CommunicationOpIdentifier]
    sends: FrozenSet[CommunicationOpIdentifier]


def _make_local_comm_batches(
        local_rank: int,
        local_comm_to_needed_local_comms: CommunicationDepGraph
        ) -> Tuple[
                Sequence[_CommunicationBatch],
                Mapping[CommunicationOpIdentifier, int]]:
    # FIXME: I'm an O(n^2) algorithm.

    current_batch_nr = 0
    comm_id_to_batch_nr = {}

    scheduled_comms: Set[CommunicationOpIdentifier] = set()
    comms_to_schedule = set(local_comm_to_needed_local_comms)

    all_comms = frozenset(local_comm_to_needed_local_comms)

    comm_batches: List[_CommunicationBatch] = []

    while len(scheduled_comms) < len(all_comms):
        # {{{ eagerly schedule nodes whose predecessors have been scheduled

        recvs_this_batch = {
                comm for comm in comms_to_schedule
                if comm.dest_rank == local_rank  # it's a receive
                if local_comm_to_needed_local_comms[comm] <= scheduled_comms}

        almost_scheduled_comms = scheduled_comms | recvs_this_batch

        sends_this_batch = {
                comm for comm in comms_to_schedule
                if comm.src_rank == local_rank  # it's a send
                if local_comm_to_needed_local_comms[comm] <= almost_scheduled_comms}

        comm_batches.append(_CommunicationBatch(
            receives=frozenset(recvs_this_batch),
            sends=frozenset(sends_this_batch)))

        comms_this_batch = recvs_this_batch | sends_this_batch
        for node in comms_this_batch:
            assert node not in comm_id_to_batch_nr
            comm_id_to_batch_nr[node] = current_batch_nr

        scheduled_comms.update(comms_this_batch)
        comms_to_schedule = comms_to_schedule - comms_this_batch

        current_batch_nr += 1

        # }}}

    assert set(comm_id_to_batch_nr.values()) == set(range(current_batch_nr))

    return comm_batches, comm_id_to_batch_nr

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

        if isinstance(expr, DictOfNamedArrays):
            for _, subexpr in sorted(expr._data.items()):
                assert isinstance(subexpr, Array)
                self.materialized_arrays.add(subexpr)

# }}}


# {{{ other old stuff

class _DominantMaterializedPredecessorsCollector(Mapper):
    """
    A Mapper whose mapper method for a node returns the materialized predecessors
    just after the point the node is evaluated.
    """
    def __init__(self, materialized_arrays: FrozenSet[Array]) -> None:
        super().__init__()
        self.materialized_arrays = materialized_arrays
        self.cache: Dict[ArrayOrNames, FrozenSet[Array]] = {}

    def _combine(self, values: Iterable[Array]) -> FrozenSet[Array]:
        from functools import reduce
        return reduce(frozenset.union,
                      (self.rec(v) for v in values),
                      frozenset())

    # type-ignore reason: return type not compatible with Mapper.rec's type
    def rec(self, expr: ArrayOrNames) -> FrozenSet[Array]:  # type: ignore[override]
        try:
            return self.cache[expr]
        except KeyError:
            result: FrozenSet[Array] = super().rec(expr)
            self.cache[expr] = result
            return result

    @cached_property
    def direct_preds_getter(self) -> DirectPredecessorsGetter:
        return DirectPredecessorsGetter()

    def _map_generic_node(self, expr: Array) -> FrozenSet[Array]:
        direct_preds = self.direct_preds_getter(expr)

        if expr in self.materialized_arrays:
            return frozenset([expr])
        else:
            return self._combine(direct_preds)

    map_placeholder = _map_generic_node
    map_data_wrapper = _map_generic_node
    map_size_param = _map_generic_node

    map_index_lambda = _map_generic_node
    map_stack = _map_generic_node
    map_concatenate = _map_generic_node
    map_roll = _map_generic_node
    map_axis_permutation = _map_generic_node
    map_basic_index = _map_generic_node
    map_contiguous_advanced_index = _map_generic_node
    map_non_contiguous_advanced_index = _map_generic_node
    map_reshape = _map_generic_node
    map_einsum = _map_generic_node
    map_distributed_recv = _map_generic_node

    def map_named_array(self, expr: NamedArray) -> FrozenSet[Array]:
        raise NotImplementedError("only LoopyCallResult named array"
                                  " supported for now.")

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays
                                 ) -> FrozenSet[Array]:
        raise NotImplementedError("Dict of named arrays not (yet) implemented")

    def map_loopy_call_result(self, expr: NamedArray) -> FrozenSet[Array]:
        # ``loopy call result` is always materialized. However, make sure to
        # traverse its arguments.
        assert expr in self.materialized_arrays
        return self._map_generic_node(expr)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> FrozenSet[Array]:
        return self.rec(expr.passthrough_data)


@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class _DominantMaterializedPredecessorsRecorder(CachedWalkMapper):
    """
    For each node in an expression graph, this mapper records the dominant
    predecessors of each node of an expression graph into
    :attr:`array_to_mat_preds`.
    """
    def __init__(self, mat_preds_getter: Callable[[Array], FrozenSet[Array]]
                 ) -> None:
        super().__init__()
        self.mat_preds_getter = mat_preds_getter
        self.array_to_mat_preds: Dict[Array, FrozenSet[Array]] = {}

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self, expr: ArrayOrNames) -> int:  # type: ignore[override]
        return id(expr)

    @cached_property
    def direct_preds_getter(self) -> DirectPredecessorsGetter:
        return DirectPredecessorsGetter()

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def post_visit(self, expr: Any) -> None:  # type: ignore[override]
        from functools import reduce
        if isinstance(expr, Array):
            self.array_to_mat_preds[expr] = reduce(
                frozenset.union,
                (self.mat_preds_getter(pred)
                 for pred in self.direct_preds_getter(expr)),
                frozenset())


def _get_array_to_dominant_materialized_deps(
        outputs: DictOfNamedArrays,
        materialized_arrays: FrozenSet[Array]) -> Map[Array, FrozenSet[Array]]:
    """
    Returns a mapping from each node in the DAG *outputs* to a :class:`frozenset`
    of its dominant materialized predecessors.
    """

    dominant_materialized_deps = _DominantMaterializedPredecessorsCollector(
        materialized_arrays)
    dominant_materialized_deps_recorder = (
        _DominantMaterializedPredecessorsRecorder(dominant_materialized_deps))
    dominant_materialized_deps_recorder(outputs)
    return Map(dominant_materialized_deps_recorder.array_to_mat_preds)


def _get_materialized_arrays_promoted_to_partition_outputs(
        ary_to_dominant_stored_preds: Mapping[Array, FrozenSet[Array]],
        stored_ary_to_part_id: Mapping[Array, int],
        materialized_arrays: FrozenSet[Array]
) -> FrozenSet[Array]:
    """
    Returns a :class:`frozenset` of materialized arrays that are used by
    multiple partitions. Materialized arrays that are used by multiple
    partitions are special in that they *must* be promoted as the outputs of a
    partition.

    Invoked as an intermediate step in :func:`find_distributed_partition`.

    :arg ary_to_dominant_stored_preds: A mapping from array to the dominant
        stored predecessors. A stored array can be either a mandatory partition
        output or a materialized array as indicated by the user.
    """
    materialized_ary_to_part_id_users: Dict[Array, Set[int]] = {}

    for ary in stored_ary_to_part_id:
        stored_preds = ary_to_dominant_stored_preds[ary]
        for pred in stored_preds:
            if pred in materialized_arrays:
                (materialized_ary_to_part_id_users
                 .setdefault(pred, set())
                 .add(stored_ary_to_part_id[ary]))

    return frozenset({ary
                      for ary, users in materialized_ary_to_part_id_users.items()
                      if users != {stored_ary_to_part_id[ary]}})


@attrs.define(frozen=True, eq=True, repr=True)
class PartIDTag(UniqueTag):
    """
    A tag applicable to a :class:`pytato.Array` recording to which part the
    array belongs.
    """
    part_id: int


class _PartIDTagAssigner(CopyMapperWithExtraArgs):
    """
    Used by :func:`find_distributed_partition` to assign each array
    node a :class:`PartIDTag`.
    """
    def __init__(self,
                 stored_array_to_part_id: Mapping[Array, int],
                 partition_outputs: FrozenSet[Array]) -> None:
        self.stored_array_to_part_id = stored_array_to_part_id
        self.partition_outputs = partition_outputs

        # type-ignore reason: incompatible  attribute type wrt base.
        self._cache: Dict[Tuple[ArrayOrNames, int],
                          Any] = {}  # type: ignore[assignment]

    # type-ignore-reason: incompatible with super class
    def get_cache_key(self,  # type: ignore[override]
                      expr: ArrayOrNames,
                      user_part_id: int
                      ) -> Tuple[ArrayOrNames, int]:

        return (expr, user_part_id)

    # type-ignore-reason: incompatible with super class
    def rec(self,  # type: ignore[override]
            expr: ArrayOrNames,
            user_part_id: int) -> Any:
        key = self.get_cache_key(expr, user_part_id)
        try:
            return self._cache[key]
        except KeyError:
            if isinstance(expr, Array):
                if expr in self.stored_array_to_part_id:
                    assert ((self.stored_array_to_part_id[expr]
                             == user_part_id)
                            or expr in self.partition_outputs)
                    # at stored array the part id changes
                    user_part_id = self.stored_array_to_part_id[expr]

                expr = expr.tagged(PartIDTag(user_part_id))

            result = super().rec(expr, user_part_id)
            self._cache[key] = result
            return result


def _remove_part_id_tag(ary: ArrayOrNames) -> Array:
    assert isinstance(ary, Array)

    # Spurious assignment because of
    # https://github.com/python/mypy/issues/12626
    result: Array = ary.without_tags(ary.tags_of_type(PartIDTag))
    return result

# }}}


# {{{ _dict_union_mpi

def _dict_union_mpi(
        dict_a: Mapping[_KeyT, _ValueT], dict_b: Mapping[_KeyT, _ValueT],
        mpi_data_type: MPI.Datatype) -> Mapping[_KeyT, _ValueT]:
    assert mpi_data_type is None
    result = dict(dict_a)
    result.update(dict_b)
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
    ``local_send_id_to_needed_local_recv_ids data structure``

    Let ``gathered_local_send_to_needed_local_recvs[i]`` be the
    ``local_send_id_to_needed_local_recv_ids`` gathered on rank ``i``. Since each
    send is carried out by exactly one rank, :math:`\bigcup_i`
    ``gathered_local_send_to_needed_local_recvs[i].keys()`` is disjoint, and
    thus simply combining all the dictionaries by key yields the rank-global
    graph of data flow between communication operations.

    Using allreduce (with disjoint union of :class:`dict` as the operation) of
    the local-pieces, this global graph will be conveyed to each rank. Each
    rank will then have the same global
    :class:`~pytato.distributed.partition.CommunicationDepGraph`
    ``comm_to_needed_comms``.

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
    ``local_send_id_to_needed_local_recv_ids`` and
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
    from pytato.array import make_dict_of_named_arrays
    from pytato.partition import find_partition

    import mpi4py.MPI as MPI

    local_rank = mpi_communicator.rank

    # {{{ Step 1: find local_send_id_to_needed_local_recv_ids

    lsrdg = _LocalSendRecvDepGatherer(local_rank=local_rank)
    lsrdg(outputs)
    local_send_id_to_needed_local_recv_ids = \
            lsrdg.local_send_id_to_needed_local_recv_ids

    dict_union_mpi_op = MPI.Op.Create(
            # type ignore reason: mpi4py misdeclares op functions as returning
            # None.
            _dict_union_mpi,  # type: ignore[arg-type]
            commute=True)
    try:
        comm_to_needed_comms = mpi_communicator.allreduce(
                local_send_id_to_needed_local_recv_ids, dict_union_mpi_op)
    finally:
        dict_union_mpi_op.Free()

    # }}}

    # {{{ Step 2: collect local_recv_id_to_needed_local_send_ids

    def get_needed_local_sends(
            comm: CommunicationOpIdentifier
            ) -> FrozenSet[CommunicationOpIdentifier]:
        if comm.src_rank == local_rank:
            # We'll stop here, we know our local data flow dependencies.
            return frozenset({comm})

        return reduce(frozenset.union,
                      (get_needed_local_sends(needed_comm)
                       for needed_comm in comm_to_needed_comms[comm]),
                      frozenset())

    local_recv_id_to_needed_local_send_ids = {
            recv_id: get_needed_local_sends(recv_id)
            for recv_id in lsrdg.local_recv_id_to_recv_node}

    # }}}

    # {{{ Step 3: Obtain a rank-local communication graph

    local_comm_to_needed_local_comms = local_recv_id_to_needed_local_send_ids.copy()
    local_comm_to_needed_local_comms.update(local_send_id_to_needed_local_recv_ids)

    if __debug__:
        from pytools.graph import validate_graph
        validate_graph(local_comm_to_needed_local_comms)

        for comm in local_comm_to_needed_local_comms:
            if comm.src_rank == comm.dest_rank:
                # FIMXE: These should just be directly connected, without
                # involving communication.
                raise ValueError("found a rank-local send")

    # }}}

    # {{{ Step 4: Partition the graph of local communicaton ops

    # The comm_batches correspond one-to-one to DistributedGraphParts
    # in the output.
    comm_batches, comm_id_to_part_id = _make_local_comm_batches(
            local_rank, local_comm_to_needed_local_comms)

    # FIXME?: comm_batches isn't being used for anything

    nparts = max(part_id for part_id in comm_id_to_part_id.values()) + 1

    # }}}

    # {{{ assign each materialized array to a batch/part

    materialized_arrays_collector = _MaterializedArrayCollector()
    materialized_arrays_collector(outputs)

    sent_array_to_send_node = {
            send.data: send
            for send in lsrdg.local_send_id_to_send_node.values()}
    sent_arrays = frozenset(sent_array_to_send_node)

    # While sent arrays are materialized, we shouldn't be including them
    # here because we're using sent arrays (which need to be materialized
    # in order to send them) as anchors to place *other* materialized data
    # into the batches.
    materialized_arrays = frozenset(
        materialized_arrays_collector.materialized_arrays) - sent_arrays

    received_arrays = frozenset(lsrdg.local_recv_id_to_recv_node.values())

    sent_array_dep_mapper = SubsetDependencyMapper(sent_arrays)
    recvd_array_dep_mapper = SubsetDependencyMapper(received_arrays)

    materialized_ary_to_part_id: Dict[Array, int] = {
            ary: max(
                max(
                    (comm_id_to_part_id[
                        _send_to_comm_id(local_rank,
                                         sent_array_to_send_node[sent_array])]
                    for sent_array in sent_array_dep_mapper(ary)),
                    default=nparts-1),
                max(
                    (comm_id_to_part_id[
                        _recv_to_comm_id(local_rank,
                                         cast(DistributedRecv, recvd_array))]
                    for recvd_array in recvd_array_dep_mapper(ary)),
                    default=nparts-1)
                )
            for ary in materialized_arrays
            }

    # }}}

    sent_ary_to_part_id = {
            sent_ary: comm_id_to_part_id[
                _send_to_comm_id(local_rank, send_node)]
            for sent_ary, send_node in sent_array_to_send_node.items()}

    stored_ary_to_part_id = materialized_ary_to_part_id.copy()
    stored_ary_to_part_id.update(sent_ary_to_part_id)

    # {{{ find which materialized arrays have users in multiple parts
    # (and promote them to part outputs)

    ary_to_dominant_materialized_deps = (
        _get_array_to_dominant_materialized_deps(outputs,
                                                      (materialized_arrays
                                                       | sent_arrays)))

    materialized_arrays_realized_as_partition_outputs = (
        _get_materialized_arrays_promoted_to_partition_outputs(
            ary_to_dominant_materialized_deps,
            stored_ary_to_part_id,
            materialized_arrays))

    # }}}

    # {{{ tag each node with its part ID

    # Why is this necessary? (I.e. isn't the mapping *stored_ary_to_part_id* enough?)
    # By assigning tags we also duplicate the non-materialized nodes that are
    # to be made available in multiple parts. Parts being disjoint is one of
    # the requirements of *find_partition*.
    part_id_tag_assigner = _PartIDTagAssigner(
        stored_ary_to_part_id,
        sent_arrays | materialized_arrays_realized_as_partition_outputs)

    outputs_tagged_with_part_id = make_dict_of_named_arrays({
        name: part_id_tag_assigner(subexpr, stored_ary_to_part_id[subexpr])
        for name, subexpr in outputs._data.items()})

    # }}}

    def get_part_id(expr: ArrayOrNames) -> int:
        if not isinstance(expr, Array):
            raise NotImplementedError("find_distributed_partition"
                                      " cannot partition DictOfNamedArrays")
        assert isinstance(expr, Array)
        tag, = expr.tags_of_type(PartIDTag)
        assert isinstance(tag, PartIDTag)
        return tag.part_id

    gp = cast(DistributedGraphPartition,
                find_partition(outputs_tagged_with_part_id,
                               get_part_id,
                               _DistributedGraphPartitioner))

    # Remove PartIDTag from arrays that may be returned from the evaluation
    # of the partitioned graph. If we don't, those may end up on inputs to
    # another graph, which may also get partitioned, which will endlessly
    # confuse that subsequent partitioning process. In addition, those
    # tags may cause arrays to look spuriously different, defeating
    # caching.
    # See https://github.com/inducer/pytato/issues/307 for further discussion.

    # Note that it does not suffice to remove those tags from just, say,
    # var_name_to_result: This may produce inconsistent Placeholder instances.
    # For the same reason, we need to use the same mapper for all nodes.
    from pytato.transform import CachedMapAndCopyMapper
    cmac = CachedMapAndCopyMapper(_remove_part_id_tag)

    def map_array(ary: Array) -> Array:
        result = cmac(ary)
        assert isinstance(result, Array)
        return result

    def map_send(send: DistributedSend) -> DistributedSend:
        return send.copy(data=cmac(send.data))

    return _map_distributed_graph_partition_nodes(map_array, map_send, gp)

# }}}

# vim: foldmethod=marker
