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

from typing import (Any, Dict, Hashable, Tuple, Optional, Set,  # noqa: F401
                    List, FrozenSet, Callable, cast, Mapping, Iterable, Sequence,
                    ClassVar, TYPE_CHECKING
                    )  # Mapping required by sphinx
from immutables import Map

import attrs

from pytools import UniqueNameGenerator
from pytools.tag import Taggable, UniqueTag, Tag
from pytato.array import (Array, _SuppliedShapeAndDtypeMixin,
                          DictOfNamedArrays, ShapeType, Placeholder,
                          make_placeholder, _get_default_axes, AxesT,
                          NamedArray)
from pytato.transform import (ArrayOrNames, CopyMapper, Mapper,
                              CachedWalkMapper, CopyMapperWithExtraArgs,
                              CombineMapper)

from pytato.partition import GraphPart, GraphPartition, PartId, GraphPartitioner
from pytato.target import BoundProgram
from pytato.analysis import DirectPredecessorsGetter
from functools import cached_property
from pytato.scalar_expr import SCALAR_CLASSES, INT_CLASSES

import numpy as np

if TYPE_CHECKING:
    import mpi4py.MPI


__doc__ = r"""
Distributed-memory evaluation of expression graphs is accomplished
by :ref:`partitioning <partitioning>` the graph to reveal communication-free
pieces of the computation. Communication (i.e. sending/receiving data) is then
accomplished at the boundaries of the parts of the resulting graph partitioning.

Recall the requirement for partitioning that, "no part may depend on its own
outputs as inputs". That sounds obvious, but in the distributed-memory case,
this is harder to decide than it looks, since we do not have full knowledge of
the computation graph.  Edges go off to other nodes and then come back.

As a first step towards making this tractable, we currently strengthen the
requirement to create partition boundaries on every edge that goes between
nodes that are/are not a dependency of a receive or that feed/do not feed a send.

.. currentmodule:: pytato
.. autoclass:: DistributedSend
.. autoclass:: DistributedSendRefHolder
.. autoclass:: DistributedRecv

.. autofunction:: make_distributed_send
.. autofunction:: staple_distributed_send
.. autofunction:: make_distributed_recv

.. currentmodule:: pytato.distributed

.. autoclass:: DistributedGraphPart
.. autoclass:: DistributedGraphPartition
.. autoclass:: verify_distributed_partition

.. currentmodule:: pytato

.. autofunction:: find_distributed_partition
.. autofunction:: number_distributed_tags
.. autofunction:: execute_distributed_partition

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. class:: Tag

    See :class:`pytools.tag.Tag`.

.. class:: CommTagType

    A type representing a communication tag.

.. class:: ShapeType

    A type representing a shape.

.. class:: AxesT

    A :class:`tuple` of :class:`Axis` objects.
"""


# {{{ distributed node types

CommTagType = Hashable


class DistributedSend(Taggable):
    """Class representing a distributed send operation.

    .. attribute:: data

        The :class:`~pytato.Array` to be sent.

    .. attribute:: dest_rank

        An :class:`int`. The rank to which :attr:`data` is to be sent.

    .. attribute:: comm_tag

        A hashable, picklable object to serve as a 'tag' for the communication.
        Only a :class:`DistributedRecv` with the same tag will be able to
        receive the data being sent here.
    """

    def __init__(self, data: Array, dest_rank: int, comm_tag: CommTagType,
                 tags: FrozenSet[Tag] = frozenset()) -> None:
        super().__init__(tags=tags)
        self.data = data
        self.dest_rank = dest_rank
        self.comm_tag = comm_tag

    def __hash__(self) -> int:
        return (
                hash(self.__class__)
                ^ hash(self.data)
                ^ hash(self.dest_rank)
                ^ hash(self.comm_tag)
                ^ hash(self.tags)
                )

    def __eq__(self, other: Any) -> bool:
        return (
                self.__class__ is other.__class__
                and self.data == other.data
                and self.dest_rank == other.dest_rank
                and self.comm_tag == other.comm_tag
                and self.tags == other.tags)

    def _with_new_tags(self, tags: FrozenSet[Tag]) -> DistributedSend:
        return DistributedSend(
                data=self.data,
                dest_rank=self.dest_rank,
                comm_tag=self.comm_tag,
                tags=tags)

    def copy(self, **kwargs: Any) -> DistributedSend:
        data: Optional[Array] = kwargs.get("data")
        dest_rank: Optional[int] = kwargs.get("dest_rank")
        comm_tag: Optional[CommTagType] = kwargs.get("comm_tag")
        tags = cast(FrozenSet[Tag], kwargs.get("tags"))
        return type(self)(
                data=data if data is not None else self.data,
                dest_rank=dest_rank if dest_rank is not None else self.dest_rank,
                comm_tag=comm_tag if comm_tag is not None else self.comm_tag,
                tags=tags if tags is not None else self.tags)

    def __repr__(self) -> str:
        # self.data takes a lot of space, shorten it
        return (f"DistributedSend(data={self.data.__class__} "
                f"at {hex(id(self.data))}, "
                f"dest_rank={self.dest_rank}, "
                f"tags={self.tags}, comm_tag={self.comm_tag})")


@attrs.define(frozen=True, eq=False, repr=False, init=False)
class DistributedSendRefHolder(Array):
    """A node acting as an identity on :attr:`passthrough_data` while also holding
    a reference to a :class:`DistributedSend` in :attr:`send`. Since
    :mod:`pytato` represents data flow, and since no data flows 'out'
    of a :class:`DistributedSend`, no node in all of :mod:`pytato` has
    a good reason to hold a reference to a send node, since there is
    no useful result of a send (at least of an :class:`~pytato.Array` type).

    This is where this node type comes in. Its value is the same as that of
    :attr:`passthrough_data`, *and* it holds a reference to the send node.

    .. note::

        This all seems a wee bit inelegant, but nobody who has written
        or reviewed this code so far had a better idea. If you do, please speak up!

    .. attribute:: send

        The :class:`DistributedSend` to which a reference is to be held.

    .. attribute:: passthrough_data

        A :class:`~pytato.Array`. The value of this node.

    .. note::

        It is the user's responsibility to ensure matching sends and receives
        are part of the computation graph on all ranks. If this rule is not heeded,
        undefined behavior (in particular deadlock) may result.
        Notably, by the nature of the data flow graph built by :mod:`pytato`,
        unused results do not appear in the graph. It is thus possible for a
        :class:`DistributedSendRefHolder` to be constructed and yet to not
        become part of the graph constructed by the user.
    """
    send: DistributedSend
    passthrough_data: Array

    _mapper_method: ClassVar[str] = "map_distributed_send_ref_holder"
    _fields: ClassVar[Tuple[str, ...]] = Array._fields + ("passthrough_data", "send")

    def __init__(self, send: DistributedSend, passthrough_data: Array,
                 tags: FrozenSet[Tag] = frozenset()) -> None:
        super().__init__(axes=passthrough_data.axes, tags=tags)
        object.__setattr__(self, "send", send)
        object.__setattr__(self, "passthrough_data", passthrough_data)

    @property
    def shape(self) -> ShapeType:
        return self.passthrough_data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.passthrough_data.dtype

    def copy(self, **kwargs: Any) -> DistributedSendRefHolder:
        # override 'Array.copy' because
        # 'DistributedSendRefHolder.axes' is a read-only field.
        send = kwargs.pop("send", self.send)
        passthrough_data = kwargs.pop("passthrough_data", self.passthrough_data)
        tags = kwargs.pop("tags", self.tags)

        if kwargs:
            raise ValueError("Cannot assign"
                             f" DistributedSendRefHolder.'{set(kwargs)}'")
        return DistributedSendRefHolder(send,
                                        passthrough_data,
                                        tags)


@attrs.define(frozen=True, eq=False)
class DistributedRecv(_SuppliedShapeAndDtypeMixin, Array):
    """Class representing a distributed receive operation.

    .. attribute:: src_rank

        An :class:`int`. The rank from which an array is to be received.

    .. attribute:: comm_tag

        A hashable, picklable object to serve as a 'tag' for the communication.
        Only a :class:`DistributedSend` with the same tag will be able to
        send the data being received here.

    .. attribute:: shape
    .. attribute:: dtype

    .. note::

        It is the user's responsibility to ensure matching sends and receives
        are part of the computation graph on all ranks. If this rule is not heeded,
        undefined behavior (in particular deadlock) may result.
        Notably, by the nature of the data flow graph built by :mod:`pytato`,
        unused results do not appear in the graph. It is thus possible for a
        :class:`DistributedRecv` to be constructed and yet to not become part
        of the graph constructed by the user.
    """
    src_rank: int
    comm_tag: CommTagType
    _shape: ShapeType
    _dtype: Any  # FIXME: sphinx does not like `_dtype: _dtype_any`

    _fields: ClassVar[Tuple[str, ...]] = Array._fields + ("shape", "dtype",
                                                          "src_rank", "comm_tag")
    _mapper_method: ClassVar[str] = "map_distributed_recv"


def make_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          send_tags: FrozenSet[Tag] = frozenset()) -> \
         DistributedSend:
    """Make a :class:`DistributedSend` object."""
    return DistributedSend(sent_data, dest_rank, comm_tag, send_tags)


def staple_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          stapled_to: Array, *,
                          send_tags: FrozenSet[Tag] = frozenset(),
                          ref_holder_tags: FrozenSet[Tag] = frozenset()) -> \
         DistributedSendRefHolder:
    """Make a :class:`DistributedSend` object wrapped in a
    :class:`DistributedSendRefHolder` object."""
    return DistributedSendRefHolder(
            DistributedSend(sent_data, dest_rank, comm_tag, send_tags),
            stapled_to, tags=ref_holder_tags)


def make_distributed_recv(src_rank: int, comm_tag: CommTagType,
                          shape: ShapeType, dtype: Any,
                          axes: Optional[AxesT] = None,
                          tags: FrozenSet[Tag] = frozenset()
                          ) -> DistributedRecv:
    """Make a :class:`DistributedRecv` object."""
    if axes is None:
        axes = _get_default_axes(len(shape))
    dtype = np.dtype(dtype)
    return DistributedRecv(src_rank, comm_tag, shape, dtype, tags=tags, axes=axes)

# }}}


# {{{ distributed info collection

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


def _map_distributed_graph_partion_nodes(
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


def _gather_distributed_comm_info(partition: GraphPartition,
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

        from pytato.distributed import DistributedSend
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
        return _gather_distributed_comm_info(partition, self.pid_to_dist_sends)


class _MandatoryPartitionOutputsCollector(CombineMapper[FrozenSet[Array]]):
    """
    Collects all nodes that, after partitioning, are necessarily outputs
    of the partition to which they belong.
    """
    def __init__(self) -> None:
        super().__init__()
        self.partition_outputs: Set[Array] = set()

    def combine(self, *args: FrozenSet[Array]) -> FrozenSet[Array]:
        from functools import reduce
        return reduce(frozenset.union, args, frozenset())

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> FrozenSet[Array]:
        return self.combine(frozenset([expr.send.data]),
                            super().map_distributed_send_ref_holder(expr))

    def _map_input_base(self, expr: Array) -> FrozenSet[Array]:
        return frozenset()

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base
    map_distributed_recv = _map_input_base


class _MaterializedArrayCollector(CachedWalkMapper):
    """
    Collects all nodes that have to be materialized during code-generation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.materialized_arrays: Set[Array] = set()

    def post_visit(self, expr: Any) -> None:
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

    @cached_property
    def direct_preds_getter(self) -> DirectPredecessorsGetter:
        return DirectPredecessorsGetter()

    def post_visit(self, expr: Any) -> None:
        from functools import reduce
        if isinstance(expr, Array):
            self.array_to_mat_preds[expr] = reduce(
                frozenset.union,
                (self.mat_preds_getter(pred)
                 for pred in self.direct_preds_getter(expr)),
                frozenset())


def _linearly_schedule_batches(
        predecessors: Map[Array, FrozenSet[Array]]) -> Map[Array, int]:
    """
    Used by :func:`find_distributed_partition`. Based on the dependencies in
    *predecessors*, each node is assigned a time such that evaluating the array
    at that point in time would not violate dependencies. This "time" or "batch
    number" is then used as a partition ID.
    """
    from functools import reduce
    current_time = 0
    ary_to_time = {}
    scheduled_nodes: Set[Array] = set()
    all_nodes = frozenset(predecessors)

    # assert that the keys contain all the nodes
    assert reduce(frozenset.union,
                  predecessors.values(),
                  cast(FrozenSet[Array], frozenset())) <= frozenset(predecessors)

    while len(scheduled_nodes) < len(all_nodes):
        # {{{ eagerly schedule nodes whose predecessors have been scheduled

        nodes_to_schedule = {node
                             for node, preds in predecessors.items()
                             if ((node not in scheduled_nodes)
                                 and (preds <= scheduled_nodes))}
        for node in nodes_to_schedule:
            assert node not in ary_to_time
            ary_to_time[node] = current_time

        scheduled_nodes.update(nodes_to_schedule)

        current_time += 1

        # }}}

    assert set(ary_to_time.values()) == set(range(current_time))
    return Map(ary_to_time)


def _assign_materialized_arrays_to_part_id(
        materialized_arrays: FrozenSet[Array],
        array_to_output_deps: Mapping[Array, FrozenSet[Array]],
        outputs_to_part_id: Mapping[Array, int]
) -> Map[Array, int]:
    """
    Returns a mapping from a materialized array to the part's ID where all the
    inputs of the array expression are available.

    Invoked as an intermediate step in :func:`find_distributed_partition`.

    .. note::

        In this heuristic we compute the materialized array as soon as its
        inputs are available. In some cases it might be worth exploring
        schedules where the evaluation of an array is delayed until one of
        its users demand it.
    """

    materialized_array_to_part_id: Dict[Array, int] = {}

    for ary in materialized_arrays:
        materialized_array_to_part_id[ary] = max(
            (outputs_to_part_id[dep]
             for dep in array_to_output_deps[ary]),
            default=-1) + 1

    return Map(materialized_array_to_part_id)


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
    def cache_key(self,  # type: ignore[override]
                  expr: ArrayOrNames,
                  user_part_id: int
                  ) -> Tuple[ArrayOrNames, int]:

        return (expr, user_part_id)

    # type-ignore-reason: incompatible with super class
    def rec(self,  # type: ignore[override]
            expr: ArrayOrNames,
            user_part_id: int) -> Any:
        key = self.cache_key(expr, user_part_id)
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


# {{{ verify_distributed_partition

@attrs.define(frozen=True)
class _SummarizedDistributedSend:
    dest_rank: int
    src_rank: int
    comm_tag: CommTagType

    shape: ShapeType
    dtype: np.dtype[Any]


@attrs.define(frozen=True)
class _DistributedPartId:
    rank: int
    part_id: PartId


@attrs.define(frozen=True)
class _DistributedName:
    rank: int
    name: str


@attrs.define(frozen=True)
class _SummarizedDistributedGraphPart:
    pid: _DistributedPartId
    needed_pids: FrozenSet[_DistributedPartId]
    user_input_names: FrozenSet[_DistributedName]
    partition_input_names: FrozenSet[_DistributedName]
    output_names: FrozenSet[_DistributedName]
    input_name_to_recv_node: Dict[_DistributedName, DistributedRecv]
    output_name_to_send_node: Dict[_DistributedName, _SummarizedDistributedSend]

    @property
    def rank(self) -> int:
        return self.pid.rank


@attrs.define(frozen=True)
class _CommIdentifier:
    source_rank: int
    dest_rank: int
    comm_tag: Hashable


def verify_distributed_partition(mpi_communicator: mpi4py.MPI.Comm,
        partition: DistributedGraphPartition) -> None:
    """
    .. warning::

        This is an MPI-collective operation.
    """
    my_rank = mpi_communicator.rank
    root_rank = 0

    # Convert local partition to _SummarizedDistributedGraphPart
    summarized_parts: \
            Dict[_DistributedPartId, _SummarizedDistributedGraphPart] = {}

    for pid, part in partition.parts.items():
        assert pid == part.pid

        sends = {}

        for name, send in part.output_name_to_send_node.items():
            n = _DistributedName(my_rank, name)
            assert n not in sends
            sends[n] = _SummarizedDistributedSend(my_rank, send.dest_rank,
                            send.comm_tag, send.data.shape, send.data.dtype)

        dpid = _DistributedPartId(my_rank, part.pid)
        summarized_parts[dpid] = _SummarizedDistributedGraphPart(
            pid=dpid,
            needed_pids=frozenset([_DistributedPartId(my_rank, pid)
                            for pid in part.needed_pids]),
            user_input_names=frozenset([_DistributedName(my_rank, name)
                            for name in part.user_input_names]),
            partition_input_names=frozenset([_DistributedName(my_rank, name)
                            for name in part.partition_input_names]),
            output_names=frozenset([_DistributedName(my_rank, name)
                            for name in part.output_names]),
            input_name_to_recv_node={_DistributedName(my_rank, name): recv
                for name, recv in part.input_name_to_recv_node.items()},
            output_name_to_send_node=sends)

    # Gather the _SummarizedDistributedGraphPart's to rank 0
    all_summarized_parts_gathered: Optional[
            Sequence[Dict[_DistributedPartId, _SummarizedDistributedGraphPart]]] = \
            mpi_communicator.gather(summarized_parts, root=root_rank)

    if mpi_communicator.rank == root_rank:
        assert all_summarized_parts_gathered

        all_summarized_parts = {
                pid: sumpart
                for rank_parts in all_summarized_parts_gathered
                for pid, sumpart in rank_parts.items()}

        # Every node in the graph is a _SummarizedDistributedGraphPart
        pid_to_needed_pids: Dict[_DistributedPartId, Set[_DistributedPartId]] = {}

        def add_needed_pid(pid: _DistributedPartId,
                           needed_pid: _DistributedPartId) -> None:
            pid_to_needed_pids.setdefault(pid, set()).add(needed_pid)

        all_recvs: Set[_CommIdentifier] = set()
        all_sends: Set[_CommIdentifier] = set()

        print(all_summarized_parts)
        output_to_defining_pid: Dict[_DistributedName, _DistributedPartId] = {}
        for sumpart in all_summarized_parts.values():
            for out_name in sumpart.output_names:
                assert out_name not in output_to_defining_pid
                output_to_defining_pid[out_name] = sumpart.pid

        comm_id_to_sending_pid: Dict[_CommIdentifier, _DistributedPartId] = {}
        for sumpart in all_summarized_parts.values():
            for sumsend in sumpart.output_name_to_send_node.values():
                comm_id = _CommIdentifier(
                        source_rank=sumsend.src_rank,
                        dest_rank=sumsend.dest_rank,
                        comm_tag=sumsend.comm_tag)

                if comm_id in comm_id_to_sending_pid:
                    raise ValueError(f"duplicate send for comm id: '{comm_id}'")
                comm_id_to_sending_pid[comm_id] = sumpart.pid

        for sumpart in all_summarized_parts.values():
            pid_to_needed_pids[sumpart.pid] = set(sumpart.needed_pids)

            # Loop through all receives, assert that combination of
            # (src_rank, dest_rank, tag) is unique.
            for dname, dist_recv in sumpart.input_name_to_recv_node.items():
                comm_id = _CommIdentifier(
                        source_rank=dist_recv.src_rank,
                        dest_rank=dname.rank,
                        comm_tag=dist_recv.comm_tag)
                assert comm_id not in all_recvs, \
                    f"Duplicate recv: '{comm_id}' --- {all_recvs=}"
                all_recvs.add(comm_id)

                # Add edges between sends and receives (cross-rank)
                try:
                    sending_pid = comm_id_to_sending_pid[comm_id]
                except KeyError:
                    raise ValueError(f"no matching send for recv on '{comm_id}'")

                add_needed_pid(sumpart.pid, sending_pid)

            # Loop through all sends, assert that combination of
            # (src_rank, dest_rank, tag) is unique.
            for dist_send in sumpart.output_name_to_send_node.values():
                comm_id = _CommIdentifier(
                        source_rank=dist_send.src_rank,
                        dest_rank=dist_send.dest_rank,
                        comm_tag=dist_send.comm_tag)
                assert comm_id not in all_sends, \
                    f"Duplicate send: {comm_id=} --- {all_sends=}"
                all_sends.add(comm_id)

            for dname, dp in output_to_defining_pid.items():
                print(dname, dp)
            # Add edges between output_names and partition_input_names (intra-rank)
            for input_name in sumpart.partition_input_names:
                defining_pid = output_to_defining_pid[input_name]
                assert defining_pid.rank == sumpart.pid.rank
                add_needed_pid(sumpart.pid, defining_pid)

        # Loop through all sends again, making sure there's exactly one recv.
        for s in all_sends:
            assert s in all_recvs, f"Missing recv: {s=} --- {all_recvs=}"

        # Try a topological sort
        from pytools.graph import compute_topological_order
        compute_topological_order(pid_to_needed_pids)

# }}}


def find_distributed_partition(outputs: DictOfNamedArrays
                               ) -> DistributedGraphPartition:
    """
    Partitions *outputs* into parts. Between two parts communication
    statements (sends/receives) are scheduled.

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

    # {{{ get partitioning helper data corresponding to the DAG

    partition_outputs = _MandatoryPartitionOutputsCollector()(outputs)

    # materialized_arrays: "extra" arrays that must be stored in a buffer
    materialized_arrays_collector = _MaterializedArrayCollector()
    materialized_arrays_collector(outputs)
    materialized_arrays = frozenset(
        materialized_arrays_collector.materialized_arrays) - partition_outputs

    # }}}

    dep_mapper = SubsetDependencyMapper(partition_outputs)

    # {{{ compute a dependency graph between outputs, schedule and partition

    output_to_deps = Map({partition_out: (dep_mapper(partition_out)
                                          - frozenset([partition_out]))
                          for partition_out in partition_outputs})

    output_to_part_id = _linearly_schedule_batches(output_to_deps)

    # }}}

    # {{{ assign each materialized array a partition ID in which it will be placed

    materialized_array_to_output_deps = Map({ary: (dep_mapper(ary)
                                                   - frozenset([ary]))
                                             for ary in materialized_arrays})
    materialized_ary_to_part_id = _assign_materialized_arrays_to_part_id(
        materialized_arrays,
        materialized_array_to_output_deps,
        output_to_part_id)

    assert frozenset(materialized_ary_to_part_id) == materialized_arrays

    # }}}

    stored_ary_to_part_id = materialized_ary_to_part_id.update(output_to_part_id)

    # {{{ find which materialized arrays have users in multiple parts
    # (and promote them to part outputs)

    ary_to_dominant_materialized_deps = (
        _get_array_to_dominant_materialized_deps(outputs,
                                                      (materialized_arrays
                                                       | partition_outputs)))

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
        partition_outputs | materialized_arrays_realized_as_partition_outputs)

    partitioned_outputs = make_dict_of_named_arrays({
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
                find_partition(partitioned_outputs,
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

    partition = _map_distributed_graph_partion_nodes(map_array, map_send, gp)

    return partition

# }}}


# {{{ construct tag numbering

def number_distributed_tags(
        mpi_communicator: mpi4py.MPI.Comm,
        partition: DistributedGraphPartition,
        base_tag: int) -> Tuple[DistributedGraphPartition, int]:
    """Return a new :class:`~pytato.distributed.DistributedGraphPartition`
    in which symbolic tags are replaced by unique integer tags, created by
    counting upward from *base_tag*.

    :returns: a tuple ``(partition, next_tag)``, where *partition* is the new
        :class:`~pytato.distributed.DistributedGraphPartition` with numerical
        tags, and *next_tag* is the lowest integer tag above *base_tag* that
        was not used.

    .. note::

        This is a potentially heavyweight MPI-collective operation on
        *mpi_communicator*.
    """
    tags = frozenset({
            recv.comm_tag
            for part in partition.parts.values()
            for recv in part.input_name_to_recv_node.values()
            } | {
            send.comm_tag
            for part in partition.parts.values()
            for send in part.output_name_to_send_node.values()})

    from mpi4py import MPI

    def set_union(
            set_a: FrozenSet[Any], set_b: FrozenSet[Any],
            mpi_data_type: MPI.Datatype) -> FrozenSet[str]:
        assert mpi_data_type is None
        assert isinstance(set_a, frozenset)
        assert isinstance(set_b, frozenset)

        return set_a | set_b

    root_rank = 0

    set_union_mpi_op = MPI.Op.Create(
            # type ignore reason: mpi4py misdeclares op functions as returning
            # None.
            set_union,  # type: ignore[arg-type]
            commute=True)
    try:
        all_tags = mpi_communicator.reduce(
                tags, set_union_mpi_op, root=root_rank)
    finally:
        set_union_mpi_op.Free()

    if mpi_communicator.rank == root_rank:
        sym_tag_to_int_tag = {}
        next_tag = base_tag
        assert isinstance(all_tags, frozenset)

        for sym_tag in all_tags:
            sym_tag_to_int_tag[sym_tag] = next_tag
            next_tag += 1

        if __debug__:
            print(f"{sym_tag_to_int_tag=}")

        mpi_communicator.bcast((sym_tag_to_int_tag, next_tag), root=root_rank)
    else:
        sym_tag_to_int_tag, next_tag = mpi_communicator.bcast(None, root=root_rank)

    from attrs import evolve as replace
    p = DistributedGraphPartition(
            parts={
                pid: replace(part,
                    input_name_to_recv_node={
                        name: recv.copy(comm_tag=sym_tag_to_int_tag[recv.comm_tag])
                        for name, recv in part.input_name_to_recv_node.items()},
                    output_name_to_send_node={
                        name: send.copy(comm_tag=sym_tag_to_int_tag[send.comm_tag])
                        for name, send in part.output_name_to_send_node.items()},
                    )
                for pid, part in partition.parts.items()
                },
            var_name_to_result=partition.var_name_to_result,
            toposorted_part_ids=partition.toposorted_part_ids), next_tag

    verify_distributed_partition(mpi_communicator, p[0])
    return p

# }}}


# {{{ distributed execute

def _post_receive(mpi_communicator: mpi4py.MPI.Comm,
                 recv: DistributedRecv) -> Tuple[Any, np.ndarray[Any, Any]]:
    if not all(isinstance(dim, INT_CLASSES) for dim in recv.shape):
        raise NotImplementedError("Parametric shapes not supported yet.")

    assert isinstance(recv.comm_tag, int)
    # mypy is right here, size params in 'recv.shape' must be evaluated
    buf = np.empty(recv.shape, dtype=recv.dtype)  # type: ignore[arg-type]

    return mpi_communicator.Irecv(
            buf=buf, source=recv.src_rank, tag=recv.comm_tag), buf


def _mpi_send(mpi_communicator: Any, send_node: DistributedSend,
             data: np.ndarray[Any, Any]) -> Any:
    # Must use-non-blocking send, as blocking send may wait for a corresponding
    # receive to be posted (but if sending to self, this may only occur later).
    return mpi_communicator.Isend(
            data, dest=send_node.dest_rank, tag=send_node.comm_tag)


def execute_distributed_partition(
        partition: DistributedGraphPartition, prg_per_partition:
        Dict[Hashable, BoundProgram],
        queue: Any, mpi_communicator: Any,
        *,
        allocator: Optional[Any] = None,
        input_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    if input_args is None:
        input_args = {}

    from mpi4py import MPI

    if len(partition.parts) != 1:
        recv_names_tup, recv_requests_tup, recv_buffers_tup = zip(*[
            (name,) + _post_receive(mpi_communicator, recv)
            for part in partition.parts.values()
            for name, recv in part.input_name_to_recv_node.items()])
        recv_names = list(recv_names_tup)
        recv_requests = list(recv_requests_tup)
        recv_buffers = list(recv_buffers_tup)
        del recv_names_tup
        del recv_requests_tup
        del recv_buffers_tup
    else:
        # Only a single partition, no recv requests exist
        recv_names = []
        recv_requests = []
        recv_buffers = []

    context: Dict[str, Any] = input_args.copy()

    pids_to_execute = set(partition.parts)
    pids_executed = set()
    recv_names_completed = set()
    send_requests = []

    # {{{ Input name refcount

    # Keep a count on how often each input name is used
    # in order to be able to free them.

    from pytools import memoize_on_first_arg

    @memoize_on_first_arg
    def _get_partition_input_name_refcount(partition: DistributedGraphPartition) \
            -> Dict[str, int]:
        partition_input_names_refcount: Dict[str, int] = {}
        for pid in set(partition.parts):
            for name in partition.parts[pid].all_input_names():
                if name in partition_input_names_refcount:
                    partition_input_names_refcount[name] += 1
                else:
                    partition_input_names_refcount[name] = 1

        return partition_input_names_refcount

    partition_input_names_refcount = \
        _get_partition_input_name_refcount(partition).copy()

    # }}}

    def exec_ready_part(part: DistributedGraphPart) -> None:
        inputs = {k: context[k] for k in part.all_input_names()}

        _evt, result_dict = prg_per_partition[part.pid](queue,
                                                        allocator=allocator,
                                                        **inputs)

        context.update(result_dict)

        for name, send_node in part.output_name_to_send_node.items():
            # FIXME: pytato shouldn't depend on pyopencl
            if isinstance(context[name], np.ndarray):
                data = context[name]
            else:
                data = context[name].get(queue)
            send_requests.append(_mpi_send(mpi_communicator, send_node, data))

        pids_executed.add(part.pid)
        pids_to_execute.remove(part.pid)

    def wait_for_some_recvs() -> None:
        complete_recv_indices = MPI.Request.Waitsome(recv_requests)

        # Waitsome is allowed to return None
        if not complete_recv_indices:
            complete_recv_indices = []

        # reverse to preserve indices
        for idx in sorted(complete_recv_indices, reverse=True):
            name = recv_names.pop(idx)
            recv_requests.pop(idx)
            buf = recv_buffers.pop(idx)

            # FIXME: pytato shouldn't depend on pyopencl
            import pyopencl as cl
            context[name] = cl.array.to_device(queue, buf, allocator=allocator)
            recv_names_completed.add(name)

    # {{{ main loop

    while pids_to_execute:
        ready_pids = {pid
                for pid in pids_to_execute
                # FIXME: Only O(n**2) altogether. Nobody is going to notice, right?
                if partition.parts[pid].needed_pids <= pids_executed
                and (set(partition.parts[pid].input_name_to_recv_node)
                    <= recv_names_completed)}
        for pid in ready_pids:
            part = partition.parts[pid]
            exec_ready_part(part)

            for p in part.all_input_names():
                partition_input_names_refcount[p] -= 1
                if partition_input_names_refcount[p] == 0:
                    del context[p]

        if not ready_pids:
            wait_for_some_recvs()

    # }}}

    for send_req in send_requests:
        send_req.Wait()

    if __debug__:
        for name, count in partition_input_names_refcount.items():
            assert count == 0
            assert name not in context

    return context

# }}}

# vim: foldmethod=marker
