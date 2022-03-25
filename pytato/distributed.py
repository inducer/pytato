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
    List, FrozenSet, Callable, cast, Mapping)  # Mapping required by sphinx

from dataclasses import dataclass

from pytools import UniqueNameGenerator
from pytools.tag import Taggable, TagsType
from pytato.array import (Array, _SuppliedShapeAndDtypeMixin,
                          DictOfNamedArrays, ShapeType,
                          Placeholder, make_placeholder,
                          _get_default_axes, AxesT)
from pytato.transform import ArrayOrNames, CopyMapper
from pytato.partition import GraphPart, GraphPartition, PartId, GraphPartitioner
from pytato.target import BoundProgram

import numpy as np

__doc__ = r"""
Distributed-memory evaluation of expression graphs is accomplished
by :ref:`partitioning <partitioning>` the graph to reveal communication-free
pieces of the computation. Communication (i.e. sending/receving data) is then
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

.. currentmodule:: pytato

.. autofunction:: find_distributed_partition
.. autofunction:: number_distributed_tags
.. autofunction:: execute_distributed_partition

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: CommTagType

    A type representing a communication tag.

.. class:: TagsType

    A :class:`frozenset` of :class:`pytools.tag.Tag`\ s.

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
                 tags: TagsType = frozenset()) -> None:
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

    def copy(self, **kwargs: Any) -> DistributedSend:
        data: Optional[Array] = kwargs.get("data")
        dest_rank: Optional[int] = kwargs.get("dest_rank")
        comm_tag: Optional[CommTagType] = kwargs.get("comm_tag")
        tags: Optional[TagsType] = kwargs.get("tags")
        return type(self)(
                data=data or self.data,
                dest_rank=dest_rank if dest_rank is not None else self.dest_rank,
                comm_tag=comm_tag if comm_tag is not None else self.comm_tag,
                tags=tags if tags is not None else self.tags)


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

    _mapper_method = "map_distributed_send_ref_holder"
    _fields = Array._fields + ("passthrough_data", "send",)

    def __init__(self, send: DistributedSend, passthrough_data: Array,
                 tags: TagsType = frozenset()) -> None:
        super().__init__(axes=passthrough_data.axes, tags=tags)
        self.send = send
        self.passthrough_data = passthrough_data

    @property
    def shape(self) -> ShapeType:
        return self.passthrough_data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.passthrough_data.dtype


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

    _fields = Array._fields + ("shape", "dtype", "src_rank", "comm_tag")
    _mapper_method = "map_distributed_recv"

    def __init__(self, src_rank: int, comm_tag: CommTagType,
                 shape: ShapeType, dtype: Any,
                 tags: Optional[TagsType] = frozenset(),
                 axes: Optional[AxesT] = None) -> None:

        if not axes:
            axes = _get_default_axes(len(shape))
        super().__init__(shape=shape, dtype=dtype, tags=tags,
                         axes=axes)
        self.src_rank = src_rank
        self.comm_tag = comm_tag


def make_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          send_tags: TagsType = frozenset()) -> \
         DistributedSend:
    """Make a :class:`DistributedSend` object."""
    return DistributedSend(sent_data, dest_rank, comm_tag, send_tags)


def staple_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          stapled_to: Array, *,
                          send_tags: TagsType = frozenset(),
                          ref_holder_tags: TagsType = frozenset()) -> \
         DistributedSendRefHolder:
    """Make a :class:`DistributedSend` object wrapped in a
    :class:`DistributedSendRefHolder` object."""
    return DistributedSendRefHolder(
            DistributedSend(sent_data, dest_rank, comm_tag, send_tags),
            stapled_to, tags=ref_holder_tags)


def make_distributed_recv(src_rank: int, comm_tag: CommTagType,
                          shape: ShapeType, dtype: Any,
                          tags: TagsType = frozenset()) \
                          -> DistributedRecv:
    """Make a :class:`DistributedRecv` object."""
    dtype = np.dtype(dtype)
    return DistributedRecv(src_rank, comm_tag, shape, dtype, tags)

# }}}


# {{{ distributed info collection

@dataclass(frozen=True)
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


@dataclass(frozen=True)
class DistributedGraphPartition(GraphPartition):
    """Store information about distributed graph partitions. This
    has the same attributes as :class:`~pytato.partition.GraphPartition`,
    however :attr:`~pytato.partition.GraphPartition.parts` now maps to
    instances of :class:`DistributedGraphPart`.
    """
    parts: Dict[PartId, DistributedGraphPart]


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
                expr.dtype, tags=expr.tags)

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


# {{{ find distributed partition

@dataclass(frozen=True, eq=True)
class DistributedPartitionId():
    fed_sends: object
    feeding_recvs: object


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


def find_distributed_partition(
        outputs: DictOfNamedArrays) -> DistributedGraphPartition:
    """Finds a partitioning in a distributed context."""

    from pytato.transform import (UsersCollector, TopoSortMapper,
                                  reverse_graph, tag_user_nodes)

    gdm = UsersCollector()
    gdm(outputs)

    graph = gdm.node_to_users

    # type-ignore-reason:
    # 'graph' also includes DistributedSend nodes, which are not Arrays
    rev_graph = reverse_graph(graph)  # type: ignore[arg-type]

    # FIXME: Inefficient... too many traversals
    node_to_feeding_recvs: Dict[ArrayOrNames, Set[ArrayOrNames]] = {}
    for node in graph:
        node_to_feeding_recvs.setdefault(node, set())
        if isinstance(node, DistributedRecv):
            tag_user_nodes(graph, tag=node,  # type: ignore[arg-type]
                            starting_point=node,
                            node_to_tags=node_to_feeding_recvs)

    node_to_fed_sends: Dict[ArrayOrNames, Set[ArrayOrNames]] = {}
    for node in rev_graph:
        node_to_fed_sends.setdefault(node, set())
        if isinstance(node, DistributedSend):
            tag_user_nodes(rev_graph, tag=node, starting_point=node,
                            node_to_tags=node_to_fed_sends)

    def get_part_id(expr: ArrayOrNames) -> DistributedPartitionId:
        return DistributedPartitionId(frozenset(node_to_fed_sends[expr]),
                                      frozenset(node_to_feeding_recvs[expr]))

    # {{{ Sanity checks

    if __debug__:
        for node, _ in node_to_feeding_recvs.items():
            for n in node_to_feeding_recvs[node]:
                assert(isinstance(n, DistributedRecv))

        for node, _ in node_to_fed_sends.items():
            for n in node_to_fed_sends[node]:
                assert(isinstance(n, DistributedSend))

        tm = TopoSortMapper()
        tm(outputs)

        for node in tm.topological_order:
            get_part_id(node)

    # }}}

    from pytato.partition import find_partition
    return cast(DistributedGraphPartition,
            find_partition(outputs, get_part_id, _DistributedGraphPartitioner))

# }}}


# {{{ construct tag numbering

def number_distributed_tags(
        mpi_communicator: Any,
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
            for name, recv in part.input_name_to_recv_node.items()
            } | {
            send.comm_tag
            for part in partition.parts.values()
            for name, send in part.output_name_to_send_node.items()})

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
        for sym_tag in all_tags:
            sym_tag_to_int_tag[sym_tag] = next_tag
            next_tag += 1

        mpi_communicator.bcast((sym_tag_to_int_tag, next_tag), root=root_rank)
    else:
        sym_tag_to_int_tag, next_tag = mpi_communicator.bcast(None, root=root_rank)

    from dataclasses import replace
    return DistributedGraphPartition(
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

# }}}


# {{{ distributed execute

def _post_receive(mpi_communicator: Any,
                 recv: DistributedRecv) -> Tuple[Any, np.ndarray[Any, Any]]:
    # FIXME: recv.shape might be parametric, evaluate
    buf = np.empty(recv.shape, dtype=recv.dtype)

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
