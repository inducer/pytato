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

from typing import (Any, Dict, Hashable, Tuple, Optional, Set, # noqa (need List, FrozenSet, Mapping for sphinx)
    List, FrozenSet, Mapping)

from dataclasses import dataclass

from pytools import UniqueNameGenerator
from pytools.tag import Taggable, TagsType
from pytato.array import (Array, _SuppliedShapeAndDtypeMixin,
                          DictOfNamedArrays, ShapeType,
                          Placeholder, make_placeholder)
from pytato.transform import ArrayOrNames, CopyMapper
from pytato.partition import GraphPart, GraphPartition, PartId
from pytato.target import BoundProgram

import numpy as np

__doc__ = """
Distributed communication
-------------------------

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
.. autofunction:: execute_distributed_partition
"""


# {{{ Distributed node types

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

        .. note:: Currently, this attribute must be of class `int`.
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
        data: Optional[Array] = kwargs["data"]
        dest_rank: Optional[int] = kwargs["dest_rank"]
        comm_tag: Optional[CommTagType] = kwargs["comm_tag"]
        tags: Optional[TagsType] = kwargs["tags"]
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
    no useful result of a send (at least of of an :class:`~pytato.Array` type).

    This is where this node type comes in. Its value is the same as that of
    :attr:`passthrough_data`, *and* it holds a reference to the send node.

    .. note::

        This all seems a wee bit inelegant, but nobody who has written
        or reviewed this code so far had a better idea. If you do, please speak up!

    .. attribute:: send

        The :class:`DistributedSend` to which a reference is to be held.

    .. attribute:: passthrough_data

        A :class:`~pytato.Array`. The value of this node.
    """

    _mapper_method = "map_distributed_send_ref_holder"
    _fields = Array._fields + ("passthrough_data", "send",)

    def __init__(self, send: DistributedSend, passthrough_data: Array,
                 tags: TagsType = frozenset()) -> None:
        super().__init__(tags=tags)
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
        Only a :class:`DistributedRecv` with the same tag will be able to
        receive the data being sent here.

    .. attribute:: shape
    .. attribute:: dtype
    """

    _fields = Array._fields + ("src_rank", "comm_tag")
    _mapper_method = "map_distributed_recv"

    def __init__(self, src_rank: int, comm_tag: CommTagType,
                 shape: ShapeType, dtype: Any,
                 tags: Optional[TagsType] = frozenset()) -> None:
        super().__init__(shape=shape, dtype=dtype, tags=tags)
        self.src_rank = src_rank
        self.comm_tag = comm_tag


def make_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          send_tags: TagsType = frozenset()) -> \
         DistributedSend:
    return DistributedSend(sent_data, dest_rank, comm_tag, send_tags)


def staple_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          stapled_to: Array, *,
                          send_tags: TagsType = frozenset(),
                          ref_holder_tags: TagsType = frozenset()) -> \
         DistributedSendRefHolder:
    return DistributedSendRefHolder(
            DistributedSend(sent_data, dest_rank, comm_tag, send_tags),
            stapled_to, tags=ref_holder_tags)


def make_distributed_recv(src_rank: int, comm_tag: CommTagType,
                          shape: ShapeType, dtype: Any,
                          tags: TagsType = frozenset()) \
                          -> DistributedRecv:
    dtype = np.dtype(dtype)
    return DistributedRecv(src_rank, comm_tag, shape, dtype, tags)

# }}}


# {{{ find distributed partition

@dataclass(frozen=True, eq=True)
class DistributedPartitionId():
    fed_sends: object
    feeding_recvs: object


def find_distributed_partition(
        outputs: DictOfNamedArrays) -> DistributedGraphPartition:
    """Finds a partitioning in a distributed context."""

    from pytato.transform import (UsersCollector, TopoSortMapper,
                                  reverse_graph, tag_user_nodes)

    # FIXME: We should probably iterate over the DictOfNamedArrays instead
    res = outputs[next(iter(outputs))]

    gdm = UsersCollector()
    gdm(res)

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
        tm(res)

        for node in tm.topological_order:
            get_part_id(node)

    # }}}

    from pytato.partition import find_partition
    return _gather_distributed_comm_info(
            find_partition(outputs, get_part_id))

# }}}


# {{{ distributed info collection

@dataclass(frozen=True)
class DistributedGraphPart(GraphPart):
    """For one graph partition, record send/receive information for input/
    output names.

    .. attribute:: input_name_to_recv_node
    .. attribute:: output_name_to_send_node
    """
    # TODO Document these

    input_name_to_recv_node: Dict[str, DistributedRecv]
    output_name_to_send_node: Dict[str, DistributedSend]


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


def _gather_distributed_comm_info(partition: GraphPartition) -> \
        DistributedGraphPartition:
    var_name_to_result = {}
    parts: Dict[PartId, DistributedGraphPart] = {}

    dist_name_generator = UniqueNameGenerator(forced_prefix="_pt_dist_")

    for part in partition.parts.values():
        comm_replacer = _DistributedCommReplacer(dist_name_generator)
        part_results = {
                var_name: comm_replacer(partition.var_name_to_result[var_name])
                for var_name in part.output_names}

        dist_sends = [
                comm_replacer.map_distributed_send(send)
                for send in part.distributed_sends]

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

    return DistributedGraphPartition(
            parts=parts,
            var_name_to_result=var_name_to_result,
            toposorted_part_ids=partition.toposorted_part_ids)

# }}}


# {{{ distributed execute

def _post_receive(comm: Any,
                 recv: DistributedRecv) -> Tuple[Any, np.ndarray[Any, Any]]:
    # FIXME: recv.shape might be parametric, evaluate
    buf = np.empty(recv.shape, dtype=recv.dtype)

    # FIXME: Why doesn't this work with the lower case mpi4py function names?
    return comm.Irecv(buf=buf, source=recv.src_rank, tag=recv.comm_tag), buf


def _mpi_send(comm: Any, send_node: DistributedSend,
             data: np.ndarray[Any, Any]) -> Any:
    # Must use-non-blocking send, as blocking send may wait for a corresponding
    # receive to be posted (but if sending to self, this may only occur later).
    return comm.Isend(data, dest=send_node.dest_rank, tag=send_node.comm_tag)


def execute_distributed_partition(
        partition: DistributedGraphPartition, prg_per_partition:
        Dict[Hashable, BoundProgram],
        queue: Any, comm: Any,
        input_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    if input_args is None:
        input_args = {}

    from mpi4py import MPI

    recv_names_tup, recv_requests_tup, recv_buffers_tup = zip(*[
            (name,) + _post_receive(comm, recv)
            for part in partition.parts.values()
            for name, recv in part.input_name_to_recv_node.items()])
    recv_names = list(recv_names_tup)
    recv_requests = list(recv_requests_tup)
    recv_buffers = list(recv_buffers_tup)

    context: Dict[str, Any] = input_args.copy()

    pids_to_execute = set(partition.parts)
    pids_executed = set()
    recv_names_completed = set()
    send_requests = []

    def exec_ready_part(part: DistributedGraphPart) -> None:
        inputs = {k: context[k] for k in part.all_input_names()}

        _evt, result_dict = prg_per_partition[part.pid](queue, **inputs)

        context.update(result_dict)

        for name, send_node in part.output_name_to_send_node.items():
            data = context[name].get(queue)
            send_requests.append(_mpi_send(comm, send_node, data))

        pids_executed.add(part.pid)
        pids_to_execute.remove(part.pid)

    def wait_for_some_recvs() -> None:
        complete_recv_indices = MPI.Request.Waitsome(recv_requests)

        # Waitsome is allowed to return None
        if not complete_recv_indices:
            complete_recv_indices = []

        for idx in sorted(complete_recv_indices, reverse=True):
            name = recv_names.pop(idx)
            recv_requests.pop(idx)
            buf = recv_buffers.pop(idx)

            import pyopencl as cl
            context[name] = cl.array.empty(queue, buf.shape, buf.dtype).set(buf)
            recv_names_completed.add(name)

    # FIXME: This keeps all variables alive that are used to get data into
    # and out of partitions. Probably not what we want long-term.

    # {{{ main loop

    while pids_to_execute:
        ready_pids = {pid
                for pid in pids_to_execute
                # FIXME: Only O(n**2) altogether. Nobody is going to notice, right?
                if partition.parts[pid].needed_pids <= pids_executed
                and (set(partition.parts[pid].input_name_to_recv_node)
                    <= recv_names_completed)}
        for pid in ready_pids:
            exec_ready_part(partition.parts[pid])

        if not ready_pids:
            wait_for_some_recvs()

    # }}}

    for send_req in send_requests:
        send_req.Wait()

    return context

# }}}

# vim: foldmethod=marker
