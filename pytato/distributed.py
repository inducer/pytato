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

from typing import Any, Dict, Hashable, Tuple, Optional, cast, List, Set  # noqa (need List, Set for sphinx)
from dataclasses import dataclass

from pytools.tag import Taggable
from pytato.array import (Array, _SuppliedShapeAndDtypeMixin, ShapeType,
                          Placeholder, make_placeholder)
from pytato.transform import CopyMapper
from pytato.partition import GraphPartitions, PartitionId
from pytato.target import BoundProgram

from pytools.tag import TagsType

import numpy as np

__doc__ = """
Distributed communication
-------------------------

.. autoclass:: DistributedSend
.. autoclass:: DistributedSendRefHolder
.. autoclass:: DistributedRecv
.. autoclass:: PerPartitionSendRecvInfo
.. autoclass:: DistributedGraphPartitions

.. autofunction:: make_distributed_send
.. autofunction:: staple_distributed_send
.. autofunction:: make_distributed_recv

.. autofunction:: gather_distributed_comm_info
.. autofunction:: execute_partitions_distributed
"""


# {{{ Distributed execution

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


# {{{ distributed info collection

@dataclass
class PerPartitionSendRecvInfo:
    """For one graph partition, record send/receive information for input/
    output names.

    .. attribute:: part_input_name_to_recv_node
    .. attribute:: part_output_name_to_send_node
    """
    # TODO Document these

    part_input_name_to_recv_node: Dict[str, DistributedRecv]
    part_output_name_to_send_node: Dict[str, DistributedSend]


@dataclass
class DistributedGraphPartitions(GraphPartitions):
    """Store information about distributed graph partitions.

    .. attribute:: partition_id_to_send_recv_info
    """
    partition_id_to_send_recv_info: Dict[PartitionId, PerPartitionSendRecvInfo]


class _DistributedCommReplacer(CopyMapper):
    """Mapper to process a DAG for realization of :class:`DistributedSend`
    and :class:`DistributedRecv` outside of normal code generation.

    -   Replaces :class:`DistributedRecv` with :class`~pytato.Placeholder`
        so that received data can be externally supplied, making a note
        in :attr:`part_input_name_to_recv_node`.

    -   Eliminates :class:`DistributedSendRefHolder` and
        :class:`DistributedSend` from the DAG, making a note of data
        to be send in :attr:`part_output_name_to_send_node`.
    """

    def __init__(self) -> None:
        super().__init__()

        from pytools import UniqueNameGenerator
        self.name_generator = UniqueNameGenerator(forced_prefix="_dist_ph_")

        self.part_input_name_to_recv_node: Dict[str, DistributedRecv] = {}
        self.part_output_name_to_send_node: Dict[str, DistributedSend] = {}

    def map_distributed_recv(self, expr: DistributedRecv) -> Placeholder:
        # no children, no need to recurse

        new_name = self.name_generator()
        self.part_input_name_to_recv_node[new_name] = expr
        return make_placeholder(new_name, self.rec_idx_or_size_tuple(expr.shape),
                expr.dtype, tags=expr.tags)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> Array:
        result = cast(DistributedSendRefHolder,
                super().map_distributed_send_ref_holder(expr))

        new_name = self.name_generator()
        self.part_output_name_to_send_node[new_name] = result.send

        return result.passthrough_data


def gather_distributed_comm_info(parts: GraphPartitions) -> \
        DistributedGraphPartitions:
    partition_id_to_input_names = {}
    partition_id_to_output_names = {}
    var_name_to_result = {}
    partition_id_to_send_recv_info = {}

    for pid in parts.toposorted_partitions:
        comm_replacer = _DistributedCommReplacer()
        part_results = {var_name: comm_replacer(parts.var_name_to_result[var_name])
                for var_name in parts.partition_id_to_output_names[pid]}

        part_results.update({
            name: send_node.data
            for name, send_node in
            comm_replacer.part_output_name_to_send_node.items()})

        partition_id_to_input_names[pid] = (
                parts.partition_id_to_input_names[pid]
                | set(comm_replacer.part_input_name_to_recv_node))

        partition_id_to_output_names[pid] = (
                parts.partition_id_to_output_names[pid]
                | set(comm_replacer.part_output_name_to_send_node))

        partition_id_to_send_recv_info[pid] = PerPartitionSendRecvInfo(
                    part_input_name_to_recv_node=(
                        comm_replacer.part_input_name_to_recv_node),
                    part_output_name_to_send_node=(
                        comm_replacer.part_output_name_to_send_node),
                    )

        for name, val in part_results.items():
            assert name not in var_name_to_result
            var_name_to_result[name] = val

    return DistributedGraphPartitions(
            toposorted_partitions=parts.toposorted_partitions,
            partition_id_to_input_names=partition_id_to_input_names,
            partition_id_to_output_names=partition_id_to_output_names,
            var_name_to_result=var_name_to_result,
            partition_id_to_send_recv_info=partition_id_to_send_recv_info,
            )


# }}}


# {{{ distributed execute

def post_receive(comm: Any,
                 recv: DistributedRecv) -> Tuple[Any, np.ndarray[Any, Any]]:
    # FIXME: recv.shape might be parametric, evaluate
    buf = np.empty(recv.shape, dtype=recv.dtype)

    # FIXME: Why doesn't this work with the lower case mpi4py function names?
    return comm.Irecv(buf=buf, source=recv.src_rank, tag=recv.comm_tag), buf


def mpi_send(comm: Any, send_node: DistributedSend,
             data: np.ndarray[Any, Any]) -> None:
    # FIXME: Might be more efficient to use non-blocking send
    comm.Send(data.data, dest=send_node.dest_rank, tag=send_node.comm_tag)


def execute_partitions_distributed(
        parts: DistributedGraphPartitions, prg_per_partition:
        Dict[Hashable, BoundProgram],
        queue: Any, comm: Any) -> Dict[str, Any]:

    recv_to_req_and_buf = {
            recv: post_receive(comm, recv)
            for pid in parts.toposorted_partitions
            for recv in (parts.partition_id_to_send_recv_info[pid]
                .part_input_name_to_recv_node.values())}

    context: Dict[str, Any] = {}
    for pid in parts.toposorted_partitions:
        send_recv_info = parts.partition_id_to_send_recv_info[pid]
        # FIXME: necessary?
        # context.update(part_receives[1].results)

        inputs = {}
        for name, recv in send_recv_info.part_input_name_to_recv_node.items():
            req, buf = recv_to_req_and_buf[recv]
            req.Wait()
            inputs.update({name: buf})

        # inputs.update({k: v[1] for k, v in part_receives[0].items()})
        # {part.name: actx.from_numpy(recv.wait())
        # {recv.results: None
        #     for recv in part_receives})

        # inputs.update(context)

        # print(f"{context=}")
        # print(prg_per_partition[pid].program)

        inputs.update({
            k: context[k] for k in parts.partition_id_to_input_names[pid]
            if k in context})

        _evt, result_dict = prg_per_partition[pid](queue, **inputs)

        context.update(result_dict)

        for name, send_node in send_recv_info.part_output_name_to_send_node.items():
            mpi_send(comm, send_node, context[name])

            # FIXME: Legal?
            #del context[name]

    return context

# }}}

# vim: foldmethod=marker
