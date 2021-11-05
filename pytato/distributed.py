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

from typing import Any, Dict, Hashable, Tuple, Optional
from dataclasses import dataclass

from pytato.array import (Array, _SuppliedShapeAndDtypeMixin, ShapeType,
                          Placeholder, make_placeholder)
from pytato.transform import CopyMapper
from pytato.partition import CodePartitions
from pytato.target import BoundProgram

from pytools.tag import TagsType

import numpy as np

__doc__ = """
.. currentmodule:: pytato.distributed

Distributed communication
-------------------------

.. class:: DistributedSend
.. class:: DistributedRecv
.. class:: DistributedCommInfo
..autofunction:: gather_distributed_comm_info
..autofunction:: execute_partitions_distributed
"""


# {{{ Distributed execution

class DistributedSend(_SuppliedShapeAndDtypeMixin, Array):
    """Class representing a distributed send operation."""

    _mapper_method = "map_distributed_send"
    _fields = Array._fields + ("data", "dest_rank", "comm_tag")

    def __init__(self, data: Array, dest_rank: int, comm_tag: Any,
                 shape: ShapeType, dtype: Any,
                 tags: Optional[TagsType] = frozenset()) -> None:
        super().__init__(shape=shape, dtype=dtype, tags=tags)
        self.data = data
        self.dest_rank = dest_rank
        self.comm_tag = comm_tag

    @property
    def shape(self) -> ShapeType:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype


class DistributedRecv(_SuppliedShapeAndDtypeMixin, Array):
    """Class representing a distributed receive operation."""

    _fields = Array._fields + ("data", "src_rank", "comm_tag")
    _mapper_method = "map_distributed_recv"

    def __init__(self, data: Array, src_rank: int, comm_tag: Any,
                 shape: ShapeType, dtype: Any,
                 tags: Optional[TagsType] = frozenset()) -> None:
        super().__init__(shape=shape, dtype=dtype, tags=tags)
        self.src_rank = src_rank
        self.comm_tag = comm_tag
        self.data = data


def make_distributed_send(data: Array, dest_rank: int, comm_tag: object,
                          shape: ShapeType, dtype: Any,
                          tags: Optional[TagsType] = frozenset()) -> \
         DistributedSend:
    return DistributedSend(data, dest_rank, comm_tag, shape, dtype, tags)


def make_distributed_recv(data: Array, src_rank: int, comm_tag: object,
                          shape: ShapeType, dtype: Any,
                          tags: Optional[TagsType] = frozenset()) \
                          -> DistributedRecv:
    return DistributedRecv(data, src_rank, comm_tag, shape, dtype, tags)

# }}}


# {{{ distributed info collection

@dataclass
class DistributedCommInfo:
    """For one graph partition, record send/receive information for input/
    output names as well as the computation in :attr:`results`.

    .. attribute:: part_input_name_to_recv_node
    .. attribute:: part_output_name_to_send_node
    .. attribute:: results
    """
    # TODO Document these

    part_input_name_to_recv_node: Dict[str, DistributedRecv]
    part_output_name_to_send_node: Dict[str, DistributedSend]
    results: Dict[str, Array]


class _DistributedCommReplacer(CopyMapper):
    """Support for distributed communication operations."""

    def __init__(self) -> None:
        super().__init__()

        from pytools import UniqueNameGenerator
        self.name_generator = UniqueNameGenerator(forced_prefix="_dist_ph_")

        self.part_input_name_to_recv_node: Dict[str, DistributedRecv] = {}
        self.part_output_name_to_send_node: Dict[str, DistributedSend] = {}

    # type-ignore-reason: incompatible with superclass
    def map_distributed_recv(self,  # type: ignore[override]
            expr: DistributedRecv) -> Placeholder:
        # no children, no need to recurse

        new_name = self.name_generator()
        self.part_input_name_to_recv_node[new_name] = expr
        return make_placeholder(new_name, expr.shape,
                expr.dtype,
                tags=expr.tags)

    # type-ignore-reason: incompatible with superclass
    def map_distributed_send(self,  # type: ignore[override]
                             expr: DistributedSend) -> Array:  #
        result = super().map_distributed_send(expr)

        new_name = self.name_generator()
        self.part_output_name_to_send_node[new_name] = result

        return expr.data


def gather_distributed_comm_info(parts: CodePartitions) -> \
        Dict[Hashable, DistributedCommInfo]:
    result = {}

    for pid in parts.toposorted_partitions:
        comm_replacer = _DistributedCommReplacer()
        part_results = {var_name: comm_replacer(parts.var_name_to_result[var_name])
                for var_name in parts.partition_id_to_output_names[pid]}

        part_results.update({
            name: send_node.data
            for name, send_node in
            comm_replacer.part_output_name_to_send_node.items()})

        result[pid] = \
                DistributedCommInfo(
                    part_input_name_to_recv_node=(
                        comm_replacer.part_input_name_to_recv_node),
                    part_output_name_to_send_node=(
                        comm_replacer.part_output_name_to_send_node),
                    results=part_results
                    )

    return result


# }}}


# {{{ distributed execute

# FIXME: Where to get communicator/actx? Argument to make_distributed_recv?
# communicator -> pass into execute_partitions_distributed
def post_receives(dci: DistributedCommInfo) -> \
        Tuple[Dict[str, Tuple[Any, Any]], DistributedCommInfo]:

    from mpi4py import MPI

    recv_reqs = {}

    for k, v in dci.part_input_name_to_recv_node.items():
        src_rank = v.src_rank
        tag = v.comm_tag

        buf = np.zeros(v.shape, dtype=v.dtype)

        recv_reqs[k] = (MPI.COMM_WORLD.Irecv(buf=buf, source=src_rank, tag=tag), buf)

    return (recv_reqs, dci)


# FIXME: Where to get communicator? Argument to make_distributed_send?
# -> pass into execute_partitions_distributed
def mpi_send(rank: int, tag: Any, data: Any) -> None:
    from mpi4py import MPI
    MPI.COMM_WORLD.Send(data.data, dest=rank, tag=tag)


def execute_partitions_distributed(parts: CodePartitions, prg_per_partition:
                        Dict[Hashable, BoundProgram], queue: Any,
                        distributed_comm_infos: Dict[Hashable,
                        DistributedCommInfo]) \
                                -> Dict[str, Any]:

    all_receives = [
            post_receives(part_dci)
            for part_dci in distributed_comm_infos.values()]

    context: Dict[str, Any] = {}
    for pid, part_dci, part_receives in zip(
            parts.toposorted_partitions, distributed_comm_infos.values(),
            all_receives):

        inputs = {"queue": queue}

        # FIXME: necessary?
        context.update(part_receives[1].results)

        for k, v in part_receives[0].items():
            v[0].Wait()
            inputs.update({k: v[1]})

        # inputs.update({k: v[1] for k, v in part_receives[0].items()})
        # {part.name: actx.from_numpy(recv.wait())
        # {recv.results: None
        #     for recv in part_receives})

        # inputs.update(context)

        # print(f"{context=}")
        # print(prg_per_partition[pid].program)

        # FIXME: necessary?
        inputs.update({
            k: context[k] for k in parts.partition_id_to_input_names[pid]
            if k in context})

        _evt, result_dict = prg_per_partition[pid](**inputs)

        context.update(result_dict)

        for name, send_node in part_dci.part_output_name_to_send_node.items():
            mpi_send(send_node.dest_rank, send_node.comm_tag, context[name])
            del context[name]

    return context

# }}}

# vim: foldmethod=marker
