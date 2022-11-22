"""
.. currentmodule:: pytato
.. autofunction::  verify_distributed_partition
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


from typing import Any, FrozenSet, Dict, Set, Optional, Sequence, TYPE_CHECKING

import numpy as np

from pytato.distributed.nodes import CommTagType, DistributedRecv
from pytato.partition import PartId
from pytato.distributed.partition import (
        DistributedGraphPartition, CommunicationOpIdentifier)
from pytato.array import ShapeType

import attrs


import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import mpi4py.MPI


# {{{ data structures

@attrs.define(frozen=True)
class _SummarizedDistributedSend:
    src_rank: int
    dest_rank: int
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
    name_to_recv_node: Dict[_DistributedName, DistributedRecv]
    name_to_send_node: Dict[_DistributedName, _SummarizedDistributedSend]

    @property
    def rank(self) -> int:
        return self.pid.rank

# }}}


# {{{ errors

class DistributedPartitionVerificationError(ValueError):
    pass


class DuplicateSendError(DistributedPartitionVerificationError):
    pass


class DuplicateRecvError(DistributedPartitionVerificationError):
    pass


class MissingSendError(DistributedPartitionVerificationError):
    pass


class MissingRecvError(DistributedPartitionVerificationError):
    pass

# }}}


# {{{ verify_distributed_partition

def verify_distributed_partition(mpi_communicator: mpi4py.MPI.Comm,
        partition: DistributedGraphPartition) -> None:
    """
    Verify that

    - a feasible execution order exists among graph parts across the global,
      partitioned, distributed data flow graph, consisting of all values of
      *partition* across all ranks.
    - sends and receives for a given triple of
      **(source rank, destination rank, tag)** are unique.
    - there is a one-to-one mapping between instances of :class:`DistributedRecv`
      and :class:`DistributedSend`

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
            name_to_recv_node={_DistributedName(my_rank, name): recv
                for name, recv in part.name_to_recv_node.items()},
            name_to_send_node={
                _DistributedName(my_rank, name):
                _SummarizedDistributedSend(
                            src_rank=my_rank,
                            dest_rank=send.dest_rank,
                            comm_tag=send.comm_tag,
                            shape=send.data.shape,
                            dtype=send.data.dtype)
                for name, send in part.name_to_send_node.items()})

    # Gather the _SummarizedDistributedGraphPart's to rank 0
    all_summarized_parts_gathered: Optional[
            Sequence[Dict[_DistributedPartId, _SummarizedDistributedGraphPart]]] = \
            mpi_communicator.gather(summarized_parts, root=root_rank)

    if mpi_communicator.rank == root_rank:
        assert all_summarized_parts_gathered

        all_summarized_parts = {
                dpid: sumpart
                for rank_parts in all_summarized_parts_gathered
                for dpid, sumpart in rank_parts.items()}

        # Every node in the graph is a _SummarizedDistributedGraphPart
        pid_to_needed_pids: Dict[_DistributedPartId, Set[_DistributedPartId]] = {}

        def add_needed_pid(pid: _DistributedPartId,
                           needed_pid: _DistributedPartId) -> None:
            pid_to_needed_pids.setdefault(pid, set()).add(needed_pid)

        all_recvs: Set[CommunicationOpIdentifier] = set()

        # {{{ gather information on who produces output

        name_to_computing_pid: Dict[_DistributedName, _DistributedPartId] = {}
        for sumpart in all_summarized_parts.values():
            for out_name in sumpart.output_names:
                assert out_name not in name_to_computing_pid
                name_to_computing_pid[out_name] = sumpart.pid

        # }}}

        # {{{ gather information on who receives which names

        name_to_receiving_pid: Dict[_DistributedName, _DistributedPartId] = {}
        for sumpart in all_summarized_parts.values():
            for recv_name in sumpart.name_to_recv_node:
                assert recv_name not in name_to_computing_pid
                assert recv_name not in name_to_receiving_pid
                name_to_receiving_pid[recv_name] = sumpart.pid

        # }}}

        # {{{ gather information on senders

        comm_id_to_sending_pid: \
                Dict[CommunicationOpIdentifier, _DistributedPartId] = {}
        for sumpart in all_summarized_parts.values():
            for sumsend in sumpart.name_to_send_node.values():
                comm_id = CommunicationOpIdentifier(
                        src_rank=sumsend.src_rank,
                        dest_rank=sumsend.dest_rank,
                        comm_tag=sumsend.comm_tag)

                if comm_id in comm_id_to_sending_pid:
                    raise DuplicateSendError(
                            f"duplicate send for comm id: '{comm_id}'")
                comm_id_to_sending_pid[comm_id] = sumpart.pid

        # }}}

        # {{{ add edges to the DAG of distributed graph parts

        for sumpart in all_summarized_parts.values():
            pid_to_needed_pids[sumpart.pid] = set(sumpart.needed_pids)

            # Loop through all receives, assert that combination of
            # (src_rank, dest_rank, tag) is unique.
            for dname, dist_recv in sumpart.name_to_recv_node.items():
                comm_id = CommunicationOpIdentifier(
                        src_rank=dist_recv.src_rank,
                        dest_rank=dname.rank,
                        comm_tag=dist_recv.comm_tag)

                if comm_id in all_recvs:
                    raise DuplicateRecvError(f"Duplicate recv: '{comm_id}'")

                all_recvs.add(comm_id)

                # Add edges between sends and receives (cross-rank)
                try:
                    sending_pid = comm_id_to_sending_pid[comm_id]
                except KeyError:
                    raise MissingSendError(
                        f"no matching send for recv on '{comm_id}'")

                add_needed_pid(sumpart.pid, sending_pid)

            # Add edges between output_names and partition_input_names (intra-rank)
            for input_name in sumpart.partition_input_names:
                defining_pid = name_to_computing_pid.get(input_name)

                if defining_pid is None:
                    defining_pid = name_to_receiving_pid.get(input_name)

                if defining_pid is None:
                    raise AssertionError(
                        f"name '{input_name}' in part {sumpart} not defined "
                        "via output or receive")

                if defining_pid == sumpart.pid:
                    # Yes, we look at our own sends. But we don't need to
                    # include an edge for them--it'll look like a cycle.
                    pass
                else:
                    assert defining_pid.rank == sumpart.pid.rank
                    add_needed_pid(sumpart.pid, defining_pid)

        # }}}

        # Loop through all sends again, making sure there exists a matching recv
        for s in comm_id_to_sending_pid:
            if s not in all_recvs:
                raise MissingRecvError(f"no matching recv for send: {s=}")

        # Do a topological sort to check for any cycles

        from pytools.graph import compute_topological_order, CycleError
        from pytato.partition import PartitionInducedCycleError
        try:
            compute_topological_order(pid_to_needed_pids)
        except CycleError:
            raise PartitionInducedCycleError

        logger.info("verify_distributed_partition completed successfully.")

# }}}

# vim: foldmethod=marker
