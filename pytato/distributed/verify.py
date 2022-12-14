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


from typing import (Any, FrozenSet, Dict, Set, Optional, Sequence,
                    TYPE_CHECKING, Mapping)

import numpy as np

from pytato.distributed.nodes import CommTagType, DistributedRecv
from pytato.partition import PartId
from pytato.distributed.partition import (DistributedGraphPartition,
        _KeyT, _ValueT, CommunicationOpIdentifier)
from pytato.array import ShapeType, DictOfNamedArrays

import attrs


import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import mpi4py.MPI
    from mpi4py import MPI


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
    input_name_to_recv_node: Dict[_DistributedName, DistributedRecv]
    output_name_to_send_node: Dict[_DistributedName, _SummarizedDistributedSend]

    @property
    def rank(self) -> int:
        return self.pid.rank


@attrs.define(frozen=True)
class _CommIdentifier:
    src_rank: int
    dest_rank: int
    comm_tag: CommTagType

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


# {{{ _dict_union_mpi

def _dict_union_mpi(
        dict_a: Mapping[_KeyT, _ValueT], dict_b: Mapping[_KeyT, _ValueT],
        mpi_data_type: MPI.Datatype) -> Mapping[_KeyT, _ValueT]:
    assert mpi_data_type is None
    result = dict(dict_a)
    result.update(dict_b)
    return result

# }}}


# {{{ _get_comm_to_needed_comms

def _get_comm_to_needed_comms(mpi_communicator: mpi4py.MPI.Comm,
        outputs: DictOfNamedArrays) -> \
        Dict[CommunicationOpIdentifier, FrozenSet[CommunicationOpIdentifier]]:
    my_rank = mpi_communicator.rank

    from pytato.distributed.partition import _LocalSendRecvDepGatherer
    lsrdg = _LocalSendRecvDepGatherer(local_rank=my_rank)
    lsrdg(outputs)
    local_send_id_to_needed_local_recv_ids = \
            lsrdg.local_send_id_to_needed_local_recv_ids

    from mpi4py import MPI
    dict_union_mpi_op = MPI.Op.Create(
            # type ignore reason: mpi4py misdeclares op functions as returning
            # None.
            _dict_union_mpi,  # type: ignore[arg-type]
            commute=True)
    try:
        # FIXME: allreduce might not be necessary for all use cases
        comm_to_needed_comms: \
            Dict[CommunicationOpIdentifier, FrozenSet[CommunicationOpIdentifier]] = \
            mpi_communicator.allreduce(
                local_send_id_to_needed_local_recv_ids, dict_union_mpi_op)
    finally:
        dict_union_mpi_op.Free()

    return comm_to_needed_comms

# }}}


# {{{ verify_distributed_dag_pre_partition

def verify_distributed_dag_pre_partition(mpi_communicator: mpi4py.MPI.Comm,
                                         outputs: DictOfNamedArrays) -> None:
    """
    Verify that a global, unpartitioned graph does not contain a cycle.

    .. warning::

        This is an MPI-collective operation.
    """
    my_rank = mpi_communicator.rank
    root_rank = 0

    comm_to_needed_comms = _get_comm_to_needed_comms(mpi_communicator, outputs)

    if my_rank == root_rank:
        from pytools.graph import compute_topological_order
        compute_topological_order(comm_to_needed_comms)


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
            input_name_to_recv_node={_DistributedName(my_rank, name): recv
                for name, recv in part.input_name_to_recv_node.items()},
            output_name_to_send_node={
                _DistributedName(my_rank, name):
                _SummarizedDistributedSend(
                            src_rank=my_rank,
                            dest_rank=send.dest_rank,
                            comm_tag=send.comm_tag,
                            shape=send.data.shape,
                            dtype=send.data.dtype)
                for name, send in part.output_name_to_send_node.items()})

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

        all_recvs: Set[_CommIdentifier] = set()

        # {{{ gather information on who produces output

        output_to_defining_pid: Dict[_DistributedName, _DistributedPartId] = {}
        for sumpart in all_summarized_parts.values():
            for out_name in sumpart.output_names:
                assert out_name not in output_to_defining_pid
                output_to_defining_pid[out_name] = sumpart.pid

        # }}}

        # {{{ gather information on senders

        comm_id_to_sending_pid: Dict[_CommIdentifier, _DistributedPartId] = {}
        for sumpart in all_summarized_parts.values():
            for sumsend in sumpart.output_name_to_send_node.values():
                comm_id = _CommIdentifier(
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
            for dname, dist_recv in sumpart.input_name_to_recv_node.items():
                comm_id = _CommIdentifier(
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
                # Input names from recv nodes have no corresponding output_name
                if input_name in sumpart.input_name_to_recv_node.keys():
                    continue
                defining_pid = output_to_defining_pid[input_name]
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
