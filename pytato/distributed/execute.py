"""
.. currentmodule:: pytato

.. autofunction:: execute_distributed_partition
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

from typing import Any, Dict, Hashable, Tuple, Optional, TYPE_CHECKING


from pytato.target import BoundProgram
from pytato.scalar_expr import INT_CLASSES

import numpy as np


from pytato.distributed.nodes import (
        DistributedRecv, DistributedSend)
from pytato.distributed.partition import (
        DistributedGraphPartition, DistributedGraphPart)

import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import mpi4py.MPI


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

    if any(part.name_to_recv_node for part in partition.parts.values()):
        recv_names_tup, recv_requests_tup, recv_buffers_tup = zip(*[
            (name,) + _post_receive(mpi_communicator, recv)
            for part in partition.parts.values()
            for name, recv in part.name_to_recv_node.items()])
        recv_names = list(recv_names_tup)
        recv_requests = list(recv_requests_tup)
        recv_buffers = list(recv_buffers_tup)
        del recv_names_tup
        del recv_requests_tup
        del recv_buffers_tup
    else:
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

        for name, send_node in part.name_to_send_node.items():
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
            import pyopencl.array as cl_array
            context[name] = cl_array.to_device(queue, buf, allocator=allocator)
            recv_names_completed.add(name)

    # {{{ main loop

    while pids_to_execute:
        ready_pids = {pid
                for pid in pids_to_execute
                # FIXME: Only O(n**2) altogether. Nobody is going to notice, right?
                if partition.parts[pid].needed_pids <= pids_executed
                and (set(partition.parts[pid].name_to_recv_node)
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
