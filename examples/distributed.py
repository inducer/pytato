#!/usr/bin/env python

from mpi4py import MPI  # pylint: disable=import-error
comm = MPI.COMM_WORLD

import pytato as pt
import pyopencl as cl
import numpy as np

from pytato import (find_partition_distributed, generate_code_for_partition,
    execute_partition_distributed, gather_distributed_comm_info,
    staple_distributed_send, make_distributed_recv)


def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.default_rng()

    x_in = rng.integers(100, size=(4, 4))
    x = pt.make_data_wrapper(x_in)

    halo = staple_distributed_send(x, dest_rank=(rank-1) % size, comm_tag=42,
            stapled_to=make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=int))

    y = x+halo

    # Find the partition
    outputs = pt.DictOfNamedArrays({"out": y})
    parts = find_partition_distributed(outputs)
    distributed_parts = gather_distributed_comm_info(parts)
    prg_per_partition = generate_code_for_partition(distributed_parts)

    if 0:
        from pytato.visualization import show_dot_graph
        show_dot_graph(distributed_parts)

    # Sanity check
    from pytato.visualization import get_dot_graph_from_partition
    get_dot_graph_from_partition(distributed_parts)

    # Execute the distributed partition
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = execute_partition_distributed(distributed_parts, prg_per_partition,
                                             queue, comm)

    final_res = [context[k] for k in outputs.keys()]

    ref_res = comm.bcast(final_res)

    np.testing.assert_allclose(ref_res, final_res)

    if rank == 0:
        print("Distributed test succeeded.")


if __name__ == "__main__":
    main()
