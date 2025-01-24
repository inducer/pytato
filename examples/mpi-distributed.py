#!/usr/bin/env python

from mpi4py import MPI  # pylint: disable=import-error


comm = MPI.COMM_WORLD

import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

import pytato as pt
from pytato import (
    execute_distributed_partition,
    find_distributed_partition,
    generate_code_for_partition,
    make_distributed_recv,
    number_distributed_tags,
    staple_distributed_send,
)


def main():
    pt.set_traceback_tag_enabled()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.default_rng()

    x_in = rng.integers(100, size=(4, 4))
    x_in_dev = cl_array.to_device(queue, x_in)
    x = pt.make_data_wrapper(x_in_dev)

    if size < 2:
        raise RuntimeError("it doesn't make sense to run the "
                           "distributed-memory test single-rank"
                           # and self-sends aren't supported for now
                           )

    mytag_x = (main, "x")
    x_plus = staple_distributed_send(x, dest_rank=(rank-1) % size,
            comm_tag=mytag_x, stapled_to=make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=mytag_x, shape=(4, 4),
                dtype=int))

    y = x+x_plus

    mytag_y = (main, "y")
    y_plus = staple_distributed_send(y, dest_rank=(rank-1) % size,
            comm_tag=mytag_y, stapled_to=make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=mytag_y, shape=(4, 4),
                dtype=int))

    z = y+y_plus

    # Find the partition
    outputs = pt.make_dict_of_named_arrays({"out": z})
    distributed_parts = find_distributed_partition(comm, outputs)

    distributed_parts, _ = number_distributed_tags(
            comm, distributed_parts, base_tag=42)
    prg_per_partition = generate_code_for_partition(distributed_parts)

    if 0:
        from pytato.visualization import show_dot_graph
        show_dot_graph(distributed_parts)

    if 0:
        # Sanity check
        from pytato.visualization import get_dot_graph_from_partition
        get_dot_graph_from_partition(distributed_parts)

    pt.verify_distributed_partition(comm, distributed_parts)

    context = execute_distributed_partition(distributed_parts, prg_per_partition,
                                             queue, comm)

    final_res = context["out"].get(queue)

    comm.isend(x_in, dest=(rank-1) % size, tag=42)
    ref_x_plus = comm.recv(source=(rank+1) % size, tag=42)

    ref_y_in = x_in + ref_x_plus

    comm.isend(ref_y_in, dest=(rank-1) % size, tag=43)
    ref_y_plus = comm.recv(source=(rank+1) % size, tag=43)

    ref_res = ref_y_in + ref_y_plus

    np.testing.assert_allclose(ref_res, final_res)

    if rank == 0:
        print("Distributed test succeeded.")


if __name__ == "__main__":
    main()
