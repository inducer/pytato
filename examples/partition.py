#!/usr/bin/env python

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.transform import (TopoSortMapper, execute_partitions,
                              generate_code_for_partitions, find_partitions)

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class MyPartitionId():
    num: int


def get_partition_id(topo_list, expr) -> MyPartitionId:
    # Partition nodes into groups of two
    res = MyPartitionId(topo_list.index(expr)//2)
    return res


def main():
    x_in = np.random.randn(2, 2)
    x = pt.make_data_wrapper(x_in)
    y = pt.stack([x@x.T, 2*x, 42+x])
    y = y + 55

    tm = TopoSortMapper()
    tm(y)

    from functools import partial
    pfunc = partial(get_partition_id, tm.topological_order)

    # Find the partitions
    parts = find_partitions(pt.DictOfNamedArrays({"out": y}), pfunc)

    from pytato.visualization import get_dot_graph_from_partitions, show_dot_graph
    show_dot_graph(get_dot_graph_from_partitions(parts))


    # Show the partitions
    # pt.show_dot_graph(y, pfunc)

    # # Execute the partitions
    # ctx = cl.create_some_context()
    # queue = cl.CommandQueue(ctx)

    # prg_per_partition = generate_code_for_partitions(parts)

    # context = execute_partitions(parts, prg_per_partition, queue)

    # final_res = context[parts.partition_id_to_output_names[
    #                         parts.toposorted_partitions[-1]][0]]

    # # Execute the unpartitioned code for comparison
    # prg = pt.generate_loopy(y)
    # evt, (out, ) = prg(queue)

    # assert np.allclose(out, final_res)

    # print("Partitioning test succeeded.")


if __name__ == "__main__":
    main()
