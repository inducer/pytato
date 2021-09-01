#!/usr/bin/env python

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.transform import (TopoSortMapper, PartitionId, execute_partitions,
                              find_partitions)

from pytato.visualization import show_dot_graph

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class MyPartitionId(PartitionId):
    num: int


part_dict = {}

def get_partition_id(topo_list, expr) -> MyPartitionId:
    res = MyPartitionId(topo_list.index(expr)//2)
    if expr not in part_dict:
        part_dict[expr] = res
        print(expr, res)
    return res


def main():
    x_in = np.random.randn(2, 2)
    x = pt.make_data_wrapper(x_in)
    y = pt.stack([x@x.T, 2*x, x + 6])
    y = pt.roll(y, shift=1, axis=1)
    y = pt.reshape(y, newshape=(-1,))

    tm = TopoSortMapper()
    tm(y)

    from functools import partial
    pfunc = partial(get_partition_id, tm.topological_order)

    # Find the partitions
    parts = find_partitions(y, pfunc)

    # Execute the partitions
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = execute_partitions(parts, queue)

    # print(context)

    # for xxx in parts.toposorted_partitions:
    #     print("--", xxx.__repr__())

    # print(parts)

    # show_dot_graph(y)
    print(len(parts.toposorted_partitions))
    for k,v in parts.partition_id_to_nodes.items():
        print(k, v)
        # show_dot_graph(v[0])
    # Execute the unpartitioned code for comparison
    prg = pt.generate_loopy(y)
    evt, (out, ) = prg(queue)

    final_res = context[parts.partition_id_to_output_names[
                            parts.toposorted_partitions[-1]][0]]


    print(out)
    print("====")
    print(final_res)
    assert np.allclose(out, final_res)

    print("Partitioning test succeeded.")


if __name__ == "__main__":
    main()
