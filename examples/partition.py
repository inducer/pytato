#!/usr/bin/env python

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.partition import (execute_partition,
                              generate_code_for_partition, find_partition)

from pytato.transform import TopoSortMapper

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
    outputs = pt.DictOfNamedArrays({"out": y})
    partition = find_partition(outputs, pfunc)

    # Show the partitions
    from pytato.visualization import get_dot_graph_from_partition
    get_dot_graph_from_partition(partition)

    # Execute the partitions
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    prg_per_partition = generate_code_for_partition(partition)

    context = execute_partition(partition, prg_per_partition, queue)

    final_res = [context[k] for k in outputs.keys()]

    # Execute the unpartitioned code for comparison
    prg = pt.generate_loopy(y)
    _, (out, ) = prg(queue)

    np.testing.assert_allclose([out], final_res)

    print("Partitioning test succeeded.")


if __name__ == "__main__":
    main()
