#!/usr/bin/env python

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.transform import (GraphToDictMapper, TopoSortMapper, PartitionId,
                              find_partitions)

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class MyPartitionId(PartitionId):
    num: int


def get_partition_id(topo_list, expr) -> MyPartitionId:
    return MyPartitionId(topo_list.index(expr))


def main():
    x_in = np.random.randn(20, 20)
    x = pt.make_data_wrapper(x_in)
    y = pt.stack([x@x.T, 2*x, x + 6])
    y = pt.roll(y, shift=1, axis=1)
    y = pt.reshape(y, newshape=(-1,))

    gdm = GraphToDictMapper()
    gdm(y)

    tm = TopoSortMapper()
    tm(y)

    from functools import partial
    pfunc = partial(get_partition_id, tm.topological_order)

    # Find the partitions
    (toposorted_partitions, prg_per_partition,
    partition_id_to_input_names, partition_id_to_output_names) \
        = find_partitions(y, pfunc)

    # Execute the partitions
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = {}
    for pid in toposorted_partitions:
        # find names that are needed
        inputs = {"queue": queue}
        for k in partition_id_to_input_names[pid]:
            if k in context:
                inputs[k] = context[k]
        res = prg_per_partition[pid](**inputs)

        context.update(res[1])

    # Execute unpartitioned code for comparison
    prg = pt.generate_loopy(y)
    evt, (out, ) = prg(queue)

    final_res = context[partition_id_to_output_names[toposorted_partitions[-1]][0]]

    assert np.allclose(out, final_res)


if __name__ == "__main__":
    main()
