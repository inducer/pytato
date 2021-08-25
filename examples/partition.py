#!/usr/bin/env python

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.transform import (GraphToDictMapper, TopoSortMapper, PartitionId,
                              find_partitions)

# from pytato.visualization import show_dot_graph

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class MyPartitionId(PartitionId):
    num: int


def get_partition_id(topo_list, expr) -> MyPartitionId:
    return MyPartitionId(topo_list.index(expr))


def main():
    x_in = np.random.randn(20, 10)

    x = pt.make_data_wrapper(x_in)
    # n = pt.make_size_param("n")
    # array = pt.make_placeholder(name="array", shape=n, dtype=np.float64)
    # stack = pt.stack([array, 2*array, array + 6])
    # y = stack @ stack.T
    y = 2*x

    gdm = GraphToDictMapper()
    gdm(y)

    tm = TopoSortMapper()
    tm(y)

    print(tm.topological_order)
    # show_dot_graph(y)

    # 1/0

    from functools import partial
    pfunc = partial(get_partition_id, tm.topological_order)

    # graph = gdm.graph_dict
    # rev_graph = reverse_graph(graph)

    # # FIXME: Inefficient... too many traversals
    # node_to_feeding_recvs = {}
    # for node in graph:
    #     node_to_feeding_recvs.setdefault(node, set())
    #     # FIXME: IndexLambda is just a place where the graph should be split:
    #     if isinstance(node, IndexLambda):
    #        tag_nodes_with_starting_point(graph, node, result=node_to_feeding_recvs)

    # node_to_fed_sends = {}
    # for node in rev_graph:
    #     node_to_fed_sends.setdefault(node, set())
    #     # FIXME: IndexLambda is just a place where the graph should be split:
    #     if isinstance(node, IndexLambda):
    #        tag_nodes_with_starting_point(rev_graph, node, result=node_to_fed_sends)

    # from functools import partial
    # pfunc = partial(get_partition_id, node_to_fed_sends,
    #                 node_to_feeding_recvs)

    # print(f"{graph=}")
    # print(f"{node_to_feeding_recvs=}")
    # print(f"{node_to_fed_sends=}")

    toposorted_partitions, prg_per_partition = find_partitions(y, pfunc)

    # execution
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = {}
    for pid in toposorted_partitions:
        # find names that are needed
        # inputs = {...}
        # prg_per_partition[f](**inputs)
        res = prg_per_partition[pid](queue)

        context.update(res[1])

    print(context)


if __name__ == "__main__":
    main()
