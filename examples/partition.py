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
    x_in = np.random.randn(20, 20)
    x = pt.make_data_wrapper(x_in)
    y = pt.stack([x@x.T, 2*x, x + 6])

    gdm = GraphToDictMapper()
    gdm(y)

    tm = TopoSortMapper()
    tm(y)

    print(tm.topological_order)

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

    (toposorted_partitions, prg_per_partition,
    partition_id_to_input_names, partition_id_to_output_names) \
        = find_partitions(y, pfunc)

    # execution
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = {}
    for pid in toposorted_partitions:
        # find names that are needed
        inputs = {"queue": queue}
        for k in partition_id_to_input_names[pid]:
            if k in context:
                inputs[k] = context[k]
        # prg_per_partition[f](**inputs)
        res = prg_per_partition[pid](**inputs)

        context.update(res[1])

    prg = pt.generate_loopy(y)
    evt, (out, ) = prg(queue)

    # print(out)

    final_res = context[partition_id_to_output_names[toposorted_partitions[-1]][0]]
    # print(final_res)

    assert np.allclose(out, final_res)


if __name__ == "__main__":
    main()
