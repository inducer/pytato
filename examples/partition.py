#!/usr/bin/env python

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.transform import (GraphToDictMapper, TopoSortMapper, PartitionId,
                              PartitionFinder)

from pytato.visualization import show_dot_graph

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

    pf = PartitionFinder(pfunc)
    pf(y)
    print(f"{pf.partition_pair_to_edges=}")
    partition_id_to_output_names = {}
    partition_id_to_input_names = {}
    partitions = set()
    partitions_dict = {}
    for (pid_producer, pid_consumer), var_names in \
            pf.partition_pair_to_edges.items():
        print((pid_producer, pid_consumer), var_names)
        partitions.add(pid_producer)
        partitions.add(pid_consumer)
        if pid_producer not in partition_id_to_input_names:
            partition_id_to_input_names[pid_producer] = []
        if pid_producer not in partition_id_to_output_names:
            partition_id_to_output_names[pid_producer] = []
        if pid_consumer not in partition_id_to_input_names:
            partition_id_to_input_names[pid_consumer] = []
        if pid_consumer not in partition_id_to_output_names:
            partition_id_to_output_names[pid_consumer] = []
        # FIXME?: Does this need to store *all* connected nodes?:
        partitions_dict.setdefault(pid_consumer, []).append(pid_producer)
        for var_name in var_names:
            partition_id_to_output_names.setdefault(
                pid_producer, []).append(var_name)
            partition_id_to_input_names.setdefault(pid_consumer, []).append(var_name)
            # print(var_name)

    from pytools.graph import compute_topological_order
    toposorted_partitions = compute_topological_order(partitions_dict)

    print("========")
    print(f"{toposorted_partitions=}")

    for pid in partitions:
        print(pid)

    for i in partition_id_to_output_names:
        print(i)

    print(partition_id_to_output_names)

    # codegen
    prg_per_partition = {pid:
            pt.generate_loopy(
                pt.DictOfNamedArrays(
                    {var_name: pf.var_name_to_result[var_name]
                        for var_name in partition_id_to_output_names[pid]
                     }))
            for pid in partitions}


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
