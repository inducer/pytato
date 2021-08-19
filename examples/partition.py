#!/usr/bin/env python

import pytato as pt
import numpy as np
from pytato.transform import (GraphToDictMapper, reverse_graph, PartitionId,
                              tag_nodes_with_starting_point, PartitionFinder)

from pytato.visualization import show_graph
from pytato.array import Stack, AxisPermutation, IndexLambda

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class MyPartitionId(PartitionId):
    fed_sends: object
    feeding_recvs: object


def get_partition_id(node_to_fed_sends, node_to_feeding_recvs, expr) -> \
                     MyPartitionId:
    return MyPartitionId(frozenset(node_to_fed_sends[expr]),
                       frozenset(node_to_feeding_recvs[expr]))


def main():
    n = pt.make_size_param("n")

    array = pt.make_placeholder(name="array", shape=n, dtype=np.float64)

    stack = pt.stack([array, 2*array, array + 6])

    y = 2*array

    gdm = GraphToDictMapper()
    gdm(y)

    graph = gdm.graph_dict
    rev_graph = reverse_graph(graph)

    # FIXME: Inefficient... too many traversals
    node_to_feeding_recvs = {}
    for node in graph:
        node_to_feeding_recvs.setdefault(node, set())
        # FIXME: IndexLambda is just a place where the graph should be split:
        if isinstance(node, IndexLambda):
            tag_nodes_with_starting_point(graph, node, result=node_to_feeding_recvs)

    node_to_fed_sends = {}
    for node in rev_graph:
        node_to_fed_sends.setdefault(node, set())
        # FIXME: IndexLambda is just a place where the graph should be split:
        if isinstance(node, IndexLambda):
            tag_nodes_with_starting_point(rev_graph, node, result=node_to_fed_sends)

    from functools import partial
    pfunc = partial(get_partition_id, node_to_fed_sends,
                    node_to_feeding_recvs)

    pf = PartitionFinder(pfunc)
    new = pf(y)
    print(pf.partition_pair_to_edges)
    partition_id_to_output_names = {}
    partition_id_to_input_names = {}
    partitions = set()
    partitions_dict = {}
    for (pid_producer, pid_consumer), var_names in \
            pf.partition_pair_to_edges.items():
        partitions.add(pid_producer)
        partitions.add(pid_consumer)
        # FIXME?: Does this need to store *all* connected nodes?
        partitions_dict.setdefault(pid_consumer, []).append(pid_producer)
        for var_name in var_names:
            partition_id_to_output_names.setdefault(
                pid_producer, []).append(var_name)
            partition_id_to_input_names.setdefault(pid_consumer, []).append(var_name)
            print(var_name)

    from pytools.graph import compute_topological_order
    toposorted_partitions = compute_topological_order(partitions_dict)

    print("========")
    print(toposorted_partitions)

    # # codegen
    prg_per_partition = {pid:
            pt.generate_loopy(
                pt.DictOfNamedArrays(
                    {var_name: pf.var_name_to_result[var_name]
                        for var_name in partition_id_to_output_names[pid]
                     }))
            for pid in partitions}
    print(f"{graph=}")
    print(f"{node_to_feeding_recvs=}")
    print(f"{node_to_fed_sends=}")

    # # execution
    # context = {}
    # for pid in topsorted_partitions:
    #     # find names that are needed
    #     inputs = {...}
    #     context.update(prg_per_partition[f](**inputs))

    # print(new)

    show_graph(y)

    # print("========")
    # print(pf.cross_partition_name_to_value)


if __name__ == "__main__":
    main()
