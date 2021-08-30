#!/usr/bin/env python

from mpi4py import MPI  # pylint: disable=import-error
comm = MPI.COMM_WORLD

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.transform import (GraphToDictMapper, TopoSortMapper, PartitionId,
                              find_partitions, reverse_graph,
                              tag_nodes_with_starting_point)

# from pytato.visualization import show_dot_graph

from dataclasses import dataclass
from pytato.array import (make_distributed_send, make_distributed_recv,
                                DistributedRecv, DistributedSend)


@dataclass(frozen=True, eq=True)
class PartitionId:
    fed_sends: object
    feeding_recvs: object


def get_partition_id(node_to_fed_sends, node_to_feeding_recvs, expr) -> \
                     PartitionId:
    return PartitionId(frozenset(node_to_fed_sends[expr]),
                       frozenset(node_to_feeding_recvs[expr]))


def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    x = pt.make_placeholder(name="myph", shape=(10,), dtype=float)
    bnd = pt.make_distributed_send(x[0], dest_rank=(rank-1) % size, comm_tag="halo")

    halo = pt.make_distributed_recv(x[9], src_rank=(rank+1) % size, comm_tag="halo",
            shape=(), dtype=float)

    y = x+bnd+halo

    bnd2 = pt.make_distributed_send(y[0], dest_rank=(rank-1) % size, comm_tag="halo")

    halo2 = pt.make_distributed_recv(y[9], src_rank=(rank+1) % size, comm_tag="halo",
            shape=(), dtype=float)
    y += bnd2 + halo2

    gdm = GraphToDictMapper()
    gdm(y)

    graph = gdm.graph_dict
    rev_graph = reverse_graph(graph)

    tm = TopoSortMapper()
    tm(y)

    print(tm.topological_order)

    graph = gdm.graph_dict
    rev_graph = reverse_graph(graph)

    # FIXME: Inefficient... too many traversals
    node_to_feeding_recvs = {}
    for node in graph:
        node_to_feeding_recvs.setdefault(node, set())
        if isinstance(node, DistributedRecv):
           tag_nodes_with_starting_point(graph, node, result=node_to_feeding_recvs)

    node_to_fed_sends = {}
    for node in rev_graph:
        node_to_fed_sends.setdefault(node, set())
        if isinstance(node, DistributedSend):
           tag_nodes_with_starting_point(rev_graph, node, result=node_to_fed_sends)

    from functools import partial
    pfunc = partial(get_partition_id, node_to_fed_sends,
                    node_to_feeding_recvs)

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
