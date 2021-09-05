#!/usr/bin/env python

from mpi4py import MPI  # pylint: disable=import-error
comm = MPI.COMM_WORLD

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.transform import (GraphToDictMapper, TopoSortMapper,
                              PartitionId as PartitionIdBase,
                              find_partitions, reverse_graph,
                              tag_nodes_with_starting_point, execute_partitions)

# from pytato.visualization import show_dot_graph

from dataclasses import dataclass
from pytato.array import (make_distributed_send, make_distributed_recv,
                                DistributedRecv, DistributedSend)


@dataclass(frozen=True, eq=True)
class PartitionId(PartitionIdBase):
    fed_sends: object
    feeding_recvs: object


def get_partition_id(node_to_fed_sends, node_to_feeding_recvs, expr) -> \
                     PartitionId:
    return PartitionId(frozenset(node_to_fed_sends[expr]),
                       frozenset(node_to_feeding_recvs[expr]))


def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    x_in = np.random.randn(20, 20)
    x = pt.make_data_wrapper(x_in)
    bnd = make_distributed_send(x[0], dest_rank=(rank-1) % size, comm_tag="halo")

    halo = make_distributed_recv(x[9], src_rank=(rank+1) % size, comm_tag="halo",
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

    # Find the partitions
    parts = find_partitions(y, pfunc)

    # Execute the partitions
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = execute_partitions(parts, queue)

    final_res = context[parts.partition_id_to_output_names[
                            parts.toposorted_partitions[-1]][0]]

    # Execute the unpartitioned code for comparison
    prg = pt.generate_loopy(y)
    evt, (out, ) = prg(queue)

    print(out)
    print("------------")
    print(final_res)

    # assert np.allclose(out, final_res)


if __name__ == "__main__":
    main()
