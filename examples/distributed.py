#!/usr/bin/env python

from mpi4py import MPI  # pylint: disable=import-error
comm = MPI.COMM_WORLD

import pytato as pt
import pyopencl as cl
import numpy as np
from pytato.transform import (UsersCollector, TopoSortMapper,
                              reverse_graph,
                              tag_user_nodes,
                              )

from pytato.partition import (find_partitions, generate_code_for_partitions)
from pytato.distributed import (execute_partitions_distributed,
                                gather_distributed_comm_info)

# from pytato.visualization import show_dot_graph

from dataclasses import dataclass
from pytato.distributed import (make_distributed_send, make_distributed_recv,
                                DistributedRecv, DistributedSend)


@dataclass(frozen=True, eq=True)
class PartitionId():
    fed_sends: object
    feeding_recvs: object


def get_partition_id(node_to_fed_sends, node_to_feeding_recvs, expr) -> \
                     PartitionId:
    return PartitionId(frozenset(node_to_fed_sends[expr]),
                       frozenset(node_to_feeding_recvs[expr]))


def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    x_in = np.random.randn(4, 4)
    x = pt.make_data_wrapper(x_in)
    bnd = make_distributed_send(
        x, dest_rank=(rank-1) % size, comm_tag=42, shape=(4, 4), dtype=float)

    halo = make_distributed_recv(
        x, src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=float)

    # TODO: send returns scalar 0?
    y = x+bnd+halo

    # bnd2 = pt.make_distributed_send(y[0], dest_rank=(rank-1)%size, comm_tag="halo")

    # halo2 = pt.make_distributed_recv(y[9], src_rank=(rank+1)%size, comm_tag="halo",
    #         shape=(), dtype=float)
    # y += bnd2 + halo2

    gdm = UsersCollector()
    gdm(y)

    graph = gdm.node_to_users
    rev_graph = reverse_graph(graph)

    tm = TopoSortMapper()
    tm(y)

    print(tm.topological_order)

    # FIXME: Inefficient... too many traversals
    node_to_feeding_recvs = {}
    for node in graph:
        node_to_feeding_recvs.setdefault(node, set())
        if isinstance(node, DistributedRecv):
            tag_user_nodes(graph, tag=node, starting_point=node,
                            node_to_tags=node_to_feeding_recvs)
    # FIXME test that node_to_feeding_recvs maps to recvs

    node_to_fed_sends = {}
    for node in rev_graph:
        node_to_fed_sends.setdefault(node, set())
        if isinstance(node, DistributedSend):
            tag_user_nodes(rev_graph, tag=node, starting_point=node,
                            node_to_tags=node_to_fed_sends)
    # FIXME test that node_to_fed_sends maps to sends

    from functools import partial
    pfunc = partial(get_partition_id, node_to_fed_sends,
                    node_to_feeding_recvs)

    # print(f"{graph=}")
    # print(f"{node_to_feeding_recvs=}")
    # print(f"{node_to_fed_sends=}")

    # Find the partitions
    outputs = pt.DictOfNamedArrays({"out": y})
    parts = find_partitions(outputs, pfunc)
    distributed_comm_infos = gather_distributed_comm_info(parts)
    prg_per_partition = generate_code_for_partitions(parts)

    # Execute the partitions
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = execute_partitions_distributed(parts, prg_per_partition,
                                             queue, distributed_comm_infos)

    final_res = context[parts.partition_id_to_output_names[
                            parts.toposorted_partitions[-1]][0]]

    # Execute the unpartitioned code for comparison
    # prg = pt.generate_loopy(y)
    # evt, (out, ) = prg(queue)

    # print(out)
    print("------------")
    print(final_res)

    # assert np.allclose(out, final_res)


if __name__ == "__main__":
    main()
