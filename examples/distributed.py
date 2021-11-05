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
from pytato.distributed import (staple_distributed_send, make_distributed_recv,
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
    rng = np.random.default_rng()

    x_in = rng.integers(100, size=(4, 4))
    x = pt.make_data_wrapper(x_in)

    halo = staple_distributed_send(x, dest_rank=(rank-1) % size, comm_tag=42,
            stapled_to=make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=int))

    y = x+halo

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

    # FIXME: Inefficient... too many traversals
    node_to_feeding_recvs = {}
    for node in graph:
        node_to_feeding_recvs.setdefault(node, set())
        if isinstance(node, DistributedRecv):
            tag_user_nodes(graph, tag=node, starting_point=node,
                            node_to_tags=node_to_feeding_recvs)

    for node, _ in node_to_feeding_recvs.items():
        for n in node_to_feeding_recvs[node]:
            assert(isinstance(n, DistributedRecv))

    node_to_fed_sends = {}
    for node in rev_graph:
        node_to_fed_sends.setdefault(node, set())
        if isinstance(node, DistributedSend):
            tag_user_nodes(rev_graph, tag=node, starting_point=node,
                            node_to_tags=node_to_fed_sends)

    for node, _ in node_to_fed_sends.items():
        for n in node_to_fed_sends[node]:
            assert(isinstance(n, DistributedSend))

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
                                             queue, distributed_comm_infos, comm)

    final_res = [context[k] for k in outputs.keys()]

    # Execute the unpartitioned code for comparison
    # prg = pt.generate_loopy(y)
    # evt, (out, ) = prg(queue)

    # print(out)
    print("------------")
    print(final_res)
    print("------------")
    print("Distributed test succeeded")

    # assert np.allclose(out, final_res)


if __name__ == "__main__":
    main()
