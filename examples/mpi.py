from mpi4py import MPI  # pylint: disable=import-error
comm = MPI.COMM_WORLD
import pytato as pt
from pytato.distributed import (GraphToDictMapper, reverse_graph,
                              tag_nodes_with_starting_point, PartitionFinder)

from pytato.visualization import show_graph

# from pytato.array import (Array, IndexLambda, DistributedRecv, DistributedSend,
#                           make_placeholder, Placeholder, Slice)

# from typing import Callable, Any, Dict, List, Tuple

from pytato.distributed import DistributedRecv, DistributedSend


def advect(*args):
    """Test function."""
    return sum([a for a in args])


from dataclasses import dataclass


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

    # print(graph)
    # print(rev_graph)

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

    pf = PartitionFinder(pfunc)
    new = pf(y)
    print(pf.partition_pair_to_edges)
    partition_id_to_output_names = {}
    partition_id_to_input_names = {}
    partitions = set()
    for (pid_producer, pid_consumer), var_names in \
            pf.partition_pair_to_edges.items():
        partitions.add(pid_producer)
        partitions.add(pid_consumer)
        print(pid_consumer)
        for var_name in var_names:
            partition_id_to_output_names.setdefault(pid_producer, []).append(var_name)
            partition_id_to_input_names.setdefault(pid_consumer, []).append(var_name)
            print(var_name)

    # pytools.graph
    # topsorted_partitions = topsort(partitions)

    # # codegen
    # prg_per_partition = {pid:
    #         pt.generate_loopy(
    #             pt.DictOfNamedArrays(
    #                 {var_name: pf.var_name_to_result[var_name]
    #                     for var_name in partition_id_to_output_names[pid]
    #                     }))
    #         for pid in partitions}

    # # execution
    # context = {}
    # for pid in topsorted_partitions:
    #     # find names that are needed
    #     inputs = {...}
    #     context.update(prg_per_partition[f](**inputs))

    print(new)

    # show_graph(y)

    print("========")
    print(pf.cross_partition_name_to_value)


if __name__ == "__main__":
    main()
