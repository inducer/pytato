from mpi4py import MPI  # pylint: disable=import-error
comm = MPI.COMM_WORLD
import pytato as pt
from pytato.transform import Mapper, CopyMapper

from pytato.array import (Array, IndexLambda, DistributedRecv, DistributedSend,
                          make_placeholder, Placeholder, Slice)

from typing import Callable, Any, Dict, List, Tuple
from dataclasses import dataclass


def advect(*args):
    """Test function."""
    return sum([a for a in args])


class GraphToDictMapper(Mapper):
    """
    Maps a graph to a dictionary representation.

    .. attribute:: graph_dict

        :class:`dict`, maps each node in the graph to the set of directly connected
        nodes, obeying the direction of each edge.
    """

    def __init__(self) -> None:
        """Initialize."""
        self.graph_dict = {}

    def map_placeholder(self, expr, *args):
        children = set()

        for dim in expr.shape:
            if isinstance(dim, Array):
                children = children | {dim}
                self.rec(dim, *args)

        for c in children:
            self.graph_dict.setdefault(c, set()).add(expr)

    def map_slice(self, expr, *args):
        self.graph_dict.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_index_lambda(self, expr: IndexLambda, *args) -> None:
        children = set()

        for child in expr.bindings.values():
            children = children | {child}
            self.rec(child)

        for dim in expr.shape:
            if isinstance(dim, Array):
                children = children | {dim}
                self.rec(dim)

        for c in children:
            self.graph_dict.setdefault(c, set()).add(expr)

    def map_distributed_send(self, expr, *args):
        self.graph_dict.setdefault(expr.data, set()).add(expr)
        self.rec(expr.data)

    def map_distributed_recv(self, expr, *args):
        self.graph_dict.setdefault(expr.data, set()).add(expr)
        self.rec(expr.data)

    def __call__(self, expr):
        return self.rec(expr)


def reverse_graph(graph):
    """Reverses a graph."""
    result = {}

    for node_key, edges in graph.items():
        for other_node_key in edges:
            result.setdefault(other_node_key, set()).add(node_key)

    return result


def tag_nodes_with_starting_point(graph, node, starting_point=None, result=None):
    """Tags nodes with their starting point."""
    if result is None:
        result = {}
    if starting_point is None:
        starting_point = node

    result.setdefault(node, set()).add(starting_point)
    if node in graph:
        for other_node_key in graph[node]:
            tag_nodes_with_starting_point(graph, other_node_key, starting_point,
                                          result)


@dataclass
class PartitionId:
    fed_sends: object
    feeding_recvs: object


def get_partition_id(node_to_fed_sends, node_to_feeding_recvs, expr) -> \
                     PartitionId:
    return PartitionId(node_to_fed_sends[expr], node_to_feeding_recvs[expr])


class PartitionFinder(CopyMapper):
    """Find partitions."""

    def __init__(self, get_partition_id:
                                   Callable[[Any], PartitionId]) -> None:
        super().__init__()
        self.get_partition_id = get_partition_id
        self.cross_partition_name_to_value = {}

        self.name_index = 0

        # "nodes" of the coarsened graph
        self.partition_id_to_nodes: Dict[PartitionId, List[Any]] = {}

        # "edges" of the coarsened graph
        self.partition_pair_to_edges: Dict[Tuple[PartitionId, PartitionId],
                List[str]] = {}

    def does_edge_cross_partition_boundary(self, node1, node2) -> bool:
        res = self.get_partition_id(node1) != self.get_partition_id(node2)
        if res:
            print("PART", node1, node2)
        else:
            print("NOPART", node1, node2)

        return res

    def make_new_name(self):
        self.name_index += 1
        res = "placeholder_" + str(self.name_index)
        assert res not in self.cross_partition_name_to_value
        return res

    def map_distributed_send(self, expr, *args):
        if self.does_edge_cross_partition_boundary(expr, expr.data):
            name = self.make_new_name()
            new_binding = make_placeholder(expr.data.shape, expr.data.dtype,
                                                  name, tags=expr.data.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.data)
        else:
            new_binding = self.rec(expr.data)
        return DistributedSend(new_binding)

    def map_distributed_recv(self, expr, *args):
        if self.does_edge_cross_partition_boundary(expr, expr.data):
            name = self.make_new_name()
            new_binding = make_placeholder(expr.data.shape, expr.data.dtype,
                                                  name, tags=expr.data.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.data)
        else:
            new_binding = self.rec(expr.data)

        return DistributedRecv(new_binding)

    def map_slice(self, expr, *args):
        if self.does_edge_cross_partition_boundary(expr, expr.array):
            name = self.make_new_name()
            new_binding = make_placeholder(expr.array.shape, expr.array.dtype,
                                                  name, tags=expr.array.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.array)
        else:
            new_binding = self.rec(expr.array)

        return Slice(array=new_binding,
                starts=expr.starts,
                stops=expr.stops,
                tags=expr.tags)

    def map_placeholder(self, expr, *args):
        new_bindings = {}
        for dim in expr.shape:
            if isinstance(dim, Array):
                name = self.make_new_name()
                if self.does_edge_cross_partition_boundary(expr, dim):
                    new_bindings[name] = make_placeholder(dim.shape, dim.dtype, name,
                                                          tags=dim.tags)
                    self.cross_partition_name_to_value[name] = self.rec(dim)
                else:
                    new_bindings[name] = self.rec(dim)

        return Placeholder(name=expr.name,
                shape=new_bindings,
                dtype=expr.dtype,
                tags=expr.tags)

    def map_index_lambda(self, expr: IndexLambda, *args) -> None:
        new_bindings = {}
        for child in expr.bindings.values():
            name = self.make_new_name()
            if self.does_edge_cross_partition_boundary(expr, child):

                new_bindings[name] = make_placeholder(child.shape, child.dtype, name,
                                                      tags=child.tags)
                self.cross_partition_name_to_value[name] = self.rec(child)
            else:
                new_bindings[name] = self.rec(child)

        new_shapes = {}
        for dim in expr.shape:
            if isinstance(dim, Array):
                name = self.make_new_name()
                if self.does_edge_cross_partition_boundary(expr, dim):
                    new_shapes[name] = make_placeholder(dim.shape, dim.dtype, name,
                                                          tags=dim.tags)
                    self.cross_partition_name_to_value[name] = self.rec(dim)
                else:
                    new_shapes[name] = self.rec(dim)

        return IndexLambda(expr=expr.expr,
                shape=new_shapes,
                dtype=expr.dtype,
                bindings=new_bindings,
                tags=expr.tags)

    def __call__(self, expr):
        return self.rec(expr)

# Partition the pytato graph whenever an edge (node1, node2) satisfies
# node_to_fed_sends[node1] != node_to_fed_sends[node2] or
# node_to_feeding_recvs[node1] != node_to_feeding_recvs[node2] If an edge
# crosses a partition boundary, replace the depended-upon node (that nominally
# lives in the other partition) with a Placeholder that lives in the current
# partition. For each partition, collect the placeholder names that it’s
# supposed to compute.


def show_graph(y: Array) -> None:
    """Show the graph."""
    dot_code = pt.get_dot_graph(y)

    from graphviz import Source  # pylint: disable=import-error
    src = Source(dot_code)
    src.view()


def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    x = pt.make_placeholder(shape=(10,), dtype=float)
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

    print(new)

    show_graph(new)

    print("========")
    print(pf.cross_partition_name_to_value)


if __name__ == "__main__":
    main()
