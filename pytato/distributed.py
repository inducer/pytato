from pytato.array import (Array, ShapeType, _SuppliedShapeAndDtypeMixin,
                          IndexLambda, Placeholder, Slice)
import numpy as np

from typing import Any, Tuple, Callable, Dict, List
from pytato.transform import Mapper, CopyMapper


# {{{ Communication nodes


class DistributedSend(Array):

    _mapper_method = "map_distributed_send"
    _fields = Array._fields + ("data",)

    def __init__(self, data: Array, dest_rank: int = 0,
                 comm_tag: object = None) -> None:
        super().__init__()
        self.data = data

    @property
    def shape(self) -> ShapeType:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype


class DistributedRecv(_SuppliedShapeAndDtypeMixin, Array):

    _fields = Array._fields + ("src_rank", "comm_tag", "data")
    _mapper_method = "map_distributed_recv"

    def __init__(self, data: Array, src_rank: int = 0, comm_tag: object = None,
                 shape: Tuple = (), dtype=float, tags=frozenset()) -> None:
        super().__init__(shape=shape, dtype=dtype, tags=tags)
        self.src_rank = src_rank
        self.comm_tag = comm_tag
        self.data = data


def make_distributed_send(data: Array, dest_rank: int, comm_tag: object) -> \
         DistributedSend:
    return DistributedSend(data, dest_rank, comm_tag)


def make_distributed_recv(data: Array, src_rank: int, comm_tag: object,
                          shape=(), dtype=float, tags=frozenset()) \
                          -> DistributedRecv:
    return DistributedRecv(data, src_rank, comm_tag, shape, dtype, tags)


# }}}


# {{{ Transformations


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


from pytato.array import make_placeholder


class PartitionFinder(CopyMapper):
    """Find partitions."""

    def __init__(self, get_partition_id:
                                   Callable[[Any], Any]) -> None:
        super().__init__()
        self.get_partition_id = get_partition_id
        self.cross_partition_name_to_value = {}

        self.name_index = 0

        # "nodes" of the coarsened graph
        self.partition_id_to_nodes: Dict[PartitionId, List[Any]] = {}

        # "edges" of the coarsened graph
        self.partition_pair_to_edges: Dict[Tuple[PartitionId, PartitionId],
                List[str]] = {}

        self.var_name_to_result: Dict[str, Array] = {}

    def does_edge_cross_partition_boundary(self, node1, node2) -> bool:
        res = self.get_partition_id(node1) != self.get_partition_id(node2)
        if res:
            print("PART", node1, node2)
        else:
            print("NOPART", node1, node2)

        return res

    def register_partition_id(self, expr: Array) -> None:
        return
        pid = self.get_partition_id(expr)
        self.partition_id_to_nodes.setdefault(pid, list()).append(expr)

    def register_placeholder(self, expr, placeholder) -> None:
        return
        pid = self.get_partition_id(expr)
        self.partion_id_to_placeholders.setdefault(pid, list()).append(placeholder)

    def make_new_name(self):
        self.name_index += 1
        res = "placeholder_" + str(self.name_index)
        assert res not in self.cross_partition_name_to_value
        return res

    def map_distributed_send(self, expr, *args):
        if self.does_edge_cross_partition_boundary(expr, expr.data):
            name = self.make_new_name()
            new_binding = make_placeholder(name, expr.data.shape, expr.data.
                                           dtype, tags=expr.data.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.data)
            self.register_placeholder(expr, new_binding)
        else:
            new_binding = self.rec(expr.data)
            self.register_partition_id(new_binding)

        self.register_partition_id(expr)
        return DistributedSend(new_binding)

    def map_distributed_recv(self, expr, *args):
        if self.does_edge_cross_partition_boundary(expr, expr.data):
            name = self.make_new_name()
            new_binding = make_placeholder(name, expr.data.shape, expr.data.dtype,
                                                  tags=expr.data.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.data)
            self.register_placeholder(expr, new_binding)
        else:
            new_binding = self.rec(expr.data)
            self.register_partition_id(new_binding)

        self.register_partition_id(expr)
        return DistributedRecv(new_binding)

    def map_slice(self, expr, *args):
        if self.does_edge_cross_partition_boundary(expr, expr.array):
            name = self.make_new_name()
            new_binding = make_placeholder(name, expr.array.shape, expr.array.dtype,
                                                tags=expr.array.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.array)
            self.register_placeholder(expr, new_binding)
        else:
            new_binding = self.rec(expr.array)
            self.register_partition_id(expr.array)

        self.register_partition_id(expr)

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
                    new_bindings[name] = make_placeholder(name, dim.shape, dim.dtype,
                                                          tags=dim.tags)
                    self.cross_partition_name_to_value[name] = self.rec(dim)
                else:
                    new_bindings[name] = self.rec(dim)
                self.register_partition_id(new_bindings[name])

        self.register_partition_id(expr)

        return Placeholder(name=expr.name,
                shape=new_bindings,
                dtype=expr.dtype,
                tags=expr.tags)

    def map_index_lambda(self, expr: IndexLambda, *args) -> None:
        new_bindings = {}
        for child in expr.bindings.values():
            name = self.make_new_name()
            if self.does_edge_cross_partition_boundary(expr, child):

                new_bindings[name] = make_placeholder(name, child.shape, child.dtype,
                                                      tags=child.tags)
                self.cross_partition_name_to_value[name] = self.rec(child)
            else:
                new_bindings[name] = self.rec(child)

        new_shapes = {}
        for dim in expr.shape:
            if isinstance(dim, Array):
                name = self.make_new_name()
                if self.does_edge_cross_partition_boundary(expr, dim):
                    new_shapes[name] = make_placeholder(name, dim.shape, dim.dtype,
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

# }}}
