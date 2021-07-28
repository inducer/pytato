from pytato.array import (Array, ShapeType, _SuppliedShapeAndDtypeMixin,
                          IndexLambda, Placeholder, Slice)
import numpy as np

from typing import Any, Tuple, Callable, Dict, List, Set, Optional
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
                 shape: Tuple[int, ...] = (), dtype=float,
                 tags=frozenset()) -> None:
        super().__init__(shape=shape, dtype=dtype, tags=tags)
        self.src_rank = src_rank
        self.comm_tag = comm_tag
        self.data = data


def make_distributed_send(data: Array, dest_rank: int, comm_tag: object) -> \
         DistributedSend:
    return DistributedSend(data, dest_rank, comm_tag)


def make_distributed_recv(data: Array, src_rank: int, comm_tag: object,
                          shape: Tuple[int, ...] = (), dtype=float,
                          tags=frozenset()) \
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
        """Initialize the GraphToDictMapper."""
        self.graph_dict: Dict[Array, Set[Array]] = {}

    def map_placeholder(self, expr: Placeholder, *args: Any) -> None:
        children: Set[Array] = set()

        for dim in expr.shape:
            if isinstance(dim, Array):
                children = children | {dim}
                self.rec(dim, *args)

        for c in children:
            self.graph_dict.setdefault(c, set()).add(expr)

    def map_slice(self, expr: Slice, *args: Any) -> None:
        self.graph_dict.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_index_lambda(self, expr: IndexLambda, *args: Any) -> None:
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

    def map_distributed_send(self, expr: DistributedSend, *args: Any) -> None:
        self.graph_dict.setdefault(expr.data, set()).add(expr)
        self.rec(expr.data)

    def map_distributed_recv(self, expr: DistributedRecv, *args: Any) -> None:
        self.graph_dict.setdefault(expr.data, set()).add(expr)
        self.rec(expr.data)

    def __call__(self, expr: Array, *args: Any, **kwargs: Any) -> Any:
        return self.rec(expr, *args)


def reverse_graph(graph: Dict[Array, Set[Array]]) -> Dict[Array, Set[Array]]:
    """Reverses a graph."""
    result: Dict[Array, Set[Array]] = {}

    for node_key, edges in graph.items():
        for other_node_key in edges:
            result.setdefault(other_node_key, set()).add(node_key)

    return result


def tag_nodes_with_starting_point(graph: Dict[Array, Set[Array]], node: Array,
        starting_point: Optional[Array] = None,
        result: Optional[Dict[Array, Set[Array]]] = None) -> None:
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
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class PartitionId:
    fed_sends: object
    feeding_recvs: object


class PartitionFinder(CopyMapper):
    """Find partitions."""

    def __init__(self, get_partition_id:
                                   Callable[[Any], Any]) -> None:
        super().__init__()
        self.get_partition_id = get_partition_id
        self.cross_partition_name_to_value: Dict[str, Array] = {}

        self.name_index = 0

        # "nodes" of the coarsened graph
        self.partition_id_to_nodes: Dict[PartitionId, List[Any]] = {}

        # "edges" of the coarsened graph
        self.partition_pair_to_edges: Dict[Tuple[PartitionId, PartitionId],
                List[str]] = {}

        self.partion_id_to_placeholders: Dict[PartitionId, List[Any]] = {}

        self.var_name_to_result: Dict[str, Array] = {}

    def does_edge_cross_partition_boundary(self, node1: Array, node2: Array) -> bool:
        p1 = self.get_partition_id(node1)
        p2 = self.get_partition_id(node2)
        crosses: bool = p1 != p2

        return crosses

    def register_partition_id(self, expr: Array, pid: Optional[PartitionId] = None) -> None:
        if not pid:
            pid = self.get_partition_id(expr)

        assert pid
        self.partition_id_to_nodes.setdefault(pid, list()).append(expr)

    def register_placeholder(self, expr: Array, placeholder: Array, pid: Optional[PartitionId] = None) -> None:
        if not pid:
            pid = self.get_partition_id(expr)
        assert pid
        self.partion_id_to_placeholders.setdefault(pid, list()).append(placeholder)

    def make_new_name(self) -> str:
        self.name_index += 1
        res = "placeholder_" + str(self.name_index)
        assert res not in self.cross_partition_name_to_value
        return res

    def set_partition_pair_to_edges(self, expr1: Array, expr2: Array, name: str) -> None:
        p1 = self.get_partition_id(expr1)
        p2 = self.get_partition_id(expr2)

        self.partition_pair_to_edges.setdefault(
                (p1, p2), list()).append(name)



    def map_distributed_send(self, expr: DistributedSend, *args: Any) -> DistributedSend:
        if self.does_edge_cross_partition_boundary(expr, expr.data):
            name = self.make_new_name()
            self.set_partition_pair_to_edges(expr, expr.data, name)
            new_binding: Array = make_placeholder(name, expr.data.shape, expr.data.
                                           dtype, tags=expr.data.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.data)
            self.register_placeholder(expr, new_binding)
        else:
            new_binding = self.rec(expr.data)
            self.register_partition_id(new_binding, self.get_partition_id(expr.data))

        self.register_partition_id(expr)
        return DistributedSend(new_binding)

    def map_distributed_recv(self, expr: DistributedRecv, *args: Any) -> DistributedRecv:
        if self.does_edge_cross_partition_boundary(expr, expr.data):
            name = self.make_new_name()
            self.set_partition_pair_to_edges(expr, expr.data, name)
            new_binding: Array = make_placeholder(name, expr.data.shape, expr.data.dtype,
                                                  tags=expr.data.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.data)
            self.register_placeholder(expr, new_binding)
        else:
            new_binding = self.rec(expr.data)
            self.register_partition_id(new_binding)

        self.register_partition_id(expr)
        return DistributedRecv(new_binding)

    def map_slice(self, expr: Slice, *args: Any) -> Slice:
        if self.does_edge_cross_partition_boundary(expr, expr.array):
            name = self.make_new_name()
            self.set_partition_pair_to_edges(expr, expr.array, name)
            new_binding: Array = make_placeholder(name, expr.array.shape, expr.array.dtype,
                                                tags=expr.array.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.array)
            self.register_placeholder(expr, new_binding)
        else:
            new_binding = self.rec(expr.array)
            self.register_partition_id(new_binding, self.get_partition_id(expr.array))

        self.register_partition_id(expr)

        return Slice(array=new_binding,
                starts=expr.starts,
                stops=expr.stops,
                tags=expr.tags)

    def map_placeholder(self, expr: Placeholder, *args: Any) -> Placeholder:
        new_bindings: Dict[str, Array] = {}
        for dim in expr.shape:
            if isinstance(dim, Array):
                name = self.make_new_name()
                if self.does_edge_cross_partition_boundary(expr, dim):
                    self.set_partition_pair_to_edges(expr, dim, name)
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

    def map_index_lambda(self, expr: IndexLambda, *args: Any) -> IndexLambda:
        new_bindings: Dict[str, Array] = {}
        for child in expr.bindings.values():
            name = self.make_new_name()
            if self.does_edge_cross_partition_boundary(expr, child):
                self.set_partition_pair_to_edges(expr, child, name)

                new_bindings[name] = make_placeholder(name, child.shape, child.dtype,
                                                      tags=child.tags)
                self.cross_partition_name_to_value[name] = self.rec(child)
            else:
                new_bindings[name] = self.rec(child)

        new_shapes: Dict[str, Array] = {}
        for dim in expr.shape:
            if isinstance(dim, Array):
                name = self.make_new_name()
                if self.does_edge_cross_partition_boundary(expr, dim):
                    self.set_partition_pair_to_edges(expr, dim, name)
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

    def __call__(self, expr: Array, *args: Any, **kwargs: Any) -> Array:
        return self.rec(expr)

# }}}
