from __future__ import annotations

__copyright__ = """
Copyright (C) 2020 Matt Wala
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Any, Callable, Dict, FrozenSet

from pytato.array import (
        Array, IndexLambda, Placeholder, MatrixProduct, Stack,
        Roll, AxisPermutation, Slice, DataWrapper, SizeParam,
        DictOfNamedArrays, Reshape, Concatenate, IndexRemappingBase,
        DistributedSend, DistributedRecv)

__doc__ = """
.. currentmodule:: pytato.transform

Transforming Computations
-------------------------

.. autoclass:: CopyMapper
.. autoclass:: DependencyMapper
.. autoclass:: WalkMapper
.. autofunction:: copy_dict_of_named_arrays
.. autofunction:: get_dependencies

"""


# {{{ mapper classes

class UnsupportedArrayError(ValueError):
    pass


class Mapper:
    def handle_unsupported_array(self, expr: Array, *args: Any,
            **kwargs: Any) -> Any:
        """Mapper method that is invoked for
        :class:`pytato.Array` subclasses for which a mapper
        method does not exist in this mapper.
        """
        raise UnsupportedArrayError("%s cannot handle expressions of type %s"
                % (type(self).__name__, type(expr)))

    def map_foreign(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("%s encountered invalid foreign object: %s"
                % (type(self).__name__, repr(expr)))

    def rec(self, expr: Array, *args: Any, **kwargs: Any) -> Any:
        method: Callable[..., Array]

        try:
            method = getattr(self, expr._mapper_method)
        except AttributeError:
            if isinstance(expr, Array):
                return self.handle_unsupported_array(expr, *args, **kwargs)
            else:
                return self.map_foreign(expr, *args, **kwargs)

        return method(expr, *args, **kwargs)

    def __call__(self, expr: Array, *args: Any, **kwargs: Any) -> Any:
        return self.rec(expr, *args, **kwargs)


class CopyMapper(Mapper):
    """Performs a deep copy of a :class:`pytato.array.Array`.
    The typical use of this mapper is to override individual ``map_`` methods
    in subclasses to permit term rewriting on an expression graph.
    """

    def __init__(self) -> None:
        self.cache: Dict[Array, Array] = {}

    def rec(self, expr: Array) -> Array:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: Array = super().rec(expr)
        self.cache[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        bindings = {
                name: self.rec(subexpr)
                for name, subexpr in expr.bindings.items()}
        return IndexLambda(expr=expr.expr,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                bindings=bindings,
                tags=expr.tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        return Placeholder(name=expr.name,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                tags=expr.tags)

    def map_matrix_product(self, expr: MatrixProduct) -> Array:
        return MatrixProduct(x1=self.rec(expr.x1),
                x2=self.rec(expr.x2),
                tags=expr.tags)

    def map_stack(self, expr: Stack) -> Array:
        arrays = tuple(self.rec(arr) for arr in expr.arrays)
        return Stack(arrays=arrays, axis=expr.axis, tags=expr.tags)

    def map_roll(self, expr: Roll) -> Array:
        return Roll(array=self.rec(expr.array),
                shift=expr.shift,
                axis=expr.axis,
                tags=expr.tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        return AxisPermutation(array=self.rec(expr.array),
                axes=expr.axes,
                tags=expr.tags)

    def map_slice(self, expr: Slice) -> Array:
        return Slice(array=self.rec(expr.array),
                starts=expr.starts,
                stops=expr.stops,
                tags=expr.tags)

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        return DataWrapper(name=expr.name,
                data=expr.data,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                tags=expr.tags)

    def map_size_param(self, expr: SizeParam) -> Array:
        return SizeParam(name=expr.name, tags=expr.tags)

    def map_distributed_send(self, expr: DistributedSend) -> DistributedSend:
        return DistributedSend(expr.data)

    def map_distributed_recv(self, expr: DistributedRecv) -> DistributedRecv:
        return DistributedRecv(expr.data)


class DependencyMapper(Mapper):
    """
    Maps a :class:`pytato.array.Array` to a :class:`frozenset` of
    :class:`pytato.array.Array`'s it depends on.
    """
    def __init__(self) -> None:
        self.cache: Dict[Array, FrozenSet[Array]] = {}

    def rec(self, expr: Array) -> FrozenSet[Array]:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: FrozenSet[Array] = super().rec(expr)
        self.cache[expr] = result
        return result

    def combine(self, *args: FrozenSet[Array]) -> FrozenSet[Array]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_index_lambda(self, expr: IndexLambda) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), *(self.rec(bnd)
                                                 for bnd in expr.bindings.values()),
                            *(self.rec(s)
                              for s in expr.shape if isinstance(s, Array)))

    def map_placeholder(self, expr: Placeholder) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]),
                            *(self.rec(s)
                              for s in expr.shape if isinstance(s, Array)))

    def map_data_wrapper(self, expr: DataWrapper) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]),
                            *(self.rec(s)
                              for s in expr.shape if isinstance(s, Array)))

    def map_size_param(self, expr: SizeParam) -> FrozenSet[Array]:
        return frozenset([expr])

    def map_matrix_product(self, expr: MatrixProduct) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), self.rec(expr.x1), self.rec(expr.x2))

    def map_stack(self, expr: Stack) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), *(self.rec(ary)
                                                 for ary in expr.arrays))

    def map_roll(self, expr: Roll) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_axis_permutation(self, expr: AxisPermutation) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_slice(self, expr: Slice) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_reshape(self, expr: Reshape) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_concatenate(self, expr: Concatenate) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), *(self.rec(ary)
                                                 for ary in expr.arrays))

    def map_distributed_send(self, expr: DistributedSend) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_distributed_recv(self, expr: DistributedRecv) -> FrozenSet[Array]:
        return self.combine(frozenset([expr]), self.rec(expr.array))


class WalkMapper(Mapper):
    """
    A mapper that walks over all the arrays in a :class:`pytato.array.Array`.

    Users may override the specific mapper methods in a derived class or
    override :meth:`WalkMapper.visit` and :meth:`WalkMapper.post_visit`.

    .. automethod:: visit
    .. automethod:: post_visit
    """
    def visit(self, expr: Any, *args: Any) -> bool:
        """
        If this method returns *True*, *expr* is traversed during the walk.
        If this method returns *False*, *expr* is not traversed as a part of
        the walk.
        """
        return True

    def post_visit(self, expr: Any, *args: Any) -> None:
        """
        Callback after *expr* has been traversed.
        """
        pass

    def map_index_lambda(self, expr: IndexLambda, *args: Any) -> None:
        if not self.visit(expr, *args):
            return

        for child in expr.bindings.values():
            self.rec(child, *args)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.rec(dim, *args)

        self.post_visit(expr, *args)

    def map_placeholder(self, expr: Placeholder, *args: Any) -> None:
        if not self.visit(expr, *args):
            return

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.rec(dim, *args)

        self.post_visit(expr, *args)

    map_data_wrapper = map_placeholder
    map_size_param = map_placeholder

    def map_matrix_product(self, expr: MatrixProduct, *args: Any) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.x1, *args)
        self.rec(expr.x2, *args)

        self.post_visit(expr, *args)

    def _map_index_remapping_base(self,
            expr: IndexRemappingBase, *args: Any) -> None:
        if not self.visit(expr, *args):
            return

        self.rec(expr.array, *args)
        self.post_visit(expr, *args)

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_slice = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def map_stack(self, expr: Stack, *args: Any) -> None:
        if not self.visit(expr, *args):
            return

        for child in expr.arrays:
            self.rec(child, *args)

        self.post_visit(expr, *args)

    map_concatenate = map_stack

    def map_distributed_send(self, expr: DistributedSend, *args: Any) -> None:
        if not self.visit(expr, *args):
            return

        self.rec(expr.data, *args)

        self.post_visit(expr, *args)

    def map_distributed_recv(self, expr: DistributedRecv, *args: Any) -> None:
        if not self.visit(expr, *args):
            return

        self.rec(expr.data, *args)

        self.post_visit(expr, *args)

# }}}


# {{{ mapper frontends

def copy_dict_of_named_arrays(source_dict: DictOfNamedArrays,
        copy_mapper: CopyMapper) -> DictOfNamedArrays:
    """Copy the elements of a :class:`~pytato.DictOfNamedArrays` into a
    :class:`~pytato.DictOfNamedArrays`.

    :param source_dict: The :class:`~pytato.DictOfNamedArrays` to copy
    :param copy_mapper: A mapper that performs copies different array types
    :returns: A new :class:`~pytato.DictOfNamedArrays` containing copies of the
        items in *source_dict*
    """
    if not source_dict:
        return DictOfNamedArrays({})

    data = {name: copy_mapper(val) for name, val in source_dict.items()}
    return DictOfNamedArrays(data)


def get_dependencies(expr: DictOfNamedArrays) -> Dict[str, FrozenSet[Array]]:
    """Returns the dependencies of each named array in *expr*.
    """
    dep_mapper = DependencyMapper()

    return {name: dep_mapper(val) for name, val in expr.items()}

# }}}


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
            new_binding = make_placeholder(expr.data.shape, expr.data.dtype,
                                                  name, tags=expr.data.tags)
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
            new_binding = make_placeholder(expr.data.shape, expr.data.dtype,
                                                  name, tags=expr.data.tags)
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
            new_binding = make_placeholder(expr.array.shape, expr.array.dtype,
                                                  name, tags=expr.array.tags)
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
                    new_bindings[name] = make_placeholder(dim.shape, dim.dtype, name,
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

# vim: foldmethod=marker
