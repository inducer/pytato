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

from typing import (Any, Callable, Dict, FrozenSet, Union, TypeVar, Set, Generic,
                    Optional, List, Tuple)

from pytato.array import (
        Array, IndexLambda, Placeholder, MatrixProduct, Stack, Roll,
        AxisPermutation, Slice, DataWrapper, SizeParam, DictOfNamedArrays,
        AbstractResultWithNamedArrays, Reshape, Concatenate, NamedArray,
        IndexRemappingBase, Einsum, InputArgumentBase)
from pytato.loopy import LoopyCall

T = TypeVar("T", Array, AbstractResultWithNamedArrays)
CombineT = TypeVar("CombineT")  # used in CombineMapper
ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]
R = FrozenSet[Array]

__doc__ = """
.. currentmodule:: pytato.transform

.. autoclass:: CopyMapper
.. autoclass:: DependencyMapper
.. autoclass:: InputGatherer
.. autoclass:: SizeParamGatherer
.. autoclass:: SubsetDependencyMapper
.. autoclass:: WalkMapper
.. autoclass:: CachedWalkMapper
.. autoclass:: TopoSortMapper
.. autofunction:: copy_dict_of_named_arrays
.. autofunction:: get_dependencies

"""


class UnsupportedArrayError(ValueError):
    pass


# {{{ mapper base class

class Mapper:
    def handle_unsupported_array(self, expr: T, *args: Any, **kwargs: Any) -> Any:
        """Mapper method that is invoked for
        :class:`pytato.Array` subclasses for which a mapper
        method does not exist in this mapper.
        """
        raise UnsupportedArrayError("%s cannot handle expressions of type %s"
                % (type(self).__name__, type(expr)))

    def map_foreign(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("%s encountered invalid foreign object: %s"
                % (type(self).__name__, repr(expr)))

    def rec(self, expr: T, *args: Any, **kwargs: Any) -> Any:
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

# }}}


# {{{ CopyMapper

class CopyMapper(Mapper):
    """Performs a deep copy of a :class:`pytato.array.Array`.
    The typical use of this mapper is to override individual ``map_`` methods
    in subclasses to permit term rewriting on an expression graph.
    """

    def __init__(self) -> None:
        self.cache: Dict[ArrayOrNames, ArrayOrNames] = {}

    def rec(self, expr: T) -> T:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]  # type: ignore
        result: T = super().rec(expr)
        self.cache[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        bindings: Dict[str, Array] = {
                name: self.rec(subexpr)
                for name, subexpr in sorted(expr.bindings.items())}
        return IndexLambda(expr=expr.expr,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                bindings=bindings,
                tags=expr.tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        assert expr.name is not None
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

    def map_concatenate(self, expr: Concatenate) -> Array:
        arrays = tuple(self.rec(arr) for arr in expr.arrays)
        return Concatenate(arrays=arrays, axis=expr.axis, tags=expr.tags)

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
        assert expr.name is not None
        return SizeParam(name=expr.name, tags=expr.tags)

    def map_einsum(self, expr: Einsum) -> Array:
        return Einsum(expr.access_descriptors,
                      tuple(self.rec(arg) for arg in expr.args))

    def map_named_array(self, expr: NamedArray) -> NamedArray:
        return type(expr)(self.rec(expr._container), expr.name)

    def map_dict_of_named_arrays(self,
            expr: DictOfNamedArrays) -> DictOfNamedArrays:
        return DictOfNamedArrays({key: self.rec(val.expr)
                                  for key, val in expr.items()})

    def map_loopy_call(self, expr: LoopyCall) -> LoopyCall:
        bindings = {name: (self.rec(subexpr) if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())}

        return LoopyCall(translation_unit=expr.translation_unit,
                         bindings=bindings,
                         entrypoint=expr.entrypoint)

    def map_reshape(self, expr: Reshape) -> Reshape:
        return Reshape(self.rec(expr.array),
                       # type-ignore reason: mypy can't tell 'rec' is being fed
                       # only arrays
                       newshape=tuple(self.rec(s)  # type: ignore
                                      if isinstance(s, Array)
                                      else s
                                      for s in expr.newshape),
                       order=expr.order,
                       tags=expr.tags)

# }}}


# {{{ CombineMapper

class CombineMapper(Mapper, Generic[CombineT]):
    def __init__(self) -> None:
        self.cache: Dict[ArrayOrNames, CombineT] = {}

    def rec(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        # type-ignore reason: type not compatible with super.rec() type
        result: CombineT = super().rec(expr)  # type: ignore
        self.cache[expr] = result
        return result

    # type-ignore reason: incompatible ret. type with super class
    def __call__(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        return self.rec(expr)

    def combine(self, *args: CombineT) -> CombineT:
        raise NotImplementedError

    def map_index_lambda(self, expr: IndexLambda) -> CombineT:
        return self.combine(*(self.rec(bnd)
                              for _, bnd in sorted(expr.bindings.items())),
                            *(self.rec(s)
                              for s in expr.shape if isinstance(s, Array)))

    def map_placeholder(self, expr: Placeholder) -> CombineT:
        return self.combine(*(self.rec(s)
                              for s in expr.shape if isinstance(s, Array)))

    def map_data_wrapper(self, expr: DataWrapper) -> CombineT:
        return self.combine(*(self.rec(s)
                              for s in expr.shape if isinstance(s, Array)))

    def map_matrix_product(self, expr: MatrixProduct) -> CombineT:
        return self.combine(self.rec(expr.x1), self.rec(expr.x2))

    def map_stack(self, expr: Stack) -> CombineT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.arrays))

    def map_roll(self, expr: Roll) -> CombineT:
        return self.combine(self.rec(expr.array))

    def map_axis_permutation(self, expr: AxisPermutation) -> CombineT:
        return self.combine(self.rec(expr.array))

    def map_slice(self, expr: Slice) -> CombineT:
        return self.combine(self.rec(expr.array))

    def map_reshape(self, expr: Reshape) -> CombineT:
        return self.combine(self.rec(expr.array))

    def map_concatenate(self, expr: Concatenate) -> CombineT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.arrays))

    def map_einsum(self, expr: Einsum) -> CombineT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.args))

    def map_named_array(self, expr: NamedArray) -> CombineT:
        return self.combine(self.rec(expr._container))

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> CombineT:
        return self.combine(*(self.rec(ary.expr)
                              for ary in expr.values()))

    def map_loopy_call(self, expr: LoopyCall) -> CombineT:
        return self.combine(*(self.rec(ary)
                              for _, ary in sorted(expr.bindings.items())
                              if isinstance(ary, Array)))

# }}}


# {{{ DependencyMapper

class DependencyMapper(CombineMapper[R]):
    """
    Maps a :class:`pytato.array.Array` to a :class:`frozenset` of
    :class:`pytato.array.Array`'s it depends on.
    .. warning::

        This returns every node in the graph! Consider a custom
        :class:`CombineMapper` or a :class:`SubsetDependencyMapper` instead.
    """

    def combine(self, *args: R) -> R:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_index_lambda(self, expr: IndexLambda) -> R:
        return self.combine(frozenset([expr]), super().map_index_lambda(expr))

    def map_placeholder(self, expr: Placeholder) -> R:
        return self.combine(frozenset([expr]), super().map_placeholder(expr))

    def map_data_wrapper(self, expr: DataWrapper) -> R:
        return self.combine(frozenset([expr]), super().map_data_wrapper(expr))

    def map_size_param(self, expr: SizeParam) -> R:
        return frozenset([expr])

    def map_matrix_product(self, expr: MatrixProduct) -> R:
        return self.combine(frozenset([expr]), super().map_matrix_product(expr))

    def map_stack(self, expr: Stack) -> R:
        return self.combine(frozenset([expr]), super().map_stack(expr))

    def map_roll(self, expr: Roll) -> R:
        return self.combine(frozenset([expr]), super().map_roll(expr))

    def map_axis_permutation(self, expr: AxisPermutation) -> R:
        return self.combine(frozenset([expr]), super().map_axis_permutation(expr))

    def map_slice(self, expr: Slice) -> R:
        return self.combine(frozenset([expr]), super().map_slice(expr))

    def map_reshape(self, expr: Reshape) -> R:
        return self.combine(frozenset([expr]), super().map_reshape(expr))

    def map_concatenate(self, expr: Concatenate) -> R:
        return self.combine(frozenset([expr]), super().map_concatenate(expr))

    def map_einsum(self, expr: Einsum) -> R:
        return self.combine(frozenset([expr]), super().map_einsum(expr))

    def map_named_array(self, expr: NamedArray) -> R:
        return self.combine(frozenset([expr]), super().map_named_array(expr))

# }}}


# {{{ SubsetDependencyMapper

class SubsetDependencyMapper(DependencyMapper):
    """
    Mapper to combine the dependencies of an expression that are a subset of
    *universe*.
    """
    def __init__(self, universe: FrozenSet[Array]):
        self.universe = universe
        super().__init__()

    def combine(self, *args: FrozenSet[Array]) -> FrozenSet[Array]:
        from functools import reduce
        return reduce(lambda acc, arg: acc | (arg & self.universe),
                      args,
                      frozenset())

# }}}


# {{{ InputGatherer

class InputGatherer(CombineMapper[FrozenSet[InputArgumentBase]]):
    """
    Mapper to combine all instances of :class:`pytato.array.InputArgumentBase` that
    an array expression depends on.
    """
    def combine(self, *args: FrozenSet[InputArgumentBase]
                ) -> FrozenSet[InputArgumentBase]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_placeholder(self, expr: Placeholder) -> FrozenSet[InputArgumentBase]:
        return self.combine(frozenset([expr]), super().map_placeholder(expr))

    def map_data_wrapper(self, expr: DataWrapper) -> FrozenSet[InputArgumentBase]:
        return self.combine(frozenset([expr]), super().map_data_wrapper(expr))

    def map_size_param(self, expr: SizeParam) -> FrozenSet[SizeParam]:
        return frozenset([expr])

# }}}


# {{{ SizeParamGatherer

class SizeParamGatherer(CombineMapper[FrozenSet[SizeParam]]):
    """
    Mapper to combine all instances of :class:`pytato.array.SizeParam` that
    an array expression depends on.
    """
    def combine(self, *args: FrozenSet[SizeParam]
                ) -> FrozenSet[SizeParam]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_size_param(self, expr: SizeParam) -> FrozenSet[SizeParam]:
        return frozenset([expr])

# }}}


# {{{ WalkMapper

class WalkMapper(Mapper):
    """
    A mapper that walks over all the arrays in a :class:`pytato.Array`.

    Users may override the specific mapper methods in a derived class or
    override :meth:`WalkMapper.visit` and :meth:`WalkMapper.post_visit`.

    .. automethod:: visit
    .. automethod:: post_visit
    """

    def visit(self, expr: Any) -> bool:
        """
        If this method returns *True*, *expr* is traversed during the walk.
        If this method returns *False*, *expr* is not traversed as a part of
        the walk.
        """
        return True

    def post_visit(self, expr: Any) -> None:
        """
        Callback after *expr* has been traversed.
        """
        pass

    def map_index_lambda(self, expr: IndexLambda) -> None:
        if not self.visit(expr):
            return

        for _, child in sorted(expr.bindings.items()):
            self.rec(child)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.rec(dim)

        self.post_visit(expr)

    def map_placeholder(self, expr: Placeholder) -> None:
        if not self.visit(expr):
            return

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.rec(dim)

        self.post_visit(expr)

    map_data_wrapper = map_placeholder
    map_size_param = map_placeholder

    def map_matrix_product(self, expr: MatrixProduct) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.x1)
        self.rec(expr.x2)

        self.post_visit(expr)

    def _map_index_remapping_base(self, expr: IndexRemappingBase) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.array)
        self.post_visit(expr)

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_slice = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def map_stack(self, expr: Stack) -> None:
        if not self.visit(expr):
            return

        for child in expr.arrays:
            self.rec(child)

        self.post_visit(expr)

    map_concatenate = map_stack

    def map_einsum(self, expr: Einsum) -> None:
        if not self.visit(expr):
            return

        for child in expr.args:
            self.rec(child)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.rec(dim)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        if not self.visit(expr):
            return

        for _, child in sorted(expr.bindings.items()):
            if isinstance(child, Array):
                self.rec(child)

        self.post_visit(expr)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        if not self.visit(expr):
            return

        for child in expr._data.values():
            self.rec(child)

        self.post_visit(expr)

    def map_named_array(self, expr: NamedArray) -> None:
        if not self.visit(expr):
            return

        self.rec(expr._container)

        self.post_visit(expr)

# }}}


# {{{ CachedWalkMapper

class CachedWalkMapper(WalkMapper):
    """
    WalkMapper that visits each node in the DAG exactly once. This loses some
    information compared to :class:`WalkMapper` as a node is visited only from
    one of its predecessors.
    """

    def __init__(self) -> None:
        self._visited_ids: Set[int] = set()

    # type-ignore reason: CachedWalkMapper.rec's type does not match
    # WalkMapper.rec's type
    def rec(self, expr: ArrayOrNames) -> None:  # type: ignore
        # Why choose id(x) as the cache key?
        # - Some downstream users (NamesValidityChecker) of this mapper rely on
        #   structurally equal objects being walked separately (e.g. to detect
        #   separate instances of Placeholder with the same name).

        if id(expr) in self._visited_ids:
            return

        # type-ignore reason: super().rec expects either 'Array' or
        # 'AbstractResultWithNamedArrays', passed 'ArrayOrNames'
        super().rec(expr)  # type: ignore
        self._visited_ids.add(id(expr))

# }}}


# {{{ TopoSortMapper

class TopoSortMapper(CachedWalkMapper):
    """A mapper that creates a list of nodes in topological order.

    .. attribute:: topological_order
        A list of all nodes in the graph in topological order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.topological_order: List[Array] = []

    def post_visit(self, expr: Any) -> None:
        self.topological_order.append(expr)

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

    data = {name: copy_mapper(val.expr) for name, val in source_dict.items()}
    return DictOfNamedArrays(data)


def get_dependencies(expr: DictOfNamedArrays) -> Dict[str, FrozenSet[Array]]:
    """Returns the dependencies of each named array in *expr*.
    """
    dep_mapper = DependencyMapper()

    return {name: dep_mapper(val.expr) for name, val in expr.items()}

# }}}


# {{{ Graph partitioning


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

    def map_matrix_product(self, expr: MatrixProduct, *args: Any) -> None:
        children = (expr.x1, expr.x2)
        for c in children:
            self.graph_dict.setdefault(c, set()).add(expr)
            self.rec(c)

    def map_stack(self, expr: Stack, *args: Any) -> None:
        for c in expr.arrays:
            self.graph_dict.setdefault(c, set()).add(expr)
            self.rec(c)

    def map_size_param(self, expr: SizeParam) -> None:
        # FIXME: Anything we need to do here?
        pass

    def map_axis_permutation(self, expr: AxisPermutation) -> None:
        self.graph_dict.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_slice(self, expr: Slice, *args: Any) -> None:
        self.graph_dict.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_data_wrapper(self, expr: DataWrapper) -> None:
        if isinstance(expr.data, Array):
            self.graph_dict.setdefault(expr.data, set()).add(expr)
            self.rec(expr.data)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.graph_dict.setdefault(dim, set()).add(expr)
                self.rec(dim)

    def map_index_lambda(self, expr: IndexLambda, *args: Any) -> None:
        children: Set[Array] = set()

        for child in expr.bindings.values():
            children = children | {child}
            self.rec(child)

        for dim in expr.shape:
            if isinstance(dim, Array):
                children = children | {dim}
                self.rec(dim)

        for c in children:
            self.graph_dict.setdefault(c, set()).add(expr)

    def __call__(self, expr: Array, *args: Any, **kwargs: Any) -> Any:
        self.graph_dict[expr] = set()  # FIXME: Is this necessary?
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
from abc import ABC


class PartitionId(ABC):
    """A class that represents a partition ID."""
    pass


class PartitionFinder(CopyMapper):
    """Find partitions."""

    def __init__(self, get_partition_id:
                                   Callable[[Array], PartitionId]) -> None:
        super().__init__()
        self.get_partition_id_init = get_partition_id
        self.cross_partition_name_to_value: Dict[str, Array] = {}

        self.name_index = 0

        # "nodes" of the coarsened graph
        self.partition_id_to_nodes: Dict[PartitionId, List[Any]] = {}

        # "edges" of the coarsened graph
        self.partition_pair_to_edges: Dict[Tuple[PartitionId, PartitionId],
                List[str]] = {}

        self.partion_id_to_placeholders: Dict[PartitionId, List[Any]] = {}

        self.var_name_to_result: Dict[str, Array] = {}

        self.expr_to_partition_id: Dict[Array, PartitionId] = {}  # FIXME: unused?

    def get_partition_id(self, expr: Array) -> PartitionId:
        try:
            return self.get_partition_id_init(expr)
        except ValueError:
            return self.expr_to_partition_id[expr]

    def does_edge_cross_partition_boundary(self, node1: Array, node2: Array) -> bool:
        return self.get_partition_id(node1) != self.get_partition_id(node2)

    def register_partition_id(self, expr: Array,
                              pid: Optional[PartitionId] = None) -> None:
        if not pid:
            pid = self.get_partition_id(expr)

        assert pid
        self.partition_id_to_nodes.setdefault(pid, list()).append(expr)

    def register_placeholder(self, name: str, expr: Array, placeholder: Array,
                             pid: Optional[PartitionId] = None) -> None:
        if not pid:
            pid = self.get_partition_id(expr)
        assert pid
        self.partion_id_to_placeholders.setdefault(pid, list()).append(placeholder)
        self.var_name_to_result[name] = expr
        self.expr_to_partition_id[expr] = pid
        print("REG PH", expr, pid)

    def make_new_name(self) -> str:
        self.name_index += 1
        res = "placeholder_" + str(self.name_index)
        assert res not in self.cross_partition_name_to_value
        return res

    def set_partition_pair_to_edges(self, expr1: Array, expr2: Array,
                                    name: str) -> None:
        p1 = self.get_partition_id(expr1)
        p2 = self.get_partition_id(expr2)

        self.partition_pair_to_edges.setdefault(
                (p1, p2), list()).append(name)

    def map_slice(self, expr: Slice, *args: Any) -> Slice:
        if self.does_edge_cross_partition_boundary(expr, expr.array):
            name = self.make_new_name()
            self.set_partition_pair_to_edges(expr, expr.array, name)
            new_binding: Array = make_placeholder(name, expr.array.shape,
                                                  expr.array.dtype,
                                                  tags=expr.array.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.array)
            self.register_placeholder(name, expr, new_binding)
        else:
            new_binding = self.rec(expr.array)
            self.register_partition_id(
                new_binding, self.get_partition_id(expr.array))

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
                    self.register_placeholder(name, expr, new_bindings[name])
                else:
                    new_bindings[name] = self.rec(dim)
                self.register_partition_id(
                    new_bindings[name], self.get_partition_id(dim))

        self.register_partition_id(expr)

        assert expr.name

        return Placeholder(name=expr.name,
                shape=new_bindings,  # type: ignore # FIXME: this is likely incorrect
                dtype=expr.dtype,
                tags=expr.tags)

    def map_matrix_product(self, expr: MatrixProduct, *args: Any) -> MatrixProduct:
        new_bindings: Dict[str, Array] = {}  # FIXME: unused
        for dim in expr.shape:
            if isinstance(dim, Array):
                name = self.make_new_name()
                if self.does_edge_cross_partition_boundary(expr, dim):
                    self.set_partition_pair_to_edges(expr, dim, name)
                    new_bindings[name] = make_placeholder(name, dim.shape, dim.dtype,
                                                          tags=dim.tags)
                    self.cross_partition_name_to_value[name] = self.rec(dim)
                    self.register_placeholder(name, expr, new_bindings[name])
                else:
                    new_bindings[name] = self.rec(dim)
                self.register_partition_id(new_bindings[name])

        if self.does_edge_cross_partition_boundary(expr, expr.x1):
            name = self.make_new_name()
            self.set_partition_pair_to_edges(expr, expr.x1, name)
            new_x1: Array = make_placeholder(name, expr.x1.shape,
                                                  expr.x1.dtype,
                                                  tags=expr.x1.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.x1)
            self.register_placeholder(name, expr, new_x1)
        else:
            new_x1 = self.rec(expr.x1)
        self.register_partition_id(
                new_x1, self.get_partition_id(expr.x1))

        if self.does_edge_cross_partition_boundary(expr, expr.x2):
            name = self.make_new_name()
            self.set_partition_pair_to_edges(expr, expr.x2, name)
            new_x2: Array = make_placeholder(name, expr.x2.shape,
                                                  expr.x2.dtype,
                                                  tags=expr.x2.tags)
            self.cross_partition_name_to_value[name] = self.rec(expr.x2)
            self.register_placeholder(name, expr, new_x2)
        else:
            new_x2 = self.rec(expr.x2)
        self.register_partition_id(
                new_x2, self.get_partition_id(expr.x2))

        self.register_partition_id(expr)

        return MatrixProduct(x1=new_x1, x2=new_x2,
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
                self.register_placeholder(name, expr, new_bindings[name])
            else:
                new_bindings[name] = self.rec(child)

            self.register_partition_id(
                new_bindings[name], self.get_partition_id(child))

        new_shapes: Dict[str, Array] = {}
        for dim in expr.shape:
            if isinstance(dim, Array):
                name = self.make_new_name()
                if self.does_edge_cross_partition_boundary(expr, dim):
                    self.set_partition_pair_to_edges(expr, dim, name)
                    new_shapes[name] = make_placeholder(name, dim.shape, dim.dtype,
                                                          tags=dim.tags)
                    self.cross_partition_name_to_value[name] = self.rec(dim)
                    self.register_placeholder(name, expr, new_shapes[name])
                else:
                    new_shapes[name] = self.rec(dim)

                self.register_partition_id(
                    new_shapes[name], self.get_partition_id(dim))

        return IndexLambda(expr=expr.expr,
                shape=new_shapes,  # type: ignore # FIXME: this is likely incorrect
                dtype=expr.dtype,
                bindings=new_bindings,
                tags=expr.tags)

    def __call__(self, expr: Array, *args: Any, **kwargs: Any) -> Array:
        return self.rec(expr)


def find_partitions(expr: Array, part_func: Callable[[Array], PartitionId]) -> Any:
    """Find partitions."""

    pf = PartitionFinder(part_func)
    pf(expr)
    # print(f"{pf.partition_pair_to_edges=}")
    partition_id_to_output_names: Dict[PartitionId, List[str]] = {}
    partition_id_to_input_names: Dict[PartitionId, List[str]] = {}
    partitions = set()
    partitions_dict: Dict[PartitionId, List[PartitionId]] = {}
    for (pid_producer, pid_consumer), var_names in \
            pf.partition_pair_to_edges.items():
        # print((pid_producer, pid_consumer), var_names)
        partitions.add(pid_producer)
        partitions.add(pid_consumer)
        if pid_producer not in partition_id_to_input_names:
            partition_id_to_input_names[pid_producer] = []
        if pid_producer not in partition_id_to_output_names:
            partition_id_to_output_names[pid_producer] = []
        if pid_consumer not in partition_id_to_input_names:
            partition_id_to_input_names[pid_consumer] = []
        if pid_consumer not in partition_id_to_output_names:
            partition_id_to_output_names[pid_consumer] = []
        # FIXME?: Does this need to store *all* connected nodes?:
        partitions_dict.setdefault(pid_consumer, []).append(pid_producer)
        for var_name in var_names:
            partition_id_to_output_names.setdefault(
                pid_producer, []).append(var_name)
            partition_id_to_input_names.setdefault(pid_consumer, []).append(var_name)
            # print(var_name)

    from pytools.graph import compute_topological_order
    toposorted_partitions = compute_topological_order(partitions_dict)

    # print("========")
    # print(f"{toposorted_partitions=}")

    # for pid in partitions:
    #     print(pid)

    # for i in partition_id_to_output_names:
    #     print(i)

    # print(partition_id_to_output_names)

    # codegen
    from pytato import generate_loopy
    prg_per_partition = {pid:
            generate_loopy(
                DictOfNamedArrays(
                    {var_name: pf.var_name_to_result[var_name]
                        for var_name in partition_id_to_output_names[pid]
                     }))
            for pid in partitions}

    return toposorted_partitions, prg_per_partition

# }}}

# vim: foldmethod=marker
