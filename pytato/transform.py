from __future__ import annotations
from dataclasses import dataclass

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

from typing import (Any, Callable, Dict, FrozenSet, Hashable, Union, TypeVar, Set,
                    Generic, Optional, List, Tuple)

from pytato.array import (
        Array, IndexLambda, Placeholder, MatrixProduct, Stack, Roll,
        AxisPermutation, DataWrapper, SizeParam, DictOfNamedArrays,
        AbstractResultWithNamedArrays, Reshape, Concatenate, NamedArray,
        IndexRemappingBase, Einsum, InputArgumentBase,
        BasicIndex, AdvancedIndexInContiguousAxes, AdvancedIndexInNoncontiguousAxes,
        IndexBase, make_placeholder)
from pytato.loopy import LoopyCall

T = TypeVar("T", Array, AbstractResultWithNamedArrays)
CombineT = TypeVar("CombineT")  # used in CombineMapper
ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]
R = FrozenSet[Array]

__doc__ = """
.. currentmodule:: pytato.transform

.. autoclass:: CopyMapper
.. autoclass:: CombineMapper
.. autoclass:: DependencyMapper
.. autoclass:: InputGatherer
.. autoclass:: SizeParamGatherer
.. autoclass:: SubsetDependencyMapper
.. autoclass:: WalkMapper
.. autoclass:: CachedWalkMapper
.. autoclass:: TopoSortMapper
.. autoclass:: GraphToDictMapper
.. autoclass:: GraphPartitioner
.. autoclass:: CodePartitions
.. autofunction:: copy_dict_of_named_arrays
.. autofunction:: get_dependencies
.. autofunction:: find_partitions
.. autofunction:: execute_partitions

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

    def _map_index_base(self, expr: IndexBase) -> Array:
        return type(expr)(self.rec(expr.array),
                          tuple(self.rec(idx) if isinstance(idx, Array) else idx
                                for idx in expr.indices))

    def map_basic_index(self, expr: BasicIndex) -> Array:
        return self._map_index_base(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> Array:
        return self._map_index_base(expr)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> Array:
        return self._map_index_base(expr)

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

    def _map_index_base(self, expr: IndexBase) -> CombineT:
        return self.combine(self.rec(expr.array),
                            *(self.rec(idx)
                              for idx in expr.indices
                              if isinstance(idx, Array)))

    def map_basic_index(self, expr: BasicIndex) -> CombineT:
        return self._map_index_base(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> CombineT:
        return self._map_index_base(expr)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> CombineT:
        return self._map_index_base(expr)

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

    def _map_index_base(self, expr: IndexBase) -> R:
        return self.combine(frozenset([expr]), super()._map_index_base(expr))

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
    map_reshape = _map_index_remapping_base

    def _map_index_base(self, expr: IndexBase) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.array)

        for idx in expr.indices:
            if isinstance(idx, Array):
                self.rec(idx)

        self.post_visit(expr)

    def map_basic_index(self, expr: BasicIndex) -> None:
        return self._map_index_base(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> None:
        return self._map_index_base(expr)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> None:
        return self._map_index_base(expr)

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

        self.post_visit(expr)

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

    :members: topological_order
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
    Maps a graph to a dictionary representation mapping a node to its children.

    .. attribute:: graph_dict
    """

    def __init__(self) -> None:
        """Initialize the GraphToDictMapper."""
        self.graph_dict: Dict[Any, Set[Any]] = {}

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays, *args: Any) -> None:
        for child in expr._data.values():
            self.graph_dict.setdefault(child, set()).add(expr)
            self.rec(child)

    def map_named_array(self, expr: NamedArray, *args: Any) -> None:
        self.graph_dict.setdefault(expr._container, set()).add(expr)
        self.rec(expr._container)

    def map_loopy_call(self, expr: LoopyCall, *args: Any) -> None:
        for _, child in sorted(expr.bindings.items()):
            if isinstance(child, Array):
                self.graph_dict.setdefault(child, set()).add(expr)
                self.rec(child)

    def map_einsum(self, expr: Einsum, *args: Any) -> None:
        for arg in expr.args:
            self.graph_dict.setdefault(arg, set()).add(expr)
            self.rec(arg)

    def map_reshape(self, expr: Reshape, *args: Any) -> None:
        children: Set[Array] = set()

        for dim in expr.shape:
            if isinstance(dim, Array):
                children = children | {dim}
                self.rec(dim, *args)

        for c in children:
            self.graph_dict.setdefault(c, set()).add(expr)

        self.graph_dict.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

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

    def map_concatenate(self, expr: Concatenate, *args: Any) -> None:
        for c in expr.arrays:
            self.graph_dict.setdefault(c, set()).add(expr)
            self.rec(c)

    def map_stack(self, expr: Stack, *args: Any) -> None:
        for c in expr.arrays:
            self.graph_dict.setdefault(c, set()).add(expr)
            self.rec(c)

    def map_roll(self, expr: Roll, *args: Any) -> None:
        self.graph_dict.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_size_param(self, expr: SizeParam) -> None:
        # FIXME: Anything we need to do here?
        pass

    def map_axis_permutation(self, expr: AxisPermutation) -> None:
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
        return self.rec(expr, *args)


def reverse_graph(graph: Dict[Array, Set[Array]]) -> Dict[Array, Set[Array]]:
    """Reverses a graph."""
    result: Dict[Array, Set[Array]] = {}

    for node_key, edges in graph.items():
        for other_node_key in edges:
            result.setdefault(other_node_key, set()).add(node_key)

    return result


def tag_nodes_with_starting_point(graph: Dict[Array, Set[Array]], tag: Any,
        starting_point: Optional[Array] = None,
        node_to_tags: Optional[Dict[Optional[Array], Set[Array]]] = None) -> None:
    """Tags nodes reachable from *starting_point*."""
    if node_to_tags is None:
        node_to_tags = {}
    node_to_tags.setdefault(starting_point, set()).add(tag)
    if starting_point in graph:
        for other_node_key in graph[starting_point]:
            tag_nodes_with_starting_point(graph, other_node_key, tag,
                                          node_to_tags)


class GraphPartitioner(CopyMapper):
    """Given a function *get_partition_id*, produces subgraphs representing
    the computation. Users should not use this class directly, but use
    :meth:`find_partitions` instead.
    """

    def __init__(self, get_partition_id:
                                   Callable[[Array], Hashable]) -> None:
        super().__init__()

        # Function to determine the Partition ID
        self.get_partition_id = get_partition_id

        # Naming for newly created PlaceHolders at partition edges
        from pytools import UniqueNameGenerator
        self.name_generator = UniqueNameGenerator(forced_prefix="_dist_ph_")

        # "edges" of the partitioned graph
        self.partition_pair_to_edges: Dict[Tuple[Hashable, Hashable],
                List[str]] = {}

        self.var_name_to_result: Dict[str, Array] = {}

    def does_edge_cross_partition_boundary(self, node1: Array, node2: Array) -> bool:
        return self.get_partition_id(node1) != self.get_partition_id(node2)

    def register_placeholder(self, name: str, expr: Array) -> None:
        self.var_name_to_result[name] = expr

    def make_new_placeholder_name(self) -> str:
        return self.name_generator()

    def add_interpartition_edge(self, target: Array, dependency: Array,
                                placeholder_name: str) -> None:
        pid_target = self.get_partition_id(target)
        pid_dependency = self.get_partition_id(dependency)

        self.partition_pair_to_edges.setdefault(
                (pid_target, pid_dependency), []).append(placeholder_name)

    def _handle_new_binding(self, expr: Array, child: Array) -> Array:
        if self.does_edge_cross_partition_boundary(expr, child):
            new_name = self.make_new_placeholder_name()
            # If an edge crosses a partition boundary, replace the
            # depended-upon node (that nominally lives in the other partition)
            # with a Placeholder that lives in the current partition. For each
            # partition, collect the placeholder names that it’s supposed to
            # compute.
            self.add_interpartition_edge(expr, child, new_name)
            new_binding: Array = make_placeholder(new_name, child.shape,
                                                  child.dtype,
                                                  tags=child.tags)
            self.register_placeholder(new_name, expr)

        else:
            new_binding = self.rec(child)

        return new_binding

    def map_reshape(self, expr: Reshape, *args: Any) -> Reshape:
        new_binding = self._handle_new_binding(expr, expr.array)

        new_shapes = tuple(self._handle_new_binding(expr, s)  # type: ignore
                                      if isinstance(s, Array)
                                      else s
                                      for s in expr.newshape)

        return Reshape(array=new_binding,
                newshape=new_shapes,
                order=expr.order,
                tags=expr.tags)

    def map_einsum(self, expr: Einsum, *args: Any) -> Einsum:
        new_bindings: List[Array] = []
        for c in expr.args:
            new_bindings.append(self._handle_new_binding(expr, c))

        return Einsum(
                     access_descriptors=expr.access_descriptors,
                     args=tuple(new_bindings),
                     tags=expr.tags)

    def map_concatenate(self, expr: Concatenate, *args: Any) -> Concatenate:
        new_bindings: List[Array] = []
        for c in expr.arrays:
            new_bindings.append(self._handle_new_binding(expr, c))

        return Concatenate(
                     arrays=tuple(new_bindings),
                     axis=expr.axis,
                     tags=expr.tags)

    def map_stack(self, expr: Stack, *args: Any) -> Stack:
        new_bindings: List[Array] = []
        for c in expr.arrays:
            new_bindings.append(self._handle_new_binding(expr, c))

        return Stack(
                     arrays=tuple(new_bindings),
                     axis=expr.axis,
                     tags=expr.tags)

    def map_roll(self, expr: Roll, *args: Any) -> Roll:
        new_binding = self._handle_new_binding(expr, expr.array)

        return Roll(array=new_binding,
                shift=expr.shift,
                axis=expr.axis,
                tags=expr.tags)

    def map_placeholder(self, expr: Placeholder, *args: Any) -> Placeholder:
        new_shapes: List[Array] = []
        for dim in expr.shape:
            if isinstance(dim, Array):
                new_shapes.append(self._handle_new_binding(expr, dim))

        assert expr.name

        return Placeholder(name=expr.name,
                shape=tuple(new_shapes),
                dtype=expr.dtype,
                tags=expr.tags)

    def map_matrix_product(self, expr: MatrixProduct, *args: Any) -> MatrixProduct:

        new_x1 = self._handle_new_binding(expr, expr.x1)
        new_x2 = self._handle_new_binding(expr, expr.x2)

        return MatrixProduct(x1=new_x1, x2=new_x2,
                tags=expr.tags)

    def map_index_lambda(self, expr: IndexLambda, *args: Any) -> IndexLambda:
        new_bindings: Dict[str, Array] = {}
        for name, child in expr.bindings.items():
            new_bindings[name] = self._handle_new_binding(expr, child)

        new_shapes: List[Array] = []
        for dim in expr.shape:
            if isinstance(dim, Array):
                new_shapes.append(self._handle_new_binding(expr, dim))

        return IndexLambda(expr=expr.expr,
                shape=tuple(new_shapes),
                dtype=expr.dtype,
                bindings=new_bindings,
                tags=expr.tags)

    def __call__(self, expr: Array, *args: Any, **kwargs: Any) -> Array:
        return self.rec(expr)


from pytato.target import BoundProgram


@dataclass
class CodePartitions:
    """Store partitions and their code."""
    toposorted_partitions: List[Hashable]
    prg_per_partition: Dict[Hashable, BoundProgram]
    partition_id_to_input_names: Dict[Hashable, List[str]]
    partition_id_to_output_names: Dict[Hashable, List[str]]


def find_partitions(expr: Array, part_func: Callable[[Array], Hashable]) ->\
        CodePartitions:
    """Partitions the *expr* according to *part_func* and generates code for
    each partition.

    :param expr: The expression to partition.
    :param part_func: A callable that returns an instance of
        :class:`Hashable` for a node.
    :returns: An instance of :class:`CodePartitions` that contains the partitions.
    """

    pf = GraphPartitioner(part_func)
    pf(expr)

    partition_id_to_output_names: Dict[Hashable, List[str]] = {
        v: [] for _, v in pf.partition_pair_to_edges.keys()}
    partition_id_to_input_names: Dict[Hashable, List[str]] = {
        k: [] for k, _ in pf.partition_pair_to_edges.keys()}

    partitions = set()

    # Used to compute the topological order
    partitions_dict: Dict[Hashable, List[Hashable]] = {}

    for (pid_target, pid_dependency), var_names in \
            pf.partition_pair_to_edges.items():
        partitions.add(pid_target)
        partitions.add(pid_dependency)

        partitions_dict.setdefault(pid_dependency, []).append(pid_target)

        for var_name in var_names:
            partition_id_to_output_names.setdefault(
                pid_target, []).append(var_name)
            partition_id_to_input_names.setdefault(
                pid_dependency, []).append(var_name)

    from pytools.graph import compute_topological_order
    toposorted_partitions = compute_topological_order(partitions_dict)

    # codegen
    from pytato import generate_loopy
    prg_per_partition = {pid:
            generate_loopy(
                DictOfNamedArrays(
                    {var_name: pf.var_name_to_result[var_name]
                        for var_name in partition_id_to_output_names[pid]
                     }))
            for pid in partitions}

    res = CodePartitions(toposorted_partitions, prg_per_partition,
            partition_id_to_input_names, partition_id_to_output_names)

    return res


def execute_partitions(parts: CodePartitions, queue: Any) -> Dict[str, Any]:
    """Executes a set of partitions on a :class:`pyopencl.CommandQueue`.

    :param parts: An instance of :class:`CodePartitions` representing the
        partitioned code.
    :param queue: An instance of :class:`pyopencl.CommandQueue` to execute the
        code on.
    :returns: A dictionary of variable names mapped to their values.
    """
    context: Dict[str, Any] = {}
    for pid in parts.toposorted_partitions:
        # find names that are needed
        inputs = {"queue": queue}

        inputs.update({
            k: context[k] for k in parts.partition_id_to_input_names[pid]
            if k in context})

        res = parts.prg_per_partition[pid](**inputs)

        context.update(res[1])

    return context

# }}}

# vim: foldmethod=marker
