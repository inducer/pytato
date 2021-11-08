from __future__ import annotations

__copyright__ = """
Copyright (C) 2020 Matt Wala
Copyright (C) 2020-21 Kaushik Kulkarni
Copyright (C) 2020-21 University of Illinois Board of Trustees
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

from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, FrozenSet, Union, TypeVar, Set, Generic,
                    List, Mapping, Iterable, Tuple, TYPE_CHECKING)

from pytato.array import (
        Array, IndexLambda, Placeholder, MatrixProduct, Stack, Roll,
        AxisPermutation, DataWrapper, SizeParam, DictOfNamedArrays,
        AbstractResultWithNamedArrays, Reshape, Concatenate, NamedArray,
        IndexRemappingBase, Einsum, InputArgumentBase,
        BasicIndex, AdvancedIndexInContiguousAxes, AdvancedIndexInNoncontiguousAxes,
        IndexBase)

from pytato.loopy import LoopyCall, LoopyCallResult
from dataclasses import dataclass
from pytato.tags import ImplStored

if TYPE_CHECKING:
    from pytato.distributed import DistributedSendRefHolder, DistributedRecv

T = TypeVar("T", Array, AbstractResultWithNamedArrays)
CombineT = TypeVar("CombineT")  # used in CombineMapper
CachedMapperT = TypeVar("CachedMapperT")  # used in CachedMapper
IndexOrShapeExpr = TypeVar("IndexOrShapeExpr")
ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]
R = FrozenSet[Array]

__doc__ = """
.. currentmodule:: pytato.transform

.. autoclass:: Mapper
.. autoclass:: CachedMapper
.. autoclass:: CopyMapper
.. autoclass:: CombineMapper
.. autoclass:: DependencyMapper
.. autoclass:: InputGatherer
.. autoclass:: SizeParamGatherer
.. autoclass:: SubsetDependencyMapper
.. autoclass:: WalkMapper
.. autoclass:: CachedWalkMapper
.. autoclass:: TopoSortMapper
.. autoclass:: CachedMapAndCopyMapper
.. autoclass:: EdgeCachedMapper
.. autofunction:: copy_dict_of_named_arrays
.. autofunction:: get_dependencies
.. autofunction:: map_and_copy
.. autofunction:: materialize_with_mpms

Dict representation of DAGs
---------------------------

.. autoclass:: UsersCollector
.. autofunction:: reverse_graph
.. autofunction:: tag_user_nodes
"""


class UnsupportedArrayError(ValueError):
    pass


# {{{ mapper base class

class Mapper:
    """A class that when called with a :class:`pytato.Array` recursively iterates over
    the DAG, calling the *_mapper_method* of each node. Users of this
    class are expected to override the methods of this class or create a
    subclass.

    .. note::

       This class might visit a node multiple times. Use a :class:`CachedMapper`
       if this is not desired.

    .. automethod:: handle_unsupported_array
    .. automethod:: map_foreign
    .. automethod:: rec
    .. automethod:: __call__
    """

    def handle_unsupported_array(self, expr: T, *args: Any, **kwargs: Any) -> Any:
        """Mapper method that is invoked for
        :class:`pytato.Array` subclasses for which a mapper
        method does not exist in this mapper.
        """
        raise UnsupportedArrayError("%s cannot handle expressions of type %s"
                % (type(self).__name__, type(expr)))

    def map_foreign(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        """Mapper method that is invoked for an object of class for which a
        mapper method does not exist in this mapper.
        """
        raise ValueError("%s encountered invalid foreign object: %s"
                % (type(self).__name__, repr(expr)))

    def rec(self, expr: T, *args: Any, **kwargs: Any) -> Any:
        """Call the mapper method of *expr* and return the result."""
        method: Callable[..., Array]

        try:
            method = getattr(self, expr._mapper_method)
        except AttributeError:
            if isinstance(expr, Array):
                return self.handle_unsupported_array(expr, *args, **kwargs)
            else:
                return self.map_foreign(expr, *args, **kwargs)

        return method(expr, *args, **kwargs)

    def __call__(self, expr: T, *args: Any, **kwargs: Any) -> Any:
        """Handle the mapping of *expr*."""
        return self.rec(expr, *args, **kwargs)

# }}}


# {{{ CachedMapper

class CachedMapper(Mapper, Generic[CachedMapperT]):
    """Mapper class that maps each node in the DAG exactly once. This loses some
    information compared to :class:`Mapper` as a node is visited only from
    one of its predecessors.
    """

    def __init__(self) -> None:
        self._cache: Dict[CachedMapperT, Any] = {}

    def cache_key(self, expr: CachedMapperT) -> Any:
        return expr

    # type-ignore-reason: incompatible with super class
    def rec(self, expr: CachedMapperT) -> Any:  # type: ignore[override]
        key = self.cache_key(expr)
        try:
            return self._cache[key]
        except KeyError:
            result = super().rec(expr)  # type: ignore[type-var]
            self._cache[key] = result
            return result

# }}}


# {{{ CopyMapper

class CopyMapper(CachedMapper[ArrayOrNames]):
    """Performs a deep copy of a :class:`pytato.array.Array`.
    The typical use of this mapper is to override individual ``map_`` methods
    in subclasses to permit term rewriting on an expression graph.

    .. note::

       This does not copy the data of a :class:`pytato.array.DataWrapper`.
    """

    def rec_idx_or_size_tuple(self, situp: Tuple[IndexOrShapeExpr, ...]
                              ) -> Tuple[IndexOrShapeExpr, ...]:
        return tuple(self.rec(s) if isinstance(s, Array) else s for s in situp)

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        bindings: Dict[str, Array] = {
                name: self.rec(subexpr)
                for name, subexpr in sorted(expr.bindings.items())}
        return IndexLambda(expr=expr.expr,
                shape=self.rec_idx_or_size_tuple(expr.shape),
                dtype=expr.dtype,
                bindings=bindings,
                tags=expr.tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        assert expr.name is not None
        return Placeholder(name=expr.name,
                shape=self.rec_idx_or_size_tuple(expr.shape),
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
        return type(expr)(
                array=self.rec(expr.array),
                indices=self.rec_idx_or_size_tuple(expr.indices))

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
                shape=self.rec_idx_or_size_tuple(expr.shape),
                tags=expr.tags)

    def map_size_param(self, expr: SizeParam) -> Array:
        assert expr.name is not None
        return SizeParam(name=expr.name, tags=expr.tags)

    def map_einsum(self, expr: Einsum) -> Array:
        return Einsum(expr.access_descriptors,
                      tuple(self.rec(arg) for arg in expr.args),
                      tags=expr.tags)

    def map_named_array(self, expr: NamedArray) -> Array:
        return type(expr)(self.rec(expr._container), expr.name, tags=expr.tags)

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

    def map_reshape(self, expr: Reshape) -> Array:
        return Reshape(self.rec(expr.array),
                       newshape=self.rec_idx_or_size_tuple(expr.newshape),
                       order=expr.order,
                       tags=expr.tags)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> Array:
        from pytato.distributed import DistributedSend, DistributedSendRefHolder
        return DistributedSendRefHolder(
                DistributedSend(
                    data=self.rec(expr.send.data),
                    dest_rank=expr.send.dest_rank,
                    comm_tag=expr.send.comm_tag),
                self.rec(expr.data))

    def map_distributed_recv(self, expr: DistributedRecv) -> Array:
        from pytato.distributed import DistributedRecv
        return DistributedRecv(
               src_rank=expr.src_rank, comm_tag=expr.comm_tag,
               shape=self.rec_idx_or_size_tuple(expr.shape),
               dtype=expr.dtype, tags=expr.tags)

# }}}


# {{{ CombineMapper

class CombineMapper(Mapper, Generic[CombineT]):
    def __init__(self) -> None:
        self.cache: Dict[ArrayOrNames, CombineT] = {}

    def rec_idx_or_size_tuple(self, situp: Tuple[IndexOrShapeExpr, ...]
                              ) -> Tuple[CombineT, ...]:
        return tuple(self.rec(s) for s in situp if isinstance(s, Array))

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
                            *self.rec_idx_or_size_tuple(expr.shape))

    def map_placeholder(self, expr: Placeholder) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    def map_data_wrapper(self, expr: DataWrapper) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

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
                            *self.rec_idx_or_size_tuple(expr.indices))

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
        return self.combine(
                self.rec(expr.array),
                *self.rec_idx_or_size_tuple(expr.newshape))

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

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> CombineT:
        return self.combine(
                self.rec(expr.send.data),
                self.rec(expr.data),
                )

    def map_distributed_recv(self, expr: DistributedRecv) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

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

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> R:
        return self.combine(
                frozenset([expr]), super().map_distributed_send_ref_holder(expr))

    def map_distributed_recv(self, expr: DistributedRecv) -> R:
        return self.combine(frozenset([expr]), super().map_distributed_recv(expr))

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

    def rec_idx_or_size_tuple(self, situp: Tuple[IndexOrShapeExpr, ...]) -> None:
        for comp in situp:
            if isinstance(comp, Array):
                self.rec(comp)

    def map_index_lambda(self, expr: IndexLambda) -> None:
        if not self.visit(expr):
            return

        for _, child in sorted(expr.bindings.items()):
            self.rec(child)

        self.rec_idx_or_size_tuple(expr.shape)

        self.post_visit(expr)

    def map_placeholder(self, expr: Placeholder) -> None:
        if not self.visit(expr):
            return

        self.rec_idx_or_size_tuple(expr.shape)

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

        self.rec_idx_or_size_tuple(expr.indices)

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

    def map_concatenate(self, expr: Concatenate) -> None:
        if not self.visit(expr):
            return

        for child in expr.arrays:
            self.rec(child)

        self.post_visit(expr)

    def map_einsum(self, expr: Einsum) -> None:
        if not self.visit(expr):
            return

        for child in expr.args:
            self.rec(child)

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

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder, *args: Any) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.send.data)
        self.rec(expr.data)

        self.post_visit(expr)

    def map_distributed_recv(self, expr: DistributedRecv, *args: Any) -> None:
        if not self.visit(expr):
            return
        self.rec_idx_or_size_tuple(expr.shape)

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


# {{{ MapAndCopyMapper

class CachedMapAndCopyMapper(CopyMapper):
    """
    Mapper that applies *map_fn* to each node and copies it. Results of
    traversals are memoized i.e. each node is mapped via *map_fn* exactly once.
    """

    def __init__(self, map_fn: Callable[[ArrayOrNames], ArrayOrNames]) -> None:
        super().__init__()
        self.map_fn: Callable[[ArrayOrNames], ArrayOrNames] = map_fn

    # type-ignore-reason:incompatible with Mapper.rec()
    def rec(self, expr: ArrayOrNames) -> ArrayOrNames:  # type: ignore[override]
        if expr in self._cache:
            return self._cache[expr]  # type: ignore[no-any-return]

        result = super().rec(self.map_fn(expr))
        self._cache[expr] = result
        return result  # type: ignore[no-any-return]

    # type-ignore-reason: Mapper.__call__ returns Any
    def __call__(self, expr: ArrayOrNames) -> ArrayOrNames:  # type: ignore[override]
        return self.rec(expr)

# }}}


# {{{ MPMS materializer

@dataclass(frozen=True, eq=True)
class MPMSMaterializerAccumulator:
    """This class serves as the return value of :class:`MPMSMaterializer`. It
    contains the set of materialized predecessors and the rewritten expression
    (i.e. the expression with tags for materialization applied).
    """
    materialized_predecessors: FrozenSet[Array]
    expr: Array


def _materialize_if_mpms(expr: Array,
                         nsuccessors: int,
                         predecessors: Iterable[MPMSMaterializerAccumulator]
                         ) -> MPMSMaterializerAccumulator:
    """
    Returns an instance of :class:`MPMSMaterializerAccumulator`, that
    materializes *expr* if it has more than 1 successors and more than 1
    materialized predecessors.
    """
    from functools import reduce

    materialized_predecessors: FrozenSet[Array] = reduce(
                                                    frozenset.union,
                                                    (pred.materialized_predecessors
                                                     for pred in predecessors),
                                                    frozenset())
    if nsuccessors > 1 and len(materialized_predecessors) > 1:
        new_expr = expr.tagged(ImplStored())
        return MPMSMaterializerAccumulator(frozenset([new_expr]), new_expr)
    else:
        return MPMSMaterializerAccumulator(materialized_predecessors, expr)


class MPMSMaterializer(Mapper):
    """See :func:`materialize_with_mpms` for an explanation."""
    def __init__(self, nsuccessors: Mapping[Array, int]):
        super().__init__()
        self.nsuccessors = nsuccessors
        self.cache: Dict[ArrayOrNames, MPMSMaterializerAccumulator] = {}

    # type-ignore reason: return type not compatible with Mapper.rec's type
    def rec(self, expr: ArrayOrNames) -> MPMSMaterializerAccumulator:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        # type-ignore reason: type not compatible with super.rec() type
        result: MPMSMaterializerAccumulator = super().rec(expr)  # type: ignore
        self.cache[expr] = result
        return result

    def _map_input_base(self, expr: InputArgumentBase
                        ) -> MPMSMaterializerAccumulator:
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_named_array(self, expr: NamedArray) -> MPMSMaterializerAccumulator:
        if not isinstance(expr, LoopyCallResult):
            raise NotImplementedError("only LoopyCallResult named array"
                                      " supported for now.")

        # loopy call result is always materialized
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_index_lambda(self, expr: IndexLambda) -> MPMSMaterializerAccumulator:
        children_rec = {bnd_name: self.rec(bnd)
                        for bnd_name, bnd in expr.bindings.items()}

        new_expr = IndexLambda(expr.expr,
                               expr.shape,
                               expr.dtype,
                               {bnd_name: bnd.expr
                                for bnd_name, bnd in children_rec.items()},
                               tags=expr.tags)
        return _materialize_if_mpms(new_expr, self.nsuccessors[expr],
                                    children_rec.values())

    def map_matrix_product(self, expr: MatrixProduct) -> MPMSMaterializerAccumulator:
        x1_rec, x2_rec = self.rec(expr.x1), self.rec(expr.x2)
        new_expr = MatrixProduct(x1_rec.expr, x2_rec.expr, expr.tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (x1_rec, x2_rec))

    def map_stack(self, expr: Stack) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_expr = Stack(tuple(ary.expr for ary in rec_arrays), expr.axis, expr.tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_concatenate(self, expr: Concatenate) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_expr = Concatenate(tuple(ary.expr for ary in rec_arrays),
                               expr.axis,
                               expr.tags)
        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_roll(self, expr: Roll) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = Roll(rec_array.expr, expr.shift, expr.axis, expr.tags)
        return _materialize_if_mpms(new_expr, self.nsuccessors[expr],
                                    (rec_array,))

    def map_axis_permutation(self, expr: AxisPermutation
                             ) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = AxisPermutation(rec_array.expr, expr.axes, expr.tags)
        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array,))

    def _map_index_base(self, expr: IndexBase) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        rec_indices = {i: self.rec(idx)
                       for i, idx in enumerate(expr.indices)
                       if isinstance(idx, Array)}

        new_expr = type(expr)(rec_array.expr,
                              tuple(rec_indices[i].expr
                                    if i in rec_indices
                                    else expr.indices[i]
                                    for i in range(
                                        len(expr.indices))),
                              expr.tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array,) + tuple(rec_indices.values())
                                    )

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_reshape(self, expr: Reshape) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = Reshape(rec_array.expr, expr.newshape, expr.order, expr.tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array,))

    def map_einsum(self, expr: Einsum) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.args]
        new_expr = Einsum(expr.access_descriptors,
                          tuple(ary.expr for ary in rec_arrays),
                          expr.tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays
                                 ) -> MPMSMaterializerAccumulator:
        raise NotImplementedError

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


def map_and_copy(expr: T,
                 map_fn: Callable[[ArrayOrNames], ArrayOrNames]
                 ) -> ArrayOrNames:
    """
    Returns a copy of *expr* with every array expression reachable from *expr*
    mapped via *map_fn*.

    .. note::

        Uses :class:`CachedMapAndCopyMapper` under the hood and because of its
        caching nature each node is mapped exactly once.
    """
    return CachedMapAndCopyMapper(map_fn)(expr)


def materialize_with_mpms(expr: DictOfNamedArrays) -> DictOfNamedArrays:
    r"""
    Materialize nodes in *expr* with MPMS materialization strategy.
    MPMS stands for Multiple-Predecessors, Multiple-Successors.

    .. note::

        - MPMS materialization strategy is a greedy materialization algorithm in
          which any node with more than 1 materialized predecessors and more than
          1 successors is materialized.
        - Materializing here corresponds to tagging a node with
          :class:`~pytato.tags.ImplStored`.
        - Does not attempt to materialize sub-expressions in
          :attr:`pytato.Array.shape`.

    .. warning::

        This is a greedy materialization algorithm and thereby this algorithm
        might be too eager to materialize. Consider the graph below:

        ::

                           I1          I2
                            \         /
                             \       /
                              \     /
                               🡦   🡧
                                 T
                                / \
                               /   \
                              /     \
                             🡧       🡦
                            O1        O2

        where, 'I1', 'I2' correspond to instances of
        :class:`pytato.array.InputArgumentBase`, and, 'O1' and 'O2' are the outputs
        required to be evaluated in the computation graph. MPMS materialization
        algorithm will materialize the intermediate node 'T' as it has 2
        predecessors and 2 successors. However, the total number of memory
        accesses after applying MPMS goes up as shown by the table below.

        ======  ========  =======
        ..        Before    After
        ======  ========  =======
        Reads          4        4
        Writes         2        3
        Total          6        7
        ======  ========  =======

    """
    from pytato.analysis import get_nusers
    materializer = MPMSMaterializer(get_nusers(expr))
    new_data = {}
    for name, ary in expr.items():
        new_data[name] = materializer(ary.expr).expr

    return DictOfNamedArrays(new_data)

# }}}


# {{{ UsersCollector

class UsersCollector(CachedMapper[ArrayOrNames]):
    """
    Maps a graph to a dictionary representation mapping a node to its users,
    i.e. all the nodes using its value.

    .. attribute:: node_to_users

       Mapping of each node in the graph to its users.

    .. automethod:: __init__
    """

    def __init__(self) -> None:
        super().__init__()
        self.node_to_users: Dict[ArrayOrNames, Set[ArrayOrNames]] = {}

    def __call__(self, expr: ArrayOrNames, *args: Any, **kwargs: Any) -> Any:
        # Root node has no predecessor
        self.node_to_users[expr] = set()
        return self.rec(expr, *args)

    def rec_idx_or_size_tuple(
            self, expr: Array, situp: Tuple[IndexOrShapeExpr, ...]
            ) -> None:
        for dim in situp:
            if isinstance(dim, Array):
                self.node_to_users.setdefault(dim, set()).add(expr)
                self.rec(dim)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        for child in expr._data.values():
            self.node_to_users.setdefault(child, set()).add(expr)
            self.rec(child)

    def map_named_array(self, expr: NamedArray) -> None:
        self.node_to_users.setdefault(expr._container, set()).add(expr)
        self.rec(expr._container)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        for _, child in sorted(expr.bindings.items()):
            if isinstance(child, Array):
                self.node_to_users.setdefault(child, set()).add(expr)
                self.rec(child)

    def map_einsum(self, expr: Einsum) -> None:
        for arg in expr.args:
            self.node_to_users.setdefault(arg, set()).add(expr)
            self.rec(arg)

    def map_reshape(self, expr: Reshape) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_placeholder(self, expr: Placeholder) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

    def map_matrix_product(self, expr: MatrixProduct) -> None:
        for child in (expr.x1, expr.x2):
            self.node_to_users.setdefault(child, set()).add(expr)
            self.rec(child)

    def map_concatenate(self, expr: Concatenate) -> None:
        for ary in expr.arrays:
            self.node_to_users.setdefault(ary, set()).add(expr)
            self.rec(ary)

    def map_stack(self, expr: Stack) -> None:
        for ary in expr.arrays:
            self.node_to_users.setdefault(ary, set()).add(expr)
            self.rec(ary)

    def map_roll(self, expr: Roll) -> None:
        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_size_param(self, expr: SizeParam) -> None:
        # no child nodes, nothing to do
        pass

    def map_axis_permutation(self, expr: AxisPermutation) -> None:
        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_data_wrapper(self, expr: DataWrapper) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

    def map_index_lambda(self, expr: IndexLambda) -> None:
        for child in expr.bindings.values():
            self.node_to_users.setdefault(child, set()).add(expr)
            self.rec(child)

        self.rec_idx_or_size_tuple(expr, expr.shape)

    def _map_index_base(self, expr: IndexBase) -> None:
        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

        for idx in expr.indices:
            if isinstance(idx, Array):
                self.node_to_users.setdefault(idx, set()).add(expr)
                self.rec(idx)

    def map_basic_index(self, expr: BasicIndex) -> None:
        self._map_index_base(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> None:
        self._map_index_base(expr)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> None:
        self._map_index_base(expr)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder, *args: Any) -> None:
        self.node_to_users.setdefault(expr.data, set()).add(expr)
        self.rec(expr.data)
        self.node_to_users.setdefault(expr.send.data, set()).add(expr)
        self.rec(expr.send.data)

    def map_distributed_recv(self, expr: DistributedRecv, *args: Any) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

# }}}


# {{{ operations on graphs in dict form

def reverse_graph(graph: Dict[ArrayOrNames, Set[ArrayOrNames]]) \
        -> Dict[ArrayOrNames, Set[ArrayOrNames]]:
    """Reverses a graph.

    :param graph: A :class:`dict` representation of a directed graph, mapping each
        node to other nodes to which it is connected by edges. A possible
        use case for this function is the graph in
        :attr:`UsersCollector.node_to_users`.
    :returns: A :class:`dict` representing *graph* with edges reversed.
    """
    result: Dict[ArrayOrNames, Set[ArrayOrNames]] = {}

    for node_key, edges in graph.items():
        # Make sure every node is in the result even if it has no users
        if node_key not in result:
            result[node_key] = set()

        for other_node_key in edges:
            result.setdefault(other_node_key, set()).add(node_key)

    return result


def tag_user_nodes(graph: Dict[ArrayOrNames, Set[ArrayOrNames]], tag: Any,
        starting_point: ArrayOrNames,
        node_to_tags: Dict[ArrayOrNames, Set[ArrayOrNames]]) \
        -> None:
    """Tags all nodes reachable from *starting_point* with *tag*.

    :param graph: A :class:`dict` representation of a directed graph, mapping each
        node to other nodes to which it is connected by edges. A possible
        use case for this function is the graph in
        :attr:`UsersCollector.node_to_users`.
    :param tag: The value to tag the nodes with.
    :param starting_point: A starting point in *graph*.
    :param node_to_tags: The resulting mapping of nodes to tags.
    """
    node_to_tags.setdefault(starting_point, set()).add(tag)
    if starting_point in graph:
        for other_node_key in graph[starting_point]:
            tag_user_nodes(graph, tag=tag, starting_point=other_node_key,
                                          node_to_tags=node_to_tags)

# }}}


# {{{ EdgeCachedMapper

class EdgeCachedMapper(CachedMapper[ArrayOrNames], ABC):
    """
    Mapper class to execute a rewriting method (:meth:`handle_edge`) on each
    edge in the graph.

    .. automethod:: handle_edge
    """

    @abstractmethod
    def handle_edge(self, expr: ArrayOrNames, child: ArrayOrNames) -> Any:
        pass

    def rec_idx_or_size_tuple(self,
            expr: Array,
            situp: Tuple[IndexOrShapeExpr, ...],
            *args: Any) -> Tuple[IndexOrShapeExpr, ...]:
        return tuple([
            self.handle_edge(expr, dim, *args) if isinstance(dim, Array) else dim
            for dim in situp])

    # {{{ map_xxx methods

    def map_named_array(self, expr: NamedArray, *args: Any) -> NamedArray:
        return NamedArray(
            container=self.handle_edge(expr, expr._container, *args),
            name=expr.name,
            tags=expr.tags)

    def map_index_lambda(self, expr: IndexLambda, *args: Any) -> IndexLambda:
        return IndexLambda(expr=expr.expr,
                shape=self.rec_idx_or_size_tuple(expr, expr.shape),
                dtype=expr.dtype,
                bindings={name: self.handle_edge(expr, child)
                          for name, child in expr.bindings.items()},
                tags=expr.tags)

    def map_einsum(self, expr: Einsum, *args: Any) -> Einsum:
        return Einsum(
                     access_descriptors=expr.access_descriptors,
                     args=tuple(self.handle_edge(expr, arg, *args)
                                for arg in expr.args),
                     tags=expr.tags)

    def map_matrix_product(self, expr: MatrixProduct, *args: Any) -> MatrixProduct:
        return MatrixProduct(x1=self.handle_edge(expr, expr.x1, *args),
                             x2=self.handle_edge(expr, expr.x2, *args),
                             tags=expr.tags)

    def map_stack(self, expr: Stack, *args: Any) -> Stack:
        return Stack(
                     arrays=tuple(self.handle_edge(expr, ary, *args)
                                  for ary in expr.arrays),
                     axis=expr.axis,
                     tags=expr.tags)

    def map_concatenate(self, expr: Concatenate, *args: Any) -> Concatenate:
        return Concatenate(
                     arrays=tuple(self.handle_edge(expr, ary, *args)
                                  for ary in expr.arrays),
                     axis=expr.axis,
                     tags=expr.tags)

    def map_roll(self, expr: Roll, *args: Any) -> Roll:
        return Roll(array=self.handle_edge(expr, expr.array, *args),
                shift=expr.shift,
                axis=expr.axis,
                tags=expr.tags)

    def map_axis_permutation(self, expr: AxisPermutation, *args: Any) \
            -> AxisPermutation:
        return AxisPermutation(
                array=self.handle_edge(expr, expr.array, *args),
                axes=expr.axes,
                tags=expr.tags)

    def map_reshape(self, expr: Reshape, *args: Any) -> Reshape:
        return Reshape(
            array=self.handle_edge(expr, expr.array, *args),
            newshape=self.rec_idx_or_size_tuple(expr, expr.newshape, *args),
            order=expr.order,
            tags=expr.tags)

    def map_basic_index(self, expr: BasicIndex, *args: Any) -> BasicIndex:
        return BasicIndex(
                array=self.handle_edge(expr, expr.array, *args),
                indices=tuple(self.handle_edge(expr, idx, *args)
                                if isinstance(idx, Array) else idx
                                for idx in expr.indices))

    def map_contiguous_advanced_index(self,
            expr: AdvancedIndexInContiguousAxes, *args: Any) \
                    -> AdvancedIndexInContiguousAxes:
        return AdvancedIndexInContiguousAxes(
                array=self.handle_edge(expr, expr.array, *args),
                indices=tuple(self.handle_edge(expr, idx, *args)
                                if isinstance(idx, Array) else idx
                                for idx in expr.indices))

    def map_non_contiguous_advanced_index(self,
            expr: AdvancedIndexInNoncontiguousAxes, *args: Any) \
            -> AdvancedIndexInNoncontiguousAxes:
        return AdvancedIndexInNoncontiguousAxes(
                array=self.handle_edge(expr, expr.array, *args),
                indices=tuple(self.handle_edge(expr, idx, *args)
                                if isinstance(idx, Array) else idx
                                for idx in expr.indices))

    def map_data_wrapper(self, expr: DataWrapper, *args: Any) -> DataWrapper:
        return DataWrapper(
                name=expr.name,
                data=expr.data,
                shape=self.rec_idx_or_size_tuple(expr, expr.shape, *args),
                tags=expr.tags)

    def map_placeholder(self, expr: Placeholder, *args: Any) -> Placeholder:
        assert expr.name

        return Placeholder(name=expr.name,
                shape=self.rec_idx_or_size_tuple(expr, expr.shape, *args),
                dtype=expr.dtype,
                tags=expr.tags)

    def map_size_param(self, expr: SizeParam, *args: Any) -> SizeParam:
        assert expr.name
        return SizeParam(name=expr.name, tags=expr.tags)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder, *args: Any) -> Any:
        from pytato.distributed import DistributedSend, DistributedSendRefHolder
        return DistributedSendRefHolder(
                send=DistributedSend(
                    data=self.handle_edge(expr, expr.send.data),
                    dest_rank=expr.send.dest_rank,
                    comm_tag=expr.send.comm_tag),
                data=self.handle_edge(expr, expr.data),
                tags=expr.tags)

    def map_distributed_recv(self, expr: DistributedRecv, *args: Any) \
            -> Any:
        from pytato.distributed import DistributedRecv
        return DistributedRecv(
            src_rank=expr.src_rank, comm_tag=expr.comm_tag,
            shape=self.rec_idx_or_size_tuple(expr, expr.shape, *args),
            dtype=expr.dtype, tags=expr.tags)

    # }}}

# }}}

# vim: foldmethod=marker
