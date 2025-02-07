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
import dataclasses
import logging
from collections.abc import Hashable
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Never,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from immutabledict import immutabledict
from typing_extensions import Self

from pymbolic.mapper.optimize import optimize_mapper
from pytools import memoize_method

from pytato.array import (
    AbstractResultWithNamedArrays,
    AdvancedIndexInContiguousAxes,
    AdvancedIndexInNoncontiguousAxes,
    Array,
    AxisPermutation,
    BasicIndex,
    Concatenate,
    DataInterface,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexLambda,
    IndexRemappingBase,
    InputArgumentBase,
    NamedArray,
    Placeholder,
    Reshape,
    Roll,
    SizeParam,
    Stack,
    _SuppliedAxesAndTagsMixin,
)
from pytato.distributed.nodes import (
    DistributedRecv,
    DistributedSend,
    DistributedSendRefHolder,
)
from pytato.function import Call, FunctionDefinition, NamedCallResult
from pytato.loopy import LoopyCall, LoopyCallResult
from pytato.tags import ImplStored


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


ArrayOrNames: TypeAlias = Array | AbstractResultWithNamedArrays
MappedT = TypeVar("MappedT",
                  Array, AbstractResultWithNamedArrays, ArrayOrNames)
IndexOrShapeExpr = TypeVar("IndexOrShapeExpr")
R = frozenset[Array]

__doc__ = """
.. autoclass:: Mapper
.. autoclass:: CacheInputs
.. autoclass:: CachedMapperCache
.. autoclass:: CachedMapper
.. autoclass:: TransformMapperCache
.. autoclass:: TransformMapper
.. autoclass:: TransformMapperWithExtraArgs
.. autoclass:: CopyMapper
.. autoclass:: CopyMapperWithExtraArgs
.. autoclass:: CombineMapper
.. autoclass:: DependencyMapper
.. autoclass:: InputGatherer
.. autoclass:: SizeParamGatherer
.. autoclass:: SubsetDependencyMapper
.. autoclass:: WalkMapper
.. autoclass:: CachedWalkMapper
.. autoclass:: TopoSortMapper
.. autoclass:: CachedMapAndCopyMapper
.. autofunction:: copy_dict_of_named_arrays
.. autofunction:: get_dependencies
.. autofunction:: map_and_copy
.. autofunction:: materialize_with_mpms
.. autofunction:: deduplicate_data_wrappers
.. automodule:: pytato.transform.lower_to_index_lambda
.. automodule:: pytato.transform.remove_broadcasts_einsum
.. automodule:: pytato.transform.einsum_distributive_law

.. currentmodule:: pytato.transform

Dict representation of DAGs
---------------------------

.. autoclass:: UsersCollector
.. autofunction:: rec_get_user_nodes


Transforming call sites
-----------------------

.. automodule:: pytato.transform.calls

.. currentmodule:: pytato.transform

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: ArrayOrNames

.. class:: MappedT

    A type variable representing the input type of a :class:`Mapper`.

.. class:: ResultT

    A type variable representing the result type of a :class:`Mapper` when mapping
    a :class:`pytato.Array` or :class:`pytato.AbstractResultWithNamedArrays`.

.. class:: FunctionResultT

    A type variable representing the result type of a :class:`Mapper` when mapping
    a :class:`pytato.function.FunctionDefinition`.

.. class:: CacheExprT

    A type variable representing an input from which to compute a cache key in order
    to cache a result.

.. class:: CacheKeyT

    A type variable representing a key computed from an input expression.

.. class:: CacheResultT

    A type variable representing a result to be cached.

.. class:: Scalar

    See :data:`pymbolic.Scalar`.

.. class:: P

    A :class:`typing.ParamSpec` used to annotate `*args` and `**kwargs`.

"""

transform_logger = logging.getLogger(__file__)


class UnsupportedArrayError(ValueError):
    pass


class ForeignObjectError(ValueError):
    pass


# {{{ mapper base class

ResultT = TypeVar("ResultT")
FunctionResultT = TypeVar("FunctionResultT")
P = ParamSpec("P")


def _verify_is_array(expr: ArrayOrNames) -> Array:
    assert isinstance(expr, Array)
    return expr


class Mapper(Generic[ResultT, FunctionResultT, P]):
    """A class that when called with a :class:`pytato.Array` recursively
    iterates over the DAG, calling the *_mapper_method* of each node. Users of
    this class are expected to override the methods of this class or create a
    subclass.

    .. note::

       This class might visit a node multiple times. Use a :class:`CachedMapper`
       if this is not desired.

    .. automethod:: handle_unsupported_array
    .. automethod:: rec
    .. automethod:: __call__
    """

    def handle_unsupported_array(self, expr: Array,
                                 *args: P.args, **kwargs: P.kwargs) -> ResultT:
        """Mapper method that is invoked for
        :class:`pytato.Array` subclasses for which a mapper
        method does not exist in this mapper.
        """
        raise UnsupportedArrayError(
                f"{type(self).__name__} cannot handle expressions of type {type(expr)}")

    def rec(self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs) -> ResultT:
        """Call the mapper method of *expr* and return the result."""
        method: Callable[..., ResultT] | None

        try:
            method = getattr(self, expr._mapper_method)
        except AttributeError:
            if isinstance(expr, Array):
                for cls in type(expr).__mro__[1:]:
                    method_name = getattr(cls, "_mapper_method", None)
                    if method_name:
                        method = getattr(self, method_name, None)
                        if method:
                            break
                else:
                    return self.handle_unsupported_array(expr, *args, **kwargs)
            else:
                raise ForeignObjectError(
                    f"{type(self).__name__} encountered invalid foreign "
                    f"object: {expr!r}") from None

        assert method is not None
        return cast("ResultT", method(expr, *args, **kwargs))

    def rec_function_definition(
            self, expr: FunctionDefinition, *args: P.args, **kwargs: P.kwargs
            ) -> FunctionResultT:
        """Call the mapper method of *expr* and return the result."""
        method: Callable[..., FunctionResultT] | None

        try:
            method = self.map_function_definition  # type: ignore[attr-defined]
        except AttributeError:
            raise ValueError(
                f"{type(self).__name__} lacks a mapper method for functions.") from None

        assert method is not None
        return method(expr, *args, **kwargs)

    @overload
    def __call__(
            self,
            expr: ArrayOrNames,
            *args: P.args,
            **kwargs: P.kwargs) -> ResultT:
        ...

    @overload
    def __call__(
            self,
            expr: FunctionDefinition,
            *args: P.args,
            **kwargs: P.kwargs) -> FunctionResultT:
        ...

    def __call__(
            self,
            expr: ArrayOrNames | FunctionDefinition,
            *args: P.args,
            **kwargs: P.kwargs) -> ResultT | FunctionResultT:
        """Handle the mapping of *expr*."""
        if isinstance(expr, ArrayOrNames):
            return self.rec(expr, *args, **kwargs)
        elif isinstance(expr, FunctionDefinition):
            return self.rec_function_definition(expr, *args, **kwargs)
        else:
            raise ForeignObjectError(
                f"{type(self).__name__} encountered invalid foreign "
                f"object: {expr!r}") from None

# }}}


# {{{ CachedMapper

CacheExprT = TypeVar("CacheExprT")
CacheResultT = TypeVar("CacheResultT")
CacheKeyT: TypeAlias = Hashable


class CacheInputs(Generic[CacheExprT, P]):
    """
    Data structure for inputs to :class:`CachedMapperCache`.

    .. attribute:: expr

        The input expression being mapped.

    .. attribute:: key

        The cache key corresponding to *expr* and any additional inputs that were
        passed.

    """
    def __init__(
            self,
            expr: CacheExprT,
            key_func: Callable[..., CacheKeyT],
            *args: P.args,
            **kwargs: P.kwargs):
        self.expr: CacheExprT = expr
        self._args: tuple[Any, ...] = args
        self._kwargs: dict[str, Any] = kwargs
        self._key_func = key_func

    @memoize_method
    def _get_key(self) -> CacheKeyT:
        return self._key_func(self.expr, *self._args, **self._kwargs)

    @property
    def key(self) -> CacheKeyT:
        return self._get_key()


class CachedMapperCache(Generic[CacheExprT, CacheResultT, P]):
    """
    Cache for mappers.

    .. automethod:: __init__

        Compute the key for an input expression.

    .. automethod:: add
    .. automethod:: retrieve
    .. automethod:: clear
    """
    def __init__(self) -> None:
        """Initialize the cache."""
        self._expr_key_to_result: dict[CacheKeyT, CacheResultT] = {}

    def add(
            self,
            inputs: CacheInputs[CacheExprT, P],
            result: CacheResultT) -> CacheResultT:
        """Cache a mapping result."""
        key = inputs.key

        assert key not in self._expr_key_to_result, \
            f"Cache entry is already present for key '{key}'."

        self._expr_key_to_result[key] = result

        return result

    def retrieve(self, inputs: CacheInputs[CacheExprT, P]) -> CacheResultT:
        """Retrieve the cached mapping result."""
        key = inputs.key
        return self._expr_key_to_result[key]

    def clear(self) -> None:
        """Reset the cache."""
        self._expr_key_to_result = {}


class CachedMapper(Mapper[ResultT, FunctionResultT, P]):
    """Mapper class that maps each node in the DAG exactly once. This loses some
    information compared to :class:`Mapper` as a node is visited only from
    one of its predecessors.

    .. automethod:: get_cache_key
    .. automethod:: get_function_definition_cache_key
    .. automethod:: clone_for_callee
    """
    def __init__(
            self,
            _cache:
                CachedMapperCache[ArrayOrNames, ResultT, P] | None = None,
            _function_cache:
                CachedMapperCache[FunctionDefinition, FunctionResultT, P] | None = None
            ) -> None:
        super().__init__()

        self._cache: CachedMapperCache[ArrayOrNames, ResultT, P] = (
            _cache if _cache is not None
            else CachedMapperCache())

        self._function_cache: CachedMapperCache[
                FunctionDefinition, FunctionResultT, P] = (
            _function_cache if _function_cache is not None
            else CachedMapperCache())

    def get_cache_key(
                self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs
            ) -> CacheKeyT:
        if args or kwargs:
            raise NotImplementedError(
                "Derived classes must override get_cache_key if using extra inputs.")
        return expr

    def get_function_definition_cache_key(
                self, expr: FunctionDefinition, *args: P.args, **kwargs: P.kwargs
            ) -> CacheKeyT:
        if args or kwargs:
            raise NotImplementedError(
                "Derived classes must override get_function_definition_cache_key if "
                "using extra inputs.")
        return expr

    def _make_cache_inputs(
            self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs
            ) -> CacheInputs[ArrayOrNames, P]:
        return CacheInputs(expr, self.get_cache_key, *args, **kwargs)

    def _make_function_definition_cache_inputs(
            self, expr: FunctionDefinition, *args: P.args, **kwargs: P.kwargs
            ) -> CacheInputs[FunctionDefinition, P]:
        return CacheInputs(
            expr, self.get_function_definition_cache_key, *args, **kwargs)

    def rec(self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs) -> ResultT:
        inputs = self._make_cache_inputs(expr, *args, **kwargs)
        try:
            return self._cache.retrieve(inputs)
        except KeyError:
            # Intentionally going to Mapper instead of super() to avoid
            # double caching when subclasses of CachedMapper override rec,
            # see https://github.com/inducer/pytato/pull/585
            return self._cache.add(inputs, Mapper.rec(self, expr, *args, **kwargs))

    def rec_function_definition(
                self, expr: FunctionDefinition, *args: P.args, **kwargs: P.kwargs
            ) -> FunctionResultT:
        inputs = self._make_function_definition_cache_inputs(expr, *args, **kwargs)
        try:
            return self._function_cache.retrieve(inputs)
        except KeyError:
            return self._function_cache.add(
                # Intentionally going to Mapper instead of super() to avoid
                # double caching when subclasses of CachedMapper override rec,
                # see https://github.com/inducer/pytato/pull/585
                inputs, Mapper.rec_function_definition(self, expr, *args, **kwargs))

    def clone_for_callee(
            self, function: FunctionDefinition) -> Self:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        # Functions are cached globally, but arrays aren't
        return type(self)(_function_cache=self._function_cache)

# }}}


# {{{ TransformMapper

class TransformMapperCache(CachedMapperCache[CacheExprT, CacheExprT, P]):
    pass


class TransformMapper(CachedMapper[ArrayOrNames, FunctionDefinition, []]):
    """Base class for mappers that transform :class:`pytato.array.Array`\\ s into
    other :class:`pytato.array.Array`\\ s.

    Enables certain operations that can only be done if the mapping results are also
    arrays (e.g., computing a cache key from them). Does not implement default
    mapper methods; for that, see :class:`CopyMapper`.
    """
    def __init__(
            self,
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        super().__init__(_cache=_cache, _function_cache=_function_cache)

# }}}


# {{{ TransformMapperWithExtraArgs

class TransformMapperWithExtraArgs(
            CachedMapper[ArrayOrNames, FunctionDefinition, P]
        ):
    """
    Similar to :class:`TransformMapper`, but each mapper method takes extra
    ``*args``, ``**kwargs`` that are propagated along a path by default.

    The logic in :class:`TransformMapper` purposely does not take the extra
    arguments to keep the cost of its each call frame low.
    """
    def __init__(
            self,
            _cache: TransformMapperCache[ArrayOrNames, P] | None = None,
            _function_cache:
                TransformMapperCache[FunctionDefinition, P] | None = None
            ) -> None:
        super().__init__(_cache=_cache, _function_cache=_function_cache)

# }}}


# {{{ CopyMapper

class CopyMapper(TransformMapper):
    """Performs a deep copy of a :class:`pytato.array.Array`.
    The typical use of this mapper is to override individual ``map_`` methods
    in subclasses to permit term rewriting on an expression graph.

    .. note::

       This does not copy the data of a :class:`pytato.array.DataWrapper`.
    """
    def rec_idx_or_size_tuple(self, situp: tuple[IndexOrShapeExpr, ...]
                              ) -> tuple[IndexOrShapeExpr, ...]:
        # type-ignore-reason: apparently mypy cannot substitute typevars
        # here.
        return tuple(self.rec(s) if isinstance(s, Array) else s  # type: ignore[misc]
                     for s in situp)

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        bindings: Mapping[str, Array] = immutabledict({
                name: self.rec(subexpr)
                for name, subexpr in sorted(expr.bindings.items())})
        return IndexLambda(expr=expr.expr,
                shape=self.rec_idx_or_size_tuple(expr.shape),
                dtype=expr.dtype,
                bindings=bindings,
                axes=expr.axes,
                var_to_reduction_descr=expr.var_to_reduction_descr,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        assert expr.name is not None
        return Placeholder(name=expr.name,
                shape=self.rec_idx_or_size_tuple(expr.shape),
                dtype=expr.dtype,
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_stack(self, expr: Stack) -> Array:
        arrays = tuple(_verify_is_array(self.rec(arr)) for arr in expr.arrays)
        return Stack(arrays=arrays, axis=expr.axis, axes=expr.axes, tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate) -> Array:
        arrays = tuple(_verify_is_array(self.rec(arr)) for arr in expr.arrays)
        return Concatenate(arrays=arrays, axis=expr.axis,
                           axes=expr.axes, tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_roll(self, expr: Roll) -> Array:
        return Roll(array=_verify_is_array(self.rec(expr.array)),
                shift=expr.shift,
                axis=expr.axis,
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        return AxisPermutation(array=_verify_is_array(self.rec(expr.array)),
                axis_permutation=expr.axis_permutation,
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase) -> Array:
        return type(expr)(_verify_is_array(self.rec(expr.array)),
                          indices=self.rec_idx_or_size_tuple(expr.indices),
                          axes=expr.axes,
                          tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

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
        return DataWrapper(
                data=expr.data,
                shape=self.rec_idx_or_size_tuple(expr.shape),
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_size_param(self, expr: SizeParam) -> Array:
        assert expr.name is not None
        return SizeParam(
            name=expr.name,
            axes=expr.axes,
            tags=expr.tags,
            non_equality_tags=expr.non_equality_tags)

    def map_einsum(self, expr: Einsum) -> Array:
        return Einsum(expr.access_descriptors,
                      tuple(_verify_is_array(self.rec(arg)) for arg in expr.args),
                      axes=expr.axes,
                      redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                      tags=expr.tags,
                      non_equality_tags=expr.non_equality_tags)

    def map_named_array(self, expr: NamedArray) -> Array:
        container = self.rec(expr._container)
        assert isinstance(container, AbstractResultWithNamedArrays)
        return type(expr)(container,
                          expr.name,
                          axes=expr.axes,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_dict_of_named_arrays(self,
            expr: DictOfNamedArrays) -> DictOfNamedArrays:
        return DictOfNamedArrays({key: _verify_is_array(self.rec(val.expr))
                                  for key, val in expr.items()},
                                 tags=expr.tags
                                 )

    def map_loopy_call(self, expr: LoopyCall) -> LoopyCall:
        bindings: Mapping[Any, Any] = immutabledict(
                    {name: (self.rec(subexpr) if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})

        return LoopyCall(translation_unit=expr.translation_unit,
                         bindings=bindings,
                         entrypoint=expr.entrypoint,
                         tags=expr.tags,
                         )

    def map_loopy_call_result(self, expr: LoopyCallResult) -> Array:
        rec_container = self.rec(expr._container)
        assert isinstance(rec_container, LoopyCall)
        return LoopyCallResult(
                _container=rec_container,
                name=expr.name,
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape) -> Array:
        return Reshape(_verify_is_array(self.rec(expr.array)),
                       newshape=self.rec_idx_or_size_tuple(expr.newshape),
                       order=expr.order,
                       axes=expr.axes,
                       tags=expr.tags,
                       non_equality_tags=expr.non_equality_tags)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> Array:
        return DistributedSendRefHolder(
                send=DistributedSend(
                    data=_verify_is_array(self.rec(expr.send.data)),
                    dest_rank=expr.send.dest_rank,
                    comm_tag=expr.send.comm_tag),
                passthrough_data=_verify_is_array(self.rec(expr.passthrough_data)),
                )

    def map_distributed_recv(self, expr: DistributedRecv) -> Array:
        return DistributedRecv(
               src_rank=expr.src_rank, comm_tag=expr.comm_tag,
               shape=self.rec_idx_or_size_tuple(expr.shape),
               dtype=expr.dtype, tags=expr.tags, axes=expr.axes,
               non_equality_tags=expr.non_equality_tags)

    def map_function_definition(self,
                                expr: FunctionDefinition) -> FunctionDefinition:
        # spawn a new mapper to avoid unsound cache hits, since the namespace of the
        # function's body is different from that of the caller.
        new_mapper = self.clone_for_callee(expr)
        new_returns = {name: new_mapper(ret)
                       for name, ret in expr.returns.items()}
        return dataclasses.replace(expr, returns=immutabledict(new_returns))

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        return Call(self.rec_function_definition(expr.function),
                    immutabledict({name: self.rec(bnd)
                         for name, bnd in expr.bindings.items()}),
                    tags=expr.tags,
                    )

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        call = self.rec(expr._container)
        assert isinstance(call, Call)
        return call[expr.name]


class CopyMapperWithExtraArgs(TransformMapperWithExtraArgs[P]):
    """
    Similar to :class:`CopyMapper`, but each mapper method takes extra
    ``*args``, ``**kwargs`` that are propagated along a path by default.

    The logic in :class:`CopyMapper` purposely does not take the extra
    arguments to keep the cost of its each call frame low.
    """
    def rec_idx_or_size_tuple(self, situp: tuple[IndexOrShapeExpr, ...],
                              *args: P.args, **kwargs: P.kwargs
                              ) -> tuple[IndexOrShapeExpr, ...]:
        # type-ignore-reason: apparently mypy cannot substitute typevars
        # here.
        return tuple(
            self.rec(s, *args, **kwargs)  # type: ignore[misc]
            if isinstance(s, Array)
            else s
            for s in situp)

    def map_index_lambda(self, expr: IndexLambda,
                         *args: P.args, **kwargs: P.kwargs) -> Array:
        bindings: Mapping[str, Array] = immutabledict({
                name: self.rec(subexpr, *args, **kwargs)
                for name, subexpr in sorted(expr.bindings.items())})
        return IndexLambda(expr=expr.expr,
                           shape=self.rec_idx_or_size_tuple(expr.shape,
                                                            *args, **kwargs),
                           dtype=expr.dtype,
                           bindings=bindings,
                           axes=expr.axes,
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_placeholder(self,
                        expr: Placeholder, *args: P.args, **kwargs: P.kwargs) -> Array:
        assert expr.name is not None
        return Placeholder(name=expr.name,
                           shape=self.rec_idx_or_size_tuple(expr.shape,
                                                            *args, **kwargs),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_stack(self, expr: Stack, *args: P.args, **kwargs: P.kwargs) -> Array:
        arrays = tuple(
            _verify_is_array(self.rec(arr, *args, **kwargs)) for arr in expr.arrays)
        return Stack(arrays=arrays, axis=expr.axis, axes=expr.axes, tags=expr.tags,
                     non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self,
                        expr: Concatenate, *args: P.args, **kwargs: P.kwargs) -> Array:
        arrays = tuple(
            _verify_is_array(self.rec(arr, *args, **kwargs)) for arr in expr.arrays)
        return Concatenate(arrays=arrays, axis=expr.axis,
                           axes=expr.axes, tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_roll(self, expr: Roll, *args: P.args, **kwargs: P.kwargs) -> Array:
        return Roll(array=_verify_is_array(self.rec(expr.array, *args, **kwargs)),
                    shift=expr.shift,
                    axis=expr.axis,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation,
                             *args: P.args, **kwargs: P.kwargs) -> Array:
        return AxisPermutation(array=_verify_is_array(
                                   self.rec(expr.array, *args, **kwargs)),
                               axis_permutation=expr.axis_permutation,
                               axes=expr.axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self,
                        expr: IndexBase, *args: P.args, **kwargs: P.kwargs) -> Array:
        assert isinstance(expr, _SuppliedAxesAndTagsMixin)
        return type(expr)(_verify_is_array(self.rec(expr.array, *args, **kwargs)),
                          indices=self.rec_idx_or_size_tuple(expr.indices,
                                                             *args, **kwargs),
                          axes=expr.axes,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_basic_index(self,
                        expr: BasicIndex, *args: P.args, **kwargs: P.kwargs) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      *args: P.args, **kwargs: P.kwargs

                                      ) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes,
                                          *args: P.args, **kwargs: P.kwargs
                                          ) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_data_wrapper(self, expr: DataWrapper,
                         *args: P.args, **kwargs: P.kwargs) -> Array:
        return DataWrapper(
                data=expr.data,
                shape=self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs),
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_size_param(self,
                       expr: SizeParam, *args: P.args, **kwargs: P.kwargs) -> Array:
        assert expr.name is not None
        return SizeParam(name=expr.name, axes=expr.axes, tags=expr.tags)

    def map_einsum(self, expr: Einsum, *args: P.args, **kwargs: P.kwargs) -> Array:
        return Einsum(expr.access_descriptors,
                      tuple(_verify_is_array(
                          self.rec(arg, *args, **kwargs)) for arg in expr.args),
                      axes=expr.axes,
                      redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                      tags=expr.tags,
                      non_equality_tags=expr.non_equality_tags)

    def map_named_array(self,
                        expr: NamedArray, *args: P.args, **kwargs: P.kwargs) -> Array:
        container = self.rec(expr._container, *args, **kwargs)
        assert isinstance(container, AbstractResultWithNamedArrays)
        return type(expr)(container,
                          expr.name,
                          axes=expr.axes,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_dict_of_named_arrays(self,
                expr: DictOfNamedArrays, *args: P.args, **kwargs: P.kwargs
            ) -> DictOfNamedArrays:
        return DictOfNamedArrays({key: _verify_is_array(
                                      self.rec(val.expr, *args, **kwargs))
                                  for key, val in expr.items()},
                                 tags=expr.tags,
                                 )

    def map_loopy_call(self, expr: LoopyCall,
                       *args: P.args, **kwargs: P.kwargs) -> LoopyCall:
        bindings: Mapping[Any, Any] = immutabledict(
                    {name: (self.rec(subexpr, *args, **kwargs)
                           if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})

        return LoopyCall(translation_unit=expr.translation_unit,
                         bindings=bindings,
                         entrypoint=expr.entrypoint,
                         tags=expr.tags,
                         )

    def map_loopy_call_result(self, expr: LoopyCallResult,
                              *args: P.args, **kwargs: P.kwargs) -> Array:
        rec_loopy_call = self.rec(expr._container, *args, **kwargs)
        assert isinstance(rec_loopy_call, LoopyCall)
        return LoopyCallResult(
                _container=rec_loopy_call,
                name=expr.name,
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape,
                    *args: P.args, **kwargs: P.kwargs) -> Array:
        return Reshape(_verify_is_array(self.rec(expr.array, *args, **kwargs)),
                       newshape=self.rec_idx_or_size_tuple(expr.newshape,
                                                           *args, **kwargs),
                       order=expr.order,
                       axes=expr.axes,
                       tags=expr.tags,
                       non_equality_tags=expr.non_equality_tags)

    def map_distributed_send_ref_holder(self, expr: DistributedSendRefHolder,
                                        *args: P.args, **kwargs: P.kwargs) -> Array:
        return DistributedSendRefHolder(
                send=DistributedSend(
                    data=_verify_is_array(self.rec(expr.send.data, *args, **kwargs)),
                    dest_rank=expr.send.dest_rank,
                    comm_tag=expr.send.comm_tag),
                passthrough_data=_verify_is_array(
                    self.rec(expr.passthrough_data, *args, **kwargs)))

    def map_distributed_recv(self, expr: DistributedRecv,
                             *args: P.args, **kwargs: P.kwargs) -> Array:
        return DistributedRecv(
               src_rank=expr.src_rank, comm_tag=expr.comm_tag,
               shape=self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs),
               dtype=expr.dtype, tags=expr.tags, axes=expr.axes,
               non_equality_tags=expr.non_equality_tags)

    def map_function_definition(
                self, expr: FunctionDefinition,
                *args: P.args, **kwargs: P.kwargs
            ) -> FunctionDefinition:
        raise NotImplementedError("Function definitions are purposefully left"
                                  " unimplemented as the default arguments to a new"
                                  " DAG traversal are tricky to guess.")

    def map_call(self, expr: Call,
                 *args: P.args, **kwargs: P.kwargs) -> AbstractResultWithNamedArrays:
        return Call(self.rec_function_definition(expr.function, *args, **kwargs),
                    immutabledict({name: self.rec(bnd, *args, **kwargs)
                         for name, bnd in expr.bindings.items()}),
                    tags=expr.tags,
                    )

    def map_named_call_result(self, expr: NamedCallResult,
                              *args: P.args, **kwargs: P.kwargs) -> Array:
        call = self.rec(expr._container, *args, **kwargs)
        assert isinstance(call, Call)
        return call[expr.name]

# }}}


# {{{ CombineMapper

class CombineMapper(CachedMapper[ResultT, FunctionResultT, []]):
    """
    Abstract mapper that recursively combines the results of user nodes
    of a given expression.

    .. automethod:: combine
    """
    def get_cache_key(self, expr: ArrayOrNames) -> CacheKeyT:
        return expr

    def get_function_definition_cache_key(self, expr: FunctionDefinition) -> CacheKeyT:
        return expr

    def rec_idx_or_size_tuple(self, situp: tuple[IndexOrShapeExpr, ...]
                              ) -> tuple[ResultT, ...]:
        return tuple(self.rec(s) for s in situp if isinstance(s, Array))

    def combine(self, *args: ResultT) -> ResultT:
        """Combine the arguments."""
        raise NotImplementedError

    def map_index_lambda(self, expr: IndexLambda) -> ResultT:
        return self.combine(*(self.rec(bnd)
                              for _, bnd in sorted(expr.bindings.items())),
                            *self.rec_idx_or_size_tuple(expr.shape))

    def map_placeholder(self, expr: Placeholder) -> ResultT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    def map_data_wrapper(self, expr: DataWrapper) -> ResultT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    def map_stack(self, expr: Stack) -> ResultT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.arrays))

    def map_roll(self, expr: Roll) -> ResultT:
        return self.combine(self.rec(expr.array))

    def map_axis_permutation(self, expr: AxisPermutation) -> ResultT:
        return self.combine(self.rec(expr.array))

    def _map_index_base(self, expr: IndexBase) -> ResultT:
        return self.combine(self.rec(expr.array),
                            *self.rec_idx_or_size_tuple(expr.indices))

    def map_basic_index(self, expr: BasicIndex) -> ResultT:
        return self._map_index_base(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> ResultT:
        return self._map_index_base(expr)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> ResultT:
        return self._map_index_base(expr)

    def map_reshape(self, expr: Reshape) -> ResultT:
        return self.combine(
                self.rec(expr.array),
                *self.rec_idx_or_size_tuple(expr.newshape))

    def map_concatenate(self, expr: Concatenate) -> ResultT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.arrays))

    def map_einsum(self, expr: Einsum) -> ResultT:
        return self.combine(*(self.rec(ary)
                              for ary in expr.args))

    def map_named_array(self, expr: NamedArray) -> ResultT:
        return self.combine(self.rec(expr._container))

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> ResultT:
        return self.combine(*(self.rec(ary.expr)
                              for ary in expr.values()))

    def map_loopy_call(self, expr: LoopyCall) -> ResultT:
        return self.combine(*(self.rec(ary)
                              for _, ary in sorted(expr.bindings.items())
                              if isinstance(ary, Array)))

    def map_loopy_call_result(self, expr: LoopyCallResult) -> ResultT:
        return self.rec(expr._container)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> ResultT:
        return self.combine(
                self.rec(expr.send.data),
                self.rec(expr.passthrough_data),
                )

    def map_distributed_recv(self, expr: DistributedRecv) -> ResultT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    def map_function_definition(self, expr: FunctionDefinition) -> FunctionResultT:
        raise NotImplementedError("Combining results from a callee expression"
                                  " is context-dependent. Derived classes"
                                  " must override map_function_definition.")

    def map_call(self, expr: Call) -> ResultT:
        raise NotImplementedError(
            "Mapping calls is context-dependent. Derived classes must override "
            "map_call.")

    def map_named_call_result(self, expr: NamedCallResult) -> ResultT:
        return self.rec(expr._container)

# }}}


# {{{ DependencyMapper

class DependencyMapper(CombineMapper[R, R]):
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

    def map_loopy_call_result(self, expr: LoopyCallResult) -> R:
        return self.combine(frozenset([expr]), super().map_loopy_call_result(expr))

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> R:
        return self.combine(
                frozenset([expr]), super().map_distributed_send_ref_holder(expr))

    def map_distributed_recv(self, expr: DistributedRecv) -> R:
        return self.combine(frozenset([expr]), super().map_distributed_recv(expr))

    def map_function_definition(self, expr: FunctionDefinition) -> R:
        # do not include arrays from the function's body as it would involve
        # putting arrays from different namespaces into the same collection.
        return frozenset()

    def map_call(self, expr: Call) -> R:
        return self.combine(self.rec_function_definition(expr.function),
                            *[self.rec(bnd) for bnd in expr.bindings.values()])

    def map_named_call_result(self, expr: NamedCallResult) -> R:
        return self.rec(expr._container)

# }}}


# {{{ SubsetDependencyMapper

class SubsetDependencyMapper(DependencyMapper):
    """
    Mapper to combine the dependencies of an expression that are a subset of
    *universe*.
    """
    def __init__(self, universe: frozenset[Array]):
        self.universe = universe
        super().__init__()

    def combine(self, *args: frozenset[Array]) -> frozenset[Array]:
        from functools import reduce
        return reduce(lambda acc, arg: acc | (arg & self.universe),
                      args,
                      frozenset())

# }}}


# {{{ InputGatherer

class InputGatherer(
        CombineMapper[frozenset[InputArgumentBase], frozenset[InputArgumentBase]]):
    """
    Mapper to combine all instances of :class:`pytato.array.InputArgumentBase` that
    an array expression depends on.
    """
    def combine(self, *args: frozenset[InputArgumentBase]
                ) -> frozenset[InputArgumentBase]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_placeholder(self, expr: Placeholder) -> frozenset[InputArgumentBase]:
        return self.combine(frozenset([expr]), super().map_placeholder(expr))

    def map_data_wrapper(self, expr: DataWrapper) -> frozenset[InputArgumentBase]:
        return self.combine(frozenset([expr]), super().map_data_wrapper(expr))

    def map_size_param(self, expr: SizeParam) -> frozenset[SizeParam]:
        return frozenset([expr])

    def map_function_definition(self, expr: FunctionDefinition
                                ) -> frozenset[InputArgumentBase]:
        # get rid of placeholders local to the function.
        new_mapper = InputGatherer()
        all_callee_inputs = new_mapper.combine(*[new_mapper(ret)
                                                 for ret in expr.returns.values()])
        result: set[InputArgumentBase] = set()
        for inp in all_callee_inputs:
            if isinstance(inp, Placeholder):
                if inp.name in expr.parameters:
                    # drop, reference to argument
                    pass
                else:
                    raise ValueError("function definition refers to non-argument "
                                     f"placeholder named '{inp.name}'")
            else:
                result.add(inp)

        return frozenset(result)

    def map_call(self, expr: Call) -> frozenset[InputArgumentBase]:
        return self.combine(self.rec_function_definition(expr.function),
            *[
                self.rec(bnd)
                for name, bnd in sorted(expr.bindings.items())])

# }}}


# {{{ SizeParamGatherer

class SizeParamGatherer(
        CombineMapper[frozenset[SizeParam], frozenset[SizeParam]]):
    """
    Mapper to combine all instances of :class:`pytato.array.SizeParam` that
    an array expression depends on.
    """
    def combine(self, *args: frozenset[SizeParam]
                ) -> frozenset[SizeParam]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_size_param(self, expr: SizeParam) -> frozenset[SizeParam]:
        return frozenset([expr])

    def map_function_definition(self, expr: FunctionDefinition
                                ) -> frozenset[SizeParam]:
        return self.combine(*[self.rec(ret)
                              for ret in expr.returns.values()])

    def map_call(self, expr: Call) -> frozenset[SizeParam]:
        return self.combine(self.rec_function_definition(expr.function),
            *[
                self.rec(bnd)
                for name, bnd in sorted(expr.bindings.items())])

# }}}


# {{{ WalkMapper

class WalkMapper(Mapper[None, None, P]):
    """
    A mapper that walks over all the arrays in a :class:`pytato.Array`.

    Users may override the specific mapper methods in a derived class or
    override :meth:`WalkMapper.visit` and :meth:`WalkMapper.post_visit`.

    .. automethod:: visit
    .. automethod:: post_visit
    """

    def clone_for_callee(
            self, function: FunctionDefinition) -> Self:
        return type(self)()

    def visit(self, expr: Any, *args: P.args, **kwargs: P.kwargs) -> bool:
        """
        If this method returns *True*, *expr* is traversed during the walk.
        If this method returns *False*, *expr* is not traversed as a part of
        the walk.
        """
        return True

    def post_visit(self, expr: Any, *args: P.args, **kwargs: P.kwargs) -> None:
        """
        Callback after *expr* has been traversed.
        """
        pass

    def rec_idx_or_size_tuple(self, situp: tuple[IndexOrShapeExpr, ...],
                              *args: P.args, **kwargs: P.kwargs) -> None:
        for comp in situp:
            if isinstance(comp, Array):
                self.rec(comp, *args, **kwargs)

    def map_index_lambda(self,
                         expr: IndexLambda, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for _, child in sorted(expr.bindings.items()):
            self.rec(child, *args, **kwargs)

        self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_placeholder(self,
                        expr: Placeholder, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    map_data_wrapper = map_placeholder
    map_size_param = map_placeholder

    def _map_index_remapping_base(self, expr: IndexRemappingBase,
                                  *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.array, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_index_base(self,
                        expr: IndexBase, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.array, *args, **kwargs)

        self.rec_idx_or_size_tuple(expr.indices, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_basic_index(self,
                        expr: BasicIndex, *args: P.args, **kwargs: P.kwargs) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      *args: P.args, **kwargs: P.kwargs) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes,
                                          *args: P.args, **kwargs: P.kwargs) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_stack(self, expr: Stack, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.arrays:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_concatenate(self,
                        expr: Concatenate, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.arrays:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_einsum(self, expr: Einsum, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.args:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays,
                                 *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr._data.values():
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder,
            *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.send.data, *args, **kwargs)
        self.rec(expr.passthrough_data, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_distributed_recv(self, expr: DistributedRecv,
                             *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_named_array(self,
                        expr: NamedArray, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr._container, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_loopy_call(self,
                       expr: LoopyCall, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for _, child in sorted(expr.bindings.items()):
            if isinstance(child, Array):
                self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_function_definition(self, expr: FunctionDefinition,
                                *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        new_mapper = self.clone_for_callee(expr)
        for subexpr in expr.returns.values():
            new_mapper(subexpr, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_call(self, expr: Call, *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec_function_definition(expr.function, *args, **kwargs)
        for bnd in expr.bindings.values():
            self.rec(bnd, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_named_call_result(self, expr: NamedCallResult,
                              *args: P.args, **kwargs: P.kwargs) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr._container, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

# }}}


# {{{ CachedWalkMapper

VisitKeyT: TypeAlias = Hashable


class CachedWalkMapper(WalkMapper[P]):
    """
    WalkMapper that visits each node in the DAG exactly once. This loses some
    information compared to :class:`WalkMapper` as a node is visited only from
    one of its predecessors.
    """

    def __init__(
            self,
            _visited_functions: set[VisitKeyT] | None = None
            ) -> None:
        super().__init__()
        self._visited_arrays_or_names: set[VisitKeyT] = set()

        self._visited_functions: set[VisitKeyT] = \
            _visited_functions if _visited_functions is not None else set()

    def get_cache_key(
            self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs
            ) -> VisitKeyT:
        raise NotImplementedError

    def get_function_definition_cache_key(
            self, expr: FunctionDefinition, *args: P.args, **kwargs: P.kwargs
            ) -> VisitKeyT:
        raise NotImplementedError

    def rec(self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs) -> None:
        cache_key = self.get_cache_key(expr, *args, **kwargs)
        if cache_key in self._visited_arrays_or_names:
            return

        super().rec(expr, *args, **kwargs)
        self._visited_arrays_or_names.add(cache_key)

    def rec_function_definition(self, expr: FunctionDefinition,
                                *args: P.args, **kwargs: P.kwargs) -> None:
        cache_key = self.get_function_definition_cache_key(expr, *args, **kwargs)
        if cache_key in self._visited_functions:
            return

        super().rec_function_definition(expr, *args, **kwargs)
        self._visited_functions.add(cache_key)

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(_visited_functions=self._visited_functions)

# }}}


# {{{ TopoSortMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class TopoSortMapper(CachedWalkMapper[[]]):
    """A mapper that creates a list of nodes in topological order.

    :members: topological_order

    .. note::

        Does not consider the nodes inside  a
        :class:`~pytato.function.FunctionDefinition`.
    """

    def __init__(
            self,
            _visited_functions: set[VisitKeyT] | None = None) -> None:
        super().__init__(_visited_functions=_visited_functions)
        self.topological_order: list[Array] = []

    def get_cache_key(self, expr: ArrayOrNames) -> int:
        return id(expr)

    def post_visit(self, expr: Any) -> None:
        self.topological_order.append(expr)

    def map_function_definition(self, expr: FunctionDefinition) -> None:
        # do nothing as it includes arrays from a different namespace.
        return

# }}}


# {{{ MapAndCopyMapper

class CachedMapAndCopyMapper(CopyMapper):
    """
    Mapper that applies *map_fn* to each node and copies it. Results of
    traversals are memoized i.e. each node is mapped via *map_fn* exactly once.
    """

    def __init__(
            self,
            map_fn: Callable[[ArrayOrNames], ArrayOrNames],
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        super().__init__(_cache=_cache, _function_cache=_function_cache)
        self.map_fn: Callable[[ArrayOrNames], ArrayOrNames] = map_fn

    def clone_for_callee(
            self, function: FunctionDefinition) -> Self:
        return type(self)(
            self.map_fn,
            _function_cache=cast(
                "TransformMapperCache[FunctionDefinition, []]", self._function_cache))

    def rec(self, expr: ArrayOrNames) -> ArrayOrNames:
        inputs = self._make_cache_inputs(expr)
        try:
            return self._cache.retrieve(inputs)
        except KeyError:
            # Intentionally going to Mapper instead of super() to avoid
            # double caching when subclasses of CachedMapper override rec,
            # see https://github.com/inducer/pytato/pull/585
            return self._cache.add(inputs, Mapper.rec(self, self.map_fn(expr)))

# }}}


# {{{ MPMS materializer

@dataclasses.dataclass(frozen=True, eq=True)
class MPMSMaterializerAccumulator:
    """This class serves as the return value of :class:`MPMSMaterializer`. It
    contains the set of materialized predecessors and the rewritten expression
    (i.e. the expression with tags for materialization applied).
    """
    materialized_predecessors: frozenset[Array]
    expr: Array


def _materialize_if_mpms(expr: Array,
                         nsuccessors: int,
                         predecessors: Iterable[MPMSMaterializerAccumulator]
                         ) -> MPMSMaterializerAccumulator:
    """
    Returns an instance of :class:`MPMSMaterializerAccumulator`, that
    materializes *expr* if it has more than 1 successor and more than 1
    materialized predecessor.
    """
    from functools import reduce

    materialized_predecessors: frozenset[Array] = reduce(
                                                    frozenset.union,
                                                    (pred.materialized_predecessors
                                                     for pred in predecessors),
                                                    frozenset())
    if nsuccessors > 1 and len(materialized_predecessors) > 1:
        new_expr = expr.tagged(ImplStored())
        return MPMSMaterializerAccumulator(frozenset([new_expr]), new_expr)
    else:
        return MPMSMaterializerAccumulator(materialized_predecessors, expr)


class MPMSMaterializer(Mapper[MPMSMaterializerAccumulator, Never, []]):
    """
    See :func:`materialize_with_mpms` for an explanation.

    .. attribute:: nsuccessors

        A mapping from a node in the expression graph (i.e. an
        :class:`~pytato.Array`) to its number of successors.
    """
    def __init__(self, nsuccessors: Mapping[Array, int]):
        super().__init__()
        self.nsuccessors = nsuccessors
        self.cache: dict[ArrayOrNames, MPMSMaterializerAccumulator] = {}

    def rec(self, expr: ArrayOrNames) -> MPMSMaterializerAccumulator:
        if expr in self.cache:
            return self.cache[expr]
        result: MPMSMaterializerAccumulator = super().rec(expr)
        self.cache[expr] = result
        return result

    def _map_input_base(self, expr: InputArgumentBase
                        ) -> MPMSMaterializerAccumulator:
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_named_array(self, expr: NamedArray) -> MPMSMaterializerAccumulator:
        raise NotImplementedError("only LoopyCallResult named array"
                                  " supported for now.")

    def map_index_lambda(self, expr: IndexLambda) -> MPMSMaterializerAccumulator:
        children_rec = {bnd_name: self.rec(bnd)
                        for bnd_name, bnd in sorted(expr.bindings.items())}

        new_expr = IndexLambda(expr=expr.expr,
                               shape=expr.shape,
                               dtype=expr.dtype,
                               bindings=immutabledict({bnd_name: bnd.expr
                                for bnd_name, bnd in sorted(children_rec.items())}),
                               axes=expr.axes,
                               var_to_reduction_descr=expr.var_to_reduction_descr,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)
        return _materialize_if_mpms(new_expr, self.nsuccessors[expr],
                                    children_rec.values())

    def map_stack(self, expr: Stack) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_expr = Stack(tuple(ary.expr for ary in rec_arrays),
                         expr.axis, axes=expr.axes, tags=expr.tags,
                         non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_concatenate(self, expr: Concatenate) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_expr = Concatenate(tuple(ary.expr for ary in rec_arrays),
                               expr.axis,
                               axes=expr.axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)
        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_roll(self, expr: Roll) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = Roll(rec_array.expr, expr.shift, expr.axis, axes=expr.axes,
                        tags=expr.tags,
                        non_equality_tags=expr.non_equality_tags)
        return _materialize_if_mpms(new_expr, self.nsuccessors[expr],
                                    (rec_array,))

    def map_axis_permutation(self, expr: AxisPermutation
                             ) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = AxisPermutation(rec_array.expr, expr.axis_permutation,
                                   axes=expr.axes, tags=expr.tags,
                                   non_equality_tags=expr.non_equality_tags)
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
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array, *tuple(rec_indices.values()))
                                    )

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_reshape(self, expr: Reshape) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        new_expr = Reshape(rec_array.expr, expr.newshape,
                           expr.order, axes=expr.axes, tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array,))

    def map_einsum(self, expr: Einsum) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.args]
        new_expr = Einsum(expr.access_descriptors,
                          tuple(ary.expr for ary in rec_arrays),
                          expr.redn_axis_to_redn_descr,
                          axes=expr.axes,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays
                                 ) -> MPMSMaterializerAccumulator:
        raise NotImplementedError

    def map_loopy_call_result(self, expr: NamedArray) -> MPMSMaterializerAccumulator:
        # loopy call result is always materialized
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> MPMSMaterializerAccumulator:
        rec_passthrough = self.rec(expr.passthrough_data)
        rec_send_data = self.rec(expr.send.data)
        new_expr = DistributedSendRefHolder(
            send=DistributedSend(rec_send_data.expr,
                                 dest_rank=expr.send.dest_rank,
                                 comm_tag=expr.send.comm_tag,
                                 tags=expr.send.tags),
            passthrough_data=rec_passthrough.expr,
            )
        return MPMSMaterializerAccumulator(
            rec_passthrough.materialized_predecessors, new_expr)

    def map_distributed_recv(self, expr: DistributedRecv
                             ) -> MPMSMaterializerAccumulator:
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_named_call_result(self, expr: NamedCallResult
                              ) -> MPMSMaterializerAccumulator:
        raise NotImplementedError("MPMSMaterializer does not support functions.")

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
        data = {}
    else:
        data = {name: _verify_is_array(copy_mapper.rec(val.expr))
                for name, val in sorted(source_dict.items())}

    return DictOfNamedArrays(data, tags=source_dict.tags)


def get_dependencies(expr: DictOfNamedArrays) -> dict[str, frozenset[Array]]:
    """Returns the dependencies of each named array in *expr*.
    """
    dep_mapper = DependencyMapper()

    return {name: dep_mapper(val.expr) for name, val in expr.items()}


def map_and_copy(expr: MappedT,
                 map_fn: Callable[[ArrayOrNames], ArrayOrNames]
                 ) -> MappedT:
    """
    Returns a copy of *expr* with every array expression reachable from *expr*
    mapped via *map_fn*.

    .. note::

        Uses :class:`CachedMapAndCopyMapper` under the hood and because of its
        caching nature each node is mapped exactly once.
    """
    return cast("MappedT", CachedMapAndCopyMapper(map_fn)(expr))


def materialize_with_mpms(expr: DictOfNamedArrays) -> DictOfNamedArrays:
    r"""
    Materialize nodes in *expr* with MPMS materialization strategy.
    MPMS stands for Multiple-Predecessors, Multiple-Successors.

    .. note::

        - MPMS materialization strategy is a greedy materialization algorithm in
          which any node with more than 1 materialized predecessor and more than
          1 successor is materialized.
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
    from pytato.analysis import get_num_nodes, get_num_tags_of_type, get_nusers
    materializer = MPMSMaterializer(get_nusers(expr))
    new_data = {}
    for name, ary in expr.items():
        new_data[name] = materializer(ary.expr).expr

    res = DictOfNamedArrays(new_data, tags=expr.tags)

    from pytato import DEBUG_ENABLED
    if DEBUG_ENABLED:
        transform_logger.info("materialize_with_mpms: materialized "
            f"{get_num_tags_of_type(res, ImplStored)} out of "
            f"{get_num_nodes(res)} nodes")

    return res

# }}}


# {{{ UsersCollector

class UsersCollector(CachedMapper[None, Never, []]):
    """
    Maps a graph to a dictionary representation mapping a node to its users,
    i.e. all the nodes using its value.

    .. attribute:: node_to_users

       Mapping of each node in the graph to its users.

    .. automethod:: __init__
    """

    def __init__(self) -> None:
        super().__init__()
        self.node_to_users: dict[ArrayOrNames,
                set[DistributedSend | ArrayOrNames]] = {}

    def __call__(self, expr: ArrayOrNames) -> None:  # type: ignore[override]
        # Root node has no predecessor
        self.node_to_users[expr] = set()
        self.rec(expr)

    def rec_idx_or_size_tuple(
            self, expr: Array, situp: tuple[IndexOrShapeExpr, ...]
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

    def map_einsum(self, expr: Einsum) -> None:
        for arg in expr.args:
            self.node_to_users.setdefault(arg, set()).add(expr)
            self.rec(arg)

        self.rec_idx_or_size_tuple(expr, expr.shape)

    def map_reshape(self, expr: Reshape) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

        self.node_to_users.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_placeholder(self, expr: Placeholder) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

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
        self.rec_idx_or_size_tuple(expr, expr.shape)

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

    def map_loopy_call(self, expr: LoopyCall) -> None:
        for _, child in sorted(expr.bindings.items()):
            if isinstance(child, Array):
                self.node_to_users.setdefault(child, set()).add(expr)
                self.rec(child)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> None:
        self.node_to_users.setdefault(expr.passthrough_data, set()).add(expr)
        self.rec(expr.passthrough_data)
        self.node_to_users.setdefault(expr.send.data, set()).add(expr.send)
        self.rec(expr.send.data)

    def map_distributed_recv(self, expr: DistributedRecv) -> None:
        self.rec_idx_or_size_tuple(expr, expr.shape)

    def map_function_definition(self, expr: FunctionDefinition) -> None:
        raise AssertionError("Control shouldn't reach at this point."
                             " Instantiate another UsersCollector to"
                             " traverse the callee function.")

    def map_call(self, expr: Call) -> None:
        for bnd in expr.bindings.values():
            self.rec(bnd)

    def map_named_call_result(self, expr: NamedCallResult) -> None:
        assert isinstance(expr._container, Call)
        for bnd in expr._container.bindings.values():
            self.node_to_users.setdefault(bnd, set()).add(expr)

        self.rec(expr._container)


def get_users(expr: ArrayOrNames) -> dict[ArrayOrNames,
                                          set[ArrayOrNames]]:
    """
    Returns a mapping from node in *expr* to its direct users.
    """
    user_collector = UsersCollector()
    user_collector(expr)
    return user_collector.node_to_users  # type: ignore[return-value]

# }}}


# {{{ operations on graphs in dict form

def _recursively_get_all_users(
        direct_users: Mapping[ArrayOrNames, set[ArrayOrNames]],
        node: ArrayOrNames) -> frozenset[ArrayOrNames]:
    result = set()
    queue = list(direct_users.get(node, set()))
    ids_already_noted_to_visit: set[int] = set()

    while queue:
        current_node = queue[0]
        queue = queue[1:]
        result.add(current_node)
        # visit each user only once.
        users_to_visit = frozenset({user
                                    for user in direct_users.get(current_node, set())
                                    if id(user) not in ids_already_noted_to_visit})

        ids_already_noted_to_visit.update({id(k)
                                           for k in users_to_visit})

        queue.extend(list(users_to_visit))

    return frozenset(result)


def rec_get_user_nodes(expr: ArrayOrNames,
                       node: ArrayOrNames,
                       ) -> frozenset[ArrayOrNames]:
    """
    Returns all direct and indirect users of *node* in *expr*.
    """
    users = get_users(expr)
    return _recursively_get_all_users(users, node)

# }}}


# {{{ deduplicate_data_wrappers

class DataWrapperDeduplicator(CopyMapper):
    """
    Mapper to replace all :class:`pytato.array.DataWrapper` instances containing
    identical data with a single instance.
    """
    def __init__(
            self,
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        super().__init__(_cache=_cache, _function_cache=_function_cache)
        self.data_wrapper_cache: dict[CacheKeyT, DataWrapper] = {}
        self.data_wrappers_encountered = 0

    def _get_data_dedup_cache_key(self, ary: DataInterface) -> CacheKeyT:
        import sys
        if "pyopencl" in sys.modules:
            from pyopencl import MemoryObjectHolder
            from pyopencl.array import Array as CLArray
            try:
                from pyopencl import SVMPointer
            except ImportError:
                SVMPointer = None  # noqa: N806

            if isinstance(ary, CLArray):
                base_data = ary.base_data
                if isinstance(ary.base_data, MemoryObjectHolder):
                    ptr = base_data.int_ptr
                elif SVMPointer is not None and isinstance(base_data, SVMPointer):
                    ptr = base_data.svm_ptr
                elif base_data is None:
                    # pyopencl represents 0-long arrays' base_data as None
                    ptr = None
                else:
                    raise ValueError("base_data of array not understood")

                return (
                        ptr,
                        ary.offset,
                        ary.shape,
                        ary.strides,
                        ary.dtype,
                        )
        if isinstance(ary, np.ndarray):
            return (
                    ary.__array_interface__["data"],
                    ary.shape,
                    ary.strides,
                    ary.dtype,
                    )
        else:
            raise NotImplementedError(str(type(ary)))

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        self.data_wrappers_encountered += 1
        cache_key = self._get_data_dedup_cache_key(expr.data)
        try:
            return self.data_wrapper_cache[cache_key]
        except KeyError:
            self.data_wrapper_cache[cache_key] = expr
            return expr


def deduplicate_data_wrappers(array_or_names: ArrayOrNames) -> ArrayOrNames:
    """For the expression graph given as *array_or_names*, replace all
    :class:`pytato.array.DataWrapper` instances containing identical data
    with a single instance.

    .. note::

        Currently only supports :class:`numpy.ndarray` and
        :class:`pyopencl.array.Array`.

    .. note::

        This function currently uses addresses of memory buffers to detect
        duplicate data, and so it may fail to deduplicate some instances
        of identical-but-separately-stored data. User code must tolerate
        this, but it must *also* tolerate this function doing a more thorough
        job of deduplication.
    """
    dedup = DataWrapperDeduplicator()
    array_or_names = dedup(array_or_names)

    if dedup.data_wrappers_encountered:
        transform_logger.debug("data wrapper de-duplication: "
                               "%d encountered, %d kept, %d eliminated",
                               dedup.data_wrappers_encountered,
                               len(dedup.data_wrapper_cache),
                               (
                                   dedup.data_wrappers_encountered
                                   - len(dedup.data_wrapper_cache)))

    return array_or_names

# }}}

# vim: foldmethod=marker
