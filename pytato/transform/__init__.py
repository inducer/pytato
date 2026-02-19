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
import logging
from collections.abc import Hashable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from constantdict import constantdict
from typing_extensions import Never, Self, override

from pymbolic.mapper.optimize import optimize_mapper

from pytato.array import (
    AbstractResultWithNamedArrays,
    AdvancedIndexInContiguousAxes,
    AdvancedIndexInNoncontiguousAxes,
    Array,
    ArrayOrScalar,
    AxisPermutation,
    BasicIndex,
    Concatenate,
    DataInterface,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexExpr,
    IndexLambda,
    IndexRemappingBase,
    InputArgumentBase,
    NamedArray,
    Placeholder,
    Reshape,
    Roll,
    ShapeType,
    SizeParam,
    Stack,
    _entries_are_identical,
    _SuppliedAxesAndTagsMixin,
)
from pytato.equality import EqualityComparer
from pytato.function import Call, FunctionDefinition, NamedCallResult
from pytato.loopy import LoopyCall, LoopyCallResult


if TYPE_CHECKING:
    from collections.abc import Callable

    from pytato.distributed.nodes import (
        DistributedRecv,
        DistributedSend,
        DistributedSendRefHolder,
    )


ArrayOrNames: TypeAlias = Array | AbstractResultWithNamedArrays
ArrayOrNamesTc = TypeVar("ArrayOrNamesTc",
                  Array, AbstractResultWithNamedArrays, DictOfNamedArrays)
ArrayOrNamesOrFunctionDefTc = TypeVar("ArrayOrNamesOrFunctionDefTc",
                  Array, AbstractResultWithNamedArrays, DictOfNamedArrays,
                  FunctionDefinition)
IndexOrShapeExpr = TypeVar("IndexOrShapeExpr")
R = frozenset[Array]

__doc__ = """
.. autoclass:: Mapper
.. autoclass:: CacheInputsWithKey
.. autoclass:: CachedMapperCache
.. autoclass:: CachedMapper
.. autoclass:: TransformMapperCache
.. autoclass:: TransformMapper
.. autoclass:: TransformMapperWithExtraArgs
.. autoclass:: CopyMapper
.. autoclass:: CopyMapperWithExtraArgs
.. autoclass:: Deduplicator
.. autoclass:: CombineMapper
.. autoclass:: DependencyMapper
.. autoclass:: InputGatherer
.. autoclass:: ListOfInputsGatherer
.. autoclass:: SizeParamGatherer
.. autoclass:: SubsetDependencyMapper
.. autoclass:: WalkMapper
.. autoclass:: CachedWalkMapper
.. autoclass:: TopoSortMapper
.. autoclass:: CachedMapAndCopyMapper
.. autoclass:: MapOnceMapper
.. autoclass:: MapAndReduceMapper
.. autoclass:: NodeCollector
.. autofunction:: copy_dict_of_named_arrays
.. autofunction:: deduplicate
.. autofunction:: get_dependencies
.. autofunction:: map_and_copy
.. autofunction:: map_and_reduce
.. autofunction:: collect_nodes
.. autofunction:: deduplicate_data_wrappers
.. automodule:: pytato.transform.lower_to_index_lambda
.. automodule:: pytato.transform.remove_broadcasts_einsum
.. automodule:: pytato.transform.einsum_distributive_law
.. automodule:: pytato.transform.materialize
.. automodule:: pytato.transform.metadata
.. automodule:: pytato.transform.dead_code_elimination

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

.. class:: ArrayOrNamesTc

    A type variable representing the input type of a :class:`Mapper`, excluding
    functions.

.. class:: ArrayOrNamesOrFunctionDefTc

    A type variable representing the input type of a :class:`Mapper`, including
    functions.

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

transform_logger = logging.getLogger(__name__)


class UnsupportedArrayError(ValueError):
    pass


class ForeignObjectError(ValueError):
    pass


class CacheCollisionError(ValueError):
    pass


class MapperCreatedDuplicateError(ValueError):
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

CacheExprT = TypeVar("CacheExprT", ArrayOrNames, FunctionDefinition)
CacheResultT = TypeVar("CacheResultT")
CacheKeyT: TypeAlias = Hashable


class CacheInputsWithKey(Generic[CacheExprT, P]):
    """
    Data structure for inputs to :class:`CachedMapperCache`.

    .. attribute:: expr

        The input expression being mapped.

    .. attribute:: args

        A :class:`tuple` of extra positional arguments.

    .. attribute:: kwargs

        A :class:`dict` of extra keyword arguments.

    .. attribute:: key

        The cache key corresponding to *expr* and any additional inputs that were
        passed.

    """
    def __init__(
            self,
            expr: CacheExprT,
            key: CacheKeyT,
            *args: P.args,
            **kwargs: P.kwargs):
        self.expr: CacheExprT = expr
        self.args: tuple[Any, ...] = args
        self.kwargs: dict[str, Any] = kwargs
        self.key: CacheKeyT = key


class CachedMapperCache(Generic[CacheExprT, CacheResultT, P]):
    """
    Cache for mappers.

    .. automethod:: __init__

        Compute the key for an input expression.

    .. automethod:: add
    .. automethod:: retrieve
    .. automethod:: clear
    """
    def __init__(self, err_on_collision: bool) -> None:
        """
        Initialize the cache.

        :arg err_on_collision: Raise an exception if two distinct input expression
            instances have the same key.
        """
        self.err_on_collision = err_on_collision

        self._input_key_to_result: dict[CacheKeyT, CacheResultT] = {}
        if self.err_on_collision:
            self._input_key_to_expr: dict[CacheKeyT, CacheExprT] = {}

    def add(
            self,
            inputs: CacheInputsWithKey[CacheExprT, P],
            result: CacheResultT) -> CacheResultT:
        """Cache a mapping result."""
        key = inputs.key

        assert key not in self._input_key_to_result, \
            f"Cache entry is already present for key '{key}'."

        self._input_key_to_result[key] = result
        if self.err_on_collision:
            self._input_key_to_expr[key] = inputs.expr

        return result

    def retrieve(self, inputs: CacheInputsWithKey[CacheExprT, P]) -> CacheResultT:
        """Retrieve the cached mapping result."""
        key = inputs.key

        result = self._input_key_to_result[key]

        if self.err_on_collision and inputs.expr is not self._input_key_to_expr[key]:
            raise CacheCollisionError

        return result

    def clear(self) -> None:
        """Reset the cache."""
        self._input_key_to_result = {}
        if self.err_on_collision:
            self._input_key_to_expr = {}


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
            err_on_collision: bool = __debug__,
            _cache:
                CachedMapperCache[ArrayOrNames, ResultT, P] | None = None,
            _function_cache:
                CachedMapperCache[FunctionDefinition, FunctionResultT, P] | None = None
            ) -> None:
        super().__init__()

        self._cache: CachedMapperCache[ArrayOrNames, ResultT, P] = (
            _cache if _cache is not None
            else CachedMapperCache(err_on_collision=err_on_collision))

        self._function_cache: CachedMapperCache[
                FunctionDefinition, FunctionResultT, P] = (
            _function_cache if _function_cache is not None
            else CachedMapperCache(err_on_collision=err_on_collision))

    def get_cache_key(
                self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs
            ) -> CacheKeyT:
        if args or kwargs:
            # Depending on whether extra arguments are passed by position or by
            # keyword, they can end up in either args or kwargs; hence key is not
            # uniquely defined in the general case
            raise NotImplementedError(
                "Derived classes must override get_cache_key if using extra inputs.")
        return expr

    def get_function_definition_cache_key(
                self, expr: FunctionDefinition, *args: P.args, **kwargs: P.kwargs
            ) -> CacheKeyT:
        if args or kwargs:
            # Depending on whether extra arguments are passed by position or by
            # keyword, they can end up in either args or kwargs; hence key is not
            # uniquely defined in the general case
            raise NotImplementedError(
                "Derived classes must override get_function_definition_cache_key if "
                "using extra inputs.")
        return expr

    def _make_cache_inputs(
            self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs
            ) -> CacheInputsWithKey[ArrayOrNames, P]:
        return CacheInputsWithKey(
            expr, self.get_cache_key(expr, *args, **kwargs), *args, **kwargs)

    def _make_function_definition_cache_inputs(
            self, expr: FunctionDefinition, *args: P.args, **kwargs: P.kwargs
            ) -> CacheInputsWithKey[FunctionDefinition, P]:
        return CacheInputsWithKey(
            expr, self.get_function_definition_cache_key(expr, *args, **kwargs),
            *args, **kwargs)

    def _cache_add(
            self,
            inputs: CacheInputsWithKey[ArrayOrNames, P],
            result: ResultT) -> ResultT:
        return self._cache.add(inputs, result)

    def _function_cache_add(
            self,
            inputs: CacheInputsWithKey[FunctionDefinition, P],
            result: FunctionResultT) -> FunctionResultT:
        return self._function_cache.add(inputs, result)

    def _cache_retrieve(self, inputs: CacheInputsWithKey[ArrayOrNames, P]) -> ResultT:
        try:
            return self._cache.retrieve(inputs)
        except CacheCollisionError as e:
            raise ValueError(
                f"cache collision detected on {type(inputs.expr)} in "
                f"{type(self)}.") from e

    def _function_cache_retrieve(
            self, inputs: CacheInputsWithKey[FunctionDefinition, P]) -> FunctionResultT:
        try:
            return self._function_cache.retrieve(inputs)
        except CacheCollisionError as e:
            raise ValueError(
                f"cache collision detected on {type(inputs.expr)} in "
                f"{type(self)}.") from e

    def rec(self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs) -> ResultT:
        inputs = self._make_cache_inputs(expr, *args, **kwargs)
        try:
            return self._cache_retrieve(inputs)
        except KeyError:
            # Intentionally going to Mapper instead of super() to avoid
            # double caching when subclasses of CachedMapper override rec,
            # see https://github.com/inducer/pytato/pull/585
            return self._cache_add(inputs, Mapper.rec(self, expr, *args, **kwargs))

    def rec_function_definition(
                self, expr: FunctionDefinition, *args: P.args, **kwargs: P.kwargs
            ) -> FunctionResultT:
        inputs = self._make_function_definition_cache_inputs(expr, *args, **kwargs)
        try:
            return self._function_cache_retrieve(inputs)
        except KeyError:
            return self._function_cache_add(
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
        return type(self)(
            err_on_collision=self._cache.err_on_collision,
            # Functions are cached globally, but arrays aren't
            _function_cache=self._function_cache)

# }}}


# {{{ TransformMapper

def _is_mapper_created_duplicate(
        expr: CacheExprT, result: CacheExprT, *,
        equality_comparer: EqualityComparer) -> bool:
    """Returns *True* if *result* is not identical to *expr* when it ought to be."""
    # For this check to work, must preserve duplicates when retrieving
    # predecessors. DirectPredecessorsGetter deduplicates by virtue of
    # storing the predecessors it finds in sets
    from pytato.analysis import ListOfDirectPredecessorsGetter
    pred_getter = ListOfDirectPredecessorsGetter(include_functions=True)
    return (
        result is not expr
        and hash(result) == hash(expr)
        and equality_comparer(result, expr)
        # Only consider "direct" duplication, not duplication resulting from
        # equality-preserving changes to predecessors. Assume that such changes are
        # OK, otherwise they would have been detected at the point at which they
        # originated. (For example, consider a DAG containing pre-existing
        # duplicates. If a subexpression of *expr* is a duplicate and is replaced
        # with a previously encountered version from the cache, a new instance of
        # *expr* must be created. This should not trigger an error.)
        and all(
            result_pred is pred
            for pred, result_pred in zip(
                pred_getter(expr),
                pred_getter(result),
                strict=True)))


class TransformMapperCache(CachedMapperCache[CacheExprT, CacheExprT, P]):
    """
    Cache for :class:`TransformMapper` and :class:`TransformMapperWithExtraArgs`.

    .. automethod:: __init__
    .. automethod:: add
    """
    def __init__(
            self,
            err_on_collision: bool,
            err_on_created_duplicate: bool) -> None:
        """
        Initialize the cache.

        :arg err_on_collision: Raise an exception if two distinct input expression
            instances have the same key.
        :arg err_on_created_duplicate: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        super().__init__(err_on_collision=err_on_collision)

        self.err_on_created_duplicate = err_on_created_duplicate

        self._result_to_cached_result: dict[CacheExprT, CacheExprT] = {}

        self._equality_comparer: EqualityComparer = EqualityComparer()

    def add(
            self,
            inputs: CacheInputsWithKey[CacheExprT, P],
            result: CacheExprT) -> CacheExprT:
        """
        Cache a mapping result.

        Returns the cached result (which may not be identical to *result* if a
        result was already cached with the same result key).
        """
        key = inputs.key

        assert key not in self._input_key_to_result, \
            f"Cache entry is already present for key '{key}'."

        try:
            # The first encountered instance of each distinct result (in terms of
            # "==") gets cached, and subsequent mappings with results that are equal
            # to prior cached results are replaced with the original instance
            result = self._result_to_cached_result[result]
        except KeyError:
            if (
                    self.err_on_created_duplicate
                    and _is_mapper_created_duplicate(
                        inputs.expr, result,
                        equality_comparer=self._equality_comparer)):
                raise MapperCreatedDuplicateError from None

            self._result_to_cached_result[result] = result

        self._input_key_to_result[key] = result
        if self.err_on_collision:
            self._input_key_to_expr[key] = inputs.expr

        return result


class TransformMapper(CachedMapper[ArrayOrNames, FunctionDefinition, []]):
    """Base class for mappers that transform :class:`pytato.array.Array`\\ s into
    other :class:`pytato.array.Array`\\ s.

    Enables certain operations that can only be done if the mapping results are also
    arrays (e.g., computing a cache key from them). Does not implement default
    mapper methods; for that, see :class:`CopyMapper`.

    .. automethod:: __init__
    .. automethod:: clone_for_callee
    """
    def __init__(
            self,
            err_on_collision: bool = __debug__,
            err_on_created_duplicate: bool = __debug__,
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        """
        :arg err_on_collision: Raise an exception if two distinct input array
            instances have the same key.
        :arg err_on_created_duplicate: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        if _cache is None:
            _cache = TransformMapperCache(
                err_on_collision=err_on_collision,
                err_on_created_duplicate=err_on_created_duplicate)

        if _function_cache is None:
            _function_cache = TransformMapperCache(
                err_on_collision=err_on_collision,
                err_on_created_duplicate=err_on_created_duplicate)

        super().__init__(
            err_on_collision=err_on_collision,
            _cache=_cache,
            _function_cache=_function_cache)

    def _cache_add(
            self,
            inputs: CacheInputsWithKey[ArrayOrNames, []],
            result: ArrayOrNames) -> ArrayOrNames:
        try:
            return self._cache.add(inputs, result)
        except MapperCreatedDuplicateError as e:
            raise ValueError(
                f"mapper-created duplicate detected on {type(inputs.expr)} in "
                f"{type(self)}.") from e

    def _function_cache_add(
            self,
            inputs: CacheInputsWithKey[FunctionDefinition, []],
            result: FunctionDefinition) -> FunctionDefinition:
        try:
            return self._function_cache.add(inputs, result)
        except MapperCreatedDuplicateError as e:
            raise ValueError(
                f"mapper-created duplicate detected on {type(inputs.expr)} in "
                f"{type(self)}.") from e

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        function_cache = cast(
            "TransformMapperCache[FunctionDefinition, []]", self._function_cache)
        return type(self)(
            err_on_collision=function_cache.err_on_collision,
            err_on_created_duplicate=function_cache.err_on_created_duplicate,
            _function_cache=function_cache)

    @override
    # This overrides incompatibly on purpose, in order to convey stronger
    # guarantees. We're not trying to be very mapper-polymorphic, so
    # IMO this inconsistency is "worth it(tm)". -AK, 2025-06-16
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            expr: ArrayOrNamesOrFunctionDefTc,
            ) -> ArrayOrNamesOrFunctionDefTc:
        """Handle the mapping of *expr*."""
        if isinstance(expr, Array):
            return cast("Array", self.rec(expr))
        elif isinstance(expr, AbstractResultWithNamedArrays):
            return cast("AbstractResultWithNamedArrays", self.rec(expr))
        else:
            return self.rec_function_definition(expr)

# }}}


# {{{ TransformMapperWithExtraArgs

class TransformMapperWithExtraArgs(
        CachedMapper[ArrayOrNames, FunctionDefinition, P]):
    """
    Similar to :class:`TransformMapper`, but each mapper method takes extra
    ``*args``, ``**kwargs`` that are propagated along a path by default.

    The logic in :class:`TransformMapper` purposely does not take the extra
    arguments to keep the cost of its each call frame low.

    .. automethod:: __init__
    .. automethod:: clone_for_callee
    """
    def __init__(
            self,
            err_on_collision: bool = __debug__,
            err_on_created_duplicate: bool = __debug__,
            _cache: TransformMapperCache[ArrayOrNames, P] | None = None,
            _function_cache:
                TransformMapperCache[FunctionDefinition, P] | None = None
            ) -> None:
        """
        :arg err_on_collision: Raise an exception if two distinct input array
            instances have the same key.
        :arg err_on_created_duplicate: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        if _cache is None:
            _cache = TransformMapperCache(
                err_on_collision=err_on_collision,
                err_on_created_duplicate=err_on_created_duplicate)

        if _function_cache is None:
            _function_cache = TransformMapperCache(
                err_on_collision=err_on_collision,
                err_on_created_duplicate=err_on_created_duplicate)

        super().__init__(
            err_on_collision=err_on_collision,
            _cache=_cache,
            _function_cache=_function_cache)

    def _cache_add(
            self,
            inputs: CacheInputsWithKey[ArrayOrNames, P],
            result: ArrayOrNames) -> ArrayOrNames:
        try:
            return self._cache.add(inputs, result)
        except MapperCreatedDuplicateError as e:
            raise ValueError(
                f"mapper-created duplicate detected on {type(inputs.expr)} in "
                f"{type(self)}.") from e

    def _function_cache_add(
            self,
            inputs: CacheInputsWithKey[FunctionDefinition, P],
            result: FunctionDefinition) -> FunctionDefinition:
        try:
            return self._function_cache.add(inputs, result)
        except MapperCreatedDuplicateError as e:
            raise ValueError(
                f"mapper-created duplicate detected on {type(inputs.expr)} in "
                f"{type(self)}.") from e

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        function_cache = cast(
            "TransformMapperCache[FunctionDefinition, P]", self._function_cache)
        return type(self)(
            err_on_collision=function_cache.err_on_collision,
            err_on_created_duplicate=function_cache.err_on_created_duplicate,
            _function_cache=function_cache)

# }}}


# {{{ CopyMapper

class CopyMapper(TransformMapper):
    """Performs a deep copy of a :class:`pytato.array.Array`.
    The typical use of this mapper is to override individual ``map_`` methods
    in subclasses to permit term rewriting on an expression graph.

    .. note::

       This does not copy the data of a :class:`pytato.array.DataWrapper`.
    """
    def rec_size_tuple(self, situp: ShapeType) -> ShapeType:
        new_situp = tuple(
            _verify_is_array(self.rec(s)) if isinstance(s, Array) else s
            for s in situp)
        return situp if _entries_are_identical(new_situp, situp) else new_situp

    def rec_idx_tuple(self, situp: tuple[IndexExpr, ...]) -> tuple[IndexExpr, ...]:
        new_situp = tuple(
            _verify_is_array(self.rec(s)) if isinstance(s, Array) else s
            for s in situp)
        return situp if _entries_are_identical(new_situp, situp) else new_situp

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        new_shape = self.rec_size_tuple(expr.shape)
        new_bindings: Mapping[str, Array] = constantdict({
                name: _verify_is_array(self.rec(subexpr))
                # FIXME: Are these sorts still necessary?
                for name, subexpr in sorted(expr.bindings.items())})
        return expr.replace_if_different(shape=new_shape, bindings=new_bindings)

    def map_placeholder(self, expr: Placeholder) -> Array:
        assert expr.name is not None
        new_shape = self.rec_size_tuple(expr.shape)
        return expr.replace_if_different(shape=new_shape)

    def map_stack(self, expr: Stack) -> Array:
        new_arrays = tuple(_verify_is_array(self.rec(arr)) for arr in expr.arrays)
        return expr.replace_if_different(arrays=new_arrays)

    def map_concatenate(self, expr: Concatenate) -> Array:
        new_arrays = tuple(_verify_is_array(self.rec(arr)) for arr in expr.arrays)
        return expr.replace_if_different(arrays=new_arrays)

    def map_roll(self, expr: Roll) -> Array:
        new_ary = _verify_is_array(self.rec(expr.array))
        return expr.replace_if_different(array=new_ary)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        new_ary = _verify_is_array(self.rec(expr.array))
        return expr.replace_if_different(array=new_ary)

    def _map_index_base(self, expr: IndexBase) -> Array:
        new_ary = _verify_is_array(self.rec(expr.array))
        new_indices = self.rec_idx_tuple(expr.indices)
        return expr.replace_if_different(array=new_ary, indices=new_indices)

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
        new_shape = self.rec_size_tuple(expr.shape)
        return expr.replace_if_different(shape=new_shape)

    def map_size_param(self, expr: SizeParam) -> Array:
        assert expr.name is not None
        return expr

    def map_einsum(self, expr: Einsum) -> Array:
        new_args = tuple(_verify_is_array(self.rec(arg)) for arg in expr.args)
        return expr.replace_if_different(args=new_args)

    def map_named_array(self, expr: NamedArray) -> Array:
        new_container = self.rec(expr._container)
        assert isinstance(new_container, AbstractResultWithNamedArrays)
        return expr.replace_if_different(_container=new_container)

    def map_dict_of_named_arrays(self,
            expr: DictOfNamedArrays) -> DictOfNamedArrays:
        new_data = {
            key: _verify_is_array(self.rec(val.expr))
            for key, val in expr.items()}
        return expr.replace_if_different(data=new_data)

    def map_loopy_call(self, expr: LoopyCall) -> LoopyCall:
        new_bindings: Mapping[str, ArrayOrScalar] = constantdict(
                    {name: (_verify_is_array(self.rec(subexpr))
                            if isinstance(subexpr, Array) else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})
        return expr.replace_if_different(bindings=new_bindings)

    def map_loopy_call_result(self, expr: LoopyCallResult) -> Array:
        new_container = self.rec(expr._container)
        assert isinstance(new_container, LoopyCall)
        return expr.replace_if_different(_container=new_container)

    def map_reshape(self, expr: Reshape) -> Array:
        new_ary = _verify_is_array(self.rec(expr.array))
        new_newshape = self.rec_size_tuple(expr.newshape)
        return expr.replace_if_different(array=new_ary, newshape=new_newshape)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> Array:
        new_send_data = _verify_is_array(self.rec(expr.send.data))
        new_passthrough = _verify_is_array(self.rec(expr.passthrough_data))
        return expr.replace_if_different(
            send=expr.send.replace_if_different(data=new_send_data),
            passthrough_data=new_passthrough)

    def map_distributed_recv(self, expr: DistributedRecv) -> Array:
        new_shape = self.rec_size_tuple(expr.shape)
        return expr.replace_if_different(shape=new_shape)

    def map_function_definition(self,
                                expr: FunctionDefinition) -> FunctionDefinition:
        # spawn a new mapper to avoid unsound cache hits, since the namespace of the
        # function's body is different from that of the caller.
        new_mapper = self.clone_for_callee(expr)
        new_returns: Mapping[str, Array] = constantdict({
            name: _verify_is_array(new_mapper(ret))
            for name, ret in expr.returns.items()})
        return expr.replace_if_different(returns=new_returns)

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        new_function = self.rec_function_definition(expr.function)
        new_bindings: Mapping[str, Array] = constantdict({
            name: _verify_is_array(self.rec(bnd))
            for name, bnd in expr.bindings.items()})
        return expr.replace_if_different(
            function=new_function, bindings=new_bindings)

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        new_call = self.rec(expr._container)
        assert isinstance(new_call, Call)
        # This is OK because:
        # 1) Call.__getitem__ is memoized
        # 2) NamedCallResult isn't allowed to modify tags, etc. (they are
        #    inherited from the call)
        return new_call[expr.name]


class CopyMapperWithExtraArgs(TransformMapperWithExtraArgs[P]):
    """
    Similar to :class:`CopyMapper`, but each mapper method takes extra
    ``*args``, ``**kwargs`` that are propagated along a path by default.

    The logic in :class:`CopyMapper` purposely does not take the extra
    arguments to keep the cost of its each call frame low.
    """
    def rec_size_tuple(self, situp: ShapeType,
                       *args: P.args, **kwargs: P.kwargs
                       ) -> ShapeType:
        new_situp = tuple(
            _verify_is_array(self.rec(s, *args, **kwargs))
            if isinstance(s, Array)
            else s
            for s in situp)
        return situp if _entries_are_identical(new_situp, situp) else new_situp

    def rec_idx_tuple(self, situp: tuple[IndexExpr, ...],
                      *args: P.args, **kwargs: P.kwargs
                      ) -> tuple[IndexExpr, ...]:
        new_situp = tuple(
            _verify_is_array(self.rec(s, *args, **kwargs))
            if isinstance(s, Array)
            else s
            for s in situp)
        return situp if _entries_are_identical(new_situp, situp) else new_situp

    def map_index_lambda(self, expr: IndexLambda,
                         *args: P.args, **kwargs: P.kwargs) -> Array:
        new_shape = self.rec_size_tuple(expr.shape, *args, **kwargs)
        new_bindings: Mapping[str, Array] = constantdict({
                name: _verify_is_array(self.rec(subexpr, *args, **kwargs))
                for name, subexpr in sorted(expr.bindings.items())})
        return expr.replace_if_different(shape=new_shape, bindings=new_bindings)

    def map_placeholder(self,
                        expr: Placeholder, *args: P.args, **kwargs: P.kwargs) -> Array:
        assert expr.name is not None
        new_shape = self.rec_size_tuple(expr.shape, *args, **kwargs)
        return expr.replace_if_different(shape=new_shape)

    def map_stack(self, expr: Stack, *args: P.args, **kwargs: P.kwargs) -> Array:
        new_arrays: tuple[Array, ...] = tuple(
            _verify_is_array(self.rec(arr, *args, **kwargs)) for arr in expr.arrays)
        return expr.replace_if_different(arrays=new_arrays)

    def map_concatenate(self,
                        expr: Concatenate, *args: P.args, **kwargs: P.kwargs) -> Array:
        new_arrays: tuple[Array, ...] = tuple(
            _verify_is_array(self.rec(arr, *args, **kwargs)) for arr in expr.arrays)
        return expr.replace_if_different(arrays=new_arrays)

    def map_roll(self, expr: Roll, *args: P.args, **kwargs: P.kwargs) -> Array:
        new_ary = _verify_is_array(self.rec(expr.array, *args, **kwargs))
        return expr.replace_if_different(array=new_ary)

    def map_axis_permutation(self, expr: AxisPermutation,
                             *args: P.args, **kwargs: P.kwargs) -> Array:
        new_ary = _verify_is_array(self.rec(expr.array, *args, **kwargs))
        return expr.replace_if_different(array=new_ary)

    def _map_index_base(self,
                        expr: IndexBase, *args: P.args, **kwargs: P.kwargs) -> Array:
        assert isinstance(expr, _SuppliedAxesAndTagsMixin)
        new_ary = _verify_is_array(self.rec(expr.array, *args, **kwargs))
        new_indices = self.rec_idx_tuple(expr.indices, *args, **kwargs)
        return expr.replace_if_different(array=new_ary, indices=new_indices)

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
        new_shape = self.rec_size_tuple(expr.shape, *args, **kwargs)
        return expr.replace_if_different(shape=new_shape)

    def map_size_param(self,
                       expr: SizeParam, *args: P.args, **kwargs: P.kwargs) -> Array:
        assert expr.name is not None
        return expr

    def map_einsum(self, expr: Einsum, *args: P.args, **kwargs: P.kwargs) -> Array:
        new_args: tuple[Array, ...] = tuple(
            _verify_is_array(self.rec(arg, *args, **kwargs)) for arg in expr.args)
        return expr.replace_if_different(args=new_args)

    def map_named_array(self,
                        expr: NamedArray, *args: P.args, **kwargs: P.kwargs) -> Array:
        new_container = self.rec(expr._container, *args, **kwargs)
        assert isinstance(new_container, AbstractResultWithNamedArrays)
        return expr.replace_if_different(_container=new_container)

    def map_dict_of_named_arrays(self,
                expr: DictOfNamedArrays, *args: P.args, **kwargs: P.kwargs
            ) -> DictOfNamedArrays:
        new_data = {
            key: _verify_is_array(self.rec(val.expr, *args, **kwargs))
            for key, val in expr.items()}
        return expr.replace_if_different(data=new_data)

    def map_loopy_call(self, expr: LoopyCall,
                       *args: P.args, **kwargs: P.kwargs) -> LoopyCall:
        new_bindings: Mapping[Any, Any] = constantdict(
                    {name: (self.rec(subexpr, *args, **kwargs)
                           if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})
        return expr.replace_if_different(bindings=new_bindings)

    def map_loopy_call_result(self, expr: LoopyCallResult,
                              *args: P.args, **kwargs: P.kwargs) -> Array:
        new_container = self.rec(expr._container, *args, **kwargs)
        assert isinstance(new_container, LoopyCall)
        return expr.replace_if_different(_container=new_container)

    def map_reshape(self, expr: Reshape,
                    *args: P.args, **kwargs: P.kwargs) -> Array:
        new_ary = _verify_is_array(self.rec(expr.array, *args, **kwargs))
        new_newshape = self.rec_size_tuple(expr.newshape, *args, **kwargs)
        return expr.replace_if_different(array=new_ary, newshape=new_newshape)

    def map_distributed_send_ref_holder(self, expr: DistributedSendRefHolder,
                                        *args: P.args, **kwargs: P.kwargs) -> Array:
        new_send_data = _verify_is_array(self.rec(expr.send.data, *args, **kwargs))
        new_passthrough = _verify_is_array(
            self.rec(expr.passthrough_data, *args, **kwargs))
        return expr.replace_if_different(
            send=expr.send.replace_if_different(data=new_send_data),
            passthrough_data=new_passthrough)

    def map_distributed_recv(self, expr: DistributedRecv,
                             *args: P.args, **kwargs: P.kwargs) -> Array:
        new_shape = self.rec_size_tuple(expr.shape, *args, **kwargs)
        return expr.replace_if_different(shape=new_shape)

    def map_function_definition(
                self, expr: FunctionDefinition,
                *args: P.args, **kwargs: P.kwargs
            ) -> FunctionDefinition:
        raise NotImplementedError("Function definitions are purposefully left"
                                  " unimplemented as the default arguments to a new"
                                  " DAG traversal are tricky to guess.")

    def map_call(self, expr: Call,
                 *args: P.args, **kwargs: P.kwargs) -> AbstractResultWithNamedArrays:
        new_function = self.rec_function_definition(expr.function, *args, **kwargs)
        new_bindings: Mapping[str, Array] = constantdict({
            name: _verify_is_array(self.rec(bnd, *args, **kwargs))
            for name, bnd in expr.bindings.items()})
        return expr.replace_if_different(
            function=new_function, bindings=new_bindings)

    def map_named_call_result(self, expr: NamedCallResult,
                              *args: P.args, **kwargs: P.kwargs) -> Array:
        new_call = self.rec(expr._container, *args, **kwargs)
        assert isinstance(new_call, Call)
        # This is OK because:
        # 1) Call.__getitem__ is memoized
        # 2) NamedCallResult isn't allowed to modify tags, etc. (they are
        #    inherited from the call)
        return new_call[expr.name]

# }}}


# {{{ deduplicate

class Deduplicator(CopyMapper):
    """Removes duplicate nodes from an expression."""
    def __init__(
            self,
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        super().__init__(
            err_on_collision=False,
            _cache=_cache,
            _function_cache=_function_cache)

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            _function_cache=cast(
                "TransformMapperCache[FunctionDefinition, []]", self._function_cache))


def deduplicate(
        expr: ArrayOrNamesOrFunctionDefTc
        ) -> ArrayOrNamesOrFunctionDefTc:
    """
    Remove duplicate nodes from an expression.

    .. note::
        Does not remove distinct instances of data wrappers that point to the same
        data (as they will not hash the same). For a utility that does that, see
        :func:`deduplicate_data_wrappers`.
    """
    return Deduplicator()(expr)

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

# FIXME: Might be able to replace this with NodeCollector. Would need to be slightly
# careful, as DependencyMapper excludes DictOfNamedArrays and NodeCollector does not
# (unless specified via collect_fn).
class DependencyMapper(CombineMapper[R, Never]):
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

    def map_call(self, expr: Call) -> R:
        # do not include arrays from the function's body as it would involve
        # putting arrays from different namespaces into the same collection.
        return self.combine(*[self.rec(bnd) for bnd in expr.bindings.values()])

    def map_named_call_result(self, expr: NamedCallResult) -> R:
        return self.rec(expr._container)

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        raise AssertionError("Control shouldn't reach this point.")

# }}}


# {{{ SubsetDependencyMapper

# FIXME: Might be able to implement this as a NodeCollector. Would need to be slightly
# careful, as DependencyMapper excludes DictOfNamedArrays and NodeCollector does not
# (unless specified via collect_fn).
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


# {{{ InputGatherer / ListOfInputsGatherer

class InputGatherer(
        CombineMapper[frozenset[InputArgumentBase], frozenset[InputArgumentBase]]):
    """
    Mapper to combine all instances of :class:`pytato.array.InputArgumentBase` that
    an array expression depends on.
    """
    @override
    def combine(self, *args: frozenset[InputArgumentBase]
                ) -> frozenset[InputArgumentBase]:
        from functools import reduce
        return reduce(
            lambda a, b: a | b,
            args,
            cast("frozenset[InputArgumentBase]", frozenset()))

    @override
    def map_placeholder(self, expr: Placeholder) -> frozenset[InputArgumentBase]:
        return self.combine(frozenset([expr]), super().map_placeholder(expr))

    @override
    def map_data_wrapper(self, expr: DataWrapper) -> frozenset[InputArgumentBase]:
        return self.combine(frozenset([expr]), super().map_data_wrapper(expr))

    def map_size_param(self, expr: SizeParam) -> frozenset[SizeParam]:
        return frozenset([expr])

    @override
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

    @override
    def map_call(self, expr: Call) -> frozenset[InputArgumentBase]:
        return self.combine(self.rec_function_definition(expr.function),
            *[
                self.rec(bnd)
                for _, bnd in sorted(expr.bindings.items())])


class ListOfInputsGatherer(
        CombineMapper[list[InputArgumentBase], list[InputArgumentBase]]):
    """
    Mapper to combine all instances of :class:`pytato.array.InputArgumentBase` that
    an array expression depends on, preserving duplicates.
    """
    @override
    def get_cache_key(self, expr: ArrayOrNames) -> CacheKeyT:
        return id(expr)

    @override
    def get_function_definition_cache_key(self, expr: FunctionDefinition) -> CacheKeyT:
        return id(expr)

    @override
    def combine(self, *args: list[InputArgumentBase]
                ) -> list[InputArgumentBase]:
        id_to_input: dict[int, InputArgumentBase] = {
            id(inp): inp
            for arg in args
            for inp in arg}
        return list(id_to_input.values())

    @override
    def map_placeholder(self, expr: Placeholder) -> list[InputArgumentBase]:
        return self.combine([expr], super().map_placeholder(expr))

    @override
    def map_data_wrapper(self, expr: DataWrapper) -> list[InputArgumentBase]:
        return self.combine([expr], super().map_data_wrapper(expr))

    def map_size_param(self, expr: SizeParam) -> list[SizeParam]:
        return [expr]

    @override
    def map_function_definition(self, expr: FunctionDefinition
                                ) -> list[InputArgumentBase]:
        # get rid of placeholders local to the function.
        new_mapper = ListOfInputsGatherer()
        all_callee_inputs = new_mapper.combine(*[new_mapper(ret)
                                                 for ret in expr.returns.values()])
        result: list[InputArgumentBase] = []
        for inp in all_callee_inputs:
            if isinstance(inp, Placeholder):
                if inp.name in expr.parameters:
                    # drop, reference to argument
                    pass
                else:
                    raise ValueError("function definition refers to non-argument "
                                     f"placeholder named '{inp.name}'")
            else:
                result.append(inp)

        return result

    @override
    def map_call(self, expr: Call) -> list[InputArgumentBase]:
        return self.combine(self.rec_function_definition(expr.function),
            *[
                self.rec(bnd)
                for _, bnd in sorted(expr.bindings.items())])

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

    def visit(
            self, expr: ArrayOrNames | FunctionDefinition,
            *args: P.args, **kwargs: P.kwargs) -> bool:
        """
        If this method returns *True*, *expr* is traversed during the walk.
        If this method returns *False*, *expr* is not traversed as a part of
        the walk.
        """
        return True

    def post_visit(
            self, expr: ArrayOrNames | FunctionDefinition,
            *args: P.args, **kwargs: P.kwargs) -> None:
        """
        Callback after *expr* has been traversed.
        """

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

    def post_visit(self, expr: ArrayOrNames | FunctionDefinition) -> None:
        if isinstance(expr, Array):
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
            return self._cache_retrieve(inputs)
        except KeyError:
            # Intentionally going to Mapper instead of super() to avoid
            # double caching when subclasses of CachedMapper override rec,
            # see https://github.com/inducer/pytato/pull/585
            return self._cache_add(inputs, Mapper.rec(self, self.map_fn(expr)))

# }}}


# {{{ MapOnceMapper

# FIXME: optimize_mapper?
class MapOnceMapper(Mapper[ResultT, FunctionResultT, [set[VisitKeyT] | None]]):
    """
    Mapper that maps each node to a result only once per call (excepting duplicates
    and different function contexts, depending on configuration). Each time the
    mapper visits a node after the first visit, the value *no_result_value* or
    *no_function_result_value* is returned instead. This is intended to be used as a
    base class for mappers that must avoid double counting of results.

    .. attribute:: no_result_value

        The value to return for nodes that have already been visited.

    .. attribute:: no_function_result_value

        The value to return for function nodes that have already been visited.

    .. attribute:: map_duplicates

        Controls whether duplicate nodes (equal nodes with different `id()`) are to
        be mapped separately or treated as the same.

    .. attribute:: map_in_different_functions

        Controls whether nodes in different function call contexts are to be mapped
        separately or treated as the same.
    """
    def __init__(
            self,
            no_result_value: ResultT,
            no_function_result_value: FunctionResultT,
            map_duplicates: bool,
            map_in_different_functions: bool,
            ) -> None:
        super().__init__()

        self.no_result_value: ResultT = no_result_value
        self.no_function_result_value: FunctionResultT = no_function_result_value

        self.map_duplicates: bool = map_duplicates
        self.map_in_different_functions: bool = map_in_different_functions

    def get_visit_key(self, expr: ArrayOrNames) -> VisitKeyT:
        return id(expr) if self.map_duplicates else expr

    def get_function_definition_visit_key(self, expr: FunctionDefinition) -> VisitKeyT:
        return id(expr) if self.map_duplicates else expr

    @override
    def rec(
            self,
            expr: ArrayOrNames,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        assert visited_node_keys is not None

        key = self.get_visit_key(expr)

        if key in visited_node_keys:
            return self.no_result_value

        # Intentionally going to Mapper instead of super() to avoid
        # double visiting when subclasses of MapOnceMapper override rec,
        # see https://github.com/inducer/pytato/pull/585
        result = Mapper.rec(self, expr, visited_node_keys)

        visited_node_keys.add(key)
        return result

    @override
    def rec_function_definition(
            self,
            expr: FunctionDefinition,
            visited_node_keys: set[VisitKeyT] | None) -> FunctionResultT:
        assert visited_node_keys is not None

        key = self.get_function_definition_visit_key(expr)

        if key in visited_node_keys:
            return self.no_function_result_value

        # Intentionally going to Mapper instead of super() to avoid
        # double visiting when subclasses of MapOnceMapper override rec,
        # see https://github.com/inducer/pytato/pull/585
        result = Mapper.rec_function_definition(self, expr, visited_node_keys)

        visited_node_keys.add(key)
        return result

    def rec_idx_or_size_tuple(
            self,
            situp: tuple[IndexOrShapeExpr, ...],
            visited_node_keys: set[VisitKeyT] | None) -> tuple[ResultT, ...]:
        return tuple(
            self.rec(s, visited_node_keys)
            for s in situp if isinstance(s, Array))

    def get_visited_node_keys_for_callee(
            self, visited_node_keys: set[VisitKeyT] | None) -> set[VisitKeyT]:
        assert visited_node_keys is not None
        return (
            visited_node_keys
            if not self.map_in_different_functions
            else set())

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            no_result_value=self.no_result_value,
            no_function_result_value=self.no_function_result_value,
            map_duplicates=self.map_duplicates,
            map_in_different_functions=self.map_in_different_functions)

    @overload
    def __call__(
            self,
            expr: ArrayOrNames,
            visited_node_keys: set[VisitKeyT] | None = None,
            ) -> ResultT:
        ...

    @overload
    def __call__(
            self,
            expr: FunctionDefinition,
            visited_node_keys: set[VisitKeyT] | None = None,
            ) -> FunctionResultT:
        ...

    @override
    def __call__(
            self,
            expr: ArrayOrNames | FunctionDefinition,
            visited_node_keys: set[VisitKeyT] | None = None,
            ) -> ResultT | FunctionResultT:
        if visited_node_keys is None:
            visited_node_keys = set()

        return super().__call__(expr, visited_node_keys)

# }}}


# {{{ MapAndReduceMapper

# FIXME: basedpyright doesn't like this
# @optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class MapAndReduceMapper(MapOnceMapper[ResultT, ResultT]):
    """
    Mapper that, for each node, calls *map_fn* on that node and then calls
    *reduce_fn* on the output together with the already-reduced results from its
    predecessors.

    .. note::

        Each node is mapped/reduced only the first time it is visited (excepting
        duplicates and different function contexts, depending on configuration).
        On subsequent visits, the returned result is `reduce_fn()`.

    .. attribute:: traverse_functions

        Controls whether or not the mapper should descend into function definitions.

    .. attribute:: map_duplicates

        Controls whether duplicate nodes (equal nodes with different `id()`) are to
        be mapped separately or treated as the same.

    .. attribute:: map_in_different_functions

        Controls whether nodes in different function call contexts are to be mapped
        separately or treated as the same.
    """
    def __init__(
            self,
            map_fn: Callable[[ArrayOrNames | FunctionDefinition], ResultT],
            # FIXME: Is there a way to make argument type annotation more specific?
            reduce_fn: Callable[..., ResultT],
            traverse_functions: bool = True,
            map_duplicates: bool = False,
            map_in_different_functions: bool = True,
            ) -> None:
        super().__init__(
            no_result_value=reduce_fn(),
            no_function_result_value=reduce_fn(),
            map_duplicates=map_duplicates,
            map_in_different_functions=map_in_different_functions)

        self.map_fn: Callable[[ArrayOrNames | FunctionDefinition], ResultT] = map_fn
        self.reduce_fn: Callable[..., ResultT] = reduce_fn

        self.traverse_functions: bool = traverse_functions

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            map_fn=self.map_fn,
            reduce_fn=self.reduce_fn,
            traverse_functions=self.traverse_functions,
            map_duplicates=self.map_duplicates,
            map_in_different_functions=self.map_in_different_functions)

    def map_index_lambda(
            self,
            expr: IndexLambda,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *self.rec_idx_or_size_tuple(expr.shape, visited_node_keys),
            *(
                self.rec(subexpr, visited_node_keys)
                for subexpr in expr.bindings.values()))

    def map_placeholder(
            self,
            expr: Placeholder,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *self.rec_idx_or_size_tuple(expr.shape, visited_node_keys))

    def map_stack(
            self,
            expr: Stack,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *(self.rec(arr, visited_node_keys) for arr in expr.arrays))

    def map_concatenate(
            self,
            expr: Concatenate,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *(self.rec(arr, visited_node_keys) for arr in expr.arrays))

    def map_roll(
            self,
            expr: Roll,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec(expr.array, visited_node_keys))

    def map_axis_permutation(
            self,
            expr: AxisPermutation,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec(expr.array, visited_node_keys))

    def _map_index_base(
            self,
            expr: IndexBase,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec(expr.array, visited_node_keys),
            *self.rec_idx_or_size_tuple(expr.indices, visited_node_keys))

    def map_basic_index(
            self,
            expr: BasicIndex,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self._map_index_base(expr, visited_node_keys)

    def map_contiguous_advanced_index(
            self,
            expr: AdvancedIndexInContiguousAxes,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self._map_index_base(expr, visited_node_keys)

    def map_non_contiguous_advanced_index(
            self,
            expr: AdvancedIndexInNoncontiguousAxes,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self._map_index_base(expr, visited_node_keys)

    def map_data_wrapper(
            self,
            expr: DataWrapper,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *self.rec_idx_or_size_tuple(expr.shape, visited_node_keys))

    def map_size_param(
            self,
            expr: SizeParam,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(self.map_fn(expr))

    def map_einsum(
            self,
            expr: Einsum,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *(self.rec(arg, visited_node_keys) for arg in expr.args))

    def map_named_array(
            self,
            expr: NamedArray,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec(expr._container, visited_node_keys))

    def map_dict_of_named_arrays(
            self,
            expr: DictOfNamedArrays,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *(self.rec(val.expr, visited_node_keys) for val in expr.values()))

    def map_loopy_call(
            self,
            expr: LoopyCall,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *(
                self.rec(subexpr, visited_node_keys)
                for subexpr in expr.bindings.values()
                if isinstance(subexpr, Array)))

    def map_loopy_call_result(
            self,
            expr: LoopyCallResult,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec(expr._container, visited_node_keys))

    def map_reshape(
            self,
            expr: Reshape,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec(expr.array, visited_node_keys),
            *self.rec_idx_or_size_tuple(expr.newshape, visited_node_keys))

    def map_distributed_send_ref_holder(
            self,
            expr: DistributedSendRefHolder,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec(expr.send.data, visited_node_keys),
            self.rec(expr.passthrough_data, visited_node_keys))

    def map_distributed_recv(
            self,
            expr: DistributedRecv,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            *self.rec_idx_or_size_tuple(expr.shape, visited_node_keys))

    def map_function_definition(
            self,
            expr: FunctionDefinition,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        if not self.traverse_functions:
            return self.no_function_result_value

        new_mapper = self.clone_for_callee(expr)
        new_visited_node_keys = self.get_visited_node_keys_for_callee(
            visited_node_keys)
        return self.reduce_fn(
            self.map_fn(expr),
            *(
                new_mapper(ret, new_visited_node_keys)
                for ret in expr.returns.values()))

    def map_call(
            self,
            expr: Call,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec_function_definition(expr.function, visited_node_keys),
            *(self.rec(bnd, visited_node_keys) for bnd in expr.bindings.values()))

    def map_named_call_result(
            self,
            expr: NamedCallResult,
            visited_node_keys: set[VisitKeyT] | None) -> ResultT:
        return self.reduce_fn(
            self.map_fn(expr),
            self.rec(expr._container, visited_node_keys))

# }}}


# {{{ NodeCollector

NodeSet: TypeAlias = frozenset[ArrayOrNames | FunctionDefinition]


# FIXME: optimize_mapper?
class NodeCollector(MapAndReduceMapper[NodeSet]):
    """
    Return the set of nodes in a DAG that match a condition specified via
    *collect_fn*.

    .. attribute:: traverse_functions

        Controls whether or not the mapper should descend into function definitions.
    """
    def __init__(
            self,
            collect_fn: Callable[[ArrayOrNames | FunctionDefinition], bool],
            traverse_functions: bool = True) -> None:
        from functools import reduce
        super().__init__(
            map_fn=lambda expr: (
                frozenset([expr]) if collect_fn(expr) else frozenset()),
            reduce_fn=lambda *args: reduce(
                cast("Callable[[NodeSet, NodeSet], NodeSet]", frozenset.union),
                args,
                cast("NodeSet", frozenset())),
            traverse_functions=traverse_functions,
            map_duplicates=False,
            map_in_different_functions=False)

        self.collect_fn: Callable[[ArrayOrNames | FunctionDefinition], bool] = \
            collect_fn

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            collect_fn=self.collect_fn,
            traverse_functions=self.traverse_functions)

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


def map_and_copy(expr: ArrayOrNamesTc,
                 map_fn: Callable[[ArrayOrNames], ArrayOrNames]
                 ) -> ArrayOrNamesTc:
    """
    Returns a copy of *expr* with every array expression reachable from *expr*
    mapped via *map_fn*.

    .. note::

        Uses :class:`CachedMapAndCopyMapper` under the hood and because of its
        caching nature each node is mapped exactly once.
    """
    return CachedMapAndCopyMapper(map_fn)(expr)


def map_and_reduce(
        expr: ArrayOrNames | FunctionDefinition,
        map_fn: Callable[[ArrayOrNames | FunctionDefinition], ResultT],
        # FIXME: Is there a way to make argument type annotation more specific?
        reduce_fn: Callable[..., ResultT],
        traverse_functions: bool = True,
        map_duplicates: bool = False,
        map_in_different_functions: bool = True) -> ResultT:
    """
    For each node, call *map_fn* on that node and then call *reduce_fn* on the output
    together with the already-reduced results from its predecessors.

    See :class:`MapAndReduceMapper` for more details.
    """
    mrm = MapAndReduceMapper(
        map_fn=map_fn,
        reduce_fn=reduce_fn,
        traverse_functions=traverse_functions,
        map_duplicates=map_duplicates,
        map_in_different_functions=map_in_different_functions)
    return mrm(expr)


def collect_nodes(
        expr: ArrayOrNames | FunctionDefinition,
        collect_fn: Callable[[ArrayOrNames | FunctionDefinition], bool],
        traverse_functions: bool = True
        ) -> frozenset[ArrayOrNames | FunctionDefinition]:
    """
    Return the set of nodes in a DAG that match a condition specified via
    *collect_fn*.

    See :class:`NodeCollector` for more details.
    """
    nc = NodeCollector(
        collect_fn=collect_fn,
        traverse_functions=traverse_functions)
    return nc(expr)


def materialize_with_mpms(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    from warnings import warn
    warn(
        "pytato.transform.materialize_with_mpms is deprecated and will be removed in "
        "2025. Use pytato.transform.materialize.materialize_with_mpms instead.",
        DeprecationWarning, stacklevel=2)

    from pytato.transform.materialize import (
        materialize_with_mpms as new_materialize_with_mpms,
    )
    return new_materialize_with_mpms(expr)

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
                if isinstance(base_data, MemoryObjectHolder):
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

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            _function_cache=cast(
                "TransformMapperCache[FunctionDefinition, []]", self._function_cache))


def deduplicate_data_wrappers(
            array_or_names: ArrayOrNamesOrFunctionDefTc
        ) -> ArrayOrNamesOrFunctionDefTc:
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
