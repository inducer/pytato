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
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    FrozenSet,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

import attrs
import numpy as np
from immutabledict import immutabledict

from pymbolic.mapper.optimize import optimize_mapper

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
    ShapeType,
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


ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]
MappedT = TypeVar("MappedT",
                  Array, AbstractResultWithNamedArrays, ArrayOrNames)
CombineT = TypeVar("CombineT")  # used in CombineMapper
TransformMapperResultT = TypeVar("TransformMapperResultT",  # used in TransformMapper
                            Array, AbstractResultWithNamedArrays, ArrayOrNames)
CacheExprT = TypeVar("CacheExprT")  # used in CachedMapperCache
CacheKeyT = TypeVar("CacheKeyT")  # used in CachedMapperCache
CacheResultT = TypeVar("CacheResultT")  # used in CachedMapperCache
CachedMapperT = TypeVar("CachedMapperT")  # used in CachedMapper
CachedMapperFunctionT = TypeVar("CachedMapperFunctionT")  # used in CachedMapper
IndexOrShapeExpr = TypeVar("IndexOrShapeExpr")
R = FrozenSet[Array]
_SelfMapper = TypeVar("_SelfMapper", bound="Mapper")

__doc__ = """
.. currentmodule:: pytato.transform

.. autoclass:: Mapper
.. autoclass:: CachedMapperCache
.. autoclass:: CachedMapper
.. autoclass:: TransformMapper
.. autoclass:: TransformMapperWithExtraArgs
.. autoclass:: CopyMapper
.. autoclass:: CopyMapperWithExtraArgs
.. autoclass:: Deduplicator
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

.. class:: MappedT

    A type variable representing the input type of a :class:`Mapper`.

.. class:: CombineT

    A type variable representing the type of a :class:`CombineMapper`.

.. class:: CachedMapperFunctionT

    A type variable used to represent the output type of a :class:`CachedMapper`
    for :class:`pytato.function.FunctionDefinition`.

.. class:: _SelfMapper

    A type variable used to represent the type of a mapper in
    :meth:`CachedMapper.clone_for_callee`.
"""

transform_logger = logging.getLogger(__file__)


class UnsupportedArrayError(ValueError):
    pass


class CacheCollisionError(ValueError):
    pass


class CacheNoOpDuplicationError(ValueError):
    pass


# {{{ mapper base class

class Mapper:
    """A class that when called with a :class:`pytato.Array` recursively
    iterates over the DAG, calling the *_mapper_method* of each node. Users of
    this class are expected to override the methods of this class or create a
    subclass.

    .. note::

       This class might visit a node multiple times. Use a :class:`CachedMapper`
       if this is not desired.

    .. automethod:: handle_unsupported_array
    .. automethod:: map_foreign
    .. automethod:: rec
    .. automethod:: __call__
    """

    def handle_unsupported_array(self, expr: MappedT,
                                 *args: Any, **kwargs: Any) -> Any:
        """Mapper method that is invoked for
        :class:`pytato.Array` subclasses for which a mapper
        method does not exist in this mapper.
        """
        raise UnsupportedArrayError(
                f"{type(self).__name__} cannot handle expressions of type {type(expr)}")

    def map_foreign(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        """Mapper method that is invoked for an object of class for which a
        mapper method does not exist in this mapper.
        """
        raise ValueError(
                f"{type(self).__name__} encountered invalid foreign object: {expr!r}")

    def rec(self, expr: MappedT, *args: Any, **kwargs: Any) -> Any:
        """Call the mapper method of *expr* and return the result."""
        method: Callable[..., Any] | None

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
                return self.map_foreign(expr, *args, **kwargs)

        assert method is not None
        return method(expr, *args, **kwargs)

    def rec_function_definition(
            self, expr: FunctionDefinition, *args: Any, **kwargs: Any
            ) -> Any:
        """Call the mapper method of *expr* and return the result."""
        method: Callable[..., Any] | None

        try:
            method = self.map_function_definition  # type: ignore[attr-defined]
        except AttributeError:
            return self.map_foreign(expr, *args, **kwargs)

        assert method is not None
        return method(expr, *args, **kwargs)

    def __call__(self, expr: MappedT, *args: Any, **kwargs: Any) -> Any:
        """Handle the mapping of *expr*."""
        return self.rec(expr, *args, **kwargs)

# }}}


# {{{ CachedMapper

class CachedMapperCache(Generic[CacheExprT, CacheKeyT, CacheResultT]):
    """
    Cache for :class:`CachedMapper`.

    .. automethod:: __init__
    .. automethod:: get_key
    .. automethod:: add
    .. automethod:: retrieve
    """
    def __init__(
            self,
            key_func: Callable[[CacheExprT], CacheKeyT],
            err_on_collision: bool) -> None:
        """
        Initialize the cache.

        :arg key_func: Function to compute a hashable cache key from an input
            expression.
        :arg err_on_collision: Raise an exception if two distinct input expression
            instances have the same key.
        """
        self.err_on_collision = err_on_collision
        self._key_func = key_func
        self._expr_key_to_result: dict[CacheKeyT, CacheResultT] = {}
        if self.err_on_collision:
            self._expr_key_to_expr: dict[CacheKeyT, CacheExprT] = {}

    # FIXME: Can this be inlined?
    def get_key(self, expr: CacheExprT) -> CacheKeyT:
        """Compute the key for an input expression."""
        return self._key_func(expr)

    def add(
            self,
            expr: CacheExprT,
            result: CacheResultT,
            key: CacheKeyT | None = None) -> CacheResultT:
        """
        Cache a mapping result.

        Returns the cached result.
        """
        if key is None:
            key = self._key_func(expr)

        assert key not in self._expr_key_to_result, \
            "Cache entry is already present for this key."

        self._expr_key_to_result[key] = result
        if self.err_on_collision:
            self._expr_key_to_expr[key] = expr

        return result

    def retrieve(
            self,
            expr: CacheExprT,
            key: CacheKeyT | None = None) -> CacheResultT:
        """Retrieve the cached mapping result."""
        if key is None:
            key = self._key_func(expr)

        result = self._expr_key_to_result[key]

        if self.err_on_collision:
            if expr is not self._expr_key_to_expr[key]:
                raise CacheCollisionError

        return result


class CachedMapper(Mapper, Generic[CachedMapperT, CachedMapperFunctionT]):
    """Mapper class that maps each node in the DAG exactly once. This loses some
    information compared to :class:`Mapper` as a node is visited only from
    one of its predecessors.

    .. automethod:: clone_for_callee
    """
    # Not sure if there's a way to simplify this stuff?
    _CacheType: type[Any] = CachedMapperCache[
        ArrayOrNames,
        Hashable,
        CachedMapperT]
    _OtherCachedMapperT = TypeVar("_OtherCachedMapperT")
    _CacheT: TypeAlias = CachedMapperCache[
        ArrayOrNames,
        Hashable,
        _OtherCachedMapperT]

    _FunctionCacheType: type[Any] = CachedMapperCache[
        FunctionDefinition,
        Hashable,
        CachedMapperFunctionT]
    _OtherCachedMapperFunctionT = TypeVar("_OtherCachedMapperFunctionT")
    _FunctionCacheT: TypeAlias = CachedMapperCache[
        FunctionDefinition,
        Hashable,
        _OtherCachedMapperFunctionT]

    def __init__(
            self,
            err_on_collision: bool | None = None,
            # Arrays are cached separately for each call stack frame, but
            # functions are cached globally
            _function_cache: _FunctionCacheT[CachedMapperFunctionT] | None = None
            ) -> None:
        super().__init__()

        if err_on_collision is None:
            err_on_collision = __debug__

        self._cache: CachedMapper._CacheT[CachedMapperT] = \
            CachedMapper._CacheType(
                lambda expr: expr,
                err_on_collision=err_on_collision)

        if _function_cache is None:
            _function_cache = CachedMapper._FunctionCacheType(
                lambda expr: expr,
                err_on_collision=err_on_collision)

        self._function_cache: CachedMapper._FunctionCacheT[CachedMapperFunctionT] = \
            _function_cache

    def _cache_add(
            self,
            expr: ArrayOrNames,
            result: CachedMapperT,
            key: Hashable | None = None) -> CachedMapperT:
        return self._cache.add(expr, result, key=key)

    def _function_cache_add(
            self,
            expr: FunctionDefinition,
            result: CachedMapperFunctionT,
            key: Hashable | None = None) -> CachedMapperFunctionT:
        return self._function_cache.add(expr, result, key=key)

    def _cache_retrieve(
            self,
            expr: ArrayOrNames,
            key: Hashable | None = None) -> CachedMapperT:
        try:
            return self._cache.retrieve(expr, key=key)
        except CacheCollisionError as e:
            raise ValueError(
                f"cache collision detected on {type(expr)} in {type(self)}.") from e

    def _function_cache_retrieve(
            self,
            expr: FunctionDefinition,
            key: Hashable | None = None) -> CachedMapperFunctionT:
        try:
            return self._function_cache.retrieve(expr, key=key)
        except CacheCollisionError as e:
            raise ValueError(
                f"cache collision detected on {type(expr)} in {type(self)}.") from e

    def rec(self, expr: ArrayOrNames) -> CachedMapperT:
        key = self._cache.get_key(expr)
        try:
            return self._cache_retrieve(expr, key=key)
        except KeyError:
            return self._cache_add(expr, super().rec(expr), key=key)

    def rec_function_definition(
            self, expr: FunctionDefinition) -> CachedMapperFunctionT:
        key = self._function_cache.get_key(expr)
        try:
            return self._function_cache_retrieve(expr, key=key)
        except KeyError:
            return self._function_cache_add(
                expr, super().rec_function_definition(expr), key=key)

    if TYPE_CHECKING:
        def __call__(self, expr: ArrayOrNames) -> CachedMapperT:
            return self.rec(expr)

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        # type-ignore-reason: self.__init__ has a different function signature
        # than Mapper.__init__
        return type(self)(  # type: ignore[call-arg]
            err_on_collision=self._cache.err_on_collision,  # type: ignore[attr-defined]
            _function_cache=self._function_cache)  # type: ignore[attr-defined]

# }}}


# {{{ TransformMapper

class TransformMapperCache(CachedMapperCache[CacheExprT, CacheKeyT, CacheExprT]):
    """
    Cache for :class:`TransformMapper`.

    .. automethod:: __init__
    .. automethod:: add
    """
    def __init__(
            self,
            key_func: Callable[[CacheExprT], CacheKeyT],
            err_on_collision: bool,
            err_on_no_op_duplication: bool) -> None:
        """
        Initialize the cache.

        :arg key_func: Function to compute a hashable cache key from an input
            expression.
        :arg err_on_collision: Raise an exception if two distinct input expression
            instances have the same key.
        :arg err_on_no_op_duplication: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        super().__init__(key_func, err_on_collision=err_on_collision)

        self.err_on_no_op_duplication = err_on_no_op_duplication

        self._result_key_to_result: dict[CacheKeyT, CacheExprT] = {}

    def add(
            self,
            expr: CacheExprT,
            result: CacheExprT,
            key: CacheKeyT | None = None,
            result_key: CacheKeyT | None = None) -> CacheExprT:
        """
        Cache a mapping result.

        Returns the cached result (which may not be identical to *result* if a
        result was already cached with the same result key).
        """
        if key is None:
            key = self._key_func(expr)
        if result_key is None:
            result_key = self._key_func(result)

        assert key not in self._expr_key_to_result, \
            "Cache entry is already present for this key."

        try:
            result = self._result_key_to_result[result_key]
        except KeyError:
            if (
                    self.err_on_no_op_duplication
                    and hash(result_key) == hash(key)
                    and result_key == key
                    and result is not expr
                    # This is questionable in two ways:
                    # 1) It will not detect duplication of things that are not
                    #    considered direct predecessors (e.g. a Call's
                    #    FunctionDefinition). Not sure how to handle such cases
                    # 2) DirectPredecessorsGetter doesn't accept FunctionDefinitions,
                    #    but CacheExprT is allowed to be one
                    and all(
                        result_pred is pred
                        for pred, result_pred in zip(
                            DirectPredecessorsGetter()(expr),
                            DirectPredecessorsGetter()(result)))):
                raise CacheNoOpDuplicationError from None

            self._result_key_to_result[result_key] = result

        self._expr_key_to_result[key] = result
        if self.err_on_collision:
            self._expr_key_to_expr[key] = expr

        return result


class TransformMapper(CachedMapper[ArrayOrNames, FunctionDefinition]):
    """Base class for mappers that transform :class:`pytato.array.Array`\\ s into
    other :class:`pytato.array.Array`\\ s.

    Enables certain operations that can only be done if the mapping results are also
    arrays (e.g., computing a cache key from them). Does not implement default
    mapper methods; for that, see :class:`CopyMapper`.

    .. automethod:: __init__
    .. automethod:: clone_for_callee
    """
    _CacheType: type[Any] = TransformMapperCache[ArrayOrNames, Hashable]
    _CacheT: TypeAlias = TransformMapperCache[ArrayOrNames, Hashable]

    _FunctionCacheType: type[Any] = TransformMapperCache[
        FunctionDefinition, Hashable]
    _FunctionCacheT: TypeAlias = TransformMapperCache[
        FunctionDefinition, Hashable]

    if TYPE_CHECKING:
        def rec(self, expr: TransformMapperResultT) -> TransformMapperResultT:
            return cast(TransformMapperResultT, super().rec(expr))

        def __call__(self, expr: TransformMapperResultT) -> TransformMapperResultT:
            return self.rec(expr)

    def __init__(
            self,
            err_on_collision: bool | None = None,
            err_on_no_op_duplication: bool | None = None,
            _function_cache: _FunctionCacheT | None = None
            ) -> None:
        """
        :arg err_on_collision: Raise an exception if two distinct input array
            instances have the same key.
        :arg err_on_no_op_duplication: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        if err_on_collision is None:
            err_on_collision = __debug__
        if err_on_no_op_duplication is None:
            err_on_no_op_duplication = __debug__

        if _function_cache is None:
            _function_cache = TransformMapper._FunctionCacheType(
                lambda expr: expr,
                err_on_collision=err_on_collision,
                err_on_no_op_duplication=err_on_no_op_duplication)

        super().__init__(
            err_on_collision=err_on_collision,
            _function_cache=_function_cache)

        self._cache: TransformMapper._CacheT = TransformMapper._CacheType(
            lambda expr: expr,
            err_on_collision=err_on_collision,
            err_on_no_op_duplication=err_on_no_op_duplication)

        self._function_cache: TransformMapper._FunctionCacheT = self._function_cache

    def _cache_add(
            self,
            expr: TransformMapperResultT,
            result: TransformMapperResultT,
            key: Hashable | None = None) -> TransformMapperResultT:
        try:
            return self._cache.add(expr, result, key=key)  # type: ignore[return-value]
        except CacheNoOpDuplicationError as e:
            raise ValueError(
                f"no-op duplication detected on {type(expr)} in "
                f"{type(self)}.") from e

    def _function_cache_add(
            self,
            expr: FunctionDefinition,
            result: FunctionDefinition,
            key: Hashable | None = None) -> FunctionDefinition:
        try:
            return self._function_cache.add(expr, result, key=key)
        except CacheNoOpDuplicationError as e:
            raise ValueError(
                f"no-op duplication detected on {type(expr)} in "
                f"{type(self)}.") from e

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        # type-ignore-reason: self.__init__ has a different function signature
        # than Mapper.__init__
        return type(self)(  # type: ignore[call-arg]
            err_on_collision=self._cache.err_on_collision,  # type: ignore[attr-defined]
            err_on_no_op_duplication=self._cache.err_on_no_op_duplication,  # type: ignore[attr-defined]
            _function_cache=self._function_cache)  # type: ignore[attr-defined]

# }}}


# {{{ TransformMapperWithExtraArgs

class TransformMapperWithExtraArgsCache(
        CachedMapperCache[CacheExprT, CacheKeyT, CacheExprT]):
    """
    Cache for :class:`TransformMapperWithExtraArgs`.

    .. automethod:: __init__
    .. automethod:: get_key
    .. automethod:: add
    .. automethod:: retrieve
    """
    def __init__(
            self,
            key_func: Callable[..., CacheKeyT],
            err_on_collision: bool,
            err_on_no_op_duplication: bool) -> None:
        """
        Initialize the cache.

        :arg key_func: Function to compute a hashable cache key from an input
            expression and extra arguments.
        :arg err_on_collision: Raise an exception if two distinct input expression
            instances have the same key.
        :arg err_on_no_op_duplication: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        super().__init__(key_func, err_on_collision=err_on_collision)

        self.err_on_no_op_duplication = err_on_no_op_duplication

        self._result_key_to_result: dict[CacheKeyT, CacheExprT] = {}

    def get_key(self, expr: CacheExprT, *args: Any, **kwargs: Any) -> CacheKeyT:
        """Compute the key for an input expression."""
        return self._key_func(expr, *args, **kwargs)

    def add(  # type: ignore[override]
            self,
            expr: CacheExprT,
            key_args: tuple[Any, ...],
            key_kwargs: dict[str, Any],
            result: CacheExprT,
            key: CacheKeyT | None = None,
            result_key: CacheKeyT | None = None) -> CacheExprT:
        """
        Cache a mapping result.

        Returns the cached result (which may not be identical to *result* if a
        result was already cached with the same result key).
        """
        if key is None:
            key = self._key_func(expr, *key_args, **key_kwargs)
        if result_key is None:
            result_key = self._key_func(result, *key_args, **key_kwargs)

        assert key not in self._expr_key_to_result, \
            "Cache entry is already present for this key."

        try:
            result = self._result_key_to_result[result_key]
        except KeyError:
            if (
                    self.err_on_no_op_duplication
                    and hash(result_key) == hash(key)
                    and result_key == key
                    and result is not expr
                    # This is questionable in two ways:
                    # 1) It will not detect duplication of things that are not
                    #    considered direct predecessors (e.g. a Call's
                    #    FunctionDefinition). Not sure how to handle such cases
                    # 2) DirectPredecessorsGetter doesn't accept FunctionDefinitions,
                    #    but CacheExprT is allowed to be one
                    and all(
                        result_pred is pred
                        for pred, result_pred in zip(
                            DirectPredecessorsGetter()(expr),
                            DirectPredecessorsGetter()(result)))):
                raise CacheNoOpDuplicationError from None

            self._result_key_to_result[result_key] = result

        self._expr_key_to_result[key] = result
        if self.err_on_collision:
            self._expr_key_to_expr[key] = expr

        return result

    def retrieve(  # type: ignore[override]
            self,
            expr: CacheExprT,
            key_args: tuple[Any, ...],
            key_kwargs: dict[str, Any],
            key: CacheKeyT | None = None) -> CacheExprT:
        """Retrieve the cached mapping result."""
        if key is None:
            key = self._key_func(expr, *key_args, **key_kwargs)

        result = self._expr_key_to_result[key]

        if self.err_on_collision:
            if expr is not self._expr_key_to_expr[key]:
                raise CacheCollisionError

        return result


class TransformMapperWithExtraArgs(CachedMapper[ArrayOrNames, FunctionDefinition]):
    """
    Similar to :class:`TransformMapper`, but each mapper method takes extra
    ``*args``, ``**kwargs`` that are propagated along a path by default.

    The logic in :class:`TransformMapper` purposely does not take the extra
    arguments to keep the cost of its each call frame low.

    .. automethod:: __init__
    .. automethod:: clone_for_callee
    """
    _CacheType: type[Any] = TransformMapperWithExtraArgsCache[
        ArrayOrNames, Hashable]
    _CacheT: TypeAlias = TransformMapperWithExtraArgsCache[
        ArrayOrNames, Hashable]

    _FunctionCacheType: type[Any] = TransformMapperWithExtraArgsCache[
        FunctionDefinition, Hashable]
    _FunctionCacheT: TypeAlias = TransformMapperWithExtraArgsCache[
        FunctionDefinition, Hashable]

    if TYPE_CHECKING:
        def __call__(
                self, expr: TransformMapperResultT, *args: Any, **kwargs: Any
                ) -> TransformMapperResultT:
            return self.rec(expr, *args, **kwargs)

    def __init__(
            self,
            err_on_collision: bool | None = None,
            err_on_no_op_duplication: bool | None = None,
            _function_cache: _FunctionCacheT | None = None
            ) -> None:
        """
        :arg err_on_collision: Raise an exception if two distinct input array
            instances have the same key.
        :arg err_on_no_op_duplication: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        if err_on_collision is None:
            err_on_collision = __debug__
        if err_on_no_op_duplication is None:
            err_on_no_op_duplication = __debug__

        def key_func(
                expr: ArrayOrNames | FunctionDefinition,
                *args: Any, **kwargs: Any) -> Hashable:
            return (expr, args, tuple(sorted(kwargs.items())))

        if _function_cache is None:
            _function_cache = TransformMapperWithExtraArgs._FunctionCacheType(
                key_func,
                err_on_collision=err_on_collision,
                err_on_no_op_duplication=err_on_no_op_duplication)

        super().__init__(
            err_on_collision=err_on_collision,
            _function_cache=_function_cache)

        self._cache: TransformMapperWithExtraArgs._CacheT = \
            TransformMapperWithExtraArgs._CacheType(
                key_func,
                err_on_collision=err_on_collision,
                err_on_no_op_duplication=err_on_no_op_duplication)

        self._function_cache: TransformMapperWithExtraArgs._FunctionCacheT = \
            self._function_cache

    def _cache_add(  # type: ignore[override]
            self,
            expr: TransformMapperResultT,
            key_args: tuple[Any, ...],
            key_kwargs: dict[str, Any],
            result: TransformMapperResultT,
            key: Hashable | None = None) -> TransformMapperResultT:
        try:
            return self._cache.add(expr, key_args, key_kwargs, result, key=key)  # type: ignore[return-value]
        except CacheNoOpDuplicationError as e:
            raise ValueError(
                f"no-op duplication detected on {type(expr)} in "
                f"{type(self)}.") from e

    def _function_cache_add(  # type: ignore[override]
            self,
            expr: FunctionDefinition,
            key_args: tuple[Any, ...],
            key_kwargs: dict[str, Any],
            result: FunctionDefinition,
            key: Hashable | None = None) -> FunctionDefinition:
        try:
            return self._function_cache.add(
                expr, key_args, key_kwargs, result, key=key)
        except CacheNoOpDuplicationError as e:
            raise ValueError(
                f"no-op duplication detected on {type(expr)} in "
                f"{type(self)}.") from e

    def _cache_retrieve(  # type: ignore[override]
            self,
            expr: TransformMapperResultT,
            key_args: tuple[Any, ...],
            key_kwargs: dict[str, Any],
            key: Hashable | None = None) -> TransformMapperResultT:
        try:
            return self._cache.retrieve(  # type: ignore[return-value]
                expr, key_args, key_kwargs, key=key)
        except CacheCollisionError as e:
            raise ValueError(
                f"cache collision detected on {type(expr)} in {type(self)}.") from e

    def _function_cache_retrieve(  # type: ignore[override]
            self,
            expr: FunctionDefinition,
            key_args: tuple[Any, ...],
            key_kwargs: dict[str, Any],
            key: Hashable | None = None) -> FunctionDefinition:
        try:
            return self._function_cache.retrieve(
                expr, key_args, key_kwargs, key=key)
        except CacheCollisionError as e:
            raise ValueError(
                f"cache collision detected on {type(expr)} in {type(self)}.") from e

    def rec(
            self,
            expr: TransformMapperResultT,
            *args: Any, **kwargs: Any) -> TransformMapperResultT:
        key = self._cache.get_key(expr, *args, **kwargs)
        try:
            return self._cache_retrieve(expr, args, kwargs, key=key)
        except KeyError:
            return self._cache_add(
                expr, args, kwargs, Mapper.rec(self, expr, *args, **kwargs), key=key)

    def rec_function_definition(
            self,
            expr: FunctionDefinition,
            *args: Any, **kwargs: Any) -> FunctionDefinition:
        key = self._function_cache.get_key(expr, *args, **kwargs)
        try:
            return self._function_cache_retrieve(expr, args, kwargs, key=key)
        except KeyError:
            return self._function_cache_add(
                expr, args, kwargs,
                Mapper.rec_function_definition(self, expr, *args, **kwargs), key=key)

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        # type-ignore-reason: self.__init__ has a different function signature
        # than Mapper.__init__
        return type(self)(  # type: ignore[call-arg]
            err_on_collision=self._cache.err_on_collision,  # type: ignore[attr-defined]
            err_on_no_op_duplication=self._cache.err_on_no_op_duplication,  # type: ignore[attr-defined]
            _function_cache=self._function_cache)  # type: ignore[attr-defined]

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
        new_situp = tuple(
            self.rec(s) if isinstance(s, Array) else s
            for s in situp)
        if all(new_s is s for s, new_s in zip(situp, new_situp)):
            return situp
        else:
            return new_situp  # type: ignore[return-value]

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape)
        new_bindings: Mapping[str, Array] = immutabledict({
                name: self.rec(subexpr)
                for name, subexpr in sorted(expr.bindings.items())})
        if (
                new_shape is expr.shape
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return IndexLambda(expr=expr.expr,
                    shape=new_shape,
                    dtype=expr.dtype,
                    bindings=new_bindings,
                    axes=expr.axes,
                    var_to_reduction_descr=expr.var_to_reduction_descr,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        assert expr.name is not None
        new_shape = self.rec_idx_or_size_tuple(expr.shape)
        if new_shape is expr.shape:
            return expr
        else:
            return Placeholder(name=expr.name,
                    shape=new_shape,
                    dtype=expr.dtype,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_stack(self, expr: Stack) -> Array:
        new_arrays = tuple(self.rec(arr) for arr in expr.arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            return expr
        else:
            return Stack(arrays=new_arrays, axis=expr.axis, axes=expr.axes,
                    tags=expr.tags, non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate) -> Array:
        new_arrays = tuple(self.rec(arr) for arr in expr.arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            return expr
        else:
            return Concatenate(arrays=new_arrays, axis=expr.axis,
                               axes=expr.axes, tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def map_roll(self, expr: Roll) -> Array:
        new_ary = self.rec(expr.array)
        if new_ary is expr.array:
            return expr
        else:
            return Roll(array=new_ary,
                    shift=expr.shift,
                    axis=expr.axis,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        new_ary = self.rec(expr.array)
        if new_ary is expr.array:
            return expr
        else:
            return AxisPermutation(array=new_ary,
                    axis_permutation=expr.axis_permutation,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase) -> Array:
        new_ary = self.rec(expr.array)
        new_indices = self.rec_idx_or_size_tuple(expr.indices)
        if new_ary is expr.array and new_indices is expr.indices:
            return expr
        else:
            return type(expr)(new_ary,
                              indices=new_indices,
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
        new_shape = self.rec_idx_or_size_tuple(expr.shape)
        if new_shape is expr.shape:
            return expr
        else:
            return DataWrapper(
                    data=expr.data,
                    shape=new_shape,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_size_param(self, expr: SizeParam) -> Array:
        assert expr.name is not None
        return expr

    def map_einsum(self, expr: Einsum) -> Array:
        new_args = tuple(self.rec(arg) for arg in expr.args)
        if all(new_arg is arg for arg, new_arg in zip(expr.args, new_args)):
            return expr
        else:
            return Einsum(expr.access_descriptors,
                          new_args,
                          axes=expr.axes,
                          redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_named_array(self, expr: NamedArray) -> Array:
        new_container = self.rec(expr._container)
        if new_container is expr._container:
            return expr
        else:
            return type(expr)(new_container,
                              expr.name,
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

    def map_dict_of_named_arrays(self,
            expr: DictOfNamedArrays) -> DictOfNamedArrays:
        new_data = {
            key: self.rec(val.expr)
            for key, val in expr.items()}
        if all(
                new_data_val is val.expr
                for val, new_data_val in zip(expr.values(), new_data.values())):
            return expr
        else:
            return DictOfNamedArrays(new_data, tags=expr.tags)

    def map_loopy_call(self, expr: LoopyCall) -> LoopyCall:
        new_bindings: Mapping[Any, Any] = immutabledict(
                    {name: (self.rec(subexpr) if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})
        if all(
                new_bnd is bnd
                for bnd, new_bnd in zip(
                    expr.bindings.values(),
                    new_bindings.values())):
            return expr
        else:
            return LoopyCall(translation_unit=expr.translation_unit,
                             bindings=new_bindings,
                             entrypoint=expr.entrypoint,
                             tags=expr.tags,
                             )

    def map_loopy_call_result(self, expr: LoopyCallResult) -> Array:
        new_container = self.rec(expr._container)
        assert isinstance(new_container, LoopyCall)
        if new_container is expr._container:
            return expr
        else:
            return LoopyCallResult(
                    container=new_container,
                    name=expr.name,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape) -> Array:
        new_ary = self.rec(expr.array)
        new_newshape = self.rec_idx_or_size_tuple(expr.newshape)
        if new_ary is expr.array and new_newshape is expr.newshape:
            return expr
        else:
            return Reshape(new_ary,
                           newshape=new_newshape,
                           order=expr.order,
                           axes=expr.axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> Array:
        new_send_data = self.rec(expr.send.data)
        if new_send_data is expr.send.data:
            new_send = expr.send
        else:
            new_send = DistributedSend(
                data=new_send_data,
                dest_rank=expr.send.dest_rank,
                comm_tag=expr.send.comm_tag)
        new_passthrough = self.rec(expr.passthrough_data)
        if new_send is expr.send and new_passthrough is expr.passthrough_data:
            return expr
        else:
            return DistributedSendRefHolder(new_send, new_passthrough)

    def map_distributed_recv(self, expr: DistributedRecv) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape)
        if new_shape is expr.shape:
            return expr
        else:
            return DistributedRecv(
                   src_rank=expr.src_rank, comm_tag=expr.comm_tag,
                   shape=new_shape, dtype=expr.dtype, tags=expr.tags,
                   axes=expr.axes, non_equality_tags=expr.non_equality_tags)

    def map_function_definition(self,
                                expr: FunctionDefinition) -> FunctionDefinition:
        # spawn a new mapper to avoid unsound cache hits, since the namespace of the
        # function's body is different from that of the caller.
        new_mapper = self.clone_for_callee(expr)
        new_returns = {name: new_mapper(ret)
                       for name, ret in expr.returns.items()}
        if all(
                new_ret is ret
                for ret, new_ret in zip(
                    expr.returns.values(),
                    new_returns.values())):
            return expr
        else:
            return attrs.evolve(expr, returns=immutabledict(new_returns))

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        new_function = self.rec_function_definition(expr.function)
        new_bindings = {
            name: self.rec(bnd)
            for name, bnd in expr.bindings.items()}
        if (
                new_function is expr.function
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return Call(new_function, immutabledict(new_bindings), tags=expr.tags)

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        new_call = self.rec(expr._container)
        assert isinstance(new_call, Call)
        return new_call[expr.name]


class CopyMapperWithExtraArgs(TransformMapperWithExtraArgs):
    """
    Similar to :class:`CopyMapper`, but each mapper method takes extra
    ``*args``, ``**kwargs`` that are propagated along a path by default.

    The logic in :class:`CopyMapper` purposely does not take the extra
    arguments to keep the cost of its each call frame low.
    """
    def rec_idx_or_size_tuple(self, situp: tuple[IndexOrShapeExpr, ...],
                              *args: Any, **kwargs: Any
                              ) -> tuple[IndexOrShapeExpr, ...]:
        # type-ignore-reason: apparently mypy cannot substitute typevars
        # here.
        return tuple(
            self.rec(s, *args, **kwargs)  # type: ignore[misc]
            if isinstance(s, Array)
            else s
            for s in situp)

    def map_index_lambda(self, expr: IndexLambda,
                         *args: Any, **kwargs: Any) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)
        new_bindings: Mapping[str, Array] = immutabledict({
                name: self.rec(subexpr, *args, **kwargs)
                for name, subexpr in sorted(expr.bindings.items())})
        if (
                new_shape is expr.shape
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return IndexLambda(expr=expr.expr,
                               shape=new_shape,
                               dtype=expr.dtype,
                               bindings=new_bindings,
                               axes=expr.axes,
                               var_to_reduction_descr=expr.var_to_reduction_descr,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def map_placeholder(self, expr: Placeholder, *args: Any, **kwargs: Any) -> Array:
        assert expr.name is not None
        new_shape = self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)
        if new_shape is expr.shape:
            return expr
        else:
            return Placeholder(name=expr.name,
                               shape=new_shape,
                               dtype=expr.dtype,
                               axes=expr.axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def map_stack(self, expr: Stack, *args: Any, **kwargs: Any) -> Array:
        new_arrays = tuple(self.rec(arr, *args, **kwargs) for arr in expr.arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            return expr
        else:
            return Stack(arrays=new_arrays, axis=expr.axis, axes=expr.axes,
                    tags=expr.tags, non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate, *args: Any, **kwargs: Any) -> Array:
        new_arrays = tuple(self.rec(arr, *args, **kwargs) for arr in expr.arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            return expr
        else:
            return Concatenate(arrays=new_arrays, axis=expr.axis,
                               axes=expr.axes, tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def map_roll(self, expr: Roll, *args: Any, **kwargs: Any) -> Array:
        new_ary = self.rec(expr.array, *args, **kwargs)
        if new_ary is expr.array:
            return expr
        else:
            return Roll(array=new_ary,
                        shift=expr.shift,
                        axis=expr.axis,
                        axes=expr.axes,
                        tags=expr.tags,
                        non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation,
                             *args: Any, **kwargs: Any) -> Array:
        new_ary = self.rec(expr.array, *args, **kwargs)
        if new_ary is expr.array:
            return expr
        else:
            return AxisPermutation(array=new_ary,
                                   axis_permutation=expr.axis_permutation,
                                   axes=expr.axes,
                                   tags=expr.tags,
                                   non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase, *args: Any, **kwargs: Any) -> Array:
        assert isinstance(expr, _SuppliedAxesAndTagsMixin)
        new_ary = self.rec(expr.array, *args, **kwargs)
        new_indices = self.rec_idx_or_size_tuple(expr.indices, *args, **kwargs)
        if new_ary is expr.array and new_indices is expr.indices:
            return expr
        else:
            return type(expr)(new_ary,
                              indices=new_indices,
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

    def map_basic_index(self, expr: BasicIndex, *args: Any, **kwargs: Any) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      *args: Any, **kwargs: Any

                                      ) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes,
                                          *args: Any, **kwargs: Any
                                          ) -> Array:
        return self._map_index_base(expr, *args, **kwargs)

    def map_data_wrapper(self, expr: DataWrapper,
                         *args: Any, **kwargs: Any) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)
        if new_shape is expr.shape:
            return expr
        else:
            return DataWrapper(
                    data=expr.data,
                    shape=new_shape,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_size_param(self, expr: SizeParam, *args: Any, **kwargs: Any) -> Array:
        assert expr.name is not None
        return expr

    def map_einsum(self, expr: Einsum, *args: Any, **kwargs: Any) -> Array:
        new_args = tuple(self.rec(arg, *args, **kwargs) for arg in expr.args)
        if all(new_arg is arg for arg, new_arg in zip(expr.args, new_args)):
            return expr
        else:
            return Einsum(expr.access_descriptors,
                          new_args,
                          axes=expr.axes,
                          redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_named_array(self, expr: NamedArray, *args: Any, **kwargs: Any) -> Array:
        new_container = self.rec(expr._container, *args, **kwargs)
        if new_container is expr._container:
            return expr
        else:
            return type(expr)(new_container,
                              expr.name,
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

    def map_dict_of_named_arrays(self,
            expr: DictOfNamedArrays, *args: Any, **kwargs: Any) -> DictOfNamedArrays:
        new_data = {
            key: self.rec(val.expr, *args, **kwargs)
            for key, val in expr.items()}
        if all(
                new_data_val is val.expr
                for val, new_data_val in zip(expr.values(), new_data.values())):
            return expr
        else:
            return DictOfNamedArrays(new_data, tags=expr.tags)

    def map_loopy_call(self, expr: LoopyCall,
                       *args: Any, **kwargs: Any) -> LoopyCall:
        new_bindings: Mapping[Any, Any] = immutabledict(
                    {name: (self.rec(subexpr, *args, **kwargs)
                           if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})
        if all(
                new_bnd is bnd
                for bnd, new_bnd in zip(
                    expr.bindings.values(),
                    new_bindings.values())):
            return expr
        else:
            return LoopyCall(translation_unit=expr.translation_unit,
                             bindings=new_bindings,
                             entrypoint=expr.entrypoint,
                             tags=expr.tags,
                             )

    def map_loopy_call_result(self, expr: LoopyCallResult,
                              *args: Any, **kwargs: Any) -> Array:
        new_container = self.rec(expr._container, *args, **kwargs)
        assert isinstance(new_container, LoopyCall)
        if new_container is expr._container:
            return expr
        else:
            return LoopyCallResult(
                    container=new_container,
                    name=expr.name,
                    axes=expr.axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape,
                    *args: Any, **kwargs: Any) -> Array:
        new_ary = self.rec(expr.array, *args, **kwargs)
        new_newshape = self.rec_idx_or_size_tuple(expr.newshape, *args, **kwargs)
        if new_ary is expr.array and new_newshape is expr.newshape:
            return expr
        else:
            return Reshape(new_ary,
                           newshape=new_newshape,
                           order=expr.order,
                           axes=expr.axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_distributed_send_ref_holder(self, expr: DistributedSendRefHolder,
                                        *args: Any, **kwargs: Any) -> Array:
        new_send_data = self.rec(expr.send.data, *args, **kwargs)
        if new_send_data is expr.send.data:
            new_send = expr.send
        else:
            new_send = DistributedSend(
                data=new_send_data,
                dest_rank=expr.send.dest_rank,
                comm_tag=expr.send.comm_tag)
        new_passthrough = self.rec(expr.passthrough_data, *args, **kwargs)
        if new_send is expr.send and new_passthrough is expr.passthrough_data:
            return expr
        else:
            return DistributedSendRefHolder(new_send, new_passthrough)

    def map_distributed_recv(self, expr: DistributedRecv,
                             *args: Any, **kwargs: Any) -> Array:
        new_shape = self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)
        if new_shape is expr.shape:
            return expr
        else:
            return DistributedRecv(
                   src_rank=expr.src_rank, comm_tag=expr.comm_tag,
                   shape=new_shape, dtype=expr.dtype, tags=expr.tags,
                   axes=expr.axes, non_equality_tags=expr.non_equality_tags)

    def map_function_definition(self, expr: FunctionDefinition,
                                *args: Any, **kwargs: Any) -> FunctionDefinition:
        raise NotImplementedError("Function definitions are purposefully left"
                                  " unimplemented as the default arguments to a new"
                                  " DAG traversal are tricky to guess.")

    def map_call(self, expr: Call,
                 *args: Any, **kwargs: Any) -> AbstractResultWithNamedArrays:
        new_function = self.rec_function_definition(expr.function, *args, **kwargs)
        new_bindings = {
            name: self.rec(bnd, *args, **kwargs)
            for name, bnd in expr.bindings.items()}
        if (
                new_function is expr.function
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return Call(new_function, immutabledict(new_bindings), tags=expr.tags)

    def map_named_call_result(self, expr: NamedCallResult,
                              *args: Any, **kwargs: Any) -> Array:
        new_call = self.rec(expr._container, *args, **kwargs)
        assert isinstance(new_call, Call)
        return new_call[expr.name]

# }}}


# {{{ DirectPredecessorsGetter

class DirectPredecessorsGetter(Mapper):
    """
    Mapper to get the
    `direct predecessors
    <https://en.wikipedia.org/wiki/Glossary_of_graph_theory#direct_predecessor>`__
    of a node.

    .. note::

        We only consider the predecessors of a nodes in a data-flow sense.
    """
    def _get_preds_from_shape(self, shape: ShapeType) -> frozenset[ArrayOrNames]:
        return frozenset({dim for dim in shape if isinstance(dim, Array)})

    def map_dict_of_named_arrays(
            self, expr: DictOfNamedArrays) -> frozenset[ArrayOrNames]:
        return frozenset(expr._data.values())

    def map_index_lambda(self, expr: IndexLambda) -> frozenset[ArrayOrNames]:
        return (frozenset(expr.bindings.values())
                | self._get_preds_from_shape(expr.shape))

    def map_stack(self, expr: Stack) -> frozenset[ArrayOrNames]:
        return (frozenset(expr.arrays)
                | self._get_preds_from_shape(expr.shape))

    def map_concatenate(self, expr: Concatenate) -> frozenset[ArrayOrNames]:
        return (frozenset(expr.arrays)
                | self._get_preds_from_shape(expr.shape))

    def map_einsum(self, expr: Einsum) -> frozenset[ArrayOrNames]:
        return (frozenset(expr.args)
                | self._get_preds_from_shape(expr.shape))

    def map_loopy_call_result(self, expr: NamedArray) -> frozenset[ArrayOrNames]:
        from pytato.loopy import LoopyCall, LoopyCallResult
        assert isinstance(expr, LoopyCallResult)
        assert isinstance(expr._container, LoopyCall)
        return (frozenset(ary
                          for ary in expr._container.bindings.values()
                          if isinstance(ary, Array))
                | self._get_preds_from_shape(expr.shape))

    def _map_index_base(self, expr: IndexBase) -> frozenset[ArrayOrNames]:
        return (frozenset([expr.array])
                | frozenset(idx for idx in expr.indices
                            if isinstance(idx, Array))
                | self._get_preds_from_shape(expr.shape))

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def _map_index_remapping_base(self, expr: IndexRemappingBase
                                  ) -> frozenset[ArrayOrNames]:
        return frozenset([expr.array])

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_input_base(self, expr: InputArgumentBase) -> frozenset[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_distributed_recv(self, expr: DistributedRecv) -> frozenset[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> frozenset[ArrayOrNames]:
        return frozenset([expr.passthrough_data])

    def map_call(self, expr: Call) -> frozenset[ArrayOrNames]:
        return frozenset(expr.bindings.values())

    def map_named_call_result(
            self, expr: NamedCallResult) -> frozenset[ArrayOrNames]:
        return frozenset([expr._container])

# }}}


# {{{ Deduplicator

class Deduplicator(CopyMapper):
    """Removes duplicate nodes from an expression."""
    def __init__(
            self,
            _function_cache: _FunctionCacheT | None = None
            ) -> None:
        super().__init__(
            err_on_collision=False, err_on_no_op_duplication=False,
            _function_cache=_function_cache)

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        # type-ignore-reason: self.__init__ has a different function signature
        # than Mapper.__init__
        return type(self)(  # type: ignore[call-arg]
            _function_cache=self._function_cache)  # type: ignore[attr-defined]

# }}}


# {{{ CombineMapper

# FIXME: Can this be made to inherit from CachedMapper?
class CombineMapper(Mapper, Generic[CombineT]):
    """
    Abstract mapper that recursively combines the results of user nodes
    of a given expression.

    .. automethod:: combine
    """
    def __init__(self) -> None:
        super().__init__()
        self.cache: dict[ArrayOrNames, CombineT] = {}
        # Don't need to pass function cache as argument here, because unlike
        # CachedMapper we're not creating a new mapper for each call
        self.function_cache: dict[FunctionDefinition, CombineT] = {}

    def rec_idx_or_size_tuple(self, situp: tuple[IndexOrShapeExpr, ...]
                              ) -> tuple[CombineT, ...]:
        return tuple(self.rec(s) for s in situp if isinstance(s, Array))

    def rec(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: CombineT = super().rec(expr)
        self.cache[expr] = result
        return result

    def rec_function_definition(
            self, expr: FunctionDefinition) -> CombineT:
        if expr in self.function_cache:
            return self.function_cache[expr]
        result: CombineT = super().rec_function_definition(expr)
        self.function_cache[expr] = result
        return result

    # type-ignore reason: incompatible ret. type with super class
    def __call__(self, expr: ArrayOrNames) -> CombineT:  # type: ignore
        return self.rec(expr)

    def combine(self, *args: CombineT) -> CombineT:
        """Combine the arguments."""
        raise NotImplementedError

    def map_index_lambda(self, expr: IndexLambda) -> CombineT:
        return self.combine(*(self.rec(bnd)
                              for _, bnd in sorted(expr.bindings.items())),
                            *self.rec_idx_or_size_tuple(expr.shape))

    def map_placeholder(self, expr: Placeholder) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    def map_data_wrapper(self, expr: DataWrapper) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

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

    def map_loopy_call_result(self, expr: LoopyCallResult) -> CombineT:
        return self.rec(expr._container)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> CombineT:
        return self.combine(
                self.rec(expr.send.data),
                self.rec(expr.passthrough_data),
                )

    def map_distributed_recv(self, expr: DistributedRecv) -> CombineT:
        return self.combine(*self.rec_idx_or_size_tuple(expr.shape))

    def map_function_definition(self, expr: FunctionDefinition) -> CombineT:
        raise NotImplementedError("Combining results from a callee expression"
                                  " is context-dependent. Derived classes"
                                  " must override map_function_definition.")

    def map_call(self, expr: Call) -> CombineT:
        raise NotImplementedError(
            "Mapping calls is context-dependent. Derived classes must override "
            "map_call.")

    def map_named_call_result(self, expr: NamedCallResult) -> CombineT:
        return self.rec(expr._container)

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

class InputGatherer(CombineMapper[FrozenSet[InputArgumentBase]]):
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

class SizeParamGatherer(CombineMapper[FrozenSet[SizeParam]]):
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

class WalkMapper(Mapper):
    """
    A mapper that walks over all the arrays in a :class:`pytato.Array`.

    Users may override the specific mapper methods in a derived class or
    override :meth:`WalkMapper.visit` and :meth:`WalkMapper.post_visit`.

    .. automethod:: visit
    .. automethod:: post_visit
    """

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        return type(self)()

    def visit(self, expr: Any, *args: Any, **kwargs: Any) -> bool:
        """
        If this method returns *True*, *expr* is traversed during the walk.
        If this method returns *False*, *expr* is not traversed as a part of
        the walk.
        """
        return True

    def post_visit(self, expr: Any, *args: Any, **kwargs: Any) -> None:
        """
        Callback after *expr* has been traversed.
        """
        pass

    def rec_idx_or_size_tuple(self, situp: tuple[IndexOrShapeExpr, ...],
                              *args: Any, **kwargs: Any) -> None:
        for comp in situp:
            if isinstance(comp, Array):
                self.rec(comp, *args, **kwargs)

    def map_index_lambda(self, expr: IndexLambda, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for _, child in sorted(expr.bindings.items()):
            self.rec(child, *args, **kwargs)

        self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_placeholder(self, expr: Placeholder, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec_idx_or_size_tuple(expr.shape)

        self.post_visit(expr, *args, **kwargs)

    map_data_wrapper = map_placeholder
    map_size_param = map_placeholder

    def _map_index_remapping_base(self, expr: IndexRemappingBase,
                                  *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.array, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_index_base(self, expr: IndexBase, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.array, *args, **kwargs)

        self.rec_idx_or_size_tuple(expr.indices, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_basic_index(self, expr: BasicIndex, *args: Any, **kwargs: Any) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      *args: Any, **kwargs: Any) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes,
                                          *args: Any, **kwargs: Any) -> None:
        return self._map_index_base(expr, *args, **kwargs)

    def map_stack(self, expr: Stack, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.arrays:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_concatenate(self, expr: Concatenate, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.arrays:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_einsum(self, expr: Einsum, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr.args:
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays,
                                 *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for child in expr._data.values():
            self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder,
            *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.send.data, *args, **kwargs)
        self.rec(expr.passthrough_data, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_distributed_recv(self, expr: DistributedRecv,
                             *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec_idx_or_size_tuple(expr.shape, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_named_array(self, expr: NamedArray, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr._container, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_loopy_call(self, expr: LoopyCall, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        for _, child in sorted(expr.bindings.items()):
            if isinstance(child, Array):
                self.rec(child, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_function_definition(self, expr: FunctionDefinition,
                                *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        new_mapper = self.clone_for_callee(expr)
        for subexpr in expr.returns.values():
            new_mapper(subexpr, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_call(self, expr: Call, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec_function_definition(expr.function, *args, **kwargs)
        for bnd in expr.bindings.values():
            self.rec(bnd, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

    def map_named_call_result(self, expr: NamedCallResult,
                              *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr._container, *args, **kwargs)

        self.post_visit(expr, *args, **kwargs)

# }}}


# {{{ CachedWalkMapper

class CachedWalkMapper(WalkMapper):
    """
    WalkMapper that visits each node in the DAG exactly once. This loses some
    information compared to :class:`WalkMapper` as a node is visited only from
    one of its predecessors.
    """

    def __init__(
            self,
            _visited_functions: set[Any] | None = None) -> None:
        super().__init__()
        self._visited_arrays_or_names: set[Any] = set()

        if _visited_functions is None:
            _visited_functions = set()

        self._visited_functions: set[Any] = _visited_functions

    def get_cache_key(self, expr: ArrayOrNames, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_function_definition_cache_key(
            self, expr: FunctionDefinition, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def rec(self, expr: ArrayOrNames, *args: Any, **kwargs: Any
            ) -> None:
        cache_key = self.get_cache_key(expr, *args, **kwargs)
        if cache_key in self._visited_arrays_or_names:
            return

        super().rec(expr, *args, **kwargs)
        self._visited_arrays_or_names.add(cache_key)

    def rec_function_definition(self, expr: FunctionDefinition,
                                *args: Any, **kwargs: Any) -> None:
        cache_key = self.get_function_definition_cache_key(expr, *args, **kwargs)
        if cache_key in self._visited_functions:
            return

        super().rec_function_definition(expr, *args, **kwargs)
        self._visited_functions.add(cache_key)

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        # type-ignore-reason: self.__init__ has a different function signature
        # than Mapper.__init__
        return type(self)(  # type: ignore[call-arg]
            _visited_functions=self._visited_functions)  # type: ignore[attr-defined]

# }}}


# {{{ TopoSortMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class TopoSortMapper(CachedWalkMapper):
    """A mapper that creates a list of nodes in topological order.

    :members: topological_order

    .. note::

        Does not consider the nodes inside  a
        :class:`~pytato.function.FunctionDefinition`.
    """

    def __init__(
            self,
            _visited_functions: set[Any] | None = None) -> None:
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
    _FunctionCacheT: TypeAlias = CopyMapper._FunctionCacheT

    def __init__(
            self,
            # FIXME: Should map_fn be applied to functions too?
            map_fn: Callable[[ArrayOrNames], ArrayOrNames],
            _function_cache: _FunctionCacheT | None = None
            ) -> None:
        super().__init__(_function_cache=_function_cache)
        self.map_fn: Callable[[ArrayOrNames], ArrayOrNames] = map_fn

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        # type-ignore-reason: self.__init__ has a different function signature
        # than Mapper.__init__ and does not have map_fn
        return type(self)(  # type: ignore[call-arg]
            self.map_fn,  # type: ignore[attr-defined]
            _function_cache=self._function_cache)  # type: ignore[attr-defined]

    def rec(self, expr: MappedT) -> MappedT:
        key = self._cache.get_key(expr)
        try:
            return self._cache_retrieve(expr, key=key)  # type: ignore[return-value]
        except KeyError:
            return self._cache_add(
                expr, Mapper.rec(self, self.map_fn(expr)), key=key)

    if TYPE_CHECKING:
        def __call__(self, expr: MappedT) -> MappedT:
            return self.rec(expr)

# }}}


# {{{ MPMS materializer

@dataclass(frozen=True, eq=True)
class MPMSMaterializerAccumulator:
    """This class serves as the return value of :class:`MPMSMaterializer`. It
    contains the set of materialized predecessors and the rewritten expression
    (i.e. the expression with tags for materialization applied).
    """
    materialized_predecessors: frozenset[Array]
    expr: Array


class MPMSMaterializerCache(
        CachedMapperCache[ArrayOrNames, ArrayOrNames, MPMSMaterializerAccumulator]):
    """
    Cache for :class:`MPMSMaterializer`.

    .. automethod:: __init__
    .. automethod:: add
    """
    def __init__(
            self,
            err_on_collision: bool,
            err_on_no_op_duplication: bool) -> None:
        """
        Initialize the cache.

        :arg err_on_collision: Raise an exception if two distinct input expression
            instances have the same key.
        :arg err_on_no_op_duplication: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        def key_func(
                expr_or_result: ArrayOrNames | MPMSMaterializerAccumulator
                ) -> ArrayOrNames:
            return (
                expr_or_result
                if isinstance(expr_or_result, ArrayOrNames)
                else expr_or_result.expr)

        super().__init__(
            key_func,
            err_on_collision=err_on_collision)

        self.err_on_no_op_duplication = err_on_no_op_duplication

        self._result_key_to_result: dict[
            ArrayOrNames, MPMSMaterializerAccumulator] = {}

    def add(
            self,
            expr: ArrayOrNames,
            result: MPMSMaterializerAccumulator,
            key: ArrayOrNames | None = None,
            result_key: ArrayOrNames | None = None) -> MPMSMaterializerAccumulator:
        """
        Cache a mapping result.

        Returns the cached result (which may not be identical to *result* if a
        result was already cached with the same result key).
        """
        if key is None:
            key = self._key_func(expr)
        if result_key is None:
            result_key = self._key_func(result)

        assert key not in self._expr_key_to_result, \
            "Cache entry is already present for this key."

        try:
            result = self._result_key_to_result[result_key]
        except KeyError:
            from pytato.analysis import DirectPredecessorsGetter
            if (
                    self.err_on_no_op_duplication
                    and hash(result_key) == hash(key)
                    and result_key == key
                    and result.expr is not expr
                    # This is questionable, as it will not detect duplication of
                    # things that are not considered direct predecessors (e.g.
                    # a Call's FunctionDefinition). Not sure how to handle such cases
                    and all(
                        result_pred is pred
                        for pred, result_pred in zip(
                            DirectPredecessorsGetter()(expr),
                            DirectPredecessorsGetter()(result.expr)))):
                raise CacheNoOpDuplicationError from None

            self._result_key_to_result[result_key] = result

        self._expr_key_to_result[key] = result
        if self.err_on_collision:
            self._expr_key_to_expr[key] = expr

        return result


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
        if not expr.tags_of_type(ImplStored):
            new_expr = expr.tagged(ImplStored())
        else:
            new_expr = expr
        return MPMSMaterializerAccumulator(frozenset([new_expr]), new_expr)
    else:
        return MPMSMaterializerAccumulator(materialized_predecessors, expr)


class MPMSMaterializer(CachedMapper):
    """
    See :func:`materialize_with_mpms` for an explanation.

    .. attribute:: nsuccessors

        A mapping from a node in the expression graph (i.e. an
        :class:`~pytato.Array`) to its number of successors.
    """
    _CacheType: type[Any] = MPMSMaterializerCache
    _CacheT: TypeAlias = MPMSMaterializerCache

    def __init__(self, nsuccessors: Mapping[Array, int]):
        err_on_collision = __debug__
        err_on_no_op_duplication = __debug__

        # Does not support functions, so function_cache is ignored
        super().__init__(err_on_collision=err_on_collision)

        self.nsuccessors = nsuccessors
        self._cache: MPMSMaterializer._CacheT = MPMSMaterializer._CacheType(
            err_on_collision=err_on_collision,
            err_on_no_op_duplication=err_on_no_op_duplication)

    def _cache_add(
            self,
            expr: ArrayOrNames,
            result: MPMSMaterializerAccumulator,
            key: Hashable | None = None) -> MPMSMaterializerAccumulator:
        try:
            return self._cache.add(expr, result, key=key)  # type: ignore[return-value]
        except CacheNoOpDuplicationError as e:
            raise ValueError(
                f"no-op duplication detected on {type(expr)} in "
                f"{type(self)}.") from e

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        raise AssertionError("Control shouldn't reach this point.")

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
        # FIXME: Why were these being sorted?
        children_rec = {bnd_name: self.rec(bnd)
                        # for bnd_name, bnd in sorted(expr.bindings.items())}
                        for bnd_name, bnd in expr.bindings.items()}
        new_children = immutabledict({
            bnd_name: bnd.expr
            # for bnd_name, bnd in sorted(children_rec.items())})
            for bnd_name, bnd in children_rec.items()})

        if all(
                new_bnd is bnd
                for bnd, new_bnd in zip(
                    expr.bindings.values(),
                    new_children.values())):
            new_expr = expr
        else:
            new_expr = IndexLambda(
                expr=expr.expr,
                shape=expr.shape,
                dtype=expr.dtype,
                bindings=new_children,
                axes=expr.axes,
                var_to_reduction_descr=expr.var_to_reduction_descr,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr, self.nsuccessors[expr],
                                    children_rec.values())

    def map_stack(self, expr: Stack) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_arrays = tuple(ary.expr for ary in rec_arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            new_expr = expr
        else:
            new_expr = Stack(new_arrays, expr.axis, axes=expr.axes, tags=expr.tags,
                             non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_concatenate(self, expr: Concatenate) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_arrays = tuple(ary.expr for ary in rec_arrays)
        if all(new_ary is ary for ary, new_ary in zip(expr.arrays, new_arrays)):
            new_expr = expr
        else:
            new_expr = Concatenate(new_arrays,
                                   expr.axis,
                                   axes=expr.axes,
                                   tags=expr.tags,
                                   non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_arrays)

    def map_roll(self, expr: Roll) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        if rec_array.expr is expr.array:
            new_expr = expr
        else:
            new_expr = Roll(rec_array.expr, expr.shift, expr.axis, axes=expr.axes,
                            tags=expr.tags,
                            non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr, self.nsuccessors[expr],
                                    (rec_array,))

    def map_axis_permutation(self, expr: AxisPermutation
                             ) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        if rec_array.expr is expr.array:
            new_expr = expr
        else:
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
        new_indices = tuple(rec_indices[i].expr
                            if i in rec_indices
                            else expr.indices[i]
                            for i in range(
                                len(expr.indices)))
        if (
                rec_array.expr is expr.array
                and all(
                    new_idx is idx
                    for idx, new_idx in zip(expr.indices, new_indices))):
            new_expr = expr
        else:
            new_expr = type(expr)(rec_array.expr,
                                  new_indices,
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
        if rec_array.expr is expr.array:
            new_expr = expr
        else:
            new_expr = Reshape(rec_array.expr, expr.newshape,
                               expr.order, axes=expr.axes, tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    (rec_array,))

    def map_einsum(self, expr: Einsum) -> MPMSMaterializerAccumulator:
        rec_args = [self.rec(ary) for ary in expr.args]
        new_args = tuple(ary.expr for ary in rec_args)
        if all(new_arg is arg for arg, new_arg in zip(expr.args, new_args)):
            new_expr = expr
        else:
            new_expr = Einsum(expr.access_descriptors,
                              new_args,
                              expr.redn_axis_to_redn_descr,
                              axes=expr.axes,
                              tags=expr.tags,
                              non_equality_tags=expr.non_equality_tags)

        return _materialize_if_mpms(new_expr,
                                    self.nsuccessors[expr],
                                    rec_args)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays
                                 ) -> MPMSMaterializerAccumulator:
        raise NotImplementedError

    def map_loopy_call_result(self, expr: NamedArray) -> MPMSMaterializerAccumulator:
        # loopy call result is always materialized
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> MPMSMaterializerAccumulator:
        rec_send_data = self.rec(expr.send.data)
        if rec_send_data.expr is expr.send.data:
            new_send = expr.send
        else:
            new_send = DistributedSend(
                rec_send_data.expr,
                dest_rank=expr.send.dest_rank,
                comm_tag=expr.send.comm_tag,
                tags=expr.send.tags)
        rec_passthrough = self.rec(expr.passthrough_data)
        if new_send is expr.send and rec_passthrough.expr is expr.passthrough_data:
            new_expr = expr
        else:
            new_expr = DistributedSendRefHolder(new_send, rec_passthrough.expr)

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
        data = {name: copy_mapper(val.expr)
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
    return CachedMapAndCopyMapper(map_fn)(expr)


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
    from pytato.analysis import get_nusers
    materializer = MPMSMaterializer(get_nusers(expr))
    new_data = {}
    for name, ary in expr.items():
        new_data[name] = materializer(ary.expr).expr

    return DictOfNamedArrays(new_data, tags=expr.tags)

# }}}


# {{{ UsersCollector

class UsersCollector(CachedMapper[ArrayOrNames, FunctionDefinition]):
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

    # type-ignore-reason: incompatible with superclass (args/kwargs, return type)
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

    def map_function_definition(self, expr: FunctionDefinition, *args: Any
                                ) -> None:
        raise AssertionError("Control shouldn't reach at this point."
                             " Instantiate another UsersCollector to"
                             " traverse the callee function.")

    def map_call(self, expr: Call, *args: Any) -> None:
        for bnd in expr.bindings.values():
            self.rec(bnd)

    def map_named_call_result(self, expr: NamedCallResult, *args: Any) -> None:
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

def _get_data_dedup_cache_key(ary: DataInterface) -> Hashable:
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

    data_wrapper_cache: dict[Hashable, DataWrapper] = {}
    data_wrappers_encountered = 0

    def cached_data_wrapper_if_present(ary: ArrayOrNames) -> ArrayOrNames:
        nonlocal data_wrappers_encountered

        if isinstance(ary, DataWrapper):
            data_wrappers_encountered += 1
            cache_key = _get_data_dedup_cache_key(ary.data)

            try:
                return data_wrapper_cache[cache_key]
            except KeyError:
                result = ary
                data_wrapper_cache[cache_key] = result
                return result
        else:
            return ary

    array_or_names = map_and_copy(array_or_names, cached_data_wrapper_if_present)

    if data_wrappers_encountered:
        transform_logger.debug("data wrapper de-duplication: "
                               "%d encountered, %d kept, %d eliminated",
                               data_wrappers_encountered,
                               len(data_wrapper_cache),
                               data_wrappers_encountered - len(data_wrapper_cache))

    return array_or_names

# }}}

# vim: foldmethod=marker
