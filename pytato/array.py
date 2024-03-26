from __future__ import annotations

__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Xiaoyu Wei
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

# {{{ docs

__doc__ = """
.. currentmodule:: pytato

.. note::
    Expression trees based on this package are picklable
    as long as no non-picklable data
    (e.g. :class:`pyopencl.array.Array`)
    is referenced from :class:`~pytato.array.DataWrapper`.

Array Interface
---------------

.. autoclass:: Array
.. autoclass:: Axis
.. autoclass:: ReductionDescriptor
.. autoclass:: NamedArray
.. autoclass:: DictOfNamedArrays
.. autoclass:: AbstractResultWithNamedArrays

NumPy-Like Interface
--------------------

These functions generally follow the interface of the corresponding functions in
:mod:`numpy`, but not all NumPy features may be supported.

.. autofunction:: matmul
.. autofunction:: roll
.. autofunction:: transpose
.. autofunction:: stack
.. autofunction:: concatenate
.. autofunction:: zeros
.. autofunction:: ones
.. autofunction:: full
.. autofunction:: eye
.. autofunction:: arange
.. autofunction:: equal
.. autofunction:: not_equal
.. autofunction:: less
.. autofunction:: less_equal
.. autofunction:: greater
.. autofunction:: greater_equal
.. autofunction:: logical_or
.. autofunction:: logical_and
.. autofunction:: logical_not
.. autofunction:: where
.. autofunction:: maximum
.. autofunction:: minimum
.. autofunction:: einsum
.. autofunction:: dot
.. autofunction:: vdot
.. autofunction:: broadcast_to
.. autofunction:: squeeze
.. autofunction:: expand_dims
.. automodule:: pytato.cmath
.. automodule:: pytato.pad
.. automodule:: pytato.reductions

.. currentmodule:: pytato.array

Concrete Array Data
-------------------

.. autoclass:: DataInterface

Built-in Expression Nodes
-------------------------

.. autoclass:: IndexLambda
.. autoclass:: Einsum
.. autoclass:: Stack
.. autoclass:: Concatenate

Index Remapping
^^^^^^^^^^^^^^^

.. autoclass:: IndexRemappingBase
.. autoclass:: Roll
.. autoclass:: AxisPermutation
.. autoclass:: Reshape
.. autoclass:: IndexBase
.. autoclass:: BasicIndex
.. autoclass:: AdvancedIndexInContiguousAxes
.. autoclass:: AdvancedIndexInNoncontiguousAxes

Input Arguments
^^^^^^^^^^^^^^^

.. autoclass:: InputArgumentBase
.. autoclass:: DataWrapper
.. autoclass:: Placeholder
.. autoclass:: SizeParam

User-Facing Node Creation
-------------------------

Node constructors such as :class:`Placeholder.__init__` and
:class:`~pytato.DictOfNamedArrays.__init__` offer limited input validation
(in favor of faster execution). Node creation from outside
:mod:`pytato` should use the following interfaces:

.. class:: ShapeComponent
.. class:: ConvertibleToShape

.. autofunction:: make_dict_of_named_arrays
.. autofunction:: make_placeholder
.. autofunction:: make_size_param
.. autofunction:: make_data_wrapper

Internal API
------------

.. autoclass:: EinsumAxisDescriptor
.. autoclass:: EinsumElementwiseAxis
.. autoclass:: EinsumReductionAxis
.. autoclass:: NormalizedSlice

Traceback functionality
-----------------------

Please consider these undocumented and subject to change at any time.

.. autofunction:: set_traceback_tag_enabled
.. class:: _PytatoFrameSummary
.. class:: _PytatoStackSummary

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: AxesT

    A :class:`tuple` of :class:`Axis` objects.

.. class:: IntegralT

    An integer data type which is a union of integral types of :mod:`numpy` and
    :class:`int`.
"""

# }}}

from abc import ABC, abstractmethod
from functools import partialmethod, cached_property
import operator
import attrs
from typing import (
        Optional, Callable, ClassVar, Dict, Any, Mapping, Tuple, Union,
        Protocol, Sequence, cast, TYPE_CHECKING, List, Iterator, TypeVar,
        FrozenSet, Collection)

import numpy as np
import pymbolic.primitives as prim
from pymbolic import var
from pytools import memoize_method
from pytools.tag import Tag, Taggable

from pytato.scalar_expr import (ScalarType, SCALAR_CLASSES,
                                ScalarExpression, IntegralT,
                                INT_CLASSES, get_reduction_induction_variables)
import re
from immutabledict import immutabledict

# {{{ get a type variable that represents the type of '...'

# https://github.com/python/typing/issues/684#issuecomment-548203158
if TYPE_CHECKING:
    from enum import Enum

    class EllipsisType(Enum):
        Ellipsis = "..."

    Ellipsis = EllipsisType.Ellipsis
else:
    EllipsisType = type(Ellipsis)

# }}}


if TYPE_CHECKING:
    _dtype_any = np.dtype[Any]
else:
    _dtype_any = np.dtype

AxesT = Tuple["Axis", ...]
ArrayT = TypeVar("ArrayT", bound="Array")


# {{{ shape

ShapeComponent = Union[IntegralT, "Array"]
ShapeType = Tuple[ShapeComponent, ...]
ConvertibleToShape = Union[
    ShapeComponent,
    Sequence[ShapeComponent]]


def _check_identifier(s: Optional[str], optional: bool) -> bool:
    if s is None:
        if optional:
            return True
        else:
            raise ValueError(f"'{s}' is not a valid identifier")

    if not s.isidentifier():
        raise ValueError(f"'{s}' is not a valid identifier")

    return True


def normalize_shape(
        shape: ConvertibleToShape,
        ) -> ShapeType:
    def normalize_shape_component(
            s: ShapeComponent) -> ShapeComponent:
        if isinstance(s, Array):
            from pytato.transform import InputGatherer

            if s.shape != ():
                raise ValueError("array valued shapes must be scalars")

            for d in InputGatherer()(s):
                if not isinstance(d, SizeParam):
                    raise NotImplementedError("shape expressions can (for now) only "
                                              "be in terms of SizeParams. Depends on"
                                              f"a  '{type(d).__name__}', "
                                              "a non-SizeParam array.")
            # TODO: Check affine-ness of the array expression.
        else:
            if not isinstance(s, INT_CLASSES):
                raise TypeError("array dimension can be an int or pytato.Array. "
                                f"Got {type(s)}.")
            assert isinstance(s, INT_CLASSES)
            if s < 0:
                raise ValueError(f"size parameter must be nonnegative (got '{s}')")

        return s

    from numbers import Number
    import collections

    if isinstance(shape, (Array, Number)):
        shape = shape,

    assert isinstance(shape, collections.abc.Sequence)
    return tuple(normalize_shape_component(s) for s in shape)

# }}}


# {{{ array interface

ConvertibleToIndexExpr = Union[int, slice, "Array", None, EllipsisType]
IndexExpr = Union[IntegralT, "NormalizedSlice", "Array", None, EllipsisType]
DtypeOrScalar = Union[_dtype_any, ScalarType]
ArrayOrScalar = Union["Array", ScalarType]


# https://github.com/numpy/numpy/issues/19302
def _np_result_type(
        # actual dtype:
        #*arrays_and_dtypes: Union[np.typing.ArrayLike, np.typing.DTypeLike],
        # our dtype:
        *arrays_and_dtypes: DtypeOrScalar,
        ) -> np.dtype[Any]:
    return np.result_type(*arrays_and_dtypes)


def _truediv_result_type(arg1: DtypeOrScalar, arg2: DtypeOrScalar) -> np.dtype[Any]:
    dtype = _np_result_type(arg1, arg2)
    # See: test_true_divide in numpy/core/tests/test_ufunc.py
    # pylint: disable=no-member
    if dtype.kind in "iu":
        return np.dtype(np.float64)
    else:
        return dtype


@attrs.frozen
class NormalizedSlice:
    """
    A normalized version of :class:`slice`. "Normalized" is explained in
    :attr:`start` and :attr:`stop`.

    .. attribute:: start

        An instance of :class:`ShapeComponent`. Normalized to satisfy the
        relation ``0 <= start <= (axis_len-1)``, where ``axis_len`` is the
        length of the axis being sliced.

    .. attribute:: stop

        An instance of :class:`ShapeComponent`. Normalized to satisfy the
        relation ``-1 <= stop <= axis_len``, where ``axis_len`` is the length of
        the axis being sliced.

    .. attribute:: step
    """
    start: ShapeComponent
    stop: ShapeComponent
    step: IntegralT


@attrs.frozen
class Axis(Taggable):
    """
    A type for recording the information about an :class:`~pytato.Array`'s
    axis.
    """
    tags: FrozenSet[Tag]

    def _with_new_tags(self, tags: FrozenSet[Tag]) -> Axis:
        from attrs import evolve as replace
        return replace(self, tags=tags)


@attrs.frozen
class ReductionDescriptor(Taggable):
    """
    Records information about a reduction dimension in an
    :class:`~pytato.Array`'.
    """
    tags: FrozenSet[Tag]

    def _with_new_tags(self, tags: FrozenSet[Tag]) -> ReductionDescriptor:
        from attrs import evolve as replace
        return replace(self, tags=tags)


@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class Array(Taggable):
    r"""
    A base class (abstract interface + supplemental functionality) for lazily
    evaluating array expressions. The interface seeks to maximize :mod:`numpy`
    compatibility, though not at all costs.

    Objects of this type are hashable and support structural equality
    comparison (and are therefore immutable).

    .. note::

        Hashability and equality testing *does* break :mod:`numpy`
        compatibility, purposefully so.

    FIXME: Point out our equivalent for :mod:`numpy`'s ``==``.

    .. attribute:: shape

        A tuple of integers or scalar-shaped :class:`~pytato.array.Array`\ s.
        Array-valued shape components may be (at most affinely) symbolic in terms of
        :class:`~pytato.array.SizeParam`\ s.

        .. note::

            Affine-ness is mainly required by code generation for
            :class:`~pytato.array.IndexLambda`, but
            :class:`~pytato.array.IndexLambda` is used to produce
            references to named arrays. Since any array that needs to be
            referenced in this way needs to obey this restriction anyway,
            a decision was made to require the same of *all* array expressions.

    .. attribute:: dtype

        An instance of :class:`numpy.dtype`.

    .. attribute:: axes

        A :class:`tuple` of :class:`~pytato.Axis` instances. One
        corresponding to each dimension of the array.

    .. attribute:: tags

        A :class:`frozenset` of :class:`pytools.tag.Tag` instances.

        Motivation: `RDF
        <https://en.wikipedia.org/wiki/Resource_Description_Framework>`__
        triples (subject: implicitly the array being tagged,
        predicate: the tag, object: the arg).

        Inherits from :class:`pytools.tag.Taggable`.

    .. automethod:: tagged
    .. automethod:: without_tags

    Array interface:

    .. automethod:: __getitem__
    .. attribute:: T

    .. method:: __mul__
    .. method:: __rmul__
    .. method:: __add__
    .. method:: __radd__
    .. method:: __sub__
    .. method:: __rsub__
    .. method:: __truediv__
    .. method:: __rtruediv__
    .. method:: __neg__
    .. method:: __pos__
    .. method:: __and__
    .. method:: __rand__
    .. method:: __or__
    .. method:: __ror__
    .. method:: __xor__
    .. method:: __rxor__
    .. method:: __abs__
    .. method:: conj
    .. automethod:: all
    .. automethod:: any
    .. automethod:: with_tagged_axis

    .. autoattribute:: real
    .. autoattribute:: imag

    Derived attributes:

    .. attribute:: ndim

    """
    axes: AxesT = attrs.field(kw_only=True)
    tags: FrozenSet[Tag] = attrs.field(kw_only=True)

    # These are automatically excluded from equality in EqualityComparer
    non_equality_tags: FrozenSet[Optional[Tag]] = attrs.field(kw_only=True,
                                                              hash=False,
                                                              default=frozenset())

    _mapper_method: ClassVar[str]

    # disallow numpy arithmetic from taking precedence
    __array_priority__: ClassVar[int] = 1

    def _is_eq_valid(self) -> bool:
        return self.__class__.__eq__ is Array.__eq__

    if __debug__:
        def __attrs_post_init__(self) -> None:
            # ensure that a developer does not uses dataclass' "__eq__"
            # or "__hash__" implementation as they have exponential complexity.
            assert self._is_eq_valid()

    def copy(self: ArrayT, **kwargs: Any) -> ArrayT:
        return attrs.evolve(self, **kwargs)

    def _with_new_tags(self: ArrayT, tags: FrozenSet[Tag]) -> ArrayT:
        return attrs.evolve(self, tags=tags)

    if TYPE_CHECKING:
        @property
        def shape(self) -> ShapeType:
            raise NotImplementedError

        @property
        def dtype(self) -> np.dtype[Any]:
            raise NotImplementedError

    @property
    def size(self) -> ShapeComponent:
        from pytools import product
        return product(self.shape)  # type: ignore[no-any-return]

    def __len__(self) -> ShapeComponent:
        if self.ndim == 0:
            raise TypeError("len() of unsized object")

        return self.shape[0]

    def __getitem__(self,
                    slice_spec: Union[ConvertibleToIndexExpr,
                                      Tuple[ConvertibleToIndexExpr, ...]]
                    ) -> Array:
        """
        .. warning::

            Out-of-bounds accesses via :class:`Array` indices are undefined
            behavior and may pass silently.
        """
        if not isinstance(slice_spec, tuple):
            slice_spec = (slice_spec,)

        from pytato.utils import _index_into
        return _index_into(
            self,
            slice_spec,
            tags=_get_default_tags(),
            non_equality_tags=frozenset({_get_created_at_tag()}))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self) -> Array:
        return AxisPermutation(self,
                               tuple(range(self.ndim)[::-1]),
                               tags=_get_default_tags(),
                               axes=_get_default_axes(self.ndim),
                               non_equality_tags=frozenset({_get_created_at_tag()}))

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True

        from pytato.equality import EqualityComparer
        return EqualityComparer()(self, other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __matmul__(self, other: Array, reverse: bool = False) -> Array:
        first = self
        second = other
        if reverse:
            first, second = second, first
        return matmul(first, second)

    __rmatmul__ = partialmethod(__matmul__, reverse=True)

    def _binary_op(self,
            op: Callable[[ScalarExpression, ScalarExpression], ScalarExpression],
            other: ArrayOrScalar,
            get_result_type: Callable[[DtypeOrScalar, DtypeOrScalar], np.dtype[Any]] = _np_result_type,  # noqa
            reverse: bool = False) -> Array:

        # {{{ sanity checks

        if not isinstance(other, (Array,) + SCALAR_CLASSES):
            return NotImplemented

        # }}}

        tags = _get_default_tags()
        non_equality_tags = frozenset({_get_created_at_tag()})

        import pytato.utils as utils
        if reverse:
            result = utils.broadcast_binary_op(other, self, op,
                                               get_result_type,
                                               tags=tags,
                                               non_equality_tags=non_equality_tags)
        else:
            result = utils.broadcast_binary_op(self, other, op,
                                               get_result_type,
                                               tags=tags,
                                               non_equality_tags=non_equality_tags)

        assert isinstance(result, Array)
        return result

    def _unary_op(self, op: Any) -> Array:
        if self.ndim == 0:
            expr = op(var("_in0"))
        else:
            indices = tuple(var(f"_{i}") for i in range(self.ndim))
            expr = op(var("_in0")[indices])

        bindings: Mapping[str, Array] = immutabledict({"_in0": self})
        return IndexLambda(
                expr=expr,
                shape=self.shape,
                dtype=self.dtype,
                bindings=bindings,
                tags=_get_default_tags(),
                axes=_get_default_axes(self.ndim),
                non_equality_tags=frozenset({_get_created_at_tag()}),
                var_to_reduction_descr=immutabledict())

    __mul__ = partialmethod(_binary_op, operator.mul)
    __rmul__ = partialmethod(_binary_op, operator.mul, reverse=True)

    __add__ = partialmethod(_binary_op, operator.add)
    __radd__ = partialmethod(_binary_op, operator.add, reverse=True)

    __sub__ = partialmethod(_binary_op, operator.sub)
    __rsub__ = partialmethod(_binary_op, operator.sub, reverse=True)

    __floordiv__ = partialmethod(_binary_op, operator.floordiv)
    __rfloordiv__ = partialmethod(_binary_op, operator.floordiv, reverse=True)

    __truediv__ = partialmethod(_binary_op, operator.truediv,
            get_result_type=_truediv_result_type)
    __rtruediv__ = partialmethod(_binary_op, operator.truediv,
            get_result_type=_truediv_result_type, reverse=True)

    __pow__ = partialmethod(_binary_op, operator.pow)
    __rpow__ = partialmethod(_binary_op, operator.pow, reverse=True)

    __neg__ = partialmethod(_unary_op, operator.neg)

    __and__ = partialmethod(_binary_op, operator.and_)
    __rand__ = partialmethod(_binary_op, operator.and_, reverse=True)
    __or__ = partialmethod(_binary_op, operator.or_)
    __ror__ = partialmethod(_binary_op, operator.or_, reverse=True)
    __xor__ = partialmethod(_binary_op, operator.xor)
    __rxor__ = partialmethod(_binary_op, operator.xor, reverse=True)

    def conj(self) -> ArrayOrScalar:
        import pytato as pt
        return pt.conj(self)

    def __abs__(self) -> Array:
        import pytato as pt
        return cast(Array, pt.abs(self))

    def __pos__(self) -> Array:
        return self

    def __bool__(self) -> None:
        raise ValueError("The truth value of an array expression is undefined.")

    @property
    def real(self) -> ArrayOrScalar:
        import pytato as pt
        return pt.real(self)

    @property
    def imag(self) -> ArrayOrScalar:
        import pytato as pt
        return pt.imag(self)

    def reshape(self, *shape: Union[int, Sequence[int]], order: str = "C") -> Array:
        import pytato as pt
        if len(shape) == 0:
            raise TypeError("reshape takes at least one argument (0 given)")
        if len(shape) == 1:
            # handle shape as single argument tuple
            return pt.reshape(self, shape[0], order=order)

        # type-ignore reason: passed: "Tuple[Union[int, Sequence[int]], ...]";
        # expected "Union[int, Sequence[int]]"
        return pt.reshape(self, shape, order=order)  # type: ignore[arg-type]

    def all(self, axis: int = 0) -> ArrayOrScalar:
        """
        Equivalent to :func:`pytato.all`.
        """
        import pytato as pt
        return pt.all(self, axis)

    def any(self, axis: int = 0) -> ArrayOrScalar:
        """
        Equivalent to :func:`pytato.any`.
        """
        import pytato as pt
        return pt.any(self, axis)

    def with_tagged_axis(self, iaxis: int,
                         tags: Union[Sequence[Tag], Tag]) -> Array:
        """
        Returns a copy of *self* with *iaxis*-th axis tagged with *tags*.
        """
        new_axes = (self.axes[:iaxis]
                    + (self.axes[iaxis].tagged(tags),)
                    + self.axes[iaxis+1:])
        return self.copy(axes=new_axes)

    @memoize_method
    def __repr__(self) -> str:
        from pytato.stringifier import Reprifier
        return Reprifier()(self)

# }}}


# {{{ mixins

@attrs.frozen(eq=False, slots=False, repr=False)
class _SuppliedShapeAndDtypeMixin:
    """A mixin class for when an array must store its own *shape* and *dtype*,
    rather than when it can derive them easily from inputs.
    """
    shape: ShapeType
    dtype: np.dtype[Any]

# }}}


# {{{ dict of named arrays

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class NamedArray(Array):
    """An entry in a :class:`AbstractResultWithNamedArrays`. Holds a reference
    back to the containing instance as well as the name by which *self* is
    known there.

    .. automethod:: __init__
    """
    _container: AbstractResultWithNamedArrays
    name: str

    _mapper_method: ClassVar[str] = "map_named_array"

    # type-ignore reason: `copy` signature incompatible with super-class
    def copy(self, *,  # type: ignore[override]
             container: Optional[AbstractResultWithNamedArrays] = None,
             name: Optional[str] = None,
             axes: Optional[AxesT] = None,
             tags: Optional[FrozenSet[Tag]] = None) -> NamedArray:
        container = self._container if container is None else container
        name = self.name if name is None else name
        tags = self.tags if tags is None else tags
        axes = self.axes if axes is None else axes

        return type(self)(container=container,
                          name=name,
                          tags=tags,
                          axes=axes)

    @property
    def expr(self) -> Array:
        if isinstance(self._container, DictOfNamedArrays):
            return self._container._data[self.name]
        else:
            raise TypeError("only permitted when container is a DictOfNamedArrays")

    @property
    def shape(self) -> ShapeType:
        return self.expr.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.expr.dtype


@attrs.frozen(eq=False, hash=True, cache_hash=True)
class AbstractResultWithNamedArrays(Mapping[str, NamedArray], Taggable, ABC):
    r"""An abstract array computation that results in multiple :class:`Array`\ s,
    each named. The way in which the values of these arrays are computed
    is determined by concrete subclasses of this class, e.g.
    :class:`pytato.loopy.LoopyCall` or :class:`DictOfNamedArrays`.

    .. automethod:: __init__
    .. automethod:: __contains__
    .. automethod:: __getitem__
    .. automethod:: __len__

    .. note::

        This container deliberately does not implement arithmetic.
    """
    tags: FrozenSet[Tag] = attrs.field(kw_only=True)
    _mapper_method: ClassVar[str]

    def _is_eq_valid(self) -> bool:
        return self.__class__.__eq__ is AbstractResultWithNamedArrays.__eq__

    if __debug__:
        def __attrs_post_init__(self) -> None:
            # ensure that a developer does not uses dataclass' "__eq__"
            # or "__hash__" implementation as they have exponential complexity.
            assert self._is_eq_valid()

    @abstractmethod
    def __contains__(self, name: object) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, name: str) -> NamedArray:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True

        from pytato.equality import EqualityComparer
        return EqualityComparer()(self, other)


@attrs.frozen(eq=False, init=False)
class DictOfNamedArrays(AbstractResultWithNamedArrays):
    """A container of named results, each of which can be computed as an
    array expression provided to the constructor.

    Implements :class:`AbstractResultWithNamedArrays`.

    .. automethod:: __init__
    """
    _data: Mapping[str, Array]

    _mapper_method: ClassVar[str] = "map_dict_of_named_arrays"

    def __init__(self, data: Mapping[str, Array], *,
                 tags: Optional[FrozenSet[Tag]] = None) -> None:
        if tags is None:
            from warnings import warn
            warn("Passing `tags=None` is deprecated and will result"
                 " in an error from 2023. To remove this message either"
                 " call make_dict_of_named_arrays or pass the `tags` argument.")
            tags = frozenset()

        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "tags", tags)

    def __hash__(self) -> int:
        return hash((frozenset(self._data.items()), self.tags))

    def __contains__(self, name: object) -> bool:
        return name in self._data

    @memoize_method
    def __getitem__(self, name: str) -> NamedArray:
        if name not in self._data:
            raise KeyError(name)
        return NamedArray(self, name,
                          axes=self._data[name].axes,
                          tags=self._data[name].tags)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return "DictOfNamedArrays(" + str(self._data) + ")"

# }}}


# {{{ index lambda

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class IndexLambda(_SuppliedShapeAndDtypeMixin, Array):
    r"""Represents an array that can be computed by evaluating
    :attr:`expr` for every value of the input indices. The
    input indices are represented by
    :class:`~pymbolic.primitives.Variable`\ s with names ``_1``,
    ``_2``, and so on.

    .. attribute:: expr

        A scalar-valued :mod:`pymbolic` expression such as
        ``a[_1] + b[_2, _1]``.

        Identifiers in the expression are resolved, in
        order, by lookups in :attr:`bindings`.

        Scalar functions in this expression must
        be identified by a dotted name representing
        a Python object (e.g. ``pytato.c99.sin``).

    .. attribute:: bindings

        A :class:`dict` mapping strings that are valid
        Python identifiers to objects implementing
        the :class:`Array` interface, making array
        expressions available for use in
        :attr:`expr`.

    .. attribute:: var_to_reduction_descr

        A mapping from reduction variables in :attr:`expr` to their
        :class:`ReductionDescriptor`.

    .. automethod:: with_tagged_reduction
    """
    expr: prim.Expression
    bindings: Mapping[str, Array] = attrs.field()
    var_to_reduction_descr: Mapping[str, ReductionDescriptor]

    _mapper_method: ClassVar[str] = "map_index_lambda"

    if __debug__:
        @bindings.validator  # type: ignore[attr-defined, misc]
        def _check_bindings(self, attribute: Any, value: Any) -> None:
            if isinstance(value, dict):
                raise TypeError("bindings may not be a dict")

    def with_tagged_reduction(self,
                              reduction_variable: str,
                              tag: Tag) -> IndexLambda:
        """
        Returns a copy of *self* with the :class:`ReductionDescriptor`
        associated with *reduction_variable* tagged with *tag*.

        :arg reduction_variable: Name of reduction variable in *self* that
            is to be tagged.
        """
        from pytato.diagnostic import NotAReductionAxis
        if not isinstance(reduction_variable, str):
            raise TypeError("Argument 'reduction_variable' expected to be str, "
                            f"got {type(reduction_variable)}.")

        assert (frozenset(self.var_to_reduction_descr)
                == get_reduction_induction_variables(self.expr))

        if reduction_variable not in self.var_to_reduction_descr:
            raise NotAReductionAxis(
                "reduction_variable can be one of"
                f" '{self.var_to_reduction_descr.keys()}',"
                f" got '{reduction_variable}'.")

        assert isinstance(self.var_to_reduction_descr, immutabledict)
        new_var_to_redn_descr = dict(self.var_to_reduction_descr)
        new_var_to_redn_descr[reduction_variable] = \
            self.var_to_reduction_descr[reduction_variable].tagged(tag)

        return type(self)(expr=self.expr,
                          shape=self.shape,
                          dtype=self.dtype,
                          bindings=self.bindings,
                          axes=self.axes,
                          var_to_reduction_descr=immutabledict
                            (new_var_to_redn_descr),
                          tags=self.tags)

# }}}


# {{{ einsum

class EinsumAxisDescriptor:
    """
    Records the access pattern of iterating over an array's axis in a
    :class:`Einsum`.
    """
    pass


@attrs.frozen(order=True)
class EinsumElementwiseAxis(EinsumAxisDescriptor):
    """
    Describes an elementwise access pattern of an array's axis.  In terms of the
    nomenclature used by :class:`IndexLambda`, ``EinsumElementwiseAxis(dim=1)`` would
    correspond to indexing the array's axis as ``_1`` in the expression.
    """
    dim: int


@attrs.frozen(order=True)
class EinsumReductionAxis(EinsumAxisDescriptor):
    """
    Describes a reduction access pattern of an array's axis.  In terms of the
    nomenclature used by :class:`IndexLambda`, ``EinsumReductionAxis(dim=0)`` would
    correspond to indexing the array's axis as ``_r0`` in the expression.
    """
    dim: int


@attrs.frozen(frozen=True, eq=False, repr=False, hash=True, cache_hash=True)
class Einsum(Array):
    """
    An array expression using the `Einstein summation convention
    <https://en.wikipedia.org/wiki/Einstein_notation>`__. See
    :func:`numpy.einsum` for a similar construct.

    .. note::

        Use :func:`pytato.einsum` to create this type of expression node in
        user code.

    .. attribute:: access_descriptors

        A :class:`tuple` of *access_descriptor* for each *arg* in
        :attr:`Einsum.args`. An *access_descriptor* is a tuple of
        :class:`EinsumAxisDescriptor` denoting how each axis of the
        argument will be operated in the einstein summation.

    .. attribute:: args

       A :class:`tuple` of array over which the Einstein summation is being
       performed.

    .. attribute:: access_descr_to_index

       Mapping from the access descriptors to the index used by the user during
       the instantiation of the :class:`Einsum` node. This is a strictly
       non-semantic attribute and only present to support a friendlier
       :meth:`with_tagged_reduction`.

    .. automethod:: with_tagged_reduction
    """

    access_descriptors: Tuple[Tuple[EinsumAxisDescriptor, ...], ...]
    args: Tuple[Array, ...]
    redn_axis_to_redn_descr: Mapping[EinsumReductionAxis,
                                     ReductionDescriptor]
    index_to_access_descr: Mapping[str, EinsumAxisDescriptor]
    _mapper_method: ClassVar[str] = "map_einsum"

    @memoize_method
    def _access_descr_to_axis_len(self
                                  ) -> Mapping[EinsumAxisDescriptor, ShapeComponent]:
        from pytato.utils import are_shape_components_equal
        descr_to_axis_len: Dict[EinsumAxisDescriptor, ShapeComponent] = {}

        for access_descrs, arg in zip(self.access_descriptors,
                                      self.args):
            assert arg.ndim == len(access_descrs)
            for arg_axis_len, descr in zip(arg.shape, access_descrs):
                if descr in descr_to_axis_len:
                    seen_axis_len = descr_to_axis_len[descr]

                    if not are_shape_components_equal(seen_axis_len,
                                                      arg_axis_len):
                        if are_shape_components_equal(arg_axis_len, 1):
                            # this axis would be broadcasted
                            pass
                        else:
                            assert are_shape_components_equal(seen_axis_len, 1)
                            descr_to_axis_len[descr] = arg_axis_len
                else:
                    descr_to_axis_len[descr] = arg_axis_len

        return immutabledict(descr_to_axis_len)

    @cached_property
    def shape(self) -> ShapeType:
        iaxis_to_len: Dict[int, ShapeComponent] = {}

        for descr, axis_len in self._access_descr_to_axis_len().items():
            if isinstance(descr, EinsumElementwiseAxis):
                iaxis_to_len[descr.dim] = axis_len
            elif isinstance(descr, EinsumReductionAxis):
                # reduction axes do not count towards einsum's shape
                pass
            else:
                raise AssertionError

        assert all(i in iaxis_to_len for i in range(len(iaxis_to_len)))
        return tuple(iaxis_to_len[i] for i in range(len(iaxis_to_len)))

    @cached_property
    def dtype(self) -> np.dtype[Any]:
        return np.result_type(*[arg.dtype for arg in self.args])

    def with_tagged_reduction(self,
                              redn_axis: Union[EinsumReductionAxis, str],
                              tag: Tag) -> Einsum:
        """
        Returns a copy of *self* with the :class:`ReductionDescriptor`
        associated with *redn_axis* tagged with *tag*.
        """
        from pytato.diagnostic import InvalidEinsumIndex, NotAReductionAxis
        # {{{ sanity checks

        if isinstance(redn_axis, str):
            try:
                redn_axis_ = self.index_to_access_descr[redn_axis]
            except KeyError:
                raise InvalidEinsumIndex(f"'{redn_axis}': not a valid axis index.")
            if isinstance(redn_axis_, EinsumReductionAxis):
                redn_axis = redn_axis_
            else:
                raise NotAReductionAxis(f"'{redn_axis}' is not"
                                        " a reduction axis.")
        elif isinstance(redn_axis, EinsumReductionAxis):
            pass
        else:
            raise TypeError("Argument 'redn_axis' expected to be"
                            f" EinsumReductionAxis, got {type(redn_axis)}")

        if redn_axis in self.redn_axis_to_redn_descr:
            assert any(redn_axis in access_descrs
                       for access_descrs in self.access_descriptors)
        else:
            raise ValueError(f"{redn_axis}: does not appear as a"
                             " reduction access descriptor.")

        # }}}

        assert isinstance(self.redn_axis_to_redn_descr, immutabledict)
        new_redn_axis_to_redn_descr = dict(self.redn_axis_to_redn_descr)
        new_redn_axis_to_redn_descr[redn_axis] = \
            self.redn_axis_to_redn_descr[redn_axis].tagged(tag)

        return type(self)(access_descriptors=self.access_descriptors,
                          args=self.args,
                          axes=self.axes,
                          redn_axis_to_redn_descr=immutabledict
                            (new_redn_axis_to_redn_descr),
                          tags=self.tags,
                          index_to_access_descr=self.index_to_access_descr,
                          non_equality_tags=self.non_equality_tags,
                          )


EINSUM_FIRST_INDEX = re.compile(r"^\s*((?P<alpha>[a-zA-Z])|(?P<ellipsis>\.\.\.))\s*")


def _normalize_einsum_out_subscript(subscript: str) -> immutabledict[str,
                                                            EinsumAxisDescriptor]:
    """
    Normalizes the output subscript of an einsum (provided in the explicit
    mode). Returns a mapping from index name to an instance of
    :class:`EinsumElementwiseAxis`.

    .. testsetup::

        >>> from pytato.array import _normalize_einsum_out_subscript

    .. doctest::

        >>> result = _normalize_einsum_out_subscript("kij")
        >>> sorted(result.keys())
        ['i', 'j', 'k']
        >>> result["i"], result["j"], result["k"]
        (EinsumElementwiseAxis(dim=1), EinsumElementwiseAxis(dim=2), EinsumElementwiseAxis(dim=0))
    """  # noqa: E501

    normalized_indices: List[str] = []
    acc = subscript.strip()
    while acc:
        match = EINSUM_FIRST_INDEX.match(acc)
        if match:
            if "alpha" in match.groupdict():
                normalized_indices.append(match.groupdict()["alpha"])
            else:
                assert "ellipsis" in match.groupdict()
                raise NotImplementedError("Broadcasting in einsums not supported")
            assert match.span()[0] == 0
            acc = acc[match.span()[-1]:]
        else:
            raise ValueError(f"Cannot parse '{acc}' in provided einsum"
                             f" '{subscript}'.")

    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("Used an input more than once to refer to the"
                         f" output axis in '{subscript}")

    return immutabledict({idx: EinsumElementwiseAxis(i)
                 for i, idx in enumerate(normalized_indices)})


def _normalize_einsum_in_subscript(subscript: str,
                                   in_operand: Array,
                                   index_to_descr: Mapping[str,
                                                        EinsumAxisDescriptor],
                                   index_to_axis_length: Mapping[str,
                                                               ShapeComponent],
                                   ) -> Tuple[Tuple[EinsumAxisDescriptor, ...],
                                              immutabledict
                                                [str, EinsumAxisDescriptor],
                                              immutabledict[str, ShapeComponent]]:
    """
    Normalizes the subscript for an input operand in an einsum. Returns
    ``(access_descrs, updated_index_to_descr, updated_to_index_to_axis_length)``,
    where, *access_descrs* is a :class:`tuple` of
    :class`EinsumAxisDescriptor` corresponding to *subscript*,
    *updated_index_to_descr* is the updated version of *index_to_descr* while
    inferring *subscript*. Similarly, *updated_index_to_axis_length* is the updated
    version of *index_to_axis_length*.


    :arg index_to_descr: A mapping from index names to instance of
        :class:`EinsumAxisDescriptor`. These constraints would most likely
        recorded during normalizing other parts of an einsum's subscripts.

    :arg index_to_axis_length: A mapping from index names to instance of
        :class:`ShapeComponent` denoting the iteration extent of the index.
        These constraints would most likely recorded during normalizing other
        parts of an einsum's subscripts.
    """
    from pytato.utils import are_shape_components_equal

    normalized_indices: List[str] = []
    acc = subscript.strip()
    while acc:
        match = EINSUM_FIRST_INDEX.match(acc)
        if match:
            if "alpha" in match.groupdict():
                normalized_indices.append(match.groupdict()["alpha"])
            else:
                assert "ellipsis" in match.groupdict()
                raise NotImplementedError("Broadcasting in einsums not supported")
            assert match.span()[0] == 0
            acc = acc[match.span()[-1]:]
        else:
            raise ValueError(f"Cannot parse '{acc}' in provided einsum"
                             f" '{subscript}'.")

    if len(normalized_indices) != in_operand.ndim:
        raise ValueError(f"Subscript '{subscript}' doesn't match the dimensionality "
                         f"of corresponding operand ({in_operand.ndim}).")

    in_operand_axis_descrs = []
    index_to_axis_length_dict = dict(index_to_axis_length)
    index_to_descr_dict = dict(index_to_descr)

    for iaxis, index_char in enumerate(normalized_indices):
        in_axis_len = in_operand.shape[iaxis]
        if index_char in index_to_descr_dict:
            if index_char in index_to_axis_length_dict:
                seen_axis_len = index_to_axis_length_dict[index_char]
                if not are_shape_components_equal(in_axis_len,
                                                  seen_axis_len):
                    if are_shape_components_equal(in_axis_len, 1):
                        # Broadcast the current axis
                        pass
                    elif are_shape_components_equal(seen_axis_len, 1):
                        # Broadcast to the length of the current axis
                        index_to_axis_length_dict[index_char] = in_axis_len
                    else:
                        raise ValueError("Got conflicting lengths for"
                                         f" '{index_char}' -- {in_axis_len},"
                                         f" {seen_axis_len}.")
            else:
                index_to_axis_length_dict[index_char] = in_axis_len
        else:
            redn_sr_no = len([descr for descr in index_to_descr_dict.values()
                              if isinstance(descr, EinsumReductionAxis)])
            redn_axis_descr = EinsumReductionAxis(redn_sr_no)
            index_to_descr_dict[index_char] = redn_axis_descr
            index_to_axis_length_dict[index_char] = in_axis_len

        in_operand_axis_descrs.append(index_to_descr_dict[index_char])

    index_to_axis_length = immutabledict(index_to_axis_length_dict)
    index_to_descr = immutabledict(index_to_descr_dict)

    return (tuple(in_operand_axis_descrs), index_to_descr, index_to_axis_length)


def einsum(subscripts: str, *operands: Array,
           index_to_redn_descr: Optional[Mapping[str, ReductionDescriptor]] = None
           ) -> Einsum:
    """
    Einstein summation *subscripts* on *operands*.
    """
    if len(operands) == 0:
        raise ValueError("must specify at least one operand")

    if index_to_redn_descr is None:
        index_to_redn_descr = {}

    if "->" not in subscripts:
        # implicit-mode: output spec matched by alphabetical ordering of
        # indices (ewwwww)
        raise NotImplementedError("Implicit mode not supported. 'subscripts'"
                                  " must contain '->', followed by the output's"
                                  " indices.")
    in_spec, out_spec = subscripts.split("->")

    in_specs = in_spec.split(",")

    if len(operands) != len(in_specs):
        raise ValueError(
            f"Number of operands should match the number "
            f"of arg specs: '{in_specs}'. Length of operands is {len(operands)}; "
            f"expecting {len(in_specs)} operands."
        )

    index_to_descr = _normalize_einsum_out_subscript(out_spec)
    index_to_axis_length: Mapping[str, ShapeComponent] = immutabledict()
    access_descriptors = []

    for in_spec, in_operand in zip(in_specs, operands):
        access_descriptor, index_to_descr, index_to_axis_length = (
            _normalize_einsum_in_subscript(in_spec,
                                           in_operand,
                                           index_to_descr,
                                           index_to_axis_length))
        access_descriptors.append(access_descriptor)

    # {{{ process index_to_redn_descr

    redn_axis_to_redn_descr = {}
    for idx, redn_descr in index_to_redn_descr.items():
        descr = index_to_descr[idx]
        if isinstance(descr, EinsumReductionAxis):
            redn_axis_to_redn_descr[descr] = redn_descr
        else:
            raise ValueError(f"'{idx}' is not a reduction dim.")

    for descr in index_to_descr.values():
        if isinstance(descr, EinsumReductionAxis):
            if descr not in redn_axis_to_redn_descr:
                redn_axis_to_redn_descr[descr] = ReductionDescriptor(frozenset())

    # }}}

    return Einsum(tuple(access_descriptors), operands,
                  tags=_get_default_tags(),
                  axes=_get_default_axes(len({descr
                                              for descr in index_to_descr.values()
                                              if isinstance(descr,
                                                            EinsumElementwiseAxis)})
                                         ),
                  redn_axis_to_redn_descr=immutabledict(redn_axis_to_redn_descr),
                  index_to_access_descr=index_to_descr,
                  non_equality_tags=frozenset({_get_created_at_tag()}),
                  )

# }}}


# {{{ stack

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class Stack(Array):
    """Join a sequence of arrays along a new axis.

    .. attribute:: arrays

        The sequence of arrays to join

    .. attribute:: axis

        The output axis

    """
    arrays: Tuple[Array, ...]
    axis: int

    _mapper_method: ClassVar[str] = "map_stack"

    @property
    def dtype(self) -> np.dtype[Any]:
        return _np_result_type(*(arr.dtype for arr in self.arrays))

    @property
    def shape(self) -> ShapeType:
        result = list(self.arrays[0].shape)
        result.insert(self.axis, len(self.arrays))
        return tuple(result)

# }}}


# {{{ concatenate

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class Concatenate(Array):
    """Join a sequence of arrays along an existing axis.

    .. attribute:: arrays

        An instance of :class:`tuple` of the arrays to join. The arrays must
        have same shape except for the dimension corresponding to *axis*.

    .. attribute:: axis

        The axis along which the *arrays* are to be concatenated.
    """
    arrays: Tuple[Array, ...]
    axis: int

    _mapper_method: ClassVar[str] = "map_concatenate"

    @property
    def dtype(self) -> np.dtype[Any]:
        return _np_result_type(*(arr.dtype for arr in self.arrays))

    @property
    def shape(self) -> ShapeType:
        # See https://github.com/python/typeshed/issues/7739
        common_axis_len = sum(ary.shape[self.axis]  # type: ignore[misc]
                              for ary in self.arrays)

        return (self.arrays[0].shape[:self.axis]
                + (common_axis_len,)
                + self.arrays[0].shape[self.axis+1:])

# }}}


# {{{ index remapping

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class IndexRemappingBase(Array):
    """Base class for operations that remap the indices of an array.

    Note that index remappings can also be expressed via
    :class:`~pytato.array.IndexLambda`.

    .. attribute:: array

        The input :class:`~pytato.Array`

    """
    array: Array

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.array.dtype

# }}}


# {{{ roll

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class Roll(IndexRemappingBase):
    """Roll an array along an axis.

    .. attribute:: shift

        Shift amount.

    .. attribute:: axis

        Shift axis.
    """
    shift: int
    axis: int

    _mapper_method: ClassVar[str] = "map_roll"

    @property
    def shape(self) -> ShapeType:
        return self.array.shape

# }}}


# {{{ axis permutation

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class AxisPermutation(IndexRemappingBase):
    r"""Permute the axes of an array.

    .. attribute:: array

    .. attribute:: axis_permutation

        A permutation of the input axes.
    """
    axis_permutation: Tuple[int, ...]

    _mapper_method: ClassVar[str] = "map_axis_permutation"

    @property
    def shape(self) -> ShapeType:
        result = []
        base_shape = self.array.shape
        for index in self.axis_permutation:
            result.append(base_shape[index])
        return tuple(result)

# }}}


# {{{ reshape

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class Reshape(IndexRemappingBase):
    """
    Reshape an array.

    .. attribute:: array

        The array to be reshaped

    .. attribute:: newshape

        The output shape

    .. attribute:: order

        Output layout order, either ``C`` or ``F``.
    """
    newshape: ShapeType
    order: str

    _mapper_method: ClassVar[str] = "map_reshape"

    if __debug__:
        def __attrs_post_init__(self) -> None:
            # FIXME: Get rid of this restriction
            assert self.order == "C"
            super().__attrs_post_init__()

    @property
    def shape(self) -> ShapeType:
        return self.newshape

# }}}


# {{{ indexing

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class IndexBase(IndexRemappingBase):
    """
    Abstract class for all index expressions on an array.

    .. attribute:: indices
    """
    indices: Tuple[IndexExpr, ...]


class BasicIndex(IndexBase):
    """
    An indexing expression with all indices being either an :class:`int` or
    :class:`slice`.
    """
    _mapper_method: ClassVar[str] = "map_basic_index"

    @property
    def shape(self) -> ShapeType:
        assert len(self.indices) == self.array.ndim
        assert all(isinstance(idx, (NormalizedSlice, INT_CLASSES))
                   for idx in self.indices)

        from pytato.utils import _normalized_slice_len
        return tuple(_normalized_slice_len(idx)
                     for idx, axis_len in zip(self.indices, self.array.shape)
                     if isinstance(idx, NormalizedSlice))


class AdvancedIndexInContiguousAxes(IndexBase):
    """
    An indexing expression with at least one :class:`Array` index and all the
    advanced indices (i.e. scalars/array) appearing contiguously in
    :attr:`IndexBase.indices`.

    The reason for the existence of this class and
    :class:`AdvancedIndexInNoncontiguousAxes` is that :mod:`numpy` treats those
    two cases differently, and we're bound to follow its precedent.
    """
    _mapper_method: ClassVar[str] = "map_contiguous_advanced_index"

    @property
    def shape(self) -> ShapeType:
        assert len(self.indices) == self.array.ndim
        assert any(isinstance(idx, Array) for idx in self.indices)
        from pytato.utils import (get_shape_after_broadcasting,
                                  _normalized_slice_len, partition)

        i_adv_indices, i_basic_indices = partition(lambda idx: isinstance(
                                                                self.indices[idx],
                                                                NormalizedSlice),
                                                   range(len(self.indices)))

        assert not any(i_adv_indices[0] < i_basic_idx < i_adv_indices[-1]
                       for i_basic_idx in i_basic_indices)

        adv_idx_shape = get_shape_after_broadcasting([self.indices[i_idx]
                                                      for i_idx in i_adv_indices])

        # type-ignored because mypy cannot figure out basic-indices only refer
        # to slices
        pre_basic_idx_shape = tuple(_normalized_slice_len(self.indices[i_idx])  # type: ignore[arg-type]  # noqa: E501
                                    for i_idx in i_basic_indices
                                    if i_idx < i_adv_indices[0])

        # type-ignored because mypy cannot figure out basic-indices only refer
        # to slices
        post_basic_idx_shape = tuple(_normalized_slice_len(self.indices[i_idx])  # type: ignore[arg-type]  # noqa: E501
                                     for i_idx in i_basic_indices
                                     if i_idx > i_adv_indices[-1])

        return pre_basic_idx_shape + adv_idx_shape + post_basic_idx_shape


class AdvancedIndexInNoncontiguousAxes(IndexBase):
    """
    An indexing expression with advanced indices (i.e. scalars/arrays)
    appearing non-contiguously in :attr:`IndexBase.indices`.

    The reason for the existence of this class and
    :class:`AdvancedIndexInContiguousAxes` is that :mod:`numpy` treats those
    two cases differently, and we're bound to follow its precedent.
    """
    _mapper_method: ClassVar[str] = "map_non_contiguous_advanced_index"

    @property
    def shape(self) -> ShapeType:
        assert len(self.indices) == self.array.ndim
        from pytato.utils import (get_shape_after_broadcasting,
                                  _normalized_slice_len, partition)

        i_adv_indices, i_basic_indices = partition(lambda idx: isinstance(
                                                                self.indices[idx],
                                                                NormalizedSlice),
                                                   range(len(self.indices)))

        assert len(i_adv_indices) >= 2
        assert any(i_adv_indices[0] < i_basic_idx < i_adv_indices[-1]
                   for i_basic_idx in i_basic_indices)

        adv_idx_shape = get_shape_after_broadcasting([self.indices[i_idx]
                                                      for i_idx in i_adv_indices])

        # type-ignored because mypy cannot figure out basic-indices only refer slices
        basic_idx_shape = tuple(_normalized_slice_len(self.indices[i_idx])  # type: ignore[arg-type]  # noqa: E501
                                for i_idx in i_basic_indices)

        return adv_idx_shape + basic_idx_shape

# }}}


# {{{ base class for arguments

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class InputArgumentBase(Array):
    r"""Base class for input arguments.

    .. note::

        Creating multiple instances of any input argument with the
        same name in an expression is not allowed.
    """

# }}}


# {{{ data wrapper

class DataInterface(Protocol):
    """A protocol specifying the minimal interface requirements for concrete
    array data supported by :class:`DataWrapper`.

    See :class:`typing.Protocol` for more information about protocols.

    Code generation targets may impose additional restrictions on the kinds of
    concrete array data they support.

    .. attribute:: shape
    .. attribute:: dtype
    """

    # That's how mypy spells "read-only attribute".
    # https://github.com/python/typing/discussions/903

    @property
    def shape(self) -> ShapeType:
        pass

    @property
    def dtype(self) -> np.dtype[Any]:
        pass


@attrs.frozen(eq=False, repr=False, hash=False)
class DataWrapper(InputArgumentBase):
    """Takes concrete array data and packages it to be compatible with the
    :class:`Array` interface.

    .. attribute:: data

        A concrete array (containing data), given as, for example,
        a :class:`numpy.ndarray`, or a :class:`pyopencl.array.Array`.
        This must offer ``shape`` and ``dtype`` attributes but is
        otherwise considered opaque. At evaluation time, its
        type must be understood by the appropriate execution backend.

        Starting with the construction of the :class:`DataWrapper`,
        this array may not be updated in-place.

    .. attribute:: shape

        The shape of the array is represented separately from array
        to allow symbolic shapes to be used, and to ease :mod:`pytato`'s
        job in recognizing shapes of arrays as equal. For example,
        if the shape of :attr:`data` is ``(3, 4)``, and :attr:`shape` is
        ``(nrows, ncolumns)``, then this represents a (global) constraint that
        that ``nrows == 3`` and ``ncolumns == 4``. Arithmetic and other
        operations in :mod:`pytato` do not currently resolve these constraints
        to assess whether shapes match, and thus it is important that a canonical
        (symbolic) form of the shape tuple is used.

    .. attribute:: name

        An (optional, string) name by which this object can be identified.
        Hypothetically, this could be used to 'swap out' the data captured
        here, but that functionality is not currently available.

    .. note::

        Since we cannot compare instances of :class:`DataInterface` being
        wrapped, a :class:`DataWrapper` instances compare equal to themselves
        (i.e. the very same instance).
    """
    data: DataInterface
    _shape: ShapeType

    _mapper_method: ClassVar[str] = "map_data_wrapper"

    @property
    def name(self) -> None:
        return None

    def _is_eq_valid(self) -> bool:
        # we override __hash__ as hashing DataInterface is impractical
        # => promise the __post_init__ that the change was intentional
        #    and valid by returning True
        return True

    def __hash__(self) -> int:
        # It would be better to hash the data, but we have no way of getting to
        # it.
        return id(self)

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

# }}}


# {{{ placeholder

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class Placeholder(_SuppliedShapeAndDtypeMixin, InputArgumentBase):
    r"""A named placeholder for an array whose concrete value is supplied by the
    user during evaluation.

    .. attribute:: name

        The name by which a value is supplied for the argument once computation
        begins.

    .. automethod:: __init__
    """
    name: str

    _mapper_method: ClassVar[str] = "map_placeholder"

# }}}


# {{{ size parameter

@attrs.frozen(eq=False, repr=False, hash=True, cache_hash=True)
class SizeParam(InputArgumentBase):
    r"""A named placeholder for a scalar that may be used as a variable in symbolic
    expressions for array sizes.

    .. attribute:: name

        The name by which a value is supplied for the argument once computation
        begins.
    """
    name: str
    axes: AxesT = attrs.field(kw_only=True, default=())

    _mapper_method: ClassVar[str] = "map_size_param"

    @property
    def shape(self) -> ShapeType:
        return ()

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(np.intp)

# }}}


# {{{ end-user facing

def _get_default_axes(ndim: int) -> AxesT:
    return tuple(Axis(frozenset()) for _ in range(ndim))


_ENABLE_TRACEBACK_TAG = False


def set_traceback_tag_enabled(enable: bool = True) -> None:
    """Enable or disable the traceback tag."""
    global _ENABLE_TRACEBACK_TAG
    _ENABLE_TRACEBACK_TAG = enable


def _get_created_at_tag(stacklevel: int = 1) -> Optional[Tag]:
    """
    Get a :class:`CreatedAt` tag storing the stack trace of an array's creation.

    :param stacklevel: the number of stack levels above this call to record as the
        array creation location
    """
    import traceback
    from pytato.tags import CreatedAt

    if not _ENABLE_TRACEBACK_TAG:
        return None

    # Drop the stack levels corresponding to extract_stack() and any additional
    # levels specified via stacklevel
    stack = traceback.extract_stack()[:-(1+stacklevel)]

    from pytato.tags import _PytatoFrameSummary, _PytatoStackSummary

    frames = tuple(_PytatoFrameSummary(s.filename, s.lineno, s.name, s.line)
                       for s in stack)

    return CreatedAt(_PytatoStackSummary(frames))


def _get_default_tags() -> FrozenSet[Tag]:
    return frozenset()


def matmul(x1: Array, x2: Array) -> Array:
    """Matrix multiplication.

    :param x1: first argument
    :param x2: second argument
    """
    if (
            isinstance(x1, SCALAR_CLASSES)
            or x1.shape == ()
            or isinstance(x2, SCALAR_CLASSES)
            or x2.shape == ()):
        raise ValueError("scalars not allowed as arguments to matmul")

    import pytato as pt

    index_names = "".join([chr(i) for i in range(ord("l"), ord("z")+1)])

    if x1.ndim == x2.ndim == 1:
        return pt.sum(x1 * x2)
    elif x1.ndim == 1:
        return cast(Array, pt.dot(x1, x2))
    elif x2.ndim == 1:
        x1_indices = index_names[:x1.ndim]
        return pt.einsum(f"{x1_indices}, {x1_indices[-1]} -> {x1_indices[:-1]}",
                         x1, x2)

    stack_indices = index_names[:max(x1.ndim-2, x2.ndim-2)]
    x1_indices = stack_indices[len(stack_indices) - x1.ndim+2:] + "ij"
    x2_indices = stack_indices[len(stack_indices) - x2.ndim+2:] + "jk"
    result_indices = stack_indices + "ik"

    return pt.einsum(f"{x1_indices}, {x2_indices} -> {result_indices}", x1, x2)


def roll(a: Array, shift: int, axis: Optional[int] = None) -> Array:
    """Roll array elements along a given axis.

    :param a: input array
    :param shift: the number of places by which elements are shifted
    :param axis: axis along which the array is shifted
    """
    if a.ndim == 0:
        return a

    if axis is None:
        if a.ndim > 1:
            raise NotImplementedError(
                    "shifting along more than one dimension is unsupported")
        else:
            axis = 0

    if not (0 <= axis < a.ndim):
        raise ValueError("invalid axis")

    if shift == 0:
        return a

    return Roll(a, shift, axis,
                tags=_get_default_tags(),
                non_equality_tags=frozenset({_get_created_at_tag()}),
                axes=_get_default_axes(a.ndim))


def transpose(a: Array, axes: Optional[Sequence[int]] = None) -> Array:
    """Reverse or permute the axes of an array.

    :param a: input array
    :param axes: if specified, a permutation of ``[0, 1, ..., a.ndim-1]``. Defaults
        to ``range(a.ndim)[::-1]``. The returned axis at index *i* corresponds to
        the input axis *axes[i]*.
    """
    if axes is None:
        axes = range(a.ndim)[::-1]

    if len(axes) != a.ndim:
        raise ValueError("axes have incorrect length")

    if set(axes) != set(range(a.ndim)):
        raise ValueError("repeated or out-of-bounds axes detected")

    return AxisPermutation(a, tuple(axes),
                           tags=_get_default_tags(),
                           non_equality_tags=frozenset({_get_created_at_tag()}),
                           axes=_get_default_axes(a.ndim))


def stack(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Join a sequence of arrays along a new axis.

    The *axis* parameter specifies the position of the new axis in the result.

    Example::

       >>> import pytato as pt
       >>> arrays = [pt.zeros(3)] * 4
       >>> pt.stack(arrays, axis=0).shape
       (4, 3)

    :param arrays: a finite sequence, each of whose elements is an
        :class:`Array` of the same shape
    :param axis: the position of the new axis, which will have length
        *len(arrays)*
    """
    from pytato.utils import are_shapes_equal

    if not arrays:
        raise ValueError("need at least one array to stack")

    for array in arrays[1:]:
        if not are_shapes_equal(array.shape, arrays[0].shape):
            raise ValueError("arrays must have the same shape")

    if not (0 <= axis <= arrays[0].ndim):
        raise ValueError("invalid axis")

    return Stack(tuple(arrays), axis,
                 tags=_get_default_tags(),
                 non_equality_tags=frozenset({_get_created_at_tag()}),
                 axes=_get_default_axes(arrays[0].ndim+1))


def concatenate(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Join a sequence of arrays along an existing axis.

    Example::

       >>> import pytato as pt
       >>> arrays = [pt.zeros(3)] * 4
       >>> pt.concatenate(arrays, axis=0).shape
       (12,)

    :param arrays: a finite sequence, each of whose elements is an
        :class:`Array` . The arrays are of the same shape except along the
        *axis* dimension.
    :param axis: The axis along which the arrays will be concatenated.
    """

    if not arrays:
        raise ValueError("need at least one array to stack")

    def shape_except_axis(ary: Array) -> ShapeType:
        return ary.shape[:axis] + ary.shape[axis+1:]

    for array in arrays[1:]:
        if shape_except_axis(array) != shape_except_axis(arrays[0]):
            raise ValueError("arrays must have the same shape except along"
                    f" dimension #{axis}.")

    if not (0 <= axis <= arrays[0].ndim):
        raise ValueError("invalid axis")

    return Concatenate(tuple(arrays), axis,
                       tags=_get_default_tags(),
                       non_equality_tags=frozenset({_get_created_at_tag()}),
                       axes=_get_default_axes(arrays[0].ndim))


def reshape(array: Array, newshape: Union[int, Sequence[int]],
            order: str = "C") -> Array:
    """
    :param array: array to be reshaped
    :param newshape: shape of the resulting array
    :param order: ``"C"`` or ``"F"``. Layout order of the result array. Only
        ``"C"`` allowed for now.

    .. note::

        reshapes of arrays with symbolic shapes not yet implemented.
    """
    from pytools import product

    if isinstance(newshape, INT_CLASSES):
        newshape = newshape,

    if newshape.count(-1) > 1:
        raise ValueError("can only specify one unknown dimension")

    if newshape.count(-1) == 1 and newshape.count(0) > 0:
        raise ValueError(f"cannot reshape {array.shape} into {newshape}")

    if not all(isinstance(axis_len, INT_CLASSES) for axis_len in array.shape):
        raise ValueError("reshape of arrays with symbolic lengths not allowed")

    if order != "C":
        raise NotImplementedError("Reshapes to a 'F'-ordered arrays")

    newshape_explicit = []

    for new_axislen in newshape:
        if not isinstance(new_axislen, INT_CLASSES):
            raise ValueError("Symbolic reshapes not allowed.")

        if new_axislen < -1:
            raise ValueError("newshape should be either a sequence of non-negative"
                             " ints or -1")

        # {{{ infer the axis length corresponding to axis marked "-1"

        if new_axislen == -1:
            size_of_rest_of_newaxes = -1 * product(newshape)

            if array.size % size_of_rest_of_newaxes != 0:
                raise ValueError(f"cannot reshape array of size {array.size}"
                        f" into ({size_of_rest_of_newaxes})")

            new_axislen = array.size // size_of_rest_of_newaxes

        # }}}

        newshape_explicit.append(new_axislen)

    if product(newshape_explicit) != array.size:
        raise ValueError(f"cannot reshape array of size {array.size}"
                f" into {newshape}")

    return Reshape(array, tuple(newshape_explicit), order,
                   tags=_get_default_tags(),
                   non_equality_tags=frozenset({_get_created_at_tag()}),
                   axes=_get_default_axes(len(newshape_explicit)))


# {{{ make_dict_of_named_arrays

def make_dict_of_named_arrays(data: Dict[str, Array], *,
                              tags: FrozenSet[Tag] = frozenset()
                              ) -> DictOfNamedArrays:
    """Make a :class:`DictOfNamedArrays` object.

    :param data: member keys and arrays
    """
    return DictOfNamedArrays(data, tags=(tags | _get_default_tags()))

# }}}


def make_placeholder(name: str,
                     shape: ConvertibleToShape,
                     dtype: Any = np.float64,
                     tags: FrozenSet[Tag] = frozenset(),
                     axes: Optional[AxesT] = None) -> Placeholder:
    """Make a :class:`Placeholder` object.

    :param name:       name of the placeholder array, generated automatically
                       if not given
    :param shape:      shape of the placeholder array
    :param dtype:      dtype of the placeholder array
                       (must be convertible to :class:`numpy.dtype`, default is
                       :class:`numpy.float64`)
    :param tags:       implementation tags
    """
    _check_identifier(name, optional=False)
    shape = normalize_shape(shape)
    dtype = np.dtype(dtype)

    if axes is None:
        axes = _get_default_axes(len(shape))

    if len(axes) != len(shape):
        raise ValueError("'axes' dimensionality mismatch:"
                         f" expected {len(shape)}, got {len(axes)}.")

    return Placeholder(name=name, shape=shape, dtype=dtype, axes=axes,
                       tags=(tags | _get_default_tags()),
                       non_equality_tags=frozenset({_get_created_at_tag()}),)


def make_size_param(name: str,
                    tags: FrozenSet[Tag] = frozenset()) -> SizeParam:
    """Make a :class:`SizeParam`.

    Size parameters may be used as variables in symbolic expressions for array
    sizes.

    :param name:       name
    :param tags:       implementation tags
    """
    _check_identifier(name, optional=False)
    return SizeParam(name, tags=(tags | _get_default_tags()),
                     non_equality_tags=frozenset({_get_created_at_tag()}),)


def make_data_wrapper(data: DataInterface,
        *,
        name: Optional[str] = None,
        shape: Optional[ConvertibleToShape] = None,
        tags: FrozenSet[Tag] = frozenset(),
        axes: Optional[AxesT] = None) -> DataWrapper:
    """Make a :class:`DataWrapper`.

    :param data:       an instance obeying the :class:`DataInterface`
    :param name:       an optional name, generated automatically if not given
    :param shape:      optional shape of the array, inferred from *data* if not given
    :param tags:       implementation tags
    """
    _check_identifier(name, optional=True)
    if shape is None:
        shape = data.shape

    if name is not None:
        from warnings import warn
        warn("Naming DataWrappers is deprecated and "
                "will be converted to a PrefixNamed tag. "
                "This will stop working in 2023. "
                "Use pytato.tags.{Named,PrefixNamed} instead.",
                DeprecationWarning, stacklevel=2)
        from pytato.tags import PrefixNamed
        tags = tags | frozenset({PrefixNamed(name)})

    shape = normalize_shape(shape)

    if axes is None:
        axes = _get_default_axes(len(shape))

    if len(axes) != len(shape):
        raise ValueError("'axes' dimensionality mismatch:"
                         f" expected {len(shape)}, got {len(axes)}.")

    return DataWrapper(data, shape, axes=axes, tags=(tags | _get_default_tags()),
                       non_equality_tags=frozenset({_get_created_at_tag()}),)

# }}}


# {{{ full

def full(shape: ConvertibleToShape, fill_value: ScalarType,
         dtype: Any = None, order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to *fill_value*.
    """
    if order != "C":
        raise ValueError("Only C-ordered arrays supported for now.")

    if dtype is None:
        dtype = np.array(fill_value).dtype
    else:
        dtype = np.dtype(dtype)

    shape = normalize_shape(shape)

    if np.isnan(fill_value):
        from pymbolic.primitives import NaN
        fill_value = NaN(dtype.type)
    else:
        fill_value = dtype.type(fill_value)

    return IndexLambda(expr=fill_value, shape=shape, dtype=dtype,
                       bindings=immutabledict(),
                       tags=_get_default_tags(),
                       non_equality_tags=frozenset({_get_created_at_tag()}),
                       axes=_get_default_axes(len(shape)),
                       var_to_reduction_descr=immutabledict())


def zeros(shape: ConvertibleToShape, dtype: Any = float,
        order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to 0.
    """
    return full(shape, 0, dtype)


def ones(shape: ConvertibleToShape, dtype: Any = float,
        order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to 1.
    """
    return full(shape, 1, dtype)

# }}}


# {{{ eye

def eye(N: int, M: Optional[int] = None, k: int = 0,  # noqa: N803
        dtype: Any = np.float64) -> Array:
    """
    Returns a 2D-array with ones on the *k*-th diagonal

    :arg N: Number of rows in the output matrix
    :arg M: Number of columns in the output matrix. Equal to *N* if *None*.
    """
    from pymbolic import parse

    if M is None:
        M = N  # noqa: N806

    if M < 0 or N < 0:
        raise ValueError("Negative dimension lengths not allowed.")

    if not isinstance(k, INT_CLASSES):
        raise ValueError(f"k must be int, got {type(k)}.")

    return IndexLambda(expr=parse(f"1 if ((_1 - _0) == {k}) else 0"),
                       shape=(N, M), dtype=dtype, bindings=immutabledict({}),
                       tags=_get_default_tags(),
                       non_equality_tags=frozenset({_get_created_at_tag()}),
                       axes=_get_default_axes(2),
                       var_to_reduction_descr=immutabledict())

# }}}


# {{{ arange

@attrs.define
class _ArangeInfo:
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]
    dtype: Optional[np.dtype[Any]]


def arange(*args: Any, **kwargs: Any) -> Array:
    """``arange([start, ]stop, [step, ]dtype=None)``

    Semantically equivalent to :func:`numpy.arange`.
    """

    explicit_dtype = False

    # {{{ argument processing

    inf = _ArangeInfo(
            start=None,
            stop=None,
            step=None,
            dtype=None)

    # Yuck. Thanks, numpy developers. ;)
    if isinstance(args[-1], np.dtype):
        inf.dtype = args[-1]
        args = args[:-1]
        explicit_dtype = True

    argc = len(args)
    if argc == 0:
        raise TypeError("stop argument required")
    elif argc == 1:
        inf.stop, = args
    elif argc == 2:
        inf.start, inf.stop = args
    elif argc == 3:
        inf.start, inf.stop, inf.step = args
    else:
        raise TypeError("arange() takes 0 to 4 positional arguments but"
                f" {argc} were given")

    admissible_names = ["start", "stop", "step", "dtype"]
    for k, v in kwargs.items():
        if k in admissible_names:
            if getattr(inf, k) is None:
                setattr(inf, k, v)
                if k == "dtype":
                    explicit_dtype = True
            else:
                raise TypeError(
                        "may not specify '%s' by position and keyword" % k)
        else:
            raise TypeError("unexpected keyword argument '%s'" % k)

    if inf.start is None:
        inf.start = 0
    if inf.step is None:
        inf.step = 1
    from numbers import Number
    if not isinstance(inf.start, Number):
        raise NotImplementedError("non-numerical start")
    if not isinstance(inf.stop, Number):
        raise NotImplementedError("non-numerical stop")
    if not isinstance(inf.step, Number):
        raise TypeError("non-numerical step")
    if inf.dtype is None:
        inf.dtype = np.array([inf.start, inf.stop, inf.step]).dtype

    # }}}

    if not explicit_dtype:
        raise TypeError("arange requires a dtype argument")

    dtype = np.dtype(inf.dtype)
    start = dtype.type(inf.start)
    step = dtype.type(inf.step)
    stop = dtype.type(inf.stop)

    from math import ceil
    # np.real() suppresses "ComplexWarning: Casting complex values to real
    # discards the imaginary part":
    size = max(0, int(ceil((np.real(stop)-np.real(start))/np.real(step))))

    from pymbolic.primitives import Variable
    return IndexLambda(expr=start + Variable("_0") * step,
                       shape=(size,), dtype=dtype, bindings=immutabledict(),
                       tags=_get_default_tags(),
                       non_equality_tags=frozenset({_get_created_at_tag()}),
                       axes=_get_default_axes(1),
                       var_to_reduction_descr=immutabledict())

# }}}


# {{{ comparison operator

def _compare(x1: ArrayOrScalar, x2: ArrayOrScalar, which: str) -> Union[Array, bool]:
    # https://github.com/python/mypy/issues/3186
    import pytato.utils as utils
    # type-ignored because 'broadcast_binary_op' returns Scalar, while
    # '_compare' returns a bool.
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.Comparison(x, which, y),
                                     lambda x, y: np.dtype(np.bool_),
                                     tags=_get_default_tags(),
                                     non_equality_tags=frozenset({
                                         _get_created_at_tag(stacklevel=2)}),
                                     )  # type: ignore[return-value]


def equal(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns (x1 == x2) element-wise.
    """
    return _compare(x1, x2, "==")


def not_equal(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns (x1 != x2) element-wise.
    """
    return _compare(x1, x2, "!=")


def less(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns (x1 < x2) element-wise.
    """
    return _compare(x1, x2, "<")


def less_equal(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns (x1 <= x2) element-wise.
    """
    return _compare(x1, x2, "<=")


def greater(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns (x1 > x2) element-wise.
    """
    return _compare(x1, x2, ">")


def greater_equal(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns (x1 >= x2) element-wise.
    """
    return _compare(x1, x2, ">=")

# }}}


# {{{ logical operations

def logical_or(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns the element-wise logical OR of *x1* and *x2*.
    """
    # https://github.com/python/mypy/issues/3186
    # type-ignored because 'broadcast_binary_op' returns Scalar, while
    # '_compare' returns a bool.
    import pytato.utils as utils
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.LogicalOr((x, y)),
                                     lambda x, y: np.dtype(np.bool_),
                                     tags=_get_default_tags(),
                                     non_equality_tags=frozenset({
                                         _get_created_at_tag()}),
                                     )  # type: ignore[return-value]


def logical_and(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns the element-wise logical AND of *x1* and *x2*.
    """
    # https://github.com/python/mypy/issues/3186
    # type-ignored because 'broadcast_binary_op' returns Scalar, while
    # '_compare' returns a bool.
    import pytato.utils as utils
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.LogicalAnd((x, y)),
                                     lambda x, y: np.dtype(np.bool_),
                                     tags=_get_default_tags(),
                                     non_equality_tags=frozenset({
                                         _get_created_at_tag()}),
                                     )  # type: ignore[return-value]


def logical_not(x: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns the element-wise logical NOT of *x*.
    """
    if isinstance(x, SCALAR_CLASSES):
        # https://github.com/python/mypy/issues/3186
        return np.logical_not(x)  # type: ignore[no-any-return]

    assert isinstance(x, Array)

    from pytato.utils import with_indices_for_broadcasted_shape
    return IndexLambda(expr=with_indices_for_broadcasted_shape(prim.Variable("_in0"),
                                                          x.shape,
                                                          x.shape),
                       shape=x.shape,
                       dtype=np.dtype(np.bool_),
                       bindings={"_in0": x},
                       tags=_get_default_tags(),
                       non_equality_tags=frozenset({_get_created_at_tag()}),
                       axes=_get_default_axes(len(x.shape)),
                       var_to_reduction_descr=immutabledict())

# }}}


# {{{ where

def where(condition: ArrayOrScalar,
          x: Optional[ArrayOrScalar] = None,
          y: Optional[ArrayOrScalar] = None) -> ArrayOrScalar:
    """
    Elementwise selector between *x* and *y* depending on *condition*.
    """
    import pytato.utils as utils

    # {{{ raise if single-argument form of pt.where is invoked

    if x is None and y is None:
        raise ValueError("Pytato does not support data-dependent array shapes.")

    if (x is None) or (y is None):
        raise ValueError("x and y must be pytato arrays")

    # }}}

    if (isinstance(condition, SCALAR_CLASSES) and isinstance(x, SCALAR_CLASSES)
            and isinstance(y, SCALAR_CLASSES)):
        # https://github.com/python/mypy/issues/3186
        return x if condition else y  # type: ignore[no-any-return]

    # {{{ find dtype

    x_dtype = x.dtype if isinstance(x, Array) else np.dtype(type(x))
    y_dtype = y.dtype if isinstance(y, Array) else np.dtype(type(y))
    dtype = np.promote_types(x_dtype, y_dtype)

    # }}}

    result_shape = utils.get_shape_after_broadcasting([condition, x, y])

    bindings: Dict[str, Array] = {}

    expr1 = utils.update_bindings_and_get_broadcasted_expr(condition, "_in0",
                                                           bindings, result_shape)
    expr2 = utils.update_bindings_and_get_broadcasted_expr(x, "_in1", bindings,
                                                           result_shape)
    expr3 = utils.update_bindings_and_get_broadcasted_expr(y, "_in2", bindings,
                                                           result_shape)

    return IndexLambda(
            expr=prim.If(expr1, expr2, expr3),
            shape=result_shape,
            dtype=dtype,
            bindings=immutabledict(bindings),
            tags=_get_default_tags(),
            non_equality_tags=frozenset({_get_created_at_tag()}),
            axes=_get_default_axes(len(result_shape)),
            var_to_reduction_descr=immutabledict())

# }}}


# {{{ (max|min)inimum

def maximum(x1: ArrayOrScalar, x2: ArrayOrScalar) -> ArrayOrScalar:
    """
    Returns the elementwise maximum of *x1*, *x2*. *x1*, *x2* being
    array-like objects that could be broadcasted together. NaNs are propagated.
    """
    from pytato.utils import get_common_dtype_of_ary_or_scalars
    common_dtype = get_common_dtype_of_ary_or_scalars([x1, x2])

    if (np.issubdtype(common_dtype, np.floating)
            or np.issubdtype(common_dtype, np.complexfloating)):
        from pytato.cmath import isnan
        return where(logical_or(isnan(x1), isnan(x2)),
                     # I don't know why pylint thinks common_dtype is a tuple.
                     common_dtype.type(np.nan),  # pylint: disable=no-member
                     where(greater(x1, x2), x1, x2))
    else:
        return where(greater(x1, x2), x1, x2)


def minimum(x1: ArrayOrScalar, x2: ArrayOrScalar) -> ArrayOrScalar:
    """
    Returns the elementwise minimum of *x1*, *x2*. *x1*, *x2* being
    array-like objects that could be broadcasted together. NaNs are propagated.
    """
    from pytato.utils import get_common_dtype_of_ary_or_scalars
    common_dtype = get_common_dtype_of_ary_or_scalars([x1, x2])

    if (np.issubdtype(common_dtype, np.floating)
            or np.issubdtype(common_dtype, np.complexfloating)):
        from pytato.cmath import isnan
        return where(logical_or(isnan(x1), isnan(x2)),
                     # I don't know why pylint thinks common_dtype is a tuple.
                     common_dtype.type(np.nan),  # pylint: disable=no-member
                     where(less(x1, x2), x1, x2))
    else:
        return where(less(x1, x2), x1, x2)

# }}}


# {{{ make_index_lambda

def make_index_lambda(
        expression: Union[str, ScalarExpression],
        bindings: Mapping[str, Array],
        shape: ShapeType,
        dtype: Any,
        var_to_reduction_descr: Optional[Mapping[str, ReductionDescriptor]] = None
) -> IndexLambda:
    if isinstance(expression, str):
        raise NotImplementedError

    if var_to_reduction_descr is None:
        var_to_reduction_descr = {}

    # {{{ sanity checks

    from pytato.scalar_expr import get_dependencies
    unknown_dep = (get_dependencies(expression, include_idx_lambda_indices=False)
            - set(bindings))

    for dep in unknown_dep:
        raise ValueError(f"Unknown variable '{dep}' in the expression.")

    # }}}

    # {{{ process var_to_reduction_descr

    processed_var_to_reduction_descr = {}
    redn_vars = get_reduction_induction_variables(expression)

    if not (frozenset(var_to_reduction_descr) <= redn_vars):
        raise ValueError(f"'{frozenset(var_to_reduction_descr) - redn_vars}': not"
                         " reduction induction variables.")

    for redn_var in redn_vars:
        redn_descr = var_to_reduction_descr.get(redn_var,
                                           ReductionDescriptor(frozenset()))
        if not isinstance(redn_descr, ReductionDescriptor):
            raise TypeError(f"reduction_dim for {redn_var} expected to be"
                            f" of type ReductionDescriptor, got {type(redn_descr)}.")
        processed_var_to_reduction_descr[redn_var] = redn_descr

    # }}}

    return IndexLambda(expr=expression,
                       bindings=immutabledict(bindings),
                       shape=shape,
                       dtype=dtype,
                       tags=_get_default_tags(),
                       non_equality_tags=frozenset({_get_created_at_tag()}),
                       axes=_get_default_axes(len(shape)),
                       var_to_reduction_descr=immutabledict
                        (processed_var_to_reduction_descr))

# }}}


# {{{ dot, vdot

def dot(a: ArrayOrScalar, b: ArrayOrScalar) -> ArrayOrScalar:
    """
    For 1-dimensional arrays *a* and *b* computes their inner product.  See
    :func:`numpy.dot` for behavior in the case when *a* and *b* aren't
    single-dimensional arrays.
    """
    import pytato as pt

    if isinstance(a, SCALAR_CLASSES) or isinstance(b, SCALAR_CLASSES):
        return a * b

    assert isinstance(a, Array)
    assert isinstance(b, Array)

    if a.ndim == b.ndim == 1:
        return pt.sum(a*b)
    elif a.ndim == b.ndim == 2:
        return a @ b
    elif a.ndim == 0 or b.ndim == 0:
        # https://github.com/python/mypy/issues/16499
        return a * b  # type: ignore[no-any-return]
    elif b.ndim == 1:
        return pt.sum(a * b, axis=(a.ndim - 1))
    else:
        idx_stream = (chr(i) for i in range(ord("i"), ord("z")))
        idx_gen: Callable[[], str] = lambda: next(idx_stream)  # noqa: E731
        a_indices = "".join(idx_gen() for _ in range(a.ndim))
        b_indices = "".join(idx_gen() for _ in range(b.ndim))
        # reduce over second-to-last axis of *b* and last axis of *a*
        b_indices = b_indices[:-2] + a_indices[-1] + b_indices[-1]
        result_indices = a_indices[:-1] + b_indices[:-2] + b_indices[-1]
        return pt.einsum(f"{a_indices}, {b_indices} -> {result_indices}", a, b)


def vdot(a: Array, b: Array) -> ArrayOrScalar:
    """
    Returns the dot-product of conjugate of *a* with *b*. If the input
    arguments are multi-dimensional arrays, they are ravel-ed first and then
    their *vdot* is computed.
    """
    import pytato as pt

    if isinstance(a, Array) and a.ndim > 1:
        a = a.reshape(-1)
    if isinstance(b, Array) and b.ndim > 1:
        b = b.reshape(-1)

    return pt.dot(pt.conj(a), b)

# }}}


# {{{ broadcast_to

def broadcast_to(array: Array, shape: ShapeType) -> Array:
    """
    Returns *array* broadcasted to *shape*.
    """
    from pytato.utils import (get_indexing_expression,
                              are_shape_components_equal)

    if len(shape) < array.ndim:
        raise ValueError(f"Cannot broadcast '{array.shape}' into '{shape}'")

    for in_dim, brdcst_dim in zip(array.shape,
                                  shape[-array.ndim:]):
        if (not are_shape_components_equal(in_dim, brdcst_dim)
                and not are_shape_components_equal(in_dim, 1)):
            raise ValueError(f"Cannot broadcast '{array.shape}' into '{shape}'")

    return IndexLambda(expr=prim.Subscript(prim.Variable("in"),
                                           get_indexing_expression(array.shape,
                                                                   shape)),
                       shape=shape,
                       dtype=array.dtype,
                       bindings=immutabledict({"in": array}),
                       tags=_get_default_tags(),
                       non_equality_tags=frozenset({_get_created_at_tag()}),
                       axes=_get_default_axes(len(shape)),
                       var_to_reduction_descr=immutabledict())

# }}}


# {{{ squeeze

def squeeze(array: Array, axis: Optional[Collection[int]] = None) -> Array:
    """
    Remove single-dimensional entries from the shape of an array.

    :arg axis: Subset of 1-long axes of *array* that must be removed. If *None*
        all 1-long axes are removed.
    """
    from pytato.utils import are_shape_components_equal

    one_d_axes = frozenset({idim
                            for idim in range(array.ndim)
                            if are_shape_components_equal(array.shape[idim], 1)})
    if axis is None:
        axis = one_d_axes
    else:
        axis = frozenset(axis)
        if not (axis <= one_d_axes):
            raise ValueError("cannot squeeze an axis which is not 1-long")

    return array[tuple(
            0 if i in axis else slice(s_i)
            for i, s_i in enumerate(array.shape))]

# }}}


# {{{ expand_dims

def expand_dims(array: Array, axis: Union[Tuple[int, ...], int]) -> Array:
    """
    Reshapes *array* by adding 1-long axes at *axis* dimensions of the returned
    array.
    """
    from pytato.tags import ExpandedDimsReshape

    if isinstance(axis, int):
        axis = axis,

    output_ndim = array.ndim + len(axis)

    normalized_axis: List[int] = []

    # {{{ sanity checks

    for ax in axis:
        if not (-output_ndim <= ax < output_ndim):
            raise ValueError(f"Dimension {ax} not present in {output_ndim}-D array.")

        normalized_axis.append(ax if ax >= 0 else (ax+output_ndim))

    if len(set(normalized_axis)) != len(normalized_axis):
        raise ValueError(f"repeated axis in '{axis}'.")

    # }}}

    new_shape = list(array.shape)

    for ax in sorted(normalized_axis):
        assert (0 <= ax < output_ndim)
        new_shape.insert(ax, 1)

    assert len(new_shape) == output_ndim

    return Reshape(array=array, newshape=tuple(new_shape), order="C",
                   tags=(_get_default_tags()
                         | {ExpandedDimsReshape(tuple(normalized_axis))}),
                   non_equality_tags=frozenset({_get_created_at_tag()}),
                   axes=_get_default_axes(len(new_shape)))

# }}}

# vim: foldmethod=marker
