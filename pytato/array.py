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
.. autofunction:: abs
.. autofunction:: sqrt
.. autofunction:: sin
.. autofunction:: cos
.. autofunction:: tan
.. autofunction:: arcsin
.. autofunction:: arccos
.. autofunction:: arctan
.. autofunction:: conj
.. autofunction:: arctan2
.. autofunction:: sinh
.. autofunction:: cosh
.. autofunction:: tanh
.. autofunction:: exp
.. autofunction:: log
.. autofunction:: log10
.. autofunction:: isnan
.. autofunction:: real
.. autofunction:: imag
.. autofunction:: zeros
.. autofunction:: ones
.. autofunction:: full
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
.. autofunction:: sum
.. autofunction:: amin
.. autofunction:: amax
.. autofunction:: prod
.. autofunction:: einsum

.. currentmodule:: pytato.array

Concrete Array Data
-------------------

.. autoclass:: DataInterface

Pre-Defined Tags
----------------

.. autoclass:: ImplementAs
.. autoclass:: ImplementationStrategy
.. autoclass:: CountNamed
.. autoclass:: ImplStored
.. autoclass:: ImplInlined
.. autoclass:: ImplDefault

Built-in Expression Nodes
-------------------------

.. autoclass:: IndexLambda
.. autoclass:: Einsum
.. autoclass:: MatrixProduct
.. autoclass:: Stack
.. autoclass:: Concatenate

Index Remapping
^^^^^^^^^^^^^^^

.. autoclass:: IndexRemappingBase
.. autoclass:: Roll
.. autoclass:: AxisPermutation
.. autoclass:: Reshape
.. autoclass:: Slice

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
.. autoclass:: ElementwiseAxis
.. autoclass:: ReductionAxis

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: T_co

    A covariant type variable used in, e.g. :meth:`pytools.tag.Taggable.copy`.
"""

# }}}

from abc import ABC, abstractmethod
from functools import partialmethod
import operator
from dataclasses import dataclass
from typing import (
        Optional, Callable, ClassVar, Dict, Any, Mapping, Tuple, Union,
        Protocol, Sequence, cast, TYPE_CHECKING, List, Iterator)

import numpy as np
import pymbolic.primitives as prim
from pymbolic import var
from pytools import memoize_method
from pytools.tag import (Tag, Taggable, UniqueTag, TagOrIterableType,
    TagsType, tag_dataclass)

from pytato.scalar_expr import (ScalarType, SCALAR_CLASSES,
                                ScalarExpression, Reduce)
import re
from pyrsistent import pmap, PMap

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


# {{{ shape

ShapeComponent = Union[int, "Array"]
ShapeType = Tuple[ShapeComponent, ...]
ConvertibleToShape = Union[
    ShapeComponent,
    Sequence[ShapeComponent]]


def _check_identifier(s: Optional[str]) -> bool:
    if s is None:
        return True

    if not s.isidentifier():
        raise ValueError(f"'{s}' is not a valid identifier")

    return True


def normalize_shape(
        shape: ConvertibleToShape,
        ) -> ShapeType:
    def normalize_shape_component(
            s: ShapeComponent) -> ShapeComponent:
        if isinstance(s, Array):
            from pytato.transform import DependencyMapper

            if s.shape != ():
                raise ValueError("array valued shapes must be scalars")

            for d in (k for k in DependencyMapper()(s)
                      if isinstance(k, InputArgumentBase)):
                if not isinstance(d, SizeParam):
                    raise NotImplementedError("shape expressions can (for now) only "
                                              "be in terms of SizeParams. Depends on"
                                              f" '{d.name}', a non-SizeParam array.")
            # TODO: Check affine-ness of the array expression.
        else:
            if not isinstance(s, int):
                raise TypeError("array dimension can be an int or pytato.Array. "
                                f"Got {type(s)}.")
            assert isinstance(s, int)
            if s < 0:
                raise ValueError(f"size parameter must be nonnegative (got '{s}')")

        return s

    from numbers import Number
    if isinstance(shape, (Array, Number)):
        shape = shape,

    # https://github.com/python/mypy/issues/3186
    return tuple(normalize_shape_component(s) for s in shape)  # type: ignore

# }}}


# {{{ array inteface

SliceItem = Union[int, slice, None, EllipsisType]
DtypeOrScalar = Union[_dtype_any, ScalarType]
ArrayOrScalar = Union["Array", ScalarType]


def _truediv_result_type(arg1: DtypeOrScalar, arg2: DtypeOrScalar) -> np.dtype[Any]:
    dtype = np.result_type(arg1, arg2)
    # See: test_true_divide in numpy/core/tests/test_ufunc.py
    if dtype.kind in "iu":
        return np.dtype(np.float64)
    else:
        return cast(_dtype_any, dtype)


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
            a decision was made to requir the same of *all* array expressions.

    .. attribute:: dtype

        An instance of :class:`numpy.dtype`.

    .. attribute:: tags

        A :class:`tuple` of :class:`pytools.tag.Tag` instances.

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

    .. autoattribute:: real
    .. autoattribute:: imag

    Derived attributes:

    .. attribute:: ndim

    """
    _mapper_method: ClassVar[str]
    # A tuple of field names. Fields must be equality comparable and
    # hashable. Dicts of hashable keys and values are also permitted.
    _fields: ClassVar[Tuple[str, ...]] = ("shape", "dtype", "tags")

    @property
    def shape(self) -> ShapeType:
        raise NotImplementedError

    @property
    def size(self) -> ShapeComponent:
        from pytools import product
        return product(self.shape)  # type: ignore

    @property
    def dtype(self) -> np.dtype[Any]:
        raise NotImplementedError

    def __getitem__(self,
            slice_spec: Union[SliceItem, Tuple[SliceItem, ...]]) -> Array:
        if not isinstance(slice_spec, tuple):
            slice_spec = (slice_spec,)

        # FIXME: This doesn't support all NumPy basic indexing constructs,
        # including:
        #
        # - newaxis
        # - Ellipsis
        # - slices with nontrivial strides
        # - slices with bounds that exceed array dimensions
        # - slices with negative indices
        # - slices that yield an array with a zero dimension (lacks codegen support)
        #
        # Symbolic expression support is also limited.

        starts = []
        stops = []
        kept_dims = []

        slice_spec_expanded = []

        for elem in slice_spec:
            if elem is Ellipsis:
                raise NotImplementedError("'...' is unsupported")
            elif elem is None:
                raise NotImplementedError("newaxis is unsupported")
            else:
                assert isinstance(elem, (int, slice))
                slice_spec_expanded.append(elem)

        slice_spec_expanded.extend(
                [slice(None, None, None)] * (self.ndim - len(slice_spec)))

        if len(slice_spec_expanded) > self.ndim:
            raise ValueError("too many dimensions in index")

        for i, elem in enumerate(slice_spec_expanded):
            if isinstance(elem, slice):
                start = elem.start
                if start is None:
                    start = 0
                stop = elem.stop
                if stop is None:
                    stop = self.shape[i]
                stride = elem.step
                if stride is not None and stride != 1:
                    raise ValueError("non-trivial strides unsupported")
                starts.append(start)
                stops.append(stop)
                kept_dims.append(i)

            elif isinstance(elem, int):
                starts.append(elem)
                stops.append(elem+1)

            else:
                raise ValueError("unknown index along dimension")

        slice_ = _make_slice(self, starts, stops)

        if len(kept_dims) != self.ndim:
            # Return an IndexLambda that elides the indexed-into dimensions
            # (as opposed to the ones that were sliced).
            indices = [0] * self.ndim
            shape = []
            for i, dim in enumerate(kept_dims):
                indices[dim] = var(f"_{i}")
                shape.append(slice_.shape[dim])
            expr = var("_in0")
            if indices:
                expr = expr[tuple(indices)]

            # FIXME: Flatten into a single IndexLambda
            return IndexLambda(expr,
                    shape=tuple(shape),
                    dtype=self.dtype,
                    bindings=dict(_in0=slice_))
        else:
            return slice_

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self) -> Array:
        return AxisPermutation(self, tuple(range(self.ndim)[::-1]))

    @memoize_method
    def __hash__(self) -> int:
        attrs = []
        for field in self._fields:
            attr = getattr(self, field)
            if isinstance(attr, dict):
                attr = frozenset(attr.items())
            attrs.append(attr)
        return hash(tuple(attrs))

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (
                isinstance(other, type(self))
                and all(
                    getattr(self, field) == getattr(other, field)
                    for field in self._fields))

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
            op: Any,
            other: ArrayOrScalar,
            get_result_type: Callable[[DtypeOrScalar, DtypeOrScalar], np.dtype[Any]] = np.result_type,  # noqa
            reverse: bool = False) -> Array:

        # {{{ sanity checks

        if not isinstance(other, (Array,) + SCALAR_CLASSES):
            return NotImplemented

        # }}}

        import pytato.utils as utils
        if reverse:
            return utils.broadcast_binary_op(other, self, op,
                                             get_result_type)  # type: ignore
        else:
            return utils.broadcast_binary_op(self, other, op,
                                             get_result_type)  # type: ignore

    def _unary_op(self, op: Any) -> Array:
        if self.ndim == 0:
            expr = op(var("_in0"))
        else:
            indices = tuple(var(f"_{i}") for i in range(self.ndim))
            expr = op(var("_in0")[indices])

        bindings = dict(_in0=self)
        return IndexLambda(
                expr,
                shape=self.shape,
                dtype=self.dtype,
                bindings=bindings)

    __mul__ = partialmethod(_binary_op, operator.mul)
    __rmul__ = partialmethod(_binary_op, operator.mul, reverse=True)

    __add__ = partialmethod(_binary_op, operator.add)
    __radd__ = partialmethod(_binary_op, operator.add, reverse=True)

    __sub__ = partialmethod(_binary_op, operator.sub)
    __rsub__ = partialmethod(_binary_op, operator.sub, reverse=True)

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

    def __abs__(self) -> Array:
        # /!\ This refers to the abs() defined in this file, not the built-in.
        return cast(Array, abs(self))

    def __pos__(self) -> Array:
        return self

    @property
    def real(self) -> ArrayOrScalar:
        import pytato as pt
        return pt.real(self)

    @property
    def imag(self) -> ArrayOrScalar:
        import pytato as pt
        return pt.imag(self)

# }}}


# {{{ mixins

class _SuppliedShapeAndDtypeMixin(object):
    """A mixin class for when an array must store its own *shape* and *dtype*,
    rather than when it can derive them easily from inputs.
    """

    def __init__(self,
            shape: ShapeType,
            dtype: np.dtype[Any],
            **kwargs: Any):
        # https://github.com/python/mypy/issues/5887
        super().__init__(**kwargs)  # type: ignore
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

# }}}


# {{{ pre-defined tag: ImplementAs

@tag_dataclass
class ImplementationStrategy(Tag):
    pass


@tag_dataclass
class ImplStored(ImplementationStrategy):
    pass


@tag_dataclass
class ImplInlined(ImplementationStrategy):
    pass


@tag_dataclass
class ImplDefault(ImplementationStrategy):
    pass


@tag_dataclass
class ImplementAs(UniqueTag):
    """
    .. attribute:: strategy
    """

    strategy: ImplementationStrategy

# }}}


# {{{ pre-defined tag: CountNamed

@tag_dataclass
class CountNamed(UniqueTag):
    """
    .. attribute:: name
    """

    name: str

# }}}


# {{{ dict of named arrays

class NamedArray(Array):
    """An entry in a :class:`AbstractResultWithNamedArrays`. Holds a reference
    back to thecontaining instance as well as the name by which *self* is
    known there.

    .. automethod:: __init__
    """
    _fields = Array._fields + ("_container", "name")
    _mapper_method = "map_named_array"

    def __init__(self,
            container: AbstractResultWithNamedArrays,
            name: str,
            tags: TagsType = frozenset()) -> None:
        super().__init__(tags=tags)
        self._container = container
        self.name = name

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


class AbstractResultWithNamedArrays(Mapping[str, NamedArray], ABC):
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

    _mapper_method: ClassVar[str]

    @abstractmethod
    def __contains__(self, name: object) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, name: str) -> NamedArray:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class DictOfNamedArrays(AbstractResultWithNamedArrays):
    """A container of named results, each of which can be computed as an
    array expression provided to the constructor.

    Implements :class:`AbstractResultWithNamedArrays`.

    .. automethod:: __init__
    """

    _mapper_method = "map_dict_of_named_arrays"

    def __init__(self, data: Mapping[str, Array]):
        super().__init__()
        self._data = data

    def __hash__(self) -> int:
        return hash(frozenset(self._data.items()))

    def __contains__(self, name: object) -> bool:
        return name in self._data

    @memoize_method
    def __getitem__(self, name: str) -> NamedArray:
        if name not in self._data:
            raise KeyError(name)
        return NamedArray(self, name)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

# }}}


# {{{ index lambda

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

    """

    _fields = Array._fields + ("expr", "bindings")
    _mapper_method = "map_index_lambda"

    def __init__(self,
            expr: prim.Expression,
            shape: ShapeType,
            dtype: np.dtype[Any],
            bindings: Optional[Dict[str, Array]] = None,
            tags: TagsType = frozenset()):

        if bindings is None:
            bindings = {}

        super().__init__(shape=shape, dtype=dtype, tags=tags)

        self.expr = expr
        self.bindings = bindings

# }}}


# {{{ einsum

class EinsumAxisDescriptor:
    """
    Records the access pattern of iterating over an array's axis in a
    :class:`Einsum`.
    """
    pass


@dataclass(eq=True, frozen=True)
class ElementwiseAxis(EinsumAxisDescriptor):
    """
    Describes an elementwise access pattern of an array's axis.  In terms of the
    nomenclature used by :class:`IndexLambda`, ``ElementwiseAxis(dim=1)`` would
    correspond to indexing the array's axis as ``_1`` in the expression.
    """
    dim: int


@dataclass(eq=True, frozen=True)
class ReductionAxis(EinsumAxisDescriptor):
    """
    Describes a reduction access pattern of an array's axis.  In terms of the
    nomenclature used by :class:`IndexLambda`, ``ReductionAxis(dim=0)`` would
    correspond to indexing the array's axis as ``_r0`` in the expression.
    """
    dim: int


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
    """
    _fields = Array._fields + ("access_descriptors", "args")
    _mapper_method = "map_einsum"

    def __init__(self,
                 access_descriptors: Tuple[Tuple[EinsumAxisDescriptor, ...], ...],
                 args: Tuple[Array, ...],
                 tags: TagsType = frozenset()):
        super().__init__(tags)
        self.access_descriptors = access_descriptors
        self.args = args

    # type-ignore reason: github.com/python/mypy/issues/1362
    @property  # type: ignore
    @memoize_method
    def shape(self) -> ShapeType:
        iaxis_to_len = {}

        for access_descr, arg in zip(self.access_descriptors,
                                     self.args):
            assert arg.ndim == len(access_descr)
            for arg_axis_len, axis_op in zip(arg.shape, access_descr):
                if isinstance(axis_op, ElementwiseAxis):
                    if axis_op.dim in iaxis_to_len:
                        assert arg.shape[axis_op.dim] == arg_axis_len
                    else:
                        iaxis_to_len[axis_op.dim] = arg_axis_len
                elif isinstance(axis_op, ReductionAxis):
                    pass
                else:
                    raise AssertionError

        assert all(i in iaxis_to_len for i in range(len(iaxis_to_len)))
        return tuple(iaxis_to_len[i] for i in range(len(iaxis_to_len)))

    # type-ignore reason: github.com/python/mypy/issues/1362
    @property  # type: ignore
    @memoize_method
    def dtype(self) -> np.dtype[Any]:
        return np.find_common_type(array_types=[arg.dtype for arg in self.args],
                                    scalar_types=[])


EINSUM_FIRST_INDEX = re.compile(r"^\s*((?P<alpha>[a-zA-Z])|(?P<ellipsis>\.\.\.))\s*")


def _normalize_einsum_out_subscript(subscript: str) -> PMap[str,
                                                            EinsumAxisDescriptor]:
    """
    Normalizes the output subscript of an einsum (provided in the explicit
    mode). Returns a mapping from index name to an instance of
    :class:`ElementwiseAxis`.

    .. testsetup::

        >>> from pytato.array import _normalize_einsum_out_subscript

    .. doctest::

        >>> result = _normalize_einsum_out_subscript("kij")
        >>> sorted(result.keys())
        ['i', 'j', 'k']
        >>> result["i"], result["j"], result["k"]
        (ElementwiseAxis(dim=1), ElementwiseAxis(dim=2), ElementwiseAxis(dim=0))
    """

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

    return pmap({idx: ElementwiseAxis(i)
                 for i, idx in enumerate(normalized_indices)})


def _normalize_einsum_in_subscript(subscript: str,
                                   in_operand: Array,
                                   index_to_descr: PMap[str,
                                                        EinsumAxisDescriptor],
                                   index_to_axis_length: PMap[str,
                                                               ShapeComponent],
                                   ) -> Tuple[Tuple[EinsumAxisDescriptor, ...],
                                              PMap[str, EinsumAxisDescriptor],
                                              PMap[str, ShapeComponent]]:
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

    for iaxis, index_char in enumerate(normalized_indices):
        in_axis_len = in_operand.shape[iaxis]
        if index_char in index_to_descr:
            if index_char in index_to_axis_length:
                seen_axis_len = index_to_axis_length[index_char]
                if not are_shape_components_equal(in_axis_len,
                                                  seen_axis_len):
                    raise ValueError(f"Got conflicting lengths for '{index_char}'"
                                     f" -- {in_axis_len}, {seen_axis_len}.")
            else:
                index_to_axis_length = index_to_axis_length.set(index_char,
                                                                in_axis_len)
        else:
            redn_sr_no = len([descr for descr in index_to_descr.values()
                              if isinstance(descr, ReductionAxis)])
            redn_axis_descr = ReductionAxis(redn_sr_no)
            index_to_descr = index_to_descr.set(index_char, redn_axis_descr)
            index_to_axis_length = index_to_axis_length.set(index_char,
                                                             in_axis_len)

        in_operand_axis_descrs.append(index_to_descr[index_char])

    return (tuple(in_operand_axis_descrs),
            index_to_descr, index_to_axis_length)


def einsum(subscripts: str, *operands: Array) -> Einsum:
    """
    Einstein summation *subscripts* on *operands*.
    """
    if len(operands) == 0:
        raise ValueError("must specify at least one operand")

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
    index_to_axis_length: PMap[str, ShapeComponent] = pmap()
    access_descriptors = []

    for in_spec, in_operand in zip(in_specs, operands):
        access_descriptor, index_to_descr, index_to_axis_length = (
            _normalize_einsum_in_subscript(in_spec,
                                           in_operand,
                                           index_to_descr,
                                           index_to_axis_length))
        access_descriptors.append(access_descriptor)

    return Einsum(tuple(access_descriptors), operands)

# }}}


# {{{ matrix product

class MatrixProduct(Array):
    """A product of two matrices, or a matrix and a vector.

    The semantics of this operation follow PEP 465 [pep465]_, i.e., the Python
    matmul (@) operator.

    .. attribute:: x1
    .. attribute:: x2

    .. [pep465] https://www.python.org/dev/peps/pep-0465/

    """
    _fields = Array._fields + ("x1", "x2")

    _mapper_method = "map_matrix_product"

    def __init__(self,
            x1: Array,
            x2: Array,
            tags: TagsType = frozenset()):
        super().__init__(tags)
        self.x1 = x1
        self.x2 = x2

    @property
    def shape(self) -> ShapeType:
        # FIXME: Broadcasting currently unsupported.
        assert 0 < self.x1.ndim <= 2
        assert 0 < self.x2.ndim <= 2

        if self.x1.ndim == 1 and self.x2.ndim == 1:
            return ()
        elif self.x1.ndim == 1 and self.x2.ndim == 2:
            return (self.x2.shape[1],)
        elif self.x1.ndim == 2 and self.x2.ndim == 1:
            return (self.x1.shape[0],)
        elif self.x1.ndim == 2 and self.x2.ndim == 2:
            return (self.x1.shape[0], self.x2.shape[1])

        raise AssertionError()

    @property
    def dtype(self) -> np.dtype[Any]:
        return cast(_dtype_any, np.result_type(self.x1.dtype, self.x2.dtype))

# }}}


# {{{ stack

class Stack(Array):
    """Join a sequence of arrays along an axis.

    .. attribute:: arrays

        The sequence of arrays to join

    .. attribute:: axis

        The output axis

    """

    _fields = Array._fields + ("arrays", "axis")
    _mapper_method = "map_stack"

    def __init__(self,
            arrays: Tuple[Array, ...],
            axis: int,
            tags: TagsType = frozenset()):
        super().__init__(tags)
        self.arrays = arrays
        self.axis = axis

    @property
    def dtype(self) -> np.dtype[Any]:
        return cast(_dtype_any,
                np.result_type(*(arr.dtype for arr in self.arrays)))

    @property
    def shape(self) -> ShapeType:
        result = list(self.arrays[0].shape)
        result.insert(self.axis, len(self.arrays))
        return tuple(result)

# }}}


# {{{ concatenate

class Concatenate(Array):
    """Join a sequence of arrays along an existing axis.

    .. attribute:: arrays

        An instance of :class:`tuple` of the arrays to join. The arrays must
        have same shape except for the dimension corresponding to *axis*.

    .. attribute:: axis

        The axis along which the *arrays* are to be concatenated.
    """

    _fields = Array._fields + ("arrays", "axis")
    _mapper_method = "map_concatenate"

    def __init__(self,
            arrays: Tuple[Array, ...],
            axis: int,
            tags: TagsType = frozenset()):
        super().__init__(tags)
        self.arrays = arrays
        self.axis = axis

    @property
    def dtype(self) -> np.dtype[Any]:
        return cast(_dtype_any,
                np.result_type(*(arr.dtype for arr in self.arrays)))

    @property
    def shape(self) -> ShapeType:
        from builtins import sum as _sum
        common_axis_len = _sum(ary.shape[self.axis] for ary in self.arrays)

        return (self.arrays[0].shape[:self.axis]
                + (common_axis_len,)
                + self.arrays[0].shape[self.axis+1:])

# }}}


# {{{ index remapping

class IndexRemappingBase(Array):
    """Base class for operations that remap the indices of an array.

    Note that index remappings can also be expressed via
    :class:`~pytato.array.IndexLambda`.

    .. attribute:: array

        The input :class:`~pytato.Array`

    """
    _fields = Array._fields + ("array",)

    def __init__(self,
            array: Array,
            tags: TagsType = frozenset()):
        super().__init__(tags)
        self.array = array

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.array.dtype

# }}}


# {{{ roll

class Roll(IndexRemappingBase):
    """Roll an array along an axis.

    .. attribute:: shift

        Shift amount.

    .. attribute:: axis

        Shift axis.
    """
    _fields = IndexRemappingBase._fields + ("shift", "axis")
    _mapper_method = "map_roll"

    def __init__(self,
            array: Array,
            shift: int,
            axis: int,
            tags: TagsType = frozenset()):
        super().__init__(array, tags)
        self.shift = shift
        self.axis = axis

    @property
    def shape(self) -> ShapeType:
        return self.array.shape

# }}}


# {{{ axis permutation

class AxisPermutation(IndexRemappingBase):
    r"""Permute the axes of an array.

    .. attribute:: axes

        A permutation of the input axes.
    """
    _fields = IndexRemappingBase._fields + ("axes",)
    _mapper_method = "map_axis_permutation"

    def __init__(self,
            array: Array,
            axes: Tuple[int, ...],
            tags: TagsType = frozenset()):
        super().__init__(array, tags)
        self.array = array
        self.axes = axes

    @property
    def shape(self) -> ShapeType:
        result = []
        base_shape = self.array.shape
        for index in self.axes:
            result.append(base_shape[index])
        return tuple(result)

# }}}


# {{{ reshape

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

    _fields = Array._fields + ("array", "newshape", "order")
    _mapper_method = "map_reshape"

    def __init__(self,
            array: Array,
            newshape: Tuple[int, ...],
            order: str,
            tags: TagsType = frozenset()):
        # FIXME: Get rid of this restriction
        assert order == "C"

        super().__init__(array, tags)
        self.newshape = newshape
        self.order = order

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.newshape

# }}}


# {{{ slice

class Slice(IndexRemappingBase):
    """Extracts a slice of constant size from an array.

    .. attribute:: starts
    .. attribute:: stops
    """
    _fields = IndexRemappingBase._fields + ("starts", "stops")
    _mapper_method = "map_slice"

    def __init__(self,
            array: Array,
            starts: Tuple[int, ...],
            stops: Tuple[int, ...],
            tags: TagsType = frozenset()):
        super().__init__(array, tags)

        self.starts = starts
        self.stops = stops

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple((stop-start)
                     for start, stop in zip(self.starts, self.stops))

# }}}


# {{{ base class for arguments

class InputArgumentBase(Array):
    r"""Base class for input arguments.

    .. attribute:: name

        The name by which a value is supplied for the argument once computation
        begins. If None, a unique name will be assigned during code-generation.

    .. note::

        Creating multiple instances of any input argument with the
        same name in an expression is not allowed.
    """

    # The name uniquely identifies this object in the namespace. Therefore,
    # subclasses don't have to update *_fields*.
    _fields = ("name",)

    def __init__(self,
            name: Optional[str],
            tags: TagsType = frozenset()):
        super().__init__(tags=tags)
        self.name = name

    def tagged(self, tags: TagOrIterableType) -> InputArgumentBase:
        raise ValueError("Cannot modify tags")

    def without_tags(self, tags: TagOrIterableType,
                        verify_existence: bool = True) -> InputArgumentBase:
        raise ValueError("Cannot modify tags")

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return self is other

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

    shape: ShapeType
    dtype: np.dtype[Any]


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
    """

    _mapper_method = "map_data_wrapper"

    def __init__(self,
            name: Optional[str],
            data: DataInterface,
            shape: ShapeType,
            tags: TagsType = frozenset()):
        super().__init__(name, tags=tags)

        self.data = data
        self._shape = shape

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

# }}}


# {{{ placeholder

class Placeholder(_SuppliedShapeAndDtypeMixin, InputArgumentBase):
    r"""A named placeholder for an array whose concrete value is supplied by the
    user during evaluation.

    .. attribute:: name
    .. automethod:: __init__
    """

    _mapper_method = "map_placeholder"

    def __init__(self,
            name: Optional[str],
            shape: ShapeType,
            dtype: np.dtype[Any],
            tags: TagsType = frozenset()):
        """Should not be called directly. Use :func:`make_placeholder`
        instead.
        """
        super().__init__(shape=shape,
                dtype=dtype,
                name=name,
                tags=tags)

# }}}


# {{{ size parameter

class SizeParam(InputArgumentBase):
    r"""A named placeholder for a scalar that may be used as a variable in symbolic
    expressions for array sizes.
    """

    _mapper_method = "map_size_param"

    @property
    def shape(self) -> ShapeType:
        return ()

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(np.intp)

# }}}


# {{{ end-user facing

def matmul(x1: Array, x2: Array) -> Array:
    """Matrix multiplication.

    :param x1: first argument
    :param x2: second argument
    """
    from pytato.utils import are_shape_components_equal
    if (
            isinstance(x1, SCALAR_CLASSES)
            or x1.shape == ()
            or isinstance(x2, SCALAR_CLASSES)
            or x2.shape == ()):
        raise ValueError("scalars not allowed as arguments to matmul")

    if len(x1.shape) > 2 or len(x2.shape) > 2:
        raise NotImplementedError("broadcasting not supported")

    if not are_shape_components_equal(x1.shape[-1], x2.shape[0]):
        raise ValueError("dimension mismatch")

    return MatrixProduct(x1, x2)


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
                    "shifing along more than one dimension is unsupported")
        else:
            axis = 0

    if not (0 <= axis < a.ndim):
        raise ValueError("invalid axis")

    if shift == 0:
        return a

    return Roll(a, shift, axis)


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

    return AxisPermutation(a, tuple(axes))


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

    return Stack(tuple(arrays), axis)


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
            raise ValueError("arrays must have the same shape expect along"
                    f" dimension #{axis}.")

    if not (0 <= axis <= arrays[0].ndim):
        raise ValueError("invalid axis")

    return Concatenate(tuple(arrays), axis)


def _make_slice(array: Array, starts: Sequence[int], stops: Sequence[int]) -> Array:
    """Extract a constant-sized slice from an array with constant offsets.

    :param array: input array
    :param starts: a sequence of length *array.ndim* containing slice
        offsets. Must be in bounds.
    :param stops: a sequence of length *array.ndim* containing slice stops
        along each dimension. Must be in bounds and ``>= start`` for each
        dimension.

    .. note::

        Support for slices is currently limited. Among other things, non-trivial
        slices (i.e., not the length of the whole dimension) involving symbolic
        expressions are unsupported.
    """
    if array.ndim != len(starts):
        raise ValueError("'starts' and 'array' do not match in number of dimensions")

    if len(starts) != len(stops):
        raise ValueError("'starts' and 'stops' do not match in number of dimensions")

    for i, (start, stop) in enumerate(zip(starts, stops)):
        symbolic_index = not (
                isinstance(array.shape[i], int)
                and isinstance(start, int)
                and isinstance(stop, int))

        if symbolic_index:
            if not (0 == start and stop == array.shape[i]):
                raise ValueError(
                        "slicing with symbolic dimensions is unsupported")
            continue

        ubound: int = cast(int, array.shape[i])
        if not (0 <= start < ubound):
            raise ValueError("index '%d' of 'begin' out of bounds" % i)

        if not (start <= stop <= ubound):
            raise ValueError("index '%d' of 'size' out of bounds" % i)

    # FIXME: Generate IndexLambda when possible
    return Slice(array, tuple(starts), tuple(stops))


def reshape(array: Array, newshape: Sequence[int], order: str = "C") -> Array:
    """
    :param array: array to be reshaped
    :param newshape: shape of the resulting array
    :param order: ``"C"`` or ``"F"``. Layout order of the result array. Only
        ``"C"`` allowed for now.

    .. note::

        reshapes of arrays with symbolic shapes not yet implemented.
    """
    from pytools import product

    if newshape.count(-1) > 1:
        raise ValueError("can only specify one unknown dimension")

    if not all(isinstance(axis_len, int) for axis_len in array.shape):
        raise ValueError("reshape of arrays with symbolic lengths not allowed")

    if order != "C":
        raise NotImplementedError("Reshapes to a 'F'-ordered arrays")

    newshape_explicit = []

    for new_axislen in newshape:
        if not isinstance(new_axislen, int):
            raise ValueError("Symbolic reshapes not allowed.")

        if not(new_axislen > 0 or new_axislen == -1):
            raise ValueError("newshape should be either sequence of positive ints or"
                    " -1")

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

    return Reshape(array, tuple(newshape_explicit), order)


# {{{ make_dict_of_named_arrays

def make_dict_of_named_arrays(data: Dict[str, Array]) -> DictOfNamedArrays:
    """Make a :class:`DictOfNamedArrays` object.

    :param data: member keys and arrays
    """
    return DictOfNamedArrays(data)

# }}}


def make_placeholder(shape: ConvertibleToShape,
        dtype: Any,
        name: Optional[str] = None,
        tags: TagsType = frozenset()) -> Placeholder:
    """Make a :class:`Placeholder` object.

    :param name:       name of the placeholder array, generated automatically
                       if not given
    :param shape:      shape of the placeholder array
    :param dtype:      dtype of the placeholder array
                       (must be convertible to :class:`numpy.dtype`)
    :param tags:       implementation tags
    """
    _check_identifier(name)
    shape = normalize_shape(shape)
    dtype = np.dtype(dtype)

    return Placeholder(name, shape, dtype, tags)


def make_size_param(name: str,
        tags: TagsType = frozenset()) -> SizeParam:
    """Make a :class:`SizeParam`.

    Size parameters may be used as variables in symbolic expressions for array
    sizes.

    :param name:       name
    :param tags:       implementation tags
    """
    _check_identifier(name)
    return SizeParam(name, tags=tags)


def make_data_wrapper(data: DataInterface,
        name: Optional[str] = None,
        shape: Optional[ConvertibleToShape] = None,
        tags: TagsType = frozenset()) -> DataWrapper:
    """Make a :class:`DataWrapper`.

    :param data:       an instance obeying the :class:`DataInterface`
    :param name:       an optional name, generated automatically if not given
    :param shape:      optional shape of the array, inferred from *data* if not given
    :param tags:       implementation tags
    """
    _check_identifier(name)
    if shape is None:
        shape = data.shape

    shape = normalize_shape(shape)

    return DataWrapper(name, data, shape, tags)

# }}}


# {{{ math functions

def _apply_elem_wise_func(inputs: Tuple[ArrayOrScalar],
                          func_name: str,
                          ret_dtype: Optional[_dtype_any] = None
                          ) -> ArrayOrScalar:
    if all(isinstance(x, SCALAR_CLASSES) for x in inputs):
        np_func = getattr(np, func_name)
        return np_func(inputs)  # type: ignore

    if not inputs:
        raise ValueError("at least one argument must be present")

    shape = None

    sym_args = []
    bindings = {}
    for index, inp in enumerate(inputs):
        if isinstance(inp, Array):
            if inp.dtype.kind not in ["f", "c"]:
                raise ValueError("only floating-point or complex "
                        "arguments supported")

            if shape is None:
                shape = inp.shape
            elif inp.shape != shape:
                # FIXME: merge this logic with arithmetic, so that broadcasting
                # is implemented properly
                raise NotImplementedError("broadcasting in function application")

            if ret_dtype is None:
                ret_dtype = inp.dtype

            bindings[f"in_{index}"] = inp
            sym_args.append(
                    prim.Subscript(var(f"in_{index}"),
                        tuple(var(f"_{i}") for i in range(len(shape)))))
        else:
            sym_args.append(inp)

    assert shape is not None
    assert ret_dtype is not None

    return IndexLambda(
            prim.Call(var(f"pytato.c99.{func_name}"), tuple(sym_args)),
            shape, ret_dtype, bindings)


def abs(x: Array) -> ArrayOrScalar:
    if x.dtype.kind == "c":
        result_dtype = np.empty(0, dtype=x.dtype).real.dtype
    else:
        result_dtype = x.dtype

    return _apply_elem_wise_func((x,), "abs", ret_dtype=result_dtype)


def sqrt(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "sqrt")


def sin(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "sin")


def cos(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "cos")


def tan(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "tan")


def arcsin(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "asin")


def arccos(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "acos")


def arctan(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "atan")


def conj(x: Array) -> ArrayOrScalar:
    if x.dtype.kind != "c":
        return x
    return _apply_elem_wise_func((x,), "conj")


def arctan2(y: Array, x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((y, x), "atan2")  # type:ignore


def sinh(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "sinh")


def cosh(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "cosh")


def tanh(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "tanh")


def exp(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "exp")


def log(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "log")


def log10(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "log10")


def isnan(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "isnan", np.dtype(np.int32))


def real(x: Array) -> ArrayOrScalar:
    if x.dtype.kind == "c":
        result_dtype = np.empty(0, dtype=x.dtype).real.dtype
    else:
        return x
    return _apply_elem_wise_func((x,), "real", ret_dtype=result_dtype)


def imag(x: Array) -> ArrayOrScalar:
    if x.dtype.kind == "c":
        result_dtype = np.empty(0, dtype=x.dtype).real.dtype
    else:
        return zeros(x.shape, dtype=x.dtype)
    return _apply_elem_wise_func((x,), "imag", ret_dtype=result_dtype)

# }}}


# {{{ full

def full(shape: ConvertibleToShape, fill_value: ScalarType,
         dtype: Any, order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to *fill_value*.
    """
    if order != "C":
        raise ValueError("Only C-ordered arrays supported for now.")

    shape = normalize_shape(shape)
    dtype = np.dtype(dtype)
    return IndexLambda(dtype.type(fill_value), shape, dtype, {})


def zeros(shape: ConvertibleToShape, dtype: Any = float,
        order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to 0.
    """
    # https://github.com/python/mypy/issues/3186
    return full(shape, 0, dtype)  # type: ignore


def ones(shape: ConvertibleToShape, dtype: Any = float,
        order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to 1.
    """
    # https://github.com/python/mypy/issues/3186
    return full(shape, 1, dtype)  # type: ignore

# }}}


# {{{ comparison operator

def _compare(x1: ArrayOrScalar, x2: ArrayOrScalar, which: str) -> Union[Array, bool]:
    # https://github.com/python/mypy/issues/3186
    import pytato.utils as utils
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.Comparison(x, which, y),
                                     lambda x, y: np.bool8)  # type: ignore


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
    import pytato.utils as utils
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.LogicalOr((x, y)),
                                     lambda x, y: np.bool8)  # type: ignore


def logical_and(x1: ArrayOrScalar, x2: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns the element-wise logical AND of *x1* and *x2*.
    """
    # https://github.com/python/mypy/issues/3186
    import pytato.utils as utils
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.LogicalAnd((x, y)),
                                     lambda x, y: np.bool8)  # type: ignore


def logical_not(x: ArrayOrScalar) -> Union[Array, bool]:
    """
    Returns the element-wise logical NOT of *x*.
    """
    if isinstance(x, SCALAR_CLASSES):
        # https://github.com/python/mypy/issues/3186
        return np.logical_not(x)  # type: ignore

    assert isinstance(x, Array)

    from pytato.utils import with_indices_for_broadcasted_shape
    return IndexLambda(with_indices_for_broadcasted_shape(prim.Variable("_in0"),
                                                          x.shape,
                                                          x.shape),
                       shape=x.shape,
                       dtype=np.bool8,
                       bindings={"_in0": x})

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
        return x if condition else y  # type: ignore

    # {{{ find dtype

    x_dtype = x.dtype if isinstance(x, Array) else np.dtype(type(x))
    y_dtype = y.dtype if isinstance(y, Array) else np.dtype(type(y))
    dtype = np.find_common_type([x_dtype, y_dtype], [])

    # }}}

    result_shape = utils.get_shape_after_broadcasting([condition, x, y])

    bindings: Dict[str, Array] = {}

    expr1 = utils.update_bindings_and_get_broadcasted_expr(condition, "_in0",
                                                           bindings, result_shape)
    expr2 = utils.update_bindings_and_get_broadcasted_expr(x, "_in1", bindings,
                                                           result_shape)
    expr3 = utils.update_bindings_and_get_broadcasted_expr(y, "_in2", bindings,
                                                           result_shape)

    return IndexLambda(prim.If(expr1, expr2, expr3),
            shape=result_shape,
            dtype=dtype,
            bindings=bindings)

# }}}


# {{{ (max|min)inimum

def maximum(x1: ArrayOrScalar, x2: ArrayOrScalar) -> ArrayOrScalar:
    """
    Returns the elementwise maximum of *x1*, *x2*. *x1*, *x2* being
    array-like objects that could be broadcasted together. NaNs are propagated.
    """
    # https://github.com/python/mypy/issues/3186
    return where(logical_or(isnan(x1), isnan(x2)), np.NaN,  # type: ignore
                 where(greater(x1, x2), x1, x2))


def minimum(x1: ArrayOrScalar, x2: ArrayOrScalar) -> ArrayOrScalar:
    """
    Returns the elementwise minimum of *x1*, *x2*. *x1*, *x2* being
    array-like objects that could be broadcasted together. NaNs are propagated.
    """
    # https://github.com/python/mypy/issues/3186
    return where(logical_or(isnan(x1), isnan(x2)), np.NaN,  # type: ignore
                 where(less(x1, x2), x1, x2))

# }}}


# {{{ make_index_lambda

def make_index_lambda(
        expression: Union[str, ScalarExpression],
        bindings: Dict[str, Array],
        shape: ShapeType,
        dtype: Any) -> IndexLambda:
    if isinstance(expression, str):
        raise NotImplementedError

    # {{{ sanity checks

    from pytato.scalar_expr import get_dependencies
    unknown_dep = (get_dependencies(expression, include_idx_lambda_indices=False)
            - set(bindings))

    for dep in unknown_dep:
        raise ValueError(f"Unknown variable '{dep}' in the expression.")

    # }}}

    return IndexLambda(expr=expression,
                       bindings=bindings,
                       shape=shape,
                       dtype=dtype)

# }}}


# {{{ reductions

def _normalize_reduction_axes(
        shape: ShapeType,
        reduction_axes: Optional[Union[int, Tuple[int]]]
        ) -> Tuple[ShapeType, Tuple[int, ...]]:
    """
    Returns a :class:`tuple` of ``(new_shape, normalized_redn_axes)``, where
    *new_shape* is the shape of the ndarray after the axes corresponding to
    *reduction_axes* are reduced and *normalized_redn_axes* is a :class:`tuple`
    of axes indices over which reduction is to be performed.

    :arg reduction_axes: Axis indices over which reduction is to be performed.
        If *reduction_axes* is None, the reduction is performed over all
        axes.
    :arg shape: Shape of the array over which reduction is to be
        performed
    """
    if reduction_axes is None:
        return (), tuple(range(len(shape)))

    if isinstance(reduction_axes, int):
        reduction_axes = reduction_axes,

    if not isinstance(reduction_axes, tuple):
        raise TypeError("Reduction axes expected to be of type 'NoneType', 'int'"
                f" or 'tuple'. (Got {type(reduction_axes)})")

    for axis in reduction_axes:
        if not (0 <= axis < len(shape)):
            raise ValueError(f"{axis} is out of bounds for array of dimension"
                    f" {len(shape)}.")

    new_shape = tuple([axis_len
        for i, axis_len in enumerate(shape)
        if i not in reduction_axes])
    return new_shape, reduction_axes


def _get_reduction_indices_bounds(shape: ShapeType,
        axes: Tuple[int, ...]) -> Tuple[
                Sequence[ScalarExpression],
                Dict[str, Tuple[ScalarExpression, ScalarExpression]]]:
    """Given *shape* and reduction axes *axes*, produce a list of inames
    ``indices`` named appropriately for reduction inames.
    Also fill a dictionary with bounds for reduction inames
    ``redn_bounds = {red_iname: (lower_bound, upper_bound)}``,
    where the bounds are given as a Python-style half-open interval.
    :returns: ``indices, redn_bounds``
    """
    indices: List[prim.Variable] = []
    redn_bounds: Dict[str, Tuple[ScalarExpression, ScalarExpression]] = {}

    n_out_dims = 0
    n_redn_dims = 0
    for idim, axis_len in enumerate(shape):
        if idim in axes:
            if not isinstance(axis_len, int):
                # TODO: add bindings for shape array expressions
                raise NotImplementedError("Parametric shapes for reduction axes"
                                          " not yet supported.")

            idx = f"_r{n_redn_dims}"
            indices.append(prim.Variable(idx))
            redn_bounds[idx] = (0, axis_len)
            n_redn_dims += 1
        else:
            indices.append(prim.Variable(f"_{n_out_dims}"))
            n_out_dims += 1

    from pyrsistent import pmap

    # insufficient type annotation in pyrsistent
    return indices, pmap(redn_bounds)  # type: ignore


def _make_reduction_lambda(op: str, a: Array,
                      axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Return a :class:`IndexLambda` that performs reduction over the *axis* axes
    of *a* with the reduction op *op*.

    :arg op: The reduction operation to perform.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes over which the reduction is to be performed. If axis is
        *None*, perform reduction over all of *a*'s axes.
    """
    new_shape, axes = _normalize_reduction_axes(a.shape, axis)
    del axis
    indices, redn_bounds = _get_reduction_indices_bounds(a.shape, axes)

    return make_index_lambda(
            Reduce(
                prim.Subscript(prim.Variable("in"), tuple(indices)),
                op,
                redn_bounds),
            {"in": a},
            new_shape,
            a.dtype)


def sum(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Sums array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be sum-reduced.
        Defaults to all axes of the input array.
    """
    return _make_reduction_lambda("sum", a, axis)


def amax(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Returns the max of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be max-reduced.
        Defaults to all axes of the input array.
    """
    return _make_reduction_lambda("max", a, axis)


def amin(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Returns the min of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be min-reduced.
        Defaults to all axes of the input array.
    """
    return _make_reduction_lambda("min", a, axis)


def prod(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Returns the product of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be product-reduced.
        Defaults to all axes of the input array.
    """
    return _make_reduction_lambda("product", a, axis)

# }}}

# vim: foldmethod=marker
