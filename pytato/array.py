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

.. autoclass:: Namespace
.. autoclass:: Array
.. autoclass:: DictOfNamedArrays

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
.. autofunction:: sin
.. autofunction:: cos
.. autofunction:: tan
.. autofunction:: arcsin
.. autofunction:: arccos
.. autofunction:: arctan
.. autofunction:: sinh
.. autofunction:: cosh
.. autofunction:: tanh
.. autofunction:: exp
.. autofunction:: log
.. autofunction:: log10
.. autofunction:: isnan
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

Supporting Functionality
------------------------

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
.. autoclass:: AdditionalOutput

Built-in Expression Nodes
-------------------------

.. autoclass:: IndexLambda
.. autoclass:: Einsum
.. autoclass:: MatrixProduct
.. autoclass:: LoopyFunction
.. autoclass:: Stack
.. autoclass:: Concatenate
.. autoclass:: AttributeLookup

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

.. class:: ConvertibleToShape

.. autofunction:: make_dict_of_named_arrays
.. autofunction:: make_placeholder
.. autofunction:: make_size_param
.. autofunction:: make_data_wrapper

Aliases
-------

(This section exists because Sphinx, our documentation tool, can't (yet)
canonicalize type references. Once Sphinx 4.0 is released, we should use the
``:canonical:`` option here.)

.. class:: Namespace

    Should be referenced as :class:`pytato.Namespace`.

.. class:: Array

    Should be referenced as :class:`pytato.Array`.

.. class:: DictOfNamedArrays

    Should be referenced as :class:`pytato.DictOfNamedArrays`.
"""

# }}}

from functools import partialmethod
from numbers import Number
import operator
from typing import (
        Optional, Callable, ClassVar, Dict, Any, Mapping, Iterator, Tuple, Union,
        Protocol, Sequence, cast, TYPE_CHECKING, List)

import numpy as np
import pymbolic.primitives as prim
from pymbolic import var
from pytools import is_single_valued, memoize_method, UniqueNameGenerator
from pytools.tag import (Tag, Taggable, UniqueTag, TagOrIterableType,
    TagsType, tag_dataclass)

import pytato.scalar_expr as scalar_expr
from pytato.scalar_expr import ScalarExpression, IntegralScalarExpression
import re


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


# {{{ namespace

class Namespace(Mapping[str, "Array"]):
    # Possible future extension: .parent attribute
    r"""
    Represents a mapping from :term:`identifier` strings to
    :term:`array expression`\ s or *None*, where *None* indicates that the name
    may not be used.  (:class:`pytato.array.Placeholder` instances register
    their names in this way to avoid ambiguity.)

    .. attribute:: name_gen
    .. automethod:: __contains__
    .. automethod:: __getitem__
    .. automethod:: __iter__
    .. automethod:: __len__
    .. automethod:: assign
    .. automethod:: copy
    .. automethod:: ref
    """
    name_gen: UniqueNameGenerator

    def __init__(self) -> None:
        self._symbol_table: Dict[str, Array] = {}
        self.name_gen = UniqueNameGenerator()

    def __contains__(self, name: object) -> bool:
        return name in self._symbol_table

    def __getitem__(self, name: str) -> Array:
        item = self._symbol_table[name]
        return item

    def __iter__(self) -> Iterator[str]:
        return iter(self._symbol_table)

    def __len__(self) -> int:
        return len(self._symbol_table)

    def copy(self) -> Namespace:
        from pytato.transform import CopyMapper, copy_namespace
        return copy_namespace(self, CopyMapper(Namespace()), only_names=None)

    def assign(self, name: str, value: Array) -> str:
        """Declare a new array.

        :param name: a Python identifier
        :param value: the array object

        :returns: *name*
        """
        if name in self._symbol_table:
            raise ValueError(f"'{name}' is already assigned")
        if not self.name_gen.is_name_conflicting(name):
            self.name_gen.add_name(name)
        self._symbol_table[name] = value

        return name

    def ref(self, name: str) -> Array:
        """
        :returns: An :term:`array expression` referring to *name*.
        """

        value = self[name]

        var_ref = prim.Variable(name)
        if value.shape:
            var_ref = var_ref[tuple("_%d" % i for i in range(len(value.shape)))]

        return IndexLambda(
                self, expr=var_ref, shape=value.shape,
                dtype=value.dtype)

    def remove_out_of_scope_data_wrappers(self) -> None:
        import sys
        data_wrappers = {name
                         for name in self
                         if (isinstance(self[name], DataWrapper)
                             and name.startswith("_pt"))}
        out_of_scope_dws = {name
                            for name in data_wrappers
                            if sys.getrefcount(self[name]) <= 2}
        for k in out_of_scope_dws:
            del self._symbol_table[k]

# }}}


# {{{ shape

ShapeType = Tuple[IntegralScalarExpression, ...]
ConvertibleToShapeComponent = Union[int, prim.Expression, str]
ConvertibleToShape = Union[
        str,
        IntegralScalarExpression,
        Tuple[ConvertibleToShapeComponent, ...]]


def _check_identifier(s: str, ns: Optional[Namespace] = None) -> bool:
    if not s.isidentifier():
        raise ValueError(f"'{s}' is not a valid identifier")

    if ns is not None:
        if s not in ns:
            raise ValueError(f"'{s}' is not known in the namespace")

    return True


class _ShapeChecker(scalar_expr.WalkMapper):
    def __init__(self, ns: Optional[Namespace] = None):
        super().__init__()
        self.ns = ns

    def map_variable(self, expr: prim.Variable) -> None:
        _check_identifier(expr.name, self.ns)
        super().map_variable(expr)


def normalize_shape(
        shape: ConvertibleToShape,
        ns: Optional[Namespace] = None
        ) -> ShapeType:
    """
    :param ns: if a namespace is given, extra checks are performed to
               ensure that expressions are well-defined.
    """
    def normalize_shape_component(
            s: ConvertibleToShapeComponent) -> ScalarExpression:
        if isinstance(s, str):
            s = scalar_expr.parse(s)

        if isinstance(s, int):
            if s < 0:
                raise ValueError(f"size parameter must be nonnegative (got '{s}')")

        elif isinstance(s, prim.Expression):
            # TODO: check expression affine-ness
            _ShapeChecker()(s)

        return s

    if isinstance(shape, str):
        shape = scalar_expr.parse(shape)

    from numbers import Number
    if isinstance(shape, (Number, prim.Expression)):
        shape = (shape,)

    # https://github.com/python/mypy/issues/3186
    return tuple(normalize_shape_component(s) for s in shape)  # type: ignore

# }}}


# {{{ array inteface

SliceItem = Union[int, slice, None, EllipsisType]
DtypeOrScalar = Union[_dtype_any, Number]


def _truediv_result_type(arg1: DtypeOrScalar, arg2: DtypeOrScalar) -> np.dtype[Any]:
    dtype = np.result_type(arg1, arg2)
    # See: test_true_divide in numpy/core/tests/test_ufunc.py
    if dtype.kind in "iu":
        return np.dtype(np.float64)
    else:
        return cast(_dtype_any, dtype)


class Array(Taggable):
    """
    A base class (abstract interface + supplemental functionality) for lazily
    evaluating array expressions. The interface seeks to maximize :mod:`numpy`
    compatibility, though not at all costs.

    Objects of this type are hashable and support structural equality
    comparison (and are therefore immutable).

    .. note::

        Hashability and equality testing *does* break :mod:`numpy`
        compatibility, purposefully so.

    FIXME: Point out our equivalent for :mod:`numpy`'s ``==``.

    .. attribute:: namespace

        A (mutable) instance of :class:`~pytato.Namespace` containing the
        names used in the computation. All arrays in a
        computation share the same namespace.

    .. attribute:: shape

        Identifiers (:class:`pymbolic.primitives.Variable`) refer to names from
        :attr:`namespace`.  A tuple of integers or :mod:`pymbolic` expressions.
        Shape may be (at most affinely) symbolic in these
        identifiers.

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

        Inherits from :class:`pytools.Taggable`.

    .. automethod:: named
    .. automethod:: tagged
    .. automethod:: without_tag

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

    Derived attributes:

    .. attribute:: ndim

    """
    _mapper_method: ClassVar[str]
    # A tuple of field names. Fields must be equality comparable and
    # hashable. Dicts of hashable keys and values are also permitted.
    _fields: ClassVar[Tuple[str, ...]] = ("shape", "dtype", "tags")

    @property
    def namespace(self) -> Namespace:
        raise NotImplementedError

    @property
    def shape(self) -> ShapeType:
        raise NotImplementedError

    @property
    def size(self) -> ScalarExpression:
        from pytools import product
        return product(self.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        raise NotImplementedError

    def named(self, name: str) -> Array:
        return self.namespace.ref(self.namespace.assign(name, self))

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
            return IndexLambda(self.namespace,
                    expr,
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
                and self.namespace is other.namespace
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
            other: Union[Array, Number],
            get_result_type: Callable[[DtypeOrScalar, DtypeOrScalar], np.dtype[Any]] = np.result_type,  # noqa
            reverse: bool = False) -> Array:

        # {{{ sanity checks

        if not isinstance(other, (Array, Number)):
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
        return IndexLambda(self.namespace,
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

    def __pos__(self) -> Array:
        return self

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


@tag_dataclass
class AdditionalOutput(UniqueTag):
    """
    .. attribute:: field
    .. attribute:: prefix
    """

    field: object
    prefix: str


# {{{ dict of named arrays

class NamedArray(Array):
    _fields = Array._fields + ("dict_of_named_arrays", "name")
    _mapper_method = "map_named_array"

    def __init__(self,
            dict_of_named_arrays: DictOfNamedArrays,
            name: str,
            tags: TagsType = frozenset()) -> None:
        super().__init__(tags=tags)
        self.dict_of_named_arrays = dict_of_named_arrays
        self.name = name

    @property
    def expr(self) -> Array:
        return self.dict_of_named_arrays._data[self.name]

    @property
    def shape(self) -> ShapeType:
        return self.expr.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.expr.dtype

    @property
    def namespace(self) -> Namespace:
        return self.dict_of_named_arrays.namespace


class DictOfNamedArrays(Mapping[str, NamedArray]):
    """A container that maps valid Python identifiers
    to instances of :class:`Array`. May occur as a result
    type of array computations.

    .. automethod:: __init__
    .. automethod:: __contains__
    .. automethod:: __getitem__
    .. automethod:: __iter__
    .. automethod:: __len__

    .. note::

        This container deliberately does not implement
        arithmetic.
    """
    _mapper_method = "map_dict_of_named_arrays"

    def __init__(self, data: Dict[str, Array]):
        self._named_arrays = {name: NamedArray(self, name)
                              for name in data}
        self._data = data

    @property
    def namespace(self) -> Namespace:
        return next(iter(self._data.values())).namespace

    def __contains__(self, name: object) -> bool:
        return name in self.namespace

    def __getitem__(self, name: str) -> NamedArray:
        return self._named_arrays[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._named_arrays)

    def __len__(self) -> int:
        return len(self._named_arrays)

    @memoize_method
    def __hash__(self) -> int:
        return hash(frozenset(self._data.items()))

# }}}


# {{{ index lambda

class IndexLambda(_SuppliedShapeAndDtypeMixin, Array):
    r"""Represents an array that can be computed by evaluating
    :attr:`expr` for every value of the input indices. The
    input indices are represented by
    :class:`~pymbolic.primitives.Variable`\ s with names ``_1``,
    ``_2``, and so on.

    .. attribute:: namespace

    .. attribute:: expr

        A scalar-valued :mod:`pymbolic` expression such as
        ``a[_1] + b[_2, _1]``.

        Identifiers in the expression are resolved, in
        order, by lookups in :attr:`bindings`, then in
        :attr:`namespace`.

        Scalar functions in this expression must
        be identified by a dotted name representing
        a Python object (e.g. ``pytato.c99.sin``).

    .. attribute:: bindings

        A :class:`dict` mapping strings that are valid
        Python identifiers to objects implementing
        the :class:`Array` interface, making array
        expressions available for use in
        :attr:`expr`.

    .. automethod:: is_reference
    """

    _fields = Array._fields + ("expr", "bindings")
    _mapper_method = "map_index_lambda"

    def __init__(self,
            namespace: Namespace,
            expr: prim.Expression,
            shape: ShapeType,
            dtype: np.dtype[Any],
            bindings: Optional[Dict[str, Array]] = None,
            tags: TagsType = frozenset()):

        if bindings is None:
            bindings = {}

        super().__init__(shape=shape, dtype=dtype, tags=tags)

        self._namespace = namespace
        self.expr = expr
        self.bindings = bindings

    @property
    def namespace(self) -> Namespace:
        return self._namespace

    @memoize_method
    def is_reference(self) -> bool:
        # FIXME: Do we want a specific 'reference' node to make all this
        # checking unnecessary?

        if isinstance(self.expr, prim.Subscript):
            assert isinstance(self.expr.aggregate, prim.Variable)
            name = self.expr.aggregate.name
            index = self.expr.index
        elif isinstance(self.expr, prim.Variable):
            name = self.expr.aggregate.name
            index = ()
        else:
            return False

        if index != tuple(var("_%d" % i) for i in range(len(self.shape))):
            return False

        try:
            val = self.namespace[name]
        except KeyError:
            assert name in self.bindings
            return False

        if self.shape != val.shape:
            return False
        if self.dtype != val.dtype:
            return False

        return True

# }}}


# {{{ einsum

class Einsum(Array):
    """
    """

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
    def namespace(self) -> Namespace:
        return self.x1.namespace

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

        assert False

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
    def namespace(self) -> Namespace:
        return self.arrays[0].namespace

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
    def namespace(self) -> Namespace:
        return self.arrays[0].namespace

    @property
    def dtype(self) -> np.dtype[Any]:
        return cast(_dtype_any,
                np.result_type(*(arr.dtype for arr in self.arrays)))

    @property
    def shape(self) -> ShapeType:
        from functools import reduce
        import operator

        common_axis_len = reduce(operator.add, (ary.shape[self.axis]
                                                for ary in self.arrays))

        return (self.arrays[0].shape[:self.axis]
                + (common_axis_len,)
                + self.arrays[0].shape[self.axis+1:])

# }}}


# {{{ attribute lookup

class AttributeLookup(Array):
    """An expression node to extract an array from a :class:`DictOfNamedArrays`.

    .. warning::

        Not yet implemented.
    """
    pass

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

    @property
    def namespace(self) -> Namespace:
        return self.array.namespace

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
        begins.

        The name is also implicitly :meth:`~pytato.Namespace.assign`\ ed in the
        :class:`~pytato.Namespace`.

    .. note::

        Creating multiple instances of any input argument with the same name
        and within the same :class:`~pytato.Namespace` is not allowed.
    """

    # The name uniquely identifies this object in the namespace. Therefore,
    # subclasses don't have to update *_fields*.
    _fields = ("name",)

    def __init__(self,
            namespace: Namespace,
            name: str,
            tags: TagsType = frozenset()):
        if name is None:
            raise ValueError("Must have explicit name")

        # Publish our name to the namespace.
        namespace.assign(name, self)

        self._namespace = namespace
        super().__init__(tags=tags)
        self.name = name

    @property
    def namespace(self) -> Namespace:
        return self._namespace

    def tagged(self, tags: TagOrIterableType) -> InputArgumentBase:
        raise ValueError("Cannot modify tags")

    def without_tags(self, tags: TagOrIterableType,
                        verify_existence: bool = True) -> InputArgumentBase:
        raise ValueError("Cannot modify tags")

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
            namespace: Namespace,
            name: str,
            data: DataInterface,
            shape: ShapeType,
            tags: TagsType = frozenset()):
        super().__init__(namespace, name, tags=tags)

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
            namespace: Namespace,
            name: str,
            shape: ShapeType,
            dtype: np.dtype[Any],
            tags: TagsType = frozenset()):
        """Should not be called directly. Use :func:`make_placeholder`
        instead.
        """
        super().__init__(shape=shape,
                dtype=dtype,
                namespace=namespace,
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
    if (
            isinstance(x1, Number)
            or x1.shape == ()
            or isinstance(x2, Number)
            or x2.shape == ()):
        raise ValueError("scalars not allowed as arguments to matmul")

    if x1.namespace is not x2.namespace:
        raise ValueError("namespace mismatch")

    if len(x1.shape) > 2 or len(x2.shape) > 2:
        raise NotImplementedError("broadcasting not supported")

    if x1.shape[-1] != x2.shape[0]:
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

       >>> arrays = [pt.zeros(3)] * 4
       >>> pt.stack(arrays, axis=0).shape
       (4, 3)

    :param arrays: a finite sequence, each of whose elements is an
        :class:`Array` of the same shape
    :param axis: the position of the new axis, which will have length
        *len(arrays)*
    """

    if not arrays:
        raise ValueError("need at least one array to stack")

    for array in arrays[1:]:
        if array.shape != arrays[0].shape:
            raise ValueError("arrays must have the same shape")

    if not (0 <= axis <= arrays[0].ndim):
        raise ValueError("invalid axis")

    return Stack(tuple(arrays), axis)


def concatenate(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Join a sequence of arrays along an existing axis.

    Example::

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

    if not all(array.namespace is arrays[0].namespace for array in arrays):
        raise ValueError("arrays must belong to the same namespace.")

    def shape_except_axis(ary: Array) -> Tuple[int, ...]:
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


def make_dict_of_named_arrays(data: Dict[str, Array]) -> DictOfNamedArrays:
    """Make a :class:`DictOfNamedArrays` object and ensure that all arrays
    share the same namespace.

    :param data: member keys and arrays
    """
    if not is_single_valued(ary.namespace for ary in data.values()):
        raise ValueError("arrays do not have same namespace")

    return DictOfNamedArrays(data)


def make_placeholder(namespace: Namespace,
        shape: ConvertibleToShape,
        dtype: Any,
        name: Optional[str] = None,
        tags: TagsType = frozenset()) -> Placeholder:
    """Make a :class:`Placeholder` object.

    :param namespace:  namespace of the placeholder array
    :param name:       name of the placeholder array, generated automatically
                       if not given
    :param shape:      shape of the placeholder array
    :param dtype:      dtype of the placeholder array
                       (must be convertible to :class:`numpy.dtype`)
    :param tags:       implementation tags
    """
    if name is None:
        name = namespace.name_gen("_pt_in")

    if not name.isidentifier():
        raise ValueError(f"'{name}' is not a Python identifier")

    shape = normalize_shape(shape, namespace)
    dtype = np.dtype(dtype)

    return Placeholder(namespace, name, shape, dtype, tags)


def make_size_param(namespace: Namespace,
        name: str,
        tags: TagsType = frozenset()) -> SizeParam:
    """Make a :class:`SizeParam`.

    Size parameters may be used as variables in symbolic expressions for array
    sizes.

    :param namespace:  namespace
    :param name:       name
    :param tags:       implementation tags
    """
    if name is None:
        raise ValueError("SizeParam instances must have a name")

    if not name.isidentifier():
        raise ValueError(f"'{name}' is not a Python identifier")

    return SizeParam(namespace, name, tags=tags)


def make_data_wrapper(namespace: Namespace,
        data: DataInterface,
        name: Optional[str] = None,
        shape: Optional[ConvertibleToShape] = None,
        tags: TagsType = frozenset()) -> DataWrapper:
    """Make a :class:`DataWrapper`.

    :param namespace:  namespace
    :param data:       an instance obeying the :class:`DataInterface`
    :param name:       an optional name, generated automatically if not given
    :param shape:      optional shape of the array, inferred from *data* if not given
    :param tags:       implementation tags
    """
    namespace.remove_out_of_scope_data_wrappers()

    if name is None:
        name = namespace.name_gen("_pt_data")

    if not name.isidentifier():
        raise ValueError(f"'{name}' is not a Python identifier")

    if shape is None:
        shape = data.shape

    shape = normalize_shape(shape, namespace)

    return DataWrapper(namespace, name, data, shape, tags)

# }}}


# {{{ math functions

def _apply_elem_wise_func(x: Array, func_name: str,
                          ret_dtype: Optional[_dtype_any] = None) -> IndexLambda:
    if x.dtype.kind != "f":
        raise ValueError(f"'{func_name}' does not support '{x.dtype}' arrays.")
    if ret_dtype is None:
        ret_dtype = x.dtype

    expr = prim.Call(
            var(f"pytato.c99.{func_name}"),
            (prim.Subscript(var("in"),
                tuple(var(f"_{i}") for i in range(len(x.shape)))),))
    return IndexLambda(x.namespace, expr, x.shape, ret_dtype, {"in": x})


def abs(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "abs")


def sin(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "sin")


def cos(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "cos")


def tan(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "tan")


def arcsin(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "asin")


def arccos(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "acos")


def arctan(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "atan")


def sinh(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "sinh")


def cosh(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "cosh")


def tanh(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "tanh")


def exp(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "exp")


def log(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "log")


def log10(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "log10")


def isnan(x: Array) -> IndexLambda:
    return _apply_elem_wise_func(x, "isnan", np.dtype(np.int32))

# }}}


# {{{ full

def full(namespace: Namespace, shape: ConvertibleToShape, fill_value: Number,
        dtype: Any, order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to *fill_value*.
    """
    if order != "C":
        raise ValueError("Only C-ordered arrays supported for now.")

    shape = normalize_shape(shape, namespace)
    dtype = np.dtype(dtype)
    return IndexLambda(namespace, dtype.type(fill_value), shape, dtype, {})


def zeros(namespace: Namespace, shape: ConvertibleToShape, dtype: Any = float,
        order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to 0.
    """
    # https://github.com/python/mypy/issues/3186
    return full(namespace, shape, 0, dtype)  # type: ignore


def ones(namespace: Namespace, shape: ConvertibleToShape, dtype: Any = float,
        order: str = "C") -> Array:
    """
    Returns an array of shape *shape* with all entries equal to 1.
    """
    # https://github.com/python/mypy/issues/3186
    return full(namespace, shape, 1, dtype)  # type: ignore

# }}}


# {{{ comparison operator

def _compare(x1: Union[Array, Number], x2: Union[Array, Number],
             which: str) -> Union[Array, bool]:
    # https://github.com/python/mypy/issues/3186
    import pytato.utils as utils
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.Comparison(x, which, y),
                                     lambda x, y: np.bool8)  # type: ignore


def equal(x1: Union[Array, Number],
          x2: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns (x1 == x2) element-wise.
    """
    return _compare(x1, x2, "==")


def not_equal(x1: Union[Array, Number],
              x2: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns (x1 != x2) element-wise.
    """
    return _compare(x1, x2, "!=")


def less(x1: Union[Array, Number],
         x2: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns (x1 < x2) element-wise.
    """
    return _compare(x1, x2, "<")


def less_equal(x1: Union[Array, Number],
               x2: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns (x1 <= x2) element-wise.
    """
    return _compare(x1, x2, "<=")


def greater(x1: Union[Array, Number],
            x2: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns (x1 > x2) element-wise.
    """
    return _compare(x1, x2, ">")


def greater_equal(x1: Union[Array, Number],
                  x2: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns (x1 >= x2) element-wise.
    """
    return _compare(x1, x2, ">=")

# }}}


# {{{ logical operations

def logical_or(x1: Union[Array, Number],
               x2: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns the element-wise logical OR of *x1* and *x2*.
    """
    # https://github.com/python/mypy/issues/3186
    import pytato.utils as utils
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.LogicalOr((x, y)),
                                     lambda x, y: np.bool8)  # type: ignore


def logical_and(x1: Union[Array, Number],
               x2: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns the element-wise logical AND of *x1* and *x2*.
    """
    # https://github.com/python/mypy/issues/3186
    import pytato.utils as utils
    return utils.broadcast_binary_op(x1, x2,
                                     lambda x, y: prim.LogicalAnd((x, y)),
                                     lambda x, y: np.bool8)  # type: ignore


def logical_not(x: Union[Array, Number]) -> Union[Array, bool]:
    """
    Returns the element-wise logical NOT of *x*.
    """
    if isinstance(x, Number):
        # https://github.com/python/mypy/issues/3186
        return np.logical_not(x)  # type: ignore

    from pytato.utils import with_indices_for_broadcasted_shape
    return IndexLambda(x.namespace,
                       with_indices_for_broadcasted_shape(prim.Variable("_in0"),
                                                          x.shape,
                                                          x.shape),
                       shape=x.shape,
                       dtype=np.bool8,
                       bindings={"_in0": x})

# }}}


# {{{ where

def where(condition: Union[Array, Number],
          x: Optional[Union[Array, Number]] = None,
          y: Optional[Union[Array, Number]] = None) -> Union[Array, Number]:
    """
    Elementwise selector between *x* and *y* depending on *condition*.
    """
    import pytato.utils as utils

    if (isinstance(condition, Number) and isinstance(x, Number)
            and isinstance(y, Number)):
        return x if condition else y

    # {{{ raise if single-argument form of pt.where is invoked

    if x is None and y is None:
        raise ValueError("Pytato does not support data-dependent array shapes.")

    if (x is None) or (y is None):
        raise ValueError("x and y must be pytato arrays")

    # }}}

    namespace = next(a.namespace for a in [condition, x, y] if isinstance(a, Array))

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

    return IndexLambda(namespace,
            prim.If(expr1, expr2, expr3),
            shape=result_shape,
            dtype=dtype,
            bindings=bindings)

# }}}


# {{{ (max|min)inimum

def maximum(x1: Union[Array, Number],
            x2: Union[Array, Number]) -> Union[Array, Number]:
    """
    Returns the elementwise maximum of *x1*, *x2*. *x1*, *x2* being
    array-like objects that could be broadcasted together. NaNs are propagated.
    """
    # https://github.com/python/mypy/issues/3186
    return where(logical_or(isnan(x1), isnan(x2)), np.NaN,  # type: ignore
                 where(greater(x1, x2), x1, x2))  # type: ignore


def minimum(x1: Union[Array, Number],
            x2: Union[Array, Number]) -> Union[Array, Number]:
    """
    Returns the elementwise minimum of *x1*, *x2*. *x1*, *x2* being
    array-like objects that could be broadcasted together. NaNs are propagated.
    """
    # https://github.com/python/mypy/issues/3186
    return where(logical_or(isnan(x1), isnan(x2)), np.NaN,  # type: ignore
                 where(less(x1, x2), x1, x2))  # type: ignore

# }}}


# {{{ make_index_lambda

INDEX_RE = re.compile("_r?(0|([1-9][0-9]*))")


def make_index_lambda(namespace: Namespace,
        expression: Union[str, ScalarExpression],
        bindings: Dict[str, Array],
        shape: ShapeType,
        dtype: Any) -> IndexLambda:
    if isinstance(expression, str):
        raise NotImplementedError("Sorry the developers were too lazy to implement"
                " a parser.")

    # {{{ sanity checks

    from pytato.scalar_expr import get_dependencies
    unknown_dep = (get_dependencies(expression)
                   - set(namespace) - set(bindings))
    for dep in unknown_dep:
        if not INDEX_RE.fullmatch(dep):
            raise ValueError(f"Unknown variable '{dep}' in the expression.")

    # }}}

    return IndexLambda(namespace, expr=expression,
            bindings=bindings,
            shape=shape,
            dtype=dtype)

# }}}


# {{{ reductions

def _preprocess_reduction_axes(
        shape: ShapeType,
        reduction_axes: Optional[Union[int, Tuple[int]]]
        ) -> Tuple[ShapeType, Tuple[int, ...]]:
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

    new_shape = []

    for i, axis_len in enumerate(shape):
        if i not in reduction_axes:
            new_shape.append(axis_len)

    return tuple(new_shape), reduction_axes


def _get_reduction_indices_bounds(shape: ShapeType,
        axes: Tuple[int, ...]) -> Tuple[
                List[ScalarExpression],
                Mapping[str, Tuple[ScalarExpression, ScalarExpression]]]:

    indices: List[prim.Variable] = []
    redn_bounds: Mapping[str, Tuple[ScalarExpression, ScalarExpression]] = {}

    n_out_dims = 0
    n_redn_dims = 0
    for idim, axis_len in enumerate(shape):
        if idim in axes:
            idx = f"_r{n_redn_dims}"
            indices.append(prim.Variable(idx))
            redn_bounds[idx] = (0, axis_len)  # type: ignore
            n_redn_dims += 1
        else:
            indices.append(prim.Variable(f"_{n_out_dims}"))
            n_out_dims += 1

    return indices, redn_bounds


def sum(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Sums array *a*'s elements along the *axis* axes.

    :arg axis: The axes along which the elements are to be sum-reduced.
        Defaults to all axes of the input arrays.
    """
    new_shape, axes = _preprocess_reduction_axes(a.shape, axis)
    del axis
    indices, redn_bounds = _get_reduction_indices_bounds(a.shape, axes)

    return make_index_lambda(a.namespace,
            scalar_expr.Reduce(
                prim.Subscript(prim.Variable("in"), tuple(indices)),
                scalar_expr.ReductionOp.SUM,
                redn_bounds),
            {"in": a},
            new_shape,
            a.dtype)


def amax(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Sums array *a*'s elements along the *axis* axes.

    :arg axis: The axes along which the elements are to be sum-reduced.
        Defaults to all axes of the input arrays.
    """
    new_shape, axes = _preprocess_reduction_axes(a.shape, axis)
    del axis
    indices, redn_bounds = _get_reduction_indices_bounds(a.shape, axes)

    return make_index_lambda(a.namespace,
            scalar_expr.Reduce(
                prim.Subscript(prim.Variable("in"), tuple(indices)),
                scalar_expr.ReductionOp.MAX,
                redn_bounds),
            {"in": a},
            new_shape,
            a.dtype)


def amin(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Sums array *a*'s elements along the *axis* axes.

    :arg axis: The axes along which the elements are to be sum-reduced.
        Defaults to all axes of the input arrays.
    """
    new_shape, axes = _preprocess_reduction_axes(a.shape, axis)
    del axis
    indices, redn_bounds = _get_reduction_indices_bounds(a.shape, axes)

    return make_index_lambda(a.namespace,
            scalar_expr.Reduce(
                prim.Subscript(prim.Variable("in"), tuple(indices)),
                scalar_expr.ReductionOp.MIN,
                redn_bounds),
            {"in": a},
            new_shape,
            a.dtype)

# }}}


class DistributedSend(Array):

    def __init__(self, data, to_rank: int = 0, tag: str = ""):
        pass


class DistributedRecv(Array):

    def __init__(self, src_rank: int = 0, tag: str = "", shape=(), dtype=float, tags=None):
        pass


# vim: foldmethod=marker
