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

Built-in Expression Nodes
-------------------------

.. autoclass:: IndexLambda
.. autoclass:: Einsum
.. autoclass:: MatrixProduct
.. autoclass:: LoopyFunction
.. autoclass:: Stack
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
        Protocol, Sequence, cast, TYPE_CHECKING)

import numpy as np
import pymbolic.primitives as prim
from pymbolic import var
from pytools import is_single_valued, memoize_method, UniqueNameGenerator
from pytools.tag import Tag, UniqueTag, TagsType, tag_dataclass

import pytato.scalar_expr as scalar_expr
from pytato.scalar_expr import ScalarExpression


# Get a type variable that represents the type of '...'
# https://github.com/python/typing/issues/684#issuecomment-548203158
if TYPE_CHECKING:
    from enum import Enum

    class EllipsisType(Enum):
        Ellipsis = "..."

    Ellipsis = EllipsisType.Ellipsis
else:
    EllipsisType = type(Ellipsis)


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
        return copy_namespace(self, CopyMapper(Namespace()))

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

# }}}


# {{{ shape

AxisLenType = Union[int, prim.Expression]
ShapeType = Tuple[AxisLenType, ...]
ConvertibleToShapeComponent = Union[int, prim.Expression, str]
ConvertibleToShape = Union[
        str,
        AxisLenType,
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
    if isinstance(shape, (int, Number, prim.Expression)):
        shape = (shape,)

    return tuple(normalize_shape_component(s) for s in shape)

# }}}


# {{{ array inteface

SliceItem = Union[int, slice, None, EllipsisType]
DtypeOrScalar = Union[np.dtype, Number]


def _truediv_result_type(arg1: DtypeOrScalar, arg2: DtypeOrScalar) -> np.dtype:
    dtype = np.result_type(arg1, arg2)
    # See: test_true_divide in numpy/core/tests/test_ufunc.py
    if dtype.kind in "iu":
        return np.float64
    else:
        return dtype


class Array:
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

    def __init__(self, tags: Optional[TagsType] = None):
        if tags is None:
            tags = frozenset()

        self.tags = tags

    def copy(self, **kwargs: Any) -> Array:
        raise NotImplementedError

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
    def dtype(self) -> np.dtype:
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

    def tagged(self, tag: Tag) -> Array:
        """
        Returns a copy of *self* tagged with *tag*.
        If *tag* is a :class:`pytools.tag.UniqueTag` and other
        tags of this type are already present, an error
        is raised.
        """
        return self.copy(tags=self.tags | frozenset([tag]))

    def without_tag(self, tag: Tag, verify_existence: bool = True) -> Array:
        new_tags = tuple(t for t in self.tags if t != tag)

        if verify_existence and len(new_tags) == len(self.tags):
            raise ValueError(f"tag '{tag}' was not present")

        return self.copy(tags=new_tags)

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
            get_result_type: Callable[[DtypeOrScalar, DtypeOrScalar], np.dtype] = np.result_type,  # noqa
            reverse: bool = False) -> Array:

        def add_indices(val: prim.Expression) -> prim.Expression:
            if self.ndim == 0:
                return val
            else:
                indices = tuple(var(f"_{i}") for i in range(self.ndim))
                return val[indices]

        if isinstance(other, Array):
            if self.namespace is not other.namespace:
                raise ValueError("Operands must belong to the same namespace.")

        if isinstance(other, Number):
            first_expr = add_indices(var("_in0"))
            second_expr = other
            bindings = dict(_in0=self)
            dtype = get_result_type(self.dtype, other)

        elif isinstance(other, Array):
            if self.shape != other.shape:
                raise NotImplementedError("broadcasting not supported")
            first_expr = add_indices(var("_in0"))
            second_expr = add_indices(var("_in1"))
            bindings = dict(_in0=self, _in1=other)
            dtype = get_result_type(self.dtype, other.dtype)

        else:
            raise ValueError("unknown argument")

        if reverse:
            first_expr, second_expr = second_expr, first_expr

        expr = op(first_expr, second_expr)

        return IndexLambda(self.namespace,
                expr,
                shape=self.shape,
                dtype=dtype,
                bindings=bindings)

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
            dtype: np.dtype,
            **kwargs: Any):
        # https://github.com/python/mypy/issues/5887
        super().__init__(**kwargs)  # type: ignore
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
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

class DictOfNamedArrays(Mapping[str, Array]):
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

    def __init__(self, data: Dict[str, Array]):
        self._data = data

    @property
    def namespace(self) -> Namespace:
        return next(iter(self._data.values())).namespace

    def __contains__(self, name: object) -> bool:
        return name in self._data

    def __getitem__(self, name: str) -> Array:
        return self._data[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

# }}}


# {{{ index lambda

class IndexLambda(_SuppliedShapeAndDtypeMixin, Array):
    """
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
            dtype: np.dtype,
            bindings: Optional[Dict[str, Array]] = None,
            tags: Optional[TagsType] = None):

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
            tags: Optional[TagsType] = None):
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
    def dtype(self) -> np.dtype:
        return np.result_type(self.x1.dtype, self.x2.dtype)

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
            tags: Optional[TagsType] = None):
        super().__init__(tags)
        self.arrays = arrays
        self.axis = axis

    @property
    def namespace(self) -> Namespace:
        return self.arrays[0].namespace

    @property
    def dtype(self) -> np.dtype:
        return np.result_type(*(arr.dtype for arr in self.arrays))

    @property
    def shape(self) -> ShapeType:
        result = list(self.arrays[0].shape)
        result.insert(self.axis, len(self.arrays))
        return tuple(result)

# }}}


# {{{ attribute lookup

class AttributeLookup(Array):
    """An expression node to extract an array from a :class:`DictOfNamedArrays`.

    .. warning::

        Not yet implemented.
    """
    pass


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
            tags: Optional[TagsType] = None):
        super().__init__(tags)
        self.array = array

    @property
    def dtype(self) -> np.dtype:
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
            tags: Optional[TagsType] = None):
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
            tags: Optional[TagsType] = None):
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
    """

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
            tags: Optional[TagsType] = None):
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
            tags: Optional[TagsType] = None):
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

    def tagged(self, tag: Tag) -> Array:
        raise ValueError("Cannot modify tags")

    def without_tag(self, tag: Tag, verify_existence: bool = True) -> Array:
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
    dtype: np.dtype


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
            tags: Optional[TagsType] = None):
        super().__init__(namespace, name, tags=tags)

        self.data = data
        self._shape = shape

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
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
            dtype: np.dtype,
            tags: Optional[TagsType] = None):
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
    def dtype(self) -> np.dtype:
        return np.dtype(np.intp)

# }}}


# {{{ loopy function

class LoopyFunction(DictOfNamedArrays):
    """
    .. note::

        This should allow both a locally stored kernel
        and one that's obtained by importing a dotted
        name.
    """

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
        tags: Optional[TagsType] = None) -> Placeholder:
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
        tags: Optional[TagsType] = None) -> SizeParam:
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
        tags: Optional[TagsType] = None) -> DataWrapper:
    """Make a :class:`DataWrapper`.

    :param namespace:  namespace
    :param data:       an instance obeying the :class:`DataInterface`
    :param name:       an optional name, generated automatically if not given
    :param shape:      optional shape of the array, inferred from *data* if not given
    :param tags:       implementation tags
    """
    if name is None:
        name = namespace.name_gen("_pt_data")

    if not name.isidentifier():
        raise ValueError(f"'{name}' is not a Python identifier")

    if shape is None:
        shape = data.shape

    shape = normalize_shape(shape, namespace)

    return DataWrapper(namespace, name, data, shape, tags)

# }}}


# vim: foldmethod=marker
