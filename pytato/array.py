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
.. autoclass:: Tag
.. autoclass:: UniqueTag
.. autoclass:: DictOfNamedArrays

NumPy-Like Interface
--------------------

.. autofunction:: matmul

Supporting Functionality
------------------------

.. autoclass:: DottedName

.. currentmodule:: pytato.array

Concrete Array Data
-------------------

.. autoclass:: DataInterface

Pre-Defined Tags
----------------

.. autoclass:: ImplementAs
.. autoclass:: CountNamed
.. autoclass:: ImplStored
.. autoclass:: ImplInlined
.. autoclass:: ImplDefault

Built-in Expression Nodes
-------------------------

.. autoclass:: IndexLambda
.. autoclass:: Einsum
.. autoclass:: MatrixProduct
.. autoclass:: Reshape
.. autoclass:: LoopyFunction

Input Arguments
^^^^^^^^^^^^^^^

.. autoclass:: InputArgumentBase
.. autoclass:: DataWrapper
.. autoclass:: Placeholder
.. autoclass:: SizeParam

User-Facing Node Creation
-------------------------

Node constructors such as :class:`Placeholder.__init__` and
:class:`DictOfNamedArrays.__init__` offer limited input validation
(in favor of faster execution). Node creation from outside
:mod:`pytato` should use the following interfaces:

.. class:: ConvertibleToShape

.. autofunction:: make_dict_of_named_arrays
.. autofunction:: make_placeholder
.. autofunction:: make_size_param
.. autofunction:: make_data_wrapper
"""

# }}}

from functools import partialmethod
from numbers import Number
import operator
from dataclasses import dataclass
from typing import (
        Optional, ClassVar, Dict, Any, Mapping, Iterator, Tuple, Union,
        FrozenSet, Protocol)

import numpy as np
import pymbolic.primitives as prim
from pytools import is_single_valued, memoize_method, UniqueNameGenerator

import pytato.scalar_expr as scalar_expr
from pytato.scalar_expr import ScalarExpression


# {{{ dotted name

class DottedName:
    """
    .. attribute:: name_parts

        A tuple of strings, each of which is a valid
        Python identifier. No name part may start with
        a double underscore.

    The name (at least morally) exists in the
    name space defined by the Python module system.
    It need not necessarily identify an importable
    object.

    .. automethod:: from_class
    """

    def __init__(self, name_parts: Tuple[str, ...]):
        if len(name_parts) == 0:
            raise ValueError("empty name parts")

        for p in name_parts:
            if not p.isidentifier():
                raise ValueError(f"{p} is not a Python identifier")

        self.name_parts = name_parts

    @classmethod
    def from_class(cls, argcls: Any) -> DottedName:
        name_parts = tuple(
                [str(part) for part in argcls.__module__.split(".")]
                + [str(argcls.__name__)])
        if not all(not npart.startswith("__") for npart in name_parts):
            raise ValueError(f"some name parts of {'.'.join(name_parts)} "
                             "start with double underscores")
        return cls(name_parts)

# }}}


# {{{ namespace

class Namespace(Mapping[str, "Array"]):
    # Possible future extension: .parent attribute
    r"""
    Represents a mapping from :term:`identifier` strings to
    :term:`array expression`\ s or *None*, where *None* indicates that the name
    may not be used.  (:class:`Placeholder` instances register their names in
    this way to avoid ambiguity.)

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
        raise NotImplementedError

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


# {{{ tag

tag_dataclass = dataclass(init=True, eq=True, frozen=True, repr=True)


@tag_dataclass
class Tag:
    """
    Generic metadata, applied to, among other things,
    instances of :class:`Array`.

    .. attribute:: tag_name

        A fully qualified :class:`DottedName` that reflects
        the class name of the tag.

    Instances of this type must be immutable, hashable,
    picklable, and have a reasonably concise :meth:`__repr__`
    of the form ``dotted.name(attr1=value1, attr2=value2)``.
    Positional arguments are not allowed.

   .. note::

       This mirrors the tagging scheme that :mod:`loopy`
       is headed towards.
    """

    @property
    def tag_name(self) -> DottedName:
        return DottedName.from_class(type(self))


class UniqueTag(Tag):
    """
    Only one instance of this type of tag may be assigned
    to a single tagged object.
    """
    pass


TagsType = FrozenSet[Tag]

# }}}


# {{{ shape

ShapeType = Tuple[ScalarExpression, ...]
ConvertibleToShapeComponent = Union[int, prim.Expression, str]
ConvertibleToShape = Union[
        str,
        ScalarExpression,
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

    return tuple(normalize_shape_component(s) for s in shape)

# }}}


# {{{ array inteface

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

        Identifiers (:class:`pymbolic.Variable`) refer to names from
        :attr:`namespace`.  A tuple of integers or :mod:`pymbolic` expressions.
        Shape may be (at most affinely) symbolic in these
        identifiers.

        .. note::

            Affine-ness is mainly required by code generation for
            :class:`IndexLambda`, but :class:`IndexLambda` is used to produce
            references to named arrays. Since any array that needs to be
            referenced in this way needs to obey this restriction anyway,
            a decision was made to requir the same of *all* array expressions.

    .. attribute:: dtype

        An instance of :class:`numpy.dtype`.

    .. attribute:: tags

        A :class:`tuple` of :class:`Tag` instances.

        Motivation: `RDF
        <https://en.wikipedia.org/wiki/Resource_Description_Framework>`__
        triples (subject: implicitly the array being tagged,
        predicate: the tag, object: the arg).

    .. automethod:: named
    .. automethod:: tagged
    .. automethod:: without_tag

    Derived attributes:

    .. attribute:: ndim

    """
    mapper_method: ClassVar[str]
    # A tuple of field names. Fields must be equality comparable and
    # hashable. Dicts of hashable keys and values are also permitted.
    fields: ClassVar[Tuple[str, ...]] = ("shape", "dtype", "tags")

    def __init__(self,
            namespace: Namespace,
            tags: Optional[TagsType] = None):
        if tags is None:
            tags = frozenset()

        self.namespace = namespace
        self.tags = tags

    def copy(self, **kwargs: Any) -> Array:
        raise NotImplementedError

    @property
    def shape(self) -> ShapeType:
        raise NotImplementedError

    @property
    def dtype(self) -> np.dtype:
        raise NotImplementedError

    def named(self, name: str) -> Array:
        return self.namespace.ref(self.namespace.assign(name, self))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def tagged(self, tag: Tag) -> Array:
        """
        Returns a copy of *self* tagged with *tag*.
        If *tag* is a :class:`UniqueTag` and other
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
        for field in self.fields:
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
                    for field in self.fields))

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __matmul__(self, other: Array) -> Array:
        return matmul(self, other)

    def _binary_op(self,
            op: Any,
            other: Union[Array, Number],
            reverse: bool = False) -> Array:
        if isinstance(other, Number):
            # TODO
            raise NotImplementedError
        else:
            if self.shape != other.shape:
                raise NotImplementedError("broadcasting")

            dtype = np.result_type(self.dtype, other.dtype)

            # FIXME: If either *self* or *other* is an IndexLambda, its expression
            # could be folded into the output, producing a fused result.
            if self.shape == ():
                expr = op(prim.Variable("_in0"), prim.Variable("_in1"))
            else:
                indices = tuple(
                        prim.Variable(f"_{i}") for i in range(self.ndim))
                expr = op(
                        prim.Variable("_in0")[indices],
                        prim.Variable("_in1")[indices])

            first, second = self, other
            if reverse:
                first, second = second, first

            bindings = dict(_in0=first, _in1=second)

            return IndexLambda(self.namespace,
                    expr,
                    shape=self.shape,
                    dtype=dtype,
                    bindings=bindings)

    __mul__ = partialmethod(_binary_op, operator.mul)
    __rmul__ = partialmethod(_binary_op, operator.mul, reverse=True)

# }}}


# {{{ mixins

class _SuppliedShapeAndDtypeMixin(object):
    """A mixin class for when an array must store its own *shape* and *dtype*,
    rather than when it can derive them easily from inputs.
    """

    def __init__(self,
            namespace: Namespace,
            shape: ShapeType,
            dtype: np.dtype,
            **kwargs: Any):
        # https://github.com/python/mypy/issues/5887
        super().__init__(namespace, **kwargs)  # type: ignore
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

    .. method:: __contains__
    .. method:: __getitem__
    .. method:: __iter__
    .. method:: __len__

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

    fields = Array.fields + ("expr", "bindings")
    mapper_method = "map_index_lambda"

    def __init__(self,
            namespace: Namespace,
            expr: prim.Expression,
            shape: ShapeType,
            dtype: np.dtype,
            bindings: Optional[Dict[str, Array]] = None,
            tags: Optional[TagsType] = None):

        if bindings is None:
            bindings = {}

        super().__init__(namespace, shape=shape, dtype=dtype, tags=tags)

        self.expr = expr
        self.bindings = bindings

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

        if index != tuple("_%d" % i for i in range(len(self.shape))):
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

    The semantics of this operation follow PEP 459 [pep459]_.

    .. attribute:: x1
    .. attribute:: x2

    .. [pep459] https://www.python.org/dev/peps/pep-0459/

    """
    fields = Array.fields + ("x1", "x2")

    mapper_method = "map_matrix_product"

    def __init__(self,
            namespace: Namespace,
            x1: Array,
            x2: Array,
            tags: Optional[TagsType] = None):
        super().__init__(namespace, tags)
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

        assert False

    @property
    def dtype(self) -> np.dtype:
        return np.result_type(self.x1.dtype, self.x2.dtype)

# }}}


# {{{ reshape

class Reshape(Array):
    """
    """

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

    # The name uniquely identifies this object in the namespace.
    fields = ("name",)

    def __init__(self,
            namespace: Namespace,
            name: str,
            tags: Optional[TagsType] = None):
        if name is None:
            raise ValueError("Must have explicit name")

        # Publish our name to the namespace
        namespace.assign(name, self)

        super().__init__(namespace=namespace, tags=tags)
        self.name = name

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

    mapper_method = "map_data_wrapper"

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
    """

    mapper_method = "map_placeholder"

    def __init__(self,
            namespace: Namespace,
            name: str,
            shape: ShapeType,
            dtype: np.dtype,
            tags: Optional[TagsType] = None):
        super().__init__(
                namespace,
                shape=shape,
                dtype=dtype,
                name=name,
                tags=tags)

# }}}


# {{{ size parameter

class SizeParam(InputArgumentBase):
    r"""A named placeholder for a scalar that may be used as a variable in symbolic
    expressions for array sizes.
    """

    mapper_method = "map_size_param"

    @property
    def shape(self) -> ShapeType:
        return ()

    @property
    def dtype(self) -> np.dtype:
        return np.intp

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


# {{{ numpy interface

def matmul(x1: Array, x2: Array) -> Array:
    """Matrix multiplication."""
    if x1.namespace is not x2.namespace:
        raise ValueError("namespace mismatch")

    if (
            isinstance(x1, Number)
            or x1.shape == ()
            or isinstance(x2, Number)
            or x2.shape == ()):
        raise ValueError("scalars not allowed as arguments to matmul")

    if len(x1.shape) > 2 or len(x2.shape) > 2:
        raise NotImplementedError("broadcasting not supported")

    if x1.shape[-1] != x2.shape[0]:
        raise ValueError("dimension mismatch")

    return MatrixProduct(x1.namespace, x1, x2)

# }}}


# {{{ end-user-facing

def make_dict_of_named_arrays(data: Dict[str, Array]) -> DictOfNamedArrays:
    """Make a :class:`DictOfNamedArrays` object and ensure that all arrays
    share the same namespace.

    :param data: member keys and arrays
    """
    if not is_single_valued(ary.namespace for ary in data.values()):
        raise ValueError("arrays do not have same namespace")

    return DictOfNamedArrays(data)


def make_placeholder(namespace: Namespace,
        name: str,
        shape: ConvertibleToShape,
        dtype: np.dtype,
        tags: Optional[TagsType] = None) -> Placeholder:
    """Make a :class:`Placeholder` object.

    :param namespace:  namespace of the placeholder array
    :param name:       name of the placeholder array
    :param shape:      shape of the placeholder array
    :param dtype:      dtype of the placeholder array
    :param tags:       implementation tags
    """
    if name is None:
        raise ValueError("Placeholder instances must have a name")

    if not name.isidentifier():
        raise ValueError(f"'{name}' is not a Python identifier")

    shape = normalize_shape(shape, namespace)

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
