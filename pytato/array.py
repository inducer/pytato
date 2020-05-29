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

.. autoclass :: Namespace
.. autoclass :: Array
.. autoclass :: Tag
.. autoclass :: UniqueTag
.. autoclass :: DictOfNamedArrays

Supporting Functionality
------------------------

.. autoclass :: DottedName

.. currentmodule:: pytato.array

Pre-Defined Tags
----------------

.. autoclass:: ImplementAs
.. autoclass:: CountNamed

Built-in Expression Nodes
-------------------------

.. autoclass:: IndexLambda
.. autoclass:: Einsum
.. autoclass:: Reshape
.. autoclass:: DataWrapper
.. autoclass:: Placeholder
.. autoclass:: LoopyFunction

User-Facing Node Creation
-------------------------

Node constructors such as :class:`Placeholder.__init__` and
:class:`DictOfNamedArrays.__init__` offer limited input validation
(in favor of faster execution). Node creation from outside
:mod:`pytato` should use the following interfaces:

.. autofunction:: make_dict_of_named_arrays
.. autofunction:: make_placeholder
"""

# }}}

import numpy as np
import pymbolic.primitives as prim

from dataclasses import dataclass
from pytools import is_single_valued
from typing import Optional, Dict, Any, Mapping, Iterator, Tuple, Union

# {{{ namespace


class Namespace():
    r"""
    Represents a mapping from :term:`identifier` strings to
    :term:`array expression`\ s or *None*, where *None* indicates that the name
    may not be used.  (:class:`Placeholder` instances register their names in
    this way to avoid ambiguity.)

    .. automethod:: __contains__
    .. automethod:: __getitem__
    .. automethod:: __iter__
    .. automethod:: assign
    .. automethod:: ref
    """

    def __init__(self) -> None:
        self._symbol_table: Dict[str, Optional[Array]] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._symbol_table

    def __getitem__(self, name: str) -> Array:
        item = self._symbol_table[name]
        if item is None:
            raise ValueError("cannot access a reserved name")
        return item

    def __iter__(self) -> Iterator[str]:
        return iter(self._symbol_table)

    def assign(self, name: str,
               value: Optional[Array]) -> str:
        """Declare a new array.

        :param name: a Python identifier
        :param value: the array object, or None if the assignment is to
                      just reserve a name

        :returns: *name*
        """
        if name in self._symbol_table:
            raise ValueError(f"'{name}' is already assigned")
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

        # FIXME: mypy thinks that this is Any
        return IndexLambda(
                self, expr=var_ref, shape=value.shape,
                dtype=value.dtype)

# }}}

# {{{ tag


class DottedName():
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


TagsType = Dict[DottedName, Tag]

# }}}

# {{{ shape


ShapeComponentType = Union[int, prim.Expression, str]
ShapeType = Tuple[ShapeComponentType, ...]


def check_shape(shape: ShapeType,
                ns: Optional[Namespace] = None) -> bool:
    """Checks for a shape tuple.

    :param shape: the shape tuple

    :param ns: if a namespace is given, extra checks are performed to
               ensure that expressions are well-defined.
    """
    for s in shape:
        if isinstance(s, int):
            assert s > 0, f"size parameter must be positive (got {s})"
        elif isinstance(s, str):
            assert str.isidentifier(s)
        elif isinstance(s, prim.Expression) and ns is not None:
            # TODO: check expression in namespace
            pass
    return True

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

       A (mutable) instance of :class:`Namespace` containing the
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

        A :class:`dict` mapping :class:`DottedName` instances
        to instances of the :class:`Tag` interface.

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

    def __init__(self, namespace: Namespace,
                 tags: Optional[TagsType] = None):
        if tags is None:
            tags = {}

        self._namespace = namespace
        self.tags = tags
        self.dtype: np.dtype = np.float64  # FIXME

    def copy(self, **kwargs: Any) -> Array:
        raise NotImplementedError

    @property
    def namespace(self) -> Namespace:
        return self._namespace

    @property
    def shape(self) -> ShapeType:
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
        raise NotImplementedError
        return self.copy()

    def without_tag(self, dotted_name: DottedName) -> Array:
        raise NotImplementedError

    # TODO:
    # - codegen interface

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


class IndexLambda(Array):
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

    # TODO: write make_index_lambda() that does dtype inference

    def __init__(
            self, namespace: Namespace, expr: prim.Expression,
            shape: ShapeType, dtype: np.dtype,
            bindings: Optional[Dict[str, Array]] = None,
            tags: Optional[Dict[DottedName, Tag]] = None):

        if bindings is None:
            bindings = {}

        super().__init__(namespace, tags=tags)

        self._shape = shape
        self._dtype = dtype
        self.expr = expr
        self.bindings = bindings

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

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

# {{{ reshape


class Reshape(Array):
    """
    """

# }}}

# {{{ data wrapper


class DataWrapper(Array):
    # TODO: Name?
    """
    Takes concrete array data and packages it to be compatible
    with the :class:`Array` interface.

    .. attribute:: data

        A concrete array (containing data), given as, for example,
        a :class:`numpy.ndarray`, or a :class:`pyopencl.array.Array`.
        This must offer ``shape`` and ``dtype`` attributes but is
        otherwise considered opaque. At evaluation time, its
        type must be understood by the appropriate execution backend.

        Starting with the construction of the :class:`DataWrapper`,
        this array may not be updated in-place.
    """

    # TODO: not really Any data
    def __init__(self, namespace: Namespace, data: Any,
                 tags: Optional[TagsType] = None):
        super().__init__(namespace, tags)

        self.data = data

    @property
    def shape(self) -> Any:  # FIXME
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

# }}}

# {{{ placeholder


class Placeholder(Array):
    """
    A named placeholder for an array whose concrete value
    is supplied by the user during evaluation.

    .. attribute:: name

        The name by which a value is supplied
        for the placeholder once computation begins.

    .. note::

        :attr:`name` is not a :term:`namespace name`. In fact,
        it is prohibited from being one. (This has to be the case: Suppose a
        :class:`Placeholder` is :meth:`~Array.tagged`, would the namespace name
        refer to the tagged or the untagged version?)
    """

    def __init__(self, namespace: Namespace,
                 name: str, shape: ShapeType,
                 tags: Optional[TagsType] = None):

        # Reserve the name, prevent others from using it.
        namespace.assign(name, None)

        super().__init__(namespace=namespace, tags=tags)

        self.name = name
        self._shape = shape

    @property
    def shape(self) -> ShapeType:
        # Matt added this to make Pylint happy.
        # Not tied to this, open for discussion about how to implement this.
        return self._shape

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

# {{{ end-user-facing


def make_dotted_name(name_parts: Tuple[str, ...]) -> DottedName:
    assert len(name_parts) > 0
    assert all(str.isidentifier(p) for p in name_parts)
    return DottedName(name_parts)


def make_dict_of_named_arrays(
        data: Dict[str, Array]) -> DictOfNamedArrays:
    """Make a :class:`DictOfNamedArrays` object and ensure that all arrays
    share the same namespace.

    :param data: member keys and arrays
    """
    if not is_single_valued(ary.namespace for ary in data.values()):
        raise ValueError("arrays do not have same namespace")
    return DictOfNamedArrays(data)


def make_placeholder(namespace: Namespace,
                     name: str,
                     shape: ShapeType,
                     tags: Optional[TagsType] = None
                     ) -> Placeholder:
    """Make a :class:`Placeholder` object.

    :param namespace:  namespace of the placeholder array
    :param name:       name of the placeholder array
    :param shape:      shape of the placeholder array
    :param tags:       implementation tags
    """
    assert str.isidentifier(name)
    assert check_shape(shape, namespace)
    return Placeholder(namespace, name, shape, tags)

# }}} End end-user-facing

# vim: foldmethod=marker
