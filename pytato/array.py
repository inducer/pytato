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

__doc__ = """


Expression trees based on this package are picklable
as long as no non-picklable data
(e.g. :class:`pyopencl.array.Array`)
is referenced from :class:`DataWrapper`.

Array Interface
---------------

.. currentmodule:: pytato

.. autoclass :: Namespace
.. autoclass :: Array
.. autoclass :: DictOfNamedArrays

Built-in Expression Nodes
-------------------------
.. currentmodule:: pytato.array

.. autoclass:: IndexLambda
.. autoclass:: Einsum
.. autoclass:: DataWrapper
.. autoclass:: Placeholder
.. autoclass:: LoopyFunction

User Interface for Making Nodes
-------------------------------
.. currentmodule:: pytato.array

.. autofunction:: make_dict_of_named_arrays
.. autofunction:: make_placeholder
"""


import pytato.typing as ptype
from pytools import is_single_valued
from typing import Optional, Dict, Any, Mapping, Iterator


class Namespace(ptype.NamespaceInterface):
    # Possible future extension: .parent attribute
    """
    .. attribute:: symbol_table

        A mapping from strings that must be valid
        Python identifiers to objects implementing the
        :class:`Array` interface.
    """

    def __init__(self) -> None:
        self._symbol_table: Dict[str, ptype.ArrayInterface] = {}

    def symbol_table(self) -> Dict[str, ptype.ArrayInterface]:
        return self._symbol_table

    def assign(self, name: str,
               value: ptype.ArrayInterface) -> None:
        """Declare a new array.

        :param name: a Python identifier
        :param value: the array object
        """
        if name in self._symbol_table:
            raise ValueError(f"'{name}' is already assigned")
        self._symbol_table[name] = value


class Array(ptype.ArrayInterface):
    """
    A base class (abstract interface +
    supplemental functionality) for lazily
    evaluating array expressions.

    .. note::

        The interface seeks to maximize :mod:`numpy`
        compatibility, though not at all costs.

    All these are abstract:

    .. attribute:: name

        A name in :attr:`namespace` that has been assigned
        to this expression. May be (and typically is) *None*.

    .. attribute:: namespace

       A (mutable) instance of :class:`Namespace` containing the
       names used in the computation. All arrays in a
       computation share the same namespace.

    .. attribute:: shape

        Identifiers (:class:`pymbolic.Variable`) refer to
        names from :attr:`namespace`.
        A tuple of integers or :mod:`pymbolic` expressions.
        Shape may be (at most affinely) symbolic in these
        identifiers.

        # FIXME: -> https://gitlab.tiker.net/inducer/pytato/-/issues/1

    .. attribute:: dtype

        An instance of :class:`numpy.dtype`.

    .. attribute:: tags

        A :class:`dict` mapping :class:`DottedName` instances
        to an argument object, whose structure is defined
        by the tag.

        Motivation: `RDF
        <https://en.wikipedia.org/wiki/Resource_Description_Framework>`__
        triples (subject: implicitly the array being tagged,
        predicate: the tag, object: the arg).

        For example::

            # tag
            DottedName("our_array_thing.impl_mode"):

            # argument
            DottedName(
                "our_array_thing.loopy_target.subst_rule")

       .. note::

           This mirrors the tagging scheme that :mod:`loopy`
           is headed towards.

    Derived attributes:

    .. attribute:: ndim

    Objects of this type are hashable and support
    structural equality comparison (and are therefore
    immutable).

    .. note::

        This *does* break :mod:`numpy` compatibility,
        purposefully so.
    """

    def __init__(self, namespace: ptype.NamespaceInterface,
                 name: Optional[str],
                 tags: Optional[ptype.TagsType] = None):
        if tags is None:
            tags = {}

        if name is not None:
            namespace.assign(name, self)

        self._namespace = namespace
        self.name = name
        self.tags = tags

    def copy(self, **kwargs: Any) -> Array:
        raise NotImplementedError

    @property
    def namespace(self) -> ptype.NamespaceInterface:
        return self._namespace

    @property
    def shape(self) -> ptype.ShapeType:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def with_tag(self, dotted_name: ptype.DottedName,
                 args: Optional[ptype.DottedName] = None) -> Array:
        """
        Returns a copy of *self* tagged with *dotted_name*
        and arguments *args*
        If a tag *dotted_name* is already present, it is
        replaced in the returned object.
        """
        if args is None:
            pass
        return self.copy()

    def without_tag(self, dotted_name: ptype.DottedName) -> Array:
        raise NotImplementedError

    def with_name(self, name: str) -> Array:
        self.namespace.assign(name, self)
        return self.copy(name=name)

    # TODO:
    # - tags
    # - codegen interface
    # - naming


class DictOfNamedArrays(Mapping[str, ptype.ArrayInterface]):
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

    def __init__(self, data: Dict[str, ptype.ArrayInterface]):
        self._data = data

    @property
    def namespace(self) -> ptype.NamespaceInterface:
        return next(iter(self._data.values())).namespace

    def __contains__(self, name: object) -> bool:
        return name in self._data

    def __getitem__(self, name: str) -> ptype.ArrayInterface:
        return self._data[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class IndexLambda(Array):
    """
    .. attribute:: index_expr

        A scalar-valued :mod:`pymbolic` expression such as
        ``a[_1] + b[_2, _1]`` depending on TODO

        Identifiers in the expression are resolved, in
        order, by lookups in :attr:`inputs`, then in
        :attr:`namespace`.

        Scalar functions in this expression must
        be identified by a dotted name
        (e.g. ``our_array_thing.c99.sin``).

    .. attribute:: binding

        A :class:`dict` mapping strings that are valid
        Python identifiers to objects implementing
        the :class:`Array` interface, making array
        expressions available for use in
        :attr:`index_expr`.
    """


class Einsum(Array):
    """
    """


class DataWrapper(Array):
    # TODO: Name?
    """
    Takes concrete array data and packages it to be compatible
    with the :class:`Array` interface.

    A way
    .. attrib

        A concrete array (containing data), given as, for example,
        a :class:`numpy.ndarray`, or a :class:`pyopencl.array.Array`.

    """


class Placeholder(Array):
    """
    A named placeholder for an array whose concrete value
    is supplied by the user during evaluation.

    .. note::

        A symbolically represented
    A symbolic
    On
        is required, and :attr:`shape` is given as data.
    """

    @property
    def shape(self) -> ptype.ShapeType:
        # Matt added this to make Pylint happy.
        # Not tied to this, open for discussion about how to implement this.
        return self._shape

    def __init__(self, namespace: ptype.NamespaceInterface,
                 name: str, shape: ptype.ShapeType,
                 tags: Optional[ptype.TagsType] = None):
        super().__init__(
            namespace=namespace,
            name=name,
            tags=tags)

        self._shape = shape


class LoopyFunction(DictOfNamedArrays):
    """
    .. note::

        This should allow both a locally stored kernel
        and one that's obtained by importing a dotted
        name.
    """


# {{{ end-user-facing


def make_dict_of_named_arrays(
        data: Dict[str, ptype.ArrayInterface]) -> DictOfNamedArrays:
    """Make a :class:`DictOfNamedArrays` object and ensure that all arrays
    share the same namespace.

    :param data: member keys and arrays
    """
    if not is_single_valued(ary.namespace for ary in data.values()):
        raise ValueError("arrays do not have same namespace")
    return DictOfNamedArrays(data)


def make_placeholder(namespace: ptype.NamespaceInterface,
                     name: str,
                     shape: ptype.ShapeType,
                     tags: Optional[ptype.TagsType] = None
                     ) -> Placeholder:
    """Make a :class:`Placeholder` object.

    :param namespace:  namespace of the placeholder array
    :param name:       name of the placeholder array
    :param shape:      shape of the placeholder array
    :param tags:       implementation tags
    """
    assert str.isidentifier(name)
    assert ptype.check_shape(shape, namespace)
    return Placeholder(namespace, name, shape, tags)

# }}} End end-user-facing
