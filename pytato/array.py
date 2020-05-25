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

Supporting Functionality
------------------------

.. autoclass :: DottedName

Built-in Expression Nodes
-------------------------
.. currentmodule:: pytato.array

.. autoclass:: IndexLambda
.. autoclass:: Einsum
.. autoclass:: DataWrapper
.. autoclass:: Placeholder
.. autoclass:: LoopyFunction
"""


import collections.abc
from pytools import single_valued, is_single_valued


class DottedName:
    """
    .. attribute:: name_parts

        A tuple of strings, each of which is a valid
        Python identifier.

    The name (at least morally) exists in the
    name space defined by the Python module system.
    It need not necessarily identify an importable
    object.
    """
    def __init__(self, name_parts):
        self.name_parts = name_parts


class Namespace:
    # Possible future extension: .parent attribute
    """
    .. attribute:: symbol_table

        A mapping from strings that must be valid
        C identifiers to objects implementing the
        :class:`Array` interface.
    """

    def __init__(self):
        self.symbol_table = {}

    def assign(self, name, value):
        if name in self.symbol_table:
            raise ValueError(f"'{name}' is already assigned")
        self.symbol_table[name] = value


class Array:
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

    def __init__(self, namespace, name, tags=None):
        if tags is None:
            tags = {}

        if name is not None:
            namespace.assign(name, self)

        self.namespace = namespace
        self.name = name
        self.tags = tags

    def copy(self, **kwargs):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return len(self.shape)

    def with_tag(self, dotted_name, args=None):
        """
        Returns a copy of *self* tagged with *dotted_name*
        and arguments *args*
        If a tag *dotted_name* is already present, it is
        replaced in the returned object.
        """
        if args is None:
            pass

    def without_tag(self, dotted_name):
        pass

    def with_name(self, name):
        self.namespace.assign_name(name, self)
        return self.copy(name=name)

    # TODO:
    # - tags
    # - codegen interface
    # - naming


class DictOfNamedArrays(collections.abc.Mapping):
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

    def __init__(self, data):
        self._data = data
        # TODO: Check that keys are valid Python identifiers

        if not is_single_valued(ary.target for ary in data.values()):
            raise ValueError("arrays do not have same target")
        if not is_single_valued(ary.namespace for ary in data.values()):
            raise ValueError("arrays do not have same namespace")

    @property
    def namespace(self):
        return single_valued(ary.namespace for ary in self._data.values())

    def __contains__(self, name):
        return name in self._data

    def __getitem__(self, name):
        return self._data[name]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
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
    def shape(self):
        # Matt added this to make Pylint happy.
        # Not tied to this, open for discussion about how to implement this.
        return self._shape

    def __init__(self, namespace, name, shape, tags=None):
        if name is None:
            raise ValueError("Placeholder instances must have a name")
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
