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
"""

# }}}


import collections.abc
from dataclasses import dataclass
from pytools import single_valued, is_single_valued
import pymbolic.primitives as prim


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
    def __init__(self, name_parts):
        self.name_parts = name_parts

    @classmethod
    def from_class(cls, argcls):
        name_parts = tuple(argcls.__module__.split(".") + [argcls.__name__])
        if not all(not npart.startswith("__") for npart in name_parts):
            raise ValueError(f"some name parts of {'.'.join(name_parts)} "
                    "start with double underscores")
        return cls(name_parts)

# }}}


# {{{ namespace

class Namespace:
    # Possible future extension: .parent attribute
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

    def __init__(self):
        self._symbol_table = {}

    def __contains__(self, name):
        return name in self._symbol_table

    def __getitem__(self, name):
        return self._symbol_table[name]

    def __iter__(self):
        return iter(self._symbol_table)

    def assign(self, name, value):
        """
        :returns: *name*
        """
        if name in self._symbol_table:
            raise ValueError(f"'{name}' is already assigned")
        self._symbol_table[name] = value

        return name

    def ref(self, name):
        """
        :returns: An :term:`array expression` referring to *name*.
        """

        value = self._symbol_table[name]

        var_ref = prim.Variable(name)
        if value.shape:
            var_ref = var_ref[tuple("_%d" % i for i in range(len(value.shape)))]

        return IndexLambda(
                self, name, expr=var_ref, shape=value.shape,
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
    def tag_name(self):
        return DottedName.from_class(type(self))


class UniqueTag(Tag):
    """
    Only one instance of this type of tag may be assigned
    to a single tagged object.
    """

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

    def __init__(self, namespace, tags=None):
        if tags is None:
            tags = {}

        self.namespace = namespace
        self.tags = tags

    def copy(self, **kwargs):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    def named(self, name):
        return self.namespace.ref(self.namespace.assign(name, self))

    @property
    def ndim(self):
        return len(self.shape)

    def tagged(self, tag: Tag):
        """
        Returns a copy of *self* tagged with *tag*.
        If *tag* is a :class:`UniqueTag` and other
        tags of this type are already present, an error
        is raised.
        """
        pass

    def without_tag(self, dotted_name):
        pass

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

    def __init__(self, namespace, name, expr, shape, dtype, bindings=None,
            tags=None):
        if bindings is None:
            bindings = {}

        super().__init__(namespace, name, tags=tags)

        self._shape = shape
        self._dtype = dtype
        self.expr = expr
        self.bindings = bindings

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def is_reference(self):
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

    def __init__(self, namespace, data, tags=None):
        super().__init__(namespace, tags)

        self.data = data

    @property
    def shape(self):
        self.data.shape

    @property
    def dtype(self):
        self.data.dtype

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

    def __init__(self, namespace, name, shape, tags=None):
        if name is None:
            raise ValueError("Placeholder instances must have a name")

        # Reserve the name, prevent others from using it.
        namespace.assign(name, None)

        super().__init__(namespace=namespace, tags=tags)

        self.name = name
        self._shape = shape

    @property
    def shape(self):
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

# vim: foldmethod=marker
