Design Decisions in Pytato
==========================

.. currentmodule:: pytato

TODO
----

- reduction inames
- finish trawling the design doc
- expression nodes in index lambda
    - what pymbolic expression nodes are OK
    - reductions
    - function identifier scoping
    - piecewise def (use ISL?)

Computation and Results
-----------------------

-   Results of computations are either implement the :class:`~Array`
    interface or are a :class:`~DictOfNamedArrays`.
    The former are referred to as an :term:`array expression`. The union type
    of both of them is referred to as an *array result*. (FIXME? name)

-   Array data is computed lazily, i.e., a representation of the desired
    computation is built, but computation/code generation is not carried
    out until instructed by the user. Evaluation/computation
    is never triggered implicitly.

-   :class:`IndexLambda` is the main means by which element-wise
    expressions, expressions involving reductions, and
    prefix sums/scans are expressed. No expression nodes should
    be created for array expressions that are expressible without
    loss of information as a :class:`IndexLambda`. :class:`IndexLambda`
    allows anything for which :mod:`numpy` might use a
    :class:`numpy.ufunc`, but for example :func:`numpy.reshape`
    is not expressible without loss of information and therefore
    realized as its own node, :class:`Reshape`.

Naming
------

-   There is one (for now) computation :class:`~Namespace` per
    computation that defines the computational "environment".
    Operations involving array expressions not using the same
    namespace are prohibited.

-   Names in the :class:`~Namespace` are under user control
    and unique. I.e. new names in the :class:`~Namespace` outside
    that are not a :ref:`reserved_identifier`
    not generated automatically without explicit user input.

-   The (array) value associated with a name is immutable once evaluated.
    In-place slice assignment may be simulated by returning a new
    node realizing a "partial replacement".

-   :attr:`Array.shape` and :attr:`Array.dtype` are evaluated eagerly.

-   Results of array computations that are scalar (i.e. an :attr:`Array.shape` of `()`)
    and have an integral :attr:`Array.dtype` (i.e. ``dtype.kind == "i"``) may be used in
    shapes once they have been assigned a name.

    For some computations such as fancy indexing::

        A[A > 0]

    it may be necessary to automatically generate names, in this
    case to describe the shape of the index array used to realize
    the access``A[A>0]``. These will be drawn from the reserved namespace
    ``_pt_shp``. Users may control the naming of these counts
    by assigning the tag :attr:`pytato.array.CountNamed`, like so::

        B = A[(A > 0).tagged(CountNamed("mycount"))]

-   :class:`Placeholder` expressions, like all array expressions,
    are considered read-only. When computation begins, the same
    actual memory may be supplied for multiple :term:`placeholder name`s,
    i.e. those arrays may alias.

    .. note::

        This does not preclude the arrays being declared with
        C's ``*restrict`` qualifier in generated code, as they
        do not alias any data that is being modified.

.. _reserved_identifier:

Reserved Identifiers
--------------------

-   Identifiers beginning with ``_pt_`` are reserved for internal use
    by :module:`pytato`. Any such internal use must be drawn from one
    of the following sub-regions, identified by their identifier
    prefixes:

    -   ``_pt_shp``: Used to automatically generate identifiers used
        in data-dependent shapes.

-   Identifiers used in index lambdas are also reserved. These include:

    -   Identifiers matching the regular expression ``_[0-9]+``. They are used
        as index ("iname") placeholders.

    -   Identifiers matching the regular expression ``_r[0-9]+``. They are used
        as reduction indices.

    -   Identifiers matching the regular expression ``_in[0-9]+``. They are used
        as automatically generated names (if required) in
        :attr:`IndexLambda.bindings`.

Glossary
========

.. glossary::

    array expression
        An object implementing the :clas:`~Array` interface

    array result
        An :term:`array expression` or an instance of
        :class:`~DictOfNamedArrays`.

    identifier
        Any string for which :meth:`str.isidentifier` returns
        *True*. See also :ref:`reserved_identifier`.

    namespace name
        The name by which an :term:`array expression` is known
        in a :class:`Namespace`.

    placeholder name
        See :attr:`Placeholder.name`.

.. vim: shiftwidth=4
