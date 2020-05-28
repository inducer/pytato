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
    The former are referred to as :term:`array expression`\ s. The union type
    of both is referred to as an :term:`array result`.

-   Array data is computed lazily, i.e., a representation of the desired
    computation is built, but computation/code generation is not carried
    out until instructed by the user. Evaluation/computation
    is never triggered implicitly.

-   :attr:`Array.dtype` is evaluated eagerly.

-   :attr:`Array.shape` is evaluated as eagerly as possible, however
    data-dependent name references in shapes are allowed. (This implies
    that the number of array axes must be statically known.)

    Consider the the example of fancy indexing::

        A[A > 0]

    Here, the length of the resulting array depends on the data contained
    in *A* and cannot be statically determined at code generation time.

    In the case of data-dependent shapes, the shape is expressed in terms of
    scalar (i.e. an :attr:`Array.shape` of `()`) values
    with an integral :attr:`Array.dtype` (i.e. ``dtype.kind == "i"``)
    referenced by name from the :attr:`Array.namespace`. Such a name
    marks the boundary between eager and lazy evaluation.

-   :class:`IndexLambda` is used to express the following functionality:

    - Broadcasting (equivalently, outer products)
    - Slicing
    - Arithmetic
    - Element-wise function application, similar :class:`numpy.ufunc`
    - Reductions
    - Prefix sums/scans

    No new expression node types should be created for operations that
    are well-expressed by :class:`IndexLambda`. An operation is well-expressed
    if it is possible to (reasonably equivalently) recover the operation
    (and its inputs and ordering with respect to other preceding/following
    operations) by examining :attr:`IndexLambda.expr`.

    FIXME: This is not sharp. I'm not sure how to make it sharp.

Naming
------

-   There is one (for now) :class:`~Namespace` per computation that defines the
    computational "environment".  Operations involving array expressions not
    using the same namespace are prohibited.

-   Names in the :class:`~Namespace` are under user control and unique. I.e.
    new names in the :class:`~Namespace` that are not a
    :ref:`reserved_identifier` are not generated automatically without explicit
    user input.

-   The (array) value associated with a name is immutable once evaluated.
    In-place slice assignment may be simulated by returning a new
    node realizing a "partial replacement".

-   For arrays with data-dependent shapes, such as fancy indexing::

        A[A > 0]

    it may be necessary to automatically generate names, in this
    case to describe the shape of the index array used to realize
    the access ``A[A>0]``. These will be drawn from the reserved namespace
    ``_pt_shp``. Users may control the naming of these counts
    by assigning the tag :attr:`pytato.array.CountNamed`, like so::

        A[(A > 0).tagged(CountNamed("mycount"))]

-   :class:`Placeholder` expressions, like all array expressions,
    are considered read-only. When computation begins, the same
    actual memory may be supplied for multiple :term:`placeholder name`\ s,
    i.e. those arrays may alias.

    .. note::

        This does not preclude the arrays being declared with
        C's ``*restrict`` qualifier in generated code, as they
        do not alias any data that is being modified.

.. _reserved_identifier:

Reserved Identifiers
--------------------

-   Identifiers beginning with ``_pt_`` are reserved for internal use
    by :mod:`pytato`. Any such internal use must be drawn from one
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
        An object implementing the :class:`~Array` interface

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
