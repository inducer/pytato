Design Decisions in Pytato
==========================

.. currentmodule:: pytato

-   Results of computations are either implement the :class:`~Array`
    interface or are a :class:`~DictOfNamedArrays`.
    The former are referred to as an :term:`array expression`. The union type
    of both of them is referred to as an *array result*. (FIXME? name)

-   There is one (for now) computation :class:`~Namespace` per
    computation that defines the computational "environment".
    Operations involving array expressions not using the same
    namespace are prohibited.

-   Names in the :class:`~Namespace` are under user control
    and unique. I.e. new names in the :class:`~Namespace` outside
    the reserved sub-namespace of identifiers beginning with
    ``_pt`` are not generated automatically without explicit user requests.

-   :attr:`Array.shape` and :attr:`Array.dtype` are evaluated eagerly.

-   Array data is computed lazily, i.e. a representation of the desired
    computation is built, but computation/code generation is not carried
    out until instructed by the user. Evaluation/computation
    is never triggered implicitly.

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

Glossary
--------

.. glossary::

    array expression
        An object implementing the :clas:`~Array` interface

    array result
        An :class:`array expression` or an instance of
        :class:`~DictOfNamedArrays`.

    identifier
        Any string matching the regular expression
        ``[a-zA-z_][a-zA-Z0-9_]+`` that does not
        start with ``_pt``, ``_lp``, or a double underscore.

Reserved Identifiers
--------------------

Identifiers beginning with ``_pt`` are reserved for internal use
by :module:`pytato`. Any such internal use must be drawn from one
of the following sub-regions, identified by their identifier
prefixes:

-   ``_pt_shp``: Used to automatically generate identifiers used
    in data-dependent shapes.
