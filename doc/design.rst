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

.. note::

    When this document refers to different ways of expressing a computation
    and transforming between them "without loss of information", what is meant
    is that the transformation is valid

    - in both directions, and
    - for all possible inputs (including those with symbolic shapes).

Computation and Results
-----------------------

-   Results of computations either implement the :class:`~Array`
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

    Consider the example of fancy indexing::

        A[A > 0]

    Here, the length of the resulting array depends on the data contained
    in *A* and cannot be statically determined at code generation time.

    In the case of data-dependent shapes, the shape is expressed in terms of
    scalar (i.e. having a :attr:`Array.shape` of `()`) values
    with an integral :attr:`Array.dtype` (i.e. having ``dtype.kind == "i"``).
    Such an expression marks the boundary between eager and lazy evaluation.

-   :attr:`Array.shape` is required to be an affine expression in terms of the
    instances of :class:`~pytato.SizeParam` in the computation graph. This
    permits shape inference to use Presburger arithmetic, meaning that shape
    inference is always decidable.

-   There is (deliberate) overlap in what various expression nodes can
    express, e.g.

    -   Array reshaping can be expressed as a :class:`pytato.Reshape`
        or as an :class:`pytato.IndexLambda`

    -   Linear algebra operations can be expressed via :class:`pytato.Einsum`
        or as an :class:`pytato.IndexLambda`

    Expression capture (the "frontend") should use the "highest-level"
    (most abstract) node type available that captures the user-intended
    operation. Lowering transformations (e.g. during code generation) may
    then convert these operations to a less abstract, more uniform
    representation.

    Operations that introduce nontrivial mappings on indices (e.g. reshape,
    strided slice, roll) are identified as potential candidates for being captured
    in their own high-level node vs. as an :class:`~pytato.array.IndexLambda`.

    Operations that *can* be expressed as :class:`~pytato.IndexLambda`
    without loss of information, *should* be expressed that way.

-   Every :class:`~pytato.Array` instance can be viewed a computation
    graph, where the :class:`~pytato.Array` instances form the nodes of
    the graph and there is an edge between a node and the array it uses. Since
    :class:`~pytato.Array` is an immutable type, the computation
    computation graphs would belong to the class of Directed Acyclic Graphs
    (DAGs). We choose the direction of the edges in the DAG to resemble the one
    typically seen in a data-flow graph, i.e. the successors of a node are its
    *users* and the predecessors of a node are the arrays that it *uses*.

    -   Borrowing the notation from `LLVM
        <https://llvm.org/doxygen/classllvm_1_1User.html>`__, we often refer to
        the direct successors of a node by *users*.

-   :func:`~pytato.zeros_like` (and similar) face some interesting conundrums
    in the context of metadata propagation (a la :func:`~pytato.unify_axes_tags`).
    A simple implementation would sever the connection between the original
    array and the new one. Unfortunately, if new metadata is added to the
    original array's axes during code generation, the new array will never
    find out. ``0 * orig_ary`` is an apparently-simple way around this,
    but with severe downsides: First, if NaNs occur in ``orig_ary``,
    they will propagate to the new array. Second, and in part
    as a consequence of the first, an actual dependency on the first
    array is created, preventing it from being eliminated even if
    otherwise unused.

    The current design uses a slight tweak of the latter approach,
    using a scalar function ``zero(...)`` that is guaranteed
    to return zero irrespective of its input. This permits
    the connection to the original array to be maintained for
    as long as desired and then finally (and thoroughly) cleared
    through :func:`~pytato.eliminate_dead_code`.


Naming
------

-   Input arrays, i.e. instances of :class:`~pytato.array.InputArgumentBase`,
    take ``Optional[str]`` as their names. If the name has not been
    provided, :mod:`pytato` assigns unique names to those arrays
    during lowering to a target IR.

-   No two non-identical array variables referenced in an expression may
    have the same name. :mod:`pytato` will detect such uses and raise an error.
    Here, "identical" is meant in the same-object ``a is b`` sense.

-   The (array) value associated with a name is immutable once evaluated.
    In-place slice assignment may be simulated by returning a new
    node realizing a "partial replacement".

-   For arrays with data-dependent shapes, such as fancy indexing::

        A[A > 0]

    it may be necessary to automatically generate names, in this
    case to describe the shape of the index array used to realize
    the access ``A[A>0]``. These will be drawn from the reserved namespace
    ``_pt_shp``. Users may control the naming of these counts
    by assigning the tag :attr:`pytato.tags.CountNamed`, like so::

        A[(A > 0).tagged(CountNamed("mycount"))]

-   :class:`pytato.Placeholder` expressions, like all array expressions,
    are considered read-only. When computation begins, the same
    actual memory may be supplied for multiple :term:`placeholder name`\ s,
    i.e. those arrays may alias.

    .. note::

        This does not preclude the arrays being declared with
        C's ``*restrict`` qualifier in generated code, as they
        do not alias any data that is being modified.

.. _reserved_identifier:

Reserved Identifiers
^^^^^^^^^^^^^^^^^^^^

-   Identifiers beginning with ``_pt_`` are reserved for internal use
    by :mod:`pytato`. Any such internal use must be drawn from one
    of the following sub-regions, identified by their identifier
    prefixes:

    -   ``_pt_shp``: Used to automatically generate identifiers used
        in data-dependent shapes.

    -   ``_pt_out``: The default name of an unnamed output argument

    -   ``_pt_in``: The default name of an unnamed placeholder argument

    -   ``_pt_data``: Used to automatically generate identifiers for
        names of :class:`~pytato.DataWrapper` arguments that are
        not supplied by the user.

    -   ``_pt_dist``: Used to automatically generate identifiers for
        placeholders at distributed communication boundaries.

-   Identifiers used in index lambdas are also reserved. These include:

    -   Identifiers matching the regular expression ``_[0-9]+``. They are used
        as index ("iname") placeholders.

    -   Identifiers matching the regular expression ``_r[0-9]+``. They are used
        as reduction indices.

    -   Identifiers matching the regular expression ``_in[0-9]+``. They are used
        as automatically generated names (if required) in
        :attr:`pytato.IndexLambda.bindings`.


.. note::

    Other than the iname names (``_[0-9]+``), these naming conventions are not
    compulsory. The above is merely intended to set aside parts of the namespace
    for this purpose that are guaranteed not to be trampled on by the user.

Tags
----

In order to convey information about the computation from DAG construction
time to processing/transformation/code generation time, each :class:`pytato.Array`
node may be tagged (via the :attr:`pytato.Array.tags` attribute) with an arbitrary
number of informational "tags". A tag is any subclass of :class:`pytools.tag.Tag`.
Guidelines for tag use:

- Tags *must not* carry semantic information; i.e. a computation must have the same
  result even if all tags are stripped.

- Tags *may* carry information related to efficient execution, i.e. it is
  permissible that evaluation of the expression is inefficient (even
  impractically so) without taking the information in the tags into
  account.

- Tags *should* be descriptive, not prescriptive.

  For example:

  - **Good:** This array is the result of differentiation.
  - **Bad:** Unroll the loops in the code computing this result.

Metadata Propagation
^^^^^^^^^^^^^^^^^^^^

Metadata (i.e. tags) is expected to be varied across application domains, not
just in its information content, but also in the rules which might govern the
way in which it propagates across the DAG. As an example, consider a Finite
Element Method solver that uses :mod:`pytato`-arrays to store the DOFs.

- A viable use of the tagging system in the solver could be to tag an array
  ``u`` to describe the first axis of an array denotes the mesh's element
  indices. In this case, after performing an operation ``u_squared = u * u`` is
  semantically valid to tag ``u_squared`` also with the tag describing that it's
  first denotes the mesh's element indices.
- Another viable use of the tagging system by the solver could be to encode the
  physical quantity that the array is storing. If ``u`` is an array tagged with
  metadata to describe that it represents the fluid's velocity. Then, the
  operation ``u_squared = u * u`` should not propagate that ``u_squared`` is
  also an array storing the fluid's velocity.


As a result, we choose to not propagate metadata (either upon construction, or
via a dedicated, universally applied pass) and instead supply tools (such as
:class:`~pytato.transform.Mapper`) to permit users to implement their own
tag/metadata propagation rules.

Memory layout
-------------

:mod:`pytato` arrays do not have a defined memory layout. Any operation in :mod:`numpy`
that relies on memory layout information to do its job is undefined in :mod:`pytato`.
At the most basic level, the attribute :attr:`numpy.ndarray.strides` is not available
on subclasses of :class:`pytato.Array`.

Dataclasses
-----------

:mod:`dataclasses` helps us reduce most of the boilerplate involved in
instantiating a new type.
We have checks in place to avoid developer errors that could happen by using
the defaults of this library. For example, :mod:`dataclasses` overrides the
implementation of ``__eq__`` for the class being implemented, which could
potentially lead to an `exponentially complex
operation <https://github.com/inducer/pytato/issues/163>`_.

Lessons learned
===============

Namespace object
----------------

In pytato's early days, there used to exist a ``Namespace`` type to define a
namespace for all input names within an array expression. This was however removed
in the later versions. As, in the process of associating names to array variables it
would privately hold references to :class:`~pytato.array.InputArgumentBase`
variables that could no longer be referenced by a user. This made it impossible for
the garbage collector to deallocate large :class:`~pytato.DataWrapper`'s,
unless the namespace itself went out-of-scope.

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

    placeholder name
        See :attr:`pytato.Placeholder.name`.

.. vim: shiftwidth=4
