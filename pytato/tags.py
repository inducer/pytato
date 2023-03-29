__doc__ = """
Pre-Defined Tags
----------------

.. autoclass:: ImplementationStrategy
.. autoclass:: ImplStored
.. autoclass:: ImplInlined
.. autoclass:: CountNamed
.. autoclass:: Named
.. autoclass:: PrefixNamed
.. autoclass:: AssumeNonNegative
.. autoclass:: ExpandedDimsReshape
"""

from dataclasses import dataclass
from typing import Tuple

from pytools.tag import Tag, UniqueTag

# {{{ pre-defined tag: ImplementationStrategy


@dataclass(frozen=True)
class ImplementationStrategy(UniqueTag):
    """
    Metadata to be attached to :class:`pytato.Array` to convey information to a
    :class:`pytato.target.Target` on how it is supposed to be lowered.
    """


@dataclass(frozen=True)
class ImplStored(ImplementationStrategy):
    """
    An :class:`ImplementationStrategy` that is tagged to an
    :class:`~pytato.Array` to indicate that the :class:`~pytato.target.Target`
    must allocate a buffer for storing all the array's elements, and, all the
    users of the array must read from that buffer.
    """


@dataclass(frozen=True)
class ImplInlined(ImplementationStrategy):
    """
    An :class:`ImplementationStrategy` that is tagged to an
    :class:`~pytato.Array` to indicate that the :class:`~pytato.target.Target`
    should inline the tagged array's expression into its users.
    """

# }}}


# {{{ pre-defined tag: Named, CountNamed, PrefixNamed

@dataclass(frozen=True)
class CountNamed(UniqueTag):
    """
    Tagged to a :class:`bool`-dtyped :class:`~pytato.Array` ``A``. If ``A``
    appears as one of the indices in :class:`~pytato.array.IndexBase`, the
    number of *True* values in ``A`` is assigned to a variable named :attr:`name`
    in the generated code.

    .. attribute:: name
    """

    name: str


class _BaseNameTag(UniqueTag):
    pass


@dataclass(frozen=True)
class Named(_BaseNameTag):
    """
    Tagged to an :class:`~pytato.Array` to indicate the
    :class:`~pytato.target.Target` that if the tagged array is allocated to a
    variable, then it must be named :attr:`name`.

    .. attribute:: name
    """

    name: str


@dataclass(frozen=True)
class PrefixNamed(_BaseNameTag):
    """
    Tagged to an :class:`~pytato.Array` to indicate the
    :class:`~pytato.target.Target` that if the tagged array is allocated to a
    variable, then its name must begin with :attr:`prefix`.

    .. attribute:: prefix
    """

    prefix: str

# }}}


@dataclass(frozen=True)
class AssumeNonNegative(Tag):
    """
    A tag attached to a :class:`~pytato.Array` to indicate the
    :class:`~pytato.target.Target` that all entries of the tagged array are
    non-negative.
    """


@dataclass(frozen=True)
class ExpandedDimsReshape(UniqueTag):
    """
    A tag that can be attached to a :class:`~pytato.array.Reshape` to indicate
    that the new dimensions created by :func:`pytato.expand_dims`.

    :attr new_dims: A :class:`tuple` of the dimensions of the reshaped array
        that were added.

    .. testsetup::

        >>> import pytato as pt

    .. doctest::

        >>> x = pt.make_placeholder("x", (10, 4), "float64")
        >>> pt.expand_dims(x, (0, 2, 4)).tags_of_type(pt.tags.ExpandedDimsReshape)
        frozenset({ExpandedDimsReshape(new_dims=(0, 2, 4))})
    """
    new_dims: Tuple[int, ...]
