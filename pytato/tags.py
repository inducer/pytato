__doc__ = """
.. currentmodule:: pytato.tags


Pre-Defined Tags
----------------

.. autoclass:: ImplementationStrategy
.. autoclass:: ImplStored
.. autoclass:: ImplInlined
.. autoclass:: CountNamed
.. autoclass:: Named
.. autoclass:: PrefixNamed
"""


from pytools.tag import UniqueTag, tag_dataclass


# {{{ pre-defined tag: ImplementationStrategy

@tag_dataclass
class ImplementationStrategy(UniqueTag):
    """
    Metadata to be attached to :class:`pytato.Array` to convey information to a
    :class:`pytato.target.Target` on how it is supposed to be lowered.
    """


@tag_dataclass
class ImplStored(ImplementationStrategy):
    """
    An :class:`ImplementationStrategy` that is tagged to an
    :class:`~pytato.Array` to indicate that the :class:`~pytato.target.Target`
    must allocate a buffer for storing all the array's elements, and, all the
    users of the array must read from that buffer.
    """


@tag_dataclass
class ImplInlined(ImplementationStrategy):
    """
    An :class:`ImplementationStrategy` that is tagged to an
    :class:`~pytato.Array` to indicate that the :class:`~pytato.target.Target`
    should inline the tagged array's expression into its users.
    """

# }}}


# {{{ pre-defined tag: Named, CountNamed, PrefixNamed

@tag_dataclass
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


@tag_dataclass
class Named(_BaseNameTag):
    """
    Tagged to an :class:`~pytato.Array` to indicate the
    :class:`~pytato.target.Target` that if the tagged array is allocated to a
    variable, then it must be named :attr:`name`.

    .. attribute:: name
    """

    name: str


@tag_dataclass
class PrefixNamed(_BaseNameTag):
    """
    Tagged to an :class:`~pytato.Array` to indicate the
    :class:`~pytato.target.Target` that if the tagged array is allocated to a
    variable, then its name must begin with :attr:`prefix`.

    .. attribute:: prefix
    """

    prefix: str

# }}}
