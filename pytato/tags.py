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
.. autoclass:: CreatedAt
.. autoclass:: ExpandedDimsReshape
.. autoclass:: FunctionIdentifier
.. autoclass:: CallImplementationTag
.. autoclass:: InlineCallTag
"""

from typing import Tuple, Hashable, Optional
from pytools.tag import Tag, UniqueTag, IgnoredForEqualityTag
from dataclasses import dataclass
from traceback import FrameSummary, StackSummary


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


@dataclass(frozen=True, eq=True)
class _PytatoFrameSummary:
    """Class to store a single call frame, similar to
    :class:`traceback.FrameSummary`, but immutable."""
    filename: str
    lineno: Optional[int]
    name: str
    line: Optional[str]

    def short_str(self, maxlen: int = 100) -> str:
        s = f"{self.filename}:{self.lineno}, in {self.name}():\n{self.line}"
        s1, s2 = s.split("\n")
        # Limit display to maxlen characters
        s1 = "[...] " + s1[len(s1)-maxlen:] if len(s1) > maxlen else s1
        s2 = s2[:maxlen] + " [...]" if len(s2) > maxlen else s2
        return s1 + "\n" + s2

    def __repr__(self) -> str:
        return f"{self.filename}:{self.lineno}, in {self.name}(): {self.line}"


@dataclass(frozen=True, eq=True)
class _PytatoStackSummary:
    """Class to store a list of :class:`_PytatoFrameSummary` call frames,
    similar to :class:`traceback.StackSummary`, but immutable."""
    frames: Tuple[_PytatoFrameSummary, ...]

    def to_stacksummary(self) -> StackSummary:
        frames = [FrameSummary(f.filename, f.lineno, f.name, line=f.line)
                  for f in self.frames]

        return StackSummary.from_list(frames)

    def short_str(self, maxlen: int = 100) -> str:
        from os.path import dirname

        # Find the first file in the frames that is not in pytato's internal
        # directories.
        for frame in reversed(self.frames):
            frame_dir = dirname(frame.filename)
            if (not frame_dir.endswith("pytato")
                    and not frame_dir.endswith("pytato/distributed")):
                return frame.short_str(maxlen)

        # Fallback in case we don't find any file that is not in the pytato/
        # directory (should be unlikely).
        return self.__repr__()

    def __repr__(self) -> str:
        return "\n  " + "\n  ".join([str(f) for f in self.frames])


# See
# https://mypy.readthedocs.io/en/stable/additional_features.html#caveats-known-issues
# on why this can not be '@tag_dataclass'.
@dataclass(init=True, eq=True, frozen=True, repr=True)
class CreatedAt(UniqueTag, IgnoredForEqualityTag):
    """
    A tag attached to a :class:`~pytato.Array` to store the traceback
    of where it was created.
    """

    traceback: _PytatoStackSummary

    def __repr__(self) -> str:
        return "CreatedAt(" + str(self.traceback) + ")"


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


@dataclass(frozen=True)
class FunctionIdentifier(UniqueTag):
    """
    A tag that can be attached to a :class:`~pytato.function.FunctionDefinition`
    node to to describe the function's identifier. One can use this to refer
    all instances of :class:`~pytato.function.FunctionDefinition`, for example in
    transformations.transform.calls.concatenate_calls`.

    .. attribute:: identifier
    """
    identifier: Hashable


@dataclass(frozen=True)
class CallImplementationTag(UniqueTag):
    """
    A tag that can be attached to a :class:`~pytato.function.Call` node to
    direct a :class:`~pytato.target.Target` how the call site should be
    lowered.
    """


@dataclass(frozen=True)
class InlineCallTag(CallImplementationTag):
    r"""
    A :class:`CallImplementationTag` that directs the
    :class:`pytato.target.Target` to inline the call site.
    """
