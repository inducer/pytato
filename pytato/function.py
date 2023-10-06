from __future__ import annotations

__doc__ = """
.. currentmodule:: pytato

.. autofunction:: trace_call

.. currentmodule:: pytato.function

.. autoclass:: Call
.. autoclass:: NamedCallResult
.. autoclass:: FunctionDefinition
.. autoclass:: ReturnType

.. class:: ReturnT

    A type variable corresponding to the return type of the function
    :func:`pytato.trace_call`.
"""

__copyright__ = """
Copyright (C) 2022 Andreas Kloeckner
Copyright (C) 2022 Kaushik Kulkarni
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

import attrs
import re
import enum

from typing import (Callable, Dict, FrozenSet, Tuple, Union, TypeVar, Optional,
                    Hashable, Sequence, ClassVar, Iterator, Iterable, Mapping)
from immutabledict import immutabledict
from functools import cached_property
from pytato.array import (Array,  AbstractResultWithNamedArrays,
                          Placeholder, NamedArray, ShapeType, _dtype_any,
                          InputArgumentBase)
from pytools.tag import Tag, Taggable

ReturnT = TypeVar("ReturnT", Array, Tuple[Array, ...], Dict[str, Array])


# {{{ Call/NamedCallResult


@enum.unique
class ReturnType(enum.Enum):
    """
    Records the function body's return type in :class:`FunctionDefinition`.
    """
    ARRAY = 0
    DICT_OF_ARRAYS = 1
    TUPLE_OF_ARRAYS = 2


# eq=False to avoid equality comparison without EqualityMaper
@attrs.define(frozen=True, eq=False, hash=True)
class FunctionDefinition(Taggable):
    r"""
    A function definition that represents its outputs as instances of
    :class:`~pytato.Array` with the inputs being
    :class:`~pytato.array.Placeholder`\ s. The outputs of the function
    can be a single :class:`pytato.Array`, a tuple of :class:`pytato.Array`\ s or an
    instance of ``Dict[str, Array]``.

    .. attribute:: parameters

        Names of the input :class:`~pytato.array.Placeholder`\ s to the
        function node. This is a superset of the names of
        :class:`~pytato.array.Placeholder` instances encountered in
        :attr:`returns`. Unused parameters are allowed.

    .. attribute:: return_type

        An instance of :class:`ReturnType`.

    .. attribute:: returns

        The outputs of the function call which are array expressions that
        depend on the *parameters*. The keys of the mapping depend on
        :attr:`return_type` as:

            - If the function returns a single :class:`pytato.Array`, then
              *returns* contains a single array expression with ``"_"`` as the
              key.
            - If the function returns a :class:`tuple` of
              :class:`pytato.Array`\ s, then *returns* contains entries with
              the key ``"_N"`` mapping the ``N``-th entry of the result-tuple.
            - If the function returns a :class:`dict` mapping identifiers to
              :class:`pytato.Array`\ s, then *returns* uses the same mapping.

    .. automethod:: get_placeholder

    .. note::

        A :class:`FunctionDefinition` comes with its own namespace based on
        :attr:`parameters`. A :class:`~pytato.transform.Mapper`-implementer
        must ensure **not** to reuse the cached result between the caller's
        expressions and a function definition's expressions to avoid unsound
        cache hits that could lead to incorrect mappings.

    .. note::

        At this point, code generation/execution does not support
        distributed-memory communication nodes (:class:`~pytato.DistributedSend`,
        :class:`~pytato.DistributedRecv`) within function bodies.
    """
    parameters: FrozenSet[str]
    return_type: ReturnType
    returns: Mapping[str, Array]
    tags: FrozenSet[Tag] = attrs.field(kw_only=True)

    @cached_property
    def _placeholders(self) -> Mapping[str, Placeholder]:
        from pytato.transform import InputGatherer
        from functools import reduce

        mapper = InputGatherer()

        all_input_args: FrozenSet[InputArgumentBase] = reduce(
            frozenset.union,
            (mapper(ary) for ary in self.returns.values()),
            frozenset()
        )

        return immutabledict({input_arg.name: input_arg
                    for input_arg in all_input_args
                    if isinstance(input_arg, Placeholder)})

    def get_placeholder(self, name: str) -> Placeholder:
        """
        Returns the instance of :class:`pytato.array.Placeholder` corresponding
        to the parameter *name* in function body.
        """
        return self._placeholders[name]

    def _with_new_tags(
            self: FunctionDefinition, tags: FrozenSet[Tag]) -> FunctionDefinition:
        return attrs.evolve(self, tags=tags)

    def __call__(self, **kwargs: Array
                 ) -> Union[Array,
                            Tuple[Array, ...],
                            Dict[str, Array]]:
        from pytato.array import _get_default_tags
        from pytato.utils import are_shapes_equal

        # {{{ sanity checks

        if self.parameters != frozenset(kwargs):
            missing_params = self.parameters - frozenset(kwargs)
            extra_params = self.parameters - frozenset(kwargs)

            raise TypeError(
                    "Incorrect arguments."
                    + (f" Missing: '{missing_params}'." if missing_params else "")
                    + (f" Extra: '{extra_params}'." if extra_params else "")
                    )

        for argname, expected_arg in self._placeholders.items():
            if expected_arg.dtype != kwargs[argname].dtype:
                raise ValueError(f"Argument '{argname}' expected to "
                                 f" be of type '{expected_arg.dtype}', got"
                                 f" '{kwargs[argname].dtype}'.")
            if not are_shapes_equal(expected_arg.shape, kwargs[argname].shape):
                raise ValueError(f"Argument '{argname}' expected to "
                                 f" have shape '{expected_arg.shape}', got"
                                 f" '{kwargs[argname].shape}'.")

        # }}}

        call_site = Call(self, bindings=immutabledict(kwargs),
                         tags=_get_default_tags())

        if self.return_type == ReturnType.ARRAY:
            return call_site["_"]
        elif self.return_type == ReturnType.TUPLE_OF_ARRAYS:
            return tuple(call_site[f"_{iarg}"]
                         for iarg in range(len(self.returns)))
        elif self.return_type == ReturnType.DICT_OF_ARRAYS:
            return {kw: call_site[kw] for kw in self.returns}
        else:
            raise NotImplementedError(self.return_type)


class NamedCallResult(NamedArray):
    """
    One of the arrays that are returned from a call to :class:`FunctionDefinition`.

    .. attribute:: call

        The function invocation that led to *self*.

    .. attribute:: name

        The name by which the returned array is referred to in
        :attr:`FunctionDefinition.returns`.
    """
    call: Call
    name: str
    _mapper_method: ClassVar[str] = "map_named_call_result"

    def __init__(self,
                 call: Call,
                 name: str) -> None:
        super().__init__(call, name,
                         axes=call.function.returns[name].axes,
                         tags=call.function.returns[name].tags)

    def with_tagged_axis(self, iaxis: int,
                         tags: Union[Sequence[Tag], Tag]) -> Array:
        raise ValueError("Tagging a NamedCallResult's axis is illegal, use"
                         " Call.with_tagged_axis instead")

    def tagged(self,
               tags: Union[Iterable[Tag], Tag, None]) -> NamedCallResult:
        raise ValueError("Tagging a NamedCallResult is illegal, use"
                         " Call.tagged instead")

    def without_tags(self,
                     tags: Union[Iterable[Tag], Tag, None],
                     verify_existence: bool = True,
                     ) -> NamedCallResult:
        raise ValueError("Untagging a NamedCallResult is illegal, use"
                         " Call.without_tags instead")

    @property
    def shape(self) -> ShapeType:
        assert isinstance(self._container, Call)
        return self._container.function.returns[self.name].shape

    @property
    def dtype(self) -> _dtype_any:
        assert isinstance(self._container, Call)
        return self._container.function.returns[self.name].dtype


# eq=False to avoid equality comparison without EqualityMapper
@attrs.define(frozen=True, eq=False, hash=True, cache_hash=True, repr=False)
class Call(AbstractResultWithNamedArrays):
    """
    Records an invocation to a :class:`FunctionDefinition`.

    .. attribute:: function

        The instance of :class:`FunctionDefinition` being called by this call site.

    .. attribute:: bindings

        A mapping from the placeholder names of :class:`FunctionDefinition` to
        their corresponding parameters in the invocation to :attr:`function`.

    """
    function: FunctionDefinition
    bindings: Mapping[str, Array]

    _mapper_method: ClassVar[str] = "map_call"

    copy = attrs.evolve

    if __debug__:
        def __attrs_post_init__(self) -> None:
            # check that the invocation parameters and the function definition
            # parameters agree with each other.
            assert frozenset(self.bindings) == self.function.parameters
            super().__attrs_post_init__()

    def __contains__(self, name: object) -> bool:
        return name in self.function.returns

    def __iter__(self) -> Iterator[str]:
        return iter(self.function.returns)

    def __getitem__(self, name: str) -> NamedCallResult:
        return NamedCallResult(self, name)

    def __len__(self) -> int:
        return len(self.function.returns)

    def _with_new_tags(self: Call, tags: FrozenSet[Tag]) -> Call:
        return attrs.evolve(self, tags=tags)

# }}}


# {{{ user-facing routines

class _Guess:
    pass


RE_ARGNAME = re.compile(r"^_pt_(\d+)$")


def trace_call(f: Callable[..., ReturnT],
               *args: Array,
               identifier: Optional[Hashable] = _Guess,
               **kwargs: Array) -> ReturnT:
    """
    Returns the expressions returned after calling *f* with the arguments
    *args* and keyword arguments *kwargs*. The subexpressions in the returned
    expressions are outlined (opposite of 'inlined') as a
    :class:`~pytato.function.FunctionDefinition`.

    :arg identifier: A hashable object that acts as
        :attr:`pytato.tags.FunctionIdentifier.identifier` for the
        :class:`~pytato.tags.FunctionIdentifier` tagged to the outlined
        :class:`~pytato.function.FunctionDefinition`. If ``None`` the function
        definition is not tagged with a
        :class:`~pytato.tags.FunctionIdentifier` tag, if ``_Guess`` the
        function identifier is guessed from ``f.__name__``.
    """
    from pytato.tags import FunctionIdentifier
    from pytato.array import _get_default_tags

    if identifier is _Guess:
        # partials might not have a __name__ attribute
        identifier = getattr(f, "__name__", None)

    for kw in kwargs:
        if RE_ARGNAME.match(kw):
            # avoid collision between argument names
            raise ValueError(f"Kw argument named '{kw}' not allowed.")

    # Get placeholders from the ``args``, ``kwargs``.
    pl_args = tuple(Placeholder(name=f"in__pt_{iarg}",
                                shape=arg.shape, dtype=arg.dtype,
                                axes=arg.axes, tags=arg.tags)
                    for iarg, arg in enumerate(args))
    pl_kwargs = {kw: Placeholder(name=f"in_{kw}", shape=arg.shape,
                                 dtype=arg.dtype, axes=arg.axes, tags=arg.tags)
                 for kw, arg in kwargs.items()}

    # Pass the placeholders
    output = f(*pl_args, **pl_kwargs)

    if isinstance(output, Array):
        returns = {"_": output}
        return_type = ReturnType.ARRAY
    elif isinstance(output, tuple):
        assert all(isinstance(el, Array) for el in output)
        returns = {f"_{iout}": out for iout, out in enumerate(output)}
        return_type = ReturnType.TUPLE_OF_ARRAYS
    elif isinstance(output, dict):
        assert all(isinstance(el, Array) for el in output.values())
        returns = output
        return_type = ReturnType.DICT_OF_ARRAYS
    else:
        raise ValueError("The function being traced must return one of"
                         f"pytato.Array, tuple, dict. Got {type(output)}.")

    # construct the function
    function = FunctionDefinition(
        frozenset(pl_arg.name for pl_arg in pl_args) | frozenset(pl_kwargs),
        return_type,
        immutabledict(returns),
        tags=_get_default_tags() | (frozenset([FunctionIdentifier(identifier)])
                                    if identifier
                                    else frozenset())
    )

    # type-ignore-reason: return type is dependent on dynamic state i.e.
    # ret_type and hence mypy is unhappy
    return function(  # type: ignore[return-value]
        **{pl.name: arg for pl, arg in zip(pl_args, args)},
        **{pl_kwargs[kw].name: arg for kw, arg in kwargs.items()}
    )

# }}}

# vim:foldmethod=marker
