"""
.. currentmodule:: pytato.transform.calls

.. autofunction:: inline_calls
.. autofunction:: tag_all_calls_to_be_inlined
"""
from __future__ import annotations


__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

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


from typing import TYPE_CHECKING

from pytato.array import (
    AbstractResultWithNamedArrays,
    Array,
    DictOfNamedArrays,
    Placeholder,
)
from pytato.function import Call, NamedCallResult
from pytato.tags import InlineCallTag
from pytato.transform import ArrayOrNames, CopyMapper, _verify_is_array


if TYPE_CHECKING:
    from collections.abc import Mapping


# {{{ inlining

class PlaceholderSubstitutor(CopyMapper):
    """
    .. attribute:: substitutions

        A mapping from the placeholder name to the array that it is to be
        substituted with.
    """

    def __init__(self, substitutions: Mapping[str, Array]) -> None:
        # Ignoring function cache, not needed
        super().__init__()
        self.substitutions = substitutions

    def map_placeholder(self, expr: Placeholder) -> Array:
        return self.substitutions[expr.name]


class Inliner(CopyMapper):
    """
    Primary mapper for :func:`inline_calls`.
    """
    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        # inline call sites within the callee.
        new_expr = super().map_call(expr)
        assert isinstance(new_expr, Call)

        if expr.tags_of_type(InlineCallTag):
            substitutor = PlaceholderSubstitutor(new_expr.bindings)

            return DictOfNamedArrays(
                {name: _verify_is_array(substitutor.rec(ret))
                 for name, ret in new_expr.function.returns.items()},
                tags=new_expr.tags
            )
        else:
            return new_expr

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        new_call_or_inlined_expr = self.rec(expr._container)
        assert isinstance(new_call_or_inlined_expr, AbstractResultWithNamedArrays)
        if isinstance(new_call_or_inlined_expr, Call):
            return new_call_or_inlined_expr[expr.name]
        else:
            return new_call_or_inlined_expr[expr.name].expr


class InlineMarker(CopyMapper):
    """
    Primary mapper for :func:`tag_all_calls_to_be_inlined`.
    """
    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        return super().map_call(expr).tagged(InlineCallTag())


def inline_calls(expr: ArrayOrNames) -> ArrayOrNames:
    """
    Returns a copy of *expr* with call sites tagged with
    :class:`pytato.tags.InlineCallTag` inlined into the expression graph.
    """
    inliner = Inliner()
    return inliner(expr)


def tag_all_calls_to_be_inlined(expr: ArrayOrNames) -> ArrayOrNames:
    """
    Returns a copy of *expr* with all reachable instances of
    :class:`pytato.function.Call` nodes tagged with
    :class:`pytato.tags.InlineCallTag`.

    .. note::

       This routine does NOT inline calls, to inline the calls
       use :func:`tag_all_calls_to_be_inlined` on this routine's
       output.
    """
    return InlineMarker()(expr)

# }}}

# vim:foldmethod=marker
