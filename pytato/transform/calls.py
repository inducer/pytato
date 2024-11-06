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

from typing import Mapping

from pytato.array import (
    AbstractResultWithNamedArrays,
    Array,
    DictOfNamedArrays,
    Placeholder,
)
from pytato.function import Call, FunctionDefinition, NamedCallResult
from pytato.tags import InlineCallTag
from pytato.transform import ArrayOrNames, CopyMapper, _SelfMapper


# {{{ inlining

class PlaceholderSubstitutor(CopyMapper):
    """
    .. attribute:: substitutions

        A mapping from the placeholder name to the array that it is to be
        substituted with.

    .. note::

        This mapper does not deduplicate subexpressions that occur in both the mapped
        expression and the substitutions. Must follow up with a
        :class:`pytato.transform.CopyMapper` if duplicates need to be removed.
    """

    def __init__(self, substitutions: Mapping[str, Array]) -> None:
        # Ignoring function cache, not needed
        super().__init__()
        self.substitutions = substitutions

    def map_placeholder(self, expr: Placeholder) -> Array:
        # Can't call rec() to remove duplicates here, because the substituted-in
        # expression may potentially contain unrelated placeholders whose names
        # collide with the ones being replaced
        return self.substitutions[expr.name]

    def map_function_definition(
            self, expr: FunctionDefinition) -> FunctionDefinition:
        # Only operates within the current stack frame
        return expr


class Inliner(CopyMapper):
    """
    Primary mapper for :func:`inline_calls`.
    """
    def __init__(
            self,
            _function_cache: dict[Hashable, CachedMapperFunctionT] | None = None
            ) -> None:
        # Must disable collision/duplication checking because we're combining
        # expressions that were previously in two different call stack frames
        # (and were thus cached separately)
        super().__init__(
            err_on_collision=False,
            err_on_no_op_duplication=False,
            _function_cache=_function_cache)

    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        return type(self)(_function_cache=self._function_cache)

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        if expr.tags_of_type(InlineCallTag):
            substitutor = PlaceholderSubstitutor(expr.bindings)

            return DictOfNamedArrays(
                {name: self.rec(substitutor(ret))
                 for name, ret in expr.function.returns.items()},
                tags=expr.tags
            )
        else:
            return super().map_call(expr)

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
        rec_expr = super().map_call(expr)
        if rec_expr.tags_of_type(InlineCallTag):
            return rec_expr
        else:
            return rec_expr.tagged(InlineCallTag())


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
