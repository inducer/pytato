"""
.. autoclass:: DeadCodeEliminator

.. currentmodule:: pytato

.. autoclass:: eliminate_dead_code
"""
from __future__ import annotations


__copyright__ = """
Copyright (C) 2025 Kaushik Kulkarni
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
from typing import TYPE_CHECKING

from constantdict import constantdict
from typing_extensions import override

import pymbolic.primitives as p

from pytato.transform import ArrayOrNamesTc, CopyMapper


if TYPE_CHECKING:
    from pytato.array import Array, IndexLambda


class DeadCodeEliminator(CopyMapper):
    @override
    def map_index_lambda(self, expr: IndexLambda) -> Array:
        if (
            isinstance(expr.expr, p.Call)
            and isinstance(expr.expr.function, p.Variable)
            and expr.expr.function.name == "pytato.zero"
        ):
            return expr.copy(expr=0, bindings=constantdict())

        return super().map_index_lambda(expr)


def eliminate_dead_code(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    """
    Removes dead subexpressions from *expr*.

    .. note::

       Currently the following sub-expressions are eliminated:

       * Arguments in calls to `pt.zero`.
    """
    mapper = DeadCodeEliminator()
    return mapper(expr)
