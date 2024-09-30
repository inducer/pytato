"""Pytato specific exceptions."""
from __future__ import annotations


__copyright__ = "Copyright (C) 2021 Kaushik Kulkarni"

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


__doc__ = """

Pytato-specific exceptions
--------------------------

.. autoclass:: NameClashError
.. autoclass:: CannotBroadcastError
.. autoclass:: UnknownIndexLambdaExpr
.. autoclass:: CannotBeLoweredToIndexLambda
"""


class NameClashError(RuntimeError):
    """
    Raised when 2 non-identical :class:`~pytato.array.InputArgumentBase`'s are
    reachable in an :class:`~pytato.array.Array`'s DAG and share the same name. Here,
    we refer to 2 objects ``a`` and ``b`` as being identical iff ``a is b``.
    """


class CannotBroadcastError(ValueError):
    pass


class UnknownIndexLambdaExpr(ValueError):  # noqa: N818
    """
    Raised when the structure :class:`pytato.array.IndexLambda` could not be
    inferred.
    """
    pass


class InvalidEinsumIndex(ValueError):  # noqa: N818
    """
    Raised when an einsum index was referred by an invalid value.
    """


class NotAReductionAxis(ValueError):  # noqa: N818
    """
    Raised when a :class:`pytato.ReductionDescriptor` was referred by an invalid
    value.
    """


class CannotBeLoweredToIndexLambda(ValueError):  # noqa: N818
    """
    Raised when a :class:`pytato.Array` was expected to be lowered to an
    :class:`~pytato.array.IndexLambda`, but it cannot be. For ex. a
    :class:`pytato.loopy.LoopyCallResult` cannot be lowered to an
    :class:`~pytato.array.IndexLambda`.
    """
