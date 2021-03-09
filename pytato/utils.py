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

import pymbolic.primitives as prim

from typing import Tuple
from pytato.array import ShapeType
from pytato.scalar_expr import ScalarExpression


def get_shape_after_broadcasting(s1: ShapeType, s2: ShapeType) -> ShapeType:
    s_big, s_small = (s1, s2) if len(s1) > len(s2) else (s2, s1)

    result = list(s_big[:-len(s_small)])

    for dim1, dim2 in zip(s_big[-len(s_small):], s_small):
        if dim1 == dim2 or (dim1 == 1) or (dim2 == 1):
            result.append(max(dim1, dim2))
        else:
            raise ValueError("operands could not be broadcast together with shapes "
                             f"{s1} {s2}.")

    return tuple(result)


def get_indexing_expression(s: ShapeType,
                            r: ShapeType) -> Tuple[ScalarExpression, ...]:
    """
    Returns the indices while broadcasting an array of shape *s* into one of shape
    *r*.
    """
    i_start = len(r) - len(s)
    indices = []
    for i, (dim1, dim2) in enumerate(zip(s, r[i_start:])):
        if dim1 != dim2:
            assert dim1 == 1
            indices.append(0)
        else:
            assert dim1 == dim2
            indices.append(prim.Variable(f"_{i+i_start}"))

    return tuple(indices)
