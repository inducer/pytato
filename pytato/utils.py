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

from typing import Tuple, List, Union
from pytato.array import Array, ShapeType
from pytato.scalar_expr import ScalarExpression


def get_shape_after_broadcasting(
        exprs: List[Union[Array, ScalarExpression]]) -> ShapeType:
    """
    Returns the shape after broadcasting *exprs* in an operation.
    """
    shapes = [expr.shape if isinstance(expr, Array) else () for expr in exprs]

    result_dim = max(len(s) for s in shapes)

    # append leading dimensions of all the shapes with 1's to match result_dim.
    augmented_shapes = [((1,)*(result_dim-len(s)) + s) for s in shapes]

    result_shape = tuple(max(s[i] for s in augmented_shapes)
                         for i in range(result_dim))

    for s in augmented_shapes:
        for axis_len, result_dim in zip(s, result_shape):
            if (axis_len != result_dim) and (axis_len != 1):
                raise ValueError("operands could not be broadcasted together with "
                        f"shapes {' '.join(str(s) for s in shapes)}.")

    return result_shape


def get_indexing_expression(shape: ShapeType,
                            result_shape: ShapeType) -> Tuple[ScalarExpression, ...]:
    """
    Returns the indices while broadcasting an array of shape *shape* into one of
    shape *result_shape*.
    """
    assert len(shape) <= len(result_shape)
    i_start = len(result_shape) - len(shape)
    indices = []
    for i, (dim1, dim2) in enumerate(zip(shape, result_shape[i_start:])):
        if dim1 != dim2:
            assert dim1 == 1
            indices.append(0)
        else:
            assert dim1 == dim2
            indices.append(prim.Variable(f"_{i+i_start}"))

    return tuple(indices)


def with_indices_for_broadcasted_shape(val: prim.Variable, shape: ShapeType,
                                       result_shape: ShapeType) -> prim.Expression:
    if len(shape) == 0:
        # scalar expr => do not index
        return val
    else:
        return val[get_indexing_expression(shape, result_shape)]
