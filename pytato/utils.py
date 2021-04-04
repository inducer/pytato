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

import numpy as np
import pymbolic.primitives as prim

from typing import Tuple, List, Union, Callable, Any, Sequence, Dict
from pytato.array import Array, ShapeType, IndexLambda, DtypeOrScalar, ArrayOrScalar
from pytato.scalar_expr import (ScalarExpression, IntegralScalarExpression,
                                SCALAR_CLASSES)


def get_shape_after_broadcasting(
        exprs: List[Union[Array, ScalarExpression]]) -> ShapeType:
    """
    Returns the shape after broadcasting *exprs* in an operation.
    """
    shapes = [expr.shape if isinstance(expr, Array) else () for expr in exprs]

    result_dim = max(len(s) for s in shapes)

    # append leading dimensions of all the shapes with 1's to match result_dim.
    augmented_shapes = [((1,)*(result_dim-len(s)) + s) for s in shapes]

    def _get_result_axis_length(axis_lengths: List[IntegralScalarExpression]
                                ) -> IntegralScalarExpression:
        result_axis_len = axis_lengths[0]
        for axis_len in axis_lengths[1:]:
            if axis_len == result_axis_len:
                pass
            elif axis_len == 1:
                pass
            elif result_axis_len == 1:
                result_axis_len = axis_len
            else:
                raise ValueError("operands could not be broadcasted together with "
                                 f"shapes {' '.join(str(s) for s in shapes)}.")
        return result_axis_len

    return tuple(_get_result_axis_length([s[i] for s in augmented_shapes])
                 for i in range(result_dim))


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


def extract_dtypes_or_scalars(
        exprs: Sequence[ArrayOrScalar]) -> List[DtypeOrScalar]:
    dtypes: List[DtypeOrScalar] = []
    for expr in exprs:
        if isinstance(expr, Array):
            dtypes.append(expr.dtype)
        else:
            assert isinstance(expr, SCALAR_CLASSES)
            dtypes.append(expr)

    return dtypes


def update_bindings_and_get_broadcasted_expr(arr: ArrayOrScalar,
                                             bnd_name: str,
                                             bindings: Dict[str, Array],
                                             result_shape: ShapeType
                                             ) -> ScalarExpression:
    """
    Returns an instance of :class:`~pytato.scalar_expr.ScalarExpression` to address
    *arr* in a :class:`pytato.array.IndexLambda` of shape *result_shape*.
    """

    if isinstance(arr, SCALAR_CLASSES):
        return arr

    assert isinstance(arr, Array)
    bindings[bnd_name] = arr
    return with_indices_for_broadcasted_shape(prim.Variable(bnd_name),
                                              arr.shape,
                                              result_shape)


def broadcast_binary_op(a1: ArrayOrScalar, a2: ArrayOrScalar,
                        op: Callable[[ScalarExpression, ScalarExpression], ScalarExpression],  # noqa:E501
                        get_result_type: Callable[[DtypeOrScalar, DtypeOrScalar], np.dtype[Any]],  # noqa:E501
                        ) -> ArrayOrScalar:
    if isinstance(a1, SCALAR_CLASSES) and isinstance(a2, SCALAR_CLASSES):
        from pymbolic.mapper.evaluator import evaluate
        return evaluate(op(a1, a2))  # type: ignore

    if isinstance(a1, Array) and isinstance(a2, Array) and (
            a1.namespace is not a2.namespace):
        raise ValueError("Operands must belong to the same namespace.")

    namespace = next(a.namespace for a in [a1, a2] if isinstance(a, Array))

    result_shape = get_shape_after_broadcasting([a1, a2])
    dtypes = extract_dtypes_or_scalars([a1, a2])
    result_dtype = get_result_type(*dtypes)

    bindings: Dict[str, Array] = {}

    expr1 = update_bindings_and_get_broadcasted_expr(a1, "_in0", bindings,
                                                     result_shape)
    expr2 = update_bindings_and_get_broadcasted_expr(a2, "_in1", bindings,
                                                     result_shape)

    return IndexLambda(namespace,
                       op(expr1, expr2),
                       shape=result_shape,
                       dtype=result_dtype,
                       bindings=bindings)
