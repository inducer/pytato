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

from numbers import Number
from typing import Tuple, List, Union, Callable, Any, Sequence, Dict
from pytato.array import (Array, ShapeType, IndexLambda, SizeParam, ShapeComponent,
                          DtypeOrScalar, ArrayOrScalar)
from pytato.scalar_expr import (ScalarExpression, IntegralScalarExpression,
                                SCALAR_CLASSES)
from pytools import UniqueNameGenerator
from pytato.transform import Mapper


__doc__ = """
Helper routines
---------------

.. autofunction:: are_shape_components_equal
.. autofunction:: are_shapes_equal
.. autofunction:: get_shape_after_broadcasting
.. autofunction:: dim_to_index_lambda_components
"""


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
            if are_shape_components_equal(axis_len, result_axis_len):
                pass
            elif are_shape_components_equal(axis_len, 1):
                pass
            elif are_shape_components_equal(result_axis_len, 1):
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
        if not are_shape_components_equal(dim1, dim2):
            assert are_shape_components_equal(dim1, 1)
            indices.append(0)
        else:
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
    if isinstance(a1, Number) and isinstance(a2, Number):
        from pytato.scalar_expr import evaluate
        return evaluate(op(a1, a2))  # type: ignore

    result_shape = get_shape_after_broadcasting([a1, a2])
    dtypes = extract_dtypes_or_scalars([a1, a2])
    result_dtype = get_result_type(*dtypes)

    bindings: Dict[str, Array] = {}

    expr1 = update_bindings_and_get_broadcasted_expr(a1, "_in0", bindings,
                                                     result_shape)
    expr2 = update_bindings_and_get_broadcasted_expr(a2, "_in1", bindings,
                                                     result_shape)

    return IndexLambda(op(expr1, expr2),
                       shape=result_shape,
                       dtype=result_dtype,
                       bindings=bindings)


# {{{ dim_to_index_lambda_components

class ShapeExpressionMapper(Mapper):
    """
    Mapper that takes a shape component and returns it as a scalar expression.
    """
    def __init__(self, var_name_gen: UniqueNameGenerator):
        self.cache: Dict[Array, ScalarExpression] = {}
        self.var_name_gen = var_name_gen
        self.bindings: Dict[str, SizeParam] = {}

    def rec(self, expr: Array) -> ScalarExpression:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: Array = super().rec(expr)
        self.cache[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda) -> ScalarExpression:
        from pytato.scalar_expr import substitute
        return substitute(expr.expr, {name: self.rec(val)
                                      for name, val in expr.bindings.items()})

    def map_size_param(self, expr: SizeParam) -> ScalarExpression:
        name = self.var_name_gen("_in")
        self.bindings[name] = expr
        return prim.Variable(name)


def dim_to_index_lambda_components(expr: ShapeComponent,
                                   vng: UniqueNameGenerator,
                                   ) -> Tuple[ScalarExpression,
                                              Dict[str, SizeParam]]:
    """
    Returns the scalar expressions and bindings to use the shape
    component within an index lambda.

    .. testsetup::

        >>> import pytato as pt
        >>> from pytato.utils import dim_to_index_lambda_components
        >>> from pytools import UniqueNameGenerator

    .. doctest::

        >>> n = pt.make_size_param("n")
        >>> expr, bnds = dim_to_index_lambda_components(3*n+8, UniqueNameGenerator())
        >>> print(expr)
        3*_in + 8
        >>> bnds  # doctest: +ELLIPSIS
        {'_in': <pytato.array.SizeParam ...>}
    """
    if isinstance(expr, int):
        return expr, {}

    assert isinstance(expr, Array)
    mapper = ShapeExpressionMapper(vng)
    result = mapper(expr)
    return result, mapper.bindings

# }}}


def are_shape_components_equal(dim1: ShapeComponent, dim2: ShapeComponent) -> bool:
    """
    Returns *True* iff *dim1* and *dim2* are have equal
    :class:`~pytato.array.SizeParam` coefficients in their expressions.
    """
    from pytato.scalar_expr import substitute, distribute

    def to_expr(dim: ShapeComponent) -> ScalarExpression:
        expr, bnds = dim_to_index_lambda_components(dim,
                                                    UniqueNameGenerator())

        return substitute(expr, {name: prim.Variable(bnd.name)
                                 for name, bnd in bnds.items()})

    dim1_expr = to_expr(dim1)
    dim2_expr = to_expr(dim2)
    # ScalarExpression.__eq__  returns Any
    return (distribute(dim1_expr-dim2_expr) == 0)  # type: ignore


def are_shapes_equal(shape1: ShapeType, shape2: ShapeType) -> bool:
    """
    Returns *True* iff *shape1* and *shape2* have the same dimensionality and the
    correpsonding components are equal as defined by
    :func:`~pytato.utils.are_shape_components_equal`.
    """
    return ((len(shape1) == len(shape2))
            and all(are_shape_components_equal(dim1, dim2)
                    for dim1, dim2 in zip(shape1, shape2)))
