from __future__ import annotations, absolute_import

import numpy as np
import pymbolic.primitives as prim

from typing import Dict, Callable, Any, Tuple

from pytato.array import (Array, IndexLambda, MatrixProduct,
                          _dtype_any,
                          Placeholder as PtPlaceholder,
                          DataWrapper as PtDataWrapper)
from pytato.match import (MatMul, Sum, Product, Id, Scalar, Dtype,
                          TupleOp,
                          Placeholder as MpPlaceholder,
                          DataWrapper as MpDataWrapper)
from pytato.transform import Mapper
from pytato.scalar_expr import ScalarType

from matchpy import Expression


class ToMatchpyExpressionMapper:
    def __init__(self) -> None:
        self.cache: Dict[Array, Expression] = {}

    def rec(self, expr: Array) -> Expression:
        if expr in self.cache:
            return self.cache[expr]

        method: Callable[[Array], Expression] = getattr(self, expr._mapper_method)

        return method(expr)

    __call__ = rec

    def map_data_wrapper(self, expr: PtDataWrapper) -> Expression:
        rec_shape = tuple(Scalar(dim) if np.isscalar(dim)
                          else self.rec(dim)
                          for dim in expr.shape)

        return MpDataWrapper(expr,
                             Id(expr.name), TupleOp(rec_shape), Dtype(expr.dtype))

    def map_placeholder(self, expr: PtPlaceholder) -> Expression:
        rec_shape = tuple(Scalar(dim) if np.isscalar(dim)
                          else self.rec(dim)
                          for dim in expr.shape)

        return MpPlaceholder(expr,
                             Id(expr.name), TupleOp(rec_shape), Dtype(expr.dtype))

    def map_index_lambda(self, expr: IndexLambda) -> Expression:
        # FIXME: Please FIXME, please.
        if (
                (isinstance(expr.expr, prim.Product)
                 and len(expr.expr.children) == 2
                 and np.isscalar(expr.expr.children[0]))
                and len(expr.bindings) == 1
                and frozenset(expr.bindings) == frozenset(["_in1"])):
            return Product(expr,
                           Scalar(expr.expr.children[0]),
                           self.rec(expr.bindings["_in1"]))
        elif (
                (isinstance(expr.expr, prim.Sum)
                 and len(expr.expr.children) == 2
                 and np.isscalar(expr.expr.children[1]))
                and len(expr.bindings) == 1
                and frozenset(expr.bindings) == frozenset(["_in0"])):
            return Sum(expr,
                       self.rec(expr.bindings["_in0"]),
                       Scalar(expr.expr.children[1]))
        elif (
                (isinstance(expr.expr, prim.Sum)
                 and len(expr.expr.children) == 2)
                and len(expr.bindings) == 2
                and frozenset(expr.bindings) == frozenset(["_in0", "_in1"])):
            assert expr.expr.children[0].aggregate.name == "_in0"
            assert expr.expr.children[1].aggregate.name == "_in1"
            return Sum(expr,
                       self.rec(expr.bindings["_in0"]),
                       self.rec(expr.bindings["_in1"]))
        else:
            raise NotImplementedError

    def map_matrix_product(self, expr: MatrixProduct) -> Expression:
        return MatMul(expr,
                      self.rec(expr.x1), self.rec(expr.x2))


class FromMatchpyExpressionMapper:
    def __init__(self) -> None:
        self.cache: Dict[Expression, Any] = {}

    def rec(self, expr: Expression) -> Any:
        if expr in self.cache:
            return self.cache[expr]

        method: Callable[[Expression], Any] = getattr(self, expr._mapper_method)

        return method(expr)

    __call__ = rec

    def map_dtype(self, expr) -> _dtype_any:
        return expr.value

    def map_scalar(self, expr: Scalar) -> ScalarType:
        return expr.value

    def map_id(self, expr: Id) -> str:
        return expr.value

    def map_tuple_op(self, expr: TupleOp[Any]) -> Tuple[Any, ...]:
        return tuple(self.rec(op) for op in expr.operands)

    def map_placeholder(self, expr: MpPlaceholder) -> Array:
        return PtPlaceholder(self.rec(expr.id),
                             self.rec(expr.shape),
                             self.rec(expr.dtype))

    def map_sum(self, expr: Sum) -> Array:
        return self.rec(expr.x1) + self.rec(expr.x2)

    def map_product(self, expr: Product) -> Array:
        return self.rec(expr.x1) * self.rec(expr.x2)

    def map_matmul(self, expr: MatMul) -> Array:
        return MatrixProduct(self.rec(expr.x1), self.rec(expr.x2))


class SubExpressionAtPathFinder(Mapper):
    def rec(self, expr: Array, path: Tuple[int, ...]) -> Array:
        method: Callable[[Array, Tuple[int, ...]], Expression]
        method = getattr(self, expr._mapper_method)
        return method(expr, path)

    __call__ = rec

    def map_data_wrapper(self, expr: PtDataWrapper, path: Tuple[int, ...]) -> Array:
        if path == ():
            return expr

        if path[0] == 0:
            return self.rec(expr.name, path[1:])
        elif path[0] == 1:
            return self.rec(expr.shape, path[1:])
        elif path[0] == 2:
            return self.rec(expr.dtype, path[1:])
        else:
            raise AssertionError

    def map_placeholder(self, expr: PtPlaceholder, path: Tuple[int, ...]) -> Array:
        if path == ():
            return expr

        if path[0] == 0:
            return self.rec(expr.name, path[1:])
        elif path[0] == 1:
            return self.rec(expr.shape, path[1:])
        elif path[0] == 2:
            return self.rec(expr.dtype, path[1:])
        else:
            raise AssertionError

    def map_index_lambda(self, expr: IndexLambda, path: Tuple[int, ...]) -> Array:
        if path == ():
            return expr

        # FIXME: Please FIXME, please.
        if (
                (isinstance(expr.expr, prim.Product)
                 and len(expr.expr.children) == 2
                 and np.isscalar(expr.expr.children[0]))
                and len(expr.bindings) == 1
                and frozenset(expr.bindings) == frozenset(["_in1"])):
            assert path[0] == 1
            return self.rec(expr.bindings["_in1"], path[1:])
        elif (
                (isinstance(expr.expr, prim.Sum)
                 and len(expr.expr.children) == 2
                 and np.isscalar(expr.expr.children[1]))
                and len(expr.bindings) == 1
                and frozenset(expr.bindings) == frozenset(["_in0"])):
            assert path[0] == 0
            return self.rec(expr.bindings["_in0"], path[1:])
        elif (
                (isinstance(expr.expr, prim.Sum)
                 and len(expr.expr.children) == 2)
                and len(expr.bindings) == 2
                and frozenset(expr.bindings) == frozenset(["_in0", "_in1"])):
            assert expr.expr.children[0].aggregate.name == "_in0"
            assert expr.expr.children[1].aggregate.name == "_in1"

            if path[0] == 0:
                return self.rec(expr.bindings["_in0"], path[1:])
            elif path[0] == 1:
                return self.rec(expr.bindings["_in1"], path[1:])
            else:
                raise AssertionError
        else:
            raise NotImplementedError

    def map_matrix_product(self,
                           expr: MatrixProduct,
                           path: Tuple[int, ...]) -> Array:
        if path == ():
            return expr

        assert (0 <= path[0] <= 1)

        if path[0] == 0:
            return self.rec(expr.x1, path[1:])
        elif path[0] == 1:
            return self.rec(expr.x2, path[1:])
        else:
            raise AssertionError
