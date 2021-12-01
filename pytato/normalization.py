import pymbolic.primitives as p
import numpy as np

from enum import Enum, auto, unique
from typing import List, Tuple, Mapping, Sequence
from pytato.array import IndexLambda, ArrayOrScalar, Array, ShapeType
from pytato.diagnostic import UnknownIndexLambdaExpr
from pytato.utils import (get_indexing_expression,
                          get_shape_after_broadcasting,
                          are_shape_components_equal)
from pytato.scalar_expr import ScalarType, ScalarExpression, Reduce, SCALAR_CLASSES
from dataclasses import dataclass


__doc__ = """

.. autoclass:: HighLevelOp

.. autofunction:: index_lambda_to_high_level_op
"""


# {{{ types of normalized index lambdas

class HighLevelOp:
    """
    Base class for all high level operations that could be raised from a
    :class:`pytato.array.IndexLambda`.
    """


@dataclass(frozen=True, eq=True, repr=True)
class FullOp(HighLevelOp):
    fill_value: ScalarType


@unique
class BinaryOpType(Enum):
    ADD = auto()
    SUB = auto()
    MULT = auto()
    LOGICAL_OR = auto()
    LOGICAL_AND = auto()
    TRUEDIV = auto()
    FLOORDIV = auto()
    POWER = auto()
    MOD = auto()


@dataclass(frozen=True, eq=True, repr=True)
class BinaryOp(HighLevelOp):
    """
    Records a ``x1 binary_op x2``-type expression.
    """
    binary_op: BinaryOpType
    x1: ArrayOrScalar
    x2: ArrayOrScalar


@dataclass(frozen=True, eq=True, repr=True)
class C99CallOp(HighLevelOp):
    function: str
    args: Tuple[ArrayOrScalar, ...]


@dataclass(frozen=True, eq=True, repr=True)
class WhereOp(HighLevelOp):
    condition: ArrayOrScalar
    then: ArrayOrScalar
    else_: ArrayOrScalar


@dataclass(frozen=True, eq=True, repr=True)
class ComparisonOp(HighLevelOp):
    operator: str
    left: ArrayOrScalar
    right: ArrayOrScalar


@dataclass(frozen=True, eq=True, repr=True)
class BroadcastOp(HighLevelOp):
    x: Array


@dataclass(frozen=True, eq=True, repr=True)
class LogicalNotOp(HighLevelOp):
    x: Array


@dataclass(frozen=True, eq=True, repr=True)
class ReduceOp(HighLevelOp):
    op: str
    x: Array
    axes: Tuple[int, ...]

# }}}


PT_C99UNARY_FUNCS = {"abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
                     "sinh", "cosh", "tanh", "exp", "log", "log10", "isnan",
                     "sqrt", "conj", "real", "imag"}

PT_C99BINARY_FUNCS = {"atan2"}


def _as_array_or_scalar(exprs: Sequence[ScalarExpression],
                        bindings: Mapping[str, Array],
                        out_shape: ShapeType
                        ) -> Tuple[ArrayOrScalar, ...]:
    """
    Helper routine invoked in :func:`index_lambda_to_high_level_op`. For every
    expression in *exprs* either infers (and returns) it as a scalar or infers
    (and returns) the array corresponding to one of *bindings* while defining
    an :class:`~pytato.array.IndexLambda` of *out_shape* shape.
    """

    result: List[ArrayOrScalar] = []
    if out_shape != get_shape_after_broadcasting(bindings.values()):
        raise UnknownIndexLambdaExpr()

    binding_to_subscript = {bnd_name: p.Subscript(
        p.Variable(bnd_name),
        get_indexing_expression(bnd.shape,
                                out_shape))
                            for bnd_name, bnd in bindings.items()}

    for expr in exprs:
        if isinstance(expr, SCALAR_CLASSES):
            result.append(expr)
        elif (isinstance(expr, p.Variable)
              and bindings[expr.name].shape == ()):
            result.append(bindings[expr.name])
        elif (isinstance(expr, p.Subscript)
              and isinstance(expr.aggregate, p.Variable)
              and (binding_to_subscript[expr.aggregate.name]
                   == expr)):
            result.append(bindings[expr.aggregate.name])
        else:
            raise UnknownIndexLambdaExpr()

    return tuple(result)


def _is_broadcastable(from_shape: ShapeType,
                      to_shape: ShapeType) -> bool:

    for in_dim, brdcst_dim in zip(from_shape,
                                  to_shape[-len(from_shape):]):
        if (not are_shape_components_equal(in_dim, brdcst_dim)
                and not are_shape_components_equal(in_dim, 1)):
            return False

    return True


_SIMPLE_PYMBOLIC_BINARY_OP_MAP = {p.Sum:        BinaryOpType.ADD,
                                  p.Product:    BinaryOpType.MULT,
                                  p.LogicalOr:  BinaryOpType.LOGICAL_OR,
                                  p.LogicalAnd: BinaryOpType.LOGICAL_AND,
                                  p.Quotient:   BinaryOpType.TRUEDIV,
                                  p.FloorDiv:   BinaryOpType.FLOORDIV,
                                  p.Power:      BinaryOpType.POWER,
                                  p.Remainder:  BinaryOpType.MOD,
                                  }


def index_lambda_to_high_level_op(expr: IndexLambda) -> HighLevelOp:
    """
    Returns a :class:`HighLevelOp` corresponding *expr*.
    """
    if np.isscalar(expr.expr):
        return FullOp(expr.expr)

    # {{{ binary ops

    try:
        if isinstance(expr.expr, (p.Quotient, p.FloorDiv, p.Remainder)):
            children = (expr.expr.numerator, expr.expr.denominator)
            bin_op = _SIMPLE_PYMBOLIC_BINARY_OP_MAP[type(expr.expr)]
        elif isinstance(expr.expr, p.Power):
            children = (expr.expr.base, expr.expr.exponent)
            bin_op = _SIMPLE_PYMBOLIC_BINARY_OP_MAP[type(expr.expr)]
        elif (isinstance(expr.expr, p.Sum)
                and len(expr.expr.children) == 2
                and isinstance(expr.expr.children[1], p.Product)
                and len(expr.expr.children[1].children) == 2
                and expr.expr.children[1].children[0] == -1):
            children = (expr.expr.children[0],
                        expr.expr.children[1].children[1])
            bin_op = BinaryOpType.SUB
        elif isinstance(expr.expr, (p.Sum, p.Product, p.LogicalAnd,
                                     p.LogicalOr)):
            children = expr.expr.children
            bin_op = _SIMPLE_PYMBOLIC_BINARY_OP_MAP[type(expr.expr)]
        else:
            raise UnknownIndexLambdaExpr

        return BinaryOp(bin_op,
                        *_as_array_or_scalar(children,
                                             expr.bindings,
                                             expr.shape))
    except UnknownIndexLambdaExpr:
        pass

    # }}}

    if (isinstance(expr.expr, p.Call)
        and expr.expr.function.name.startswith("pytato.c99.")
        and expr.expr.function.name[11:] in (PT_C99UNARY_FUNCS
                                             | PT_C99BINARY_FUNCS)):
        # TODO: Check types agree with function signature
        try:
            return C99CallOp(expr.expr.function.name[11:],
                             _as_array_or_scalar(expr.expr.parameters,
                                                 expr.bindings,
                                                 expr.shape))
        except UnknownIndexLambdaExpr:
            pass

    if isinstance(expr.expr, p.Comparison):
        try:
            return ComparisonOp(expr.expr.operator,
                                *_as_array_or_scalar((expr.expr.left,
                                                      expr.expr.right),
                                                     expr.bindings,
                                                     expr.shape))
        except UnknownIndexLambdaExpr:
            pass

    if isinstance(expr.expr, p.If):
        try:
            return WhereOp(*_as_array_or_scalar((expr.expr.condition,
                                                 expr.expr.then,
                                                 expr.expr.else_),
                                                expr.bindings,
                                                expr.shape))
        except UnknownIndexLambdaExpr:
            pass

    if isinstance(expr.expr, p.LogicalNot):
        try:
            ary, = _as_array_or_scalar((expr.expr,),
                                       expr.bindings,
                                       expr.shape)
            assert isinstance(ary, Array)
            return LogicalNotOp(ary)
        except UnknownIndexLambdaExpr:
            pass

    if (isinstance(expr.expr, Reduce)
        and isinstance(expr.expr.inner_expr, p.Subscript)
        and (expr.expr.inner_expr.aggregate.name in expr.bindings)
        and (expr.bindings[expr.expr.inner_expr.aggregate.name].ndim
             == len(expr.expr.inner_expr.index_tuple))):

        # {{{ extract the bounds

        idx_to_bounds = {}
        for i, dim in enumerate(expr.shape):
            if not isinstance(dim, int):
                raise NotImplementedError("Raising reductions with parametric bounds"
                                          " not yet supported.")

            idx_to_bounds[f"_{i}"] = (0, dim)

        for name, (lbound, ubound) in expr.expr.bounds.items():
            if not (isinstance(lbound, int)
                    and isinstance(ubound, int)):
                raise NotImplementedError("Raising reductions with parametric bounds"
                                          " not yet supported.")

            idx_to_bounds[name] = (lbound, ubound)

        # }}}

        if all((isinstance(idx, p.Variable)
                and idx_to_bounds[idx.name] == (0, dim))
               for idx, dim in zip(expr.expr.inner_expr.index_tuple,
                                   expr
                                   .bindings[expr.expr.inner_expr.aggregate.name]
                                   .shape)):
            return ReduceOp(expr.expr.op,
                            expr.bindings[expr
                                          .expr
                                          .inner_expr
                                          .aggregate.name],
                            axes=tuple(i
                                       for i, idx in enumerate(expr
                                                               .expr
                                                               .inner_expr
                                                               .index_tuple)
                                       if idx.name in expr.expr.bounds)
                            )

    if (isinstance(expr.expr, p.Subscript)
        and isinstance(expr.expr.aggregate, p.Variable)
        and expr.expr.aggregate.name in expr.bindings
        and _is_broadcastable(
            expr.bindings[expr.expr.aggregate.name].shape,
            expr.shape)):
        return BroadcastOp(expr.bindings[expr.expr.aggregate.name])

    raise UnknownIndexLambdaExpr(expr.expr)
