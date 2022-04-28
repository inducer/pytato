import numpy as np
import pymbolic.primitives as p

from enum import Enum, auto, unique
from typing import List, Tuple, Mapping, Sequence
from pytato.array import IndexLambda, ArrayOrScalar, Array, ShapeType
from pytato.diagnostic import UnknownIndexLambdaExpr
from pytato.utils import (get_indexing_expression,
                          get_shape_after_broadcasting,
                          are_shape_components_equal)
from pytato.scalar_expr import ScalarType, ScalarExpression, Reduce, SCALAR_CLASSES
from pytato.reductions import ReductionOperation
from dataclasses import dataclass
from pyrsistent import pmap
from pyrsistent.typing import PMap


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
    BITWISE_OR = auto()
    BITWISE_AND = auto()
    BITWISE_XOR = auto()
    TRUEDIV = auto()
    FLOORDIV = auto()
    POWER = auto()
    MOD = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()


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
class BroadcastOp(HighLevelOp):
    x: Array


@dataclass(frozen=True, eq=True, repr=True)
class LogicalNotOp(HighLevelOp):
    x: Array


@dataclass(frozen=True, eq=True, repr=True)
class ReduceOp(HighLevelOp):
    """
    .. attribute:: axes

        A mapping from the dimension index of the array to be reduced to the
        name of the reduction variable in :attr:`~pytato.array.IndexLambda.expr`.
    """
    op: ReductionOperation
    x: Array
    axes: PMap[int, str]

# }}}


PT_C99UNARY_FUNCS = {"abs", "sin", "cos", "tan", "asin", "acos", "atan",
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
        elif isinstance(expr, p.NaN):
            if expr.data_type:
                result.append(expr.data_type(float("nan")))
            else:
                result.append(np.nan)
        else:
            raise UnknownIndexLambdaExpr()

    return tuple(result)


def _is_idx_lambda_broadcast_op(expr: IndexLambda) -> bool:
    if (isinstance(expr.expr, p.Subscript)
            and isinstance(expr.expr.aggregate, p.Variable)):
        input_name = expr.expr.aggregate.name
    elif isinstance(expr.expr, p.Variable):
        input_name = expr.expr.name
    else:
        return False

    from_shape = expr.bindings[input_name].shape
    to_shape = expr.shape

    for in_dim, brdcst_dim in zip(from_shape,
                                  to_shape[-len(from_shape):]):
        if (not are_shape_components_equal(in_dim, brdcst_dim)
                and not are_shape_components_equal(in_dim, 1)):
            return False

    return True


def _is_normal_reduce_expr(expr: IndexLambda) -> bool:
    if not isinstance(expr.expr, Reduce):
        return False

    if not isinstance(expr.expr.inner_expr, p.Subscript):
        return False

    input_ary = expr.bindings[expr.expr.inner_expr.aggregate.name]

    i_out_dim = 0

    for idim, idx in enumerate(expr.expr.inner_expr.index_tuple):
        if not isinstance(idx, p.Variable):
            return False

        if idx.name in expr.expr.bounds:
            lbound, ubound = expr.expr.bounds[idx.name]
            if (not isinstance(lbound, int) or not isinstance(ubound, int)):
                raise NotImplementedError("Parametric bound expressions not"
                                          " supported.")
            if not are_shape_components_equal(lbound, 0):
                return False
            if not are_shape_components_equal(ubound, input_ary.shape[idim]):
                return False
        else:
            if idx.name == f"_{i_out_dim}":
                if not are_shape_components_equal(input_ary.shape[idim],
                                                  expr.shape[i_out_dim]):
                    return False
                i_out_dim += 1
            else:
                return False

    return True


_SIMPLE_PYMBOLIC_BINARY_OP_MAP = {p.Sum:        BinaryOpType.ADD,
                                  p.Product:    BinaryOpType.MULT,
                                  p.LogicalOr:  BinaryOpType.LOGICAL_OR,
                                  p.LogicalAnd: BinaryOpType.LOGICAL_AND,
                                  p.BitwiseOr:  BinaryOpType.BITWISE_OR,
                                  p.BitwiseAnd: BinaryOpType.BITWISE_AND,
                                  p.BitwiseXor: BinaryOpType.BITWISE_XOR,
                                  p.Quotient:   BinaryOpType.TRUEDIV,
                                  p.FloorDiv:   BinaryOpType.FLOORDIV,
                                  p.Power:      BinaryOpType.POWER,
                                  p.Remainder:  BinaryOpType.MOD,
                                  }
_COMPARISON_OP_TO_BINARY_OP_MAP = {"==": BinaryOpType.EQUAL,
                                   "!=": BinaryOpType.NOT_EQUAL,
                                   "<":  BinaryOpType.LESS,
                                   ">":  BinaryOpType.GREATER,
                                   "<=": BinaryOpType.LESS_EQUAL,
                                   ">=": BinaryOpType.GREATER_EQUAL,
                                   }


def index_lambda_to_high_level_op(expr: IndexLambda) -> HighLevelOp:
    """
    Returns a :class:`HighLevelOp` corresponding *expr*.
    """
    if isinstance(expr.expr, SCALAR_CLASSES):
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
                                    p.LogicalOr, p.BitwiseOr, p.BitwiseAnd,
                                    p.BitwiseXor)):
            children = expr.expr.children
            bin_op = _SIMPLE_PYMBOLIC_BINARY_OP_MAP[type(expr.expr)]
        elif isinstance(expr.expr, p.Comparison):
            children = (expr.expr.left, expr.expr.right)
            bin_op = _COMPARISON_OP_TO_BINARY_OP_MAP[expr.expr.operator]
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

    if _is_normal_reduce_expr(expr):
        return ReduceOp(expr.expr.op,
                        expr.bindings[expr
                                      .expr
                                      .inner_expr
                                      .aggregate.name],
                        axes=pmap({i: idx.name
                                   for i, idx in enumerate(expr
                                                           .expr
                                                           .inner_expr
                                                           .index_tuple)
                                   if idx.name in expr.expr.bounds})
                        )

    if _is_idx_lambda_broadcast_op(expr):
        if isinstance(expr.expr, p.Subscript):
            return BroadcastOp(expr.bindings[expr.expr.aggregate.name])
        else:
            assert isinstance(expr.expr, p.Variable)
            return BroadcastOp(expr.bindings[expr.expr.name])

    raise UnknownIndexLambdaExpr(expr.expr)
