"""Pattern matching support for Pytato DAGs using MatchPy."""
from __future__ import annotations, absolute_import


import abc

from typing import (Union, ClassVar, Optional, Iterator, Mapping,
                    Generic, TypeVar, Tuple)
from dataclasses import dataclass, fields, field

from pytato.scalar_expr import ScalarType
from pytato.array import (ArrayOrScalar,  _dtype_any,
                          Array)

from matchpy import (Operation, Arity, Symbol, Expression,
                     Atom as BaseAtom,
                     Wildcard as BaseWildcard)


NumpyOpOrScalarT = Union["NumpyOp", ScalarType]
IndexExpr = Union[Symbol, "Slice", "NumpyOp"]
ConstantT = TypeVar("ConstantT")
TupleOpT = TypeVar("TupleOpT")

np_op_dataclass = dataclass(frozen=True, eq=True)
_NOT_OPERAND_METADATA = {"not_an_operand": True}


@np_op_dataclass
class Constant(BaseAtom, Generic[ConstantT]):
    """
    A dtype constant.
    """
    value: ConstantT
    variable_name: Optional[str] = None

    @property
    def head(self):
        return self

    def __lt__(self, other):
        if not isinstance(other, (Expression, Array)):
            return NotImplemented
        if type(other) is type(self):
            if self.value == other.value:
                return (self.variable_name or "") < (other.variable_name or "")
            return str(self.value) < str(other.value)
        return type(self).__name__ < type(other).__name__


class Scalar(Constant[ScalarType]):
    _mapper_method: [str] = "map_scalar"


class Dtype(Constant[_dtype_any]):
    _mapper_method: ClassVar[str] = "map_dtype"


class Id(Constant[str]):
    _mapper_method: ClassVar[str] = "map_id"


@np_op_dataclass
class TupleOp(Operation, Generic[TupleOpT]):
    _operands: Tuple[Expression]
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)

    arity: ClassVar[Arity] = Arity.variadic
    name: ClassVar[str] = "tuple"
    _mapper_method: ClassVar[str] = "map_tuple_op"
    unpacked_args_to_init: ClassVar[bool] = True

    @property
    def operands(self):
        return self._operands


@np_op_dataclass
class NumpyOp(abc.ABC, Operation):
    """
    A base class for all Numpy-like operations that can end up as a
    :class:`pytato.IndexLambda` after calling functions from :mod:`pytato`'s
    frontend.
    """
    pt_expr: Optional[Array] = field(metadata=_NOT_OPERAND_METADATA)
    unpacked_args_to_init: ClassVar[bool] = True

    @property
    def operands(self) -> Tuple[Expression]:
        return tuple(getattr(self, field.name)
                     for field in fields(self)
                     if not field.metadata.get("not_an_operand", False))

    def __lt__(self, other):
        if not isinstance(other, (Expression, Array)):
            return NotImplemented
        if type(other) is type(self):
            if self.operands == other.operands:
                return (self.variable_name or "") < (other.variable_name or "")
            return str(self.operands) < str(other.operands)
        return type(self).__name__ < type(other).__name__

    def __add__(self, other):
        return Sum(None, self, other)

    def __radd__(self, other):
        return Sum(None, other, self)

    def __mul__(self, other):
        return Product(None, self, other)

    def __rmul__(self, other):
        return Product(None, other, self)

    def __matmul__(self, other):
        return MatMul(None, self, other)


@np_op_dataclass
class _Input(NumpyOp):
    id: Id
    shape: TupleOp[Expression]
    dtype: Symbol
    arity: ClassVar[Arity] = Arity.unary
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)


@np_op_dataclass
class DataWrapper(_Input):
    name: ClassVar[str] = "DataWrapper"
    _mapper_method: ClassVar[str] = "map_data_wrapper"


@np_op_dataclass
class Placeholder(_Input):
    name: ClassVar[str] = "Placeholder"
    _mapper_method: ClassVar[str] = "map_placeholder"


@np_op_dataclass
class _BinaryOp(NumpyOp):
    x1: NumpyOpOrScalarT
    x2: NumpyOpOrScalarT
    arity: ClassVar[Arity] = Arity.binary
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)


@np_op_dataclass
class Sum(_BinaryOp):
    name: ClassVar[str] = "Sum"
    associative: ClassVar[bool] = True
    _mapper_method: ClassVar[str] = "map_sum"

    # TODO: [kk, discuss] -- Should we mark this op as commutative?


@np_op_dataclass
class Product(_BinaryOp):
    name: ClassVar[str] = "Product"
    associative: ClassVar[bool] = True
    _mapper_method: ClassVar[str] = "map_product"


@np_op_dataclass
class TrueDiv(_BinaryOp):
    name: ClassVar[str] = "TrueDiv"


@np_op_dataclass
class FloorDiv(_BinaryOp):
    name: ClassVar[str] = "FloorDiv"


@np_op_dataclass
class Modulo(_BinaryOp):
    name: ClassVar[str] = "Modulo"


@np_op_dataclass
class LogicalOr(_BinaryOp):
    name: ClassVar[str] = "LogicalOr"
    associative: ClassVar[bool] = True


@np_op_dataclass
class LogicalAnd(_BinaryOp):
    name: ClassVar[str] = "LogicalAnd"
    associative: ClassVar[bool] = True


@np_op_dataclass
class _UnaryOp(NumpyOp):
    x: NumpyOp
    arity: ClassVar[Arity] = Arity.unary
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)


@np_op_dataclass
class LogicalNot(NumpyOp):
    x: NumpyOp


@np_op_dataclass
class Negate(NumpyOp):
    x: NumpyOp


@np_op_dataclass
class SingleArgCall(NumpyOp):
    func: Symbol
    x: NumpyOp
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)
    name: ClassVar[str] = "SingleArgCall"
    arity: ClassVar[Arity] = Arity.binary


@np_op_dataclass
class TwoArgCall(NumpyOp):
    func: Symbol
    x1: NumpyOpOrScalarT
    x2: NumpyOpOrScalarT
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)
    name: ClassVar[str] = "TwoArgCall"
    arity: ClassVar[Arity] = Arity.ternary


@np_op_dataclass
class Stack(NumpyOp):
    arrays: TupleOp[NumpyOp]
    axis: Symbol
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)
    name: ClassVar[str] = "Stack"
    arity: ClassVar[Arity] = Arity.binary


@np_op_dataclass
class Concatenate(NumpyOp):
    arrays: TupleOp[NumpyOp]
    axis: Symbol
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)
    name: ClassVar[str] = "Concatenate"
    arity: ClassVar[Arity] = Arity.binary


@np_op_dataclass
class MatMul(NumpyOp):
    x1: NumpyOp
    x2: NumpyOp
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)
    name: ClassVar[str] = "MatMul"
    arity: ClassVar[Arity] = Arity.binary
    _mapper_method: ClassVar[str] = "map_matmul"


@np_op_dataclass
class Reshape(NumpyOp):
    x: NumpyOp
    newshape: TupleOp[Expression]
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)


@np_op_dataclass
class Roll(NumpyOp):
    x: NumpyOp
    shift: Symbol
    axis: Symbol
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)


@np_op_dataclass
class AxisPermutation(NumpyOp):
    x: NumpyOp
    axes: TupleOp[Symbol]
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)


@np_op_dataclass
class Slice:
    start: Expression
    stop: Expression
    step: Symbol
    name: ClassVar[str] = "Slice"
    arity: ClassVar[Arity] = Arity.ternary
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)


@np_op_dataclass
class Index(NumpyOp):
    x: NumpyOp
    indices: TupleOp[IndexExpr]


class Wildcard(BaseWildcard):
    def __add__(self, other):
        return Sum(None, self, other)

    def __radd__(self, other):
        return Sum(None, other, self)

    def __mul__(self, other):
        return Product(None, self, other)

    def __rmul__(self, other):
        return Product(None, other, self)

    def __matmul__(self, other):
        return MatMul(None, self, other)

    @classmethod
    def dot(cls, name=None) -> "Wildcard":
        # FIXME: This should go into matchpy itself.
        return cls(min_count=1, fixed_size=True, variable_name=name)


def match_anywhere(subject, pattern) -> Iterator[Mapping[str, ArrayOrScalar]]:
    from matchpy import match_anywhere, Pattern
    from .tofrom import ToMatchpyExpressionMapper, FromMatchpyExpressionMapper

    m_subject = ToMatchpyExpressionMapper()(subject)
    m_pattern = Pattern(pattern)

    # FIXME: Convert these matches into
    # 1. Wildcards -> Array entries
    # 2. From the path, extract the entire matched sub-expression

    return match_anywhere(m_subject, m_pattern)

    # from_matchpy_expr = FromMatchpyExpressionMapper()
    # for subst, _ in match_anywhere(matchpy_subject, Pattern(pattern)):
    #     yield {name: from_matchpy_expr(expr) for name, expr in subst.items()}
