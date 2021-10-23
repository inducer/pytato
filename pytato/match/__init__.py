"""Pattern matching support for Pytato DAGs using MatchPy."""
from __future__ import annotations, absolute_import


import abc

from typing import (Union, ClassVar, Optional, Iterator, Mapping,
                    Generic, TypeVar, Sequence)
from dataclasses import dataclass, fields, field

from pytato.scalar_expr import ScalarType
from pytato.array import (ArrayOrScalar,  _dtype_any,
                          DataWrapper as PtDataWrapper)

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
        if not isinstance(other, Expression):
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


class TupleOp(Operation, Generic[TupleOpT]):
    arity = Arity.variadict
    name = "tuple"
    _mapper_method = "map_tuple_op"

    def __init__(self, operands: Sequence[TupleOpT],
                 variable_name=None) -> None:
        super().__init__(tuple(operands), variable_name)


class NumpyOp(abc.ABC, Operation):
    """
    A base class for all Numpy-like operations that can end up as a
    :class:`pytato.IndexLambda` after calling functions from :mod:`pytato`'s
    frontend.
    """
    unpacked_args_to_init: ClassVar[bool] = True

    @property
    def operands(self):
        return tuple(getattr(self, field.name)
                     for field in fields(self)
                     if not field.metadata.get("not_an_operand", False))

    def __add__(self, other):
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __mul__(self, other):
        return Product(self, other)

    def __rmul__(self, other):
        return Product(other, self)

    def __matmul__(self, other):
        return MatMul(self, other)


@np_op_dataclass
class _Input(NumpyOp):
    id: Id
    shape: TupleOp[Expression]
    dtype: Symbol
    arity: ClassVar[Arity] = Arity.unary


@np_op_dataclass
class DataWrapper(_Input):
    name: ClassVar[str] = "DataWrapper"
    _pt_counterpart: PtDataWrapper = field(metadata=_NOT_OPERAND_METADATA)
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)
    _mapper_method: ClassVar[str] = "map_data_wrapper"


@np_op_dataclass
class Placeholder(_Input):
    variable_name: Optional[str] = field(default=None,
                                         metadata=_NOT_OPERAND_METADATA)
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
    commutative: ClassVar[bool] = True
    associative: ClassVar[bool] = True
    _mapper_method: ClassVar[str] = "map_sum"


@np_op_dataclass
class Product(_BinaryOp):
    name: ClassVar[str] = "Product"
    commutative: ClassVar[bool] = True
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
    commutative: ClassVar[bool] = True
    associative: ClassVar[bool] = True


@np_op_dataclass
class LogicalAnd(_BinaryOp):
    name: ClassVar[str] = "LogicalAnd"
    commutative: ClassVar[bool] = True
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
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __mul__(self, other):
        return Product(self, other)

    def __rmul__(self, other):
        return Product(other, self)

    def __matmul__(self, other):
        return MatMul(self, other)

    @classmethod
    def dot(cls, name=None) -> "Wildcard":
        return cls(min_count=1, fixed_size=True, variable_name=name)


def match_anywhere(subject, pattern) -> Iterator[Mapping[str, ArrayOrScalar]]:
    from matchpy import match_anywhere, Pattern
    from .tofrom import ToMatchpyExpressionMapper, FromMatchpyExpressionMapper

    matchpy_subject = ToMatchpyExpressionMapper()(subject)
    from_matchpy_expr = FromMatchpyExpressionMapper()

    for subst, _ in match_anywhere(matchpy_subject, Pattern(pattern)):
        yield {name: from_matchpy_expr(expr) for name, expr in subst.items()}
