from __future__ import annotations

__copyright__ = """Copyright (C) 2021 Kaushik Kulkarni"""

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


from typing import (Dict, Tuple, TypeVar, Iterable, Optional, Callable, Any,
                    cast)
from dataclasses import dataclass
from pytato.array import (Array, ArrayOrScalar, AxisPermutation, Concatenate,
                          DataWrapper, Einsum, IndexBase, IndexLambda,
                          MatrixProduct, Placeholder, Reshape, Roll, SizeParam,
                          Stack, ShapeComponent, NormalizedSlice, IndexExpr)
from pytato.normalization import (BinaryOp, BinaryOpType, FullOp, LogicalNotOp,
                                  ReduceOp, index_lambda_to_high_level_op,
                                  ComparisonOp, BroadcastOp)
from pytato.scalar_expr import (SCALAR_CLASSES, ScalarExpression,
                                StringifyMapper as StringifyMapperBase,
                                ExpressionBase)
from pytato.transform import Mapper
from pytato.utils import are_shape_components_equal

import pymbolic.primitives as p


T = TypeVar("T")


# {{{ helper routines

def first_true(iterable: Iterable[T], default: T,
               pred: Optional[Callable[[T], bool]] = None) -> T:
    """
    Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.
    """
    # Taken from <https://docs.python.org/3/library/itertools.html#itertools-recipes>
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


def _get_einsum_spec(einsum: Einsum) -> str:
    from pytato.array import (ElementwiseAxis, EinsumAxisDescriptor)

    idx_stream = (chr(i) for i in range(ord("i"), ord("z")))
    idx_gen: Callable[[], str] = lambda: next(idx_stream)  # noqa: E731
    axis_descr_to_idx: Dict[EinsumAxisDescriptor, str] = {}
    input_specs = []
    for access_descr in einsum.access_descriptors:
        spec = ""
        for axis_descr in access_descr:
            try:
                spec += axis_descr_to_idx[axis_descr]
            except KeyError:
                axis_descr_to_idx[axis_descr] = idx_gen()
                spec += axis_descr_to_idx[axis_descr]

        input_specs.append(spec)

    output_spec = "".join(axis_descr_to_idx[ElementwiseAxis(i)]
                          for i in range(einsum.ndim))

    return f"{', '.join(input_specs)} -> {output_spec}"

# }}}


@dataclass(frozen=True, eq=True)
class MatrixProductExpr(ExpressionBase):
    x1: ScalarExpression
    x2: ScalarExpression

    def __getinitargs__(self) -> Tuple[ScalarExpression, ScalarExpression]:
        return (self.x1, self.x2)

    mapper_method = "map_matrix_product"


# {{{ stringify mapper

class StringifyMapper(StringifyMapperBase):
    def map_matrix_product(self,
                           expr: MatrixProduct,
                           enclosing_prec: int,
                           *args: Any, **kwargs: Any) -> str:
        from pymbolic.mapper.stringifier import PREC_PRODUCT
        kwargs["force_parens_around"] = (p.Quotient, p.FloorDiv, p.Remainder)
        # type-ignore-reason: self.parenthesize_if_needed returns 'Any' (not 'str')
        return self.parenthesize_if_needed(   # type: ignore[no-any-return]
            " @ ".join([self.rec(expr.x1,
                                 PREC_PRODUCT,
                                 **kwargs),
                        self.rec(expr.x2,
                                 PREC_PRODUCT,
                                 **kwargs)]),
            enclosing_prec, PREC_PRODUCT)

# }}}


def _is_slice_trivial(slice_: NormalizedSlice,
                      dim: ShapeComponent) -> bool:
    """
    Return *True* only if *slice_* is equivalent to the trivial slice i.e.
    traverses an axis of length *dim* in unit steps.
    """
    return (are_shape_components_equal(slice_.start, 0)
            and are_shape_components_equal(slice_.stop, dim)
            and slice_.step == 1)


class ToPymbolicExpression(Mapper):
    def __init__(self) -> None:
        super().__init__()
        self.cache: Dict[ArrayOrScalar, ScalarExpression] = {}

    # type-ignore reason: return type not compatible with Mapper.rec's type
    def rec(self, expr: ArrayOrScalar) -> ScalarExpression:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        # type-ignore reason: type not compatible with super.rec() type
        result: ScalarExpression = super().rec(expr)  # type: ignore
        self.cache[expr] = result
        return result

    def map_foreign(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(expr, SCALAR_CLASSES):
            return expr

        return super().map_foreign(expr, *args, **kwargs)

    def map_size_param(self, expr: SizeParam) -> ScalarExpression:
        return p.Variable(expr.name)

    def map_data_wrapper(self, expr: DataWrapper) -> ScalarExpression:
        if isinstance(expr.name, str):
            return p.Variable(expr.name)
        else:
            assert expr.name is None
            return p.Call(p.Variable("DataWrapper"),
                          (p.Variable(f"<{type(expr.data)} at {id(expr.data)}>"),))

    def map_placeholder(self, expr: Placeholder) -> ScalarExpression:
        assert isinstance(expr.name, str)
        return p.Variable(expr.name)

    def map_index_lambda(self, expr: IndexLambda) -> ScalarExpression:
        hlo = index_lambda_to_high_level_op(expr)
        if isinstance(hlo, BinaryOp):
            if hlo.binary_op in (BinaryOpType.TRUEDIV,
                                 BinaryOpType.FLOORDIV,
                                 BinaryOpType.MOD):
                return hlo.binary_op.value(self.rec(hlo.x1),
                                           self.rec(hlo.x2))
            else:
                return hlo.binary_op.value(((self.rec(hlo.x1),
                                             self.rec(hlo.x2))))
        elif isinstance(hlo, ReduceOp):
            if hlo.axes == tuple(i for i in range(hlo.x.ndim)):
                return p.Call(p.Variable(hlo.op), (self.rec(hlo.x),))
            elif len(hlo.axes) == 1:
                return p.CallWithKwargs(p.Variable(hlo.op),
                                        (self.rec(hlo.x),),
                                        {"axis": hlo.axes[0]})
            else:
                return p.CallWithKwargs(p.Variable(hlo.op),
                                        (self.rec(hlo.x),),
                                        {"axis": hlo.axes})
        elif isinstance(hlo, FullOp):
            shape_rec = tuple(self.rec(dim) for dim in expr.shape)
            if hlo.fill_value == 0:
                return p.Call("zeros", (shape_rec,))
            elif hlo.fill_value == 1:
                return p.Call("ones", (shape_rec,))
            else:
                return p.Call("full", (hlo.fill_value, shape_rec,))
        elif isinstance(hlo, LogicalNotOp):
            return p.Call("logical_not", (self.rec(hlo.x),))
        elif isinstance(hlo, ComparisonOp):
            return p.Comparison(self.rec(hlo.left),
                                hlo.operator,
                                self.rec(hlo.right))
        elif isinstance(hlo, BroadcastOp):
            return p.Call(p.Variable("broadcast_to"),
                          ((self.rec(hlo.x),)
                           + (tuple(self.rec(dim) for dim in expr.shape),))
                          )
        else:
            kwargs = {name: self.rec(bnd)
                      for name, bnd in expr.bindings.items()}
            kwargs["shape"] = tuple(self.rec(dim) for dim in expr.shape)
            return p.CallWithKwargs("IndexLambda",
                                    parameters=(),
                                    kw_parameters=kwargs)

    def map_matrix_product(self, expr: MatrixProduct) -> ScalarExpression:
        return MatrixProductExpr(self.rec(expr.x1), self.rec(expr.x2))

    def map_stack(self, expr: Stack) -> ScalarExpression:
        if expr.axis == 0:
            kwargs = {}
        else:
            kwargs = {"axis": expr.axis}

        return p.CallWithKwargs(p.Variable("stack"),
                                ([self.rec(ary) for ary in expr.arrays],),
                                kw_parameters=kwargs)

    def map_concatenate(self, expr: Concatenate) -> ScalarExpression:
        if expr.axis == 0:
            kwargs = {}
        else:
            kwargs = {"axis": expr.axis}

        return p.CallWithKwargs(p.Variable("concatenate"),
                                ([self.rec(ary) for ary in expr.arrays],),
                                kw_parameters=kwargs)

    def map_roll(self, expr: Roll) -> ScalarExpression:
        if expr.array.ndim == 1:
            kwargs = {}
        else:
            kwargs = {"axis": expr.axis}

        return p.CallWithKwargs(p.Variable("roll"),
                                (self.rec(expr.array), expr.shift),
                                kw_parameters=kwargs)

    def map_axis_permutation(self, expr: AxisPermutation
                             ) -> ScalarExpression:
        if expr.axes == tuple(range(expr.array.ndim)[::-1]):
            kwargs = {}
        else:
            kwargs = {"axes": expr.axes}

        return p.CallWithKwargs(p.Variable("transpose"),
                                (self.rec(expr.array),),
                                kw_parameters=kwargs)

    def _map_index_base(self, expr: IndexBase) -> ScalarExpression:
        last_non_trivial_index = first_true(
            range(expr.array.ndim)[::-1],
            default=-1,
            pred=lambda i: not (isinstance(expr.indices[i], NormalizedSlice)
                                and _is_slice_trivial(
                                        cast(NormalizedSlice, expr.indices[i]),
                                        expr.array.shape[i]))
        )
        if last_non_trivial_index == -1:
            return self.rec(expr.array)

        def _rec_idx(idx: IndexExpr, dim: ShapeComponent) -> ScalarExpression:
            if isinstance(idx, int):
                return idx
            elif isinstance(idx, NormalizedSlice):
                step = idx.step if idx.step != 1 else None
                if idx.step > 0:
                    start = (None
                             if are_shape_components_equal(0,
                                                           idx.start)
                             else self.rec(idx.start))

                    stop = (None
                            if are_shape_components_equal(dim, idx.stop)
                            else self.rec(idx.stop))
                else:
                    start = (None
                             if are_shape_components_equal(dim-1, idx.start)
                             else self.rec(idx.start))

                    stop = (None
                            if are_shape_components_equal(-1, idx.stop)
                            else self.rec(idx.stop))

                return p.Slice((start, stop, step))
            else:
                assert isinstance(idx, Array)
                return self.rec(idx)

        return p.Subscript(self.rec(expr.array),
                           tuple(_rec_idx(idx, dim)
                                 for idx, dim in zip(
                                         expr.indices[:last_non_trivial_index+1],
                                         expr.array.shape)))

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_reshape(self, expr: Reshape) -> ScalarExpression:
        return p.Call(p.Variable("reshape"),
                      ((self.rec(expr.array),)
                       + ((tuple(self.rec(dim) for dim in expr.shape)
                           if expr.ndim > 1
                           else self.rec(expr.shape[0])),))
                      )

    def map_einsum(self, expr: Einsum) -> ScalarExpression:
        einsum_spec = _get_einsum_spec(expr)
        return p.Call(p.Variable("einsum"),
                      ((p.Variable(f'"{einsum_spec}"'),)
                       + tuple(self.rec(arg) for arg in expr.args)))


def stringify(expr: Array) -> str:
    pym_expr = ToPymbolicExpression()(expr)
    return StringifyMapper()(pym_expr)  # type: ignore[no-any-return]
