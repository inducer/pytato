from __future__ import annotations


__copyright__ = """
Copyright (C) 2022 Kaushik Kulkarni
"""

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

import ast
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import (
    TypedDict,
    TypeVar,
    cast,
)

import numpy as np
from immutabledict import immutabledict
from typing_extensions import NotRequired

from pytools import UniqueNameGenerator

from pytato.array import (
    Array,
    ArrayOrScalar,
    AxisPermutation,
    Concatenate,
    DataInterface,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexExpr,
    IndexLambda,
    NamedArray,
    NormalizedSlice,
    Placeholder,
    Reshape,
    Roll,
    ShapeComponent,
    SizeParam,
    Stack,
)
from pytato.raising import BinaryOpType, C99CallOp
from pytato.reductions import (
    AllReductionOperation,
    AnyReductionOperation,
    MaxReductionOperation,
    MinReductionOperation,
    ProductReductionOperation,
    ReductionOperation,
    SumReductionOperation,
)
from pytato.target.python import BoundPythonProgram, NumpyLikePythonTarget
from pytato.transform import CachedMapper
from pytato.utils import are_shape_components_equal, get_einsum_specification


T = TypeVar("T")


def _can_colorize_output() -> bool:
    try:
        import pygments  # noqa: F401
        return True
    except ImportError:
        return False


def _get_default_colorize_code() -> bool:
    return ((not sys.stdout.isatty())
            # https://no-color.org/
            and "NO_COLOR" not in os.environ)


MISMATCHED_C99_CALL_TO_NP_FUNC = {
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "atan2": "arctan2"}


def _c99_callop_numpy_name(hlo: C99CallOp) -> str:
    if hlo.function in MISMATCHED_C99_CALL_TO_NP_FUNC:
        return MISMATCHED_C99_CALL_TO_NP_FUNC[hlo.function]
    else:
        return hlo.function


def first_true(iterable: Iterable[T], default: T,
               pred: Callable[[T], bool] | None = None) -> T:
    """
    Returns the first true value in *iterable*. If no true value is found,
    returns *default* If *pred* is not None, returns the first item for which
    pred(item) is true.
    """
    # Taken from <https://docs.python.org/3/library/itertools.html#itertools-recipes>
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


def _is_slice_trivial(slice_: NormalizedSlice,
                      dim: ShapeComponent) -> bool:
    """
    Return *True* only if *slice_* is equivalent to the trivial slice i.e.
    traverses an axis of length *dim* in unit steps.
    """
    return (are_shape_components_equal(slice_.start, 0)
            and are_shape_components_equal(slice_.stop, dim)
            and slice_.step == 1)


SIMPLE_BINOP_TO_AST_OP = {BinaryOpType.ADD:         ast.Add,
                          BinaryOpType.SUB:         ast.Sub,
                          BinaryOpType.MULT:        ast.Mult,
                          BinaryOpType.TRUEDIV:     ast.Div,
                          BinaryOpType.FLOORDIV:    ast.FloorDiv,
                          BinaryOpType.MOD:         ast.Mod,
                          BinaryOpType.POWER:       ast.Pow,
                          BinaryOpType.BITWISE_OR:  ast.BitOr,
                          BinaryOpType.BITWISE_XOR: ast.BitXor,
                          BinaryOpType.BITWISE_AND: ast.BitAnd,
                          }

COMPARISON_OP_TO_CALL = {BinaryOpType.LESS:    "less",
                         BinaryOpType.GREATER: "greater",
                         BinaryOpType.LESS_EQUAL: "less_equal",
                         BinaryOpType.GREATER_EQUAL: "greater_equal",
                         BinaryOpType.EQUAL: "equal",
                         BinaryOpType.NOT_EQUAL: "not_equal",
                         }

LOGICAL_OP_TO_CALL = {BinaryOpType.LOGICAL_OR:  "logical_or",
                      BinaryOpType.LOGICAL_AND: "logical_and",
                      }

PYTATO_REDUCTION_TO_NP_REDUCTION: Mapping[type[ReductionOperation], str] = {
    SumReductionOperation: "sum",
    ProductReductionOperation: "product",
    MaxReductionOperation: "max",
    MinReductionOperation: "min",
    AllReductionOperation: "all",
    AnyReductionOperation: "any",
}


class NumpyCodegenMapper(CachedMapper[str, []]):
    """
    .. note::

        - This mapper stores mutable state for building the program. The same
          mapper instance must be re-used with care.
    """
    def __init__(self, numpy: str, numpy_backend: str, vng: Callable[[str], str]):
        super().__init__()
        self.numpy = numpy
        self.numpy_backend = numpy_backend
        self.vng = vng

        self.lines: list[ast.stmt] = []
        self.arg_names: set[str] = set()
        self.bound_arguments: dict[str, DataInterface] = {}

    def _record_line_and_return_lhs(self, lhs: str, rhs: ast.expr) -> str:
        self.lines.append(ast.Assign(targets=[ast.Name(lhs)],
                                     value=rhs))
        return lhs

    def map_index_lambda(self, expr: IndexLambda) -> str:
        from pytato.raising import (
            BinaryOp,
            BinaryOpType,
            BroadcastOp,
            FullOp,
            ReduceOp,
            WhereOp,
            index_lambda_to_high_level_op,
        )
        hlo = index_lambda_to_high_level_op(expr)
        lhs = self.vng("_pt_tmp")
        rhs: ast.expr

        def _rec_ary_or_constant(e: ArrayOrScalar) -> ast.expr:
            if isinstance(e, Array):
                return ast.Name(self.rec(e))
            else:
                if np.isnan(e):
                    e_np = np.array(e)
                    # generates code like: `np.float64("nan")`.
                    return ast.Call(
                        func=ast.Attribute(value=ast.Name(self.numpy),
                                           attr=e_np.dtype.name),
                        args=[ast.Constant(value="nan")],
                        keywords=[])
                else:
                    return ast.Constant(e)

        if isinstance(hlo, FullOp):
            if hlo.fill_value == 1:
                if expr.dtype == np.dtype(float):
                    rhs = ast.Call(
                        ast.Attribute(ast.Name(self.numpy_backend),
                                      "ones"),
                        args=[ast.Tuple(elts=[ast.Constant(d)
                                              for d in expr.shape])],
                        keywords=[])
                else:
                    rhs = ast.Call(
                        ast.Attribute(ast.Name(self.numpy_backend),
                                      "ones"),
                        args=[ast.Tuple(elts=[ast.Constant(d)
                                              for d in expr.shape])],
                        keywords=[ast.keyword(
                            arg="dtype",
                            value=ast.Attribute(ast.Name(self.numpy),
                                                f"{expr.dtype.type.__name__}"),
                        )])
            elif hlo.fill_value == 0:
                if expr.dtype == np.dtype(float):
                    rhs = ast.Call(
                        ast.Attribute(ast.Name(self.numpy_backend),
                                      "zeros"),
                        args=[ast.Tuple(elts=[ast.Constant(d)
                                              for d in expr.shape])],
                        keywords=[])
                else:
                    rhs = ast.Call(
                        ast.Attribute(ast.Name(self.numpy_backend),
                                      "zeros"),
                        args=[ast.Tuple(elts=[ast.Constant(d)
                                              for d in expr.shape])],
                        keywords=[ast.keyword(
                            arg="dtype",
                            value=ast.Attribute(ast.Name(self.numpy),
                                                f"{expr.dtype.type.__name__}"),
                        )])
            else:
                rhs = ast.Call(
                    ast.Attribute(ast.Name(self.numpy_backend),
                                  "full"),
                    args=[ast.Tuple(elts=[ast.Constant(d)
                                          for d in expr.shape]),
                          _rec_ary_or_constant(hlo.fill_value),
                          ],
                    keywords=[ast.keyword(
                        arg="dtype",
                        value=ast.Attribute(ast.Name(self.numpy),
                                            f"{expr.dtype.type.__name__}"),
                    )])
        elif isinstance(hlo, BinaryOp):
            if hlo.binary_op in {BinaryOpType.ADD, BinaryOpType.SUB,
                                 BinaryOpType.MULT, BinaryOpType.POWER,
                                 BinaryOpType.TRUEDIV, BinaryOpType.FLOORDIV,
                                 BinaryOpType.MOD, BinaryOpType.BITWISE_OR,
                                 BinaryOpType.BITWISE_XOR,
                                 BinaryOpType.BITWISE_AND,
                                 }:
                rhs = ast.BinOp(left=_rec_ary_or_constant(hlo.x1),
                                op=SIMPLE_BINOP_TO_AST_OP[hlo.binary_op](),
                                right=_rec_ary_or_constant(hlo.x2))
            elif hlo.binary_op in {BinaryOpType.EQUAL, BinaryOpType.NOT_EQUAL,
                                   BinaryOpType.LESS, BinaryOpType.LESS_EQUAL,
                                   BinaryOpType.GREATER,
                                   BinaryOpType.GREATER_EQUAL}:
                rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend),
                                             COMPARISON_OP_TO_CALL[hlo.binary_op]),
                               args=[_rec_ary_or_constant(hlo.x1),
                                     _rec_ary_or_constant(hlo.x2)],
                               keywords=[])
            elif hlo.binary_op in {BinaryOpType.LOGICAL_OR,
                                   BinaryOpType.LOGICAL_AND}:
                rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend),
                                             LOGICAL_OP_TO_CALL[hlo.binary_op]),
                               args=[_rec_ary_or_constant(hlo.x1),
                                     _rec_ary_or_constant(hlo.x2)],
                               keywords=[])
            else:
                raise NotImplementedError(hlo.binary_op)
        elif isinstance(hlo, C99CallOp):
            rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend),
                                         _c99_callop_numpy_name(hlo)),
                           args=[_rec_ary_or_constant(arg)
                                 for arg in hlo.args],
                           keywords=[])
        elif isinstance(hlo, WhereOp):
            rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend),
                                         "where"),
                           args=[_rec_ary_or_constant(hlo.condition),
                                 _rec_ary_or_constant(hlo.then),
                                 _rec_ary_or_constant(hlo.else_)],
                           keywords=[])
        elif isinstance(hlo, BroadcastOp):
            if not all(isinstance(d, int) for d in expr.shape):
                raise NotImplementedError("Parametric shape in broadcast_to")

            rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend),
                                         "broadcast_to"),
                           args=[ast.Name(self.rec(hlo.x)),
                                 ast.Tuple(elts=[ast.Constant(d)
                                                 for d in expr.shape])],
                           keywords=[])
        elif isinstance(hlo, ReduceOp):
            if type(hlo.op) not in PYTATO_REDUCTION_TO_NP_REDUCTION:
                raise NotImplementedError(hlo.op)
            np_fn_name = PYTATO_REDUCTION_TO_NP_REDUCTION[type(hlo.op)]
            if all(i in hlo.axes for i in range(hlo.x.ndim)):
                rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend),
                                             np_fn_name),
                               args=[ast.Name(self.rec(hlo.x))],
                               keywords=[])
            else:
                if len(hlo.axes) == 1:
                    axis, = hlo.axes.keys()
                    axis_ast: ast.expr = ast.Constant(axis)
                else:
                    axis_ast = ast.Tuple(elts=[ast.Constant(e)
                                               for e in sorted(hlo.axes.keys())])
                rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend),
                                             np_fn_name),
                               args=[ast.Name(self.rec(hlo.x))],
                               keywords=[ast.keyword(arg="axis",
                                                     value=axis_ast)])
        else:
            raise NotImplementedError(type(hlo))

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_placeholder(self, expr: Placeholder) -> str:
        self.arg_names.add(expr.name)
        return expr.name

    def map_stack(self, expr: Stack) -> str:
        assert isinstance(expr.axis, int)

        rec_ids = [self.rec(ary) for ary in expr.arrays]
        lhs = self.vng("_pt_tmp")
        rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend), "stack"),
                       args=[ast.List([ast.Name(id_)
                                       for id_ in rec_ids])],
                       keywords=[ast.keyword(arg="axis",
                                             value=ast.Constant(expr.axis))])

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_concatenate(self, expr: Concatenate) -> str:
        assert isinstance(expr.axis, int)

        rec_ids = [self.rec(ary) for ary in expr.arrays]
        lhs = self.vng("_pt_tmp")
        rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend), "concatenate"),
                       args=[ast.List([ast.Name(id_)
                                       for id_ in rec_ids])],
                       keywords=[ast.keyword(arg="axis",
                                             value=ast.Constant(expr.axis))])

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_roll(self, expr: Roll) -> str:
        lhs = self.vng("_pt_tmp")
        rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend), "roll"),
                       args=[ast.Name(self.rec(expr.array)),
                             ],
                       keywords=[ast.keyword(arg="shift",
                                             value=ast.Constant(expr.shift)),
                                 ast.keyword(arg="axis",
                                             value=ast.Constant(expr.axis))])

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_axis_permutation(self, expr: AxisPermutation) -> str:
        lhs = self.vng("_pt_tmp")
        if expr.axis_permutation == tuple(range(expr.ndim))[::-1]:
            rhs: ast.expr = ast.Attribute(ast.Name(self.rec(expr.array)), "T")
        else:
            rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend), "transpose"),
                           args=[ast.Name(self.rec(expr.array))],
                           keywords=[ast.keyword(
                               arg="axes",
                               value=ast.List(elts=[ast.Constant(a)
                                                    for a in expr.axis_permutation]))
                                     ])

        return self._record_line_and_return_lhs(lhs, rhs)

    def _map_index_base(self, expr: IndexBase) -> str:
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

        lhs = self.vng("_pt_tmp")

        def _rec_idx(idx: IndexExpr, dim: ShapeComponent) -> ast.expr:
            if isinstance(idx, int):
                return ast.Constant(idx)
            elif isinstance(idx, NormalizedSlice):
                step = idx.step if idx.step != 1 else None
                if idx.step > 0:
                    start = (None
                             if are_shape_components_equal(0,
                                                           idx.start)
                             else idx.start)

                    stop = (None
                            if are_shape_components_equal(dim, idx.stop)
                            else idx.stop)
                else:
                    start = (None
                             if are_shape_components_equal(dim-1, idx.start)
                             else idx.start)

                    stop = (None
                            if are_shape_components_equal(-1, idx.stop)
                            else idx.stop)

                from ast import expr as expr_t

                class SliceKwargs(TypedDict):
                    step: NotRequired[expr_t]
                    lower: NotRequired[expr_t]
                    upper: NotRequired[expr_t]

                kwargs: SliceKwargs = {}
                if step is not None:
                    assert isinstance(step, int)
                    kwargs["step"] = ast.Constant(step)
                if start is not None:
                    assert isinstance(start, int)
                    kwargs["lower"] = ast.Constant(start)
                if stop is not None:
                    assert isinstance(stop, int)
                    kwargs["upper"] = ast.Constant(stop)

                return ast.Slice(**kwargs)
            else:
                assert isinstance(idx, Array)
                return ast.Name(self.rec(idx))

        rhs = ast.Subscript(value=ast.Name(self.rec(expr.array)),
                            slice=ast.Tuple(
                                elts=[
                                    _rec_idx(idx, dim)
                                    for idx, dim in zip(
                                            expr.indices[:last_non_trivial_index+1],
                                            expr.array.shape,
                                            strict=False)]))

        return self._record_line_and_return_lhs(lhs, rhs)

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_data_wrapper(self, expr: DataWrapper) -> str:
        name = self.vng("_pt_data") if expr.name is None else expr.name
        self.arg_names.add(name)
        self.bound_arguments[name] = expr.data
        return name

    def map_size_param(self, expr: SizeParam) -> str:
        # would demand a more complicated BoundProgram implementation.
        raise NotImplementedError("SizeParams not yet supported  in numpy-targets.")

    def map_einsum(self, expr: Einsum) -> str:
        lhs = self.vng("_pt_tmp")
        args = [ast.Name(self.rec(arg)) for arg in expr.args]
        rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend), "einsum"),
                        args=[ast.Constant(get_einsum_specification(expr)),
                              *args],
                       keywords=[],
                       )

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_reshape(self, expr: Reshape) -> str:
        lhs = self.vng("_pt_tmp")
        if not all(isinstance(d, int) for d in expr.shape):
            raise NotImplementedError("Non-integral reshapes.")
        rhs = ast.Call(ast.Attribute(ast.Name(self.numpy_backend), "reshape"),
                        args=[ast.Name(self.rec(expr.array)),
                              ast.Tuple(elts=[ast.Constant(d)
                                              for d in expr.shape])],
                       keywords=[],
                       )

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_named_array(self, expr: NamedArray) -> str:
        return self.rec(expr.expr)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> str:
        lhs = self.vng("_pt_tmp")

        from ast import expr as expr_t
        keys: list[expr_t | None] = []
        values: list[expr_t] = []
        for name, subexpr in sorted(expr._data.items()):
            keys.append(ast.Constant(name))
            values.append(ast.Name(self.rec(subexpr)))

        rhs = ast.Dict(keys=keys, values=values)

        return self._record_line_and_return_lhs(lhs, rhs)


def generate_numpy_like(expr: Array | Mapping[str, Array] | DictOfNamedArrays,
                        target: NumpyLikePythonTarget,
                        function_name: str,
                        show_code: bool,
                        entrypoint_decorators: tuple[str, ...],
                        extra_preambles: tuple[ast.stmt, ...],
                        colorize_show_code: bool | None = None,
                        ) -> BoundPythonProgram:
    import collections

    from pytato.transform import InputGatherer

    if ((not isinstance(expr, DictOfNamedArrays))
            and isinstance(expr, collections.abc.Mapping)):
        from pytato.array import make_dict_of_named_arrays
        expr = make_dict_of_named_arrays(dict(expr))

    assert isinstance(expr, Array | DictOfNamedArrays)

    var_name_gen = UniqueNameGenerator()

    var_name_gen.add_names({input_expr.name
                            for input_expr in InputGatherer()(expr)
                            if isinstance(input_expr,
                                          Placeholder | SizeParam | DataWrapper)
                            if input_expr.name is not None})
    if isinstance(expr, DictOfNamedArrays):
        var_name_gen.add_names(expr)

    var_name_gen.add_names({target.numpy_like_module_name_shorthand,
                            "np",
                            function_name})

    cgen_mapper = NumpyCodegenMapper(
        numpy_backend=target.numpy_like_module_name_shorthand,
        numpy="np",
        vng=var_name_gen)
    result_var = cgen_mapper(expr)

    lines = cgen_mapper.lines
    lines.append(ast.Return(ast.Name(result_var)))

    from ast import expr as expr_t
    decorator_list: list[expr_t] = [ast.Name(dec) for dec in entrypoint_decorators]

    module = ast.Module(
        body=[ast.Import(names=[ast.alias(name=target.numpy_like_module_name,
                                          asname=(
                                              target
                                              .numpy_like_module_name_shorthand
                                          ))]),
              ast.Import(names=[ast.alias(name="numpy", asname="np")]),
              *extra_preambles,
              ast.FunctionDef(
                  name=function_name,
                  args=ast.arguments(
                      args=[],
                      posonlyargs=[],
                      kwonlyargs=[ast.arg(arg=name)
                                  for name in cgen_mapper.arg_names],
                      kw_defaults=[None for _ in cgen_mapper.arg_names],
                      defaults=[]),
                  body=lines,
                  decorator_list=decorator_list)
              ],
        type_ignores=[])

    program = ast.unparse(ast.fix_missing_locations(module))

    if show_code:
        if colorize_show_code is None:
            colorize_show_code = _get_default_colorize_code()
        assert isinstance(colorize_show_code, bool)

        if _can_colorize_output() and colorize_show_code:
            from pygments import highlight
            from pygments.formatters import TerminalTrueColorFormatter
            from pygments.lexers import PythonLexer
            print(highlight(program,
                            formatter=TerminalTrueColorFormatter(),
                            lexer=PythonLexer()))
        else:
            print(program)

    return target.bind_program(
        program,
        function_name,
        expected_arguments=frozenset(cgen_mapper.arg_names),
        bound_arguments=immutabledict(cgen_mapper.bound_arguments))
