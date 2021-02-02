from __future__ import annotations

import numbers
import numpy as np
import mlir.astnodes as ast
import islpy as isl
from mlir.builder import IRBuilder
from typing import Union, Mapping, Any, Dict, Tuple, List, FrozenSet, Set
from pytato.array import (Array, DictOfNamedArrays, Namespace,
        SizeParam, Placeholder, IndexLambda, DataInterface, InputArgumentBase)
from dataclasses import dataclass
import pytato.scalar_expr as scalar_expr
from pytato.codegen import normalize_outputs, preprocess
from pytato.transform import Mapper
import pymbolic.primitives as prim
from pymbolic.mapper.substitutor import make_subst_func
from functools import reduce, partialmethod
from pytato.program import BoundProgram
from loopy.symbolic import isl_set_from_expr
from pytools import memoize_method
from pyrsistent import pmap


# {{{ FIXME: remove after github.com/inducer/pytato/pull/20 is merged

class NamedArray(Array):
    def __init__(self, name, shape, dtype):
        super().__init__(frozenset())
        self.name = name
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

# }}}


@dataclass
class DimOperation(prim.Expression):
    array_name: str
    axis: int

    def __str__(self):
        return f"dim({self.array_name}, {self.axis})"

    def __hash__(self):
        return hash((self.array_name, self.axis))

    mapper_method = "map_dim_op"


@dataclass
class CodeGenState:
    builder: IRBuilder
    file: ast.MLIRFile
    module: ast.Module
    function: ast.Function
    namespace: Namespace
    size_param_array_dim_bset: isl.BasicSet
    expr_to_pymbolic_name: Mapping[Array, str]
    pymbolic_name_to_ssa: Mapping[str, ast.SsaId]
    arguments: List[str]
    stored_names: Set[str]

    def pymbolic_name_to_expr(self, name: str) -> Array:
        try:
            return next(iter(key for key, val in self.expr_to_pymbolic_name.items()
                             if val == name))
        except StopIteration:
            raise ValueError(f"Unknown pymbolic name '{name}'")


def get_initial_codegen_state(outputs: DictOfNamedArrays) -> CodeGenState:
    namespace = outputs.namespace
    builder = IRBuilder()
    mlir_file = builder.make_mlir_file()
    module = mlir_file.module
    with builder.goto_block(builder.make_block(module.region)):
        fn = builder.function("_pt_kernel")

    bset = get_size_param_array_dim_basic_set(outputs)
    return CodeGenState(builder, mlir_file, module, fn, namespace, bset, {}, {}, [],
            set())


@dataclass
class BoundMLIRProgram(BoundProgram):
    options: List[str]
    # arguments: arrays ordered by the position in the function
    arguments: List[Union[Placeholder, NamedArray]]
    size_param_array_dim_bset: isl.BasicSet

    @property
    @memoize_method
    def arg_ids(self) -> List[str]:
        return [arg.name for arg in self.arguments]

    @memoize_method
    def infer_shapes(self,
            shapes: Dict[str, Union[Tuple[int, ...]]]) -> Dict[str, Tuple[int, ...]]:

        # {{{ intersect with the known shapes set with user-provided shapes

        shapes_bset = self.size_param_array_dim_bset

        for arg_name, shape in shapes.items():
            if arg_name in self.arg_ids:
                for iaxis, dim_len in enumerate(shape):
                    shapes_bset &= isl_set_from_expr(shapes_bset.get_space(),
                            prim.Comparison(prim.Variable(str(DimOperation(arg_name,
                                                                           iaxis))),
                                            "==",
                                            dim_len))
            else:
                # proabably a size param
                if arg_name not in self.size_param_array_dim_bset.get_var_dict():
                    raise ValueError(f"Got unexpected argument {arg_name}.")

                if not isinstance(shape, numbers.Integral):
                    raise TypeError(f"Size param '{arg_name}' expected to be an int")

                shapes_bset &= isl_set_from_expr(shapes_bset.get_space(),
                        prim.Comparison(prim.Variable(arg_name),
                                        "==",
                                        shape))

        # }}}

        # {{{ validity checks

        if shapes_bset.is_empty():
            raise ValueError(f"Provided shapes: '{shapes}' has no intersection with"
                    " the system of shape equations: "
                    f"'{self.size_param_array_dim_bset}'")

        if shapes_bset.count_val().to_python() != 1:
            raise ValueError(f"Provided shapes: '{shapes}' has on intersecting with "
                    "the system of shape equations: "
                    f"'{self.size_param_array_dim_bset}' did not produce "
                    "a unique solution.")

        # }}}

        shapes_bset, = shapes_bset.get_basic_sets()

        # {{{ solve for the dim lengths

        result = {}

        dom_var_dict = shapes_bset.get_var_dict()

        for arg in self.arguments:
            inferred_shape = []
            for iaxis in range(arg.ndim):
                _, pos = dom_var_dict[str(DimOperation(arg_name, iaxis))]
                inferred_shape.append(shapes_bset.dim_max_val(pos).to_python())

            result[arg.name] = tuple(inferred_shape)

        # }}}

        return result

    @memoize_method
    def arg_id_to_dtype(self, name: str) -> np.dtype:
        return next(iter(arg.dtype for arg in self.arguments if arg.name == name))

    def __call__(self, **kwargs: Union[DataInterface, numbers.Integral]) -> Any:
        args_to_mlir_knl = []
        return_dict = {}

        if set(kwargs.keys()) & set(self.bound_arguments.keys()):
            raise ValueError("Got arguments that were previously bound: "
                    f"{set(kwargs.keys()) & set(self.bound_arguments.keys())}.")

        updated_kwargs = self.bound_arguments.copy()
        updated_kwargs.update(kwargs)

        input_shapes = pmap({kw: arg.shape if isinstance(arg, DataInterface) else arg
                for kw, arg in updated_kwargs.items()})

        output_shapes = self.infer_shapes(input_shapes)

        for pt_ary in self.arguments:
            if isinstance(pt_ary, Placeholder):
                if pt_ary.name not in updated_kwargs:
                    raise ValueError(f"Argument '{pt_ary.name}' not provided.")

                if not isinstance(updated_kwargs[pt_ary.name], np.ndarray):
                    raise TypeError("BoundMLIRProgram.__call__ expects a numpy"
                        f" array as argument for '{pt_ary.name}', got"
                        f"'{type(updated_kwargs[pt_ary.name])}.'")

                if pt_ary.dtype != updated_kwargs[pt_ary.name].dtype:
                    raise TypeError(f"dtype mismatch for {pt_ary.name}. Expected "
                            f"{pt_ary.dtype}, got "
                            f"{updated_kwargs[pt_ary.name].dtype}.")

                args_to_mlir_knl.append(updated_kwargs[pt_ary.name])
            else:
                assert isinstance(pt_ary, NamedArray)
                if pt_ary.name in updated_kwargs:
                    if not isinstance(updated_kwargs[pt_ary.name], np.ndarray):
                        raise TypeError("BoundMLIRProgram.__call__ expects a numpy"
                            f" array as argument for '{pt_ary.name}', got"
                            f"'{type(updated_kwargs[pt_ary.name])}.'")

                    if pt_ary.dtype != updated_kwargs[pt_ary.name].dtype:
                        raise TypeError(f"dtype mismatch for {pt_ary.name}. "
                                f"Expected {pt_ary.dtype}, got "
                                f"{updated_kwargs[pt_ary.name].dtype}.")

                    args_to_mlir_knl.append(updated_kwargs[pt_ary.name])
                else:
                    # user didn't provide the address for the array => create own.
                    empty_ary = np.empty(dtype=pt_ary.dtype,
                            shape=output_shapes[pt_ary.name])
                    args_to_mlir_knl.append(empty_ary)
                    return_dict[pt_ary.name] = empty_ary

        from mlir.run import mlir_opt, call_function
        source = mlir_opt(self.program.dump(), self.options)
        call_function(source, "_pt_kernel", args_to_mlir_knl)
        return return_dict


"""
    .. attribute:: arguments

        ith entry in the list corresponds to the pytato array name of the ith
        argument of the built MLIR function.
"""


class IndexLambdaSubstitutor(scalar_expr.IdentityMapper):
    """
    Substitutes the usage of an :class:`pytato.array.IndexLambda` according
    to the provided *bindings*.

    ::
        >>> bindings = {"in": scalar_expr.parse("x[_1 + 7, _0] + y[_0, _1]")}
        >>> input_expr = scalar_expr.parse("in[_1 + 3, _0] + 17")
        >>> print(IndexLambdaSubstitutor(bindings)(input_expr))
        >>> x[_0 + 7, _1 + 3] + y[_1 + 3, _0] + 17
    """
    def __init__(self, bindings: Mapping[str, scalar_expr.ScalarExpression]) -> None:
        self.bindings = bindings

    def map_subscript(self, expr: prim.Subscript):
        idx_map = {f"_{i}": idx
                   for i, idx in enumerate(expr.index_tuple)}
        subst_mapper = scalar_expr.SubstitutionMapper(make_subst_func(idx_map))
        return subst_mapper(self.bindings[expr.aggregate.name])

    def map_variable(self, expr: prim.Variable):
        try:
            return self.bindings[expr.name]
        except KeyError:
            return expr


class Pymbolicifier(Mapper):
    """
    Maps an :class:`~pytato.array.Array` to an indexed expression with the indices
    ``_0, _1, ...`` corresponding to the output's indices.

    ::
        >>> from pytato.codegen import preprocess
        >>> x = pt.make_placeholder(ns, shape=(6, 6), dtype=float, name="u")
        >>> y = pt.make_placeholder(ns, shape=(4, 9), dtype=float, name="v")
        >>> z = 10*pt.reshape(x, (-1, )) + 4*pt.reshape(y, (-1, ))
        >>> processed_z = preprocess(
        ...     pt.make_dict_of_named_arrays({"z": z})).outputs["z"]
        >>> print(Pymbolicifier({x: "u", y: "v"})(processed_z))
        >>> 10*u[_0 // 6, _0 % 6] + 4*v[_0 // 9, _0 % 9]
    """
    def __init__(self, expr_to_pymbolic_name: Dict[Array, str],
            available_arrays: Set[str]) -> None:
        self.expr_to_pymbolic_name = expr_to_pymbolic_name
        self.available_arrays = available_arrays

    def map_index_lambda(self, expr: IndexLambda):
        if expr in self.expr_to_pymbolic_name:
            # index lambda has a name, probably stored as a named array
            # => just return the expression in terms of the named array
            name = self.expr_to_pymbolic_name[expr]
            if name in self.available_arrays:
                return prim.Subscript(prim.Variable(name),
                                      tuple(prim.Variable(f"_{i}")
                                            for i in range(expr.ndim)))

        return IndexLambdaSubstitutor({name: self.rec(val)
            for name, val in expr.bindings.items()})(expr.expr)

    def map_placeholder(self, expr: Placeholder):
        return prim.Subscript(prim.Variable(expr.name), tuple(prim.Variable(f"_{i}")
            for i in range(expr.ndim)))

    def map_size_param(self, expr: SizeParam):
        return prim.Variable(self.expr_to_pymbolic_name[expr])


class ScalarExpressionBlockWriter(scalar_expr.IdentityMapper):
    """
    Maps a scalar expression to the mlir ops.

    .. note::

        Cannot handle indexed expressions.
    """
    def __init__(self) -> None:
        self.scalar_expr_to_ssa: Dict[scalar_expr.ScalarExpression,
                                      Tuple[ast.SsaId, np.dtype]] = {}

    def map_constant(self, expr: numbers.Number,
            state: CodeGenState) -> Tuple[ast.SsaId, np.dtype]:
        if expr in self.scalar_expr_to_ssa:
            return self.scalar_expr_to_ssa[expr]

        dtype = np.array(expr).dtype
        mlir_type = np_dtype_to_mlir_dtype(dtype)
        if dtype.kind == "f":
            result = state.builder.float_constant(expr, mlir_type)
        elif dtype.kind == "i":
            result = state.builder.float_constant(expr, mlir_type)
        else:
            raise NotImplementedError(f"{dtype}")

        self.scalar_expr_to_ssa[expr] = result, dtype
        return result, dtype

    def map_dim_op(self, expr: DimOperation,
            state: CodeGenState) -> Tuple[ast.SsaId, np.dtype]:
        ary_ssa = state.pymbolic_name_to_ssa[expr.array_name]
        idx_ssa = state.builder.index_constant(expr.axis)
        result_as_idx = state.builder.dim(ary_ssa, idx_ssa,
                to_memref_type(state.pymbolic_name_to_expr(expr.array_name)))
        result_as_int = state.builder.index_cast(result_as_idx, state.builder.INDEX,
                state.builder.INT64)
        return result_as_int, np.dtype(np.intp)

    def _map_binary_op(self, expr: scalar_expr.ScalarExpression,
                       state: CodeGenState,
                       scalar_op_int_method: str,
                       scalar_op_float_method: str) -> Tuple[ast.SsaId, np.dtype]:

        if expr in self.scalar_expr_to_ssa:
            return self.scalar_expr_to_ssa[expr]

        def emit_binary_op(ssa1: ast.SsaId, dtype1: np.dtype,
                ssa2: ast.SsaId, dtype2: np.dtype):
            res_dtype = (
                    np.empty(0, dtype=dtype1)
                    + np.empty(0, dtype=dtype2)
                    ).dtype
            if res_dtype.kind == "i":
                op = getattr(state.builder, scalar_op_int_method)
            elif res_dtype.kind == "f":
                op = getattr(state.builder, scalar_op_float_method)
            else:
                raise NotImplementedError(f"{res_dtype}")

            ssa1 = astype(state.builder, ssa1, dtype1, res_dtype)
            ssa2 = astype(state.builder, ssa2, dtype2, res_dtype)
            result = op(ssa1, ssa2, np_dtype_to_mlir_dtype(res_dtype))

            return result, res_dtype

        result = reduce(lambda x, y: emit_binary_op(*x, *y),
                      (self.rec(child, state) for child in expr.children))

        self.scalar_expr_to_ssa[expr] = result

        return result

    map_sum = partialmethod(_map_binary_op, scalar_op_int_method="addi",
            scalar_op_float_method="addf")
    map_product = partialmethod(_map_binary_op, scalar_op_int_method="muli",
            scalar_op_float_method="mulf")


class LinalgGenericBlockWriter(ScalarExpressionBlockWriter):
    """
    Maps an indexed expression, with ``_0, _1, ...`` as the indices,
    to its corresponding :class:`mlir.dialects.linalg.LinalgGeneric`.

    :attribute lgen: The linalg generic operation being built. Updated during mapper
        method calls.
    :attribute scalar_expr_to_ssa: A mapping from scalar expression to it's ssa id
        and dtype in the region of the block.
    """
    def __init__(self, lgen: ast.dialects.linalg.LinalgGeneric) -> None:
        super().__init__()
        self.lgen = lgen

    def map_subscript(self, expr: prim.Subscript,
            state: CodeGenState) -> Tuple[ast.SsaId, np.dtype]:
        if expr in self.scalar_expr_to_ssa:
            return self.scalar_expr_to_ssa[expr]

        pt_array = state.pymbolic_name_to_expr(expr.aggregate.name)
        mref_type = to_memref_type(pt_array)

        # {{{ build the affine map

        from pymbolic.mapper.evaluator import evaluate

        dims = [f"_{i}" for i in range(len(expr.index_tuple))]

        if scalar_expr.get_dependencies(expr.index_tuple) > set(dims):
            raise NotImplementedError("only pure indices as dependencies "
                    "allowed for now.")

        multi_dim_affine_expr = evaluate(expr.index_tuple, {
            f"_{i}": state.builder.make_affine_dim(f"_{i}")
            for i in range(len(expr.index_tuple))})

        # }}}

        result = state.builder.linalg_generic_add_in(self.lgen,
                state.pymbolic_name_to_ssa[expr.aggregate.name],
                mref_type,
                state.builder.make_affine_map(multi_dim_affine_expr, dims))

        self.scalar_expr_to_ssa[expr] = result, pt_array.dtype

        return result, pt_array.dtype

    def map_variable(self, expr: prim.Variable,
            state: CodeGenState) -> Tuple[ast.SsaId, np.dtype]:
        pt_array = state.pymbolic_name_to_expr(expr.name)
        return state.pymbolic_name_to_ssa[expr.name], pt_array.dtype


# {{{ builder helpers

def np_dtype_to_mlir_dtype(dtype: np.dtype):
    if dtype == np.float64:
        return IRBuilder.F64
    elif dtype == np.float32:
        return IRBuilder.F32
    elif dtype == np.int32:
        return IRBuilder.INT32
    elif dtype == np.int64:
        return IRBuilder.INT64
    else:
        raise NotImplementedError("No mapping known for type {dtype}.")


def to_memref_type(expr: Array):
    shape = [dim if isinstance(dim, numbers.Integral) else None
             for dim in expr.shape]
    return IRBuilder.MemRefType(shape=shape,
            dtype=np_dtype_to_mlir_dtype(expr.dtype))


def add_function_arg(state: CodeGenState, expr: Array) -> ast.SsaId:
    """
    Returns the ssa id of the added function argument to *state*'s
    :attr:`CodeGenState.function`.
    """
    arg, = state.builder.add_function_args(state.function,
                                           [to_memref_type(expr)],
                                           [expr.name])
    state.arguments.append(expr)
    state.expr_to_pymbolic_name[expr] = expr.name
    state.pymbolic_name_to_ssa[expr.name] = arg
    # return arg


def emit_size_param(state: CodeGenState, expr: SizeParam) -> ast.SsaId:
    bset = state.size_param_array_dim_bset

    # {{{ project out all other size params

    for val in state.namespace.values():
        if isinstance(val, SizeParam):
            if val != expr:
                dt, pos = bset.get_var_dict()[val.name]
                bset = bset.project_out(dt, pos, 1)

    # }}}

    # get expr's dt, pos
    dt, pos = bset.get_var_dict()[expr.name]

    # grab one constraint having 'expr' terms
    cnstrnt = next(iter(cnstrnt for cnstrnt in bset.get_constraints()
        if expr.name in cnstrnt.get_coefficients_by_name()))

    aff = cnstrnt.get_bound(dt, pos)

    def aff_to_expr(aff):
        denom = aff.get_denominator_val().to_python()

        result = (aff.get_constant_val()*denom).to_python()
        for dt in [isl.dim_type.in_, isl.dim_type.param]:
            for i in range(aff.dim(dt)):
                coeff = (aff.get_coefficient_val(dt, i)*denom).to_python()
                if coeff:
                    dim_id = aff.get_dim_id(dt, i)
                    result += coeff*dim_id.user

        for i in range(aff.dim(isl.dim_type.div)):
            coeff = (aff.get_coefficient_val(isl.dim_type.div, i)*denom).to_python()
            if coeff:
                result += coeff*aff_to_expr(aff.get_div(i))

        return result // denom

    ssa, dtype = ScalarExpressionBlockWriter()(aff_to_expr(aff), state)

    state.expr_to_pymbolic_name[expr] = expr.name
    state.pymbolic_name_to_ssa[expr.name] = ssa


def build_linalg_generic(state: CodeGenState,
                         out_ssa: ast.SsaId,
                         pym_expr: scalar_expr.ScalarExpression,
                         pt_expr: Array):
    """
    Adds a :class:`ast.dialects.linalg.LinalgGeneric`operation to *state*'s block.
    """
    lgen = state.builder.linalg_generic(["parallel"]*pt_expr.ndim)

    with state.builder.goto_block(state.builder.make_block(lgen.region)):
        state.builder.linalg_generic_add_out(lgen, out_ssa,
                to_memref_type(pt_expr),
                state.builder.make_identity_map(pt_expr.ndim)
                )
        ssa, dtype = LinalgGenericBlockWriter(lgen)(pym_expr, state)
        state.builder.linalg_yield(ssa, np_dtype_to_mlir_dtype(dtype))


def astype(builder, src_ssa: ast.SsaId,
           src_type: np.dtype,
           dst_type: np.dtype) -> ast.SsaId:
    if src_type == dst_type:
        return src_ssa

    if src_type.kind == "i":
        if dst_type.kind == "f":
            return builder.sitofp(src_ssa, np_dtype_to_mlir_dtype(src_type),
                    np_dtype_to_mlir_dtype(dst_type))

    raise NotImplementedError(f"{src_type} -> {dst_type}: type-cast not implemented")

# }}}


def get_size_param_array_dim_basic_set(outputs):
    namespace = outputs.namespace
    isl_ctx = isl.DEFAULT_CONTEXT

    # {{{ add dims to the ISL set.

    bset = isl.BasicSet.universe(isl.Space.set_alloc(isl_ctx, 0, 0))

    for expr in namespace.values():
        if isinstance(expr, Placeholder):
            for iaxis in range(expr.ndim):
                bset = bset.add_dims(isl.dim_type.set, 1)
                dim_op = DimOperation(expr.name, iaxis)
                bset = bset.set_dim_id(
                        isl.dim_type.set,
                        bset.dim(isl.dim_type.set)-1,
                        isl.Id(context=isl_ctx, name=str(dim_op), user=dim_op))
        elif isinstance(expr, SizeParam):
            bset = bset.add_dims(isl.dim_type.set, 1)
            bset = bset.set_dim_id(
                    isl.dim_type.set,
                    bset.dim(isl.dim_type.set)-1,
                    isl.Id(context=isl_ctx, name=expr.name,
                           user=prim.Variable(expr.name)))
        else:
            # The shape expressions only exert relations between the placeholders
            # and the size params.
            raise NotImplementedError(f"Not implemented for {type(expr)}")

    for name, expr in outputs.items():
        for iaxis in range(expr.ndim):
            bset = bset.add_dims(isl.dim_type.set, 1)
            dim_op = DimOperation(name, iaxis)
            bset = bset.set_dim_id(
                    isl.dim_type.set,
                    bset.dim(isl.dim_type.set)-1,
                    isl.Id(context=isl_ctx, name=str(dim_op), user=dim_op))

    # }}}

    # {{{ add constraints to bset

    # grab shape expressions from placeholder's axis lengths
    for expr in namespace.values():
        if isinstance(expr, Placeholder):
            for iaxis in range(expr.ndim):
                bset &= isl_set_from_expr(bset.get_space(),
                        prim.Comparison(prim.Variable(str(DimOperation(expr.name,
                                                                       iaxis))),
                                        "==",
                                        expr.shape[iaxis]))

    for name, expr in outputs.items():
        for iaxis in range(expr.ndim):
            bset &= isl_set_from_expr(bset.get_space(),
                    prim.Comparison(prim.Variable(str(DimOperation(name,
                                                                   iaxis))),
                                    "==",
                                    expr.shape[iaxis]))

    # }}}

    bset, = bset.get_basic_sets()

    return bset


def generate_mlir(
        result: Union[Array, DictOfNamedArrays, Dict[str, Array]],
        options: List[str] = ["-convert-linalg-to-loops",
                              "-convert-scf-to-std"]) -> BoundProgram:
    """
    Code generation entry point.

    :param result: Outputs of the computation.
    """
    orig_outputs: DictOfNamedArrays = normalize_outputs(result)
    del result

    preproc_result = preprocess(orig_outputs)

    outputs = preproc_result.outputs
    compute_order = preproc_result.compute_order
    namespace = outputs.namespace

    # {{{ get all dependencies of 'output'

    from pytato.transform import get_dependencies
    # https://github.com/python/mypy/issues/2013
    deps: FrozenSet[Array] = set().union(
            *list(get_dependencies(outputs).values()))  # type: ignore
    all_deps = frozenset([dep.name
                            for dep in deps
                            if isinstance(dep, InputArgumentBase)])

    # }}}

    state = get_initial_codegen_state(outputs)

    with state.builder.goto_block(state.builder.make_block(state.function.region)):
        # {{{ register placeholders as function arguments

        for name, val in sorted(namespace.items(),
                                key=lambda x: x[0]  # lexicographic order of names
                                ):
            if isinstance(val, Placeholder):
                add_function_arg(state, val)
                state.stored_names.add(name)

        # }}}

        # {{{ register outputs as function_arguments

        for name in compute_order:
            expr = outputs[name]
            add_function_arg(state,
                             NamedArray(name=name,
                                        shape=expr.shape,
                                        dtype=expr.dtype))

        # }}}

        # {{{ emit size params if needed

        for name, val in sorted(namespace.items(),
                                key=lambda x: x[0]  # lexicographic order of names
                                ):
            if isinstance(val, SizeParam):
                # only emit size param if it is one of the dependencies
                if val.name in all_deps:
                    emit_size_param(state, val)

        # }}}

        for name in compute_order:
            expr = outputs[name]

            build_linalg_generic(state,
                                 state.pymbolic_name_to_ssa[name],
                                 Pymbolicifier(state.expr_to_pymbolic_name,
                                               state.stored_names)(expr),
                                 expr)
            state.stored_names.add(name)

        state.builder.ret()

    return BoundMLIRProgram(state.file,
            preproc_result.bound_arguments,
            options, state.arguments, state.size_param_array_dim_bset)


# vim: fdm=marker
