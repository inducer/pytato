from __future__ import annotations

import numpy as np
import mlir.astnodes as ast
from mlir.builder import IRBuilder
from typing import Union, Mapping, Any, Dict, Tuple
from pytato.array import (Array, DictOfNamedArrays, Namespace, InputArgumentBase,
        SizeParam, Placeholder, IndexLambda)
from dataclasses import dataclass
import pytato.scalar_expr as scalar_expr
from pytato.codegen import normalize_outputs, preprocess
from pytato.transform import Mapper
import pymbolic.primitives as prim
from pymbolic.mapper.substitutor import make_subst_func
from functools import reduce
from numbers import Number


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


@dataclass
class BoundProgram:
    program: ast.MLIRFile
    bound_arguments: Mapping[str, Any]


@dataclass
class CodeGenState:
    builder: IRBuilder
    file: ast.MLIRFile
    module: ast.Module
    function: ast.Function
    namespace: Namespace
    expr_to_pymbolic_name: Mapping[Array, str]
    pymbolic_name_to_ssa: Mapping[str, ast.SsaId]

    def pymbolic_name_to_expr(self, name: str) -> Array:
        try:
            return next(iter(key for key, val in self.expr_to_pymbolic_name.items()
                             if val == name))
        except StopIteration:
            raise ValueError(f"Unknown pymbolic name '{name}'")


def get_initial_codegen_state(namespace) -> CodeGenState:
    builder = IRBuilder()
    mlir_file = builder.make_mlir_file()
    module = mlir_file.module
    with builder.goto_block(builder.make_block(module.region)):
        fn = builder.function("_pt_kernel")

    return CodeGenState(builder, mlir_file, module, fn, namespace, {}, {})


class BoundMLIRProgram(BoundProgram):
    def __call__(self, *arg: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


class IndexLambdaSubstitutor(scalar_expr.IdentityMapper):
    def __init__(self, bindings: Mapping[str, scalar_expr.scalar_expr]) -> None:
        self.bindings = bindings

    def map_subscript(self, expr: prim.Subscript):
        idx_map = {f"_{i}": idx
                   for i, idx in enumerate(expr.index_tuple)}
        subst_mapper = scalar_expr.SubstitutionMapper(make_subst_func(idx_map))
        return subst_mapper(self.bindings[expr.aggregate.name])


class Pymbolicifier(Mapper):
    def __init__(self, expr_to_pymbolic_name: Dict[Array, str]):
        self.expr_to_pymbolic_name = expr_to_pymbolic_name

    def map_index_lambda(self, expr: IndexLambda):
        if expr in self.expr_to_pymbolic_name:
            name = self.expr_to_pymbolic_name[expr]
            return prim.Subscript(prim.Variable(name),
                                  tuple(prim.Variable(f"_{i}")
                                        for i in range(expr.ndim)))

        return IndexLambdaSubstitutor({name: self.rec(val)
            for name, val in expr.bindings.items()})(expr.expr)

    def map_placeholder(self, expr: Placeholder):
        return prim.Subscript(prim.Variable(expr.name), tuple(prim.Variable(f"_{i}")
            for i in range(expr.ndim)))


class LinalgGenericBlockWriter(scalar_expr.IdentityMapper):
    """
    """
    def __init__(self, lgen: ast.dialects.linalg.LinalgGeneric) -> None:
        self.lgen = lgen
        self.scalar_expr_to_ssa: Dict[scalar_expr.scalar_expr, ast.SsaId] = {}

    def map_subscript(self, expr: prim.Subscript,
            state: CodeGenState) -> Tuple[ast.SsaId, np.dtype]:
        if expr in self.scalar_expr_to_ssa:
            return self.scalar_expr_to_ssa[expr]

        pt_array = state.pymbolic_name_to_expr(expr.aggregate.name)
        mref_type = state.builder.MemRefType(shape=pt_array.shape,
                dtype=np_dtype_to_mlir_dtype(pt_array.dtype))

        # {{{ build the affine map

        from pymbolic.mapper.evaluator import evaluate

        dims = [f"_{i}" for i in range(len(expr.index_tuple))]

        if scalar_expr.get_dependencies(expr.index_tuple) > set(dims):
            raise NotImplementedError("only pure indices as dependencies"
                    " allowed for now.")

        multi_dim_affine_expr = evaluate(expr.index_tuple, {
            f"_{i}": state.builder.make_affine_dim(f"_{i}")
            for i in range(len(expr.index_tuple))})

        result = state.builder.linalg_generic_add_in(self.lgen,
                state.pymbolic_name_to_ssa[expr.aggregate.name],
                mref_type,
                state.builder.make_affine_map(multi_dim_affine_expr, dims))

        # }}}

        self.scalar_expr_to_ssa[expr] = result

        return result, pt_array.dtype

    def map_constant(self, expr: Number,
            state: CodeGenState) -> Tuple[ast.SsaId, np.dtype]:
        dtype = np.array(expr).dtype
        if dtype.kind == "f":
            result = state.builder.float_constant(expr, dtype)
        elif dtype.kind == "i":
            result = state.builder.float_constant(expr, dtype)
        else:
            raise NotImplementedError(f"{dtype}")

        return result, dtype

    def map_sum(self, expr: prim.Sum,
            state: CodeGenState) -> Tuple[ast.SsaId, np.dtype]:
        def add_values_in_ssa(ssa1: ast.SsaId, dtype1: np.dtype,
                ssa2: ast.SsaId, dtype2: np.dtype):
            res_dtype = (
                    np.empty(0, dtype=dtype1)
                    + np.empty(0, dtype=dtype2)
                    ).dtype
            if res_dtype.kind == "f":
                result = state.builder.addf(ssa1, ssa2,
                        np_dtype_to_mlir_dtype(res_dtype))
            elif res_dtype.kind == "i":
                result = state.builder.addi(ssa1, ssa2,
                        np_dtype_to_mlir_dtype(res_dtype))
            else:
                raise NotImplementedError(f"{res_dtype}")

            return result, res_dtype

        return reduce(add_values_in_ssa,
                      (self.rec(child) for child in expr.children))

    def map_product(self, expr: prim.Add,
            state: CodeGenState) -> Tuple[ast.SsaId, np.dtype]:
        def mult_values_in_ssa(ssa1: ast.SsaId, dtype1: np.dtype,
                ssa2: ast.SsaId, dtype2: np.dtype):
            res_dtype = (
                    np.empty(0, dtype=dtype1)
                    * np.empty(0, dtype=dtype2)
                    ).dtype
            if res_dtype.kind == "f":
                result = state.builder.mulf(ssa1, ssa2,
                        np_dtype_to_mlir_dtype(res_dtype))
            elif res_dtype.kind == "i":
                result = state.builder.muli(ssa1, ssa2,
                        np_dtype_to_mlir_dtype(res_dtype))
            else:
                raise NotImplementedError(f"{res_dtype}")

            return result, res_dtype

        return reduce(lambda x, y: mult_values_in_ssa(*x, *y),
                      (self.rec(child, state) for child in expr.children))


def generate_mlir(
        result: Union[Array, DictOfNamedArrays, Dict[str, Array]]) -> BoundProgram:
    r"""Code generation entry point.

    :param result: Outputs of the computation.
    """
    orig_outputs: DictOfNamedArrays = normalize_outputs(result)
    del result

    preproc_result = preprocess(orig_outputs)
    outputs = preproc_result.outputs
    compute_order = preproc_result.compute_order
    namespace = outputs.namespace

    state = get_initial_codegen_state(namespace)

    # Generate code for graph leaves
    for name, val in sorted(namespace.items(),
                            key=lambda x: x[0]  # lexicographic order of names
                            ):
        if isinstance(val, SizeParam):
            raise ValueError("SizeParams requires Pytato to have a shape inference"
                    " engine of its own")
        elif isinstance(val, Placeholder):
            assert all(isinstance(dim, int) for dim in val.shape)
            mref_type = state.builder.MemRefType(shape=val.shape,
                    dtype=np_dtype_to_mlir_dtype(val.dtype))
            arg, = state.builder.add_function_args(state.function,
                                                   [mref_type],
                                                   [val.name])
            state.expr_to_pymbolic_name[val] = name
            state.pymbolic_name_to_ssa[name] = arg
        else:
            assert isinstance(val, InputArgumentBase)
            raise NotImplementedError(f"Not implemented for type {type(val)}.")

    with state.builder.goto_block(state.builder.make_block(state.function.region)):
        for name in compute_order:
            expr = outputs[name]

            # {{{ register the output arrays as one of the function args

            # TODO: Should be shut inside a function

            mref_type = state.builder.MemRefType(shape=expr.shape,
                    dtype=np_dtype_to_mlir_dtype(expr.dtype))
            arg, = state.builder.add_function_args(state.function,
                                                   [mref_type],
                                                   [name])
            # }}}

            pymbolic_expr = Pymbolicifier(state.expr_to_pymbolic_name)(expr)
            lgen = state.builder.linalg_generic(["parallel"]*expr.ndim)

            with state.builder.goto_block(state.builder.make_block(lgen.region)):
                # TODO: This should also be cleaned (referring to mref_dtype
                # multiple times isn't ideal)
                state.builder.linalg_generic_add_out(lgen, arg, mref_type,
                        state.builder.make_identity_map(expr.ndim))
                ssa, dtype = LinalgGenericBlockWriter(lgen)(pymbolic_expr, state)
                state.builder.linalg_yield(ssa, np_dtype_to_mlir_dtype(dtype))

            state.expr_to_pymbolic_name[expr] = name
            state.pymbolic_name_to_ssa[name] = arg

        state.builder.ret()


    print(state.file.dump())
    1/0

    return BoundMLIRProgram(state.builder.file, preproc_result.bound_arguments)


# vim: fdm=marker
