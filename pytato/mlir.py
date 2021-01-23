from __future__ import annotations

import numpy as np
import mlir.astnodes as ast
from mlir.builder import IRBuilder
from typing import Union, Mapping, Any, Dict
from pytato.array import (Array, DictOfNamedArrays, Namespace, InputArgumentBase,
        SizeParam, Placeholder, IndexLambda)
from dataclasses import dataclass
import pytato.scalar_expr as scalar_expr
from pytato.codegen import normalize_outputs, preprocess
from pytato.transform import Mapper
import pymbolic.primitives as prim
from pymbolic.mapper.substitutor import make_subst_func


def np_dtype_to_mlir_dtype(dtype: np.dtype):
    if dtype == np.float64:
        return IRBuilder.F64
    elif dtype == np.float32:
        return IRBuilder.F32
    elif dtype == np.int32:
        return IRBuilder.I32
    elif dtype == np.int64:
        return IRBuilder.I64
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
    expr_to_ssa: Mapping[Array, ast.SsaId]


def get_initial_codegen_state(namespace) -> CodeGenState:
    builder = IRBuilder()
    mlir_file = builder.make_mlir_file()
    module = mlir_file.module
    with builder.goto_block(builder.make_block(module.region)):
        fn = builder.function("_pt_kernel")

    return CodeGenState(builder, mlir_file, module, fn, namespace, {})


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
    def map_index_lambda(self, expr: IndexLambda):
        return IndexLambdaSubstitutor({name: self.rec(val)
            for name, val in expr.bindings.items()})(expr.expr)

    def map_placeholder(self, expr: Placeholder):
        return prim.Subscript(prim.Variable(expr.name), tuple(prim.Variable(f"_{i}")
            for i in range(expr.ndim)))


class LinalgGenericizer(scalar_expr.IdentityMapper):
    ...


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
            state.expr_to_ssa[val] = arg
        else:
            assert isinstance(val, InputArgumentBase)
            raise NotImplementedError(f"Not implemented for type {type(val)}.")

    with state.builder.goto_block(state.builder.make_block(state.function.region)):
        for name in compute_order:
            expr = outputs[name]
            assert isinstance(expr, IndexLambda)
            pymbolic_expr = Pymbolicifier()(expr)
            print(pymbolic_expr)
            1/0
            with state.builder.linalg_generic() as lgen:
                with state.builder.goto_bock(state.builder.make_block(lgen.region)):
                    LinalgGenericizer()(pymbolic_expr, state.builder, lgen)

        state.builder.ret()

    return BoundMLIRProgram(state.builder.file, preproc_result.bound_arguments)


# vim: fdm=marker
