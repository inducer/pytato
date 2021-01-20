from __future__ import annotations

import numpy as np
import mlir.astnodes as ast
from mlir.builder import IRBuilder
from typing import Union, Mapping, Any, Dict
from pytato.array import Array, DictOfNamedArrays, Namespace
from dataclasses import dataclass
from pytato.codegen import normalize_outputs, preprocess
from pytato.transform import Mapper


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

    def from_numpy_dtype(self, dtype: np.dtype):
        if dtype == np.float64:
            return self.builder.F64
        else:
            raise NotImplementedError(f"Unknown type {dtype}")


def get_initial_codegen_state(namespace) -> CodeGenState:
    builder = IRBuilder()
    mlir_file = builder.make_mlir_file()
    module = mlir_file.module
    with builder.goto_block(builder.make_block(module.region)):
        fn = builder.function("_pt_kernel")

    return CodeGenState(builder, mlir_file, module, fn, namespace)


class BoundMLIRProgram(BoundProgram):
    def __call__(self, *arg: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


class CodeGenMapper(Mapper):
    def __init__(self):
        self.cache: Dict[Array, Any] = {}

    def map_placeholder(self, expr: ast.Placeholder,
            state: CodeGenState) -> ast.SsaId:
        if expr in self.cache:
            return self.cache[expr]

        mlir_dtype = state.from_numpy_dtype(expr.dtype)
        if not all(isinstance(dim, int) for dim in expr.shape):
            raise NotImplementedError("Symbolic shapes not implemented.")
        memref_type = state.builder.MemRefType(shape=expr.shape, dtype=mlir_dtype)
        return state.builder.add_function_args(state.function,
                                               [memref_type],
                                               expr.name)[0]


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
    cg_mapper = CodeGenMapper()
    for name, val in sorted(namespace.items(),
                            key=lambda x: x[0]  # lexicographic order of names
                            ):
        _ = cg_mapper(val, state)

    # Generate code for outputs.
    builder = state.builder
    with builder.goto_block(builder.make_block(builder.function.region)):
        for name in compute_order:
            expr = outputs[name]
            raise NotImplementedError(expr)

        builder.ret()

    return BoundMLIRProgram(builder.file, preproc_result.bound_arguments)


# vim: fdm=marker
