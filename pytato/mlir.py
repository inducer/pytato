from __future__ import annotations

import numpy as np
import mlir.astnodes as ast
from mlir.builder import IRBuilder
from typing import Union, Mapping, Any, Dict, Tuple, List
from pytato.array import (Array, DictOfNamedArrays, Namespace, InputArgumentBase,
        SizeParam, Placeholder, IndexLambda, DataInterface)
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
class BoundProgram:
    program: ast.MLIRFile
    bound_arguments: Mapping[str, Any]
    options: List[str]


@dataclass
class CodeGenState:
    builder: IRBuilder
    file: ast.MLIRFile
    module: ast.Module
    function: ast.Function
    namespace: Namespace
    expr_to_pymbolic_name: Mapping[Array, str]
    pymbolic_name_to_ssa: Mapping[str, ast.SsaId]
    arguments: List[str]

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

    return CodeGenState(builder, mlir_file, module, fn, namespace, {}, {}, [])


@dataclass
class BoundMLIRProgram(BoundProgram):
    arguments: List[Union[Placeholder, NamedArray]]

    def __call__(self, **kwargs: DataInterface) -> Any:
        args_to_mlir_knl = []
        return_dict = {}

        if set(kwargs.keys()) & set(self.bound_arguments.keys()):
            raise ValueError("Got arguments that were previously bound: "
                    f"{set(kwargs.keys()) & set(self.bound_arguments.keys())}.")

        updated_kwargs = self.bound_arguments.copy()
        updated_kwargs.update(kwargs)

        for pt_ary in self.arguments:
            if isinstance(pt_ary, Placeholder):
                # FIXME: Need to do some type, shape checking
                if pt_ary.name not in updated_kwargs:
                    raise ValueError(f"Argument '{pt_ary.name}' not provided.")

                if not isinstance(updated_kwargs[pt_ary.name], np.ndarray):
                    raise TypeError("BoundMLIRProgram.__call__ expects a numpy"
                        f" array as argument for '{pt_ary.name}', got"
                        f"'{type(updated_kwargs[pt_ary.name])}.'")

                args_to_mlir_knl.append(updated_kwargs[pt_ary.name])
            else:
                assert isinstance(pt_ary, NamedArray)
                if pt_ary.name in updated_kwargs:
                    # FIXME: Need to do some type, shape checking
                    args_to_mlir_knl.append(updated_kwargs[pt_ary.name])
                else:
                    # user didn't provide the address for the array => create own.
                    empty_ary = np.empty(dtype=pt_ary.dtype, shape=pt_ary.shape)
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
        mlir_type = np_dtype_to_mlir_dtype(dtype)
        if dtype.kind == "f":
            result = state.builder.float_constant(expr, mlir_type)
        elif dtype.kind == "i":
            result = state.builder.float_constant(expr, mlir_type)
        else:
            raise NotImplementedError(f"{dtype}")

        return result, dtype

    def map_sum(self, expr: prim.Add,
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

        return reduce(lambda x, y: add_values_in_ssa(*x, *y),
                      (self.rec(child, state) for child in expr.children))

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
        result: Union[Array, DictOfNamedArrays, Dict[str, Array]],
        options: List[str] = ["-convert-linalg-to-loops",
                              "-convert-scf-to-std"]) -> BoundProgram:
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
            state.arguments.append(val)
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
            state.arguments.append(NamedArray(name=name, shape=expr.shape,
                dtype=expr.dtype))
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

    return BoundMLIRProgram(state.file,
            preproc_result.bound_arguments,
            options, state.arguments)


# vim: fdm=marker
