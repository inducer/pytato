from __future__ import annotations

import numpy as np
import mlir.astnodes as ast
from mlir.builder import IRBuilder
from typing import Union, Mapping, Any, Dict, Tuple, List
from pytato.array import (Array, DictOfNamedArrays, Namespace,
        SizeParam, Placeholder, IndexLambda, DataInterface)
from dataclasses import dataclass
import pytato.scalar_expr as scalar_expr
from pytato.codegen import normalize_outputs, preprocess
from pytato.transform import Mapper
import pymbolic.primitives as prim
from pymbolic.mapper.substitutor import make_subst_func
from functools import reduce
from numbers import Number
from pytato.program import BoundProgram


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
    options: List[str]
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
    """
    Substitutes the usage of an :class:`pytato.array.IndexLambda` according
    to the provided *bindings*.

    ::
        >>> bindings = {"in": scalar_expr.parse("x[_1 + 7, _0] + y[_0, _1]")}
        >>> input_expr = scalar_expr.parse("in[_1 + 3, _0] + 17")
        >>> print(IndexLambdaSubstitutor(bindings)(input_expr))
        >>> x[_0 + 7, _1 + 3] + y[_1 + 3, _0] + 17
    """
    def __init__(self, bindings: Mapping[str, scalar_expr.scalar_expr]) -> None:
        self.bindings = bindings

    def map_subscript(self, expr: prim.Subscript):
        idx_map = {f"_{i}": idx
                   for i, idx in enumerate(expr.index_tuple)}
        subst_mapper = scalar_expr.SubstitutionMapper(make_subst_func(idx_map))
        return subst_mapper(self.bindings[expr.aggregate.name])


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
    def __init__(self, expr_to_pymbolic_name: Dict[Array, str]):
        self.expr_to_pymbolic_name = expr_to_pymbolic_name

    def map_index_lambda(self, expr: IndexLambda):
        if expr in self.expr_to_pymbolic_name:
            # index lambda has a name, probably stored as a named array
            # => just return the expression in terms of the named array
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
    Maps an indexed expression, with ``_0, _1, ...`` as the indices,
    to a :class:`mlir.dialects.linalg.LinalgGeneric` for it.

    :attribute lgen: The linalg generic operation being built. Updated during mapper
        method calls.
    :attribute scalar_expr_to_ssa: A mapping from scalar expression to it's ssa id
        and dtype in the region of the block.
    """
    def __init__(self, lgen: ast.dialects.linalg.LinalgGeneric) -> None:
        self.lgen = lgen
        self.scalar_expr_to_ssa: Dict[scalar_expr.scalar_expr,
                                      Tuple[ast.SsaId, np.dtype]] = {}

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
    return IRBuilder.MemRefType(shape=expr.shape,
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
    return arg


def build_linalg_generic(state: CodeGenState,
                         out_ssa: ast.SsaId,
                         pym_expr: scalar_expr.scalar_expr,
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

# }}}


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

    state = get_initial_codegen_state(namespace)

    # Generate code for DAG's leaves
    for name, val in sorted(namespace.items(),
                            key=lambda x: x[0]  # lexicographic order of names
                            ):
        if isinstance(val, SizeParam):
            raise ValueError("SizeParams requires Pytato to have a shape inference"
                    " engine of its own")
        elif isinstance(val, Placeholder):
            assert all(isinstance(dim, int) for dim in val.shape)
            arg = add_function_arg(state, val)
            state.expr_to_pymbolic_name[val] = name
            state.pymbolic_name_to_ssa[name] = arg
        else:
            raise NotImplementedError(f"Not implemented for type {type(val)}.")

    with state.builder.goto_block(state.builder.make_block(state.function.region)):
        for name in compute_order:
            expr = outputs[name]

            build_linalg_generic(state,
                                 add_function_arg(state,
                                                  NamedArray(name=name,
                                                             shape=expr.shape,
                                                             dtype=expr.dtype)),
                                  Pymbolicifier(state.expr_to_pymbolic_name)(expr),
                                  expr)

            state.expr_to_pymbolic_name[expr] = name
            state.pymbolic_name_to_ssa[name] = arg

        state.builder.ret()

    return BoundMLIRProgram(state.file,
            preproc_result.bound_arguments,
            options, state.arguments)


# vim: fdm=marker
