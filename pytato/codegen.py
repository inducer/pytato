from __future__ import annotations

__copyright__ = """Copyright (C) 2020 Matt Wala"""

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

import dataclasses
from functools import partialmethod
from typing import Union, Dict, Tuple, Callable, List, Any

import pymbolic.primitives as prim
from pymbolic import var

from pytato.array import (Array, DictOfNamedArrays, IndexLambda,
                          DataWrapper, Roll, AxisPermutation, Slice,
                          IndexRemappingBase, Stack, Placeholder, Reshape,
                          Concatenate, DataInterface, SizeParam, InputArgumentBase)
from pytato.scalar_expr import ScalarExpression, IntegralScalarExpression
from pytato.transform import CopyMapper, WalkMapper
from pytato.target import Target
from pytato.loopy import LoopyFunction
from pytools import UniqueNameGenerator
SymbolicIndex = Tuple[IntegralScalarExpression, ...]


__doc__ = """
References
----------

.. class:: DictOfNamedArrays

    Should be referenced as :class:`pytato.DictOfNamedArrays`.

.. class:: DataInterface

    Should be referenced as :class:`pytato.array.DataInterface`.

Code Generation Helpers
-------------------------

.. currentmodule:: pytato.codegen

.. autoclass:: CodeGenPreprocessor
.. autoclass:: PreprocessResult

.. autofunction:: preprocess
.. autofunction:: normalize_outputs
"""


# {{{ preprocessing for codegen

class CodeGenPreprocessor(CopyMapper):
    """A mapper that preprocesses graphs to simplify code generation.

    The following node simplifications are performed:

    ======================================  =====================================
    Source Node Type                        Target Node Type
    ======================================  =====================================
    :class:`~pytato.array.DataWrapper`      :class:`~pytato.array.Placeholder`
    :class:`~pytato.array.Roll`             :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.AxisPermutation`  :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Slice`            :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Reshape`          :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Concatenate`      :class:`~pytato.array.IndexLambda`
    ======================================  =====================================
    """

    # TODO:
    # Stack -> IndexLambda
    # MatrixProduct -> Einsum

    def __init__(self) -> None:
        super().__init__()
        self.bound_arguments: Dict[str, DataInterface] = {}
        self.var_name_gen: UniqueNameGenerator = UniqueNameGenerator()
        self.kernels_seen = set()

    def map_size_param(self, expr: SizeParam) -> Array:
        name = expr.name
        assert name is not None
        return SizeParam(name=name, tags=expr.tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        name = expr.name
        if name is None:
            name = self.var_name_gen("_pt_in")
        return Placeholder(name=name,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                tags=expr.tags)

    def map_loopy_function(self, expr: LoopyFunction) -> LoopyFunction:
        import pytools
        import loopy as lp
        program = expr.program.copy(target=self.target.get_loopy_target())
        namegen = pytools.UniqueNameGenerator(self.kernels_seen)
        entrypoint = expr.entrypoint

        # {{{ eliminate callable name collision

        clbls_to_rename = self.kernels_seen & set(program.callables_table)

        for clbl_to_rename in clbls_to_rename:
            new_name = namegen(clbl_to_rename)
            if clbl_to_rename == entrypoint:
                entrypoint = new_name

            program = lp.rename_callable(program, clbl_to_rename, new_name)

        # }}}

        self.kernels_seen.update(set(program.callables_table))

        bindings = {name: self.rec(subexpr)
                    for name, subexpr in expr.bindings.items()}

        return LoopyFunction(namespace=self.namespace,
                program=program,
                bindings=bindings,
                entrypoint=entrypoint)

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        name = expr.name
        if name is None:
            name = self.var_name_gen("_pt_in")

        self.bound_arguments[name] = expr.data
        return Placeholder(name=name,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                tags=expr.tags)

    def map_stack(self, expr: Stack) -> Array:

        def get_subscript(array_index: int) -> SymbolicIndex:
            result = []
            for i in range(expr.ndim):
                if i != expr.axis:
                    result.append(var(f"_{i}"))
            return tuple(result)

        # I = axis index
        #
        # => If(_I == 0,
        #        _in0[_0, _1, ...],
        #        If(_I == 1,
        #            _in1[_0, _1, ...],
        #            ...
        #                _inNm1[_0, _1, ...] ...))
        for i in range(len(expr.arrays) - 1, -1, -1):
            subarray_expr = var(f"_in{i}")[get_subscript(i)]
            if i == len(expr.arrays) - 1:
                stack_expr = subarray_expr
            else:
                from pymbolic.primitives import If, Comparison
                stack_expr = If(Comparison(var(f"_{expr.axis}"), "==", i),
                        subarray_expr,
                        stack_expr)

        bindings = {f"_in{i}": self.rec(array)
                for i, array in enumerate(expr.arrays)}

        return IndexLambda(expr=stack_expr,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                bindings=bindings)

    def map_concatenate(self, expr: Concatenate) -> Array:
        from pymbolic.primitives import If, Comparison, Subscript

        def get_subscript(array_index: int, offset: ScalarExpression) -> Subscript:
            aggregate = var(f"_in{array_index}")
            index = [var(f"_{i}") if i != expr.axis else (var(f"_{i}") - offset)
                     for i in range(len(expr.shape))]
            return Subscript(aggregate, tuple(index))

        lbounds: List[Any] = [0]
        ubounds: List[Any] = [expr.arrays[0].shape[expr.axis]]

        for i, array in enumerate(expr.arrays[1:], start=1):
            ubounds.append(ubounds[i-1]+array.shape[expr.axis])
            lbounds.append(ubounds[i-1])

        # I = axis index
        #
        # => If(0<=_I < arrays[0].shape[axis],
        #        _in0[_0, _1, ..., _I, ...],
        #        If(arrays[0].shape[axis]<= _I < (arrays[1].shape[axis]
        #                                         +arrays[0].shape[axis]),
        #            _in1[_0, _1, ..., _I-arrays[0].shape[axis], ...],
        #            ...
        #                _inNm1[_0, _1, ...] ...))
        for i in range(len(expr.arrays) - 1, -1, -1):
            lbound, ubound = lbounds[i], ubounds[i]
            subarray_expr = get_subscript(i, lbound)
            if i == len(expr.arrays) - 1:
                stack_expr = subarray_expr
            else:
                stack_expr = If(Comparison(var(f"_{expr.axis}"), ">=", lbound)
                                and Comparison(var(f"_{expr.axis}"), "<", ubound),
                                subarray_expr,
                                stack_expr)

        bindings = {f"_in{i}": self.rec(array)
                for i, array in enumerate(expr.arrays)}

        return IndexLambda(expr=stack_expr,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                bindings=bindings)

    def map_roll(self, expr: Roll) -> Array:
        from pytato.utils import dim_to_index_lambda_components

        index_expr = var("_in0")
        indices = [var(f"_{d}") for d in range(expr.ndim)]
        axis = expr.axis
        axis_len_expr, bindings = dim_to_index_lambda_components(
            expr.shape[axis],
            UniqueNameGenerator({"_in0"}))

        indices[axis] = (indices[axis] - expr.shift) % axis_len_expr

        if indices:
            index_expr = index_expr[tuple(indices)]

        bindings["_in0"] = expr.array  # type: ignore

        return IndexLambda(expr=index_expr,
                           shape=tuple(self.rec(s) if isinstance(s, Array) else s
                                       for s in expr.shape),
                           dtype=expr.dtype,
                           bindings={name: self.rec(bnd)
                                     for name, bnd in bindings.items()})

    # {{{ index remapping (roll, axis permutation, slice)

    def handle_index_remapping(self,
            indices_getter: Callable[[CodeGenPreprocessor, Array], SymbolicIndex],
            expr: IndexRemappingBase) -> Array:
        indices = indices_getter(self, expr)

        index_expr = var("_in0")
        if indices:
            index_expr = index_expr[indices]

        array = self.rec(expr.array)

        return IndexLambda(expr=index_expr,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                bindings=dict(_in0=array))

    def _indices_for_axis_permutation(self, expr: AxisPermutation) -> SymbolicIndex:
        indices = [None] * expr.ndim
        for from_index, to_index in enumerate(expr.axes):
            indices[to_index] = var(f"_{from_index}")
        return tuple(indices)

    def _indices_for_slice(self, expr: Slice) -> SymbolicIndex:
        return tuple(var(f"_{d}") + expr.starts[d] for d in range(expr.ndim))

    def _indices_for_reshape(self, expr: Reshape) -> SymbolicIndex:
        newstrides = [1]  # reshaped array strides
        for new_axis_len in reversed(expr.shape[1:]):
            assert isinstance(new_axis_len, int)
            newstrides.insert(0, newstrides[0]*new_axis_len)

        flattened_idx = sum(prim.Variable(f"_{i}")*stride
                            for i, stride in enumerate(newstrides))

        oldstrides = [1]  # input array strides
        for axis_len in reversed(expr.array.shape[1:]):
            assert isinstance(axis_len, int)
            oldstrides.insert(0, oldstrides[0]*axis_len)

        assert isinstance(expr.array.shape[-1], int)
        oldsizetills = [expr.array.shape[-1]]  # input array size till for axes idx
        for old_axis_len in reversed(expr.array.shape[:-1]):
            assert isinstance(old_axis_len, int)
            oldsizetills.insert(0, oldsizetills[0]*old_axis_len)

        return tuple(((flattened_idx % sizetill) // stride)
                     for stride, sizetill in zip(oldstrides, oldsizetills))

    # https://github.com/python/mypy/issues/8619
    map_axis_permutation = (
            partialmethod(handle_index_remapping, _indices_for_axis_permutation))  # type: ignore  # noqa
    map_slice = partialmethod(handle_index_remapping, _indices_for_slice)  # type: ignore  # noqa
    map_reshape = partialmethod(handle_index_remapping, _indices_for_reshape) # noqa

    # }}}

# }}}


def normalize_outputs(result: Union[Array, DictOfNamedArrays,
                                    Dict[str, Array]]) -> DictOfNamedArrays:
    """Convert outputs of a computation to the canonical form.

    Performs a conversion to :class:`~pytato.DictOfNamedArrays` if necessary.

    :param result: Outputs of the computation.
    """
    if not isinstance(result, (Array, DictOfNamedArrays, dict)):
        raise TypeError("outputs of the computation should be "
                "either an Array or a DictOfNamedArrays")

    if isinstance(result, Array):
        outputs = DictOfNamedArrays({"_pt_out": result})
    elif isinstance(result, dict):
        outputs = DictOfNamedArrays(result)
    else:
        assert isinstance(result, DictOfNamedArrays)
        outputs = result

    return outputs


# {{{ input naming check

class NamesValidityChecker(WalkMapper):
    def __init__(self) -> None:
        self.name_to_input: Dict[str, InputArgumentBase] = {}

    def post_visit(self, expr: Any) -> None:
        if isinstance(expr, InputArgumentBase):
            if expr.name is None:
                # Name to be automatically assigned
                return

            try:
                ary = self.name_to_input[expr.name]
            except KeyError:
                self.name_to_input[expr.name] = expr
            else:
                if ary is not expr:
                    from pytato.diagnostic import NameClashError
                    raise NameClashError("Received two separate instances of inputs "
                                         f"named '{expr.name}'.")


def check_validity_of_outputs(exprs: DictOfNamedArrays) -> None:
    name_validation_mapper = NamesValidityChecker()

    for ary in exprs.values():
        name_validation_mapper(ary)

# }}}


@dataclasses.dataclass(init=True, repr=False, eq=False)
class PreprocessResult:
    outputs: DictOfNamedArrays
    compute_order: Tuple[str, ...]
    bound_arguments: Dict[str, DataInterface]


def preprocess(outputs: DictOfNamedArrays, target: Target) -> PreprocessResult:
    """Preprocess a computation for code generation."""
    from pytato.transform import copy_dict_of_named_arrays, get_dependencies

    check_validity_of_outputs(outputs)

    # {{{ compute the order in which the outputs must be computed

    # semantically order does not matter, but doing a toposort ordering of the
    # outputs leads to a FLOP optimal choice

    from pytools.graph import compute_topological_order

    deps = get_dependencies(outputs)

    # only look for dependencies between the outputs
    deps = {name: (val & frozenset(outputs.exprs))
            for name, val in deps.items()}

    # represent deps in terms of output names
    output_expr_to_name = {output.expr: name for name, output in outputs.items()}
    dag = {name: (frozenset([output_expr_to_name[output] for output in val])
                  - frozenset([name]))
           for name, val in deps.items()}

    output_order: List[str] = compute_topological_order(dag)[::-1]

    # }}}

    mapper = CodeGenPreprocessor(target)

    new_outputs = copy_dict_of_named_arrays(outputs, mapper)

    return PreprocessResult(outputs=new_outputs,
            compute_order=tuple(output_order),
            bound_arguments=mapper.bound_arguments)

# vim: fdm=marker
