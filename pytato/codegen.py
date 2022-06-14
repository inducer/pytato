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
                          DataWrapper, Roll, AxisPermutation,
                          IndexRemappingBase, Stack, Placeholder, Reshape,
                          Concatenate, DataInterface, SizeParam,
                          InputArgumentBase, Einsum,
                          AdvancedIndexInContiguousAxes,
                          AdvancedIndexInNoncontiguousAxes, BasicIndex,
                          NormalizedSlice)

from pytato.scalar_expr import (ScalarExpression, IntegralScalarExpression,
                                INT_CLASSES, IntegralT)
from pytato.transform import CopyMapper, CachedWalkMapper, SubsetDependencyMapper
from pytato.target import Target
from pytato.loopy import LoopyCall
from pytato.tags import AssumeNonNegative
from pytools import UniqueNameGenerator
from immutables import Map
import loopy as lp
SymbolicIndex = Tuple[IntegralScalarExpression, ...]


__doc__ = """
.. currentmodule:: pytato.codegen

.. autoclass:: CodeGenPreprocessor
.. autoclass:: PreprocessResult

.. autofunction:: preprocess
.. autofunction:: normalize_outputs
"""


# {{{ _generate_name_for_temp

def _generate_name_for_temp(
        expr: Array, var_name_gen: UniqueNameGenerator,
        default_prefix: str = "_pt_temp") -> str:
    from pytato.tags import _BaseNameTag, Named, PrefixNamed
    if expr.tags_of_type(_BaseNameTag):
        if expr.tags_of_type(Named):
            name_tag, = expr.tags_of_type(Named)
            if var_name_gen.is_name_conflicting(name_tag.name):
                raise ValueError(f"Cannot assign the name {name_tag.name} to the"
                                 f" temporary corresponding to {expr} as it "
                                 "conflicts with an existing name. ")
            var_name_gen.add_name(name_tag.name)
            return name_tag.name
        elif expr.tags_of_type(PrefixNamed):
            prefix_tag, = expr.tags_of_type(PrefixNamed)
            return var_name_gen(prefix_tag.prefix)
        else:
            raise NotImplementedError(type(list(expr.tags_of_type(_BaseNameTag))[0]))
    else:
        return var_name_gen(default_prefix)

# }}}


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
    :class:`~pytato.array.IndexBase`        :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Reshape`          :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Concatenate`      :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Einsum`           :class:`~pytato.array.IndexLambda`
    ======================================  =====================================
    """

    # TODO:
    # Stack -> IndexLambda

    def __init__(self, target: Target) -> None:
        super().__init__()
        self.bound_arguments: Dict[str, DataInterface] = {}
        self.var_name_gen: UniqueNameGenerator = UniqueNameGenerator()
        self.target = target
        self.kernels_seen: Dict[str, lp.LoopKernel] = {}

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
                axes=expr.axes,
                tags=expr.tags)

    def map_loopy_call(self, expr: LoopyCall) -> LoopyCall:
        from pytato.target.loopy import LoopyTarget
        if not isinstance(self.target, LoopyTarget):
            raise ValueError("Got a LoopyCall for a non-loopy target.")
        translation_unit = expr.translation_unit.copy(
                                        target=self.target.get_loopy_target())
        namegen = UniqueNameGenerator(set(self.kernels_seen))
        entrypoint = expr.entrypoint

        # {{{ eliminate callable name collision

        for name, clbl in translation_unit.callables_table.items():
            if isinstance(clbl, lp.kernel.function_interface.CallableKernel):
                if name in self.kernels_seen and (
                        translation_unit[name] != self.kernels_seen[name]):
                    # callee name collision => must rename

                    # {{{ see if it's one of the other kernels

                    for other_knl in self.kernels_seen.values():
                        if other_knl.copy(name=name) == translation_unit[name]:
                            new_name = other_knl.name
                            break
                    else:
                        # didn't find any other equivalent kernel, rename to
                        # something unique
                        new_name = namegen(name)

                    # }}}

                    if name == entrypoint:
                        # if the colliding name is the entrypoint, then rename the
                        # entrypoint as well.
                        entrypoint = new_name

                    translation_unit = lp.rename_callable(
                                            translation_unit, name, new_name)
                    name = new_name

                self.kernels_seen[name] = translation_unit[name]

        # }}}

        bindings = {name: (self.rec(subexpr) if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())}

        return LoopyCall(translation_unit=translation_unit,
                         bindings=bindings,
                         entrypoint=entrypoint)

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        name = _generate_name_for_temp(expr, self.var_name_gen, "_pt_data")

        self.bound_arguments[name] = expr.data
        return Placeholder(name=name,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                axes=expr.axes,
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
                axes=expr.axes,
                bindings=bindings,
                var_to_reduction_descr=Map(),
                tags=expr.tags)

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
                bindings=bindings,
                axes=expr.axes,
                var_to_reduction_descr=Map(),
                tags=expr.tags)

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
                                     for name, bnd in bindings.items()},
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags)

    def map_einsum(self, expr: Einsum) -> Array:
        import operator
        from functools import reduce
        from pytato.scalar_expr import Reduce
        from pytato.utils import (dim_to_index_lambda_components,
                                  are_shape_components_equal)
        from pytato.array import EinsumElementwiseAxis, EinsumReductionAxis

        bindings = {f"in{k}": self.rec(arg) for k, arg in enumerate(expr.args)}
        redn_bounds: Dict[str, Tuple[ScalarExpression, ScalarExpression]] = {}
        args_as_pym_expr: List[prim.Subscript] = []
        namegen = UniqueNameGenerator(set(bindings))
        var_to_redn_descr = {}

        # {{{ add bindings coming from the shape expressions

        for access_descr, (iarg, arg) in zip(expr.access_descriptors,
                                            enumerate(expr.args)):
            subscript_indices = []
            for iaxis, axis in enumerate(access_descr):
                if not are_shape_components_equal(
                            arg.shape[iaxis],
                            expr._access_descr_to_axis_len()[axis]):
                    # axis is broadcasted
                    assert are_shape_components_equal(arg.shape[iaxis], 1)
                    subscript_indices.append(0)
                    continue

                if isinstance(axis, EinsumElementwiseAxis):
                    subscript_indices.append(prim.Variable(f"_{axis.dim}"))
                else:
                    assert isinstance(axis, EinsumReductionAxis)
                    redn_idx_name = f"_r{axis.dim}"
                    if redn_idx_name not in redn_bounds:
                        # convert the ShapeComponent to a ScalarExpression
                        redn_bound, redn_bound_bindings = (
                            dim_to_index_lambda_components(
                                arg.shape[iaxis], namegen))
                        redn_bounds[redn_idx_name] = (0, redn_bound)

                        bindings.update({k: self.rec(v)
                                         for k, v in redn_bound_bindings.items()})
                        var_to_redn_descr[redn_idx_name] = (
                            expr.redn_axis_to_redn_descr[axis])

                    subscript_indices.append(prim.Variable(redn_idx_name))

            args_as_pym_expr.append(prim.Subscript(prim.Variable(f"in{iarg}"),
                                                   tuple(subscript_indices)))

        # }}}

        inner_expr = reduce(operator.mul, args_as_pym_expr[1:],
                            args_as_pym_expr[0])

        if redn_bounds:
            from pytato.reductions import SumReductionOperation
            inner_expr = Reduce(inner_expr,
                                SumReductionOperation(),
                                redn_bounds)

        return IndexLambda(expr=inner_expr,
                           shape=tuple(self.rec(s) if isinstance(s, Array) else s
                                       for s in expr.shape),
                           dtype=expr.dtype,
                           bindings=bindings,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(var_to_redn_descr),
                           tags=expr.tags)

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
                bindings=dict(_in0=array),
                axes=expr.axes,
                var_to_reduction_descr=Map(),
                tags=expr.tags)

    def _indices_for_axis_permutation(self, expr: AxisPermutation) -> SymbolicIndex:
        indices = [None] * expr.ndim
        for from_index, to_index in enumerate(expr.axis_permutation):
            indices[to_index] = var(f"_{from_index}")
        return tuple(indices)

    def _indices_for_reshape(self, expr: Reshape) -> SymbolicIndex:
        if expr.array.shape == ():
            # RHS must be a scalar i.e. RHS' indices are empty
            assert expr.size == 1
            return ()

        newstrides: List[IntegralT] = [1]  # reshaped array strides
        for new_axis_len in reversed(expr.shape[1:]):
            assert isinstance(new_axis_len, INT_CLASSES)
            newstrides.insert(0, newstrides[0]*new_axis_len)

        flattened_idx = sum(prim.Variable(f"_{i}")*stride
                            for i, stride in enumerate(newstrides))

        oldstrides: List[IntegralT] = [1]  # input array strides
        for axis_len in reversed(expr.array.shape[1:]):
            assert isinstance(axis_len, INT_CLASSES)
            oldstrides.insert(0, oldstrides[0]*axis_len)

        assert isinstance(expr.array.shape[-1], INT_CLASSES)
        oldsizetills = [expr.array.shape[-1]]  # input array size till for axes idx
        for old_axis_len in reversed(expr.array.shape[:-1]):
            assert isinstance(old_axis_len, INT_CLASSES)
            oldsizetills.insert(0, oldsizetills[0]*old_axis_len)

        return tuple(((flattened_idx % sizetill) // stride)
                     for stride, sizetill in zip(oldstrides, oldsizetills))

    # https://github.com/python/mypy/issues/8619
    map_axis_permutation = (
            partialmethod(handle_index_remapping, _indices_for_axis_permutation))  # type: ignore  # noqa
    map_reshape = partialmethod(handle_index_remapping, _indices_for_reshape) #type: ignore # noqa

    # }}}

    def map_basic_index(self, expr: BasicIndex) -> IndexLambda:
        vng = UniqueNameGenerator()
        indices = []

        in_ary = vng("in")
        bindings = {in_ary: self.rec(expr.array)}
        islice_idx = 0

        for idx, axis_len in zip(expr.indices, expr.array.shape):
            if isinstance(idx, INT_CLASSES):
                if isinstance(axis_len, INT_CLASSES):
                    indices.append(idx % axis_len)
                else:
                    bnd_name = vng("in")
                    bindings[bnd_name] = axis_len
                    indices.append(idx % prim.Variable(bnd_name))
            elif isinstance(idx, NormalizedSlice):
                indices.append(idx.start
                               + idx.step * prim.Variable(f"_{islice_idx}"))
                islice_idx += 1
            else:
                raise NotImplementedError

        return IndexLambda(expr=prim.Subscript(prim.Variable(in_ary),
                                               tuple(indices)),
                           bindings=bindings,
                           shape=expr.shape,
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags,
                           )

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> IndexLambda:
        from pytato.utils import (get_shape_after_broadcasting,
                                  get_indexing_expression)

        i_adv_indices = tuple(i
                              for i, idx_expr in enumerate(expr.indices)
                              if isinstance(idx_expr, (Array, INT_CLASSES)))
        adv_idx_shape = get_shape_after_broadcasting([expr.indices[i_idx]
                                                      for i_idx in i_adv_indices])

        vng = UniqueNameGenerator()
        indices = []

        in_ary = vng("in")
        bindings = {in_ary: self.rec(expr.array)}
        islice_idx = 0

        for i_idx, (idx, axis_len) in enumerate(zip(expr.indices, expr.array.shape)):
            if isinstance(idx, INT_CLASSES):
                if isinstance(axis_len, INT_CLASSES):
                    indices.append(idx % axis_len)
                else:
                    bnd_name = vng("in")
                    bindings[bnd_name] = self.rec(axis_len)
                    indices.append(idx % prim.Variable(bnd_name))
            elif isinstance(idx, NormalizedSlice):
                indices.append(idx.start
                               + idx.step * prim.Variable(f"_{islice_idx}"))
                islice_idx += 1
            elif isinstance(idx, Array):
                if isinstance(axis_len, INT_CLASSES):
                    bnd_name = vng("in")
                    bindings[bnd_name] = self.rec(idx)
                    indirect_idx_expr = prim.Subscript(
                        prim.Variable(bnd_name),
                        get_indexing_expression(
                            idx.shape,
                            (1,)*i_adv_indices[0]+adv_idx_shape))

                    if not idx.tags_of_type(AssumeNonNegative):
                        # We define "upper" out-of bounds access to be undefined
                        # behavior.  (numpy raises an exception, too)
                        indirect_idx_expr = indirect_idx_expr % axis_len

                    indices.append(indirect_idx_expr)
                else:
                    raise NotImplementedError("Advanced indexing over"
                                              " parametric axis lengths.")
            else:
                raise NotImplementedError(f"Indices of type {type(idx)}.")

            if i_idx == i_adv_indices[-1]:
                islice_idx += len(adv_idx_shape)

        return IndexLambda(expr=prim.Subscript(prim.Variable(in_ary),
                                               tuple(indices)),
                           bindings=bindings,
                           shape=expr.shape,
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags,
                           )

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> IndexLambda:
        from pytato.utils import (get_shape_after_broadcasting,
                                  get_indexing_expression)
        i_adv_indices = tuple(i
                              for i, idx_expr in enumerate(expr.indices)
                              if isinstance(idx_expr, (Array, INT_CLASSES)))
        adv_idx_shape = get_shape_after_broadcasting([expr.indices[i_idx]
                                                      for i_idx in i_adv_indices])

        vng = UniqueNameGenerator()
        indices = []

        in_ary = vng("in")
        bindings = {in_ary: self.rec(expr.array)}

        islice_idx = len(adv_idx_shape)

        for idx, axis_len in zip(expr.indices, expr.array.shape):
            if isinstance(idx, INT_CLASSES):
                if isinstance(axis_len, INT_CLASSES):
                    indices.append(idx % axis_len)
                else:
                    bnd_name = vng("in")
                    bindings[bnd_name] = self.rec(axis_len)
                    indices.append(idx % prim.Variable(bnd_name))
            elif isinstance(idx, NormalizedSlice):
                indices.append(idx.start
                               + idx.step * prim.Variable(f"_{islice_idx}"))
                islice_idx += 1
            elif isinstance(idx, Array):
                if isinstance(axis_len, INT_CLASSES):
                    bnd_name = vng("in")
                    bindings[bnd_name] = self.rec(idx)

                    indirect_idx_expr = prim.Subscript(prim.Variable(bnd_name),
                                                       get_indexing_expression(
                                                           idx.shape,
                                                           adv_idx_shape))

                    if not idx.tags_of_type(AssumeNonNegative):
                        # We define "upper" out-of bounds access to be undefined
                        # behavior.  (numpy raises an exception, too)
                        indirect_idx_expr = indirect_idx_expr % axis_len

                    indices.append(indirect_idx_expr)
                else:
                    raise NotImplementedError("Advanced indexing over"
                                              " parametric axis lengths.")
            else:
                raise NotImplementedError(f"Indices of type {type(idx)}.")

        return IndexLambda(expr=prim.Subscript(prim.Variable(in_ary),
                                               tuple(indices)),
                           bindings=bindings,
                           shape=expr.shape,
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags,
                           )
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

class NamesValidityChecker(CachedWalkMapper):
    def __init__(self) -> None:
        self.name_to_input: Dict[str, InputArgumentBase] = {}
        super().__init__()

    def post_visit(self, expr: Any) -> None:
        if isinstance(expr, (Placeholder, SizeParam, DataWrapper)):
            if expr.name is not None:
                try:
                    ary = self.name_to_input[expr.name]
                except KeyError:
                    self.name_to_input[expr.name] = expr
                else:
                    if ary is not expr:
                        from pytato.diagnostic import NameClashError
                        raise NameClashError(
                                "Received two separate instances of inputs "
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
    from pytato.transform import copy_dict_of_named_arrays

    check_validity_of_outputs(outputs)

    # {{{ compute the order in which the outputs must be computed

    # semantically order does not matter, but doing a toposort ordering of the
    # outputs leads to a FLOP optimal choice

    from pytools.graph import compute_topological_order

    get_deps = SubsetDependencyMapper(frozenset(out.expr
                                                for out in outputs.values()))

    # only look for dependencies between the outputs
    deps = {name: get_deps(output.expr)
            for name, output in outputs.items()}

    # represent deps in terms of output names
    output_expr_to_name = {output.expr: name for name, output in outputs.items()}
    dag = {name: (frozenset([output_expr_to_name[output] for output in val])
                  - frozenset([name]))
           for name, val in deps.items()}

    output_order: List[str] = compute_topological_order(dag, key=lambda x: x)[::-1]

    # }}}

    mapper = CodeGenPreprocessor(target)

    new_outputs = copy_dict_of_named_arrays(outputs, mapper)

    return PreprocessResult(outputs=new_outputs,
            compute_order=tuple(output_order),
            bound_arguments=mapper.bound_arguments)

# vim: fdm=marker
