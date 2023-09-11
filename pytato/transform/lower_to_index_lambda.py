"""
.. currentmodule:: pytato.transform.lower_to_index_lambda

.. autofunction:: to_index_lambda
"""

__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

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

import pymbolic.primitives as prim

from typing import List, Any, Dict, Tuple, TypeVar
from immutables import Map
from pytools import UniqueNameGenerator
from pytato.array import (Array, IndexLambda, Stack, Concatenate,
                          Einsum, Reshape, Roll, AxisPermutation,
                          BasicIndex, AdvancedIndexInContiguousAxes,
                          AdvancedIndexInNoncontiguousAxes,
                          NormalizedSlice, ShapeType,
                          AbstractResultWithNamedArrays)
from pytato.scalar_expr import ScalarExpression, INT_CLASSES, IntegralT
from pytato.diagnostic import CannotBeLoweredToIndexLambda
from pytato.tags import AssumeNonNegative
from pytato.transform import Mapper

ToIndexLambdaT = TypeVar("ToIndexLambdaT", Array, AbstractResultWithNamedArrays)


def _get_reshaped_indices(expr: Reshape) -> Tuple[ScalarExpression, ...]:
    if expr.array.shape == ():
        # RHS must be a scalar i.e. RHS' indices are empty
        assert expr.size == 1
        return ()

    if expr.order not in ["C", "F"]:
        raise NotImplementedError("Order expected to be 'C' or 'F'",
                                  f" found {expr.order}")

    if expr.order == "C":
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
        oldsizetills = [expr.array.shape[-1]]  # input array size
                                               # till for axes idx
        for old_axis_len in reversed(expr.array.shape[:-1]):
            assert isinstance(old_axis_len, INT_CLASSES)
            oldsizetills.insert(0, oldsizetills[0]*old_axis_len)

    else:
        newstrides: List[IntegralT] = [1]  # reshaped array strides
        for new_axis_len in expr.shape[:-1]:
            assert isinstance(new_axis_len, INT_CLASSES)
            newstrides.append(newstrides[-1]*new_axis_len)

        flattened_idx = sum(prim.Variable(f"_{i}")*stride
                            for i, stride in enumerate(newstrides))

        oldstrides: List[IntegralT] = [1]  # input array strides
        for axis_len in expr.array.shape[:-1]:
            assert isinstance(axis_len, INT_CLASSES)
            oldstrides.append(oldstrides[-1]*axis_len)

        assert isinstance(expr.array.shape[0], INT_CLASSES)
        oldsizetills = [expr.array.shape[0]]  # input array size till for axes idx
        for old_axis_len in expr.array.shape[1:]:
            assert isinstance(old_axis_len, INT_CLASSES)
            oldsizetills.append(oldsizetills[-1]*old_axis_len)

    return tuple(((flattened_idx % sizetill) // stride)
                 for stride, sizetill in zip(oldstrides, oldsizetills))


class ToIndexLambdaMixin:
    def _rec_shape(self, shape: ShapeType) -> ShapeType:
        return tuple(self.rec(s) if isinstance(s, Array)
                     else s
                     for s in shape)

    def rec(self, expr: ToIndexLambdaT, *args: Any, **kwargs: Any) -> ToIndexLambdaT:
        # type-ignore-reason: mypy is right as we are attempting to make
        # guarantees about other super-classes.
        return super().rec(expr, *args, **kwargs)  # type: ignore[no-any-return,misc]

    def map_index_lambda(self, expr: IndexLambda) -> IndexLambda:
        return IndexLambda(expr=expr.expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings={name: self.rec(bnd)
                                     for name, bnd in expr.bindings.items()},
                           axes=expr.axes,
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           tags=expr.tags)

    def map_stack(self, expr: Stack) -> IndexLambda:
        subscript = tuple(prim.Variable(f"_{i}")
                          for i in range(expr.ndim)
                          if i != expr.axis)

        # I = axis index
        #
        # => If(_I == 0,
        #        _in0[_0, _1, ...],
        #        If(_I == 1,
        #            _in1[_0, _1, ...],
        #            ...
        #                _inNm1[_0, _1, ...] ...))
        for i in range(len(expr.arrays) - 1, -1, -1):
            subarray_expr = prim.Variable(f"_in{i}")[subscript]
            if i == len(expr.arrays) - 1:
                stack_expr = subarray_expr
            else:
                from pymbolic.primitives import If, Comparison
                stack_expr = If(Comparison(prim.Variable(f"_{expr.axis}"), "==", i),
                        subarray_expr,
                        stack_expr)

        bindings = {f"_in{i}": self.rec(array)
                    for i, array in enumerate(expr.arrays)}

        return IndexLambda(expr=stack_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           bindings=bindings,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags)

    def map_concatenate(self, expr: Concatenate) -> IndexLambda:
        from pymbolic.primitives import If, Comparison, Subscript

        def get_subscript(array_index: int, offset: ScalarExpression) -> Subscript:
            aggregate = prim.Variable(f"_in{array_index}")
            index = [prim.Variable(f"_{i}")
                     if i != expr.axis
                     else (prim.Variable(f"_{i}") - offset)
                     for i in range(len(expr.shape))]
            return Subscript(aggregate, tuple(index))

        lbounds: List[Any] = [0]
        ubounds: List[Any] = [expr.arrays[0].shape[expr.axis]]

        for i, array in enumerate(expr.arrays[1:], start=1):
            ubounds.append(ubounds[i-1]+array.shape[expr.axis])
            lbounds.append(ubounds[i-1])

        # I = axis index
        #
        # => If(_I < arrays[0].shape[axis],
        #        _in0[_0, _1, ..., _I, ...],
        #        If(_I < (arrays[1].shape[axis]+arrays[0].shape[axis]),
        #            _in1[_0, _1, ..., _I-arrays[0].shape[axis], ...],
        #            ...
        #                _inNm1[_0, _1, ...] ...))
        for i in range(len(expr.arrays) - 1, -1, -1):
            lbound, ubound = lbounds[i], ubounds[i]
            subarray_expr = get_subscript(i, lbound)
            if i == len(expr.arrays) - 1:
                concat_expr = subarray_expr
            else:
                concat_expr = If(Comparison(prim.Variable(f"_{expr.axis}"),
                                            "<", ubound),
                                 subarray_expr,
                                 concat_expr)

        bindings = {f"_in{i}": self.rec(array)
                    for i, array in enumerate(expr.arrays)}

        return IndexLambda(expr=concat_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings=bindings,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags)

    def map_einsum(self, expr: Einsum) -> IndexLambda:
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
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings=bindings,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(var_to_redn_descr),
                           tags=expr.tags)

    def map_roll(self, expr: Roll) -> IndexLambda:
        from pytato.utils import dim_to_index_lambda_components

        index_expr = prim.Variable("_in0")
        indices = [prim.Variable(f"_{d}") for d in range(expr.ndim)]
        axis = expr.axis
        axis_len_expr, bindings = dim_to_index_lambda_components(
            expr.shape[axis],
            UniqueNameGenerator({"_in0"}))

        indices[axis] = (indices[axis] - expr.shift) % axis_len_expr

        if indices:
            index_expr = index_expr[tuple(indices)]

        # type-ignore-reason: `bindings` was returned as Dict[str, SizeParam]
        bindings["_in0"] = expr.array  # type: ignore[assignment]

        return IndexLambda(expr=index_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings={name: self.rec(bnd)
                                     for name, bnd in bindings.items()},
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags)

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
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags,
                           )

    def map_non_contiguous_advanced_index(
            self, expr: AdvancedIndexInNoncontiguousAxes) -> IndexLambda:
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
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags,
                           )

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
                    bindings[bnd_name] = self.rec(axis_len)
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
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags,
                           )

    def map_reshape(self, expr: Reshape) -> IndexLambda:
        indices = _get_reshaped_indices(expr)
        index_expr = prim.Variable("_in0")[tuple(indices)]
        return IndexLambda(expr=index_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings={"_in0": self.rec(expr.array)},
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> IndexLambda:
        indices = [None] * expr.ndim
        for from_index, to_index in enumerate(expr.axis_permutation):
            indices[to_index] = prim.Variable(f"_{from_index}")

        index_expr = prim.Variable("_in0")[tuple(indices)]

        return IndexLambda(expr=index_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings={"_in0": self.rec(expr.array)},
                           axes=expr.axes,
                           var_to_reduction_descr=Map(),
                           tags=expr.tags)


class ToIndexLambdaMapper(Mapper, ToIndexLambdaMixin):

    def handle_unsupported_array(self, expr: Any) -> Any:  # type: ignore[override]
        raise CannotBeLoweredToIndexLambda(type(expr))

    def rec(self, expr: Array) -> Array:  # type: ignore[override]
        return expr

    __call__ = Mapper.rec


def to_index_lambda(expr: Array) -> IndexLambda:
    """
    Lowers *expr* to :class:`~pytato.array.IndexLambda` if possible, otherwise
    raises a :class:`pytato.diagnostic.CannotBeLoweredToIndexLambda`.

    :returns: The lowered :class:`~pytato.array.IndexLambda`.
    """
    return ToIndexLambdaMapper()(expr)  # type: ignore[no-any-return]

# vim:fdm=marker
