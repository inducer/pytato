"""
.. currentmodule:: pytato.transform.lower_to_index_lambda

.. autofunction:: to_index_lambda
"""
from __future__ import annotations


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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from immutabledict import immutabledict
from typing_extensions import Never

import pymbolic.primitives as prim
from pymbolic import ArithmeticExpression
from pytools import UniqueNameGenerator

from pytato.array import (
    AbstractResultWithNamedArrays,
    AdvancedIndexInContiguousAxes,
    AdvancedIndexInNoncontiguousAxes,
    Array,
    AxisPermutation,
    BasicIndex,
    Concatenate,
    Einsum,
    IndexExpr,
    IndexLambda,
    NormalizedSlice,
    Reshape,
    Roll,
    ShapeComponent,
    ShapeType,
    Stack,
    _entries_are_identical,
    _get_einsum_access_descr_to_axis_len,
)
from pytato.diagnostic import CannotBeLoweredToIndexLambda
from pytato.scalar_expr import INT_CLASSES, ScalarExpression
from pytato.tags import AssumeNonNegative
from pytato.transform import (
    Mapper,
    _verify_is_array,
)
from pytato.utils import normalized_slice_does_not_change_axis


if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


ToIndexLambdaT = TypeVar("ToIndexLambdaT", Array, AbstractResultWithNamedArrays)


@dataclass(frozen=True)
class _ReshapeShapeGroup:
    old_ax_shape_group: tuple[ShapeComponent, ...]
    new_ax_shape_group: tuple[ShapeComponent, ...]


def _generate_index_expressions(
        old_shape: ShapeType,
        new_shape: ShapeType,
        order: str,
        index_vars: list[prim.Variable]) -> tuple[ScalarExpression, ...]:

    old_strides: list[ArithmeticExpression] = [1]
    new_strides: list[ArithmeticExpression] = [1]
    old_strides = old_strides[:len(old_shape)]
    new_strides = new_strides[:len(new_shape)]

    if not old_shape:
        assert new_shape == (1,)
        return (0,)

    if old_shape == new_shape:
        # Avoid generating modulo expressions for direct pass-through
        assert len(old_shape) == 1
        return (index_vars[0],)

    old_size_tills = [old_shape[-1] if order == "C" else old_shape[0]]

    old_stride_axs = (old_shape[::-1][:-1] if order == "C" else
                      old_shape[:-1])
    for old_ax in old_stride_axs:
        old_strides.append(old_strides[-1]*old_ax)

    new_stride_axs = (new_shape[::-1][:-1] if order == "C" else
                      new_shape[:-1])
    for new_ax in new_stride_axs:
        new_strides.append(new_strides[-1]*new_ax)

    old_size_till_axs = (old_shape[:-1][::-1] if order == "C" else
                         old_shape[1:])
    for old_ax in old_size_till_axs:
        old_size_tills.append(old_size_tills[-1]*old_ax)

    if order == "C":
        old_strides = old_strides[::-1]
        new_strides = new_strides[::-1]
        old_size_tills = old_size_tills[::-1]

    flattened_index_expn = sum(
        index_var*new_stride
        for index_var, new_stride in zip(index_vars, new_strides, strict=True))

    return tuple(
        # Mypy has a point: complex numbers don't support '//'.
        (flattened_index_expn % old_size_till) // old_stride  # type: ignore[operator]
        for old_size_till, old_stride in zip(old_size_tills, old_strides, strict=True))


def _get_reshaped_indices(
        order: str, old_shape: ShapeType, new_shape: ShapeType
    ) -> tuple[ScalarExpression, ...]:

    if order.upper() not in ["C", "F"]:
        raise NotImplementedError("Order expected to be 'C' or 'F'",
                                  " (case insensitive). Found order = ",
                                  f"{order}")

    # index variables need to be unique and depend on the new shape length
    index_vars = [prim.Variable(f"_{i}") for i in range(len(new_shape))]

    # {{{ check for scalars

    if old_shape == ():
        from pytools import product
        assert product(new_shape) == 1
        return ()

    if new_shape == ():
        return _generate_index_expressions(old_shape, new_shape, order,
                                           index_vars)

    if 0 in old_shape and 0 in new_shape:
        return _generate_index_expressions(old_shape, new_shape, order,
                                           index_vars)

    # }}}

    # {{{ generate subsets of old axes mapped to subsets of new axes

    axis_mapping: list[_ReshapeShapeGroup] = []

    old_index = 0
    new_index = 0

    while old_index < len(old_shape) and new_index < len(new_shape):
        old_ax_len_product = old_shape[old_index]
        new_ax_len_product = new_shape[new_index]

        # Specially handle (i.e. skip) axes of length 1 at the start of an index group
        if old_ax_len_product != new_ax_len_product:
            if old_ax_len_product == 1:
                axis_mapping.append(_ReshapeShapeGroup(
                    old_ax_shape_group=(old_ax_len_product,),
                    new_ax_shape_group=()))
                old_index += 1
                continue
            if new_ax_len_product == 1:
                axis_mapping.append(_ReshapeShapeGroup(
                    old_ax_shape_group=(),
                    new_ax_shape_group=(new_ax_len_product,)))
                new_index += 1
                continue

        old_product_end = old_index + 1
        new_product_end = new_index + 1

        while old_ax_len_product != new_ax_len_product:
            if not isinstance(old_ax_len_product, INT_CLASSES) or \
                not isinstance(new_ax_len_product, INT_CLASSES):
                raise TypeError("Cannot determine which axes were expanded or "
                                "collapsed symbolically")

            if new_ax_len_product < old_ax_len_product:
                new_ax_len_product *= new_shape[new_product_end]
                new_product_end += 1
            else:
                old_ax_len_product *= old_shape[old_product_end]
                old_product_end += 1

        axis_mapping.append(_ReshapeShapeGroup(
            old_ax_shape_group=old_shape[old_index:old_product_end],
            new_ax_shape_group=new_shape[new_index:new_product_end]))

        old_index = old_product_end
        new_index = new_product_end

    # handle trailing 1s

    # At most one of the while loops below should execute.
    assert not (
        old_index < len(old_shape)
        and
        new_index < len(new_shape)
    )

    while old_index < len(old_shape):
        assert old_shape[old_index] == 1
        axis_mapping.append(_ReshapeShapeGroup(
            old_ax_shape_group=(old_shape[old_index],),
            new_ax_shape_group=()))
        old_index += 1

    while new_index < len(new_shape):
        assert new_shape[new_index] == 1
        axis_mapping.append(_ReshapeShapeGroup(
                                old_ax_shape_group=(),
                                new_ax_shape_group=(new_shape[new_index],),
                            ))
        new_index += 1

    # }}}

    # {{{ compute index expressions for sub shapes

    index_vars_begin = 0
    index_expressions = []
    for shape_group in axis_mapping:
        sub_old_shape = shape_group.old_ax_shape_group
        sub_new_shape = shape_group.new_ax_shape_group

        index_vars_end = index_vars_begin + len(sub_new_shape)
        sub_index_vars = index_vars[index_vars_begin:index_vars_end]
        index_vars_begin = index_vars_end

        if not sub_old_shape:
            # No need to generate index into old array
            assert sub_new_shape == (1,)
        else:
            index_expressions.append(_generate_index_expressions(
                sub_old_shape, sub_new_shape, order, sub_index_vars))

    # }}}

    return sum(index_expressions, ())


class ToIndexLambdaMixin:
    def rec_size_tuple(self, situp: ShapeType) -> ShapeType:
        new_situp = tuple(
            _verify_is_array(self.rec(s)) if isinstance(s, Array) else s
            for s in situp)
        return situp if _entries_are_identical(new_situp, situp) else new_situp

    def rec_idx_tuple(self, situp: tuple[IndexExpr, ...]) -> tuple[IndexExpr, ...]:
        new_situp = tuple(
            _verify_is_array(self.rec(s)) if isinstance(s, Array) else s
            for s in situp)
        return situp if _entries_are_identical(new_situp, situp) else new_situp

    if TYPE_CHECKING:
        def rec(
                self, expr: ToIndexLambdaT, *args: Any,
                **kwargs: Any) -> ToIndexLambdaT:
            # type-ignore-reason: mypy is right as we are attempting to make
            # guarantees about other super-classes.
            return super().rec(  # type: ignore[no-any-return,misc]
                expr, *args, **kwargs)

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        new_shape = self.rec_size_tuple(expr.shape)
        new_bindings: Mapping[str, Array] = immutabledict({
                name: _verify_is_array(self.rec(subexpr))
                for name, subexpr in sorted(expr.bindings.items())})
        return expr.replace_if_different(shape=new_shape, bindings=new_bindings)

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
                from pymbolic.primitives import Comparison, If
                stack_expr = If(Comparison(prim.Variable(f"_{expr.axis}"), "==", i),
                        subarray_expr,
                        stack_expr)

        bindings = {f"_in{i}": self.rec(array)
                    for i, array in enumerate(expr.arrays)}

        return IndexLambda(expr=stack_expr,
                           shape=self.rec_size_tuple(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           bindings=immutabledict(bindings),
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate) -> IndexLambda:
        from pymbolic.primitives import Comparison, If, Subscript

        def get_subscript(array_index: int, offset: ScalarExpression) -> Subscript:
            aggregate = prim.Variable(f"_in{array_index}")
            index = [prim.Variable(f"_{i}")
                     if i != expr.axis
                     else (prim.Variable(f"_{i}") - offset)
                     for i in range(len(expr.shape))]
            return Subscript(aggregate, tuple(index))

        rec_arrays: tuple[Array, ...] = tuple(self.rec(ary) for ary in expr.arrays)

        lbounds: list[Any] = [0]
        ubounds: list[Any] = [rec_arrays[0].shape[expr.axis]]

        for i, array in enumerate(rec_arrays[1:], start=1):
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
                concat_expr: ArithmeticExpression = subarray_expr
            else:
                concat_expr = If(Comparison(prim.Variable(f"_{expr.axis}"),
                                            "<", ubound),
                                 subarray_expr,
                                 concat_expr)

        bindings = {f"_in{i}": array
                    for i, array in enumerate(rec_arrays)}

        return IndexLambda(expr=concat_expr,
                           shape=self.rec_size_tuple(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict(bindings),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_einsum(self, expr: Einsum) -> IndexLambda:
        import operator
        from functools import reduce

        from pytato.array import EinsumElementwiseAxis, EinsumReductionAxis
        from pytato.scalar_expr import Reduce
        from pytato.utils import (
            are_shape_components_equal,
            dim_to_index_lambda_components,
        )

        rec_args: tuple[Array, ...] = tuple(self.rec(arg) for arg in expr.args)

        bindings = {f"_in{k}": arg for k, arg in enumerate(rec_args)}
        redn_bounds: dict[str, tuple[ScalarExpression, ScalarExpression]] = {}
        args_as_pym_expr: list[prim.Subscript] = []
        namegen = UniqueNameGenerator(set(bindings))
        var_to_redn_descr = {}

        # {{{ add bindings coming from the shape expressions

        access_descr_to_axis_len = _get_einsum_access_descr_to_axis_len(
            expr.access_descriptors, rec_args)

        for access_descr, (iarg, arg) in zip(expr.access_descriptors,
                                            enumerate(rec_args), strict=True):
            subscript_indices: list[ArithmeticExpression] = []
            for iaxis, axis in enumerate(access_descr):
                if not are_shape_components_equal(
                            arg.shape[iaxis],
                            access_descr_to_axis_len[axis]):
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

            args_as_pym_expr.append(prim.Subscript(prim.Variable(f"_in{iarg}"),
                                                   tuple(subscript_indices)))

        # }}}

        inner_expr: ArithmeticExpression = reduce(
                operator.mul, args_as_pym_expr[1:],
                args_as_pym_expr[0])

        if redn_bounds:
            from pytato.reductions import SumReductionOperation
            inner_expr = Reduce(inner_expr,
                                SumReductionOperation(),
                                immutabledict(redn_bounds))

        return IndexLambda(expr=inner_expr,
                           shape=self.rec_size_tuple(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict(bindings),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(var_to_redn_descr),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_roll(self, expr: Roll) -> IndexLambda:
        from pytato.utils import dim_to_index_lambda_components

        rec_array = self.rec(expr.array)

        index_expr: prim.ExpressionNode = prim.Variable("_in0")
        indices: list[ArithmeticExpression] = [
            prim.Variable(f"_{d}") for d in range(expr.ndim)]
        axis = expr.axis
        axis_len_expr, bindings = dim_to_index_lambda_components(
            rec_array.shape[axis],
            UniqueNameGenerator({"_in0"}))

        # Mypy has a point: the type system does not prove that the operands are
        # not complex-valued or bool.
        indices[axis] = (indices[axis] - expr.shift) % axis_len_expr  # type: ignore[operator, call-overload]

        if indices:
            index_expr = index_expr[tuple(indices)]

        # type-ignore-reason: `bindings` was returned as Dict[str, SizeParam]
        bindings["_in0"] = rec_array  # type: ignore[assignment]

        return IndexLambda(expr=index_expr,
                           shape=self.rec_size_tuple(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict(bindings),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> IndexLambda:
        from pytato.utils import get_indexing_expression, get_shape_after_broadcasting

        rec_array = self.rec(expr.array)
        rec_indices = self.rec_idx_tuple(expr.indices)

        i_adv_indices = tuple(i
                              for i, idx_expr in enumerate(rec_indices)
                              if isinstance(idx_expr, (Array, *INT_CLASSES)))
        adv_idx_shape = get_shape_after_broadcasting([
                    cast("Array | int | np.integer[Any]", rec_indices[i_idx])
                    for i_idx in i_adv_indices])

        vng = UniqueNameGenerator()
        indices: list[ArithmeticExpression] = []
        in_ary = vng("in")
        bindings = {in_ary: rec_array}
        islice_idx = 0

        for i_idx, (idx, axis_len) in enumerate(
                        zip(rec_indices, rec_array.shape, strict=True)):
            if isinstance(idx, INT_CLASSES):
                if isinstance(axis_len, INT_CLASSES):
                    indices.append(idx % axis_len)
                else:
                    bnd_name = vng("in")
                    bindings[bnd_name] = axis_len
                    indices.append(idx % prim.Variable(bnd_name))
            elif isinstance(idx, NormalizedSlice):
                if normalized_slice_does_not_change_axis(idx, axis_len):
                    indices.append(prim.Variable(f"_{islice_idx}"))
                else:
                    indices.append(idx.start
                               + idx.step * prim.Variable(f"_{islice_idx}"))
                islice_idx += 1
            elif isinstance(idx, Array):
                if isinstance(axis_len, INT_CLASSES):
                    bnd_name = vng("in")
                    bindings[bnd_name] = idx
                    indirect_idx_expr: ArithmeticExpression = prim.Subscript(
                        prim.Variable(bnd_name),
                        get_indexing_expression(
                            idx.shape,
                            (1,)*i_adv_indices[0]+adv_idx_shape))

                    if not idx.tags_of_type(AssumeNonNegative):
                        # We define "upper" out-of bounds access to be undefined
                        # behavior.  (numpy raises an exception, too)

                        # Mypy has a point: the type system does not prove that the
                        # operands are not complex-valued.
                        indirect_idx_expr = indirect_idx_expr % axis_len  # type: ignore[operator]

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
                           bindings=immutabledict(bindings),
                           shape=self.rec_size_tuple(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags,
                           )

    def map_non_contiguous_advanced_index(
            self, expr: AdvancedIndexInNoncontiguousAxes) -> IndexLambda:
        from pytato.utils import get_indexing_expression, get_shape_after_broadcasting

        rec_array = self.rec(expr.array)
        rec_indices = self.rec_idx_tuple(expr.indices)

        i_adv_indices = tuple(i
                              for i, idx_expr in enumerate(rec_indices)
                              if isinstance(idx_expr, (Array, *INT_CLASSES)))
        adv_idx_shape = get_shape_after_broadcasting([
            cast("Array | int | np.integer[Any]", rec_indices[i_idx])
            for i_idx in i_adv_indices])

        vng = UniqueNameGenerator()
        indices: list[ArithmeticExpression] = []

        in_ary = vng("in")
        bindings = {in_ary: rec_array}

        islice_idx = len(adv_idx_shape)

        for idx, axis_len in zip(rec_indices, rec_array.shape, strict=True):
            if isinstance(idx, INT_CLASSES):
                if isinstance(axis_len, INT_CLASSES):
                    indices.append(idx % axis_len)
                else:
                    bnd_name = vng("in")
                    bindings[bnd_name] = axis_len
                    indices.append(idx % prim.Variable(bnd_name))
            elif isinstance(idx, NormalizedSlice):
                if normalized_slice_does_not_change_axis(idx, axis_len):
                    indices.append(prim.Variable(f"_{islice_idx}"))
                else:
                    indices.append(idx.start
                               + idx.step * prim.Variable(f"_{islice_idx}"))
                islice_idx += 1
            elif isinstance(idx, Array):
                if isinstance(axis_len, INT_CLASSES):
                    bnd_name = vng("in")
                    bindings[bnd_name] = idx

                    indirect_idx_expr: ArithmeticExpression = prim.Subscript(
                                                        prim.Variable(bnd_name),
                                                        get_indexing_expression(
                                                            idx.shape,
                                                            adv_idx_shape))

                    if not idx.tags_of_type(AssumeNonNegative):
                        # We define "upper" out-of bounds access to be undefined
                        # behavior.  (numpy raises an exception, too)

                        # Mypy has a point: complex numbers do not have '%'.
                        indirect_idx_expr = indirect_idx_expr % axis_len  # type: ignore[operator]

                    indices.append(indirect_idx_expr)
                else:
                    raise NotImplementedError("Advanced indexing over"
                                              " parametric axis lengths.")
            else:
                raise NotImplementedError(f"Indices of type {type(idx)}.")

        return IndexLambda(expr=prim.Subscript(prim.Variable(in_ary),
                                               tuple(indices)),
                           bindings=immutabledict(bindings),
                           shape=self.rec_size_tuple(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags,
                           )

    def map_basic_index(self, expr: BasicIndex) -> IndexLambda:
        rec_array = self.rec(expr.array)
        rec_indices = self.rec_idx_tuple(expr.indices)

        vng = UniqueNameGenerator()
        indices: list[ArithmeticExpression] = []

        in_ary = vng("in")
        bindings = {in_ary: rec_array}
        islice_idx = 0

        for idx, axis_len in zip(rec_indices, rec_array.shape, strict=True):
            if isinstance(idx, INT_CLASSES):
                if isinstance(axis_len, INT_CLASSES):
                    indices.append(idx % axis_len)
                else:
                    bnd_name = vng("in")
                    bindings[bnd_name] = axis_len
                    indices.append(idx % prim.Variable(bnd_name))
            elif isinstance(idx, NormalizedSlice):
                if normalized_slice_does_not_change_axis(idx, axis_len):
                    indices.append(prim.Variable(f"_{islice_idx}"))
                else:
                    indices.append(idx.start
                               + idx.step * prim.Variable(f"_{islice_idx}"))
                islice_idx += 1
            else:
                raise NotImplementedError

        return IndexLambda(expr=prim.Subscript(prim.Variable(in_ary),
                                               tuple(indices)),
                           bindings=immutabledict(bindings),
                           shape=self.rec_size_tuple(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags,
                           )

    def map_reshape(self, expr: Reshape) -> IndexLambda:
        rec_array = self.rec(expr.array)
        rec_newshape = self.rec_size_tuple(expr.shape)
        indices = _get_reshaped_indices(expr.order, rec_array.shape, rec_newshape)
        index_expr = prim.Variable("_in0")[tuple(indices)]
        return IndexLambda(expr=index_expr,
                           shape=rec_newshape,
                           dtype=expr.dtype,
                           bindings=immutabledict({"_in0": rec_array}),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> IndexLambda:
        rec_array = self.rec(expr.array)

        indices: list[ArithmeticExpression | None] = [None] * expr.ndim
        for from_index, to_index in enumerate(expr.axis_permutation):
            indices[to_index] = prim.Variable(f"_{from_index}")

        index_expr = prim.Variable("_in0")[
            cast("tuple[ArithmeticExpression]", tuple(indices))]

        return IndexLambda(expr=index_expr,
                           shape=self.rec_size_tuple(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict({"_in0": rec_array}),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)


class ToIndexLambdaMapper(Mapper[Array, Never, []], ToIndexLambdaMixin):

    def handle_unsupported_array(self, expr: Array) -> Array:
        raise CannotBeLoweredToIndexLambda(type(expr))

    def rec(self, expr: Array) -> Array:  # type: ignore[override]
        return expr

    def __call__(self, expr: Array) -> Array:  # type: ignore[override]
        return Mapper.rec(self, expr)


def to_index_lambda(expr: Array) -> IndexLambda:
    """
    Lowers *expr* to :class:`~pytato.array.IndexLambda` if possible, otherwise
    raises a :class:`pytato.diagnostic.CannotBeLoweredToIndexLambda`.

    :returns: The lowered :class:`~pytato.array.IndexLambda`.
    """
    res = ToIndexLambdaMapper()(expr)
    assert isinstance(res, IndexLambda)
    return res

# vim:fdm=marker
