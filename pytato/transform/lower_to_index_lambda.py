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
from typing import TYPE_CHECKING, Any, Never, TypeVar, cast

from immutabledict import immutabledict

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
    IndexLambda,
    NormalizedSlice,
    Reshape,
    Roll,
    ShapeComponent,
    ShapeType,
    Stack,
)
from pytato.diagnostic import CannotBeLoweredToIndexLambda
from pytato.scalar_expr import INT_CLASSES, ScalarExpression
from pytato.tags import AssumeNonNegative
from pytato.transform import Mapper


if TYPE_CHECKING:
    import numpy as np


ToIndexLambdaT = TypeVar("ToIndexLambdaT", Array, AbstractResultWithNamedArrays)


@dataclass(frozen=True)
class _ReshapeIndexGroup:
    old_ax_indices: tuple[ShapeComponent, ...]
    new_ax_indices: tuple[ShapeComponent, ...]


def _generate_index_expressions(
        old_shape: ShapeType,
        new_shape: ShapeType,
        order: str,
        index_vars: list[prim.Variable]) -> tuple[ScalarExpression, ...]:

    old_strides: list[ArithmeticExpression] = [1]
    new_strides: list[ArithmeticExpression] = [1]
    old_strides = old_strides[:len(old_shape)]
    new_strides = new_strides[:len(new_shape)]

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
            for old_size_till, old_stride in zip(old_size_tills,
                                                 old_strides,
                                                 strict=True))


def _get_reshaped_indices(expr: Reshape) -> tuple[ScalarExpression, ...]:

    if expr.order.upper() not in ["C", "F"]:
        raise NotImplementedError("Order expected to be 'C' or 'F'",
                                  " (case insensitive). Found order = ",
                                  f"{expr.order}")

    order = expr.order
    old_shape = expr.array.shape
    new_shape = expr.shape

    # index variables need to be unique and depend on the new shape length
    index_vars = [prim.Variable(f"_{i}") for i in range(len(new_shape))]

    # {{{ check for scalars

    if old_shape == ():
        assert expr.size == 1
        return ()

    if new_shape == ():
        return _generate_index_expressions(old_shape, new_shape, order,
                                           index_vars)

    if 0 in old_shape and 0 in new_shape:
        return _generate_index_expressions(old_shape, new_shape, order,
                                           index_vars)

    # }}}

    # {{{ generate subsets of old axes mapped to subsets of new axes

    axis_mapping: list[_ReshapeIndexGroup] = []

    old_index = 0
    new_index = 0

    while old_index < len(old_shape) and new_index < len(new_shape):
        old_ax_len_product = old_shape[old_index]
        new_ax_len_product = new_shape[new_index]

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

        old_ax_indices = old_shape[old_index:old_product_end]
        new_ax_indices = new_shape[new_index:new_product_end]

        axis_mapping.append(_ReshapeIndexGroup(
            old_ax_indices=old_ax_indices,
            new_ax_indices=new_ax_indices))

        old_index = old_product_end
        new_index = new_product_end

    # handle trailing 1s
    final_reshaped_indices = axis_mapping.pop(-1)
    old_ax_indices = final_reshaped_indices.old_ax_indices
    new_ax_indices = final_reshaped_indices.new_ax_indices

    while old_index < len(old_shape):
        old_ax_indices += tuple([old_shape[old_index]])  # noqa: C409
        old_index += 1

    while new_index < len(new_shape):
        new_ax_indices += tuple([new_shape[new_index]])  # noqa: C409
        new_index += 1

    axis_mapping.append(_ReshapeIndexGroup(old_ax_indices=old_ax_indices,
                                           new_ax_indices=new_ax_indices))

    # }}}

    # {{{ compute index expressions for sub shapes

    index_vars_begin = 0
    index_expressions = []
    for reshaped_indices in axis_mapping:
        sub_old_shape = reshaped_indices.old_ax_indices
        sub_new_shape = reshaped_indices.new_ax_indices

        index_vars_end = index_vars_begin + len(sub_new_shape)
        sub_index_vars = index_vars[index_vars_begin:index_vars_end]
        index_vars_begin = index_vars_end

        sub_exprs: tuple[ScalarExpression, ...] = ()
        nind = 0
        oind = 0
        while nind < len(sub_new_shape) and oind < len(sub_old_shape):
            if sub_new_shape[nind] == sub_old_shape[oind]:
                sub_exprs = (*sub_exprs, sub_index_vars[nind])
                nind += 1
                oind += 1
            elif sub_new_shape[nind] == 1:
                nind += 1
            elif sub_old_shape[oind] == 1:
                sub_exprs = (*sub_exprs, 0)  # Only one element.
                oind += 1
            else:
                # Generate the rest of the expressions.
                sub_exprs = (*sub_exprs,
                             *_generate_index_expressions(sub_old_shape[oind:],
                                                  sub_new_shape[nind:], order,
                                                  sub_index_vars[nind:]))
                break
        if len(sub_exprs) < len(sub_old_shape):
            tmp = _generate_index_expressions(sub_old_shape[oind:],
                                              sub_new_shape[nind-1:], order,
                                              sub_index_vars[nind-1:])

            sub_exprs = (*sub_exprs, *tmp)
        index_expressions.append(sub_exprs)

    # }}}

    return sum(index_expressions, ())


class ToIndexLambdaMixin:
    def _rec_shape(self, shape: ShapeType) -> ShapeType:
        return tuple(self.rec(s) if isinstance(s, Array)
                     else s
                     for s in shape)

    if TYPE_CHECKING:
        def rec(
                self, expr: ToIndexLambdaT, *args: Any,
                **kwargs: Any) -> ToIndexLambdaT:
            # type-ignore-reason: mypy is right as we are attempting to make
            # guarantees about other super-classes.
            return super().rec(  # type: ignore[no-any-return,misc]
                expr, *args, **kwargs)

    def map_index_lambda(self, expr: IndexLambda) -> IndexLambda:
        return IndexLambda(expr=expr.expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict({name: self.rec(bnd)
                                         for name, bnd
                                         in sorted(expr.bindings.items())}),
                           axes=expr.axes,
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

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
                           shape=self._rec_shape(expr.shape),
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

        lbounds: list[Any] = [0]
        ubounds: list[Any] = [expr.arrays[0].shape[expr.axis]]

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
                concat_expr: ArithmeticExpression = subarray_expr
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

        bindings = {f"_in{k}": self.rec(arg) for k, arg in enumerate(expr.args)}
        redn_bounds: dict[str, tuple[ScalarExpression, ScalarExpression]] = {}
        args_as_pym_expr: list[prim.Subscript] = []
        namegen = UniqueNameGenerator(set(bindings))
        var_to_redn_descr = {}

        # {{{ add bindings coming from the shape expressions

        for access_descr, (iarg, arg) in zip(expr.access_descriptors,
                                            enumerate(expr.args), strict=True):
            subscript_indices: list[ArithmeticExpression] = []
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
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict(bindings),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(var_to_redn_descr),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_roll(self, expr: Roll) -> IndexLambda:
        from pytato.utils import dim_to_index_lambda_components

        index_expr: prim.Expression = prim.Variable("_in0")
        indices: list[ArithmeticExpression] = [
            prim.Variable(f"_{d}") for d in range(expr.ndim)]
        axis = expr.axis
        axis_len_expr, bindings = dim_to_index_lambda_components(
            expr.shape[axis],
            UniqueNameGenerator({"_in0"}))

        # Mypy has a point: the type system does not prove that the operands are
        # not complex-valued or bool.
        indices[axis] = (indices[axis] - expr.shift) % axis_len_expr  # type: ignore[operator, call-overload]

        if indices:
            index_expr = index_expr[tuple(indices)]

        # type-ignore-reason: `bindings` was returned as Dict[str, SizeParam]
        bindings["_in0"] = expr.array  # type: ignore[assignment]

        return IndexLambda(expr=index_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict({name: self.rec(bnd)
                                     for name, bnd in bindings.items()}),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> IndexLambda:
        from pytato.utils import get_indexing_expression, get_shape_after_broadcasting

        i_adv_indices = tuple(i
                              for i, idx_expr in enumerate(expr.indices)
                              if isinstance(idx_expr, (Array, *INT_CLASSES)))
        adv_idx_shape = get_shape_after_broadcasting([
                    cast("Array | int | np.integer[Any]", expr.indices[i_idx])
                    for i_idx in i_adv_indices])

        vng = UniqueNameGenerator()
        indices: list[ArithmeticExpression] = []
        in_ary = vng("in")
        bindings = {in_ary: self.rec(expr.array)}
        islice_idx = 0

        for i_idx, (idx, axis_len) in enumerate(
                        zip(expr.indices, expr.array.shape, strict=True)):
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
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags,
                           )

    def map_non_contiguous_advanced_index(
            self, expr: AdvancedIndexInNoncontiguousAxes) -> IndexLambda:
        from pytato.utils import get_indexing_expression, get_shape_after_broadcasting
        i_adv_indices = tuple(i
                              for i, idx_expr in enumerate(expr.indices)
                              if isinstance(idx_expr, (Array, *INT_CLASSES)))
        adv_idx_shape = get_shape_after_broadcasting([
            cast("Array | int | np.integer[Any]", expr.indices[i_idx])
            for i_idx in i_adv_indices])

        vng = UniqueNameGenerator()
        indices: list[ArithmeticExpression] = []

        in_ary = vng("in")
        bindings = {in_ary: self.rec(expr.array)}

        islice_idx = len(adv_idx_shape)

        for idx, axis_len in zip(expr.indices, expr.array.shape, strict=True):
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
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags,
                           )

    def map_basic_index(self, expr: BasicIndex) -> IndexLambda:
        vng = UniqueNameGenerator()
        indices: list[ArithmeticExpression] = []

        in_ary = vng("in")
        bindings = {in_ary: self.rec(expr.array)}
        islice_idx = 0

        for idx, axis_len in zip(expr.indices, expr.array.shape, strict=True):
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
                           bindings=immutabledict(bindings),
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags,
                           )

    def map_reshape(self, expr: Reshape) -> IndexLambda:
        indices = _get_reshaped_indices(expr)
        index_expr = prim.Variable("_in0")[tuple(indices)]
        return IndexLambda(expr=index_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict({"_in0": self.rec(expr.array)}),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> IndexLambda:
        indices: list[ArithmeticExpression | None] = [None] * expr.ndim
        for from_index, to_index in enumerate(expr.axis_permutation):
            indices[to_index] = prim.Variable(f"_{from_index}")

        index_expr = prim.Variable("_in0")[
            cast("tuple[ArithmeticExpression]", tuple(indices))]

        return IndexLambda(expr=index_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict({"_in0": self.rec(expr.array)}),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)


class ToIndexLambdaMapper(Mapper[Array, Never, []], ToIndexLambdaMixin):

    def handle_unsupported_array(self, expr: Array) -> Array:
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
    res = ToIndexLambdaMapper()(expr)
    assert isinstance(res, IndexLambda)
    return res

# vim:fdm=marker
