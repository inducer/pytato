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

from typing import List, Any, Dict, Tuple, TypeVar, TYPE_CHECKING
from immutabledict import immutabledict
from pytools import UniqueNameGenerator
from pytato.array import (Array, IndexLambda, Stack, Concatenate,
                          Einsum, Reshape, Roll, AxisPermutation,
                          BasicIndex, AdvancedIndexInContiguousAxes,
                          AdvancedIndexInNoncontiguousAxes,
                          NormalizedSlice, ShapeType,
                          AbstractResultWithNamedArrays)
from pytato.scalar_expr import ScalarExpression, INT_CLASSES
from pytato.diagnostic import CannotBeLoweredToIndexLambda
from pytato.tags import AssumeNonNegative
from pytato.transform import Mapper

ToIndexLambdaT = TypeVar("ToIndexLambdaT", Array, AbstractResultWithNamedArrays)


def _get_reshaped_indices(expr: Reshape) -> Tuple[ScalarExpression, ...]:
    if expr.array.shape == ():
        # RHS must be a scalar i.e. RHS' indices are empty
        assert expr.size == 1
        return ()

    if expr.order.upper() not in ["C", "F"]:
        raise NotImplementedError("Order expected to be 'C' or 'F'",
                                  f" (case insensitive) found {expr.order}")

    order = expr.order
    oldshape = expr.array.shape
    newshape = expr.shape

    # {{{ construct old -> new axis mapping

    # NOTE: there are cases where the mapping of old -> new axes is obfuscated
    # by the reshape, i.e. (5, 12, 3) -> (3, 2, 3, 10). In cases like this, we
    # default to linearizing all indices.
    # In cases where the reshape is simple, i.e. (5, 16) -> (5, 4, 4) or
    # (5, 4, 4, 4) -> (5, 16, 4), we leave the untouched axes alone and only
    # bother computing new indices for the collapsed/expanded axes.
    ambiguous_reshape = False
    oldax_to_newax = {}

    # case 1 & 2: collapsed/expanded old axes
    if len(newshape) != len(oldshape):
        expanded_old_dims = (True if len(newshape) > len(oldshape) else False)

        if expanded_old_dims:
            longshape = newshape
            shortshape = oldshape
        else:
            longshape = oldshape
            shortshape = newshape

        ilongax = 0
        for ishortax, shortax in enumerate(shortshape):
            if shortax != longshape[ilongax]:

                # collect expanded axes
                acc = 1
                ilongaxs = []
                while acc < shortax:
                    if ilongax >= len(longshape):
                        ambiguous_reshape = True
                        break

                    ilongaxs.append(ilongax)
                    acc *= longshape[ilongax]
                    ilongax += 1

                # check for axes of length 1
                if ilongax < len(longshape) and longshape[ilongax] == 1:
                    while ilongax < len(longshape) and longshape[ilongax] == 1:
                        ilongaxs.append(ilongax)
                        ilongax += 1

                if len(newshape) > len(oldshape):
                    oldax_to_newax[(ishortax,)] = tuple(ilongaxs)
                else:
                    oldax_to_newax[tuple(ilongaxs)] = (ishortax,)
            else:
                if len(newshape) > len(oldshape):
                    oldax_to_newax[(ishortax,)] = (ilongax,)
                else:
                    oldax_to_newax[(ilongax,)] = (ishortax,)

            if ambiguous_reshape:
                break

            ilongax += 1

            if ilongax >= len(longshape):
                break

    # case 3: permute axes of oldshape
    else:
        if oldshape == newshape:
            oldax_to_newax = { (i,): (i,) for i in range(len(oldshape)) }
        else:
            for ioldax, oldax in enumerate(oldshape):
                inewax = 0
                while inewax < len(newshape):
                    if oldax == newshape[inewax]:
                        break
                    inewax += 1

                if inewax >= len(newshape):
                    ambiguous_reshape = True
                    break

                oldax_to_newax[(ioldax,)] = (inewax,)

    # }}}

    # {{{ check that we've caught everything

    # need a mapping of all old axes -> all new axes
    if (len(sum(oldax_to_newax.keys(), ())) != len(oldshape) or
        len(sum(oldax_to_newax.values(), ())) != len(newshape)):
        ambiguous_reshape = True

    # }}}

    # {{{ process old -> new axis mapping

    def compute_new_indices(ioldaxs, inewaxs, order, fallback=False):

        if not fallback:
            sub_oldshape = [oldshape[ioldax] for ioldax in ioldaxs]
            sub_newshape = [newshape[inewax] for inewax in inewaxs]

            index_vars = [prim.Variable(f"_{i}") for i in inewaxs]

            # indices match, no need to do anything
            if ioldaxs == inewaxs:
                return tuple(index_vars)

        else:
            sub_oldshape = oldshape
            sub_newshape = newshape

            index_vars = [prim.Variable(f"_{i}") for i in range(len(newshape))]

        oldstrides = [1]
        oldstride_axes = (
            reversed(sub_oldshape[1:]) if order == "C" else sub_oldshape[:-1])

        for ax_len in oldstride_axes:
            oldstrides.append(ax_len*oldstrides[-1])

        oldsizetill_axes = (
            reversed(sub_oldshape[:-1]) if order == "C" else sub_oldshape[:-1])
        oldsizetills = [sub_oldshape[-1] if order == "C" else sub_oldshape[0]]

        for ax_len in oldsizetill_axes:
            oldsizetills.append(ax_len*oldsizetills[-1])

        newstride_axes = (
            reversed(sub_newshape[1:] if order == "C" else sub_newshape[:-1]))
        newstrides = [1]
        for ax_len in newstride_axes:
            newstrides.append(ax_len*newstrides[-1])

        if order == "C":
            newstrides = newstrides[::-1]
            oldstrides = oldstrides [::-1]
            oldsizetills = oldsizetills[::-1]

        flattened_idx = sum(
            index_var*stride
            for index_var, stride in zip(index_vars, newstrides)
        )

        return tuple(((flattened_idx % sizetill) // stride)
                       for stride, sizetill in zip(oldstrides, oldsizetills))

    if ambiguous_reshape:
        print("Ambiguous reshape found when trying to compute reshaped indices."
              " Defaulting to linearizing all axes.")
        return compute_new_indices(oldshape, newshape, order, fallback=True)

    ret = [
        compute_new_indices(ioldaxs, inewaxs, order)
        for ioldaxs, inewaxs in oldax_to_newax.items()
    ]

    # }}}

    # create a flat tuple instead of a (possibly) tuple of tuples
    return sum(ret, ())

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
                           bindings=immutabledict(bindings),
                           var_to_reduction_descr=immutabledict(),
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
                           bindings=immutabledict(bindings),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
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
                           bindings=immutabledict(bindings),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(var_to_redn_descr),
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
                           bindings=immutabledict({name: self.rec(bnd)
                                     for name, bnd in bindings.items()}),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
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
                           bindings=immutabledict(bindings),
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
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
                           bindings=immutabledict(bindings),
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
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
                           bindings=immutabledict(bindings),
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags,
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
                           tags=expr.tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> IndexLambda:
        indices = [None] * expr.ndim
        for from_index, to_index in enumerate(expr.axis_permutation):
            indices[to_index] = prim.Variable(f"_{from_index}")

        index_expr = prim.Variable("_in0")[tuple(indices)]

        return IndexLambda(expr=index_expr,
                           shape=self._rec_shape(expr.shape),
                           dtype=expr.dtype,
                           bindings=immutabledict({"_in0": self.rec(expr.array)}),
                           axes=expr.axes,
                           var_to_reduction_descr=immutabledict(),
                           tags=expr.tags)


class ToIndexLambdaMapper(Mapper, ToIndexLambdaMixin):

    def handle_unsupported_array(self, expr: Any) -> Any:
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
