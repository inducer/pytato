from __future__ import annotations


"""
.. currentmodule:: pytato.transform

TODO:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytato.transform.parameter_study
"""
__copyright__ = """
Copyright (C) 2020-1 University of Illinois Board of Trustees
"""

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
from typing import (
    Iterable,
    Mapping,
)

from immutabledict import immutabledict

import pymbolic.primitives as prim
from pytools import unique
from pytools.tag import UniqueTag

from pytato.array import (
    AbstractResultWithNamedArrays,
    Array,
    AxesT,
    Axis,
    AxisPermutation,
    Concatenate,
    Einsum,
    EinsumAxisDescriptor,
    EinsumElementwiseAxis,
    IndexBase,
    IndexLambda,
    NormalizedSlice,
    Placeholder,
    Reshape,
    Roll,
    Stack,
)
from pytato.distributed.nodes import (
    DistributedRecv,
    DistributedSendRefHolder,
)
from pytato.function import (
    Call,
    FunctionDefinition,
    NamedCallResult,
)
from pytato.scalar_expr import IdentityMapper, IntegralT
from pytato.transform import CopyMapper


@dataclass(frozen=True)
class ParameterStudyAxisTag(UniqueTag):
    """
        A tag to indicate that the axis is being used
        for independent trials like in a parameter study.
        If you want to vary multiple input variables in the
        same study then you need to have the same type of
        class: 'ParameterStudyAxisTag'.
    """
    axis_size: int


StudiesT = tuple[ParameterStudyAxisTag, ...]
ArraysT = tuple[Array, ...]
KnownShapeType = tuple[IntegralT, ...]


class ExpansionMapper(CopyMapper):

    def __init__(self, placeholder_name_to_parameter_studies: Mapping[str, StudiesT]):
        super().__init__()
        self.placeholder_name_to_parameter_studies = placeholder_name_to_parameter_studies  # noqa

    def _shapes_and_axes_from_predecessors(self, curr_expr: Array,
                                         mapped_preds: ArraysT) -> \
                                                 tuple[KnownShapeType,
                                                       AxesT,
                                                       dict[Array, tuple[int, ...]]]:

        assert not any(axis.tags_of_type(ParameterStudyAxisTag) for
                       axis in curr_expr.axes)

        # We are post pending the axes we are using for parameter studies.
        new_shape: KnownShapeType = ()
        studies_axes: AxesT = ()

        study_to_arrays: dict[frozenset[ParameterStudyAxisTag], ArraysT] = {}

        active_studies: set[ParameterStudyAxisTag] = set()

        for arr in mapped_preds:
            for axis in arr.axes:
                tags = axis.tags_of_type(ParameterStudyAxisTag)
                if tags:
                    assert len(tags) == 1  # only one study per axis.
                    active_studies = active_studies.union(tags)
                    if tags in study_to_arrays.keys():
                        study_to_arrays[tags] = (*study_to_arrays[tags], arr)
                    else:
                        study_to_arrays[tags] = (arr,)

        ps, na, arr_to_studies = self._studies_to_shape_and_axes_and_arrays_in_canonical_order(active_studies, # noqa
                                                                             new_shape,
                                                                             studies_axes,
                                                                             study_to_arrays)
        # Add in the arrays that are not a part of a parameter study.
        # This is done to avoid any KeyErrors later.

        for arr in unique(mapped_preds):  # pytools unique maintains the order.
            if arr not in arr_to_studies.keys():
                arr_to_studies[arr] = ()
            else:
                assert len(arr_to_studies[arr]) > 0

        assert len(arr_to_studies) == len(list(unique(mapped_preds)))

        return ps, na, arr_to_studies

    def _studies_to_shape_and_axes_and_arrays_in_canonical_order(self,
                    studies: Iterable[ParameterStudyAxisTag],
                    new_shape: KnownShapeType, new_axes: AxesT,
                    study_to_arrays: dict[frozenset[ParameterStudyAxisTag], ArraysT]) \
                            -> tuple[KnownShapeType, AxesT, dict[Array,
                                                                 tuple[int, ...]]]:

        # This is where we specify the canonical ordering of the studies.

        array_to_canonical_ordered_studies: dict[Array, tuple[int, ...]] = {}
        studies_axes = new_axes

        for ind, study in enumerate(sorted(studies,
                                           key=lambda x: str(x.__class__))):
            new_shape = (*new_shape, study.axis_size)
            studies_axes = (*studies_axes, Axis(tags=frozenset((study,))))
            if study_to_arrays:
                for arr in study_to_arrays[frozenset((study,))]:
                    if arr in array_to_canonical_ordered_studies.keys():
                        array_to_canonical_ordered_studies[arr] = (*array_to_canonical_ordered_studies[arr], ind) # noqa
                    else:
                        array_to_canonical_ordered_studies[arr] = (ind,)

        return new_shape, studies_axes, array_to_canonical_ordered_studies

    def map_placeholder(self, expr: Placeholder) -> Array:
        # This is where we could introduce extra axes.
        if expr.name in self.placeholder_name_to_parameter_studies.keys():
            new_axes = expr.axes
            studies = self.placeholder_name_to_parameter_studies[expr.name]
            new_shape, new_axes, _ = self._studies_to_shape_and_axes_and_arrays_in_canonical_order( # noqa
                                                                    studies,
                                                                    (),
                                                                    expr.axes,
                                                                    {})

            return Placeholder(name=expr.name,
                               shape=self.rec_idx_or_size_tuple((*expr.shape,
                                                                 *new_shape,)),
                               dtype=expr.dtype,
                               axes=new_axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

        return super().map_placeholder(expr)

    def map_roll(self, expr: Roll) -> Array:
        new_predecessor = self.rec(expr.array)
        _, new_axes, _ = self._shapes_and_axes_from_predecessors(expr,
                                                                (new_predecessor,))

        return Roll(array=new_predecessor,
                    shift=expr.shift,
                    axis=expr.axis,
                    axes=expr.axes + new_axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        new_predecessor = self.rec(expr.array)
        postpend_shape, new_axes, _ = self._shapes_and_axes_from_predecessors(expr,
                                                                             (new_predecessor,))
        # Include the axes we are adding to the system.
        axis_permute = expr.axis_permutation + tuple([i + len(expr.axis_permutation)
                                             for i in range(len(postpend_shape))])

        return AxisPermutation(array=new_predecessor,
                               axis_permutation=axis_permute,
                               axes=expr.axes + new_axes,
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase) -> Array:
        new_predecessor = self.rec(expr.array)
        postpend_shape, new_axes, _ = self._shapes_and_axes_from_predecessors(expr,
                                                                             (new_predecessor,))
        # Update the indices.
        new_indices = expr.indices
        for shape in postpend_shape:
            new_indices = (*new_indices, NormalizedSlice(0, shape, 1))

        return type(expr)(new_predecessor,
                          indices=self.rec_idx_or_size_tuple(new_indices),
                          axes=expr.axes + new_axes,
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape) -> Array:
        new_predecessor = self.rec(expr.array)
        postpend_shape, new_axes, _ = self._shapes_and_axes_from_predecessors(expr,
                                                                             (new_predecessor,))
        return Reshape(new_predecessor,
                       newshape=self.rec_idx_or_size_tuple(expr.newshape +
                                                           postpend_shape),
                       order=expr.order,
                       axes=expr.axes + new_axes,
                       tags=expr.tags,
                       non_equality_tags=expr.non_equality_tags)

        # {{{ Operations with multiple predecessors.

    def _mult_pred_same_shape(self, expr: Stack | Concatenate) -> tuple[ArraysT,
                                                                        AxesT]:

        """
            This method will convert predecessors who were originally the same
            shape in a single instance program to the same shape in this multiple
            instance program.
        """

        new_predecessors = tuple(self.rec(arr) for arr in expr.arrays)

        studies_shape, new_axes, arrays_to_study_num_present = self._shapes_and_axes_from_predecessors(expr, new_predecessors) # noqa

        # This is going to be expensive.

        # Now we need to update the expressions.
        # Now that we have the appropriate shape,
        # we need to update the input arrays to match.

        cp_map = CopyMapper()
        corrected_new_arrays: ArraysT = ()
        for iarr, array in enumerate(new_predecessors):
            tmp = cp_map(array)  # Get a copy of the array.
            # We need to grow the array to the new size.
            if arrays_to_study_num_present:
                studies_present = arrays_to_study_num_present[array]
                for ind, size in enumerate(studies_shape):
                    if ind not in studies_present:
                        build: ArraysT = tuple([cp_map(tmp) for _ in range(size)])

                        #  Note we are stacking the arrays into the appropriate shape.
                        tmp = Stack(arrays=build,
                                    axis=len(expr.arrays[iarr].axes) + ind,
                                    axes=new_axes[:ind],
                                    tags=tmp.tags,
                                    non_equality_tags=tmp.non_equality_tags)

            if studies_shape:
                assert tmp.shape[-(len(studies_shape)):] == studies_shape
            else:
                assert tmp.shape[-(len(studies_shape)):] == tmp.shape

            corrected_new_arrays = (*corrected_new_arrays, tmp)

        return corrected_new_arrays, new_axes

    def map_stack(self, expr: Stack) -> Array:
        new_arrays, new_axes_for_end = self._mult_pred_same_shape(expr)
        return Stack(arrays=new_arrays,
                     axis=expr.axis,
                     axes=expr.axes + new_axes_for_end,
                     tags=expr.tags,
                     non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate) -> Array:
        new_arrays, new_axes_for_end = self._mult_pred_same_shape(expr)

        return Concatenate(arrays=new_arrays,
                           axis=expr.axis,
                           axes=expr.axes + new_axes_for_end,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        # Update bindings first.
        new_bindings: dict[str, Array] = {name: self.rec(bnd)
                                             for name, bnd in
                                          sorted(expr.bindings.items())}
        new_arrays = (*new_bindings.values(),)

        array_to_bnd_name: dict[Array, str] = {bnd: name for name, bnd
                                               in sorted(new_bindings.items())}

        # Determine the new parameter studies that are being conducted.
        postpend_shape, new_axes, array_to_studies_num = self._shapes_and_axes_from_predecessors(expr, # noqa
                                                                             new_arrays)

        varname_to_studies_nums = {array_to_bnd_name[array]: studies for array,
                              studies in array_to_studies_num.items()}

        # Now we need to update the expressions.
        scalar_expr = ParamAxisExpander()(expr.expr, varname_to_studies_nums,
                                          len(expr.shape))

        return IndexLambda(expr=scalar_expr,
                           bindings=immutabledict(new_bindings),
                           shape=(*expr.shape, *postpend_shape,),
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           dtype=expr.dtype,
                           axes=(*expr.axes, *new_axes,),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_einsum(self, expr: Einsum) -> Array:

        new_predecessors = tuple(self.rec(arg) for arg in expr.args)
        _, new_axes, arrays_to_study_num_present = self._shapes_and_axes_from_predecessors(expr, new_predecessors) # noqa

        access_descriptors: tuple[tuple[EinsumAxisDescriptor, ...], ...] = ()
        for ival, array in enumerate(new_predecessors):
            one_descr = expr.access_descriptors[ival]
            if arrays_to_study_num_present:
                for ind in arrays_to_study_num_present[array]:
                    one_descr = (*one_descr,
                                 # Adding in new element axes to the end of the arrays.
                                 EinsumElementwiseAxis(dim=len(expr.shape) + ind))
            access_descriptors = (*access_descriptors, one_descr)

        return Einsum(access_descriptors,
                     new_predecessors,
                     axes=expr.axes + new_axes,
                     redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                     tags=expr.tags,
                     non_equality_tags=expr.non_equality_tags)

    # }}} Operations with multiple predecessors.

    # {{{ Function definitions
    def map_function_definition(self, expr: FunctionDefinition) -> FunctionDefinition:
        raise NotImplementedError(" Expanding functions is not yet supported.")

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        raise NotImplementedError(" Expanding functions is not yet supported.")

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        raise NotImplementedError(" Expanding functions is not yet supported.")

    # }}}

    # {{{ Distributed Programming Constructs
    def map_distributed_recv(self, expr: DistributedRecv) -> DistributedRecv:
        # This data will depend solely on the rank sending it to you.
        raise NotImplementedError(" Expanding distributed programming constructs is"
                                  " not yet supported.")

    def map_distributed_send_ref_holder(self, expr: DistributedSendRefHolder) \
                                        -> Array:
        # We are sending the data. This data may increase in size due to the
        # parameter studies.
        raise NotImplementedError(" Expanding distributed programming constructs is"
                                  " not yet supported.")

    # }}}


class ParamAxisExpander(IdentityMapper):

    def map_subscript(self, expr: prim.Subscript,
                      varname_to_studies_num: Mapping[str, tuple[int, ...]],
                      num_original_inds: int) -> prim.Subscript:
        # We know that we are not changing the variable that we are indexing into.
        # This is stored in the aggregate member of the class Subscript.

        # We only need to modify the indexing which is stored in the index member.
        name = expr.aggregate.name
        if name in varname_to_studies_num.keys():
            #  These are the single instance information.
            index = self.rec(expr.index, varname_to_studies_num,
                             num_original_inds)

            new_vars: tuple[prim.Variable, ...] = ()
            my_studies: tuple[int, ...] = varname_to_studies_num[name]

            for num in my_studies:
                new_vars = *new_vars, prim.Variable(f"_{num_original_inds + num}"),

            if isinstance(index, tuple):
                index = index + new_vars
            else:
                index = tuple(index) + new_vars
            return type(expr)(aggregate=expr.aggregate, index=index)
        return expr

    def map_variable(self, expr: prim.Variable,
                      varname_to_studies: Mapping[str, tuple[int, ...]],
                     num_original_inds: int) -> prim.Expression:
        # We know that a variable is a leaf node. So we only need to update it
        # if the variable is part of a study.

        if expr.name in varname_to_studies.keys():
            #  These are the single instance information.
            #  In the multiple instance we will need to index into the variable.

            new_vars: tuple[prim.Variable, ...] = ()
            my_studies: tuple[int, ...] = varname_to_studies[expr.name]

            for num in my_studies:
                new_vars = *new_vars, prim.Variable(f"_{num_original_inds + num}"),

            return prim.Subscript(aggregate=expr, index=new_vars)
        return expr
