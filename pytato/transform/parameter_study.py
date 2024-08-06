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
                                                       dict[int, tuple[int, ...]]]:

        assert not any(axis.tags_of_type(ParameterStudyAxisTag) for
                       axis in curr_expr.axes)

        # We are post pending the axes we are using for parameter studies.

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

        ps, na, arr_num_to_study_nums = self._studies_to_shape_and_axes_and_arrays_in_canonical_order(active_studies, # noqa
                                                study_to_arrays, mapped_preds)

        # Add in the arrays that are not a part of a parameter study.
        # This is done to avoid any KeyErrors later.

        for arr_num in range(len(mapped_preds)):
            if arr_num not in arr_num_to_study_nums.keys():
                arr_num_to_study_nums[arr_num] = ()
            else:
                assert len(arr_num_to_study_nums[arr_num]) > 0

        assert len(arr_num_to_study_nums) == len(mapped_preds)

        for axis in na:
            assert axis.tags_of_type(ParameterStudyAxisTag)

        return ps, na, arr_num_to_study_nums

    def _studies_to_shape_and_axes_and_arrays_in_canonical_order(self,
                    studies: Iterable[ParameterStudyAxisTag],
                    study_to_arrays: dict[frozenset[ParameterStudyAxisTag], ArraysT],
                    mapped_preds: ArraysT) -> tuple[KnownShapeType, AxesT,
                                                    dict[int, tuple[int, ...]]]:

        # This is where we specify the canonical ordering of the studies.
        array_num_to_study_nums: dict[int, tuple[int, ...]] = {}
        new_shape: KnownShapeType = ()
        studies_axes: AxesT = ()

        for ind, study in enumerate(sorted(studies,
                                           key=lambda x: str(x.__class__))):
            new_shape = (*new_shape, study.axis_size)
            studies_axes = (*studies_axes, Axis(tags=frozenset((study,))))
            for arr_num, arr in enumerate(mapped_preds):
                if arr in study_to_arrays[frozenset((study,))]:
                    if arr_num in array_num_to_study_nums.keys():
                        array_num_to_study_nums[arr_num] = (*array_num_to_study_nums[arr_num], ind) # noqa
                    else:
                        array_num_to_study_nums[arr_num] = (ind,)

        assert len(new_shape) == len(studies)
        assert len(studies_axes) == len(studies)

        return new_shape, studies_axes, array_num_to_study_nums

    def map_placeholder(self, expr: Placeholder) -> Array:
        # This is where we could introduce extra axes.
        if expr.name in self.placeholder_name_to_parameter_studies.keys():
            new_axes = expr.axes
            studies = self.placeholder_name_to_parameter_studies[expr.name]

            new_shape: KnownShapeType = ()
            new_axes: AxesT = ()

            # We know that there are no predecessors and we know the studies to add.
            # We need to get them in the right order.
            new_shape, new_axes, _ = self._studies_to_shape_and_axes_and_arrays_in_canonical_order( # noqa
                                                                    studies, {}, ())

            return Placeholder(name=expr.name,
                               shape=self.rec_idx_or_size_tuple((*expr.shape,
                                                                 *new_shape,)),
                               dtype=expr.dtype,
                               axes=(*expr.axes, *new_axes,),
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
                    axes=(*expr.axes, *new_axes,),
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
                               axes=(*expr.axes, *new_axes,),
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
                          axes=(*expr.axes, *new_axes,),
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
                       axes=(*expr.axes, *new_axes,),
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

        studies_shape, new_axes, arr_num_to_study_nums = self._shapes_and_axes_from_predecessors(expr, new_predecessors) # noqa

        # This is going to be expensive.
        correct_shape_preds: ArraysT = ()

        for iarr, array in enumerate(new_predecessors):
            # Broadcast out to the right shape.
            num_single_inst_axes = len(expr.arrays[iarr].shape)
            scale_expr = prim.Subscript(prim.Variable("_in0"),
                                        index=tuple([prim.Variable(f"_{ind}") for
                                               ind in range(num_single_inst_axes)]))
            new_array = IndexLambda(expr=scale_expr,
                                    bindings=immutabledict({"_in0": array}),
                                    tags=array.tags,
                                    non_equality_tags=array.non_equality_tags,
                                    dtype=array.dtype,
                                    var_to_reduction_descr=immutabledict({}),
                                    shape=(*expr.arrays[iarr].shape, *studies_shape,),
                                    axes=(*expr.arrays[iarr].axes, *new_axes,))

            correct_shape_preds = (*correct_shape_preds, new_array,)

        for arr in correct_shape_preds:
            assert arr.shape == correct_shape_preds[0].shape 

        return correct_shape_preds, new_axes

    def map_stack(self, expr: Stack) -> Array:
        new_arrays, new_axes = self._mult_pred_same_shape(expr)
        return Stack(arrays=new_arrays,
                     axis=expr.axis,
                     axes=(*expr.axes, *new_axes,),
                     tags=expr.tags,
                     non_equality_tags=expr.non_equality_tags)

    def map_concatenate(self, expr: Concatenate) -> Array:
        new_arrays, new_axes = self._mult_pred_same_shape(expr)

        return Concatenate(arrays=new_arrays,
                           axis=expr.axis,
                           axes=(*expr.axes, *new_axes,),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        # Update bindings first.

        new_binds: dict[str, Array] = {name: self.rec(bnd)
                                             for name, bnd in
                                          sorted(expr.bindings.items())}
        new_arrays = (*new_binds.values(),)

        # The arrays may be the same for a predecessors.
        # However, the index will be unique.

        array_num_to_bnd_name: dict[int, str] = {ind: name for ind, (name, _)
                                                 in enumerate(sorted(new_binds.items()))} # noqa

        # Determine the new parameter studies that are being conducted.
        postpend_shape, new_axes, arr_num_to_study_nums = self._shapes_and_axes_from_predecessors(expr, # noqa
                                                                             new_arrays)

        varname_to_studies_nums = {array_num_to_bnd_name[iarr]: studies for iarr,
                              studies in arr_num_to_study_nums.items()}

        for vn_key in varname_to_studies_nums.keys():
            assert vn_key in new_binds.keys()

        for vn_key in new_binds.keys():
            assert vn_key in varname_to_studies_nums.keys()

        # Now we need to update the expressions.
        scalar_expr = ParamAxisExpander()(expr.expr, varname_to_studies_nums,
                                          len(expr.shape))
        

        # Data dump the index lambda to a file which I can read.
        with open("expansion_map_indexlambda.txt", "a+") as my_file:
            my_file.write("\n")
            my_file.write("\n")
            my_file.write(str(scalar_expr))
            my_file.write("\n")
            my_file.write(str({name: len(bnd.axes) for name, bnd in sorted(new_binds.items())}))


        return IndexLambda(expr=scalar_expr,
                           bindings=immutabledict(new_binds),
                           shape=(*expr.shape, *postpend_shape,),
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           dtype=expr.dtype,
                           axes=(*expr.axes, *new_axes,),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_einsum(self, expr: Einsum) -> Array:

        new_predecessors = tuple(self.rec(arg) for arg in expr.args)
        _, new_axes, arr_num_to_study_nums = self._shapes_and_axes_from_predecessors(expr, new_predecessors) # noqa

        access_descriptors: tuple[tuple[EinsumAxisDescriptor, ...], ...] = ()
        for ival, array in enumerate(new_predecessors):
            one_descr = expr.access_descriptors[ival]
            if arr_num_to_study_nums:
                for ind in arr_num_to_study_nums[ival]:
                    one_descr = (*one_descr,
                                 # Adding in new element axes to the end of the arrays.
                                 EinsumElementwiseAxis(dim=len(expr.shape) + ind))
            access_descriptors = (*access_descriptors, one_descr)

            # One descriptor per axis.
            assert len(one_descr) == len(array.shape)

        return Einsum(access_descriptors,
                     new_predecessors,
                     axes=(*expr.axes, *new_axes,),
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
                      num_orig_elem_inds: int) -> prim.Subscript:
        """
            `arg' num_orig_elem_inds specifies the maximum number of indices
            in a scalar expression that can be used to index into an array provided
            that index is not used as part of a reduction.
        """
        # We know that we are not changing the variable that we are indexing into.
        # This is stored in the aggregate member of the class Subscript.

        # We only need to modify the indexing which is stored in the index member.
        assert isinstance(expr.aggregate, prim.Variable)

        name = expr.aggregate.name
        if name in varname_to_studies_num.keys():
            #  These are the single instance information.

            index = self.rec(expr.index, varname_to_studies_num,
                             num_orig_elem_inds)

            new_vars: tuple[prim.Variable, ...] = ()
            my_studies: tuple[int, ...] = varname_to_studies_num[name]

            for num in my_studies:
                new_vars = *new_vars, prim.Variable(f"_{num_orig_elem_inds + num}"),

            if isinstance(index, tuple):
                index = index + new_vars
            else:
                index = tuple(index) + new_vars

            return type(expr)(aggregate=expr.aggregate, index=index)

        return super().map_subscript(expr, varname_to_studies_num, num_orig_elem_inds)

    def map_variable(self, expr: prim.Variable,
                      varname_to_studies: Mapping[str, tuple[int, ...]],
                     num_orig_elem_inds: int) -> prim.Expression:
        # We know that a variable is a leaf node. So we only need to update it
        # if the variable is part of a study.

        if expr.name in varname_to_studies.keys():
            # The variable may need to be updated.

            my_studies: tuple[int, ...] = varname_to_studies[expr.name]

            if len(my_studies) == 0:
                # No studies
                return expr

            new_vars: tuple[prim.Variable, ...] = ()

            assert my_studies
            assert len(my_studies) > 0

            for num in my_studies:
                new_vars = *new_vars, prim.Variable(f"_{num_orig_elem_inds + num}"),

            return prim.Subscript(aggregate=expr, index=new_vars)

        # Since the variable is not in a study we can just return the answer.
        return super().map_variable(expr, varname_to_studies, num_orig_elem_inds)

    def map_substitution(self, expr: prim.Substitution, varname_to_studies,
                         num_orig_elem_inds):

        breakpoint()
        raise NotImplementedError("Substitution needs to be expanded.")
