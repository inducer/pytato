from __future__ import annotations


__doc__ = """
.. currentmodule:: pytato.transform.parameter_study

.. autoclass:: ParameterStudyAxisTag
.. autoclass:: ParameterStudyVectorizer
.. autoclass:: IndexLambdaScalarExpressionVectorizer
"""

__copyright__ = "Copyright (C) 2024 Nicholas Koskelo"

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
    Mapping,
    Sequence,
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


KnownShapeType = tuple[IntegralT, ...]


@dataclass(frozen=True)
class ParameterStudyAxisTag(UniqueTag):
    """
    A tag to indicate that the axis is being used
    for independent trials like in a parameter study.
    If you want to vary multiple input variables in the
    same study then you need to have the same type of
    :class:`ParameterStudyAxisTag`.
    """
    size: int


class IndexLambdaScalarExpressionVectorizer(IdentityMapper):
    """
    The goal of this mapper is to convert a single instance scalar expression
    into a single instruction multiple data scalar expression. We assume that any
    array variables in the original scalar expression will be indexed completely.
    Also, new axes for the studies are assumed to be on the end of
    those array variables.
    """

    def __init__(self, varname_to_studies_num: Mapping[str, tuple[int, ...]],
                    num_orig_elem_inds: int):
        """
        `arg` varname_to_studies_num: is a mapping from the variable name used
        in the scalar expression to the studies present in the multiple instance
        expression. Note that the varnames must be for the array variables only.

        `arg` num_orig_elem_inds: is the number of element axes in the result of
        the single instance expression.
        """

        super().__init__()
        self.varname_to_studies_num = varname_to_studies_num
        self.num_orig_elem_inds = num_orig_elem_inds

    def map_subscript(self, expr: prim.Subscript) -> prim.Subscript:
        # We only need to modify the indexing which is stored in the index member.
        assert isinstance(expr.aggregate, prim.Variable)

        name = expr.aggregate.name
        if name in self.varname_to_studies_num.keys():
            #  These are the single instance information.

            index = self.rec(expr.index)

            additional_inds = (prim.Variable(f"_{self.num_orig_elem_inds + num}") for
                               num in self.varname_to_studies_num[name])

            return type(expr)(aggregate=expr.aggregate,
                              index=(*index, *additional_inds,))

        return super().map_subscript(expr)

    def map_variable(self, expr: prim.Variable) -> prim.Expression:
        # We know that a variable is a leaf node. So we only need to update it
        # if the variable is part of a study.
        if expr.name in self.varname_to_studies_num.keys():
            # The variable may need to be updated.

            my_studies: tuple[int, ...] = self.varname_to_studies_num[expr.name]

            if len(my_studies) == 0:
                # No studies
                return expr

            assert my_studies
            assert len(my_studies) > 0

            new_vars = (prim.Variable(f"_{self.num_orig_elem_inds + num}") # noqa E501
                            for num in my_studies)

            return prim.Subscript(aggregate=expr, index=tuple(new_vars))

        # Since the variable is not in a study we can just return the answer.
        return super().map_variable(expr)


def _param_study_to_index(tag: ParameterStudyAxisTag) -> str:
    """
    Get the canonical index string associated with the input tag.
    """
    return str(tag.__class__)  # Update to use the qualname or name.


class ParameterStudyVectorizer(CopyMapper):
    r"""
    This mapper will expand a DAG into a DAG for parameter studies. An array is part
    of a parameter study if one of its axes is tagged with a with a tag from
    :class:`ParameterStudyAxisTag` and all of the axes after that axis are also tagged
    with a distinct :class:`ParameterStudyAxisTag` tag. An array may be a member of
    multiple parameter studies. The new DAG for parameter studies which will be
    equivalent to running your original DAG once for each input in the parameter study
    space. The parameter study space is defined as the Cartesian product of all
    the input parameter studies. When calling this mapper you must specify which input
    :class:`~pytato.array.Placeholder` arrays are part of what parameter study.

    To maintain the equivalence with repeated calling the single instance DAG, the
    DAG for parameter studies will not create any expressions which depend on the
    specific instance of a parameter study.

    It is not required that each input be part of a parameter study as we will
    broadcast the input to the appropriate size.

    The mapper does not support distributed programming or function definitions.

    .. note::

    Any new axes used for parameter studies will be added to the end of the arrays.
    """

    def __init__(self,
                 place_name_to_parameter_studies: Mapping[str,
                                                    tuple[ParameterStudyAxisTag, ...]],
                 study_to_size: Mapping[ParameterStudyAxisTag, int]):
        super().__init__()
        self.place_name_to_parameter_studies = place_name_to_parameter_studies  # E501
        self.study_to_size = study_to_size

    def _get_canonical_ordered_studies(
            self, mapped_preds: tuple[Array, ...]) -> Sequence[ParameterStudyAxisTag]:
        active_studies: set[ParameterStudyAxisTag] = set()
        for arr in mapped_preds:
            for axis in arr.axes:
                tags = axis.tags_of_type(ParameterStudyAxisTag)
                if tags:
                    assert len(tags) == 1  # only one study per axis.
                    active_studies = active_studies.union(tags)

        return sorted(active_studies, key=_param_study_to_index)

    def _canonical_ordered_studies_to_shapes_and_axes(
            self, studies: Sequence[ParameterStudyAxisTag]) -> tuple[list[int],
                                                                     list[Axis]]:
        """
        Get the shapes and axes in the canonical ordering.
        """

        return [self.study_to_size[study] for study in studies], \
                [Axis(tags=frozenset((study,))) for study in studies]

    def map_placeholder(self, expr: Placeholder) -> Array:

        if expr.name in self.place_name_to_parameter_studies.keys():
            canonical_studies = sorted(self.place_name_to_parameter_studies[expr.name],
                                       key=_param_study_to_index) # noqa E501

            # We know that there are no predecessors and we know the studies to add.
            # We need to get them in the right order.
            end_shape, end_axes = self._canonical_ordered_studies_to_shapes_and_axes(canonical_studies) # noqa E501

            return Placeholder(name=expr.name,
                               shape=self.rec_idx_or_size_tuple((*expr.shape,
                                                                 *end_shape,)),
                               dtype=expr.dtype,
                               axes=(*expr.axes, *end_axes,),
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

        return super().map_placeholder(expr)

    def map_roll(self, expr: Roll) -> Array:
        new_predecessor = self.rec(expr.array)

        canonical_studies = self._get_canonical_ordered_studies((new_predecessor,))
        _, end_axes = self._canonical_ordered_studies_to_shapes_and_axes(canonical_studies) # noqa E501

        return Roll(array=new_predecessor,
                    shift=expr.shift,
                    axis=expr.axis,
                    axes=(*expr.axes, *end_axes,),
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        new_predecessor = self.rec(expr.array)

        canonical_studies = self._get_canonical_ordered_studies((new_predecessor,))
        end_shapes, end_axes = self._canonical_ordered_studies_to_shapes_and_axes(canonical_studies) # noqa E501

        # Include the axes we are adding to the system.
        n_single_inst_axes: int = len(expr.axis_permutation)
        axis_permute_gen = (i + n_single_inst_axes for i in range(len(end_shapes)))

        return AxisPermutation(array=new_predecessor,
                               axis_permutation=(*expr.axis_permutation,
                                                 *axis_permute_gen,),
                               axes=(*expr.axes, *end_axes,),
                               tags=expr.tags,
                               non_equality_tags=expr.non_equality_tags)

    def _map_index_base(self, expr: IndexBase) -> Array:
        new_predecessor = self.rec(expr.array)

        canonical_studies = self._get_canonical_ordered_studies((new_predecessor,))
        end_shape, end_axes = self._canonical_ordered_studies_to_shapes_and_axes(canonical_studies) # noqa E501

        # Update the indices.
        end_indices = (NormalizedSlice(0, shape, 1) for shape in end_shape)

        return type(expr)(new_predecessor,
                          indices=self.rec_idx_or_size_tuple((*expr.indices,
                                                              *end_indices)),
                          axes=(*expr.axes, *end_axes,),
                          tags=expr.tags,
                          non_equality_tags=expr.non_equality_tags)

    def map_reshape(self, expr: Reshape) -> Array:
        new_predecessor = self.rec(expr.array)

        canonical_studies = self._get_canonical_ordered_studies((new_predecessor,))
        end_shape, end_axes = self._canonical_ordered_studies_to_shapes_and_axes(canonical_studies) # noqa E501

        return Reshape(new_predecessor,
                       newshape=self.rec_idx_or_size_tuple((*expr.shape,
                                                            *end_shape,)),
                       order=expr.order,
                       axes=(*expr.axes, *end_axes,),
                       tags=expr.tags,
                       non_equality_tags=expr.non_equality_tags)

        # {{{ Operations with multiple predecessors.

    def _broadcast_predecessors_to_same_shape(self, expr: Stack | Concatenate) \
                                                -> tuple[tuple[Array, ...], AxesT]:

        """
        This method will convert predecessors who were originally the same
        shape in a single instance program to the same shape in this multiple
        instance program.
        """

        new_predecessors = tuple(self.rec(arr) for arr in expr.arrays)

        canonical_studies = self._get_canonical_ordered_studies(new_predecessors)
        studies_shape, new_axes = self._canonical_ordered_studies_to_shapes_and_axes(canonical_studies) # noqa E501

        if not studies_shape:
            # We do not need to do anything as the expression we have is unmodified.
            return new_predecessors, tuple(new_axes)

        correct_shape_preds: tuple[Array, ...] = ()
        for iarr, array in enumerate(new_predecessors):
            # Broadcast out to the right shape.
            n_single_inst_axes = len(expr.arrays[iarr].shape)

            # We assume there is at most one axis
            # tag of type ParameterStudyAxisTag per axis.

            index = tuple(prim.Variable(f"_{ind}") if not \
                        array.axes[ind].tags_of_type(ParameterStudyAxisTag) \
                    else prim.Variable(f"_{n_single_inst_axes + canonical_studies.index(tuple(array.axes[ind].tags_of_type(ParameterStudyAxisTag))[0])}") # noqa E501
                    for ind in range(len(array.shape)))

            assert len(index) == len(array.axes)

            new_array = IndexLambda(expr=prim.Subscript(prim.Variable("_in0"),
                                                        index=index),
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

        return correct_shape_preds, tuple(new_axes)

    def map_stack(self, expr: Stack) -> Array:
        new_arrays, new_axes = self._broadcast_predecessors_to_same_shape(expr)
        out = Stack(arrays=new_arrays,
                     axis=expr.axis,
                     axes=(*expr.axes, *new_axes,),
                     tags=expr.tags,
                     non_equality_tags=expr.non_equality_tags)

        assert out.ndim == len(out.shape)
        assert len(out.shape) == len(out.arrays[0].shape) + 1

        return out

    def map_concatenate(self, expr: Concatenate) -> Array:
        new_arrays, new_axes = self._broadcast_predecessors_to_same_shape(expr)

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
                                                 in enumerate(sorted(new_binds.items()))} # noqa E501

        # Determine the new parameter studies that are being conducted.
        canonical_studies = self._get_canonical_ordered_studies(new_arrays)
        postpend_shapes, post_axes = self._canonical_ordered_studies_to_shapes_and_axes(canonical_studies) # noqa E501

        varname_to_studies_nums: dict[str, tuple[int, ...]] = {bnd_name: () for 
                                                              _, bnd_name
                                                 in array_num_to_bnd_name.items()}

        for iarr, array in enumerate(new_arrays):
            for axis in array.axes:
                tags = axis.tags_of_type(ParameterStudyAxisTag)
                if tags:
                    assert len(tags) == 1
                    study: ParameterStudyAxisTag = next(iter(tags))
                    name: str = array_num_to_bnd_name[iarr]
                    varname_to_studies_nums[name] = (*varname_to_studies_nums[name],
                                                     canonical_studies.index(study),)

        #varname_to_studies_nums = {array_num_to_bnd_name[iarr]: studies for iarr,
        #                      studies in arr_num_to_study_nums.items()}

        assert all(vn_key in new_binds.keys() for
                   vn_key in varname_to_studies_nums.keys())

        assert all(vn_key in varname_to_studies_nums.keys() for
                   vn_key in new_binds.keys())

        # Now we need to update the expressions.
        scalar_expr_mapper = IndexLambdaScalarExpressionVectorizer(varname_to_studies_nums, len(expr.shape)) # noqa E501

        return IndexLambda(expr=scalar_expr_mapper(expr.expr),
                           bindings=immutabledict(new_binds),
                           shape=(*expr.shape, *postpend_shapes,),
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           dtype=expr.dtype,
                           axes=(*expr.axes, *post_axes,),
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_einsum(self, expr: Einsum) -> Array:

        new_predecessors = tuple(self.rec(arg) for arg in expr.args)
        canonical_studies = self._get_canonical_ordered_studies(new_predecessors)
        studies_shape, end_axes = self._canonical_ordered_studies_to_shapes_and_axes(canonical_studies) # noqa E501

        access_descriptors: tuple[tuple[EinsumAxisDescriptor, ...], ...] = ()
        for ival, array in enumerate(new_predecessors):
            one_descr = expr.access_descriptors[ival]
            for axis in array.axes:
                tags = axis.tags_of_type(ParameterStudyAxisTag)
                if tags:
                    # Need to append a descriptor
                    assert len(tags) == 1
                    study: ParameterStudyAxisTag = next(iter(tags))
                    one_descr = (*one_descr,
                                 EinsumElementwiseAxis(dim=len(expr.shape)
                                         + canonical_studies.index(study)))

            access_descriptors = (*access_descriptors, one_descr)

            # One descriptor per axis.
            assert len(one_descr) == len(array.shape)

        return Einsum(access_descriptors,
                     new_predecessors,
                     axes=(*expr.axes, *end_axes,),
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
