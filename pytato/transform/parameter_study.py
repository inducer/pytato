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

from immutabledict import immutabledict
from dataclasses import dataclass
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pytato.array import (
    Array,
    AxesT,
    Axis,
    AxisPermutation,
    Concatenate,
    Einsum,
    IndexBase,
    IndexLambda,
    NormalizedSlice,
    Placeholder,
    Reshape,
    Roll,
    Stack,
)

from pytato.scalar_expr import IdentityMapper, IntegralT

import pymbolic.primitives as prim

from pytools.tag import UniqueTag, Tag

from pytato.transform import CopyMapper


@dataclass(frozen=True)
class ParameterStudyAxisTag(UniqueTag):
    """
        A tag for acting on axes of arrays.
        To enable multiple parameter studies on the same variable name
        specify a different axis number and potentially a different size.

        Currently does not allow multiple variables of different names to be in
        the same parameter study.
    """
    axis_num: int
    axis_size: int


StudiesT = Tuple[ParameterStudyAxisTag, ...]
ArraysT = Tuple[Array, ...]
KnownShapeType = Tuple[IntegralT, ...]


class ExpansionMapper(CopyMapper):

    def __init__(self, placeholder_name_to_parameter_studies: Mapping[str, StudiesT]):
        super().__init__()
        self.placeholder_name_to_parameter_studies = placeholder_name_to_parameter_studies  # noqa

    def _shapes_and_axes_from_predecessor(self, curr_expr: Array,
                                         mapped_preds: ArraysT) -> \
                                                 Tuple[KnownShapeType,
                                                       AxesT,
                                                       Dict[Array, Tuple[int, ...]]]:
        # Initialize with something for the typing.

        assert not any(axis.tags_of_type(ParameterStudyAxisTag) for
                       axis in curr_expr.axes)

        # We are post pending the axes we are using for parameter studies.
        new_shape: KnownShapeType = ()
        studies_axes: AxesT = ()

        study_to_arrays: Dict[FrozenSet[ParameterStudyAxisTag], ArraysT] = {}

        active_studies: Set[ParameterStudyAxisTag] = set()

        for arr in mapped_preds:
            for axis in arr.axes:
                tags = axis.tags_of_type(ParameterStudyAxisTag)
                if tags:
                    active_studies = active_studies.union(tags)
                    if tags in study_to_arrays.keys():
                        study_to_arrays[tags] = (*study_to_arrays[tags], arr)
                    else:
                        study_to_arrays[tags] = (arr,)

        return self._studies_to_shape_and_axes_and_arrays_in_canonical_order(active_studies, # noqa
                                                                             new_shape,
                                                                             studies_axes,
                                                                             study_to_arrays)

    def _studies_to_shape_and_axes_and_arrays_in_canonical_order(self,
                    studies: Iterable[ParameterStudyAxisTag],
                    new_shape: KnownShapeType, new_axes: AxesT,
                    study_to_arrays: Dict[FrozenSet[ParameterStudyAxisTag], ArraysT]) \
                            -> Tuple[KnownShapeType, AxesT, Dict[Array,
                                                                 Tuple[int, ...]]]:

        # This is where we specify the canonical ordering.

        array_to_canonical_ordered_studies: Dict[Array, Tuple[int, ...]] = {}
        studies_axes = new_axes

        for ind, study in enumerate(sorted(studies,
                                           key=lambda x: str(x.__class__))):
            new_shape = (*new_shape, study.axis_size)
            studies_axes = (*studies_axes, Axis(tags=frozenset((study,))))
            print(study_to_arrays)
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
        _, new_axes, _ = self._shapes_and_axes_from_predecessor(expr,
                                                                (new_predecessor,))

        return Roll(array=new_predecessor,
                    shift=expr.shift,
                    axis=expr.axis,
                    axes=expr.axes + new_axes,
                    tags=expr.tags,
                    non_equality_tags=expr.non_equality_tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        new_predecessor = self.rec(expr.array)
        postpend_shape, new_axes, _ = self._shapes_and_axes_from_predecessor(expr,
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
        postpend_shape, new_axes, _ = self._shapes_and_axes_from_predecessor(expr,
                                                                             (new_predecessor,))
        # Update the indicies.
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
        postpend_shape, new_axes, _ = self._shapes_and_axes_from_predecessor(expr,
                                                                             (new_predecessor,))
        return Reshape(new_predecessor,
                       newshape=self.rec_idx_or_size_tuple(expr.newshape +
                                                           postpend_shape),
                       order=expr.order,
                       axes=expr.axes + new_axes,
                       tags=expr.tags,
                       non_equality_tags=expr.non_equality_tags)

        # {{{ Operations with multiple predecessors.

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

    def _mult_pred_same_shape(self, expr: Union[Stack, Concatenate]) -> Tuple[ArraysT,
                                                                              AxesT]:

        new_predecessors = tuple(self.rec(arr) for arr in expr.arrays)

        studies_shape, new_axes, arrays_to_study_num_present = self._shapes_and_axes_from_predecessor(expr, new_predecessors) # noqa

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
            corrected_new_arrays = (*corrected_new_arrays, tmp)

        return corrected_new_arrays, new_axes

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        # Update bindings first.
        new_bindings: Dict[str, Array] = {name: self.rec(bnd)
                                             for name, bnd in
                                          sorted(expr.bindings.items())}

        # Determine the new parameter studies that are being conducted.
        from pytools import unique

        all_axis_tags: StudiesT = ()
        varname_to_studies: Dict[str, Dict[UniqueTag, bool]] = {}
        for name, bnd in sorted(new_bindings.items()):
            axis_tags_for_bnd: Set[Tag] = set()
            varname_to_studies[name] = {}
            for i in range(len(bnd.axes)):
                axis_tags_for_bnd = axis_tags_for_bnd.union(bnd.axes[i].tags_of_type(ParameterStudyAxisTag)) # noqa
            for tag in axis_tags_for_bnd:
                if isinstance(tag, ParameterStudyAxisTag):
                    # Defense
                    varname_to_studies[name][tag] = True
                    all_axis_tags = *all_axis_tags, tag,

        cur_studies: Sequence[ParameterStudyAxisTag] = list(unique(all_axis_tags))
        study_to_axis_number: Dict[ParameterStudyAxisTag, int] = {}

        new_shape = expr.shape
        new_axes = expr.axes

        for study in cur_studies:
            if isinstance(study, ParameterStudyAxisTag):
                # Just defensive programming
                # The active studies are added to the end of the bindings.
                study_to_axis_number[study] = len(new_shape)
                new_shape = (*new_shape, study.axis_size,)
                new_axes = (*new_axes, Axis(tags=frozenset((study,))),)
                #  This assumes that the axis only has 1 tag,
                #  because there should be no dependence across instances.

        # Now we need to update the expressions.
        scalar_expr = ParamAxisExpander()(expr.expr, varname_to_studies,
                                          study_to_axis_number)

        return IndexLambda(expr=scalar_expr,
                           bindings=immutabledict(new_bindings),
                           shape=new_shape,
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           dtype=expr.dtype,
                           axes=new_axes,
                           tags=expr.tags,
                           non_equality_tags=expr.non_equality_tags)

    def map_einsum(self, expr: Einsum) -> Array:

        return super().map_einsum(expr)

    # }}} Operations with multiple predecessors.


class ParamAxisExpander(IdentityMapper):

    def map_subscript(self, expr: prim.Subscript,
                      varname_to_studies: Mapping[str,
                                                  Mapping[ParameterStudyAxisTag, bool]],
                      study_to_axis_number: Mapping[ParameterStudyAxisTag, int]) -> \
                                                                    prim.Subscript:
        # We know that we are not changing the variable that we are indexing into.
        # This is stored in the aggregate member of the class Subscript.

        # We only need to modify the indexing which is stored in the index member.
        name = expr.aggregate.name
        if name in varname_to_studies.keys():
            #  These are the single instance information.
            index = self.rec(expr.index, varname_to_studies,
                             study_to_axis_number)

            new_vars: Tuple[prim.Variable, ...] = ()

            for key, num in sorted(study_to_axis_number.items(),
                                   key=lambda item: item[1]):
                if key in varname_to_studies[name]:
                    new_vars = *new_vars, prim.Variable(f"_{num}"),

            if isinstance(index, tuple):
                index = index + new_vars
            else:
                index = tuple(index) + new_vars
            return type(expr)(aggregate=expr.aggregate, index=index)
        return expr
