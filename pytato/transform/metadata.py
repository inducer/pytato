"""
.. currentmodule:: pytato

.. autofunction:: unify_axes_tags

.. currentmodule:: pytato.transform.metadata

.. autoclass:: AxisTagAttacher

.. autoclass:: AxisIgnoredForPropagationTag

.. autoclass:: AxesTagsEquationCollector
"""
from __future__ import annotations


__copyright__ = """
Copyright (C) 2022 Kaushik Kulkarni
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

import logging
import re
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeAlias,
    Never,
    TypeVar,
    cast,
)

from bidict import bidict

import pymbolic.primitives as prim
from pytools import UniqueNameGenerator
from pytools.tag import Tag

from pytato.array import (
    AbstractResultWithNamedArrays,
    AdvancedIndexInContiguousAxes,
    Array,
    AxisPermutation,
    BasicIndex,
    Concatenate,
    DictOfNamedArrays,
    Einsum,
    EinsumReductionAxis,
    IndexLambda,
    InputArgumentBase,
    NamedArray,
    NormalizedSlice,
    Reshape,
    Stack,
)
from pytato.distributed.nodes import DistributedRecv, DistributedSendRefHolder
from pytato.function import NamedCallResult
from pytato.scalar_expr import (
    IDX_LAMBDA_RESERVED_INDEX_PATTERN,
    CombineMapper,
    get_dependencies as get_dependencies_scalar
)
from pytato.scalar_expr import SCALAR_CLASSES
from pytato.transform import ArrayOrNames, CopyMapper, Mapper, TransformMapperCache
from pytato.utils import are_shape_components_equal, are_shapes_equal


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

    from pytato.function import FunctionDefinition, NamedCallResult
    from pytato.loopy import LoopyCall


GraphNodeT = TypeVar("GraphNodeT")

BindingName: TypeAlias = str
P = ParamSpec("P")

BINDING_NAME_RESERVED_PATTERN = re.compile(r"^(_in?(0|([1-9][0-9]*)))$")


# {{{ BindingSubscriptsCollector


class BindingSubscriptsCollector(CombineMapper[dict[BindingName,
                                               set[tuple[prim.Expression, ...]]],
                                                      []]):
    """
    Return all the subscript expressions used by a variable specified by BindingName.
    Ex:
    _in1[_0,_1] would result in an dictionary entry {"_in1": ("_0", "_1")}.
    """
    def combine(self,
                values: Iterable[dict[BindingName,
                                         set[tuple[prim.Expression, ...]]]]) \
                        -> dict[BindingName, set[tuple[prim.Expression, ...]]]:
        out: dict[BindingName, set[tuple[prim.Expression, ...]]] = {}
        import operator
        from functools import reduce
        return reduce(operator.or_, values, out)

    def map_subscript(self, expr: prim.Subscript) -> dict[BindingName,
                                                    set[tuple[prim.Expression, ...]]]:
        """
        Record the indexing expression if the Subscript expression has a prim.Variable
        as its aggregate.
        """

        if isinstance(expr.aggregate, prim.Variable):
            name = expr.aggregate.name
            base: dict[BindingName,
                       set[tuple[prim.Expression, ...]]] = {name: {expr.index_tuple}}

            """
            for ind, subexpr in enumerate(expr.index_tuple):
                sub = self.rec(subexpr)
                if sub:
                    # we have nested subscripts.
                    for key, val in sub.items():
                        # The new key will be comma separated.
                        newkey = name + "," + str(ind) + "," + key
                        base.update({newkey: val})
            """
            return self.combine([base] + [self.rec(subexpr) for _, subexpr in enumerate(expr.index_tuple)])
            #return {expr.aggregate.name: {expr.index_tuple}}
        return {}

    def map_algebraic_leaf(self, expr: prim.Expression) -> dict[BindingName,
                                                    set[tuple[prim.Expression, ...]]]:

        return {}

    def map_constant(self, expr: object) -> dict[BindingName,
                                                    set[tuple[prim.Expression, ...]]]:
        return {}
# }}}

# {{{ AxesTagsEquationCollector

class AxesTagsEquationCollector(Mapper[None, Never, []]):
    r"""
    Records equations arising from operand/output axes equivalence for an array
    operation. This mapper implements a default set of propagation rules,
    refer to documentation of mapper methods for their propagation semantics.

    .. attribute:: tag_t

        The type of the tags that are to be propagated.

    .. attribute:: equations

        A :class:`list` of equations. Each equation is represented by 2-tuple
        as ``("u", "v")`` that is mathematically interpreted as
        :math:`\{u \doteq v\}`.

    .. attribute:: known_tag_to_var

        A mapping from an instance of :class:`Tag` to a :class:`str` by which
        it will be referenced in :attr:`equations`.

    .. attribute:: axis_to_var

        A :class:`~bidict.bidict` from a :class:`tuple` of the form ``(array,
        iaxis)`` to the :class:`str` by which it will be referenced in
        :attr:`equations`.

    .. automethod:: map_index_lambda
    .. automethod:: map_placeholder
    .. automethod:: map_data_wrapper
    .. automethod:: map_size_param
    .. automethod:: map_reshape
    .. automethod:: map_basic_index
    .. automethod:: map_contiguous_advanced_index
    .. automethod:: map_stack
    .. automethod:: map_concatenate

    .. note::

        Users are encouraged to derive this mapper to implement domain-specific
        axis tags propagation semantics.
    """
    def __init__(self, tag_t: type[Tag]) -> None:
        self.tag_t: type[Tag] = tag_t
        super().__init__()

        # {{{ mutable state held by the mapper

        self._visited_nodes: set[ArrayOrNames] = set()

        self.var_name_gen: UniqueNameGenerator = UniqueNameGenerator()
        self.var_name_gen.add_names(["c", ""])

        # axis_to_var: mapping from (array, iaxis) to the variable to be
        # used for unification.
        self.axis_to_var: bidict[tuple[Array, int | str], str] = bidict()
        self.known_tag_to_var: dict[Tag, str] = {}

        self.equations: list[tuple[str, str]] = []

        # }}}

    # {{{ unification helpers

    def get_var_for_axis(self, ary: Array, iaxis: int | str) -> str:
        key = (ary, iaxis)

        try:
            return self.axis_to_var[key]
        except KeyError:
            new_var = self.var_name_gen("ax")
            self.axis_to_var[key] = new_var
            return new_var

    def get_var_for_tag(self, tag: Tag) -> str:
        key = tag
        try:
            return self.known_tag_to_var[key]
        except KeyError:
            new_var = self.var_name_gen("tag")
            self.known_tag_to_var[key] = new_var
            return new_var

    def record_equation(self, lhs: str, rhs: str) -> None:
        r"""
        Adds the equation :math:`\{\text{lhs}\doteq\text{rhs}}` to
        :attr:`equations`.
        """
        self.equations.append((lhs, rhs))

    def record_equations_from_axes_tags(self, ary: Array) -> None:
        """
        Records equations for *ary*\'s axis tags of type :attr:`tag_t`.
        """
        for iaxis, axis in enumerate(ary.axes):
            lhs_var = self.get_var_for_axis(ary, iaxis)
            for tag in axis.tags_of_type(self.tag_t):
                rhs_var = self.get_var_for_tag(tag)
                self.record_equation(lhs_var, rhs_var)

    # }}}

    # {{{ mapper interface

    def rec(self, expr: ArrayOrNames) -> None:
        if expr in self._visited_nodes:
            return

        if isinstance(expr, Array):
            self.record_equations_from_axes_tags(expr)

        super().rec(expr)
        self._visited_nodes.add(expr)

    def _map_input_base(self, expr: InputArgumentBase) -> None:
        """
        A :class:`pytato.InputArgumentBase` does not have any operands i.e. no
        propagation equations are recorded.
        """

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_index_lambda(self, expr: IndexLambda) -> None:
        """
        Equality conditions are added between an axis of the operands which is indexed
        by a :class:`~pymbolic.Variable` which has a name that follows the reserved
        iname format, "_[0-9]+", and the axis of the output specified by the iname.
        """
        for bnd in expr.bindings.values():
            breakpoint()
            self.rec(bnd)

        index_expr_used = BindingSubscriptsCollector()(expr.expr)

        

        breakpoint()
        for vname, set_of_ind_tuple in index_expr_used.items():
            for ind_tuple in set_of_ind_tuple:
                for axis_ind, var_ind_name in enumerate(ind_tuple):

                    variables_used = get_dependencies_scalar(var_ind_name)
                    if isinstance(var_ind_name, prim.Variable):
                        lhs: str = self.get_var_for_axis(expr.bindings[vname],
                                                     axis_ind)
                        matched_pattern = IDX_LAMBDA_RESERVED_INDEX_PATTERN.fullmatch(var_ind_name.name)
                        if matched_pattern:
                            # matched with an axis index.
                            self.record_equation(lhs, self.get_var_for_axis(expr,
                                                  int(matched_pattern.group("index"))))
                        elif var_ind_name.name in expr.var_to_reduction_descr.keys():
                            # matched with a reduction axis.
                            # We are assuming that this axis is eliminated from the
                            # axes of the output array. So, the metadata will only be keep
                            # in the reduction descriptor object which is indexed by the
                            # var_ind_name.name
                            self.record_equation(lhs,
                                self.get_var_for_axis(expr, var_ind_name.name))

                        elif BINDING_NAME_RESERVED_PATTERN.fullmatch(var_ind_name.name):
                            # This means that we had an index of index.
                            # So, the metadata propagation with this index is data
                            # dependent.
                            pass
                        else:
                            pass
                            #warning("Variable does not match an index pattern. It will
                            #be ignored for metadata propagation.")

                    # We need to add an equation if the index name is the only variable
                    # for that axis. This includes if there is scaled indexing.
                    for ind_name in variables_used:
                        breakpoint()
                        lhs: str = self.get_var_for_axis(expr.bindings[vname],
                                                         axis_ind)
                        matched_pattern = IDX_LAMBDA_RESERVED_INDEX_PATTERN.fullmatch(ind_name)
                        if matched_pattern:
                            # matched with an axis index of the output.
                            self.record_equation(lhs, self.get_var_for_axis(expr,
                                                  int(matched_pattern.group("index"))))
                        elif ind_name in expr.var_to_reduction_descr.keys():
                            # matched with a reduction axis.
                            # We are assuming that this axis is eliminated from the
                            # axes of the output array. So, the metadata will only be keep
                            # in the reduction descriptor object which is indexed by the
                            # var_ind_name.name
                            self.record_equation(lhs,
                                self.get_var_for_axis(expr, ind_name))


        return

    def map_stack(self, expr: Stack) -> None:
        """
        Records an equality equation between the axes of arrays being stacked
        and their corresponding axis in *expr*. No equation is added for the
        newly created axis i.e. :attr:`pytato.array.Stack.axis`.
        """
        raise NotImplementedError
        for ary in expr.arrays:
            self.rec(ary)

        for iaxis in range(expr.ndim):
            for ary in expr.arrays:
                if iaxis < expr.axis:
                    self.record_equation(
                        self.get_var_for_axis(ary, iaxis),
                        self.get_var_for_axis(expr, iaxis)
                    )
                elif iaxis == expr.axis:
                    pass
                elif iaxis > expr.axis:
                    self.record_equation(
                        self.get_var_for_axis(ary, iaxis-1),
                        self.get_var_for_axis(expr, iaxis)
                    )
                else:
                    raise AssertionError

    def map_concatenate(self, expr: Concatenate) -> None:
        """
        Records an equality equation between the axes of arrays being
        concatenated and their corresponding axis in *expr*. No equation is
        added for the concatenated axis i.e.
        :attr:`pytato.array.Concatenate.axis`.
        """
        raise NotImplementedError
        for ary in expr.arrays:
            self.rec(ary)

        for ary in expr.arrays:
            assert ary.ndim == expr.ndim
            for iaxis in range(expr.ndim):
                if iaxis != expr.axis:
                    # non-concatenated axes share the dimensions.
                    self.record_equation(
                        self.get_var_for_axis(ary, iaxis),
                        self.get_var_for_axis(expr, iaxis)
                    )

    def map_axis_permutation(self, expr: AxisPermutation
                             ) -> None:
        """
        Records an equality equation for every axis of *expr*\'s operand and
        its corresponding axis in *expr* as specified by
        :attr:`pytato.array.AxisPermutation.axis_permutation`.
        """
        raise NotImplementedError
        self.rec(expr.array)

        assert expr.ndim == expr.array.ndim

        for out_axis in range(expr.ndim):
            in_axis = expr.axis_permutation[out_axis]
            self.record_equation(
                self.get_var_for_axis(expr, out_axis),
                self.get_var_for_axis(expr.array, in_axis)
            )

    def map_basic_index(self, expr: BasicIndex) -> None:
        """
        Records an equality equation for each trivially sliced axis of the
        array being indexed and its corresponding axis in *expr*. A trivially
        sliced axis is one which goes along the entire length of the axis with
        a positive unit stride.
        """
        raise NotImplementedError
        self.rec(expr.array)

        i_out_axis = 0

        assert len(expr.indices) == expr.array.ndim

        for i_in_axis, idx in enumerate(expr.indices):
            if isinstance(idx, int):
                pass
            else:
                assert isinstance(idx, NormalizedSlice)
                if (idx.step == 1
                        and are_shape_components_equal(idx.start, 0)
                        and are_shape_components_equal(idx.stop,
                                                       expr.array.shape[i_in_axis])):

                    self.record_equation(
                        self.get_var_for_axis(expr.array, i_in_axis),
                        self.get_var_for_axis(expr, i_out_axis)
                    )

                i_out_axis += 1

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> None:
        """
        For sliced indices adds all the equations as prescribed by
        :meth:`AxesTagsEquationCollector.map_basic_index`. For the advanced
        indices adds an equality equation for each non-broadcasted axis of an
        indexing array to its corresponding axis in *expr*.
        """
        raise NotImplementedError
        from pytato.utils import get_shape_after_broadcasting, partition

        self.rec(expr.array)
        for idx in expr.indices:
            if isinstance(idx, Array):
                self.rec(idx)

        i_adv_indices, i_basic_indices = partition(
            lambda idx: isinstance(expr.indices[idx], NormalizedSlice),
            range(len(expr.indices)))
        npre_advanced_basic_indices = len([i_idx
                                           for i_idx in i_basic_indices
                                           if i_idx < i_adv_indices[0]])
        npost_advanced_basic_indices = len([i_idx
                                            for i_idx in i_basic_indices
                                            if i_idx > i_adv_indices[-1]])

        indirection_arrays: list[Array] = cast("list[Array]",
                                               [expr.indices[i_idx]
                                                for i_idx in i_adv_indices
                                                if isinstance(expr.indices[i_idx],
                                                           Array)
                                                ])

        assert are_shapes_equal(
            get_shape_after_broadcasting(indirection_arrays),
            expr.shape[
                npre_advanced_basic_indices:expr.ndim-npost_advanced_basic_indices])

        # {{{ add equations from indirection arrays with the output

        for subexpr in indirection_arrays:
            for i_in_axis, i_out_axis in zip(
                    range(subexpr.ndim),
                    range(expr.ndim
                          - npost_advanced_basic_indices
                          - subexpr.ndim,
                          expr.ndim-npost_advanced_basic_indices),
                    strict=True):
                in_dim = subexpr.shape[i_in_axis]
                out_dim = expr.shape[i_out_axis]
                if are_shape_components_equal(in_dim, out_dim):
                    self.record_equation(
                        self.get_var_for_axis(subexpr, i_in_axis),
                        self.get_var_for_axis(expr, i_out_axis))
                else:
                    # broadcasted axes, cannot belong to the same
                    # discretization entity.
                    assert are_shape_components_equal(in_dim, 1)
        # }}}

        # {{{ add equations from slices in indexed array's axes to output axes

        for i_in_axis, idx in enumerate(expr.indices[:npre_advanced_basic_indices]):
            assert isinstance(idx, NormalizedSlice)
            if (idx.step == 1
                    and are_shape_components_equal(idx.start, 0)
                    and are_shape_components_equal(idx.stop,
                                                   expr.array.shape[i_in_axis])):
                assert are_shape_components_equal(expr.shape[i_in_axis],
                                                  expr.array.shape[i_in_axis])
                self.record_equation(
                    self.get_var_for_axis(expr.array, i_in_axis),
                    self.get_var_for_axis(expr, i_in_axis))

        for i, idx in enumerate(
                expr.indices[expr.array.ndim-npost_advanced_basic_indices:]):
            i_in_axis = i + (expr.array.ndim - npost_advanced_basic_indices)
            i_out_axis = i + (expr.ndim - npost_advanced_basic_indices)
            assert isinstance(idx, NormalizedSlice)
            if (idx.step == 1
                    and are_shape_components_equal(idx.start, 0)
                    and are_shape_components_equal(idx.stop,
                                                   expr.array.shape[i_in_axis])):
                assert are_shape_components_equal(expr.shape[i_out_axis],
                                                  expr.array.shape[i_in_axis])
                self.record_equation(
                    self.get_var_for_axis(expr.array, i_in_axis),
                    self.get_var_for_axis(expr, i_out_axis))
        # }}}

    def map_reshape(self, expr: Reshape) -> None:
        """
        Reshaping generally does not preserve the axis between its input and
        output and so no constraints are enforced except when the
        :class:`pytato.Reshape` has come from a :func:`pytato.expand_dims`.
        """
        raise NotImplementedError
        from pytato.tags import ExpandedDimsReshape

        self.rec(expr.array)

        expand_dims_tags = expr.tags_of_type(ExpandedDimsReshape)

        if expand_dims_tags:
            expand_dims_tag, = expand_dims_tags
            i_in_axis = 0
            for i_out_axis in range(expr.ndim):
                if i_out_axis not in expand_dims_tag.new_dims:
                    self.record_equation(
                        self.get_var_for_axis(expr.array, i_in_axis),
                        self.get_var_for_axis(expr, i_out_axis)
                    )
                    i_in_axis += 1

            assert i_in_axis == expr.array.ndim

    def map_einsum(self, expr: Einsum) -> None:
        """
        Equality conditions are added between axes of the operands and outputs
        that have the same index when instantiated through
        :func:`pytato.einsum` thereby having the same the
        :class:`~pytato.array.EinsumAxisDescriptor`.
        """
        raise NotImplementedError
        from pytato.array import EinsumAxisDescriptor, EinsumElementwiseAxis

        for arg in expr.args:
            self.rec(arg)

        descr_to_var: dict[EinsumAxisDescriptor, str] = {}
        for iaxis in range(expr.ndim):
            descr_to_var[EinsumElementwiseAxis(iaxis)] = self.get_var_for_axis(expr,
                                                                               iaxis)

        for access_descrs, arg in zip(expr.access_descriptors,
                                      expr.args, strict=True):
            for iarg_axis, descr in enumerate(access_descrs):
                in_tag_var = self.get_var_for_axis(arg, iarg_axis)

                if descr in descr_to_var:
                    self.record_equation(descr_to_var[descr], in_tag_var)
                else:
                    descr_to_var[descr] = in_tag_var

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        raise NotImplementedError
        for _, subexpr in sorted(expr._data.items()):
            self.rec(subexpr)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        """
        Does not add any equations.
        """
        raise NotImplementedError
        for _, subexpr in sorted(expr.bindings.items()):
            if isinstance(subexpr, Array):
                self.rec(subexpr)

        # there's really no good way to propagate the metadata in this case.
        # One *could* raise the loopy kernel instruction expressions to
        # high level ops, but that's quite involved and probably not worth it.

    def map_named_array(self, expr: NamedArray) -> None:
        raise NotImplementedError
        self.rec(expr._container)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> None:
        """
        Since the value held by a :class:`pytato.DistributedSendRefHolder`
        is the value held by
        :attr:`pytato.DistributedSendRefHolder.passthrough_data`, equality
        equations are added between each axis of *expr* and its corresponding
        axis in the pass-through data.
        """
        raise NotImplementedError
        self.rec(expr.passthrough_data)
        self.rec(expr.send.data)
        for idim in range(expr.ndim):
            self.record_equation(
                self.get_var_for_axis(expr.passthrough_data, idim),
                self.get_var_for_axis(expr, idim),
            )

    def map_distributed_recv(self,
                             expr: DistributedRecv) -> None:
        """
        :class:`pytato.DistributedRecv` does not have any operands and so no
        more equations are deduced.
        """
        raise NotImplementedError

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        raise NotImplementedError(
            "AxesTagsEquationCollector does not currently support expressions "
            "containing functions.")

    # }}}

# }}}


# {{{ AxisTagAttacher

class AxisTagAttacher(CopyMapper):
    """
    A mapper that tags the axes in a DAG as prescribed by *axis_to_tags*.
    """
    def __init__(self,
                 axis_to_tags: Mapping[tuple[Array, int | str], Collection[Tag]],
                 tag_corresponding_redn_descr: bool,
                 _cache: TransformMapperCache[ArrayOrNames] | None = None,
                 _function_cache:
                    TransformMapperCache[FunctionDefinition] | None = None):
        super().__init__(_cache=_cache, _function_cache=_function_cache)
        self.axis_to_tags: Mapping[tuple[Array, int | str], Collection[Tag]] = axis_to_tags
        self.tag_corresponding_redn_descr: bool = tag_corresponding_redn_descr

    def _attach_tags(self, expr: Array, rec_expr: Array) -> Array:
        assert rec_expr.ndim == expr.ndim

        result = rec_expr

        for iaxis in range(expr.ndim):
            result = result.with_tagged_axis(
                iaxis, self.axis_to_tags.get((expr, iaxis), []))

        # {{{ tag reduction descrs

        if self.tag_corresponding_redn_descr:
            if isinstance(expr, Einsum):
                assert isinstance(result, Einsum)
                for arg, access_descrs in zip(expr.args,
                                              expr.access_descriptors,
                                              strict=True):
                    for iaxis, access_descr in enumerate(access_descrs):
                        if isinstance(access_descr, EinsumReductionAxis):
                            result = result.with_tagged_reduction(
                                access_descr,
                                self.axis_to_tags.get((arg, iaxis), [])
                            )

            if isinstance(expr, IndexLambda):
                assert isinstance(result, IndexLambda)
                if expr_copy.var_to_reduction_descr:
                    # This is a reduction operation.
                    # We need to find the axes that are reduced over
                    # and update the tag/tag them appropriately.
                    for redn_var in expr.var_to_reduction_descr.keys():
                        result = result.with_tagged_reduction(
                            redn_var,
                            self.axis_to_tags.get((expr, redn_var), [])
                        )

        # }}}

        return result

    def rec(self, expr: ArrayOrNames) -> ArrayOrNames:
        key = self._cache.get_key(expr)
        try:
            return self._cache.retrieve(expr, key=key)
        except KeyError:
            result = Mapper.rec(self, expr)
            if not isinstance(
                    expr, AbstractResultWithNamedArrays | DistributedSendRefHolder):
                assert isinstance(expr, Array)
                # type-ignore reason: passed "ArrayOrNames"; expected "Array"
                result = self._attach_tags(expr, result)  # type: ignore[arg-type]
            return self._cache.add(expr, result, key=key)

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        raise NotImplementedError(
            "AxisTagAttacher does not currently support expressions containing "
            "functions.")

# }}}


class AxisIgnoredForPropagationTag(Tag):
    """
    Disallows tags from propagating across axes equipped with this tag.
    Effectively removes an edge from the undirected graph whose edges represent
    propagation pathways. The default tag propagation behavior in the case
    of an einsum is to propagate all tags across non-reduction axes. Since this
    is not always desirable, this tag can be used to disable the default
    behavior at axis-granularity.
    """
    pass


def unify_axes_tags(
        expr: ArrayOrNames,
        *,
        tag_t: type[Tag] = Tag,
        equations_collector_t: type[
            AxesTagsEquationCollector] = AxesTagsEquationCollector,
        unify_redn_descrs: bool = True,
) -> ArrayOrNames:
    r"""
    Returns a copy of *expr* with tags of type *tag_t* propagated along the
    array operations with the tags propagation semantics implemented in
    *equations_collector_t*. By propagation, we mean that we solve the
    unification equations assembled in
    :attr:`AxesTagsEquationCollector.equations` to obtain a mapping from an
    :class:`~pytato.Array`\ 's axis to the new tags it is to be tagged with. We
    use this mapping to add more tags to an array's axis.

    .. note::

        - This routine by itself does not raise if a particular array's axis is
          tagged with multiple tags of type *tag_t*. If such behavior is not
          expected, ensure that *tag_t* is a subclass of
          :class:`~pytools.tag.UniqueTag`.
    """
    equations_collector = equations_collector_t(tag_t)

    # First we will convert the expression to a series of IndexLambda operations.

    from pytato.transform.lower_to_index_lambda import ToIndexLambdaMixin, to_index_lambda
    from pytato.transform import TransformMapper
    from pytato.diagnostic import CannotBeLoweredToIndexLambda
    mapped_expr = to_index_lambda(expr)

    class MyIndexMapper(TransformMapper, ToIndexLambdaMixin):
        def handle_unsupported_array(self, expr: Any) -> Any:
            raise CannotBeLoweredToIndexLambda(type(expr))

        def map_placeholder(self, expr: Placeholder) -> Placeholder:
            return expr


    mymapper = MyIndexMapper()
    mapped_expr = mymapper(expr)

    breakpoint()
    equations_collector(mapped_expr)

    # start BFS traversal with the known tags as the sources.
    # From the equations build a Propagation Graph
    # Defn. A Propagation graph is a graph where nodes denote variables and an
    # edge between 2 nodes denotes an equality criterion.

    from pytools.graph import (
        get_reachable_nodes,
        undirected_graph_from_edges,
    )

    known_tag_vars = frozenset(equations_collector.known_tag_to_var.values())

    # Reduction axes are specified by a str but all other axes are specified
    # by an integer. Note that the axes are still uniquely identified.
    axis_to_solved_tags: dict[tuple[Array, int | str], set[Tag]] = {}

    propagation_graph = undirected_graph_from_edges(
        equations_collector.equations
    )

    ignored_vars = set({
        tag_var for tag, tag_var in equations_collector.known_tag_to_var.items()
        if isinstance(tag, AxisIgnoredForPropagationTag)
    })

    for (ary, ax), ax_var in equations_collector.axis_to_var.items():
        # Reduction axes do not follow AxisIgnoredForPropagation.
        # They cannot propagate the information to descendant of the array anyway.
        if isinstance(ax, int):
            if ary.axes[ax].tags_of_type(AxisIgnoredForPropagationTag):
                ignored_vars.update({ax_var})

    for tag, var in equations_collector.known_tag_to_var.items():
        reachable_nodes = get_reachable_nodes(propagation_graph, var,
                                              exclude_nodes=ignored_vars)
        for reachable_var in (reachable_nodes - known_tag_vars):
            axis_to_solved_tags.setdefault(
                equations_collector.axis_to_var.inverse[reachable_var],
                set()
            ).add(tag)

    return AxisTagAttacher(axis_to_solved_tags,
                           tag_corresponding_redn_descr=unify_redn_descrs,
                           )(mapped_expr)

# vim: fdm=marker
