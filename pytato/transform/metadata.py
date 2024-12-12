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
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
)

from bidict import bidict

import pymbolic.primitives as prim
from pymbolic.typing import Expression
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
    IDX_LAMBDA_INAME,
    IDX_LAMBDA_JUST_REDUCTIONS,
    CombineMapper,
)
from pytato.transform import ArrayOrNames, CopyMapper, Mapper
from pytato.utils import are_shape_components_equal, are_shapes_equal


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from pytato.function import NamedCallResult
    from pytato.loopy import LoopyCall


GraphNodeT = TypeVar("GraphNodeT")

BindingName: TypeAlias = str
P = ParamSpec("P")


class IndexExpressionsUsedInIndexLambda(CombineMapper[Mapping[BindingName,
                                                     set[tuple[Expression, ...]]],
                                                      []]):
    """
    Determine which axes are used in the scalar expressionand which ones just
    flow through the expression.
    """
    def combine(self,
                values: Iterable[Mapping[BindingName, set[tuple[Expression, ...]]]]) \
                        -> Mapping[BindingName, set[tuple[Expression, ...]]]:
        out: dict[BindingName, set[tuple[Expression, ...]]] = {}
        for val in values:
            out.update(val)
        return out

    def map_subscript(self, expr: prim.Subscript) -> Mapping[BindingName,
                                                    set[tuple[Expression, ...]]]:
        """
        Record the indexing usage for the variable if we are tracking
        the specific variable.
        """

        if isinstance(expr.aggregate, prim.Variable):
            name: BindingName = expr.aggregate.name
            base = {name: set(expr.index_tuple)}
            return self.combine([base, self.rec(expr.index)])
        return {}

    def map_algebraic_leaf(self, expr: prim.ExpressionNode) -> Mapping[BindingName,
                                                      set[tuple[Expression, ...]]]:

        return {}

    def map_constant(self, expr: object) -> Mapping[BindingName,
                                                      set[tuple[Expression, ...]]]:
        return {}


# {{{ AxesTagsEquationCollector

class AxesTagsEquationCollector(Mapper[None, []]):
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
        self.axis_to_var: bidict[tuple[Array, int], str] = bidict()
        self.known_tag_to_var: dict[Tag, str] = {}

        self.equations: list[tuple[str, str]] = []

        # }}}

    # {{{ unification helpers

    def get_var_for_axis(self, ary: Array, iaxis: int) -> str:
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
        The propagation semantics for a :class:`~pytato.IndexLambda` are
        implemented only for operations that can be raised to a
        :class:`~pytato.raising.HighLevelOp`. In such cases, an equality
        equation is recorded for every non-broadcasted axis of an operand and
        its corresponding axis of *expr*.
        """
        for bnd in expr.bindings.values():
            self.rec(bnd)

        index_expr_used = IndexExpressionsUsedInIndexLambda()(expr.expr)

        if __debug__:
            out_shape = expr.shape
            assert len(out_shape) == expr.ndim

        for vname, ind_tuple in index_expr_used.items():
            for axis_ind in range(len(ind_tuple)):
                var_ind_name = ind_tuple[axis_ind]
                if isinstance(var_ind_name, prim.Variable):
                    if IDX_LAMBDA_JUST_REDUCTIONS.fullmatch(var_ind_name.name):
                        # Reduction axis. We can ignore it.
                        pass
                    elif var_ind_name.name[:3] == "_in":
                        # Variable name axis.
                        pass
                    elif IDX_LAMBDA_INAME.fullmatch(var_ind_name.name):
                        # matched with an iname.
                        inum = int(var_ind_name.name[1:])
                        val = (self.get_var_for_axis(expr.bindings[vname], axis_ind),
                               self.get_var_for_axis(expr, inum))
                        self.equations.append(val)
                    else:
                        raise ValueError(f"Unknown index name used in {vname}")
        return

    def map_stack(self, expr: Stack) -> None:
        """
        Records an equality equation between the axes of arrays being stacked
        and their corresponding axis in *expr*. No equation is added for the
        newly created axis i.e. :attr:`pytato.array.Stack.axis`.
        """
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
        for _, subexpr in sorted(expr._data.items()):
            self.rec(subexpr)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        """
        Does not add any equations.
        """
        for _, subexpr in sorted(expr.bindings.items()):
            if isinstance(subexpr, Array):
                self.rec(subexpr)

        # there's really no good way to propagate the metadata in this case.
        # One *could* raise the loopy kernel instruction expressions to
        # high level ops, but that's quite involved and probably not worth it.

    def map_named_array(self, expr: NamedArray) -> None:
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
                 axis_to_tags: Mapping[tuple[Array, int], Collection[Tag]],
                 tag_corresponding_redn_descr: bool):
        super().__init__()
        self.axis_to_tags: Mapping[tuple[Array, int], Collection[Tag]] = axis_to_tags
        self.tag_corresponding_redn_descr: bool = tag_corresponding_redn_descr

    def rec(self, expr: ArrayOrNames) -> Any:
        if isinstance(expr, AbstractResultWithNamedArrays | DistributedSendRefHolder):
            return super().rec(expr)
        else:
            assert isinstance(expr, Array)
            key = self.get_cache_key(expr)
            try:
                return self._cache[key]
            except KeyError:
                expr_copy = Mapper.rec(self, expr)
                assert isinstance(expr_copy, Array)
                assert expr_copy.ndim == expr.ndim

                for iaxis in range(expr.ndim):
                    expr_copy = expr_copy.with_tagged_axis(
                        iaxis, self.axis_to_tags.get((expr, iaxis), []))

                # {{{ tag reduction descrs

                if self.tag_corresponding_redn_descr:
                    if isinstance(expr, Einsum):
                        assert isinstance(expr_copy, Einsum)
                        for arg, access_descrs in zip(expr.args,
                                                      expr.access_descriptors,
                                                      strict=True):
                            for iaxis, access_descr in enumerate(access_descrs):
                                if isinstance(access_descr, EinsumReductionAxis):
                                    expr_copy = expr_copy.with_tagged_reduction(
                                        access_descr,
                                        self.axis_to_tags.get((arg, iaxis), [])
                                    )

                    if isinstance(expr, IndexLambda):
                        if expr.var_to_reduction_descr:
                            # This is a reduction operation.
                            # We need to find the axes that are reduced over
                            # and update the tag/tag them appropriately.
                            for iaxis in range(len(expr.expr.inner_expr.index_tuple)):
                                name = expr.expr.inner_expr.index_tuple[iaxis].name
                                if name in expr.var_to_reduction_descr.keys():
                                    assert len(list(expr.bindings.keys())) == 1
                                    my_arr: Array = next(iter(expr.bindings.values()))

                                    assert isinstance(expr_copy, IndexLambda)
                                    expr_copy = expr_copy.with_tagged_reduction(
                                            name,
                                            self.axis_to_tags.get((my_arr, iaxis), [])
                                            )
                # }}}

                self._cache[key] = expr_copy
                return expr_copy

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

    equations_collector(expr)

    # start BFS traversal with the known tags as the sources.
    # From the equations build a Propagation Graph
    # Defn. A Propagation graph is a graph where nodes denote variables and an
    # edge between 2 nodes denotes an equality criterion.

    from pytools.graph import (
        get_reachable_nodes,
        undirected_graph_from_edges,
    )

    known_tag_vars = frozenset(equations_collector.known_tag_to_var.values())
    axis_to_solved_tags: dict[tuple[Array, int], set[Tag]] = {}

    propagation_graph = undirected_graph_from_edges(
        equations_collector.equations
    )

    for tag, var in equations_collector.known_tag_to_var.items():
        if isinstance(tag, AxisIgnoredForPropagationTag):
            continue

        reachable_nodes = get_reachable_nodes(propagation_graph, var)
        for reachable_var in (reachable_nodes - known_tag_vars):
            axis_to_solved_tags.setdefault(
                equations_collector.axis_to_var.inverse[reachable_var],
                set()
            ).add(tag)

    return AxisTagAttacher(axis_to_solved_tags,
                           tag_corresponding_redn_descr=unify_redn_descrs,
                           )(expr)

# vim: fdm=marker
