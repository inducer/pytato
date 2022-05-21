"""
.. currentmodule:: pytato

.. autofunction:: unify_axes_tags

.. currentmodule:: pytato.transform.metadata

.. autoclass:: AxisTagAttacher

.. autoclass:: AxesTagsEquationCollector
"""

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


from typing import (TYPE_CHECKING, Type, Set, Tuple, List, Dict, FrozenSet,
                    Mapping, Iterable, Any, TypeVar, cast)
from bidict import bidict
from pytato.scalar_expr import SCALAR_CLASSES
from pytato.transform import ArrayOrNames, Mapper, CopyMapper

from pytato.array import (InputArgumentBase, Stack, Concatenate, IndexLambda,
                          AxisPermutation, BasicIndex,
                          AdvancedIndexInContiguousAxes,
                          Array, Reshape, Einsum, NormalizedSlice,
                          DictOfNamedArrays, NamedArray,
                          AbstractResultWithNamedArrays, ArrayOrScalar,
                          EinsumReductionAxis)
from pytato.distributed import DistributedRecv, DistributedSendRefHolder
from pytato.utils import are_shape_components_equal, are_shapes_equal
from pytato.raising import (index_lambda_to_high_level_op,
                            BinaryOp, FullOp, WhereOp,
                            BroadcastOp, C99CallOp, ReduceOp)

from pytato.diagnostic import UnknownIndexLambdaExpr

from pytools import UniqueNameGenerator
from pytools.tag import Tag
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pytato.loopy import LoopyCall


GraphNodeT = TypeVar("GraphNodeT")


# {{{ AxesTagsEquationCollector

class AxesTagsEquationCollector(Mapper):
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
    def __init__(self, tag_t: Type[Tag]):
        self.tag_t: Type[Tag] = tag_t
        super().__init__()

        # {{{ mutable state held by the mapper

        self._visited_nodes: Set[ArrayOrNames] = set()

        self.var_name_gen: UniqueNameGenerator = UniqueNameGenerator()
        self.var_name_gen.add_names(["c", ""])

        # axis_to_var: mapping from (array, iaxis) to the variable to be
        # used for unification.
        self.axis_to_var: bidict[Tuple[Array, int], str] = bidict()
        self.known_tag_to_var: Dict[Tag, str] = {}

        self.equations: List[Tuple[str, str]] = []

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

    # type-ignore reason: signature not compatible with Mapper.rec
    def rec(self, expr: ArrayOrNames) -> None:  # type: ignore[override]
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

        try:
            hlo = index_lambda_to_high_level_op(expr)
        except UnknownIndexLambdaExpr:
            from warnings import warn
            warn(f"'{expr}' is an unknown index lambda type"
                 " no tags were propagated across it.")
            # no propagation semantics implemented for such cases
            return

        if isinstance(hlo, BinaryOp):
            subexprs: Tuple[ArrayOrScalar, ...] = (hlo.x1, hlo.x2)
        elif isinstance(hlo, WhereOp):
            subexprs = (hlo.condition, hlo.then, hlo.else_)
        elif isinstance(hlo, FullOp):
            # A full-op does not impose any equations
            subexprs = ()
        elif isinstance(hlo, BroadcastOp):
            subexprs = (hlo.x,)
        elif isinstance(hlo, C99CallOp):
            subexprs = hlo.args
        elif isinstance(hlo, ReduceOp):

            # {{{ ReduceOp doesn't quite involve broadcasting

            i_out_axis = 0
            for i_in_axis in range(hlo.x.ndim):
                if i_in_axis not in hlo.axes:
                    self.record_equation(
                        self.get_var_for_axis(hlo.x, i_in_axis),
                        self.get_var_for_axis(expr, i_out_axis)
                    )
                    i_out_axis += 1

            assert i_out_axis == expr.ndim

            # }}}

            return

        else:
            raise NotImplementedError(type(hlo))

        for subexpr in subexprs:
            if isinstance(subexpr, Array):
                for i_in_axis, i_out_axis in zip(
                        range(subexpr.ndim),
                        range(expr.ndim-subexpr.ndim, expr.ndim)):
                    in_dim = subexpr.shape[i_in_axis]
                    out_dim = expr.shape[i_out_axis]
                    if are_shape_components_equal(in_dim, out_dim):
                        self.record_equation(
                            self.get_var_for_axis(subexpr, i_in_axis),
                            self.get_var_for_axis(expr, i_out_axis)
                        )
                    else:
                        # i_in_axis is broadcasted => do not propagate
                        assert are_shape_components_equal(in_dim, 1)
            else:
                assert isinstance(subexpr, SCALAR_CLASSES)

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
        from pytato.utils import partition, get_shape_after_broadcasting

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

        indirection_arrays: List[Array] = cast(List[Array],
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
                          expr.ndim-npost_advanced_basic_indices)):
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
        from pytato.array import EinsumElementwiseAxis, EinsumAxisDescriptor

        for arg in expr.args:
            self.rec(arg)

        descr_to_var: Dict[EinsumAxisDescriptor, str] = {}
        for iaxis in range(expr.ndim):
            descr_to_var[EinsumElementwiseAxis(iaxis)] = self.get_var_for_axis(expr,
                                                                               iaxis)

        for access_descrs, arg in zip(expr.access_descriptors,
                                      expr.args):
            for iarg_axis, descr in enumerate(access_descrs):
                in_tag_var = self.get_var_for_axis(arg, iarg_axis)

                if descr in descr_to_var:
                    self.record_equation(descr_to_var[descr], in_tag_var)
                else:
                    descr_to_var[descr] = in_tag_var

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        for _, subexpr in sorted(expr._data.items()):
            self.rec(subexpr)

    def map_loopy_call(self, expr: "LoopyCall") -> None:
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
        axis in the passthrough data.
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

    # }}}

# }}}


def _get_propagation_graph_from_constraints(
        equations: List[Tuple[str, str]]) -> Mapping[str, FrozenSet[str]]:
    import immutables
    propagation_graph: Dict[str, Set[str]] = {}
    for lhs, rhs in equations:
        assert lhs != rhs
        propagation_graph.setdefault(lhs, set()).add(rhs)
        propagation_graph.setdefault(rhs, set()).add(lhs)

    return immutables.Map({k: frozenset(v)
                           for k, v in propagation_graph.items()})


def get_reachable_nodes(undirected_graph: Mapping[GraphNodeT, Iterable[GraphNodeT]],
                        source_node: GraphNodeT) -> FrozenSet[GraphNodeT]:
    """
    Returns a :class:`frozenset` of all nodes in *undirected_graph* that are
    reachable from *source_node*.
    """
    nodes_visited: Set[GraphNodeT] = set()
    nodes_to_visit = {source_node}
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        nodes_visited.add(current_node)

        neighbors = undirected_graph[current_node]
        nodes_to_visit.update({node
                               for node in neighbors
                               if node not in nodes_visited})

    return frozenset(nodes_visited)


class AxisTagAttacher(CopyMapper):
    """
    A mapper that tags the axes in a DAG as prescribed by *axis_to_tags*.
    """
    def __init__(self,
                 axis_to_tags: Mapping[Tuple[Array, int], Iterable[Tag]],
                 tag_corresponding_redn_descr: bool):
        super().__init__()
        self.axis_to_tags: Mapping[Tuple[Array, int], Iterable[Tag]] = axis_to_tags
        self.tag_corresponding_redn_descr: bool = tag_corresponding_redn_descr

    # type-ignore reason: overrides the type of Mapper.rec
    def rec(self, expr: ArrayOrNames) -> Any:  # type: ignore[override]
        if isinstance(expr, (AbstractResultWithNamedArrays,
                             DistributedSendRefHolder)):
            return super().rec(expr)
        else:
            assert isinstance(expr, Array)
            key = self.get_cache_key(expr)
            try:
                return self._cache[key]
            except KeyError:
                expr_copy = Mapper.rec(self, expr)
                assert expr_copy.ndim == expr.ndim

                for iaxis in range(expr.ndim):
                    expr_copy = expr_copy.with_tagged_axis(
                        iaxis, self.axis_to_tags.get((expr, iaxis), []))

                # {{{ tag reduction descrs

                if self.tag_corresponding_redn_descr:
                    if isinstance(expr, Einsum):
                        for arg, access_descrs in zip(expr.args,
                                                      expr.access_descriptors):
                            for iaxis, access_descr in enumerate(access_descrs):
                                if isinstance(access_descr, EinsumReductionAxis):
                                    expr_copy = expr_copy.with_tagged_reduction(
                                        access_descr,
                                        self.axis_to_tags.get((arg, iaxis), [])
                                    )

                    if isinstance(expr, IndexLambda):
                        try:
                            hlo = index_lambda_to_high_level_op(expr)
                        except UnknownIndexLambdaExpr:
                            pass
                        else:
                            if isinstance(hlo, ReduceOp):
                                for iaxis, redn_var in hlo.axes.items():
                                    expr_copy = expr_copy.with_tagged_reduction(
                                        redn_var,
                                        self.axis_to_tags.get((hlo.x, iaxis), [])
                                    )

                # }}}

                self._cache[key] = expr_copy
                return expr_copy

    # type-ignore reason: overrides the type of Mapper.__call__
    def __call__(self, expr: ArrayOrNames) -> ArrayOrNames:  # type: ignore[override]
        result = self.rec(expr)
        assert isinstance(result, (Array, AbstractResultWithNamedArrays))
        return result


def unify_axes_tags(
        expr: ArrayOrNames,
        *,
        tag_t: Type[Tag] = Tag,
        equations_collector_t: Type[
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

    propagation_graph = _get_propagation_graph_from_constraints(
        equations_collector.equations)

    known_tag_vars = frozenset(equations_collector.known_tag_to_var.values())
    axis_to_solved_tags: Dict[Tuple[Array, int], Set[Tag]] = {}

    for tag, var in equations_collector.known_tag_to_var.items():
        for reachable_var in (get_reachable_nodes(propagation_graph, var)
                              - known_tag_vars):
            axis_to_solved_tags.setdefault(
                equations_collector.axis_to_var.inverse[reachable_var],
                set()
            ).add(tag)

    return AxisTagAttacher(axis_to_solved_tags,
                           tag_corresponding_redn_descr=unify_redn_descrs,
                           )(expr)

# vim: fdm=marker
