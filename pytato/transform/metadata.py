"""
.. currentmodule:: pytato

.. autofunction:: unify_axes_tags

.. currentmodule:: pytato.transform.metadata

.. autoclass:: AxisTagAttacher

.. autoclass:: AxisIgnoredForPropagationTag

.. autoclass:: AxesTagsEquationCollector
"""
from __future__ import annotations

from pytato.utils import are_shape_components_equal


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
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from bidict import bidict
from typing_extensions import Never

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
    Reshape,
    Stack,
)
from pytato.distributed.nodes import DistributedRecv, DistributedSendRefHolder
from pytato.function import NamedCallResult
from pytato.scalar_expr import (
    IDX_LAMBDA_AXIS_INDEX,
    CombineMapper,
)
from pytato.transform import (
    ArrayOrNames,
    ArrayOrNamesOrFunctionDefTc,
    CopyMapper,
    Mapper,
    TransformMapperCache,
)
from pytato.transform.lower_to_index_lambda import to_index_lambda


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
                                               set[tuple[Expression, ...]]],
                                                      []]):
    """
    Return all the subscript expressions used by a variable specified by BindingName.
    Ex:
    ``_in1[_0,_1]`` would result in an dictionary entry ``{"_in1": ("_0", "_1")}``.
    """
    def combine(self,
                values: Iterable[dict[BindingName,
                                         set[tuple[Expression, ...]]]]) \
                        -> dict[BindingName, set[tuple[Expression, ...]]]:
        out: dict[BindingName, set[tuple[Expression, ...]]] = {}
        import operator
        from functools import reduce
        return reduce(operator.or_, values, out)

    def map_subscript(self, expr: prim.Subscript) -> dict[BindingName,
                                                    set[tuple[Expression, ...]]]:
        """
        Record the indexing expression if the Subscript expression has a prim.Variable
        as its aggregate.
        """

        base_result = super().map_subscript(expr)
        if isinstance(expr.aggregate, prim.Variable):
            return {expr.aggregate.name: {expr.index_tuple}, **base_result}
        return base_result

    def map_algebraic_leaf(self, expr: Expression) -> dict[BindingName,
                                                    set[tuple[Expression, ...]]]:

        return {}

    def map_constant(self, expr: object) -> dict[BindingName,
                                                    set[tuple[Expression, ...]]]:
        return {}
# }}}

# {{{ AxesTagsEquationCollector


class AxesTagsEquationCollector(Mapper[None, Never, []]):
    r"""
    Records equations arising from operand/output axes equivalence for an array
    operation. An equation is recorded for "straight-through" axes in expressions,
    i.e. ones that pass through an operation unmodified.

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

        Users may subclass this mapper to implement domain-specific
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
        # used for unification. If isinstance(iaxis, str), then we are dealing
        # with a reduction axis and so that string will be uniquely defined.
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
        for bnd in expr.bindings.values():
            self.rec(bnd)

        self.add_equations_using_index_lambda_version_of_expr(expr)

    def add_equations_using_index_lambda_version_of_expr(self, expr: Array) -> None:
        """
        Equations are added between an axis of the bindings of *expr* and an axis of
        *expr* if the binding's axis is indexed by by a :class:`~pymbolic.Variable`
        which has a name that follows the reserved iname format, "_[0-9]+", and the axis
        of the output specified by the iname.
        """
        idx_lambda = expr if isinstance(expr, IndexLambda) else to_index_lambda(expr)

        index_expr_used = BindingSubscriptsCollector()(idx_lambda.expr)

        for vname, set_of_ind_tuple in index_expr_used.items():
            for ind_tuple in set_of_ind_tuple:
                for iaxis, var_ind_name in enumerate(ind_tuple):
                    if isinstance(var_ind_name, prim.Variable):
                        lhs: str = self.get_var_for_axis(idx_lambda.bindings[vname],
                                                     iaxis)
                        ind_name: str = var_ind_name.name
                        matched_pattern = IDX_LAMBDA_AXIS_INDEX.fullmatch(ind_name)
                        if matched_pattern:
                            idx_lambda_axis_index = int(matched_pattern.group("index"))
                            if are_shape_components_equal(
                                    idx_lambda.shape[idx_lambda_axis_index],
                                    idx_lambda.bindings[vname].shape[iaxis]):
                                self.record_equation(
                                    lhs,
                                    self.get_var_for_axis(expr, idx_lambda_axis_index))

                        elif ind_name in idx_lambda.var_to_reduction_descr:
                            # Matched with a reduction axis.
                            # We are assuming that this axis is eliminated from the
                            # axes of the output array. Thus, the metadata can only
                            # be propagated to and from the reduction descriptor
                            # which is indexed by the string ind_name.
                            self.record_equation(lhs,
                                self.get_var_for_axis(expr, ind_name))

                        # Other cases are considered "complicated" and we won't
                        # handle them here.

    def map_stack(self, expr: Stack) -> None:
        for ary in expr.arrays:
            self.rec(ary)

        self.add_equations_using_index_lambda_version_of_expr(expr)

    def map_concatenate(self, expr: Concatenate) -> None:
        for ary in expr.arrays:
            self.rec(ary)
        self.add_equations_using_index_lambda_version_of_expr(expr)

    def map_axis_permutation(self, expr: AxisPermutation
                             ) -> None:
        self.rec(expr.array)
        self.add_equations_using_index_lambda_version_of_expr(expr)

    def map_basic_index(self, expr: BasicIndex) -> None:
        self.rec(expr.array)
        self.add_equations_using_index_lambda_version_of_expr(expr)

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> None:
        self.rec(expr.array)
        self.add_equations_using_index_lambda_version_of_expr(expr)

    def map_reshape(self, expr: Reshape) -> None:
        """
        Reshaping generally does not preserve the axis between its input and
        output and so no constraints are enforced except when the
        :class:`pytato.Reshape` has come from a :func:`pytato.expand_dims`.
        """
        # Cannot use add_equations_using_index_lambda_version_of_expr here because
        # reshapes that represent a change in the meaning of an axis may produce
        # index expressions that look like direct passthroughs (for example,
        # reshaping an array of shape (m*n,) to (m, n) when n == 1).
        # We also cannot preserve "unchanged" axes, because it may be unclear which
        # axes those actually are. For example, when reshaping (m, n) -> (m, 1, n),
        # we can't tell from the shape alone which, if any, axes are unchanged
        # (unless made explicit using the ExpandedDimsReshape tag).
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
        for arg in expr.args:
            self.rec(arg)
        self.add_equations_using_index_lambda_version_of_expr(expr)

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
        pass

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
                 _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
                 _function_cache:
                    TransformMapperCache[FunctionDefinition, []] | None = None):
        super().__init__(_cache=_cache, _function_cache=_function_cache)
        self.axis_to_tags: Mapping[tuple[Array, int | str],
                                   Collection[Tag]] = axis_to_tags
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
                if result.var_to_reduction_descr:
                    # This is a reduction operation.
                    # We need to find the axes that are reduced over
                    # and update the tag/tag them appropriately.
                    for redn_var in expr.var_to_reduction_descr:
                        result = result.with_tagged_reduction(
                            redn_var,
                            self.axis_to_tags.get((expr, redn_var), [])
                        )

        # }}}

        return result

    def rec(self, expr: ArrayOrNames) -> ArrayOrNames:
        inputs = self._make_cache_inputs(expr)
        try:
            return self._cache_retrieve(inputs)
        except KeyError:
            # Intentionally going to Mapper instead of super() to avoid
            # double caching when subclasses of CachedMapper override rec,
            # see https://github.com/inducer/pytato/pull/585
            result = Mapper.rec(self, expr)
            if not isinstance(
                    expr, AbstractResultWithNamedArrays | DistributedSendRefHolder):
                assert isinstance(expr, Array)
                # type-ignore reason: passed "ArrayOrNames"; expected "Array"
                result = self._attach_tags(expr, result)  # type: ignore[arg-type]
            return self._cache_add(inputs, result)

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
        expr: ArrayOrNamesOrFunctionDefTc,
        *,
        tag_t: type[Tag] = Tag,
        equations_collector_t: type[
            AxesTagsEquationCollector] = AxesTagsEquationCollector,
        unify_redn_descrs: bool = True,
) -> ArrayOrNamesOrFunctionDefTc:
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

    # Reduction axes are specified by a str but all other axes are specified
    # by an integer. Note that the axes are still uniquely identified.
    axis_to_solved_tags: dict[tuple[Array, int | str], set[Tag]] = {}

    propagation_graph = undirected_graph_from_edges(
        equations_collector.equations
    )

    ignored_vars = {
        tag_var for tag, tag_var in equations_collector.known_tag_to_var.items()
        if isinstance(tag, AxisIgnoredForPropagationTag)
    } | {
        ax_var
        for (ary, ax), ax_var in equations_collector.axis_to_var.items()

        # FIXME? Reduction axes ignore AxisIgnoredForPropagation.
        # They cannot propagate the information to descendant of the array anyway.
        if isinstance(ax, int)

        and ary.axes[ax].tags_of_type(AxisIgnoredForPropagationTag)
    }

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
                           )(expr)

# vim: fdm=marker
