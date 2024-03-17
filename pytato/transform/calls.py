from __future__ import annotations

__doc__ = """
.. currentmodule:: pytato.transform.calls

.. autofunction:: inline_calls
.. autofunction:: concatenate_calls
.. autofunction:: tag_all_calls_to_be_inlined

.. autoclass:: CallSiteLocation
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

import itertools
import attrs
import pymbolic.primitives as prim
import pytato.scalar_expr as scalar_expr

from typing import (Tuple, FrozenSet, Collection, Mapping, Any, List, Dict,
                    TYPE_CHECKING, Sequence, Callable, Set, Generator)
from pytato.transform import (ArrayOrNames, CopyMapper, CombineMapper, Mapper,
                              CachedMapper, _SelfMapper,
                              CachedWalkMapper)
from pytato.transform.lower_to_index_lambda import to_index_lambda
from pytato.array import (AbstractResultWithNamedArrays, Array,
                          DictOfNamedArrays, IndexBase, Placeholder,
                          SizeParam, InputArgumentBase, concatenate,
                          IndexLambda, Roll, Stack, Concatenate,
                          Einsum, AxisPermutation,
                          Reshape, BasicIndex, DataWrapper, ShapeComponent,
                          ShapeType)
from functools import partialmethod
from immutables import Map
from pytools import memoize_method, memoize_on_first_arg

from pytato.function import Call, NamedCallResult, FunctionDefinition
from pytato.tags import InlineCallTag
from pytato.utils import are_shape_components_equal
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pytato.loopy import LoopyCallResult

ArrayOnStackT = Tuple[Tuple[Call, ...], Array]


# {{{ inlining

class PlaceholderSubstitutor(CopyMapper):
    """
    .. attribute:: substitutions

        A mapping from the placeholder name to the array that it is to be
        substituted with.
    """

    def __init__(self, substitutions: Mapping[str, Array]) -> None:
        super().__init__()
        self.substitutions = substitutions

    def map_placeholder(self, expr: Placeholder) -> Array:
        return self.substitutions[expr.name]


class Inliner(CopyMapper):
    """
    Primary mapper for :func:`inline_calls`.
    """
    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        # inline call sites within the callee.
        new_expr = super().map_call(expr)
        assert isinstance(new_expr, Call)

        if expr.tags_of_type(InlineCallTag):
            substitutor = PlaceholderSubstitutor(new_expr.bindings)

            return DictOfNamedArrays(
                {name: substitutor(ret)
                 for name, ret in new_expr.function.returns.items()},
                tags=new_expr.tags
            )
        else:
            return new_expr

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        new_call_or_inlined_expr = self.rec(expr._container)
        assert isinstance(new_call_or_inlined_expr, AbstractResultWithNamedArrays)
        if isinstance(new_call_or_inlined_expr, Call):
            return new_call_or_inlined_expr[expr.name]
        else:
            return new_call_or_inlined_expr[expr.name].expr


class InlineMarker(CopyMapper):
    """
    Primary mapper for :func:`tag_all_calls_to_be_inlined`.
    """
    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        return super().map_call(expr).tagged(InlineCallTag())


def inline_calls(expr: ArrayOrNames) -> ArrayOrNames:
    """
    Returns a copy of *expr* with call sites tagged with
    :class:`pytato.tags.InlineCallTag` inlined into the expression graph.
    """
    inliner = Inliner()
    return inliner(expr)


def tag_all_calls_to_be_inlined(expr: ArrayOrNames) -> ArrayOrNames:
    """
    Returns a copy of *expr* with all reachable instances of
    :class:`pytato.function.Call` nodes tagged with
    :class:`pytato.tags.InlineCallTag`.

    .. note::

       This routine does NOT inline calls, to inline the calls
       use :func:`tag_all_calls_to_be_inlined` on this routine's
       output.
    """
    return InlineMarker()(expr)

# }}}


# {{{ Concatenatability

@attrs.define(frozen=True)
class Concatenatability:
    """
    Describes how a particular array expression can be concatenated.
    """


@attrs.define(frozen=True)
class ConcatableAlongAxis(Concatenatability):
    """
    Used to describe an array expression that is concatenatable along *axis*.
    """
    axis: int


@attrs.define(frozen=True)
class ConcatableIfConstant(Concatenatability):
    """
    Used to describe an array expression in a function body that can be
    concatenated only if the expression is the same across call-sites.
    """

# }}}


# {{{ concatenate_calls

@attrs.define(frozen=True)
class CallSiteLocation:
    r"""
    Records a call-site's location in a :mod:`pytato` expression.

    .. attribute:: call

        The instance of :class:`~pytato.function.Call` being called at this
        location.

    .. attribute:: stack

        The call sites within which this particular call is called.
        For eg. if ``stack = (c1, c2)``, then :attr:`call` is called within
        ``c2``\ 's function body which itself is called from ``c1``\ 's
        function body.
    """
    call: Call
    stack: Tuple[Call, ...]


class CallsiteCollector(CombineMapper[FrozenSet[CallSiteLocation]]):
    r"""
    Collects all the call sites in a :mod:`pytato` expression.

    .. attribute:: stack

        The stack of calls at which the calls are being collected. This
        attribute is used to specify :attr:`CallSiteLocation.stack` in the
        :class:`CallSiteLocation`\ s being built. Must be altered (by creating
        a new instance of the mapper) before entering the function body of a
        new :class:`~pytato.function.Call`.
    """
    def __init__(self, stack: Tuple[Call, ...]) -> None:
        self.stack = stack
        super().__init__()

    def combine(self, *args: FrozenSet[CallSiteLocation]
                ) -> FrozenSet[CallSiteLocation]:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_size_param(self, expr: SizeParam) -> FrozenSet[CallSiteLocation]:
        return frozenset()

    def map_call(self, expr: Call) -> FrozenSet[CallSiteLocation]:
        new_mapper_for_fn = CallsiteCollector(stack=self.stack + (expr,))
        return self.combine(frozenset([CallSiteLocation(expr, self.stack)]),
                            *[new_mapper_for_fn(ret)
                              for ret in expr.function.returns.values()])


class _NamedCallResultReplacerPostConcatenate(CopyMapper):
    """
    Mapper to replace instances of :class:`~pytato.function.NamedCallResult` as
    per :attr:`replacement_map`.

    .. attribute:: stack_to_replace_on

        The stack onto which the replacement must be performed.

    .. attribute:: current_stack

        Records the stack to track which function body the mapper is
        traversing. Must be altered (by creating a new instance) before
        entering the function body of a new :class:`~pytato.function.Call`.
    """
    def __init__(self,
                 replacement_map: Mapping[NamedCallResult, Array],
                 current_stack: Tuple[Call, ...],
                 stack_to_replace_on: Tuple[Call, ...]) -> None:
        self.replacement_map = replacement_map
        self.current_stack = current_stack
        self.stack_to_replace_on = stack_to_replace_on
        super().__init__()

    @memoize_method
    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        raise AssertionError("Control should not reach here."
                             " Call clone_with_new_call_on_stack instead.")

    @memoize_method
    def clone_with_new_call_on_stack(self: _SelfMapper, expr: Call) -> _SelfMapper:
        # type-ignore-reason: Mapper class does not define these attributes.
        return type(self)(  # type: ignore[call-arg]
            self.replacement_map,  # type: ignore[attr-defined]
            self.current_stack + (expr,),  # type: ignore[attr-defined]
            self.stack_to_replace_on,  # type: ignore[attr-defined]
        )

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        new_stack_to_enter = self.current_stack + (expr,)
        if self.stack_to_replace_on[:len(new_stack_to_enter)] == new_stack_to_enter:
            # leading call-sites on the stack match the stack on which
            # replacement must be performed.
            new_mapper = self.clone_with_new_call_on_stack(expr)

            return Call(new_mapper.map_function_definition(expr.function),
                        Map({name: self.rec(bnd)
                             for name, bnd in expr.bindings.items()}),
                        tags=expr.tags)
        else:
            return Call(expr.function,  # do not map the exprs in function's body.
                        Map({name: self.rec(bnd)
                             for name, bnd in expr.bindings.items()}),
                        tags=expr.tags)

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        if self.current_stack == self.stack_to_replace_on:
            try:
                return self.replacement_map[expr]
            except KeyError:
                return super().map_named_call_result(expr)
        else:
            return super().map_named_call_result(expr)


def _have_same_axis_length(arrays: Collection[Array],
                           iaxis: int) -> bool:
    """
    Returns *True* only if every array in *arrays* have the same axis length
    along *iaxis*.
    """
    axis_length = next(iter(arrays)).shape[iaxis]
    return all(are_shape_components_equal(other_ary.shape[iaxis],
                                          axis_length)
               for other_ary in arrays)


def _have_same_axis_length_except(arrays: Collection[Array],
                                  iaxis: int) -> bool:
    """
    Returns *True* only if every array in *arrays* have the same
    dimensionality and have axes with the same lengths except along the
    *iaxis*-axis.
    """
    ndim = next(iter(arrays)).ndim
    return (all(ary.ndim == ndim for ary in arrays)
            and all(_have_same_axis_length(arrays, idim)
                    for idim in range(ndim)
                    if idim != iaxis))


@attrs.define(frozen=True)
class _InputConcatabilityGetterAcc:
    r"""
    Return type for :class:`_InputConcatabilityGetter`. An instance of this class is
    returned after mapping a :class:`~pytato.Array` expression.

    .. attribute:: seen_inputs

        A :class:`frozenset` of all :class:`pytato.InputArgumentBase`
        predecessors of a node.

    .. attribute:: input_concatability

        Records the constraints that come along with concatenating the array
        being mapped. The constraints are recorded as a mapping from the axes
        of the array being mapped to the axes of the input arguments. This
        mapping informs us which axes in the :class:`InputArgumentBase`\ s'
        must be concatenated to soundly concatenate a particular axis in the
        array being mapped. The axes in this mapping are represented using
        :class:`int`. If certain axes are missing in this mapping, then
        concatenation cannot be performed along those axes for the mapped
        array.
    """
    seen_inputs: FrozenSet[InputArgumentBase]
    input_concatability: Mapping[Concatenatability,
                                 Mapping[InputArgumentBase, Concatenatability]]

    def __post_init__(self) -> None:
        assert all(
            frozenset(input_concat.keys()) == self.seen_inputs
            for input_concat in self.input_concatability.values())

    __attrs_post_init__ = __post_init__


class NonConcatableExpression(RuntimeError):
    """
    Used internally by :class:`_ScalarExprConcatabilityMapper`.
    """


class _InvalidConcatenatability(RuntimeError):
    """
    Used internally by :func:`_get_ary_to_concatenatabilities`.
    """


class _ScalarExprConcatabilityMapper(scalar_expr.CombineMapper):
    """
    Maps :attr:`~pytato.array.IndexLambda.expr` to the axes of the bindings
    that must be concatenated to concatenate the IndexLambda's
    :attr:`iaxis`-axis.

    .. attribute:: allow_indirect_addr

        If *True* indirect access are allowed. However, concatenating along the
        iaxis-axis would be sound only if the binding which is being indexed
        into is same for all the expressions to be concatenated.
    """
    def __init__(self, iaxis: int, allow_indirect_addr: bool) -> None:
        self.iaxis = iaxis
        self.allow_indirect_addr = allow_indirect_addr
        super().__init__()

    def combine(self, values: Collection[Mapping[str, Concatenatability]]
                ) -> Mapping[str, Concatenatability]:
        result: Dict[str, Concatenatability] = {}
        for value in values:
            for bnd_name, iaxis in value.items():
                try:
                    if result[bnd_name] != iaxis:
                        # only one axis of a particular binding can be
                        # concatenated. If multiple axes must be concatenated
                        # that means the index lambda is not
                        # iaxis-concatenatable.
                        raise NonConcatableExpression
                except KeyError:
                    result[bnd_name] = iaxis

        return Map(result)

    def map_variable(self, expr: prim.Variable) -> Mapping[str, Concatenatability]:
        if expr.name == f"_{self.iaxis}":
            raise NonConcatableExpression
        else:
            return Map()

    def map_constant(self, expr: Any) -> Mapping[str, Concatenatability]:
        return Map()

    map_nan = map_constant

    def map_subscript(self, expr: prim.Subscript
                      ) -> Mapping[str, Concatenatability]:
        name: str = expr.aggregate.name
        rec_indices: List[Mapping[str, Concatenatability]] = []
        for iaxis, idx in enumerate(expr.index_tuple):
            if idx == prim.Variable(f"_{self.iaxis}"):
                rec_indices.append({name: ConcatableAlongAxis(iaxis)})
            else:
                rec_idx = self.rec(idx)
                if rec_idx:
                    if not self.allow_indirect_addr:
                        raise NonConcatableExpression
                    else:
                        # indirect accesses cannot be concatenated in the general
                        # case unless the indexee is the same for the
                        # expression graphs being concatenated.
                        pass
                rec_indices.append(rec_idx)

        combined_rec_indices = dict(self.combine(rec_indices))

        if name not in combined_rec_indices:
            combined_rec_indices[name] = ConcatableIfConstant()

        return Map(combined_rec_indices)


@memoize_on_first_arg
def _get_binding_to_concatenatability(expr: scalar_expr.ScalarExpression,
                                      iaxis: int,
                                      allow_indirect_addr: bool,
                                      ) -> Mapping[str, Concatenatability]:
    """
    Maps *expr* using :class:`_ScalarExprConcatabilityMapper`.
    """
    mapper = _ScalarExprConcatabilityMapper(iaxis, allow_indirect_addr)
    return mapper(expr)  # type: ignore[no-any-return]


def _combine_input_accs(
    operand_accs: Tuple[_InputConcatabilityGetterAcc, ...],
    concat_to_operand_concatabilities: Mapping[Concatenatability,
                                               Tuple[Concatenatability, ...]
                                               ],
) -> _InputConcatabilityGetterAcc:
    """
    For an index lambda ``I`` with operands ``I1, I2, .. IN`` that specify their
    concatenatability constraints using *operand_accs*, this routine returns
    the axes concatenation constaints of ``I``.

    :arg concat_to_operand_concatabilities: Mapping of the form ``concat_I ->
        (C_I1, C_I2, ..., C_IN)`` specifying the concatabilities of the
        operands ``I1, I2, .., IN`` in order to concatenate the
        ``I`` axis via the criterion ``conncat_I``.
    """

    from functools import reduce

    input_concatabilities: Dict[Concatenatability, Map[InputArgumentBase,
                                                       Concatenatability]] = {}
    seen_inputs: FrozenSet[InputArgumentBase] = reduce(
        frozenset.union,
        (operand_acc.seen_inputs for operand_acc in operand_accs),
        frozenset())

    # The core logic here is to filter the iaxis in out_axis_to_operand_axes
    # so that all the operands agree on how the input arguments must be
    # concatenated.

    for out_concat, operand_concatabilities in (concat_to_operand_concatabilities
                                                .items()):
        is_i_out_axis_concatenatable = True
        input_concatability: Dict[InputArgumentBase, Concatenatability] = {}

        for operand_concatability, operand_acc in zip(operand_concatabilities,
                                                      operand_accs,
                                                      strict=True):
            if operand_concatability not in (
                    operand_acc.input_concatability):
                # required operand concatability cannot be achieved
                # => out_concat cannot be concatenated
                is_i_out_axis_concatenatable = False
                break

            for input_arg, input_concat in (
                    operand_acc
                    .input_concatability[operand_concatability]
                    .items()):
                try:
                    if input_concatability[input_arg] != input_concat:
                        is_i_out_axis_concatenatable = False
                        break
                except KeyError:
                    input_concatability[input_arg] = input_concat
            if not is_i_out_axis_concatenatable:
                break

        if is_i_out_axis_concatenatable:
            input_concatabilities[out_concat] = Map(input_concatability)

    return _InputConcatabilityGetterAcc(seen_inputs,
                                        Map(input_concatabilities))


@attrs.define(frozen=True)
class FunctionConcatenability:
    r"""
    Records a valid concatenatability criterion for a
    :class:`pytato.function.FunctionDefinition`.

    .. attribute:: output_to_concatenatability

        A mapping from the name of a
        :class:`FunctionDefinition`\ 's returned array to how it should be
        concatenated.

    .. attribute:: input_to_concatenatability

        A mapping from a :class:`FunctionDefinition`\ 's parameter to how it
        should be concatenated.

    .. note::

        A :class:`FunctionDefinition` typically has multiple valid
        concatenability constraints. This class only records one of those valid
        constraints.
    """
    output_to_concatenatability: Mapping[str, Concatenatability]
    input_to_concatenatability: Mapping[str, Concatenatability]

    def __str__(self) -> str:
        outputs = []
        for name, concat in self.output_to_concatenatability.items():
            outputs.append(f"{name} => {concat}")

        inputs = []
        for name, concat in self.input_to_concatenatability.items():
            inputs.append(f"{name} => {concat}")

        output_str = "\n".join(outputs)
        input_str = "\n".join(inputs)

        return (f"Outputs:\n--------\n{output_str}\n"
                f"========\nInputs:\n-------\n{input_str}\n"
                "========")


def _combine_named_result_accs(
        named_result_accs: Mapping[str, _InputConcatabilityGetterAcc]
) -> Tuple[FunctionConcatenability, ...]:
    """
    Combines the concantenatability constraints of named results of a
    :class:`FunctionDefinition` and returns a :class:`tuple` of the valid
    concatenatable constraints.
    """
    potential_concatenatable_ouptut_axes = itertools.product(*[
        [(name, concat) for concat in acc.input_concatability]
        for name, acc in named_result_accs.items()])

    valid_concatenatabilities: List[FunctionConcatenability] = []

    for output_concats in potential_concatenatable_ouptut_axes:
        is_concatenatable = True
        input_concatability: Dict[InputArgumentBase, Concatenatability] = {}

        for result_name, iresult_axis in output_concats:
            for input_arg, i_input_axis in (
                    named_result_accs[result_name]
                    .input_concatability[iresult_axis]
                    .items()):
                try:
                    if input_concatability[input_arg] != i_input_axis:
                        is_concatenatable = False
                        break
                except KeyError:
                    input_concatability[input_arg] = i_input_axis

            if not is_concatenatable:
                break

        if is_concatenatable:
            pl_concatabilities = {pl.name: concat
                                  for pl, concat in input_concatability.items()
                                  if isinstance(pl, Placeholder)}
            valid_concatenatabilities.append(
                FunctionConcatenability(Map(output_concats),
                                        Map(pl_concatabilities))
            )

    return tuple(valid_concatenatabilities)


class _InputConcatabilityGetter(CachedMapper[ArrayOrNames]):
    """
    Maps :class:`pytato.array.Array` expressions to
    :class:`_InputConcatenatabilityGetterAcc` that summarizes constraints
    induced on the concatenatability of the inputs of the expression by  the
    expression's concatenatability.
    """

    def _map_input_arg_base(self, expr: InputArgumentBase
                            ) -> _InputConcatabilityGetterAcc:
        input_concatenatability: Dict[Concatenatability,
                                      Map[InputArgumentBase,
                                          Concatenatability]] = {}
        for idim in range(expr.ndim):
            input_concatenatability[ConcatableAlongAxis(idim)] = Map(
                {expr: ConcatableAlongAxis(idim)})

        input_concatenatability[ConcatableIfConstant()] = Map(
            {expr: ConcatableIfConstant()})

        return _InputConcatabilityGetterAcc(frozenset([expr]),
                                            Map(input_concatenatability))

    map_placeholder = _map_input_arg_base
    map_data_wrapper = _map_input_arg_base

    def _map_index_lambda_like(self, expr: Array,
                               allow_indirect_addr: bool
                               ) -> _InputConcatabilityGetterAcc:
        expr = to_index_lambda(expr)
        input_accs = tuple(self.rec(expr.bindings[name])
                           for name in sorted(expr.bindings.keys()))
        expr_concat_to_input_concats: Dict[Concatenatability,
                                           Tuple[Concatenatability, ...]] = {}

        for iaxis in range(expr.ndim):
            try:
                bnd_name_to_concat = _get_binding_to_concatenatability(
                    expr.expr, iaxis, allow_indirect_addr)
                expr_concat_to_input_concats[ConcatableAlongAxis(iaxis)] = (
                    tuple(concat
                          for _, concat in sorted(bnd_name_to_concat.items(),
                                                  key=lambda x: x[0]))
                )
            except NonConcatableExpression:
                pass

        expr_concat_to_input_concats[ConcatableIfConstant()] = tuple(
            ConcatableIfConstant() for _ in expr.bindings)

        return _combine_input_accs(input_accs, expr_concat_to_input_concats)

    map_index_lambda = partialmethod(_map_index_lambda_like,
                                      allow_indirect_addr=False)
    map_einsum = partialmethod(_map_index_lambda_like,
                                allow_indirect_addr=False)
    map_basic_index = partialmethod(_map_index_lambda_like,
                                     allow_indirect_addr=False)
    map_roll = partialmethod(_map_index_lambda_like,
                              allow_indirect_addr=False)
    map_stack = partialmethod(_map_index_lambda_like,
                               allow_indirect_addr=False)
    map_concatenate = partialmethod(_map_index_lambda_like,
                                     allow_indirect_addr=False)
    map_axis_permutation = partialmethod(_map_index_lambda_like,
                                          allow_indirect_addr=False)
    map_reshape = partialmethod(_map_index_lambda_like,
                                 allow_indirect_addr=False)

    map_contiguous_advanced_index = partialmethod(_map_index_lambda_like,
                                                   allow_indirect_addr=True)
    map_non_contiguous_advanced_index = partialmethod(_map_index_lambda_like,
                                                       allow_indirect_addr=True)

    def map_named_call_result(self, expr: NamedCallResult
                              ) -> _InputConcatabilityGetterAcc:
        assert isinstance(expr._container, Call)
        valid_concatenatabilities = _get_valid_concatenatability_constraints(
            expr._container.function)

        expr_concat_possibilities = {
            valid_concatenability.output_to_concatenatability[expr.name]
            for valid_concatenability in valid_concatenatabilities
        }

        input_concatenatabilities: Dict[Concatenatability,
                                        Mapping[InputArgumentBase,
                                                Concatenatability]] = {}
        rec_bindings = {bnd_name: self.rec(binding)
                        for bnd_name, binding in expr._container.bindings.items()}
        callee_acc = self.rec(expr._container.function.returns[expr.name])
        seen_inputs: Set[InputArgumentBase] = set()

        for seen_input in callee_acc.seen_inputs:
            if isinstance(seen_input, Placeholder):
                seen_inputs.update(rec_bindings[seen_input.name].seen_inputs)
            elif isinstance(seen_input, (DataWrapper, SizeParam)):
                seen_inputs.add(seen_input)
            else:
                raise NotImplementedError(type(seen_input))

        for concat_possibility in expr_concat_possibilities:
            caller_input_concatabilities: Dict[InputArgumentBase,
                                               Concatenatability] = {}

            is_concat_possibility_valid = True
            for callee_input_arg, callee_input_concat in (
                    callee_acc.input_concatability[concat_possibility].items()):
                caller_acc = rec_bindings[callee_input_arg.name]
                if isinstance(callee_input_arg, Placeholder):
                    if callee_input_concat in caller_acc.input_concatability:
                        for caller_input_arg, caller_input_concat in (
                                caller_acc
                                .input_concatability[callee_input_concat]
                                .items()):
                            try:
                                if (caller_input_concatabilities[caller_input_arg]
                                        != caller_input_concat):
                                    is_concat_possibility_valid = False
                                    break
                            except KeyError:
                                caller_input_concatabilities[callee_input_arg] = (
                                    caller_input_concat)
                        if not is_concat_possibility_valid:
                            break
                    else:
                        is_concat_possibility_valid = False
                        break
                elif isinstance(callee_input_arg, (DataWrapper, SizeParam)):
                    try:
                        if (caller_input_concatabilities[callee_input_arg]
                                != callee_input_concat):
                            is_concat_possibility_valid = False
                            break
                    except KeyError:
                        caller_input_concatabilities[callee_input_arg] = (
                            callee_input_concat)
                else:
                    raise NotImplementedError(type(callee_input_arg))

            if is_concat_possibility_valid:
                input_concatenatabilities[concat_possibility] = Map(
                    caller_input_concatabilities)

        return _InputConcatabilityGetterAcc(frozenset(seen_inputs),
                                            Map(input_concatenatabilities))

    def map_loopy_call_result(self, expr: "LoopyCallResult"
                              ) -> _InputConcatabilityGetterAcc:
        raise ValueError("Loopy Calls are illegal to concatenate. Maybe"
                         " rewrite the operation as array operations?")


def _verify_arrays_can_be_concated_along_axis(
        arrays: Collection[Array],
        fields_that_must_be_same: Collection[str],
        iaxis: int) -> None:
    """
    Performs some common checks if *arrays* from different function bodies can be
    concatenated.

    .. attribute:: arrays

        Corresponding expressions from the function bodies for call-site that
        are being checked for concatenation along *iaxis*.
    """
    if not _have_same_axis_length_except(arrays, iaxis):
        raise _InvalidConcatenatability("Cannot be concatenate the calls.")
    if len({ary.__class__ for ary in arrays}) != 1:
        raise _InvalidConcatenatability("Cannot be concatenate the calls.")
    for field in fields_that_must_be_same:
        if len({getattr(ary, field) for ary in arrays}) != 1:
            raise _InvalidConcatenatability("Cannot be concatenate the calls.")


def _verify_arrays_same(arrays: Collection[Array]) -> None:
    if len(set(arrays)) != 1:
        raise _InvalidConcatenatability("Cannot be concatenated as arrays across "
                                        " functions is not the same.")


def _get_concatenated_shape(arrays: Collection[Array], iaxis: int) -> ShapeType:
    # type-ignore-reason: mypy expects 'ary.shape[iaxis]' as 'int' since the
    # 'start' is an 'int'
    concatenated_axis_length = sum(ary.shape[iaxis]  # type: ignore[misc]
                                   for ary in arrays)
    template_ary = next(iter(arrays))

    return tuple(dim
                 if idim != iaxis
                 else concatenated_axis_length
                 for idim, dim in enumerate(template_ary.shape)
                 )


class _ConcatabilityCollector(CachedWalkMapper):
    def __init__(self, current_stack: Tuple[Call, ...]) -> None:
        self.ary_to_concatenatability: Dict[ArrayOnStackT, Concatenatability] = {}
        self.current_stack = current_stack
        self.call_sites_on_hold: Set[Call] = set()
        super().__init__()

    # type-ignore-reason: CachedWalkMaper takes variadic `*args, **kwargs`.
    def get_cache_key(self,  # type: ignore[override]
                      expr: ArrayOrNames,
                      *args: Any,
                      ) -> Tuple[ArrayOrNames, Any]:
        return (expr, args)

    def _record_concatability(self, expr: Array,
                              concatenatability: Concatenatability,
                              ) -> None:
        key = (self.current_stack, expr)
        assert key not in self.ary_to_concatenatability
        self.ary_to_concatenatability[key] = concatenatability

    @memoize_method
    def clone_for_callee(
            self: _SelfMapper, function: FunctionDefinition) -> _SelfMapper:
        raise AssertionError("Control should not reach here."
                             " Call clone_with_new_call_on_stack instead.")

    @memoize_method
    def clone_with_new_call_on_stack(self: _SelfMapper, expr: Call) -> _SelfMapper:
        # type-ignore-reason: Mapper class does not define these attributes.
        return type(self)(  # type: ignore[call-arg]
            self.current_stack + (expr,),  # type: ignore[attr-defined]
        )

    def _map_input_arg_base(self,
                            expr: InputArgumentBase,
                            concatenatability: Concatenatability,
                            exprs_from_other_calls: Tuple[Array, ...],
                            ) -> None:
        if isinstance(concatenatability, ConcatableIfConstant):
            _verify_arrays_same((expr,) + exprs_from_other_calls)
        elif isinstance(concatenatability, ConcatableAlongAxis):
            _verify_arrays_can_be_concated_along_axis(
                (expr,) + exprs_from_other_calls,
                ["dtype", "name"],
                concatenatability.axis)
        else:
            raise NotImplementedError(type(concatenatability))

        self._record_concatability(expr, concatenatability)

    map_placeholder = _map_input_arg_base  # type: ignore[assignment]
    map_data_wrapper = _map_input_arg_base  # type: ignore[assignment]

    def _map_index_lambda_like(self,
                               expr: Array,
                               concatenatability: Concatenatability,
                               exprs_from_other_calls: Tuple[Array, ...],
                               allow_indirect_addr: bool,
                               ) -> None:
        self._record_concatability(expr, concatenatability)

        idx_lambda = to_index_lambda(expr)
        idx_lambdas_from_other_calls = tuple(to_index_lambda(ary)
                                             for ary in exprs_from_other_calls)

        if isinstance(concatenatability, ConcatableIfConstant):
            _verify_arrays_same((idx_lambda,) + idx_lambdas_from_other_calls)
            for ary in idx_lambda.bindings.values():
                self.rec(ary, concatenatability)
        elif isinstance(concatenatability, ConcatableAlongAxis):
            _verify_arrays_can_be_concated_along_axis(
                (idx_lambda, ) + idx_lambdas_from_other_calls,
                ["dtype", "expr"],
                concatenatability.axis)
            bnd_name_to_concat = _get_binding_to_concatenatability(
                idx_lambda.expr, concatenatability.axis, allow_indirect_addr)
            for bnd_name, bnd_concat in bnd_name_to_concat.items():
                self.rec(idx_lambda.bindings[bnd_name], bnd_concat,
                         tuple(ary.bindings[bnd_name]
                               for ary in idx_lambdas_from_other_calls))
        else:
            raise NotImplementedError(type(concatenatability))

    map_index_lambda = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_einsum = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_basic_index = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_roll = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_stack = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_concatenate = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_axis_permutation = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_reshape = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)

    map_contiguous_advanced_index = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=True)
    map_non_contiguous_advanced_index = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=True)

    # type-ignore-reason: CachedWalkMapper.map_call takes in variadic args, kwargs
    def map_call(self,  # type: ignore[override]
                 expr: Call,
                 exprs_from_other_calls: Tuple[Call, ...]) -> None:
        if not all(
                (self.current_stack, named_result) in self.ary_to_concatenatability
                for named_result in expr.values()):
            self.call_sites_on_hold.add(expr)
        else:
            self.call_sites_on_hold -= {expr}
            new_mapper = self.clone_with_new_call_on_stack(expr)
            for name, val_in_callee in expr.function.returns.items():
                new_mapper(val_in_callee,
                           self.ary_to_concatenatability[(self.current_stack,
                                                          expr[name])],
                           tuple(other_call.function.returns[name]
                                 for other_call in exprs_from_other_calls)
                           )

            if new_mapper.call_sites_on_hold:
                raise NotImplementedError("Call sites that do not all use all"
                                          " the returned values not yet"
                                          " supported for concatenation.")

            for ary, concat in new_mapper.ary_to_concatenatability.items():
                assert ary not in self.ary_to_concatenatability
                self.ary_to_concatenatability[ary] = concat

            for name, binding in expr.bindings.items():
                concat = (
                    new_mapper
                    .ary_to_concatenatability[(self.current_stack + (expr,),
                                               expr.function.get_placeholder(name))]
                )
                self.rec(binding,
                         concat,
                         tuple(other_call.bindings[name]
                               for other_call in exprs_from_other_calls))

    # type-ignore-reason: CachedWalkMapper's method takes in variadic args, kwargs
    def map_named_call_result(self, expr: NamedCallResult,  # type: ignore[override]
                              concatenatability: Concatenatability,
                              exprs_from_other_calls: Tuple[Array, ...],
                              ) -> None:
        self._record_concatability(expr, concatenatability)
        if any(not isinstance(ary, NamedCallResult)
               for ary in exprs_from_other_calls):
            raise _InvalidConcatenatability()

        # type-ignore-reason: mypy does not respect the conditional which
        # asserts that all arrays in `exprs_from_other_calls` are
        # NamedCallResult.
        self.rec(expr._container,
                 tuple(ary._container  # type: ignore[attr-defined]
                       for ary in exprs_from_other_calls)
                 )

    def map_loopy_call_result(self, expr: "LoopyCallResult"
                              ) -> None:
        raise ValueError("Loopy Calls are illegal to concatenate. Maybe"
                         " rewrite the operation as array operations?")


class _Concatenator(Mapper):
    def __init__(self,
                 current_stack: Tuple[Call, ...],
                 ary_to_concatenatability: Map[ArrayOnStackT, Concatenatability],
                 ) -> None:
        self.current_stack = current_stack
        self.ary_to_concatenatability = ary_to_concatenatability

        self._cache: Dict[Tuple[Array, Tuple[Array, ...]], Array] = {}

    # type-ignore-reason: super-type Mapper does not allow the extra args.
    def rec(self, expr: Array,  # type: ignore[override]
            exprs_from_other_calls: Tuple[Array, ...]
            ) -> Array:
        key = (expr, exprs_from_other_calls)
        try:
            return self._cache[key]
        except KeyError:
            result: Array = super().rec(expr,
                                        exprs_from_other_calls)
            self._cache[key] = result
            return result

    @memoize_method
    def clone_with_new_call_on_stack(self, expr: Call) -> _Concatenator:
        return _Concatenator(
            self.current_stack + (expr,),
            self.ary_to_concatenatability,
        )

    def _get_concatenatability(self, expr: Array) -> Concatenatability:
        return self.ary_to_concatenatability[(self.current_stack, expr)]

    def map_placeholder(self,
                        expr: Placeholder,
                        exprs_from_other_calls: Tuple[Array, ...]
                        ) -> Array:
        concat = self._get_concatenatability(expr)
        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            new_shape = _get_concatenated_shape(
                (expr,) + exprs_from_other_calls, concat.axis)
            return Placeholder(name=expr.name,
                               dtype=expr.dtype,
                               shape=new_shape,
                               tags=expr.tags,
                               axes=expr.axes)
        else:
            raise NotImplementedError(type(concat))

    def map_data_wrapper(self,
                         expr: DataWrapper,
                         exprs_from_other_calls: Tuple[Array, ...]
                         ) -> Array:
        concat = self._get_concatenatability(expr)
        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            return concatenate((expr,) + exprs_from_other_calls, concat.axis)
        else:
            raise NotImplementedError(type(concat))

    def map_index_lambda(self,
                         expr: IndexLambda,
                         exprs_from_other_calls: Tuple[Array, ...]
                         ) -> Array:
        concat = self._get_concatenatability(expr)
        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, IndexLambda)
                       for ary in exprs_from_other_calls)

            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are IndexLambda.
            new_bindings = {
                bnd_name: self.rec(
                           subexpr,
                           tuple(ary.bindings[bnd_name]  # type: ignore[attr-defined]
                                 for ary in exprs_from_other_calls))
                for bnd_name, subexpr in expr.bindings.items()
            }
            new_shape = _get_concatenated_shape((expr,) + exprs_from_other_calls,
                                                concat.axis)
            return IndexLambda(expr=expr.expr,
                               shape=new_shape,
                               dtype=expr.dtype,
                               bindings=Map(new_bindings),
                               var_to_reduction_descr=expr.var_to_reduction_descr,
                               tags=expr.tags,
                               axes=expr.axes)
        else:
            raise NotImplementedError(type(concat))

    def map_einsum(self, expr: Einsum,
                   exprs_from_other_calls: Tuple[Array, ...]) -> Array:
        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, Einsum) for ary in exprs_from_other_calls)

            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Einsum.
            new_args = [self.rec(arg,
                                 tuple(ary.args[iarg]  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls))
                        for iarg, arg in enumerate(expr.args)]

            return Einsum(expr.access_descriptors,
                          tuple(new_args),
                          expr.redn_axis_to_redn_descr,
                          expr.index_to_access_descr,
                          tags=expr.tags,
                          axes=expr.axes,
                          )
        else:
            raise NotImplementedError(type(concat))

    def _map_index_base(self, expr: IndexBase,
                        exprs_from_other_calls: Tuple[Array, ...]) -> Array:
        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, IndexBase) for ary in exprs_from_other_calls)

            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are IndexBase.
            new_indices = [
                self.rec(idx,
                         tuple(ary.indices[i_idx]  # type: ignore[attr-defined]
                               for ary in exprs_from_other_calls))
                if isinstance(idx, Array)
                else idx
                for i_idx, idx in enumerate(expr.indices)
            ]
            new_array = self.rec(expr.array,
                                 tuple(ary.array  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls))

            return type(expr)(array=new_array,
                              indices=tuple(new_indices),
                              tags=expr.tags,
                              axes=expr.axes)
        else:
            raise NotImplementedError(type(concat))

    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base
    map_basic_index = _map_index_base

    def map_roll(self,
                 expr: Roll,
                 exprs_from_other_calls: Tuple[Array, ...]
                 ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert concat.axis != expr.axis
            assert all(isinstance(ary, Roll) for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Roll.
            return Roll(self.rec(expr.array,
                                 tuple(ary.array  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls)),
                        shift=expr.shift,
                        axis=expr.axis,
                        tags=expr.tags,
                        axes=expr.axes)
        else:
            raise NotImplementedError(type(concat))

    def map_stack(self, expr: Stack,
                  exprs_from_other_calls: Tuple[Array, ...]
                  ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, Stack) for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Stack.
            if any(len(ary.arrays) != len(expr.arrays)  # type: ignore[attr-defined]
                   for ary in exprs_from_other_calls):
                raise ValueError("Cannot concatenate stack expressions"
                                 " with different number of arrays.")

            new_arrays = tuple(
                self.rec(array,
                         tuple(subexpr.arrays[iarray]  # type: ignore[attr-defined]
                               for subexpr in exprs_from_other_calls)
                         )
                for iarray, array in enumerate(expr.arrays))

            return Stack(new_arrays,
                         expr.axis,
                         tags=expr.tags,
                         axes=expr.axes)
        else:
            raise NotImplementedError(type(concat))

    def map_concatenate(self, expr: Concatenate,
                        exprs_from_other_calls: Tuple[Array, ...]
                        ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, Concatenate)
                       for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Concatenate.
            if any(len(ary.arrays) != len(expr.arrays)  # type: ignore[attr-defined]
                   for ary in exprs_from_other_calls):
                raise ValueError("Cannot concatenate concatenate-expressions"
                                 " with different number of arrays.")

            new_arrays = tuple(
                self.rec(array,
                         tuple(subexpr.arrays[iarray]  # type: ignore[attr-defined]
                               for subexpr in exprs_from_other_calls)
                         )
                for iarray, array in enumerate(expr.arrays)
            )

            return Concatenate(new_arrays,
                               expr.axis,
                               tags=expr.tags,
                               axes=expr.axes)
        else:
            raise NotImplementedError(type(concat))

    def map_axis_permutation(self, expr: AxisPermutation,
                             exprs_from_other_calls: Tuple[Array, ...]
                             ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, AxisPermutation)
                       for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are AxisPermutation.
            new_array = self.rec(expr.array,
                                 tuple(ary.array  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls))
            return AxisPermutation(new_array,
                                   expr.axis_permutation,
                                   tags=expr.tags,
                                   axes=expr.axes)
        else:
            raise NotImplementedError(type(concat))

    def map_reshape(self, expr: Reshape,
                    exprs_from_other_calls: Tuple[Array, ...]
                    ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            new_newshape = _get_concatenated_shape(
                (expr,) + exprs_from_other_calls, concat.axis)

            assert all(isinstance(ary, Reshape) for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Reshape.
            new_array = self.rec(expr.array,
                                 tuple(ary.array  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls))
            return Reshape(new_array,
                           new_newshape,
                           expr.order,
                           tags=expr.tags,
                           axes=expr.axes)
        else:
            raise NotImplementedError(type(concat))

    @memoize_method
    def map_call(self, expr: Call, other_callsites: Tuple[Call, ...]) -> Call:
        new_bindings = {name: self.rec(bnd,
                                       tuple(callsite.bindings[name]
                                             for callsite in other_callsites))
                        for name, bnd in expr.bindings.items()}
        new_mapper = self.clone_with_new_call_on_stack(expr)
        fn_defn = expr.function
        new_fn_defn = FunctionDefinition(
            fn_defn.parameters,
            fn_defn.return_type,
            Map({ret: new_mapper(ret_val,
                                 tuple(other_call.function.returns[ret]
                                       for other_call in other_callsites)
                                 )
                 for ret, ret_val in fn_defn.returns.items()}),
            tags=fn_defn.tags,
        )
        return Call(new_fn_defn,
                    Map(new_bindings),
                    tags=expr.tags)

    def map_named_call_result(self,
                              expr: NamedCallResult,
                              exprs_from_other_calls: Tuple[Array, ...]
                              ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, NamedCallResult)
                       for ary in exprs_from_other_calls)
            assert isinstance(expr._container, Call)
            new_call = self.map_call(
                expr._container,
                tuple(ary._container  # type: ignore[attr-defined]
                      for ary in exprs_from_other_calls))
            return new_call[expr.name]
        else:
            raise NotImplementedError(type(concat))

    def map_loopy_call_result(self, expr: "LoopyCallResult",
                              exprs_from_other_calls: Tuple[Array, ...],
                              ) -> _InputConcatabilityGetterAcc:
        raise ValueError("Loopy Calls are illegal to concatenate. Maybe"
                         " rewrite the operation as array operations?")


@memoize_on_first_arg
def _get_valid_concatenatability_constraints(fn: FunctionDefinition,
                                             ) -> Tuple[FunctionConcatenability,
                                                        ...]:
    mapper = _InputConcatabilityGetter()
    output_accs = {name: mapper(output)
                   for name, output in fn.returns.items()}

    return _combine_named_result_accs(output_accs)


def _get_ary_to_concatenatabilities(call_sites: Sequence[Call],
                                    ) -> Generator[Map[ArrayOnStackT,
                                                       Concatenatability],
                                                   None,
                                                   None]:
    """
    Generates a :class:`Concatenatability` criterion for each array in the
    expression graph of *call_sites*'s function body if they traverse identical
    function bodies.
    """
    fn_body = next(iter(call_sites)).function
    fn_concatenatabilities = _get_valid_concatenatability_constraints(fn_body)

    for fn_concatenatability in fn_concatenatabilities:
        collector = _ConcatabilityCollector(current_stack=())

        # select a template call site to start the traversal.
        template_call, *other_calls = call_sites

        try:
            # verify the constraints on parameters are satisfied
            for name, input_concat in (fn_concatenatability
                                       .input_to_concatenatability
                                       .items()):
                if isinstance(input_concat, ConcatableIfConstant):
                    _verify_arrays_same([cs.bindings[name] for cs in call_sites])
                elif isinstance(input_concat, ConcatableAlongAxis):
                    _verify_arrays_can_be_concated_along_axis(
                        [cs.bindings[name] for cs in call_sites],
                        [],
                        input_concat.axis)
                else:
                    raise NotImplementedError(type(input_concat))

            # verify the constraints on function bodies are satisfied
            for name, output_concat in (fn_concatenatability
                                        .output_to_concatenatability
                                        .items()):
                collector(template_call.function.returns[name],
                          output_concat,
                          tuple(other_call.function.returns[name]
                                for other_call in other_calls))
        except _InvalidConcatenatability:
            pass
        else:
            if collector.call_sites_on_hold:
                raise NotImplementedError("Expressions that use part of"
                                          " function's returned values are not"
                                          " yet supported.")

            logger.info("Found a valid concatenatability --\n"
                        f"{fn_concatenatability}")

            yield Map(collector.ary_to_concatenatability)


def _get_replacement_map_post_concatenating(call_sites: Sequence[Call],
                                            ) -> Mapping[NamedCallResult,
                                                         Array]:
    """
    .. note::

        We require *call_sites* to be ordered to determine the concatenation
        order.
    """
    assert call_sites, "Empty `call_sites`."

    ary_to_concatenatabilities = _get_ary_to_concatenatabilities(call_sites)

    try:
        ary_to_concatenatability = next(ary_to_concatenatabilities)
    except StopIteration:
        raise ValueError("No valid concatenatibilities found.")
    else:
        if __debug__:
            try:
                next(ary_to_concatenatabilities)
            except StopIteration:
                # unique concatenatibility
                pass
            else:
                from warnings import warn
                # TODO: Take some input from the user to resolve this ambiguity.
                warn("Multiple concatenation possibilities found. This may"
                     " lead to non-deterministic transformed expression graphs.")

    # {{{ actually perform the concatenation

    template_call_site, *other_call_sites = call_sites

    concatenator = _Concatenator(current_stack=(),
                                 ary_to_concatenatability=ary_to_concatenatability)

    # new_returns: concatenated function body
    new_returns: Dict[str, Array] = {}
    for output_name in template_call_site.keys():
        new_returns[output_name] = concatenator(
            template_call_site.function.returns[output_name],
            tuple(csite.function.returns[output_name]
                  for csite in other_call_sites))

    # }}}

    # construct new function body
    new_function = FunctionDefinition(
        template_call_site.function.parameters,
        template_call_site.function.return_type,
        Map(new_returns),
        tags=template_call_site.function.tags,
    )

    result: Dict[NamedCallResult, Array] = {}

    new_call_bindings: Dict[str, Array] = {}

    # construct new bindings
    for param_name in template_call_site.bindings:
        param_placeholder = template_call_site.function.get_placeholder(param_name)
        param_concat = ary_to_concatenatability[((), param_placeholder)]
        if isinstance(param_concat, ConcatableAlongAxis):
            new_binding = concatenate([csite.bindings[param_name]
                                              for csite in call_sites],
                                             param_concat.axis)
        elif isinstance(param_concat, ConcatableIfConstant):
            _verify_arrays_same([csite.bindings[param_name]
                                 for csite in call_sites])
            new_binding = template_call_site.bindings[param_name]
        else:
            raise NotImplementedError(type(param_concat))
        new_call_bindings[param_name] = new_binding

    # construct new call
    new_call = Call(
        function=new_function,
        bindings=Map(new_call_bindings),
        tags=template_call_site.tags)

    # slice into new_call's outputs to replace the old expressions.
    for output_name, output_ary in (template_call_site
                                    .function
                                    .returns
                                    .items()):
        start_idx: ShapeComponent = 0
        for cs in call_sites:
            concat = ary_to_concatenatability[((), output_ary)]
            if isinstance(concat, ConcatableIfConstant):
                result[cs[output_name]] = new_call[output_name]
            elif isinstance(concat, ConcatableAlongAxis):
                ndim = output_ary.ndim
                indices = [slice(None) for i in range(ndim)]
                indices[concat.axis] = slice(
                    start_idx, start_idx+cs[output_name].shape[concat.axis])

                sliced_output = new_call[output_name][tuple(indices)]
                assert isinstance(sliced_output, BasicIndex)
                result[cs[output_name]] = sliced_output
                start_idx = start_idx + cs[output_name].shape[concat.axis]
            else:
                raise NotImplementedError(type(concat))

    return Map(result)


def concatenate_calls(expr: ArrayOrNames,
                      call_site_filter: Callable[[CallSiteLocation], bool],
                      *,
                      warn_if_no_calls: bool = True,
                      err_if_no_calls: bool = False,
                      ) -> ArrayOrNames:
    r"""
    Returns a copy of *expr* after concatenating all call-sites ``C`` such that
    ``call_site_filter(C) is True``.

    :arg call_site_filter: A callable to select which instances of
        :class:`~pytato.function.Call`\ s must be concatenated.
    """
    all_call_sites = CallsiteCollector(stack=())(expr)

    filtered_call_sites = {callsite
                           for callsite in all_call_sites
                           if call_site_filter(callsite)}

    if not filtered_call_sites:
        if err_if_no_calls:
            raise ValueError("No calls to concatenate.")
        elif warn_if_no_calls:
            from warnings import warn
            warn("No calls to concatenate.", stacklevel=2)
        else:
            pass
        return expr
    elif len({csite.stack for csite in filtered_call_sites}) == 1:
        pass
    else:
        raise ValueError("Call-sites to concatenate are called"
                         " at multiple stack frames. This is not allowed.")

    old_expr_to_new_expr_map = (
        _get_replacement_map_post_concatenating([csite.call
                                                 for csite in filtered_call_sites]))

    stack, = {csite.stack for csite in filtered_call_sites}

    result = _NamedCallResultReplacerPostConcatenate(
        replacement_map=old_expr_to_new_expr_map,
        current_stack=(),
        stack_to_replace_on=stack)(expr)

    assert isinstance(result, (Array, AbstractResultWithNamedArrays))
    return result

# }}}

# vim:foldmethod=marker
