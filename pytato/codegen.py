from __future__ import annotations

__copyright__ = """Copyright (C) 2020 Matt Wala"""

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

import dataclasses
from typing import (Union, Optional, Mapping, Dict, Tuple, FrozenSet, Set)

import islpy as isl
import loopy as lp
import pymbolic.primitives as prim
from pymbolic import var
import pytools

from pytato.array import (
        Array, DictOfNamedArrays, Placeholder, ShapeType, IndexLambda,
        SizeParam, DataWrapper, InputArgumentBase, MatrixProduct, Roll,
        AxisPermutation)
from pytato.program import BoundProgram
from pytato.target import Target, PyOpenCLTarget
import pytato.scalar_expr as scalar_expr
from pytato.scalar_expr import ScalarExpression
import pytato.transform


__doc__ = """
Generating Code
---------------

.. currentmodule:: pytato

.. autofunction:: generate_loopy

Code Generation Internals
-------------------------

.. currentmodule:: pytato.codegen

.. autoclass:: LoopyExpressionContext
.. autoclass:: ImplementedResult
.. autoclass:: StoredResult
.. autoclass:: InlinedResult
.. autoclass:: SubstitutionRuleResult

.. autoclass:: CodeGenState
.. autoclass:: CodeGenMapper

.. autoclass:: InlinedExpressionGenMapper

.. autofunction:: domain_for_shape
.. autofunction:: add_output

"""


# {{{ generated array expressions

# SymbolicIndex and ShapeType are semantically distinct but identical at the
# type level.
SymbolicIndex = ShapeType
ReductionBounds = Dict[int, Tuple[ScalarExpression, ScalarExpression]]


@dataclasses.dataclass(init=True, repr=False, eq=False)
class LoopyExpressionContext(object):
    """Mutable state used while generating :mod:`loopy` expressions.
    Wraps :class:`CodeGenState` with more expression-specific information.

    This data is passed through :class:`InlinedExpressionGenMapper` via arguments,
    and is also used by :meth:`ImplementedResult.to_loopy_expression` to
    retrieve contextual data.

    .. attribute:: state

        The :class:`CodeGenState`.

    .. attribute:: local_namespace

        A (read-only) local name mapping used for name lookup when generating
        code.

    .. attribute:: depends_on

        The set of statement IDs that need to be included in
        :attr:`loopy.InstructionBase.depends_on`.

    .. attribute:: reduction_bounds

        A mapping from reduction iname number (for reduction inames ``_r0``,
        ``_r1``, ...) to reduction bounds in the expression.

    .. automethod:: update_depends_on
    .. automethod:: lookup

    """
    state: CodeGenState
    _depends_on: FrozenSet[str] = \
            dataclasses.field(default_factory=frozenset)
    local_namespace: Mapping[str, Array] = \
            dataclasses.field(default_factory=dict)
    reduction_bounds: ReductionBounds = \
            dataclasses.field(default_factory=dict)

    def lookup(self, name: str) -> Array:
        try:
            return self.local_namespace[name]
        except KeyError:
            return self.state.namespace[name]

    @property
    def depends_on(self) -> FrozenSet[str]:
        return self._depends_on

    def update_depends_on(self, other: FrozenSet[str]) -> None:
        self._depends_on = self._depends_on | other


class ImplementedResult(object):
    """Generated code for a node in the computation graph (i.e., an array
    expression).

    .. attribute:: array

        The :class:`pytato.Array` associated with this code.

    .. automethod:: to_loopy_expression
    """
    def __init__(self, array: Array):
        self.array = array

    def to_loopy_expression(self, indices: SymbolicIndex,
            expr_context: LoopyExpressionContext) -> ScalarExpression:
        """Return a :mod:`loopy` expression for this result.

        :param indices: symbolic expressions for the indices of the array
        :param expr_context: the associated expression context. The fields are
            treated as follows:

               - *depends_on* is populated with any dependencies needed for the
                 generated expression.

               - *reduction_bounds* is populated with reduction bounds for the
                 reduction inames in the returned expression. If
                 *reduction_bounds* is nonempty, then the returned inames are
                 ensured to be disjoint from those present.
        """
        raise NotImplementedError


class StoredResult(ImplementedResult):
    """An array expression generated as a :mod:`loopy` array.

    See also: :class:`pytato.array.ImplStored`.
    """
    def __init__(self, name: str, array: Array):
        super().__init__(array)
        self.name = name

    # TODO: Handle dependencies.
    def to_loopy_expression(self, indices: SymbolicIndex,
            expr_context: LoopyExpressionContext) -> ScalarExpression:
        if indices == ():
            return prim.Variable(self.name)
        else:
            return prim.Variable(self.name)[indices]


class InlinedResult(ImplementedResult):
    """An array expression generated as a :mod:`loopy` expression containing inlined
    sub-expressions.

    See also: :class:`pytato.array.ImplInlined`.
    """
    def __init__(self, expr: ScalarExpression, array: Array,
            reduction_bounds: ReductionBounds, depends_on: FrozenSet[str]):
        super().__init__(array)
        self.expr = expr
        self.reduction_bounds = dict(reduction_bounds)
        self.depends_on = depends_on

    def to_loopy_expression(self, indices: SymbolicIndex,
            expr_context: LoopyExpressionContext) -> ScalarExpression:
        assert len(indices) == self.array.ndim
        substitutions = {f"_{d}": i for d, i in enumerate(indices)}

        reduction_start = 0
        if expr_context.reduction_bounds:
            reduction_start = 1 + max(expr_context.reduction_bounds)

        for i, bounds in self.reduction_bounds.items():
            j = i + reduction_start
            substitutions[f"_r{i}"] = var(f"_r{j}")
            expr_context.reduction_bounds[j] = bounds

        expr_context.update_depends_on(self.depends_on)

        return scalar_expr.substitute(self.expr, substitutions)


class SubstitutionRuleResult(ImplementedResult):
    # TODO: implement
    pass

# }}}


# {{{ codegen

@dataclasses.dataclass(init=True, repr=False, eq=False)
class CodeGenState:
    """A container for data kept by :class:`CodeGenMapper`.

    .. attribute:: namespace

        The (global) namespace

    .. attribute:: kernel

        The partial :class:`loopy.LoopKernel` being built.

    .. attribute:: results

        A mapping from :class:`pytato.Array` instances to
        instances of :class:`ImplementedResult`.

    .. attribute:: var_name_gen
    .. attribute:: insn_id_gen

    .. automethod:: update_kernel
    """
    namespace: Mapping[str, Array]
    _kernel: lp.LoopKernel
    results: Dict[Array, ImplementedResult]

    var_name_gen: pytools.UniqueNameGenerator = dataclasses.field(init=False)
    insn_id_gen: pytools.UniqueNameGenerator = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.var_name_gen = self._kernel.get_var_name_generator()
        self.insn_id_gen = self._kernel.get_instruction_id_generator()

    @property
    def kernel(self) -> lp.LoopKernel:
        return self._kernel

    def update_kernel(self, kernel: lp.LoopKernel) -> None:
        self._kernel = kernel


class CodeGenMapper(pytato.transform.Mapper):
    """A mapper for generating code for nodes in the computation graph.
    """
    exprgen_mapper: InlinedExpressionGenMapper

    def __init__(self) -> None:
        self.exprgen_mapper = InlinedExpressionGenMapper(self)

    def map_size_param(self, expr: SizeParam,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        arg = lp.ValueArg(expr.name, dtype=expr.dtype)
        kernel = state.kernel.copy(args=state.kernel.args + [arg])
        state.update_kernel(kernel)

        result = StoredResult(expr.name, expr)
        state.results[expr] = result
        return result

    def handle_array_input_argument(self, expr: Union[Placeholder, DataWrapper],
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        shape_context = LoopyExpressionContext(state)
        shape = []
        for component in expr.shape:
            shape.append(self.exprgen_mapper(component, shape_context))
            # Not supported yet.
            assert not shape_context.depends_on
            assert not shape_context.reduction_bounds

        arg = lp.GlobalArg(expr.name,
                shape=tuple(shape),
                dtype=expr.dtype,
                order="C")
        kernel = state.kernel.copy(args=state.kernel.args + [arg])
        state.update_kernel(kernel)

        result = StoredResult(expr.name, expr)
        state.results[expr] = result
        return result

    map_placeholder = handle_array_input_argument
    map_data_wrapper = handle_array_input_argument

    def map_matrix_product(self, expr: MatrixProduct,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        x1_result = self.rec(expr.x1, state)
        x2_result = self.rec(expr.x2, state)

        expr_context = LoopyExpressionContext(state)
        expr_context.reduction_bounds[0] = (0, expr.x2.shape[0])

        # Figure out inames.
        x1_inames = []
        for i in range(expr.x1.ndim):
            if i == expr.x1.ndim - 1:
                x1_inames.append(var("_r0"))
            else:
                x1_inames.append(var(f"_{i}"))
        x2_inames = []
        for i in range(expr.x2.ndim):
            if i == 0:
                x2_inames.append(var("_r0"))
            else:
                offset = i + len(x1_inames) - 2
                x2_inames.append(var(f"_{offset}"))

        inner_expr = x1_result.to_loopy_expression(
                tuple(x1_inames), expr_context)
        inner_expr *= x2_result.to_loopy_expression(
                tuple(x2_inames), expr_context)

        import loopy.library.reduction as red
        result_expr = lp.Reduction(
                operation=red.parse_reduction_op("sum"),
                inames=("_r0",),
                expr=inner_expr,
                allow_simultaneous=False)

        result = InlinedResult(result_expr,
                expr,
                expr_context.reduction_bounds,
                expr_context.depends_on)

        state.results[expr] = result
        return result

    def map_roll(self, expr: Roll,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        indices = [var(f"_{d}") for d in range(expr.ndim)]
        indices[expr.axis] = (
                indices[expr.axis] + expr.shift) % expr.shape[expr.axis]

        expr_context = LoopyExpressionContext(state)
        loopy_expr = (
                self.rec(expr.array, state)
                .to_loopy_expression(tuple(indices), expr_context))

        result = InlinedResult(loopy_expr,
                expr,
                expr_context.reduction_bounds,
                expr_context.depends_on)

        state.results[expr] = result
        return result

    def map_axis_permutation(self, expr: AxisPermutation,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        indices = [None] * expr.ndim
        for from_index, to_index in enumerate(expr.axes):
            indices[to_index] = var(f"_{from_index}")

        expr_context = LoopyExpressionContext(state)
        loopy_expr = (
                self.rec(expr.array, state)
                .to_loopy_expression(tuple(indices), expr_context))

        result = InlinedResult(loopy_expr,
                expr,
                expr_context.reduction_bounds,
                expr_context.depends_on)

        state.results[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        # TODO: Respect tags.

        expr_context = LoopyExpressionContext(state,
                local_namespace=expr.bindings)
        loopy_expr = self.exprgen_mapper(expr.expr, expr_context)

        result = InlinedResult(loopy_expr, expr, {}, frozenset())
        state.results[expr] = result
        return result

# }}}


# {{{ inlined expression gen mapper

class InlinedExpressionGenMapper(scalar_expr.IdentityMapper):
    """A mapper for generating :mod:`loopy` expressions with inlined
    sub-expressions.

    The inputs to this mapper are scalar expression as found in
    :class:`pytato.array.IndexLambda`, or expressions that are
    compatible (e.g., shape expressions).

    The outputs of this mapper are scalar expressions suitable for wrapping in
    :class:`InlinedResult`.
    """
    codegen_mapper: CodeGenMapper

    def __init__(self, codegen_mapper: CodeGenMapper):
        self.codegen_mapper = codegen_mapper

    def __call__(self, expr: ScalarExpression,
            expr_context: LoopyExpressionContext) -> ScalarExpression:
        return self.rec(expr, expr_context)

    def map_subscript(self, expr: prim.Subscript,
            expr_context: LoopyExpressionContext) -> ScalarExpression:
        assert isinstance(expr.aggregate, prim.Variable)
        result: ImplementedResult = self.codegen_mapper(
                expr_context.lookup(expr.aggregate.name), expr_context.state)
        return result.to_loopy_expression(expr.index, expr_context)

    # TODO: map_reduction()

    def map_variable(self, expr: prim.Variable,
            expr_context: LoopyExpressionContext) -> ScalarExpression:
        result: ImplementedResult = self.codegen_mapper(
                expr_context.lookup(expr.name),
                expr_context.state)
        return result.to_loopy_expression((), expr_context)

# }}}


# {{{ utils

def domain_for_shape(dim_names: Tuple[str, ...],
         shape: ShapeType,
         reductions: Dict[str, Tuple[ScalarExpression, ScalarExpression]],
         ) -> isl.BasicSet:  # noqa
    """Create an :class:`islpy.BasicSet` that expresses an appropriate index domain
    for an array of (potentially symbolic) shape *shape* having reduction
    dimensions *reductions*.

    :param dim_names: A tuple of strings, the names of the axes. These become set
        dimensions in the returned domain.

    :param shape: A tuple of constant or quasi-affine :mod:`pymbolic`
        expressions. The variables in these expressions become parameter
        dimensions in the returned set.  Must have the same length as
        *dim_names*.

    :arg reductions: A map from reduction inames to (lower, upper) bounds
        (as half-open integer ranges). The variables in the bounds become
        parameter dimensions in the returned set.
    """
    assert len(dim_names) == len(shape)

    # Collect parameters.
    param_names_set: Set[str] = set()
    for sdep in map(scalar_expr.get_dependencies, shape):
        param_names_set |= sdep

    for bounds in reductions.values():
        for sdep in map(scalar_expr.get_dependencies, bounds):
            # FIXME: Assumes that reduction bounds are not data-dependent.
            param_names_set |= sdep

    set_names = sorted(tuple(dim_names) + tuple(reductions))
    param_names = sorted(param_names_set)

    # Build domain.
    dom = isl.BasicSet.universe(
            isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
            set=set_names,
            params=param_names))

    # Add constraints.
    from loopy.symbolic import aff_from_expr
    affs = isl.affs_from_space(dom.space)

    for iname, dim in zip(dim_names, shape):
        dom &= affs[0].le_set(affs[iname])
        dom &= affs[iname].lt_set(aff_from_expr(dom.space, dim))

    for iname, (left, right) in reductions.items():
        dom &= aff_from_expr(dom.space, left).le_set(affs[iname])
        dom &= affs[iname].lt_set(aff_from_expr(dom.space, right))

    dom, = dom.get_basic_sets()

    return dom


def add_output(name: str, expr: Array, state: CodeGenState,
        mapper: CodeGenMapper) -> None:
    """Add an output argument to the kernel.
    """
    result = mapper(expr, state)

    # Get expression.
    inames = tuple(
            state.var_name_gen(f"{name}_dim{d}")
            for d in range(expr.ndim))
    indices = tuple(prim.Variable(iname) for iname in inames)
    expr_context = LoopyExpressionContext(state)
    copy_expr = result.to_loopy_expression(indices, expr_context)

    # Make the reduction inames/substitute them into expression.
    reduction_inames = tuple(
            state.var_name_gen(f"{name}_red{i}")
            for i in range(len(expr_context.reduction_bounds)))
    substitutions = {
            f"_r{i}": var(reduction_inames[i])
            for i in expr_context.reduction_bounds}
    copy_expr = scalar_expr.substitute(copy_expr, substitutions)

    # Make the instruction
    from loopy.kernel.instruction import make_assignment
    if indices:
        assignee = prim.Variable(name)[indices]
    else:
        assignee = prim.Variable(name)
    insn = make_assignment((assignee,),
            copy_expr,
            id=state.insn_id_gen(f"{name}_copy"),
            within_inames=frozenset(inames),
            depends_on=expr_context.depends_on)

    # Get the domain.
    reduction_bounds = {
            reduction_inames[i]: bounds
            for i, bounds in expr_context.reduction_bounds.items()}
    domain = domain_for_shape(inames, expr.shape, reduction_bounds)

    # Get the argument.
    arg = lp.GlobalArg(name,
            shape=expr.shape,
            dtype=expr.dtype,
            order="C",
            is_output_only=True)

    # Update the kernel.
    kernel = state.kernel
    kernel = kernel.copy(args=kernel.args + [arg],
            instructions=kernel.instructions + [insn],
            domains=kernel.domains + [domain])
    state.update_kernel(kernel)

# }}}


def generate_loopy(result: Union[Array, DictOfNamedArrays],
        target: Optional[Target] = None,
        options: Optional[lp.Options] = None) -> BoundProgram:
    r"""Code generation entry point.

    :param result: Outputs of the computation.
    :param target: Code generation target.
    :param options: Code generation options for the kernel.
    :returns: A wrapped generated :mod:`loopy` kernel
    """
    # {{{ get namespace and outputs

    outputs: DictOfNamedArrays

    if isinstance(result, Array):
        outputs = DictOfNamedArrays({"_pt_out": result})
        namespace = outputs.namespace
    else:
        assert isinstance(result, DictOfNamedArrays)
        outputs = result

    namespace = outputs.namespace
    del result

    # }}}

    if target is None:
        target = PyOpenCLTarget()

    # Set up codegen state.
    kernel = lp.make_kernel("{:}", [],
            target=target.get_loopy_target(),
            options=options,
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION)

    state = CodeGenState(namespace=namespace,
            _kernel=kernel,
            results=dict())

    # Reserve names of input and output arguments.
    for val in namespace.values():
        if isinstance(val, InputArgumentBase):
            state.var_name_gen.add_name(val.name)
    state.var_name_gen.add_names(outputs)

    # Generate code for graph nodes.
    mapper = CodeGenMapper()
    for name, val in namespace.items():
        _ = mapper(val, state)

    # Generate code for outputs.
    for name, expr in outputs.items():
        add_output(name, expr, state, mapper)

    # Collect bound arguments.
    bound_arguments = {}
    for name, val in namespace.items():
        if isinstance(val, DataWrapper):
            bound_arguments[name] = val.data

    return target.bind_program(
            program=state.kernel,
            bound_arguments=bound_arguments)
