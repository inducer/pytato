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

import collections
import contextlib
import dataclasses
from typing import (
        Any, Union, Optional, Mapping, Iterator, Dict, Tuple, FrozenSet,
        Set)
import typing

import islpy as isl
import loopy as lp
import pymbolic.primitives as prim

from pytato.array import (
        Array, DictOfNamedArrays, Placeholder, ShapeType, IndexLambda)
from pytato.program import BoundProgram, Target, PyOpenCLTarget
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
ReductionBounds = Dict[str, Tuple[ScalarExpression, ScalarExpression]]


@dataclasses.dataclass(init=True, repr=False, eq=False)
class LoopyExpressionContext(object):
    """Contextual data for generating :mod:`loopy` expressions.

    This data is passed through :class:`LoopyExpressionGenMapper` via arguments,
    and is also used by :meth:`ImplementedResult.to_loopy_expression` to
    retrieve contextual data.

    .. attribute:: state

        The :class:`CodeGenState`.

    .. attribute:: depends_on

        The set of statement IDs that need to be included in
        :attr:`loopy.kernel.data.instruction.InstructionBase.depends_on`.

    .. attribute:: reduction_bounds

        A mapping from inames to reduction bounds in the expression.
    """
    state: CodeGenState
    _depends_on: FrozenSet[str]
    reduction_bounds: ReductionBounds

    @property
    def namespace(self) -> typing.ChainMap[str, Array]:
        return self.state.namespace

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
        """Return a :mod:`loopy` expression for this result."""
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
    def __init__(self, expr: ScalarExpression, array: Array):
        super().__init__(array)
        self.expr = expr

    # TODO: Handle dependencies and reduction domains.
    def to_loopy_expression(self, indices: SymbolicIndex,
            expr_context: LoopyExpressionContext) -> ScalarExpression:
        return scalar_expr.substitute(
                self.expr,
                {f"_{d}": i for d, i in zip(range(self.array.ndim), indices)})


class SubstitutionRuleResult(ImplementedResult):
    # TODO: implement
    pass

# }}}


# {{{ codegen

@dataclasses.dataclass(init=True, repr=False, eq=False)
class CodeGenState:
    """A container for data kept by :class:`CodeGenMapper`.

    .. attribute:: namespace

        The namespace

    .. attribute:: kernel

        The partial :class:`loopy.LoopKernel` being built.

    .. attribute:: results

        A mapping from :class:`pytato.array.Array` instances to
        instances of :class:`ImplementedResult`.

    .. attribute:: var_name_gen
    .. attribute:: insn_id_gen

    .. automethod:: update_kernel
    .. automethod:: chain_namespaces
    .. automethod:: make_expression_context
    """
    namespace: typing.ChainMap[str, Array]
    _kernel: lp.LoopKernel
    results: Dict[Array, ImplementedResult]

    # Both of these have type Callable[[str], str], but mypy's support for that
    # is broken (https://github.com/python/mypy/issues/6910)
    var_name_gen: Any = dataclasses.field(init=False)
    insn_id_gen: Any = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.var_name_gen = self._kernel.get_var_name_generator()
        self.insn_id_gen = self._kernel.get_instruction_id_generator()

    @property
    def kernel(self) -> lp.LoopKernel:
        return self._kernel

    def update_kernel(self, kernel: lp.LoopKernel) -> None:
        self._kernel = kernel

    @contextlib.contextmanager
    def chain_namespaces(
            self,
            local_namespace: Mapping[str, Array]) -> Iterator[CodeGenState]:
        """A context manager for overriding with a local scope."""
        self.namespace.maps.insert(0, local_namespace)
        yield self
        self.namespace.maps.pop(0)

    def make_expression_context(
            self,
            depends_on: FrozenSet[str] = frozenset(),
            reduction_bounds: Optional[ReductionBounds] = None
            ) -> LoopyExpressionContext:
        """Get a new :class:`LoopyExpressionContext`."""
        if reduction_bounds is None:
            reduction_bounds = {}
        return LoopyExpressionContext(self,
                _depends_on=depends_on,
                reduction_bounds=reduction_bounds)


class CodeGenMapper(pytato.transform.Mapper):
    """A mapper for generating code for nodes in the computation graph.
    """
    exprgen_mapper: InlinedExpressionGenMapper

    def __init__(self) -> None:
        self.exprgen_mapper = InlinedExpressionGenMapper(self)

    def map_placeholder(self, expr: Placeholder,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        arg = lp.GlobalArg(expr.name,
                shape=expr.shape,
                dtype=expr.dtype,
                order="C")
        kernel = state.kernel.copy(args=state.kernel.args + [arg])
        state.update_kernel(kernel)

        result = StoredResult(expr.name, expr)
        state.results[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        # TODO: Respect tags.

        with state.chain_namespaces(expr.bindings) as chained_state:
            expr_context = chained_state.make_expression_context()
            loopy_expr = self.exprgen_mapper(expr.expr, expr_context)

        result = InlinedResult(loopy_expr, expr)
        state.results[expr] = result
        return result

# }}}


# {{{ inlined expression gen mapper

class InlinedExpressionGenMapper(scalar_expr.IdentityMapper):
    """A mapper for generating :mod:`loopy` expressions with inlined
    sub-expressions.

    The inputs to this mapper are scalar expression as found in
    :class:`pytato.IndexLambda`, or expressions that are compatible (e.g., shape
    expressions).

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
                expr_context.namespace[expr.aggregate.name], expr_context.state)
        return result.to_loopy_expression(expr.index, expr_context)

    # TODO: map_reduction()

    def map_variable(self, expr: prim.Variable,
            expr_context: LoopyExpressionContext) -> ScalarExpression:
        result: ImplementedResult = self.codegen_mapper(
                expr_context.namespace[expr.name],
                expr_context.state)
        return result.to_loopy_expression((), expr_context)

# }}}


# {{{ utils

def domain_for_shape(
        dim_names: Tuple[str, ...], shape: ShapeType) -> isl.BasicSet:
    """Create a :class:`islpy.BasicSet` that expresses an appropriate index domain
    for an array of (potentially symbolic) shape *shape*.

    :param dim_names: A tuple of strings, the names of the axes. These become set
        dimensions in the returned domain.

    :param shape: A tuple of constant or quasi-affine :mod:`pymbolic`
        expressions. The variables in these expressions become parameter
        dimensions in the returned set.  Must have the same length as
        *dim_names*.
    """
    assert len(dim_names) == len(shape)

    # Collect parameters.
    param_names_set: Set[str] = set()
    for sdep in map(scalar_expr.get_dependencies, shape):
        param_names_set |= sdep

    set_names = sorted(dim_names)
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

    dom, = dom.get_basic_sets()

    return dom


def add_output(name: str, expr: Array, state: CodeGenState,
        mapper: CodeGenMapper) -> None:
    """Add an output argument to the kernel.
    """
    # FIXE: Scalar outputs are not supported yet.
    assert expr.shape != ()

    result = mapper(expr, state)
    name = state.var_name_gen(name)

    inames = tuple(
            state.var_name_gen(f"{name}_dim{d}")
            for d in range(expr.ndim))
    domain = domain_for_shape(inames, expr.shape)

    arg = lp.GlobalArg(name,
            shape=expr.shape,
            dtype=expr.dtype,
            order="C",
            is_output_only=True)

    indices = tuple(prim.Variable(iname) for iname in inames)
    expr_context = state.make_expression_context()
    copy_expr = result.to_loopy_expression(indices, expr_context)

    # TODO: Contextual data not supported yet.
    assert not expr_context.reduction_bounds
    assert not expr_context.depends_on

    from loopy.kernel.instruction import make_assignment
    insn = make_assignment((prim.Variable(name)[indices],),
            copy_expr,
            id=state.insn_id_gen(f"{name}_copy"),
            within_inames=frozenset(inames),
            depends_on=expr_context.depends_on)

    kernel = state.kernel
    kernel = kernel.copy(args=kernel.args + [arg],
            instructions=kernel.instructions + [insn],
            domains=kernel.domains + [domain])
    state.update_kernel(kernel)

# }}}


def generate_loopy(result: Union[Array, DictOfNamedArrays],
        target: Optional[Target] = None) -> BoundProgram:
    r"""Code generation entry point.

    :param result: Outputs of the computation.
    :returns: A wrapped generated :mod:`loopy` kernel
    """
    # {{{ get namespace and outputs

    outputs: DictOfNamedArrays

    if isinstance(result, Array):
        outputs = DictOfNamedArrays({"out": result})
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
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION)

    state = CodeGenState(namespace=collections.ChainMap(namespace),
            _kernel=kernel,
            results=dict())

    # Generate code for graph nodes.
    mapper = CodeGenMapper()
    for name, val in namespace.items():
        _ = mapper(val, state)

    # Generate code for outputs.
    for name, expr in outputs.items():
        add_output(name, expr, state, mapper)

    return target.bind_program(program=state.kernel, bound_arguments=dict())
