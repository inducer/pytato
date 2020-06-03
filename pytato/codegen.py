from __future__ import annotations

# Codegen output class.


import collections
import dataclasses
from typing import cast, Any, ContextManager, Callable, Union, Optional, Mapping, Iterator, Dict, Tuple, Set, FrozenSet
import typing

import contextlib
import loopy as lp
import numpy as np
import pymbolic.mapper
import pymbolic.primitives as prim
import pytools

from pytato.array import Array, DictOfNamedArrays, Placeholder, Output, Namespace, ShapeType, IndexLambda
from pytato.program import BoundProgram, Target, PyOpenCLTarget
import pytato.symbolic as sym
from pytato.symbolic import ScalarExpression


# These are semantically distinct but identical at the type level.
SymbolicIndex = ShapeType


# {{{ nodal result

class NodalResult(object):

    def __init__(self, shape: ShapeType, dtype: np.dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_loopy_expression(self, indices: SymbolicIndex, context: ExpressionContext, reduction_inames: Optional[Tuple[str, ...]] = None) -> ScalarExpression:
        raise NotImplementedError


class ArrayResult(NodalResult):

    def __init__(self, name: str, shape: ShapeType, dtype: np.dtype):
        # TODO: Stuff dependencies in here.
        super().__init__(shape, dtype)
        self.name = name

    def to_loopy_expression(self, indices: SymbolicIndex, context: ExpressionContext, reduction_inames: Optional[Tuple[str, ...]] = None) -> ScalarExpression:
        if indices == ():
            return prim.Variable(self.name)
        else:
            return prim.Variable(self.name)[indices]


class ExpressionResult(NodalResult):

    def __init__(self, expr: ScalarExpression, shape: ShapeType, dtype: np.dtype):
        super().__init__(shape, dtype)
        self.expr = expr

    def to_loopy_expression(self, indices: SymbolicIndex, context: ExpressionContext, reduction_inames: Optional[Tuple[str, ...]] = None) -> ScalarExpression:
        return sym.substitute(
                self.expr,
                dict(zip(
                        (f"_{d}" for d in range(self.ndim)),
                        indices)))


class SubstitutionRuleResult(NodalResult):
    # TODO: implement
    pass

# }}}


class ExpressionGenMapper(sym.IdentityMapper):
    """A mapper for generating :mod:`loopy` expressions.

    The inputs to this mapper are :class:`IndexLambda` expressions, or
    expressions that are closely equivalent (e.g., shape expressions). In
    particular
    """
    codegen_mapper: CodeGenMapper

    def __init__(self, codegen_mapper: CodeGenMapper):
        self.codegen_mapper = codegen_mapper

    def __call__(self,
            expr: ScalarExpression,
            indices: Tuple[ScalarExpression, ...],
            context: ExpressionContext) -> ScalarExpression:
        return self.rec(expr, indices, context)

    def map_subscript(self, expr: prim.Subscript, indices: SymbolicIndex, context: ExpressionContext) -> ScalarExpression:
        assert isinstance(expr.aggregate, prim.Variable)
        result: NodalResult = self.codegen_mapper(
                context.namespace[expr.aggregate.name],
                context.state)
        assert len(expr.index) == len(indices)
        mapped_indices = sym.substitute(
                expr.index,
                dict(zip(
                        (f"_{d}" for d in range(len(indices))),
                        indices)))
        return result.to_loopy_expression(mapped_indices, context)

    # TODO: map_reduction()

    def map_variable(self, expr: prim.Variable, indices: SymbolicIndex, context: ExpressionContext) -> ScalarExpression:
        result: NodalResult = self.codegen_mapper(
                context.namespace[expr.name],
                context.state)
        return result.to_loopy_expression((), context)


class CodeGenMapper(pymbolic.mapper.Mapper):
    """A mapper for generating code for nodes in the computation graph.
    """
    exprgen_mapper: ExpressionGenMapper

    def __init__(self) -> None:
        self.exprgen_mapper = ExpressionGenMapper(self)

    def map_placeholder(self, expr: Placeholder, state: CodeGenState) -> NodalResult:
        if expr in state.results:
            return state.results[expr]

        arg = lp.GlobalArg(expr.name, shape=expr.shape, dtype=expr.dtype, order="C")
        kernel = state.kernel.copy(args=state.kernel.args + [arg])
        state.update_kernel(kernel)

        result = ArrayResult(expr.name, expr.dtype, expr.shape)
        state.results[expr] = result
        return result

    def map_output(self, expr: Output, state: CodeGenState) -> NodalResult:
        if expr in state.results:
            return state.results[expr]

        # FIXE: Scalar outputs are not supported yet.
        assert expr.shape != ()

        inner_result = self.rec(expr.array, state)
        
        inames = tuple(
                state.var_name_gen(f"{expr.name}_dim{d}")
                for d in range(expr.ndim))
        domain = sym.domain_for_shape(inames, expr.shape)

        arg = lp.GlobalArg(
                expr.name,
                shape=expr.shape,
                dtype=expr.dtype,
                order="C",
                is_output_only=True)

        indices = tuple(prim.Variable(iname) for iname in inames)
        context = state.make_expression_context()
        copy_expr = inner_result.to_loopy_expression(indices, context)
        # TODO: Context data supported yet.
        assert not context.reduction_bounds
        assert not context.depends_on, context.depends_on

        from loopy.kernel.instruction import make_assignment
        insn = make_assignment(
                (prim.Variable(expr.name)[indices],),
                copy_expr,
                id=state.insn_id_gen(f"{expr.name}_copy"),
                within_inames=frozenset(inames),
                depends_on=context.depends_on)

        kernel = state.kernel
        kernel = kernel.copy(
                args=kernel.args + [arg],
                instructions=kernel.instructions + [insn],
                domains=kernel.domains + [domain])
        state.update_kernel(kernel)

        result = ArrayResult(expr.name, expr.dtype, expr.shape)
        state.results[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda, state: CodeGenState) -> NodalResult:
        if expr in state.results:
            return state.results[expr]

        # TODO: tags

        with state.chain_namespaces(expr.bindings) as chained_state:
            expr_context = chained_state.make_expression_context()
            indices = tuple(prim.Variable(f"_{d}") for d in range(expr.ndim))
            generated_expr = self.exprgen_mapper(expr.expr, indices, expr_context)

        result = ExpressionResult(generated_expr, expr.shape, expr.dtype)
        state.results[expr] = result
        return result


@dataclasses.dataclass(init=True, repr=False, eq=False)
class CodeGenState:
    """
    This data is threaded through :class:`CodeGenMapper`.

    .. attribute:: namespace
    .. attribute:: kernel
    .. attribute:: results
    .. attribute:: var_name_gen
    .. attribute:: insn_id_gen
    """
    namespace: typing.ChainMap[str, Array]
    _kernel: lp.LoopKernel
    results: Dict[Array, NodalResult]
    # Both of these have type Callable[[str], str], but mypy's support for that is broken.
    var_name_gen: Any = dataclasses.field(init=False)
    insn_id_gen: Any = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.var_name_gen = self._kernel.get_var_name_generator()
        self.insn_id_gen = self._kernel.get_var_name_generator()

    @property
    def kernel(self) -> lp.LoopKernel:
        return self._kernel

    def update_kernel(self, kernel: lp.LoopyKernel) -> None:
        self._kernel = kernel

    @contextlib.contextmanager
    def chain_namespaces(self, local_namespace: Mapping[str, Array]) -> Iterator[CodeGenState]:
        self.namespace.maps.insert(0, local_namespace)
        yield self
        self.namespace.maps.pop(0)

    def make_expression_context(self, depends_on: FrozenSet[str] = frozenset(), reduction_bounds: Optional[ReductionBounds] = None) -> ExpressionContext:
        if reduction_bounds is None:
            reduction_bounds = {}
        return ExpressionContext(self, _depends_on=depends_on, reduction_bounds=reduction_bounds)


ReductionBounds = Dict[str, Tuple[ScalarExpression, ScalarExpression]]


@dataclasses.dataclass(init=True, repr=False, eq=False)
class ExpressionContext(object):
    """Contextual data for generating :mod:`loopy` expressions.

    This data is threaded through :class:`ExpressionGenMapper`.

    .. attribute:: state
    .. attribute:: _depends_on
    .. attribute:: reduction_bounds
    """
    state: CodeGenState
    _depends_on: FrozenSet[str]
    reduction_bounds: Dict[str, Tuple[ScalarExpression, ScalarExpression]]

    @property
    def namespace(self) -> typing.ChainMap[str, Array]:
        return self.state.namespace
    
    @property
    def depends_on(self) -> FrozenSet[str]:
        return self._depends_on

    def update_depends_on(self, other: FrozenSet[str]) -> None:
        self._depends_on = self._depends_on | other


def generate_loopy(result: Union[Namespace, Array, DictOfNamedArrays],
                   target: Optional[Target] = None) -> BoundProgram:
    # {{{ get namespace

    if isinstance(result, Array):
        if isinstance(result, Output):
            result = result.namespace
        else:
            result = DictOfNamedArrays({"_out": result})

    if isinstance(result, DictOfNamedArrays):
        namespace = result.namespace._chain()

        # Augment with Output nodes.
        name_gen = pytools.UniqueNameGenerator(set(namespace))
        for name, val in result.items():
            out_name = name_gen(name)
            Output(namespace, out_name, val)

        result = namespace.copy()

    assert isinstance(result, Namespace)
    
    # Make an internal copy.
    result = result.copy()

    # }}}

    if target is None:
        target = PyOpenCLTarget()

    # Set up codegen state.
    kernel = lp.make_kernel(
            "{:}", [], target=target.get_loopy_target(),
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION)
    
    state = CodeGenState(
            namespace=collections.ChainMap(result),
            _kernel=kernel,
            results=dict())

    # Generate code for graph nodes.
    mapper = CodeGenMapper()
    for name, val in result.items():
        _ = mapper(val, state)

    return target.bind_program(program=state.kernel, bound_arguments=dict())
