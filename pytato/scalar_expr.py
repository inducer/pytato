"""
.. class:: ScalarExpression

    Like ``ArithmeticExpressionT`` in :mod:`pymbolic`, but also allows
    Boolean values.

.. autofunction:: parse
.. autofunction:: get_dependencies
.. autofunction:: substitute

.. class:: Expression

    See :data:`pymbolic.typing.Expression`.
"""

# FIXME: Unclear why the direct links to pymbolic don't work

from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
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

import re
from collections.abc import Iterable, Mapping, Set
from typing import (
    TYPE_CHECKING,
    Any,
    Never,
    cast,
)

import numpy as np
from constantdict import constantdict
from typing_extensions import TypeIs

import pymbolic.primitives as prim
from pymbolic import ArithmeticExpression, Bool, Expression, expr_dataclass
from pymbolic.mapper import (
    CombineMapper as CombineMapperBase,
    IdentityMapper as IdentityMapperBase,
    P,
    ResultT,
    WalkMapper as WalkMapperBase,
)
from pymbolic.mapper.collector import TermCollector as TermCollectorBase
from pymbolic.mapper.dependency import (
    DependenciesT,
    DependencyMapper as DependencyMapperBase,
)
from pymbolic.mapper.distributor import DistributeMapper as DistributeMapperBase
from pymbolic.mapper.evaluator import EvaluationMapper as EvaluationMapperBase
from pymbolic.mapper.stringifier import StringifyMapper as StringifyMapperBase
from pymbolic.mapper.substitutor import SubstitutionMapper as SubstitutionMapperBase
from pymbolic.typing import Integer


if TYPE_CHECKING:
    from pytato.reductions import ReductionOperation


# {{{ scalar expressions

INT_CLASSES = (int, np.integer)
PYTHON_SCALAR_CLASSES = (int, float, complex, bool)
SCALAR_CLASSES = prim.VALID_CONSTANT_CLASSES

IntegralScalarExpression = Integer | prim.ExpressionNode
ScalarExpression = ArithmeticExpression | Bool


def is_integral_scalar_expression(expr: object) -> TypeIs[IntegralScalarExpression]:
    return isinstance(expr, int | np.integer | prim.ExpressionNode)


def parse(s: str) -> ScalarExpression:
    from pymbolic.parser import Parser
    res = Parser()(s)
    if not prim.is_arithmetic_expression(res):
        raise ValueError(f"'{s}' is not an arithmetic expression")

    return res

# }}}


# {{{ mapper classes

class WalkMapper(WalkMapperBase[[]]):
    def map_reduce(self, expr: Reduce) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.inner_expr)

        self.post_visit(expr)


class CombineMapper(CombineMapperBase[ResultT, P]):
    def map_type_cast(self,
                      expr: TypeCast, *args: P.args, **kwargs: P.kwargs) -> ResultT:
        return self.rec(expr.inner_expr, *args, **kwargs)

    def map_reduce(self, expr: Reduce, *args: P.args, **kwargs: P.kwargs) -> ResultT:
        return self.combine([*(self.rec(bnd, *args, **kwargs)
                               for _, bnd in sorted(expr.bounds.items())),
                             self.rec(expr.inner_expr, *args, **kwargs)])


class IdentityMapper(IdentityMapperBase[P]):
    def map_reduce(self,
                   expr: Reduce,
                   *args: P.args, **kwargs: P.kwargs) -> Expression:
        return Reduce(
                      cast("ArithmeticExpression",
                           self.rec(expr.inner_expr, *args, **kwargs)),
                      expr.op,
                      constantdict({
                                        name: (
                                            self.rec(lower, *args, **kwargs),
                                            self.rec(upper, *args, **kwargs)
                                        )
                                        for name, (lower, upper) in expr.bounds.items()
                                    }))

    def map_type_cast(self,
                expr: TypeCast, *args: P.args, **kwargs: P.kwargs) -> Expression:
        return TypeCast(expr.dtype,
                        cast("ArithmeticExpression",
                             self.rec(expr.inner_expr, *args, **kwargs)))


class SubstitutionMapper(SubstitutionMapperBase):
    def map_reduce(self, expr: Reduce) -> ScalarExpression:
        inner_expr = self.rec(expr.inner_expr)
        assert not isinstance(inner_expr, tuple)
        return Reduce(inner_expr,
                      op=expr.op,
                      bounds=constantdict(
                          {name: self.rec(bound)
                           for name, bound in expr.bounds.items()}))


IDX_LAMBDA_REDUCTION_AXIS_INDEX = re.compile(r"^(_r?(?P<index>0|[1-9][0-9]*))$")
IDX_LAMBDA_AXIS_INDEX = re.compile(r"^(_(?P<index>0|[1-9][0-9]*))$")


class DependencyMapper(DependencyMapperBase[P]):
    def __init__(self, *,
                 include_idx_lambda_indices: bool = True,
                 include_subscripts: bool = True,
                 include_lookups: bool = True,
                 include_calls: bool = True,
                 include_cses: bool = False,
                 composite_leaves: bool | None = None) -> None:
        super().__init__(include_subscripts=include_subscripts,
                         include_lookups=include_lookups,
                         include_calls=include_calls,
                         include_cses=include_cses,
                         composite_leaves=composite_leaves)
        self.include_idx_lambda_indices = include_idx_lambda_indices

    def map_variable(self,
                expr: prim.Variable, *args: P.args, **kwargs: P.kwargs
            ) -> DependenciesT:
        if ((not self.include_idx_lambda_indices)
                and IDX_LAMBDA_REDUCTION_AXIS_INDEX.fullmatch(str(expr))):
            return set()
        else:
            return super().map_variable(expr, *args, **kwargs)

    def map_reduce(self, expr: Reduce,
            *args: P.args, **kwargs: P.kwargs) -> DependenciesT:
        return self.combine([
            self.rec(expr.inner_expr, *args, **kwargs),
            set().union(*(self.rec((lb, ub), *args, **kwargs)
                        for (lb, ub) in expr.bounds.values()))])


class EvaluationMapper(EvaluationMapperBase[ResultT]):

    def map_reduce(self, expr: Reduce) -> Never:
        # TODO: not trivial to evaluate symbolic reduction nodes
        raise NotImplementedError()


class DistributeMapper(DistributeMapperBase):

    def map_reduce(self, expr: Reduce) -> None:
        # TODO: not trivial to distribute symbolic reduction nodes
        raise NotImplementedError()


class TermCollector(TermCollectorBase):

    def map_reduce(self, expr: Reduce) -> None:
        raise NotImplementedError()


class StringifyMapper(StringifyMapperBase[P]):
    def map_reduce(self,
                   expr: Any, enclosing_prec: int, *args: P.args, **kwargs: P.kwargs
               ) -> str:
        from pymbolic.mapper.stringifier import PREC_COMPARISON, PREC_NONE
        bounds_expr = " and ".join(
                f"{self.rec(lb, PREC_COMPARISON, *args, **kwargs)}"
                f"<={name}<{self.rec(ub, PREC_COMPARISON, *args, **kwargs)}"
                for name, (lb, ub) in expr.bounds.items())
        bounds_expr = "{" + bounds_expr + "}"
        return (f"{expr.op}({bounds_expr}, "
                f"{self.rec(expr.inner_expr, PREC_NONE, *args, **kwargs)})")

    def map_type_cast(self,
                      expr: TypeCast, enclosing_prec: int,
                      *args: P.args, **kwargs: P.kwargs) -> str:
        from pymbolic.mapper.stringifier import PREC_NONE
        inner_str = self.rec(expr.inner_expr, PREC_NONE, *args, **kwargs)
        return f"cast({expr.dtype}, {inner_str})"

# }}}


# {{{ mapper frontends

def get_dependencies(expression: Expression,
        include_idx_lambda_indices: bool = True) -> frozenset[str]:
    """Return the set of variable names in an expression.

    :param expression: A scalar expression, or an expression derived from such
        (e.g., a tuple of scalar expressions)
    """
    mapper = DependencyMapper[[]](
            composite_leaves=False,
            include_idx_lambda_indices=include_idx_lambda_indices)
    return frozenset(dep.name
                     for dep in mapper(expression)
                     if isinstance(dep, prim.Variable))


def substitute(expression: Expression,
        variable_assignments: Mapping[str, Any] | None) -> Expression:
    """Perform variable substitution in an expression.

    :param expression: A scalar expression, or an expression derived from such
        (e.g., a tuple of scalar expressions)
    :param variable_assignments: A mapping from variable names to substitutions
    """
    if variable_assignments is None:
        variable_assignments = {}

    from pymbolic.mapper.substitutor import make_subst_func
    return SubstitutionMapper(make_subst_func(variable_assignments))(expression)


def evaluate(
            expression: Expression, context: Mapping[str, ResultT] | None = None
        ) -> ResultT:
    """
    Evaluates *expression* by substituting the variable values as provided in
    *context*.
    """
    if context is None:
        context = {}
    return EvaluationMapper(context)(expression)


def distribute(expr: Expression, parameters: frozenset[Any] = frozenset(),
               commutative: bool = True) -> Expression:
    if commutative:
        return DistributeMapper(TermCollector(parameters))(expr)
    else:
        return DistributeMapper(lambda x: x)(expr)

# }}}


# {{{ custom scalar expression nodes

class ExpressionBase(prim.ExpressionNode):
    def make_stringifier(self,
                 originating_stringifier: StringifyMapperBase[[]] | None = None
             ) -> StringifyMapperBase[[]]:
        return StringifyMapper()


@expr_dataclass()
class Reduce(ExpressionBase):
    """
    .. autoattribute:: inner_expr
    .. autoattribute:: op
    .. autoattribute:: bounds
    """

    inner_expr: ScalarExpression
    """The expression to be reduced over."""

    op: ReductionOperation

    bounds: Mapping[str, tuple[ArithmeticExpression, ArithmeticExpression]]
    """
    A mapping from reduction inames to tuples ``(lower_bound, upper_bound)``
    identifying half-open bounds intervals.  Must be hashable.
    """

    if __debug__:
        def __post_init__(self) -> None:
            hash(self.bounds)


@expr_dataclass()
class TypeCast(ExpressionBase):
    """
    .. autoattribute:: dtype
    .. autoattribute:: dtype
    """
    dtype: np.dtype[Any]
    inner_expr: ScalarExpression

# }}}


class InductionVariableCollector(CombineMapper[Set[str], []]):
    def combine(self, values: Iterable[Set[str]]) -> frozenset[str]:
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    def map_reduce(self, expr: Reduce) -> Set[str]:
        return self.combine([frozenset(expr.bounds.keys()),
                             super().map_reduce(expr)])

    def map_algebraic_leaf(self, expr: prim.ExpressionNode) -> frozenset[str]:
        return frozenset()

    def map_constant(self, expr: object) -> Set[str]:
        return frozenset()


def get_reduction_induction_variables(expr: Expression) -> Set[str]:
    """
    Returns the induction variables for the reduction nodes.
    """
    return InductionVariableCollector()(expr)

# vim: foldmethod=marker
