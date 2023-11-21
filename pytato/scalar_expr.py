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

from typing import (
        Any, Union, Mapping, FrozenSet, Set, Tuple, Optional, TYPE_CHECKING,
        Iterable)

from pymbolic.mapper import (WalkMapper as WalkMapperBase, IdentityMapper as
        IdentityMapperBase)
from pymbolic.mapper.substitutor import (SubstitutionMapper as
        SubstitutionMapperBase)
from pymbolic.mapper.dependency import (DependencyMapper as
        DependencyMapperBase)
from pymbolic.mapper.evaluator import (EvaluationMapper as
        EvaluationMapperBase)
from pymbolic.mapper.distributor import (DistributeMapper as
        DistributeMapperBase)
from pymbolic.mapper.stringifier import (StringifyMapper as
        StringifyMapperBase)
from pymbolic.mapper import CombineMapper as CombineMapperBase
from pymbolic.mapper.collector import TermCollector as TermCollectorBase
from immutabledict import immutabledict
import pymbolic.primitives as prim
import numpy as np
import re

if TYPE_CHECKING:
    from pytato.reductions import ReductionOperation


__doc__ = """
.. currentmodule:: pytato.scalar_expr

.. data:: ScalarExpression

    A :class:`type` for scalar-valued symbolic expressions. Expressions are
    composable and manipulable via :mod:`pymbolic`.

    Concretely, this is an alias for
    ``Union[Number, np.bool_, bool, pymbolic.primitives.Expression]``.

.. autofunction:: parse
.. autofunction:: get_dependencies
.. autofunction:: substitute

"""

# {{{ scalar expressions

IntegralT = Union[int, np.integer[Any]]
BoolT = Union[bool, np.bool_]
INT_CLASSES = (int, np.integer)
IntegralScalarExpression = Union[IntegralT, prim.Expression]
ScalarType = Union[np.number[Any], int, np.bool_, bool, float]

ScalarExpression = Union[ScalarType, prim.Expression]
SCALAR_CLASSES = prim.VALID_CONSTANT_CLASSES


def parse(s: str) -> ScalarExpression:
    from pymbolic.parser import Parser
    return Parser()(s)

# }}}


# {{{ mapper classes

class WalkMapper(WalkMapperBase):
    def map_reduce(self, expr: Reduce) -> None:
        if not self.visit(expr):
            return

        self.rec(expr.inner_expr)

        self.post_visit(expr)


class CombineMapper(CombineMapperBase):
    def map_reduce(self, expr: Reduce, *args: Any, **kwargs: Any) -> Any:
        return self.combine([*(self.rec(bnd, *args, **kwargs)
                               for _, bnd in sorted(expr.bounds.items())),
                             self.rec(expr.inner_expr, *args, **kwargs)])


class IdentityMapper(IdentityMapperBase):
    pass


class SubstitutionMapper(SubstitutionMapperBase):
    def map_reduce(self, expr: Reduce) -> ScalarExpression:
        return Reduce(self.rec(expr.inner_expr),
                      op=expr.op,
                      bounds=immutabledict(
                          {name: self.rec(bound)
                           for name, bound in expr.bounds.items()}))


IDX_LAMBDA_RE = re.compile("_r?(0|([1-9][0-9]*))")


class DependencyMapper(DependencyMapperBase):
    def __init__(self, *,
                 include_idx_lambda_indices: bool = True,
                 include_subscripts: bool = True,
                 include_lookups: bool = True,
                 include_calls: bool = True,
                 include_cses: bool = False,
                 composite_leaves: Optional[bool] = None) -> None:
        super().__init__(include_subscripts=include_subscripts,
                         include_lookups=include_lookups,
                         include_calls=include_calls,
                         include_cses=include_cses,
                         composite_leaves=composite_leaves)
        self.include_idx_lambda_indices = include_idx_lambda_indices

    def map_variable(self, expr: prim.Variable) -> Set[prim.Variable]:
        if ((not self.include_idx_lambda_indices)
                and IDX_LAMBDA_RE.fullmatch(str(expr))):
            return set()
        else:
            return super().map_variable(expr)  # type: ignore

    def map_reduce(self, expr: Reduce,
            *args: Any, **kwargs: Any) -> Set[prim.Variable]:
        return self.combine([  # type: ignore
            self.rec(expr.inner_expr),
            set().union(*(self.rec((lb, ub)) for (lb, ub) in expr.bounds.values()))])


class EvaluationMapper(EvaluationMapperBase):

    def map_reduce(self, expr: Reduce, *args: Any, **kwargs: Any) -> None:
        # TODO: not trivial to evaluate symbolic reduction nodes
        raise NotImplementedError()


class DistributeMapper(DistributeMapperBase):

    def map_reduce(self, expr: Reduce, *args: Any, **kwargs: Any) -> None:
        # TODO: not trivial to distribute symbolic reduction nodes
        raise NotImplementedError()


class TermCollector(TermCollectorBase):

    def map_reduce(self, expr: Reduce, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()


class StringifyMapper(StringifyMapperBase):
    def map_reduce(self, expr: Any, enclosing_prec: Any, *args: Any) -> str:
        from pymbolic.mapper.stringifier import (
                PREC_COMPARISON as PC,
                PREC_NONE as PN)
        bounds_expr = " and ".join(
                f"{self.rec(lb, PC)}<={name}<{self.rec(ub, PC)}"
                for name, (lb, ub) in expr.bounds.items())
        bounds_expr = "{" + bounds_expr + "}"
        return (f"{expr.op}({bounds_expr}, {self.rec(expr.inner_expr, PN)})")

# }}}


# {{{ mapper frontends

def get_dependencies(expression: Any,
        include_idx_lambda_indices: bool = True) -> FrozenSet[str]:
    """Return the set of variable names in an expression.

    :param expression: A scalar expression, or an expression derived from such
        (e.g., a tuple of scalar expressions)
    """
    mapper = DependencyMapper(
            composite_leaves=False,
            include_idx_lambda_indices=include_idx_lambda_indices)
    return frozenset(dep.name for dep in mapper(expression))


def substitute(expression: Any,
        variable_assigments: Optional[Mapping[str, Any]]) -> Any:
    """Perform variable substitution in an expression.

    :param expression: A scalar expression, or an expression derived from such
        (e.g., a tuple of scalar expressions)
    :param variable_assigments: A mapping from variable names to substitutions
    """
    if variable_assigments is None:
        variable_assigments = {}

    from pymbolic.mapper.substitutor import make_subst_func
    return SubstitutionMapper(make_subst_func(variable_assigments))(expression)


def evaluate(expression: Any, context: Optional[Mapping[str, Any]] = None) -> Any:
    """
    Evaluates *expression* by substituting the variable values as provided in
    *context*.
    """
    if context is None:
        context = {}
    return EvaluationMapper(context)(expression)


def distribute(expr: Any, parameters: FrozenSet[Any] = frozenset(),
               commutative: bool = True) -> Any:
    if commutative:
        return DistributeMapper(TermCollector(parameters))(expr)
    else:
        return DistributeMapper(lambda x: x)(expr)

# }}}


# {{{ custom scalar expression nodes

class ExpressionBase(prim.Expression):
    def make_stringifier(self, originating_stringifier: Any = None) -> str:
        return StringifyMapper()


class Reduce(ExpressionBase):
    """
    .. attribute:: inner_expr

        A :class:`ScalarExpression` to be reduced over.

    .. attribute:: op

        A :class:`pytato.reductions.ReductionOperation`.

    .. attribute:: bounds

        A mapping from reduction inames to tuples ``(lower_bound, upper_bound)``
        identifying half-open bounds intervals.  Must be hashable.
    """
    inner_expr: ScalarExpression
    op: ReductionOperation
    bounds: Mapping[str, Tuple[ScalarExpression, ScalarExpression]]

    def __init__(self, inner_expr: ScalarExpression,
            op: ReductionOperation, bounds: Any) -> None:
        self.inner_expr = inner_expr
        self.op = op
        self.bounds = bounds

    def __hash__(self) -> int:
        return hash((self.inner_expr,
                self.op,
                tuple(self.bounds.keys()),
                tuple(self.bounds.values())))

    def __getinitargs__(self) -> Tuple[ScalarExpression, ReductionOperation, Any]:
        return (self.inner_expr, self.op, self.bounds)

    init_arg_names = ("inner_expr", "op", "bounds")
    mapper_method = "map_reduce"

# }}}


class InductionVariableCollector(CombineMapper):
    def combine(self, values: Iterable[FrozenSet[str]]) -> FrozenSet[str]:
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    def map_reduce(self, expr: Reduce) -> FrozenSet[str]:
        return self.combine([frozenset(expr.bounds.keys()),
                             super().map_reduce(expr)])

    def map_algebraic_leaf(self, expr: prim.Expression) -> FrozenSet[str]:
        return frozenset()

    def map_constant(self, expr: Any) -> FrozenSet[str]:
        return frozenset()


def get_reduction_induction_variables(expr: prim.Expression) -> FrozenSet[str]:
    """
    Returns the induction variables for the reduction nodes.
    """
    return InductionVariableCollector()(expr)  # type: ignore[no-any-return]

# vim: foldmethod=marker
