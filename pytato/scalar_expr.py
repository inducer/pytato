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

from numbers import Number
from typing import Any, Union, Mapping, FrozenSet, Set, Optional, Tuple

from loopy.symbolic import Reduction
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
from pymbolic.mapper.collector import TermCollector as TermCollectorBase
import pymbolic.primitives as prim
import numpy as np
from dataclasses import dataclass, field
import math
from enum import Enum

__doc__ = """
.. currentmodule:: pytato.scalar_expr

Scalar Expressions
------------------

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

IntegralScalarExpression = Union[int, prim.Expression]
ScalarType = Union[Number, np.bool_, bool]
ScalarExpression = Union[ScalarType, prim.Expression]
SCALAR_CLASSES = prim.VALID_CONSTANT_CLASSES


def parse(s: str) -> ScalarExpression:
    from pymbolic.parser import Parser
    return Parser()(s)

# }}}


# {{{ mapper classes

class WalkMapper(WalkMapperBase):

    def map_reduction(self, expr: Reduction, *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr, *args, **kwargs):
            return

        self.rec(expr.expr, *args, **kwargs)


class IdentityMapper(IdentityMapperBase):

    def map_reduction(self, expr: Reduction,
            *args: Any, **kwargs: Any) -> Reduction:
        new_inames = []
        for iname in expr.inames:
            new_iname = self.rec(prim.Variable(iname), *args, **kwargs)
            if not isinstance(new_iname, prim.Variable):
                raise ValueError(
                        f"reduction iname {iname} can only be renamed"
                        " to another iname")
            new_inames.append(new_iname.name)

        return Reduction(expr.operation,
                tuple(new_inames),
                self.rec(expr.expr, *args, **kwargs),
                allow_simultaneous=expr.allow_simultaneous)


class SubstitutionMapper(SubstitutionMapperBase):

    def map_reduction(self, expr: Reduction) -> Reduction:
        new_inames = []
        for iname in expr.inames:
            new_iname = self.subst_func(iname)
            if new_iname is None:
                new_iname = prim.Variable(iname)
            else:
                if not isinstance(new_iname, prim.Variable):
                    raise ValueError(
                            f"reduction iname {iname} can only be renamed"
                            " to another iname")
            new_inames.append(new_iname.name)

        return Reduction(expr.operation,
                tuple(new_inames),
                self.rec(expr.expr),
                allow_simultaneous=expr.allow_simultaneous)


class DependencyMapper(DependencyMapperBase):

    def map_reduction(self, expr: Reduction,
            *args: Any, **kwargs: Any) -> Set[prim.Variable]:
        deps: Set[prim.Variable] = self.rec(expr.expr, *args, **kwargs)
        return deps - set(prim.Variable(iname) for iname in expr.inames)


class EvaluationMapper(EvaluationMapperBase):
    pass


class DistributeMapper(DistributeMapperBase):
    pass


class TermCollector(TermCollectorBase):
    pass

# }}}


# {{{ mapper frontends

def get_dependencies(expression: Any) -> FrozenSet[str]:
    """Return the set of variable names in an expression.

    :param expression: A scalar expression, or an expression derived from such
        (e.g., a tuple of scalar expressions)
    """
    mapper = DependencyMapper(composite_leaves=False)
    return frozenset(dep.name for dep in mapper(expression))


def substitute(expression: Any, variable_assigments: Mapping[str, Any]) -> Any:
    """Perform variable substitution in an expression.

    :param expression: A scalar expression, or an expression derived from such
        (e.g., a tuple of scalar expressions)
    :param variable_assigments: A mapping from variable names to substitutions
    """
    from pymbolic.mapper.substitutor import make_subst_func
    return SubstitutionMapper(make_subst_func(variable_assigments))(expression)


def evaluate(expression: Any, context: Mapping[str, Any] = {}) -> Any:
    """
    Evaluates *expression* by substituting the variable values as provided in
    *context*.
    """
    return EvaluationMapper(context)(expression)


def distribute(expr: Any, parameters: Set[Any] = set(),
               commutative: bool = True) -> Any:
    if commutative:
        return DistributeMapper(TermCollector(parameters))(expr)
    else:
        return DistributeMapper(lambda x: x)(expr)

# }}}


class ReductionOp(Enum):
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    PRODUCT = "product"

    @property
    def neutral_element(self):
        if self.value == "max":
            return -math.inf
        elif self.value == "min":
            return math.inf
        elif self.value == "sum":
            return 0
        elif self.value == "product":
            return 1
        else:
            raise NotImplementedError(f"Unknown reduction op {self}.")


@dataclass
class Reduce(prim.Expression):
    inner_expr: ScalarExpression
    op: ReductionOp
    bounds: Mapping[str, Tuple[ScalarExpression, ScalarExpression]]
    neutral_element: Optional[ScalarExpression] = None
    mapper_method: str = field(init=False, default="map_reduce")

    def __post_init__(self):
        self.neutral_element = self.neutral_element or self.op.neutral_element

    def __hash__(self):
        return hash((self.inner_expr,
                self.op,
                tuple(self.bounds.keys()),
                tuple(self.bounds.values()),
                self.neutral_element))

    def __str__(self):
        bounds_expr = " and ".join(f"{lb}<={key}<{ub}"
                for key, (lb, ub) in self.bounds.items())
        bounds_expr = "{" + bounds_expr + "}"
        return (f"{self.op.value}({bounds_expr}, {self.inner_expr},"
                f" {self.neutral_element})")


# vim: foldmethod=marker
