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
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Union,
)

import attrs
import numpy as np
from immutabledict import immutabledict

import pymbolic.primitives as prim
from pymbolic.mapper import (
    CombineMapper as CombineMapperBase,
    IdentityMapper as IdentityMapperBase,
    WalkMapper as WalkMapperBase,
)
from pymbolic.mapper.collector import TermCollector as TermCollectorBase
from pymbolic.mapper.dependency import DependencyMapper as DependencyMapperBase
from pymbolic.mapper.distributor import DistributeMapper as DistributeMapperBase
from pymbolic.mapper.evaluator import EvaluationMapper as EvaluationMapperBase
from pymbolic.mapper.stringifier import StringifyMapper as StringifyMapperBase
from pymbolic.mapper.substitutor import SubstitutionMapper as SubstitutionMapperBase


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
Scalar = Union[np.number[Any], int, np.bool_, bool, float, complex]

ScalarExpression = Union[Scalar, prim.Expression]
PYTHON_SCALAR_CLASSES = (int, float, complex, bool)
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
    def map_type_cast(self, expr: TypeCast, *args: Any, **kwargs: Any) -> Any:
        return self.rec(expr.inner_expr)

    def map_reduce(self, expr: Reduce, *args: Any, **kwargs: Any) -> Any:
        return self.combine([*(self.rec(bnd, *args, **kwargs)
                               for _, bnd in sorted(expr.bounds.items())),
                             self.rec(expr.inner_expr, *args, **kwargs)])


class IdentityMapper(IdentityMapperBase):
    def map_reduce(self, expr: Reduce, *args: Any, **kwargs: Any) -> Any:
        return Reduce(
                      self.rec(expr.inner_expr, *args, **kwargs),
                      expr.op,
                      immutabledict({
                                        name: (
                                            self.rec(lower, *args, **kwargs),
                                            self.rec(upper, *args, **kwargs)
                                        )
                                        for name, (lower, upper) in expr.bounds.items()
                                    }))

    def map_type_cast(self, expr: TypeCast, *args: Any, **kwargs: Any) -> Any:
        return TypeCast(expr.dtype, self.rec(expr.inner_expr, *args, **kwargs))


class SubstitutionMapper(SubstitutionMapperBase):
    def map_reduce(self, expr: Reduce) -> ScalarExpression:
        return Reduce(self.rec(expr.inner_expr),
                      op=expr.op,
                      bounds=immutabledict(
                          {name: self.rec(bound)
                           for name, bound in expr.bounds.items()}))


IDX_LAMBDA_RE = re.compile("_r?(0|([1-9][0-9]*))")
IDX_LAMBDA_INAME = re.compile("^(_(0|([1-9][0-9]*)))$")
IDX_LAMBDA_JUST_REDUCTIONS = re.compile("^(_r(0|([1-9][0-9]*)))$")

class DependencyMapper(DependencyMapperBase):
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

    def map_variable(self, expr: prim.Variable) -> set[prim.Variable]:
        if ((not self.include_idx_lambda_indices)
                and IDX_LAMBDA_RE.fullmatch(str(expr))):
            return set()
        else:
            return super().map_variable(expr)  # type: ignore

    def map_reduce(self, expr: Reduce,
            *args: Any, **kwargs: Any) -> set[prim.Variable]:
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
        from pymbolic.mapper.stringifier import PREC_COMPARISON, PREC_NONE
        bounds_expr = " and ".join(
                f"{self.rec(lb, PREC_COMPARISON)}"
                f"<={name}<{self.rec(ub, PREC_COMPARISON)}"
                for name, (lb, ub) in expr.bounds.items())
        bounds_expr = "{" + bounds_expr + "}"
        return (f"{expr.op}({bounds_expr}, {self.rec(expr.inner_expr, PREC_NONE)})")

    def map_type_cast(self, expr: TypeCast, enclosing_prec: Any) -> str:
        from pymbolic.mapper.stringifier import PREC_NONE
        return f"cast({expr.dtype}, {self.rec(expr.inner_expr, PREC_NONE)})"

# }}}


# {{{ mapper frontends

def get_dependencies(expression: Any,
        include_idx_lambda_indices: bool = True) -> frozenset[str]:
    """Return the set of variable names in an expression.

    :param expression: A scalar expression, or an expression derived from such
        (e.g., a tuple of scalar expressions)
    """
    mapper = DependencyMapper(
            composite_leaves=False,
            include_idx_lambda_indices=include_idx_lambda_indices)
    return frozenset(dep.name for dep in mapper(expression))


def substitute(expression: Any,
        variable_assignments: Mapping[str, Any] | None) -> Any:
    """Perform variable substitution in an expression.

    :param expression: A scalar expression, or an expression derived from such
        (e.g., a tuple of scalar expressions)
    :param variable_assignments: A mapping from variable names to substitutions
    """
    if variable_assignments is None:
        variable_assignments = {}

    from pymbolic.mapper.substitutor import make_subst_func
    return SubstitutionMapper(make_subst_func(variable_assignments))(expression)


def evaluate(expression: Any, context: Mapping[str, Any] | None = None) -> Any:
    """
    Evaluates *expression* by substituting the variable values as provided in
    *context*.
    """
    if context is None:
        context = {}
    return EvaluationMapper(context)(expression)


def distribute(expr: Any, parameters: frozenset[Any] = frozenset(),
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


@attrs.frozen(eq=True, hash=True, cache_hash=True)
class Reduce(ExpressionBase):
    """
    .. autoattribute:: inner_expr
    .. autoattribute:: op
    .. autoattribute:: bounds
    """

    inner_expr: ScalarExpression
    """The expression to be reduced over."""

    op: ReductionOperation

    bounds: Mapping[str, tuple[ScalarExpression, ScalarExpression]]
    """
    A mapping from reduction inames to tuples ``(lower_bound, upper_bound)``
    identifying half-open bounds intervals.  Must be hashable.
    """

    def update_persistent_hash(self, key_hash: Any, key_builder: Any) -> None:
        key_builder.rec(key_hash, self.inner_expr)
        key_builder.rec(key_hash, self.op)
        key_builder.rec(key_hash, tuple(self.bounds.keys()))
        key_builder.rec(key_hash, tuple(self.bounds.values()))

    def __getinitargs__(self) -> tuple[ScalarExpression, ReductionOperation, Any]:
        return (self.inner_expr, self.op, self.bounds)

    if __debug__:
        def __attrs_post_init__(self) -> None:
            hash(self.bounds)

    init_arg_names = ("inner_expr", "op", "bounds")
    mapper_method = "map_reduce"


@attrs.frozen(eq=True, hash=True, cache_hash=True)
class TypeCast(ExpressionBase):
    """
    .. autoattribute:: dtype
    .. autoattribute:: dtype
    """
    dtype: np.dtype[Any]
    inner_expr: ScalarExpression

    def __getinitargs__(self) -> tuple[np.dtype[Any], ScalarExpression]:
        return (self.dtype, self.inner_expr)

    init_arg_names = ("dtype", "inner_expr")
    mapper_method = "map_type_cast"

# }}}


class InductionVariableCollector(CombineMapper):
    def combine(self, values: Iterable[frozenset[str]]) -> frozenset[str]:
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    def map_reduce(self, expr: Reduce) -> frozenset[str]:
        return self.combine([frozenset(expr.bounds.keys()),
                             super().map_reduce(expr)])

    def map_algebraic_leaf(self, expr: prim.Expression) -> frozenset[str]:
        return frozenset()

    def map_constant(self, expr: Any) -> frozenset[str]:
        return frozenset()


def get_reduction_induction_variables(expr: prim.Expression) -> frozenset[str]:
    """
    Returns the induction variables for the reduction nodes.
    """
    return InductionVariableCollector()(expr)  # type: ignore[no-any-return]


def contains_reduction(expr: prim.Expression) -> bool:
    """
    Returns true if any operation in the scalar expression, expr, is a reduction
    operation.
    """
    return bool(get_reduction_induction_variables(expr))
# vim: foldmethod=marker
