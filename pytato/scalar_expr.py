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
from typing import Any, Union, Mapping, FrozenSet, Set

from loopy.symbolic import Reduction
from pymbolic.mapper import (WalkMapper as WalkMapperBase, IdentityMapper as
        IdentityMapperBase)
from pymbolic.mapper.substitutor import (make_subst_func,
        SubstitutionMapper as SubstitutionMapperBase)
from pymbolic.mapper.dependency import (DependencyMapper as
        DependencyMapperBase)
import pymbolic.primitives as prim

__doc__ = """
.. currentmodule:: pytato.scalar_expr

Scalar Expressions
------------------

.. data:: ScalarExpression

    A :class:`type` for scalar-valued symbolic expressions. Expressions are
    composable and manipulable via :mod:`pymbolic`.

    Concretely, this is an alias for
    ``Union[Number, pymbolic.primitives.Expression]``.

.. autofunction:: parse
.. autofunction:: get_dependencies
.. autofunction:: substitute

"""

# {{{ scalar expressions

IntegralScalarExpression = Union[int, prim.Expression]
ScalarExpression = Union[Number, prim.Expression]


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

# }}}


# {{{ IndexLambdaSubstitutor

class IndexLambdaSubstitutor(IdentityMapper):
    """
    Substitutes the usage of an :class:`pytato.array.IndexLambda` according
    to the provided *bindings*.

    ::
        >>> bindings = {"in": scalar_expr.parse("x[_1 + 7, _0] + y[_0, _1]")}
        >>> input_expr = scalar_expr.parse("in[_1 + 3, _0] + 17")
        >>> print(IndexLambdaSubstitutor(bindings)(input_expr))
        >>> x[_0 + 7, _1 + 3] + y[_1 + 3, _0] + 17
    """
    def __init__(self, bindings: Mapping[str, ScalarExpression]) -> None:
        self.bindings = bindings

    def map_subscript(self, expr: prim.Subscript) -> prim.Subscript:
        idx_map = {f"_{i}": idx
                   for i, idx in enumerate(expr.index_tuple)}
        subst_mapper = SubstitutionMapper(make_subst_func(idx_map))
        return subst_mapper(self.bindings[expr.aggregate.name])

    def map_variable(self, expr: prim.Variable) -> prim.Variable:
        try:
            return self.bindings[expr.name]
        except KeyError:
            return super().map_variable(expr)

# }}}

# vim: foldmethod=marker
