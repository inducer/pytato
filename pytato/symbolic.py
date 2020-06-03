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
from typing import Any, Union, Mapping, FrozenSet, Set, Tuple

import islpy as isl
from pymbolic.mapper import (WalkMapper as WalkMapperBase, IdentityMapper as
        IdentityMapperBase)
from pymbolic.mapper.substitutor import (SubstitutionMapper as
        SubstitutionMapperBase)
from pymbolic.mapper.dependency import (DependencyMapper as
        DependencyMapperBase)
import pymbolic.primitives as prim

__doc__ = """
.. currentmodule:: pytato.symbolic

Symbolic Infrastructure
-----------------------

.. data:: ScalarExpression

    A type alias for ``Union[Number, pymbolic.primitives.Expression]``.

.. autofunction:: parse
.. autofunction:: get_dependencies
.. autofunction:: substitute
.. autofunction:: domain_for_shape

"""

# {{{ scalar expressions

ScalarExpression = Union[Number, prim.Expression]


def parse(s: str) -> ScalarExpression:
    from pymbolic.parser import Parser
    return Parser()(s)


# }}}

# {{{ mapper classes


class WalkMapper(WalkMapperBase):
    pass


class IdentityMapper(IdentityMapperBase):
    pass


class SubstitutionMapper(SubstitutionMapperBase):
    pass


class DependencyMapper(DependencyMapperBase):
    pass


# }}}

# {{{ mapper frontends


def get_dependencies(expression: Any) -> FrozenSet[str]:
    """Return the set of variable names in an expression.

    :param expression: A :mod:`pymbolic` expression
    """
    mapper = DependencyMapper(composite_leaves=False)
    return frozenset(dep.name for dep in mapper(expression))


def substitute(expression: Any, variable_assigments: Mapping[str, Any]) -> Any:
    """Perform variable substitution in an expression.

    :param expression: A :mod:`pymbolic` expression
    :param variable_assigments: A mapping from variable names to substitutions
    """
    from pymbolic.mapper.substitutor import make_subst_func
    return SubstitutionMapper(make_subst_func(variable_assigments))(expression)


# }}}


def domain_for_shape(dim_names: Tuple[str, ...], shape: Tuple[ScalarExpression,
        ...]) -> isl.BasicSet:
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
    for sdep in map(get_dependencies, shape):
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


# vim: foldmethod=marker
