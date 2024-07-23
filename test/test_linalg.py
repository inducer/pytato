#!/usr/bin/env python

__copyright__ = """
Copyright (C) 2023 Kaushik Kulkarni
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

import sys

import numpy as np

from pyopencl.tools import (  # noqa: F401
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

import pytato as pt


def test_apply_einsum_distributive_law_0():
    from pytato.transform.einsum_distributive_law import (
        DoDistribute,
        DoNotDistribute,
        EinsumDistributiveLawDescriptor,
        apply_distributive_property_to_einsums,
    )

    def how_to_distribute(
            expr: pt.Einsum) -> EinsumDistributiveLawDescriptor:
        if pt.analysis.is_einsum_similar_to_subscript(
                expr, "ij,j->i"):
            return DoDistribute(ioperand=1)
        else:
            return DoNotDistribute()

    x1 = pt.make_placeholder("x1", 4, np.float64)
    x2 = pt.make_placeholder("x2", 4, np.float64)
    A1 = pt.make_placeholder("A1", (10, 4), np.float64)
    A2 = pt.make_placeholder("A2", (10, 4), np.float64)
    y = (7*A1 + 8*A2) @ (2*x1-3*x2)
    y_transformed = apply_distributive_property_to_einsums(y, how_to_distribute)

    assert y_transformed == ((2 * ((7*A1 + 8*A2) @ x1) - 3 * ((7*A1 + 8*A2) @
                                                              x2)))


def test_apply_einsum_distributive_law_1():
    from pytato.transform.einsum_distributive_law import (
        DoDistribute,
        DoNotDistribute,
        EinsumDistributiveLawDescriptor,
        apply_distributive_property_to_einsums,
    )

    def how_to_distribute(
            expr: pt.Einsum) -> EinsumDistributiveLawDescriptor:
        if pt.analysis.is_einsum_similar_to_subscript(
                expr, "ij,j->i"):
            return DoDistribute(ioperand=0)
        else:
            return DoNotDistribute()

    x1 = pt.make_placeholder("x1", 4, np.float64)
    x2 = pt.make_placeholder("x2", 4, np.float64)
    A1 = pt.make_placeholder("A1", (10, 4), np.float64)
    A2 = pt.make_placeholder("A2", (10, 4), np.float64)
    y = (7*A1 + 8*pt.sin(A2)) @ (2*x1-3*x2)
    y_transformed = apply_distributive_property_to_einsums(y, how_to_distribute)
    print(y_transformed)
    assert y_transformed == (7 * (A1 @ (2*x1-3*x2)) + 8 * (pt.sin(A2) @ (2*x1-3*x2)))


def test_apply_einsum_distributive_law_2():
    from pytato.transform.einsum_distributive_law import (
        DoDistribute,
        DoNotDistribute,
        EinsumDistributiveLawDescriptor,
        apply_distributive_property_to_einsums,
    )

    def how_to_distribute(
            expr: pt.Einsum) -> EinsumDistributiveLawDescriptor:
        if (pt.analysis.is_einsum_similar_to_subscript(
                expr, "ij,j->i") and
                pt.utils.are_shape_components_equal(expr.args[1].shape[0],
                                                    10)):
            return DoDistribute(ioperand=1)
        else:
            return DoNotDistribute()

    x1 = pt.make_placeholder("x1", 4, np.float64)
    x2 = pt.make_placeholder("x2", 4, np.float64)
    A1 = pt.make_placeholder("A1", (10, 10), np.float64)
    A2 = pt.make_placeholder("A2", (10, 10), np.float64)
    B = pt.make_placeholder("B", (10, 4), np.float64)
    y = (7*A1 + 8*A2) @ (2*(B@x1)-3*(B@x2))
    y_transformed = apply_distributive_property_to_einsums(y, how_to_distribute)

    assert y_transformed == (2 * ((7*A1 + 8*A2) @ (B@x1))
                             - 3 * ((7*A1 + 8*A2) @ (B@x2)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
