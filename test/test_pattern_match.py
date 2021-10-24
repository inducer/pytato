import sys

import numpy as np
import pyopencl.array as cl_array  # noqa
import pyopencl.cltypes as cltypes  # noqa
import pyopencl.tools as cl_tools  # noqa
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import pytato as pt
from pytato.match import Wildcard


def test_match_linear_combo():
    a = pt.make_placeholder("a", shape=(10, 10), dtype=np.float64)
    b = pt.make_placeholder("b", shape=(10, 10), dtype=np.float64)
    c = pt.make_placeholder("c", shape=(10, 10), dtype=np.float64)
    x1 = pt.make_placeholder("x1", shape=(10,), dtype=np.float64)
    x2 = pt.make_placeholder("x2", shape=(10,), dtype=np.float64)

    expr = 3 * (a @ (2 * b @ x2 + 3 * c @ x1)) + 4

    w1_ = Wildcard.dot("w1_")
    w2_ = Wildcard.dot("w2_")
    w3_ = Wildcard.dot("w3_")
    w4_ = Wildcard.dot("w4_")
    w5_ = Wildcard.dot("w5_")

    pattern = w1_ @ (w2_ @ w3_ + w4_ @ w5_)

    matches = pt.match.match_anywhere(expr, pattern)
    (subst, match), = list(matches)

    assert subst["w1_"] == a
    assert subst["w2_"] == 2 * b
    assert subst["w3_"] == x2
    assert subst["w4_"] == 3 * c
    assert subst["w5_"] == x1
    assert match == (a @ (2 * b @ x2 + 3 * c @ x1))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
