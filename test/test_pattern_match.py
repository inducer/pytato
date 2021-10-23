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
    A = pt.make_placeholder("A", shape=(10, 10), dtype=np.float64)
    B = pt.make_placeholder("B", shape=(10, 10), dtype=np.float64)
    C = pt.make_placeholder("C", shape=(10, 10), dtype=np.float64)
    x1 = pt.make_placeholder("x1", shape=(10,), dtype=np.float64)
    x2 = pt.make_placeholder("x2", shape=(10,), dtype=np.float64)

    expr = 3 * (A @ (2 * B @ x2 + 3 * C @ x1)) + 4

    w1_ = Wildcard.dot("w1_")
    w2_ = Wildcard.dot("w2_")
    w3_ = Wildcard.dot("w3_")
    w4_ = Wildcard.dot("w4_")
    w5_ = Wildcard.dot("w5_")

    pattern = w1_ @ (w2_ @ w3_ + w4_ @ w5_)

    matches = pt.match.match_anywhere(expr, pattern)
    (match, _, _), = list(matches)

    assert match["w1_"] == A
    assert match["w2_"] == 2 * B
    assert match["w3_"] == x2
    assert match["w4_"] == 3 * C
    assert match["w5_"] == x1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
