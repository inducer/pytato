import pytato as pt
import numpy as np
from pytato.mlir import generate_mlir
from numpy.random import default_rng
from pytato.scalar_expr import parse


def test_static_shapes():
    ns = pt.Namespace()
    rng = default_rng()

    x_in = rng.random(10)
    y_in = rng.random(10)

    x = pt.make_data_wrapper(ns, x_in)
    y = pt.make_data_wrapper(ns, y_in)
    z = x + 2*y
    return_dict = generate_mlir({"z": z})()

    np.testing.assert_allclose(return_dict["z"], x_in+2*y_in)


def test_parametric_shapes():
    rng = default_rng()
    N = rng.integers(50, 100)
    x_in = rng.random(N)

    ns = pt.Namespace()
    pt.make_size_param(ns, "n")
    x = pt.make_placeholder(ns, name="x", shape=("n",), dtype=float)

    y = 2*x

    return_dict = generate_mlir({"y": y})(x=x_in)
    np.testing.assert_allclose(return_dict["y"], 2*x_in)


def test_parameters_in_exprs():
    rng = default_rng()
    M = rng.integers(10, 20)
    N = rng.integers(10, 20)
    x_in = rng.random((M, M+N))

    ns = pt.Namespace()
    m = pt.make_size_param(ns, "m")
    n = pt.make_size_param(ns, "n")
    x = pt.make_placeholder(ns, name="x", shape=("m", "m+n"), dtype=float)

    y = pt.IndexLambda(ns, parse("in1*in2*in3[_0, _1]"), bindings={"in1": n,
                                                                "in2": m,
                                                                "in3": x},
                       shape=parse("(m,m+n)"), dtype=np.dtype(float))
    return_dict = generate_mlir({"y": y})(x=x_in)

    np.testing.assert_allclose(return_dict["y"], M*N*x_in)
