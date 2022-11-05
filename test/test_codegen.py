#!/usr/bin/env python

__copyright__ = """Copyright (C) 2020-2021 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
Copyright (C) 2021 Matthias Diener
Copyright (C) 2021 Kaushik Kulkarni
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

from typing import Union

import itertools
import operator
import sys

import loopy as lp
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array  # noqa
import pyopencl.cltypes as cltypes  # noqa
import pyopencl.tools as cl_tools  # noqa
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
import pytest  # noqa
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa

import pytato as pt
from testlib import assert_allclose_to_numpy, get_random_pt_dag
import pymbolic.primitives as p


def test_basic_codegen(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x = pt.make_placeholder("x", (5,), np.int64)
    prog = pt.generate_loopy(x * x)
    x_in = np.array([1, 2, 3, 4, 5])
    _, (out,) = prog(queue, x=x_in)
    assert (out == x_in * x_in).all()


def test_named_clash(ctx_factory):
    x = pt.make_placeholder("x", (5,), np.int64)

    from pytato.tags import ImplStored, Named
    expr = (
            (2*x).tagged((Named("xx"), ImplStored()))
            + (3*x).tagged((Named("xx"), ImplStored())))

    with pytest.raises(ValueError):
        pt.generate_loopy(expr)


def test_scalar_placeholder(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x = pt.make_placeholder("x", (), np.int64)
    prog = pt.generate_loopy(x)
    x_in = np.array(1)
    _, (x_out,) = prog(queue, x=x_in)
    assert np.array_equal(x_out, x_in)


# https://github.com/inducer/pytato/issues/15
@pytest.mark.xfail  # shape inference solver: not yet implemented
def test_size_param(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = pt.make_size_param(name="n")
    pt.make_placeholder(name="x", shape=n, dtype=np.int64)
    prog = pt.generate_loopy(n)
    x_in = np.array([1, 2, 3, 4, 5])
    _, (n_out,) = prog(queue, x=x_in)
    assert n_out == 5


@pytest.mark.parametrize("x1_ndim", (1, 2))
@pytest.mark.parametrize("x2_ndim", (1, 2))
def test_matmul_basic(ctx_factory, x1_ndim, x2_ndim):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    def get_array(ndim):
        arr = np.array([[1, 2], [3, 4]])
        return arr[(0,) * (arr.ndim - ndim)]

    x1_in = get_array(x1_ndim)
    x2_in = get_array(x2_ndim)

    x1 = pt.make_data_wrapper(x1_in)
    x2 = pt.make_data_wrapper(x2_in)
    prog = pt.generate_loopy(x1 @ x2)
    _, (out,) = prog(queue)

    assert (out == x1_in @ x2_in).all()


def _do_matmul(x1_shape, x2_shape, queue):
    from numpy.random import default_rng
    rng = default_rng(seed=0)

    x1_in = rng.standard_normal(x1_shape)
    x2_in = rng.standard_normal(x2_shape)

    x1 = pt.make_data_wrapper(x1_in)
    x2 = pt.make_data_wrapper(x2_in)

    prog = pt.generate_loopy(x1 @ x2)
    _, (out,) = prog(queue)

    np_out = np.matmul(x1_in, x2_in)

    assert np_out.shape == out.shape
    np.testing.assert_allclose(np_out, out)


@pytest.mark.parametrize("x1_shape", ((7,),))
@pytest.mark.parametrize("x2_shape", ((7,), (7, 3), (7, 7, 4), (7, 7, 7, 9)))
def test_matmul_dimone(ctx_factory, x1_shape, x2_shape):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    _do_matmul(x1_shape, x2_shape, queue)


@pytest.mark.parametrize("x1_shape", ((9, 7, 11, 4), (9, 7, 1, 4), (7, 4)))
@pytest.mark.parametrize("x2_shape", ((7, 4, 5), (7, 4, 3)))
def test_matmul_higherdim(ctx_factory, x1_shape, x2_shape):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    _do_matmul(x1_shape, x2_shape, queue)


def test_data_wrapper(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # Without name/shape
    x_in = np.array([1, 2, 3, 4, 5])
    x = pt.make_data_wrapper(x_in)
    prog = pt.generate_loopy(x)
    _, (x_out,) = prog(queue)
    assert (x_out == x_in).all()

    # With name/shape
    x_in = np.array([[1, 2], [3, 4], [5, 6]])
    n = pt.make_size_param("n")
    x = pt.make_data_wrapper(x_in, name="x", shape=(n, 2))
    prog = pt.generate_loopy(x)
    _, (x_out,) = prog(queue)
    assert (x_out == x_in).all()


def test_codegen_with_DictOfNamedArrays(ctx_factory):  # noqa
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x = pt.make_placeholder("x", (5,), np.int64)
    y = pt.make_placeholder("y", (5,), np.int64)
    x_in = np.array([1, 2, 3, 4, 5])
    y_in = np.array([6, 7, 8, 9, 10])

    result = pt.make_dict_of_named_arrays({"x_out": x, "y_out": y})

    # With return_dict.
    prog = pt.generate_loopy(result)

    _, outputs = prog(queue, x=x_in, y=y_in)
    assert (outputs["x_out"] == x_in).all()
    assert (outputs["y_out"] == y_in).all()


@pytest.mark.parametrize("shift", (-1, 1, 0, -20, 20))
@pytest.mark.parametrize("axis", (0, 1))
def test_roll(ctx_factory, shift, axis):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    n = pt.make_size_param("n")
    x = pt.make_placeholder(name="x", shape=(n, n), dtype=np.float64)

    x_in = np.arange(1., 10.).reshape(3, 3)
    assert_allclose_to_numpy(pt.roll(x, shift=shift, axis=axis),
                              queue,
                              {x: x_in})


@pytest.mark.parametrize("axes", (
    (), (0, 1), (1, 0),
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)))
def test_axis_permutation(ctx_factory, axes):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    ndim = len(axes)
    shape = (3, 4, 5)[:ndim]

    from numpy.random import default_rng
    rng = default_rng()

    x_in = rng.random(size=shape)

    x = pt.make_data_wrapper(x_in)
    assert_allclose_to_numpy(pt.transpose(x, axes), queue)


def test_transpose(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    shape = (2, 8)

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=shape)

    x = pt.make_data_wrapper(x_in)
    assert_allclose_to_numpy(x.T, queue)


# Doesn't include: ? (boolean), g (float128), G (complex256)
ARITH_DTYPES = "bhilqpBHILQPfdFD"


def reverse_args(f):
    def wrapper(*args):
        return f(*reversed(args))
    return wrapper


@pytest.mark.parametrize("which", ("add", "sub", "mul", "truediv", "pow",
                                   "equal", "not_equal", "less", "less_equal",
                                   "greater", "greater_equal", "logical_and",
                                   "logical_or"))
@pytest.mark.parametrize("reverse", (False, True))
def test_scalar_array_binary_arith(ctx_factory, which, reverse):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    not_valid_in_complex = which in ["equal", "not_equal", "less", "less_equal",
                                     "greater", "greater_equal",
                                     "logical_and", "logical_or"]

    try:
        pt_op = getattr(operator, which)
        np_op = getattr(operator, which)
    except AttributeError:
        pt_op = getattr(pt, which)
        np_op = getattr(np, which)

    if reverse:
        pt_op = reverse_args(pt_op)
        np_op = reverse_args(np_op)

    x_orig = 7
    y_orig = np.array([1, 2, 3, 4, 5])

    for first_dtype in (int, float, complex, np.int32, np.float64,
                        np.complex128):
        x_in = first_dtype(x_orig)

        if first_dtype in [complex, np.complex128] and not_valid_in_complex:
            continue

        exprs = {}
        for dtype in ARITH_DTYPES:
            if dtype in "FDG" and not_valid_in_complex:
                continue
            y = pt.make_data_wrapper(
                    y_orig.astype(dtype), name=f"y{dtype}")
            exprs[dtype] = pt_op(x_in, y)

        prog = pt.generate_loopy(exprs)

        _, outputs = prog(queue)

        for dtype in exprs:
            out = outputs[dtype]
            out_ref = np_op(x_in, y_orig.astype(dtype))

            assert out.dtype == out_ref.dtype, (out.dtype, out_ref.dtype)
            # In some cases ops are done in float32 in loopy but float64 in numpy.
            assert np.allclose(out, out_ref), (out, out_ref)


@pytest.mark.parametrize("which", ("add", "sub", "mul", "truediv", "pow",
                                   "equal", "not_equal", "less", "less_equal",
                                   "greater", "greater_equal", "logical_or",
                                   "logical_and"))
@pytest.mark.parametrize("reverse", (False, True))
def test_array_array_binary_arith(ctx_factory, which, reverse):
    if which == "sub":
        pytest.skip("https://github.com/inducer/loopy/issues/131")

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    not_valid_in_complex = which in ["equal", "not_equal", "less", "less_equal",
                                     "greater", "greater_equal",
                                     "logical_and", "logical_or"]

    try:
        pt_op = getattr(operator, which)
        np_op = getattr(operator, which)
    except AttributeError:
        pt_op = getattr(pt, which)
        np_op = getattr(np, which)

    if reverse:
        pt_op = reverse_args(pt_op)
        np_op = reverse_args(np_op)

    x_orig = np.array([1, 2, 3, 4, 5])
    y_orig = np.array([10, 9, 8, 7, 6])

    for first_dtype in ARITH_DTYPES:
        if first_dtype in "FDG" and not_valid_in_complex:
            continue

        x_in = x_orig.astype(first_dtype)
        x = pt.make_data_wrapper(x_in, name="x")

        exprs = {}
        for dtype in ARITH_DTYPES:
            if dtype in "FDG" and not_valid_in_complex:
                continue
            y = pt.make_data_wrapper(
                    y_orig.astype(dtype), name=f"y{dtype}")
            exprs[dtype] = pt_op(x, y)

        prog = pt.generate_loopy(exprs)

        _, outputs = prog(queue)

        for dtype in ARITH_DTYPES:
            if dtype in "FDG" and not_valid_in_complex:
                continue
            out = outputs[dtype]
            out_ref = np_op(x_in, y_orig.astype(dtype))

            assert out.dtype == out_ref.dtype, (out.dtype, out_ref.dtype)
            # In some cases ops are done in float32 in loopy but float64 in numpy.
            assert np.allclose(out, out_ref), (out, out_ref)


@pytest.mark.parametrize("which", ("__and__", "__or__", "__xor__"))
def test_binary_logic(ctx_factory, which):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    pt_op = getattr(operator, which)
    np_op = getattr(operator, which)

    x_orig = np.array([1, 2, 3, 4, 5])
    y_orig = np.array([5, 4, 3, 2, 1])

    x = pt.make_data_wrapper(x_orig)
    y = pt.make_data_wrapper(y_orig)

    prog = pt.generate_loopy(pt_op(x, y))

    _, out = prog(queue)

    out_ref = np_op(x_orig, y_orig)

    assert out[0].dtype == out_ref.dtype
    assert np.array_equal(out[0], out_ref)


@pytest.mark.parametrize("which", ("neg", "pos"))
def test_unary_arith(ctx_factory, which):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    op = getattr(operator, which)

    x_orig = np.array([1, 2, 3, 4, 5])

    exprs = {}
    for dtype in ARITH_DTYPES:
        exprs[dtype] = op(
                pt.make_data_wrapper(x_orig.astype(dtype)))

    prog = pt.generate_loopy(exprs)

    _, outputs = prog(queue)

    for dtype in ARITH_DTYPES:
        out = outputs[dtype]
        out_ref = op(x_orig.astype(dtype))

        assert out.dtype == out_ref.dtype
        assert np.array_equal(out, out_ref)


def generate_test_slices_for_dim(dim_bound):
    # Include scalars to test indexing.
    for i in range(dim_bound):
        yield i

    for i in range(0, dim_bound):
        for j in range(i + 1, 1 + dim_bound):
            yield slice(i, j, None)


def generate_test_slices(shape):
    yield from itertools.product(*map(generate_test_slices_for_dim, shape))


@pytest.mark.parametrize("shape", [(3,), (2, 2), (1, 2, 1)])
def test_slice(ctx_factory, shape):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from numpy.random import default_rng
    rng = default_rng()

    x_in = rng.random(size=shape)
    x = pt.make_data_wrapper(x_in)

    outputs = {}
    ref_outputs = {}

    i = 0
    for slice_ in generate_test_slices(shape):
        outputs[f"out_{i}"] = x[slice_]
        ref_outputs[f"out_{i}"] = x_in[slice_]
        i += 1

    prog = pt.generate_loopy(outputs)

    _, outputs = prog(queue)

    for output in outputs:
        x_out = outputs[output]
        x_ref = ref_outputs[output]
        assert (x_out == x_ref).all()


@pytest.mark.parametrize("input_dims", (1, 2, 3))
def test_stack(ctx_factory, input_dims):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    shape = (2, 2, 2)[:input_dims]

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=shape)
    y_in = rng.random(size=shape)

    x = pt.make_data_wrapper(x_in)
    y = pt.make_data_wrapper(y_in)

    for axis in range(0, 1 + input_dims):
        assert_allclose_to_numpy(pt.stack((x, y), axis=axis), queue)


def test_concatenate(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from numpy.random import default_rng
    rng = default_rng()
    x0_in = rng.random(size=(3, 9, 3))
    x1_in = rng.random(size=(3, 11, 3))
    x2_in = rng.random(size=(3, 22, 3))

    x0 = pt.make_data_wrapper(x0_in)
    x1 = pt.make_data_wrapper(x1_in)
    x2 = pt.make_data_wrapper(x2_in)

    assert_allclose_to_numpy(pt.concatenate((x0, x1, x2), axis=1), queue)


@pytest.mark.parametrize("oldshape", [(36,),
                                      (3, 3, 4),
                                      (12, 3),
                                      (2, 2, 3, 3, 1)])
@pytest.mark.parametrize("newshape", [(-1,),
                                      (-1, 6),
                                      (4, 9),
                                      (9, -1),
                                      (36, -1),
                                      36])
def test_reshape(ctx_factory, oldshape, newshape):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=oldshape)

    x = pt.make_data_wrapper(x_in)

    assert_allclose_to_numpy(pt.reshape(x, newshape=newshape), queue)
    assert_allclose_to_numpy(x.reshape(newshape), queue)
    if isinstance(newshape, tuple):
        assert_allclose_to_numpy(x.reshape(*newshape), queue)


def test_dict_of_named_array_codegen_avoids_recomputation():
    x = pt.make_placeholder(shape=(10, 4), dtype=float, name="x")
    y = 2*x
    z = y + 4*x

    yz = pt.make_dict_of_named_arrays({"y": y, "z": z})

    knl = pt.generate_loopy(yz).kernel
    assert ("y" in knl.id_to_insn["z_store"].read_dependency_names())


def test_dict_to_loopy_kernel(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(10)

    x = pt.make_data_wrapper(x_in)
    y = 2*x
    z = 3*x

    _, result_dict = pt.generate_loopy({"y": y, "z": z})(
            queue)
    np.testing.assert_allclose(result_dict["y"], 2*x_in)
    np.testing.assert_allclose(result_dict["z"], 3*x_in)


def test_only_deps_as_knl_args():
    # See https://gitlab.tiker.net/inducer/pytato/-/issues/13
    x = pt.make_placeholder(name="x", shape=(10, 4), dtype=float)
    y = pt.make_placeholder(name="y", shape=(10, 4), dtype=float)  # noqa:F841

    z = 2*x
    knl = pt.generate_loopy(z).kernel

    assert "x" in knl.arg_dict
    assert "y" not in knl.arg_dict


@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.complex128))
@pytest.mark.parametrize("function_name", ("abs", "sin", "cos", "tan", "arcsin",
    "arccos", "arctan", "sinh", "cosh", "tanh", "exp", "log", "log10", "sqrt",
    "conj", "__abs__", "real", "imag"))
def test_math_functions(ctx_factory, dtype, function_name):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    if np.dtype(dtype).kind == "c" and function_name in ["arcsin", "arccos",
                                                         "arctan", "log10"]:
        pytest.skip("Unsupported by loopy.")

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=(10, 4)).astype(dtype)

    x = pt.make_data_wrapper(x_in)
    try:
        pt_func = getattr(pt, function_name)
        np_func = getattr(np, function_name)
    except AttributeError:
        pt_func = getattr(operator, function_name)
        np_func = getattr(operator, function_name)

    _, (y,) = pt.generate_loopy(pt_func(x))(queue)

    y_np = np_func(x_in)

    np.testing.assert_allclose(y, y_np, rtol=1e-6)
    assert y.dtype == y_np.dtype


@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.complex128))
@pytest.mark.parametrize("function_name", ("arctan2",))
def test_binary_math_functions(ctx_factory, dtype, function_name):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    if np.dtype(dtype).kind == "c" and function_name in ["arctan2"]:
        pytest.skip("Unsupported by loopy.")

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=(10, 4)).astype(dtype)
    y_in = rng.random(size=(10, 4)).astype(dtype)

    x = pt.make_data_wrapper(x_in)
    y = pt.make_data_wrapper(y_in)
    pt_func = getattr(pt, function_name)
    np_func = getattr(np, function_name)

    _, (out,) = pt.generate_loopy(pt_func(x, y))(queue)

    out_np = np_func(x_in, y_in)
    np.testing.assert_allclose(out, out_np, rtol=1e-6)
    assert out.dtype == out_np.dtype


@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.float32, np.float64))
def test_full_zeros_ones(ctx_factory, dtype):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    _, (z,) = pt.generate_loopy(pt.zeros(10, dtype))(queue)
    _, (o,) = pt.generate_loopy(pt.ones(10, dtype))(queue)
    _, (t,) = pt.generate_loopy(pt.full(10, 2, dtype))(queue)

    for ary in (z, o, t):
        assert ary.dtype == dtype

    assert (z == 0).all()
    assert (o == 1).all()
    assert (t == 2).all()


def test_passsing_bound_arguments_raises(ctx_factory):
    queue = cl.CommandQueue(ctx_factory())

    x = pt.make_data_wrapper(np.ones(10), name="x")
    prg = pt.generate_loopy(42*x)

    with pytest.raises(ValueError):
        evt, (out2,) = prg(queue, x=np.random.rand(10))


@pytest.mark.parametrize("shape1, shape2", (
                                            [(10, 4), ()],
                                            [(), (10, 4)],
                                            [(3,), (32, 32, 3)],
                                            [(32, 32, 3), (3,)],
                                            [(32, 22, 1), (3,)],
                                            [(4, 1, 3), (1, 7, 1)],
                                            [(4, 1, 3), (1, p.Variable("n")+2, 1)],
                                           ))
def test_broadcasting(ctx_factory, shape1, shape2):
    from numpy.random import default_rng
    from pymbolic.mapper.evaluator import evaluate

    queue = cl.CommandQueue(ctx_factory())

    rng = default_rng()
    n = rng.integers(20, 40)
    pt_n = pt.make_size_param("n")

    x_in = rng.random(evaluate(shape1, {"n": n})).astype(np.int8)
    y_in = rng.random(evaluate(shape2, {"n": n})).astype(np.int8)
    x = pt.make_data_wrapper(x_in, shape=evaluate(shape1, {"n": pt_n}))
    y = pt.make_data_wrapper(y_in, shape=evaluate(shape2, {"n": pt_n}))

    prg = pt.generate_loopy(x+y)

    if "n" in prg.kernel.arg_dict:
        evt, (out,) = prg(queue, n=n)
    else:
        evt, (out,) = prg(queue)

    np.testing.assert_allclose(out, x_in+y_in)


@pytest.mark.parametrize("which", ("maximum", "minimum"))
def test_maximum_minimum(ctx_factory, which):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from numpy.random import default_rng
    rng = default_rng()

    def _get_rand_with_nans(shape):
        arr = rng.random(size=shape)
        mask = rng.choice([False, True], shape)
        arr[mask] = np.nan
        return arr

    x1_in = _get_rand_with_nans((10, 4))
    x2_in = _get_rand_with_nans((10, 4))

    x1 = pt.make_data_wrapper(x1_in)
    x2 = pt.make_data_wrapper(x2_in)
    pt_func = getattr(pt, which)
    np_func = getattr(np, which)

    _, (y,) = pt.generate_loopy(pt_func(x1, x2))(queue)
    np.testing.assert_allclose(y, np_func(x1_in, x2_in), rtol=1e-6)


def test_call_loopy(ctx_factory):
    from pytato.loopy import call_loopy
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    x_in = np.random.rand(10, 4)
    x = pt.make_placeholder(shape=(10, 4), dtype=np.float64, name="x")
    y = 2*x

    knl = lp.make_function(
            "{[i, j]: 0<=i<10 and 0<=j<4}",
            """
            Z[i] = 10*sum(j, Y[i, j])
            """, name="callee")

    loopyfunc = call_loopy(knl, bindings={"Y": y}, entrypoint="callee")
    z = loopyfunc["Z"]

    evt, (z_out, ) = pt.generate_loopy(2*z)(queue, x=x_in)

    assert (z_out == 40*(x_in.sum(axis=1))).all()


def test_call_loopy_with_same_callee_names(ctx_factory):
    from pytato.loopy import call_loopy

    queue = cl.CommandQueue(ctx_factory())

    u_in = np.random.rand(10)
    twice = lp.make_function(
            "{[i]: 0<=i<10}",
            """
            y[i] = 2*x[i]
            """, name="callee")

    thrice = lp.make_function(
            "{[i]: 0<=i<10}",
            """
            y[i] = 3*x[i]
            """, name="callee")

    u = pt.make_data_wrapper(u_in)
    cuatro_u = 2*call_loopy(twice, {"x": u}, "callee")["y"]
    nueve_u = 3*call_loopy(thrice, {"x": u}, "callee")["y"]

    out = pt.make_dict_of_named_arrays({"cuatro_u": cuatro_u, "nueve_u": nueve_u})

    evt, out_dict = pt.generate_loopy(out,
                                      options=lp.Options(return_dict=True))(queue)
    np.testing.assert_allclose(out_dict["cuatro_u"], 4*u_in)
    np.testing.assert_allclose(out_dict["nueve_u"], 9*u_in)


def test_exprs_with_named_arrays(ctx_factory):
    queue = cl.CommandQueue(ctx_factory())
    x_in = np.random.rand(10, 4)
    x = pt.make_data_wrapper(x_in)
    y1y2 = pt.make_dict_of_named_arrays({"y1": 2*x, "y2": 3*x})
    res = 21*y1y2["y1"]
    evt, (out,) = pt.generate_loopy(res)(queue)

    np.testing.assert_allclose(out, 42*x_in)


def test_call_loopy_with_parametric_sizes(ctx_factory):

    x_in = np.random.rand(10, 4)

    from pytato.loopy import call_loopy

    queue = cl.CommandQueue(ctx_factory())

    m = pt.make_size_param("M")
    n = pt.make_size_param("N")
    x = pt.make_placeholder(shape=(m, n), dtype=np.float64, name="x")
    y = 3*x

    knl = lp.make_kernel(
            "{[i, j]: 0<=i<m and 0<=j<n}",
            """
            Z[i] = 7*sum(j, Y[i, j])
            """, name="callee", lang_version=(2018, 2))

    loopyfunc = call_loopy(knl, bindings={"Y": y, "m": m, "n": n})
    z = loopyfunc["Z"]

    evt, (z_out, ) = pt.generate_loopy(2*z)(queue, x=x_in)
    np.testing.assert_allclose(z_out, 42*(x_in.sum(axis=1)))


def test_call_loopy_with_scalar_array_inputs(ctx_factory):
    import loopy as lp
    from numpy.random import default_rng
    from pytato.loopy import call_loopy

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    rng = default_rng()
    x_in = rng.random(size=())

    knl = lp.make_kernel(
        "{:}",
        """
        y = 2*x
        """)

    x = pt.make_placeholder(name="x", shape=(), dtype=float)
    y = call_loopy(knl, {"x": 3*x})["y"]

    evt, (out,) = pt.generate_loopy(y)(queue, x=x_in)
    np.testing.assert_allclose(out, 6*x_in)


@pytest.mark.parametrize("axis", (None, 1, 0))
@pytest.mark.parametrize("redn", ("sum", "amax", "amin", "prod", "all", "any"))
@pytest.mark.parametrize("shape", [(2, 2), (1, 2, 1), (3, 4, 5)])
def test_reductions(ctx_factory, axis, redn, shape):
    queue = cl.CommandQueue(ctx_factory())

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=shape)

    x = pt.make_data_wrapper(x_in)
    np_func = getattr(np, redn)
    pt_func = getattr(pt, redn)
    prg = pt.generate_loopy(pt_func(x, axis=axis))

    evt, (out,) = prg(queue)

    assert np.all(abs(1 - out/np_func(x_in, axis)) < 1e-14)


@pytest.mark.parametrize("spec,argshapes", ([("im,mj,km->ijk",
                                              [(3, 3)]*3),

                                             ("ik,kj->ij",  # A @ B
                                              [(4, 3), (3, 5)]),

                                             ("ij,ij->ij",  # A * B
                                              [(4, 4)]*2),

                                             ("ij,ji->ij",  # A * B.T
                                              [(4, 4)]*2),

                                             ("ij,kj->ik",  # inner(A, B)
                                              [(4, 4)]*2),

                                             ("ij,j->j",    # A @ x
                                              [(4, 4), (4,)]),

                                             ("ij->ij",  # identity
                                              [(10, 4)]),

                                             ("ij->ji",  # transpose
                                              [(10, 4)]),

                                             ("ii->i",  # diag
                                              [(5, 5)]),

                                             (" ij ->  ",  # np.sum
                                              [(10, 4)]),
                                             ("dij,ej,ej,dej->ei",  # diff: curvimesh
                                              [(2, 10, 10), (100, 10),
                                               (100, 10), (2, 100, 10)]),

                                             ("dij,ej,ej,dej->ei",  # diff: simplex
                                              [(2, 10, 10), (100, 1),
                                               (100, 1), (2, 100, 10)]),

                                             ("ij,ij->ij",  # broadcasting
                                              [(1, 3), (3, 1)]),
                                             ]))
def test_einsum(ctx_factory, spec, argshapes):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    np_operands = [np.random.rand(*argshape) for argshape in argshapes]
    pt_operands = [pt.make_data_wrapper(x_in) for x_in in np_operands]

    np_out = np.einsum(spec, *np_operands)
    pt_expr = pt.einsum(spec, *pt_operands)

    _, (pt_out,) = pt.generate_loopy(pt_expr)(queue)
    assert np_out.shape == pt_out.shape
    np.testing.assert_allclose(np_out, pt_out)


def test_einsum_with_parametrized_shapes(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    m = pt.make_size_param("m")
    n = pt.make_size_param("n")

    m_in = np.random.randint(2, 20)
    n_in = np.random.randint(2, 20)

    def _get_a_shape(_m, _n):
        return (2*_m+1, 3*_n+7)

    def _get_x_shape(_m, _n):
        return (3*_n+7, )

    A_in = np.random.rand(*_get_a_shape(m_in, n_in))  # noqa: N806
    x_in = np.random.rand(*_get_x_shape(m_in, n_in))
    A = pt.make_data_wrapper(A_in, shape=_get_a_shape(m, n))  # noqa: N806
    x = pt.make_data_wrapper(x_in, shape=_get_x_shape(m, n))

    np_out = np.einsum("ij, j ->  i", A_in, x_in)
    pt_expr = pt.einsum("ij, j ->  i", A, x)

    _, (pt_out,) = pt.generate_loopy(pt_expr)(cq, m=m_in, n=n_in)

    assert np_out.shape == pt_out.shape
    np.testing.assert_allclose(np_out, pt_out)


def test_arguments_passing_to_loopy_kernel_for_non_dependent_vars(ctx_factory):
    from numpy.random import default_rng
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = default_rng()
    ctx = cl.create_some_context()
    x_in = rng.random((3, 3))
    x = pt.make_data_wrapper(x_in)
    _, (out,) = pt.generate_loopy(0 * x)(cq)

    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, 0)


def test_call_loopy_shape_inference1(ctx_factory):
    from pytato.loopy import call_loopy
    import loopy as lp
    from numpy.random import default_rng

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    rng = default_rng()

    A_in = rng.random((20, 37))  # noqa

    knl = lp.make_kernel(
            ["{[i, j]: 0<=i<(2*n + 3*m + 2) and 0<=j<(6*n + 4*m + 3)}",
             "{[ii, jj]: 0<=ii<m and 0<=jj<n}"],
            """
            <> tmp = sum([i, j], A[i, j])
            out[ii, jj] = tmp*(ii + jj)
            """, lang_version=(2018, 2))

    A = pt.make_placeholder(name="x", shape=(20, 37), dtype=np.float64)  # noqa: N806
    y_pt = call_loopy(knl, {"A": A})["out"]

    _, (out,) = pt.generate_loopy(y_pt)(queue, x=A_in)

    np.testing.assert_allclose(out,
                               A_in.sum() * (np.arange(4).reshape(4, 1)
                                             + np.arange(3)))


def test_call_loopy_shape_inference2(ctx_factory):
    from pytato.loopy import call_loopy
    import loopy as lp
    from numpy.random import default_rng

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    rng = default_rng()

    A_in = rng.random((38, 71))  # noqa

    knl = lp.make_kernel(
            ["{[i, j]: 0<=i<(2*n + 3*m + 2) and 0<=j<(6*n + 4*m + 3)}",
             "{[ii, jj]: 0<=ii<m and 0<=jj<n}"],
            """
            <> tmp = sum([i, j], A[i, j])
            out[ii, jj] = tmp*(ii + jj)
            """, lang_version=(2018, 2))

    n1 = pt.make_size_param("n1")
    n2 = pt.make_size_param("n2")
    A = pt.make_placeholder(name="x",  # noqa: N806
                            shape=(4*n1 + 6*n2 + 2, 12*n1 + 8*n2 + 3),
                            dtype=np.float64)

    y_pt = call_loopy(knl, {"A": A})["out"]

    _, (out,) = pt.generate_loopy(y_pt)(queue, x=A_in, n1=3, n2=4)

    np.testing.assert_allclose(out,
                               A_in.sum() * (np.arange(8).reshape(8, 1)
                                             + np.arange(6)))


@pytest.mark.parametrize("n", [4, 3, 5])
@pytest.mark.parametrize("m", [2, 7, None])
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_eye(ctx_factory, n, m, k):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    np_eye = np.eye(n, m, k)
    pt_eye = pt.eye(n, m, k)

    _, (out,) = pt.generate_loopy(pt_eye)(cq)

    assert np_eye.shape == out.shape
    np.testing.assert_allclose(out, np_eye)


def test_arange(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    from numpy.random import default_rng
    rng = default_rng(seed=0)
    for _ in range(30):
        nargs = rng.integers(1, 3, endpoint=True)
        bounds = {0: (-300, 300), 1: (-300, 300), 2: (-10, 10)}
        args = [rng.integers(*bounds[iarg], endpoint=True) for iarg in range(nargs)]

        np_res = np.arange(*args, dtype=np.int64)
        pt_res_sym = pt.arange(*args, dtype=np.int64)

        print(np_res.shape, args)
        _, (pt_res,) = pt.generate_loopy(pt_res_sym)(cq)

        assert np.array_equal(pt_res, np_res)


@pytest.mark.parametrize("which,num_args", ([("maximum", 2),
                                             ("minimum", 2),
                                             ]))
def test_pt_ops_on_scalar_args_computed_eagerly(ctx_factory, which, num_args):
    from numpy.random import default_rng
    rng = default_rng()
    args = [rng.random() for _ in range(num_args)]

    pt_func = getattr(pt, which)
    np_func = getattr(np, which)

    np.testing.assert_allclose(pt_func(*args), np_func(*args))


@pytest.mark.parametrize("a_shape,b_shape", ([((10,), (10,)),
                                              ((10, 4), (4, 10)),
                                              ((10, 2, 2), (2,)),
                                              ((10, 5, 2, 7), (3, 7, 4))]))
@pytest.mark.parametrize("a_dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("b_dtype", [np.float32, np.complex64])
def test_dot(ctx_factory, a_shape, b_shape, a_dtype, b_dtype):
    from numpy.random import default_rng
    rng = default_rng()
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    if np.issubdtype(a_dtype, np.complexfloating):
        a_in = (rng.random(a_shape) + 1j * rng.random(a_shape)).astype(a_dtype)
    else:
        a_in = (rng.random(a_shape)).astype(a_dtype)

    if np.issubdtype(b_dtype, np.complexfloating):
        b_in = (rng.random(b_shape) + 1j * rng.random(b_shape)).astype(b_dtype)
    else:
        b_in = (rng.random(b_shape)).astype(b_dtype)

    a = pt.make_data_wrapper(a_in)
    b = pt.make_data_wrapper(b_in)

    np_result = np.dot(a_in, b_in)
    _, (pt_result,) = pt.generate_loopy(pt.dot(a, b))(cq)

    assert pt_result.shape == np_result.shape
    assert pt_result.dtype == np_result.dtype
    np.testing.assert_allclose(np_result, pt_result, rtol=1e-6)


@pytest.mark.parametrize("a_shape,b_shape", ([((10,), (10,)),
                                              ((10, 4), (4, 10))]))
@pytest.mark.parametrize("a_dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("b_dtype", [np.float32, np.complex64])
def test_vdot(ctx_factory, a_shape, b_shape, a_dtype, b_dtype):
    from numpy.random import default_rng
    rng = default_rng()
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    if np.issubdtype(a_dtype, np.complexfloating):
        a_in = (rng.random(a_shape) + 1j * rng.random(a_shape)).astype(a_dtype)
    else:
        a_in = (rng.random(a_shape)).astype(a_dtype)

    if np.issubdtype(b_dtype, np.complexfloating):
        b_in = (rng.random(b_shape) + 1j * rng.random(b_shape)).astype(b_dtype)
    else:
        b_in = (rng.random(b_shape)).astype(b_dtype)

    a = pt.make_data_wrapper(a_in)
    b = pt.make_data_wrapper(b_in)

    np_result = np.vdot(a_in, b_in)
    _, (pt_result,) = pt.generate_loopy(pt.vdot(a, b))(cq)

    assert pt_result.shape == np_result.shape
    assert pt_result.dtype == np_result.dtype
    np.testing.assert_allclose(np_result, pt_result, rtol=1e-6)


def test_reduction_adds_deps(ctx_factory):
    from numpy.random import default_rng

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    rng = default_rng()
    x_in = rng.random(10)
    x = pt.make_data_wrapper(x_in)
    y = 2*x
    z = pt.sum(y)
    pt_prg = pt.generate_loopy({"y": y, "z": z})

    assert ("y_store"
            in pt_prg.program.default_entrypoint.id_to_insn["z_store"].depends_on)

    _, out_dict = pt_prg(queue)
    np.testing.assert_allclose(np.sum(2*x_in),
                               out_dict["z"])


def test_broadcast_to(ctx_factory):
    from numpy.random import default_rng

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    rng = default_rng()

    # fuzz testing
    for _ in range(20):
        broadcasted_ndim = rng.integers(1, 7)
        broadcasted_shape = tuple(int(dim)
                                  for dim in rng.integers(3, 7, broadcasted_ndim))

        input_ndim = rng.integers(0, broadcasted_ndim+1)
        input_shape = [
            dim if rng.choice((0, 1)) else 1
            for dim in broadcasted_shape[broadcasted_ndim-input_ndim:]]

        x_in = rng.random(input_shape, dtype=np.float32)
        x = pt.make_data_wrapper(x_in)
        evt, (x_brdcst,) = pt.generate_loopy(
                                pt.broadcast_to(x, broadcasted_shape))(queue)

        np.testing.assert_allclose(np.broadcast_to(x_in, broadcasted_shape),
                                   x_brdcst)


def test_advanced_indexing_with_broadcasting(ctx_factory):
    from numpy.random import default_rng

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = default_rng()
    x_in = rng.random((3, 3, 3, 3))
    idx1_in = np.array([[-1], [1]])
    idx2_in = np.array([0, 2])

    x = pt.make_data_wrapper(x_in)
    idx1 = pt.make_data_wrapper(idx1_in)
    idx2 = pt.make_data_wrapper(idx2_in)

    # Case 1
    evt, (pt_out,) = pt.generate_loopy(x[:, ::-1, idx1, idx2])(cq)
    np.testing.assert_allclose(pt_out, x_in[:, ::-1, idx1_in, idx2_in])

    # Case 2
    evt, (pt_out,) = pt.generate_loopy(x[-4:4:-1, idx1, idx2, :])(cq)
    np.testing.assert_allclose(pt_out, x_in[-4:4:-1, idx1_in, idx2_in, :])

    # Case 3
    evt, (pt_out,) = pt.generate_loopy(x[idx1, idx2, -2::-1, :])(cq)
    np.testing.assert_allclose(pt_out, x_in[idx1_in, idx2_in, -2::-1, :])

    # Case 4 (non-contiguous advanced indices)
    evt, (pt_out,) = pt.generate_loopy(x[:, idx1, -2::-1, idx2])(cq)
    np.testing.assert_allclose(pt_out, x_in[:, idx1_in, -2::-1, idx2_in])

    # Case 5 (non-contiguous advanced indices with ellipsis)
    evt, (pt_out,) = pt.generate_loopy(x[idx1, ..., idx2])(cq)
    np.testing.assert_allclose(pt_out, x_in[idx1_in, ..., idx2_in])


def test_advanced_indexing_fuzz(ctx_factory):
    from numpy.random import default_rng

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)
    rng = default_rng(seed=0)

    NSAMPLES = 50  # noqa: N806

    for i in range(NSAMPLES):
        input_ndim = rng.integers(1, 8)

        # choosing shape so that the max input size is approx 20MB
        input_shape = [rng.integers(low=1, high=10)
                       for i in range(input_ndim)]

        indirection_shape = [rng.integers(low=1, high=5)
                             for i in range(rng.integers(1, 4))]

        x_in = rng.random(input_shape, dtype=np.float32)
        x = pt.make_data_wrapper(x_in)

        n_indices = rng.integers(low=1, high=input_ndim+1)
        np_indices = []

        for i in range(n_indices):
            axis_len = input_shape[i]
            idx_type = rng.choice(["INT", "SLICE", "ARRAY"])
            if idx_type == "INT":
                if axis_len == 0:
                    # rng.integers does not like low == high
                    np_indices.append(0)
                else:
                    np_indices.append(int(rng.integers(low=-axis_len,
                                                       high=axis_len)))
            elif idx_type == "SLICE":
                start, stop, step = rng.integers(low=-2*axis_len, high=2*axis_len,
                                                 size=3)
                step = 1 if step == 0 else step

                np_indices.append(slice(int(start), int(stop), int(step)))
            elif idx_type == "ARRAY":
                np_indices.append(rng.integers(low=-axis_len, high=axis_len,
                                               size=indirection_shape))
            else:
                raise NotImplementedError

        pt_indices = [idx if isinstance(idx, (int, slice))
                      else pt.make_data_wrapper(idx)
                      for idx in np_indices]

        evt, (pt_out,) = pt.generate_loopy(x[tuple(pt_indices)])(cq)

        np.testing.assert_allclose(pt_out, x_in[tuple(np_indices)],
                                   err_msg=(f"input_shape={input_shape}, "
                                            f"indirection_shape={indirection_shape},"
                                            f" indices={pt_indices}"))


def test_reshape_on_scalars(ctx_factory):
    # Reported by alexfikl
    # See https://github.com/inducer/pytato/issues/157
    from numpy.random import default_rng
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)
    rng = default_rng()

    # () -> (1,)
    x_in = rng.random(())
    x = pt.make_data_wrapper(x_in)
    assert_allclose_to_numpy(pt.reshape(x, (1,)), cq)

    # (1,) -> ()
    x_in = rng.random((1,))
    x = pt.make_data_wrapper(x_in)
    assert_allclose_to_numpy(pt.reshape(x, ()), cq)

    # (1, 0, 2, 4) -> (192, 168, 0, 1)
    x_in = rng.random((1, 0, 2, 4))
    x = pt.make_data_wrapper(x_in)
    assert_allclose_to_numpy(pt.reshape(x, (192, 168, 0, 1)), cq)


def test_materialize_reduces_flops(ctx_factory):
    x1 = pt.make_placeholder("x1", (10, 4), np.float64)
    x2 = pt.make_placeholder("x2", (10, 4), np.float64)
    x3 = pt.make_placeholder("x3", (10, 4), np.float64)
    x4 = pt.make_placeholder("x4", (10, 4), np.float64)
    x5 = pt.make_placeholder("x5", (10, 4), np.float64)
    cse = x1 + x2 + x3
    y1 = x4 * cse
    y2 = cse / x5
    bad_graph = pt.make_dict_of_named_arrays({"y1": y1, "y2": y2})

    good_graph = pt.transform.materialize_with_mpms(bad_graph)

    bad_t_unit = pt.generate_loopy(bad_graph)
    good_t_unit = pt.generate_loopy(good_graph)
    bad_flops = (lp.get_op_map(bad_t_unit.program,
                              subgroup_size="guess")
                 .filter_by(dtype=[np.float64])
                 .eval_and_sum({}))
    good_flops = (lp.get_op_map(good_t_unit.program,
                               subgroup_size="guess")
                  .filter_by(dtype=[np.float64])
                  .eval_and_sum({}))
    assert good_flops == (bad_flops - 80)


def test_named_temporaries(ctx_factory):
    x = pt.make_placeholder("x", (10, 4), np.float32)
    y = pt.make_placeholder("y", (10, 4), np.float32)
    tmp1 = 2 * x + 3 * y
    tmp2 = 7 * x + 8 * y

    dag = pt.make_dict_of_named_arrays({"out1": 10 * tmp1 + 11 * tmp2,
                                        "out2": 22 * tmp1 + 53 * tmp2
                                        })
    dag = pt.transform.materialize_with_mpms(dag)

    def mark_materialized_nodes_as_cse(ary: Union[pt.Array,
                                                  pt.AbstractResultWithNamedArrays]
                                       ) -> pt.Array:
        if isinstance(ary, pt.AbstractResultWithNamedArrays):
            return ary

        if ary.tags_of_type(pt.tags.ImplStored):
            return ary.tagged(pt.tags.PrefixNamed("cse"))
        else:
            return ary

    dag = pt.transform.map_and_copy(dag, mark_materialized_nodes_as_cse)
    t_unit = pt.generate_loopy(dag).program
    assert len([tv.name.startswith("cse")
               for tv in t_unit.default_entrypoint.temporary_variables.values()]
               ) == 2


@pytest.mark.parametrize("shape", [(1, 3, 1), (1, 1), (2, 2, 3)])
def test_squeeze(ctx_factory, shape):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    x_in = np.ones(shape=shape, dtype=np.float64)
    x = pt.make_data_wrapper(x_in)

    np_result = np.squeeze(x_in).shape

    _, (pt_result,) = pt.generate_loopy(pt.squeeze(x))(cq)

    np.testing.assert_allclose(pt_result.shape, np_result)


def test_random_dag_against_numpy(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    from testlib import RandomDAGContext, make_random_dag
    axis_len = 5

    # We'd like to know if Numpy divides by zero.
    with np.errstate(all="raise"):
        for i in range(50):
            print(i)  # progress indicator for somewhat slow test

            seed = 120 + i
            rdagc_pt = RandomDAGContext(np.random.default_rng(seed=seed),
                    axis_len=axis_len, use_numpy=False)
            rdagc_np = RandomDAGContext(np.random.default_rng(seed=seed),
                    axis_len=axis_len, use_numpy=True)

            ref_result = make_random_dag(rdagc_np)
            dag = make_random_dag(rdagc_pt)
            from pytato.transform import materialize_with_mpms
            dict_named_arys = pt.make_dict_of_named_arrays({"result": dag})
            dict_named_arys = materialize_with_mpms(dict_named_arys)
            if 0:
                pt.show_dot_graph(dict_named_arys)

            _, pt_result = pt.generate_loopy(dict_named_arys)(cq)

            assert np.allclose(pt_result["result"], ref_result)


def test_assume_non_negative_indirect_address(ctx_factory):
    from numpy.random import default_rng
    from pytato.scalar_expr import WalkMapper

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    class OnRemainderRaiser(WalkMapper):
        def map_remainder(self, expr):
            raise RuntimeError

    rng = default_rng()

    a_np = rng.random((10,))
    b_np = rng.permutation(np.arange(10))

    a = pt.make_data_wrapper(a_np)
    b = pt.make_data_wrapper(b_np).tagged(pt.tags.AssumeNonNegative())

    pt_prg = pt.generate_loopy(a[b])

    for insn in pt_prg.program.default_entrypoint.instructions:
        OnRemainderRaiser()(insn.expression)

    evt, (out,) = pt_prg(cq)

    np.testing.assert_allclose(out, a_np[b_np])


def test_axis_tag_to_loopy_iname_tag_propagate():
    from testlib import FooInameTag, BarInameTag, BazInameTag

    x = pt.make_placeholder("x", (10, 4), np.float32)
    y = 2 * x
    y = (y
         .with_tagged_axis(0, FooInameTag())
         .with_tagged_axis(1, BarInameTag()))
    x_sum = pt.sum(
        x,
        axis_to_reduction_descr={
            1: pt.ReductionDescriptor(frozenset([FooInameTag()]))})
    x_einsum = pt.einsum(
        "ij->",
        x,
        index_to_redn_descr={"i": pt.ReductionDescriptor(frozenset([BarInameTag()]))}
    )

    t_unit = pt.generate_loopy({"y": y,
                                "x_sum": x_sum,
                                "x_einsum": x_einsum},
                               axis_tag_t_to_not_propagate=frozenset([BazInameTag])
                               ).program

    assert len(t_unit
               .default_entrypoint
               .inames["y_dim0"]
               .tags_of_type(FooInameTag)) == 1
    assert len(t_unit
               .default_entrypoint
               .inames["y_dim0"]
               .tags_of_type(BarInameTag)) == 0

    assert len(t_unit
               .default_entrypoint
               .inames["y_dim1"]
               .tags_of_type(FooInameTag)) == 0
    assert len(t_unit
               .default_entrypoint
               .inames["y_dim1"]
               .tags_of_type(BarInameTag)) == 1
    assert len(t_unit
               .default_entrypoint
               .inames["_pt_sum_r1"]
               .tags_of_type(FooInameTag)) == 1
    assert len(t_unit
               .default_entrypoint
               .inames["_pt_sum_r0_0"]
               .tags_of_type(BarInameTag)) == 1

    # there shouldn't be any inames tagged with BazInameTag
    assert len([iname
                for iname in t_unit.default_entrypoint.inames.values()
                if iname.tags_of_type(BazInameTag)]) == 0


def test_array_tags_propagated_to_loopy():
    from testlib import FooTag, BarTag, BazTag

    x1 = pt.make_placeholder("x1", (10, 4), dtype=np.float64)
    x2 = pt.make_placeholder("x2", (10, 4), dtype=np.float64)
    y = 2 * x1 + ((3 * x2)
                  .tagged((BarTag(), BazTag(),
                           pt.tags.ImplStored(),
                           pt.tags.Named("tmp")))
                  )

    pt_prg = pt.generate_loopy({"y": y.tagged((FooTag(), BazTag()))},
                               array_tag_t_to_not_propagate=frozenset([BazTag]))
    assert pt_prg.program.default_entrypoint.arg_dict["y"].tags_of_type(FooTag)
    assert not pt_prg.program.default_entrypoint.arg_dict["y"].tags_of_type(BazTag)
    assert (pt_prg.program.default_entrypoint.temporary_variables["tmp"]
            .tags_of_type(BarTag))
    assert not (pt_prg.program.default_entrypoint.temporary_variables["tmp"]
                .tags_of_type(BazTag))


def test_scalars_are_typed(ctx_factory):
    # See https://github.com/inducer/pytato/issues/246
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    x_in = np.random.rand(3, 3)
    x = pt.make_data_wrapper(x_in)
    evt, (pt_out,) = pt.generate_loopy(x * 3.14j)(cq)
    np_out = x_in * 3.14j
    assert pt_out.dtype == np_out.dtype
    np.testing.assert_allclose(pt_out, np_out)


def test_regression_reduction_in_conditional(ctx_factory):
    # Regression test for https://github.com/inducer/pytato/pull/255
    # which was ultimately caused by https://github.com/inducer/loopy/issues/533
    # Reproducer from
    # https://github.com/inducer/pytato/pull/255#issuecomment-1028248714

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    def kernel(usr_np, _pt_data_9):
        _pt_tmp_53 = _pt_data_9 @ _pt_data_9
        _pt_tmp_42 = usr_np.maximum(_pt_tmp_53, _pt_tmp_53)
        _pt_tmp_27 = usr_np.sum(_pt_tmp_42)
        _pt_tmp_0 = usr_np.maximum(_pt_tmp_27, _pt_tmp_53)
        return _pt_tmp_0

    def get_np_input_args():
        return {
            "_pt_data_9": np.ones((2, 2)),
        }

    np_inputs = get_np_input_args()
    np_result = kernel(np, **np_inputs)
    pt_dag = kernel(pt, **{kw: pt.make_data_wrapper(arg)
                           for kw, arg in np_inputs.items()})

    knl = pt.generate_loopy(pt_dag, options=lp.Options(write_cl=True))

    _, (pt_result,) = knl(cq)

    from pytato.analysis import get_num_nodes
    print(get_num_nodes(pt_dag))

    np.testing.assert_allclose(pt_result, np_result)


def test_zero_size_cl_array_dedup(ctx_factory):
    # At pytato@0d8b909 this regression would fail as
    # 'deduplicate_data_wrappers' could not handle 0-long buffers
    import pyopencl.array as cla

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)
    x_cl1 = cla.empty(cq=cq, shape=(0,), dtype="float64")
    x_cl2 = cla.empty(cq=cq, shape=(1,), dtype="float64")
    x1 = pt.make_data_wrapper(x_cl1)
    x2 = pt.make_data_wrapper(x_cl1)
    x3 = pt.make_data_wrapper(x_cl2)
    x4 = pt.make_data_wrapper(x_cl2)

    out = pt.make_dict_of_named_arrays({"out1": 2*x1,
                                        "out2": 2*x2,
                                        "out3": x3 + x4
                                        })

    dedup_dw_out = pt.transform.deduplicate_data_wrappers(out)

    num_nodes_old = pt.analysis.get_num_nodes(out)
    num_nodes_new = pt.analysis.get_num_nodes(dedup_dw_out)
    # 'x2' would be merged with 'x1' as both of them point to the same data
    # 'x3' would be merged with 'x4' as both of them point to the same data
    assert num_nodes_new == (num_nodes_old - 2)


# {{{ test_deterministic_codegen


def _check_deterministic_codegen(ref_dag, ref_t_unit, iproc, results):
    t_unit = pt.generate_loopy(ref_dag).program
    results[iproc] = int(ref_t_unit == t_unit)


def test_deterministic_codegen():
    import multiprocessing as mp
    import os
    original_hash_seed = os.environ.pop("PYTHONHASHSEED", None)

    nprocs = 4

    mp_ctx = mp.get_context("spawn")

    ntests = 15
    for i in range(ntests):
        seed = 120 + i
        results = mp_ctx.Array("i", (0, ) * nprocs)
        print(f"Step {i} {seed}")

        ref_dag = get_random_pt_dag(seed)
        ref_t_unit = pt.generate_loopy(ref_dag).program

        # {{{ spawn nprocs-processes and verify they all compare equally

        procs = [mp_ctx.Process(target=_check_deterministic_codegen,
                                args=(ref_dag,
                                      ref_t_unit,
                                      iproc, results))
                 for iproc in range(nprocs)]

        for iproc, proc in enumerate(procs):
            # See
            # https://replit.com/@KaushikKulkarn1/spawningprocswithhashseedv2?v=1#main.py
            os.environ["PYTHONHASHSEED"] = str(iproc)
            proc.start()

        for proc in procs:
            proc.join()

        if original_hash_seed is not None:
            os.environ["PYTHONHASHSEED"] = original_hash_seed

        assert set(results[:]) == {1}

        # }}}

# }}}


def test_no_computation_for_empty_arrays(ctx_factory):
    # fails at faee1d2f4ffabed96e758a2c0fbd1b1fc1e78719
    import pyopencl.array as cla

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    x_cl1 = cla.empty(cq=cq, shape=(0, 17, 1), dtype="float64")
    x_cl2 = cla.empty(cq=cq, shape=(0, 1, 20), dtype="float64")

    x1 = pt.make_data_wrapper(x_cl1)
    x2 = pt.make_data_wrapper(x_cl2)

    res = x1 + x2

    bprg = pt.generate_loopy(res)
    assert not bprg.program.default_entrypoint.instructions


def test_expand_dims(ctx_factory):
    from numpy.random import default_rng

    ntests = 50
    rng = default_rng(seed=0)
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    for _ in range(ntests):
        in_dim = rng.integers(2, 7)
        n_new_axes = rng.integers(0, 7)
        in_shape = rng.integers(2, 7, in_dim)
        np_input = rng.random(in_shape, "float32")
        pt_input = pt.make_data_wrapper(np_input)
        axis = tuple(int(rng.integers(-(n_new_axes+in_dim), n_new_axes+in_dim))
                     for _ in range(n_new_axes))

        try:
            np_output = np.expand_dims(np_input, axis=axis)
        except ValueError:
            with pytest.raises(ValueError):
                pt.expand_dims(pt_input, axis=axis)
        else:
            _, (pt_output, ) = pt.generate_loopy(pt.expand_dims(pt_input,
                                                                axis=axis))(cq)

            np.testing.assert_allclose(np_output, pt_output)


def test_two_rolls(ctx_factory):
    # see https://github.com/inducer/pytato/issues/341
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)

    n = pt.make_size_param("n")
    x = pt.make_placeholder(name="x", shape=(n, n), dtype=np.float64)

    x_in = np.arange(1., 10.).reshape(3, 3)
    evt, (pt_out,) = pt.generate_loopy(pt.roll(x, -2, 0) + pt.roll(x, -1, 0)
                                       )(cq, x=x_in)
    np_out = np.roll(x_in, -2, 0) + np.roll(x_in, -1, 0)
    np.testing.assert_allclose(np_out, pt_out)


def test_lp_substitution_result(ctx_factory):
    from pytato.target.loopy import ImplSubstitution

    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)

    rng = np.random.default_rng(0)

    a_np = rng.random((10, 5))
    x_np = rng.random((5, ))

    a = pt.make_data_wrapper(a_np)
    x = pt.make_data_wrapper(x_np)

    twice_a = 2*a
    thrice_x = 3*x
    twice_a = twice_a.tagged(ImplSubstitution())
    thrice_x = thrice_x.tagged(ImplSubstitution())

    out = pt.einsum("ij,j->i", twice_a, thrice_x)
    pt_prg = pt.generate_loopy(out)

    assert len(pt_prg.program.default_entrypoint.substitutions) == 2
    _, (pt_eval_out,) = pt_prg(cq)

    np.testing.assert_allclose(np.einsum("ij,j->i", 2*a_np, 3*x_np), pt_eval_out)


def test_rewrite_einsums_with_no_broadcasts(ctx_factory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = np.random.default_rng()

    a = pt.make_data_wrapper(rng.random((10, 4, 1)))
    b = pt.make_data_wrapper(rng.random((10, 1, 4)))
    c = pt.einsum("ijk,ijk->ijk", a, b)
    expr = pt.einsum("ijk,ijk,ijk->i", a, b, c)

    new_expr = pt.rewrite_einsums_with_no_broadcasts(expr)

    _, (ref_out,) = pt.generate_loopy(expr)(cq)
    _, (out,) = pt.generate_loopy(new_expr)(cq)

    np.testing.assert_allclose(ref_out, out)


def test_no_redundant_stores_with_impl_stored(ctx_factory):
    # See <https://github.com/inducer/pytato/issues/415>
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = np.random.default_rng(0)
    x_np = rng.random((10, 4))

    x = pt.make_placeholder("x", (10, 4), np.float64)
    y = 2*x
    y = y.tagged(pt.tags.ImplStored())
    prg = pt.generate_loopy(y)

    assert len(prg.program.default_entrypoint.temporary_variables) == 0
    np.testing.assert_allclose(prg(cq, x=x_np)[1][0], 2*x_np)


def test_placeholders_do_not_diverge_after_removing_impl_stored(ctx_factory):
    # Note: An earlier attempt at fixing
    # <https://github.com/inducer/pytato/issues/415> would create multiple
    # instances of placeholders in the graph leading to incoherent codes.

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = np.random.default_rng(0)
    x_np = rng.random((10,))

    x = pt.make_placeholder("x", 10, np.float64).tagged(pt.tags.ImplStored())
    prg = pt.generate_loopy({"out1": 3*x, "out2": x})
    _, out = prg(cq, x=x_np)
    np.testing.assert_allclose(out["out1"], 3*x_np)
    np.testing.assert_allclose(out["out2"], x_np)


def _get_masking_array_for_test_pad(array, pad_widths):
    from pytato.pad import _normalize_pad_width
    pad_widths = _normalize_pad_width(array, pad_widths)

    def _get_mask_array_idx(*idxs):
        return np.where(
            sum([((idx < pad_width[0])
                  | (idx >= (pad_width[0]+axis_len))
                  ).astype(np.int32)
                 for idx, axis_len, pad_width in zip(idxs,
                                                     array.shape,
                                                     pad_widths)]) > 1,
            0*idxs[0],
            0*idxs[0] + 1)

    return np.fromfunction(
        _get_mask_array_idx,
        shape=tuple(dim + pad_width[0] + pad_width[1]
                    for dim, pad_width in zip(array.shape, pad_widths)),
    )


def test_pad(ctx_factory):

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)
    rng = np.random.default_rng(0)

    for _ in range(10):
        ndim = rng.integers(1, 8)
        ary_shape = rng.integers(2, 7, ndim)
        ary_np = rng.random(tuple(ary_shape), dtype=np.float32)
        ary = pt.make_data_wrapper(ary_np)

        # test constant pad length - I
        pad_width = rng.integers(0, 3)
        np_out = np.pad(ary_np, pad_width)
        out = pt.pad(ary, pad_width)

        _, (pt_out,) = pt.generate_loopy(out)(cq)
        mask_array = _get_masking_array_for_test_pad(ary_np, pad_width)

        np.testing.assert_allclose(np_out * mask_array, pt_out * mask_array)

        # testing constant pad length - II
        pad_width = tuple(rng.integers(0, 3, 2))
        np_out = np.pad(ary_np, pad_width)
        out = pt.pad(ary, pad_width)

        _, (pt_out,) = pt.generate_loopy(out)(cq)
        mask_array = _get_masking_array_for_test_pad(ary_np, pad_width)

        np.testing.assert_allclose(np_out * mask_array, pt_out * mask_array)

        # test unequal padding - I
        pad_width = [tuple(pad) for pad in rng.integers(0, 3, (ndim, 2))]
        np_out = np.pad(ary_np, pad_width, constant_values=32)
        out = pt.pad(ary, pad_width, constant_values=32)

        _, (pt_out,) = pt.generate_loopy(out)(cq)
        mask_array = _get_masking_array_for_test_pad(ary_np, pad_width)

        np.testing.assert_allclose(np_out * mask_array, pt_out * mask_array)

        # test unequal padding - II
        pad_width = [tuple(pad) for pad in rng.integers(0, 3, (ndim, 2))]
        np_out = np.pad(ary_np, pad_width, constant_values=(32, 42))
        out = pt.pad(ary, pad_width, constant_values=(32, 42))

        _, (pt_out,) = pt.generate_loopy(out)(cq)
        mask_array = _get_masking_array_for_test_pad(ary_np, pad_width)

        np.testing.assert_allclose(np_out * mask_array, pt_out * mask_array)

        # test unequal padding - III
        pad_width = [tuple(pad) for pad in rng.integers(0, 3, (ndim, 2))]
        constant_values = [tuple(val) for val in rng.random((ndim, 2))]

        np_out = np.pad(ary_np, pad_width, constant_values=constant_values)
        out = pt.pad(ary, pad_width, constant_values=constant_values)

        _, (pt_out,) = pt.generate_loopy(out)(cq)
        mask_array = _get_masking_array_for_test_pad(ary_np, pad_width)

        np.testing.assert_allclose(np_out * mask_array, pt_out * mask_array)


def test_function_call(ctx_factory, visualize=False):
    from functools import partial
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)

    def f(x):
        return 2*x

    def g(tracer, x):
        return tracer(f, x), 3*x

    def h(x, y):
        return {"twice": 2*x+y, "thrice": 3*x+y}

    def build_expression(tracer):
        x = pt.arange(500, dtype=np.float32)
        twice_x = tracer(f, x)
        twice_x_2, thrice_x_2 = tracer(partial(g, tracer), x)

        result = tracer(h, x, 2*x)
        twice_x_3 = result["twice"]
        thrice_x_3 = result["thrice"]

        return {"foo": 3.14 + twice_x_3,
                "bar": 4 * thrice_x_3,
                "baz": 65 * twice_x,
                "quux": 7 * twice_x_2}

    result_with_functions = pt.tag_all_calls_to_be_inlined(
        pt.make_dict_of_named_arrays(build_expression(pt.trace_call)))
    result_without_functions = pt.make_dict_of_named_arrays(
        build_expression(lambda fn, *args: fn(*args)))

    # test that visualizing graphs with functions works
    dot = pt.get_dot_graph(result_with_functions)

    if visualize:
        pt.show_dot_graph(dot)

    _, outputs = pt.generate_loopy(result_with_functions)(cq, out_host=True)
    _, expected = pt.generate_loopy(result_without_functions)(cq, out_host=True)

    assert len(outputs) == len(expected)

    for key in outputs.keys():
        np.testing.assert_allclose(outputs[key], expected[key])


@pytest.mark.parametrize("should_concatenate_bar", (False, True))
def test_nested_function_calls(ctx_factory, should_concatenate_bar):
    from functools import partial

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = np.random.default_rng(0)
    x_np = rng.random((10,))

    x = pt.make_placeholder("x", 10, np.float64).tagged(pt.tags.ImplStored())
    prg = pt.generate_loopy({"out1": 3*x, "out2": x})
    _, out = prg(cq, x=x_np)
    np.testing.assert_allclose(out["out1"], 3*x_np)
    np.testing.assert_allclose(out["out2"], x_np)
    ref_tracer = lambda f, *args, identifier: f(*args)  # noqa: E731

    def foo(tracer, x, y):
        return 2*x + 3*y

    def bar(tracer, x, y):
        foo_x_y = tracer(partial(foo, tracer), x, y, identifier="foo")
        return foo_x_y * x * y

    def call_bar(tracer, x, y):
        return tracer(partial(bar, tracer), x, y, identifier="bar")

    x1_np, y1_np = rng.random((2, 13, 29))
    x2_np, y2_np = rng.random((2, 4, 29))
    x1, y1 = pt.make_data_wrapper(x1_np), pt.make_data_wrapper(y1_np)
    x2, y2 = pt.make_data_wrapper(x2_np), pt.make_data_wrapper(y2_np)
    result = pt.make_dict_of_named_arrays({"out1": call_bar(pt.trace_call, x1, y1),
                                           "out2": call_bar(pt.trace_call, x2, y2)}
                                          )
    result = pt.tag_all_calls_to_be_inlined(result)
    if should_concatenate_bar:
        from pytato.transform.calls import CallsiteCollector
        assert len(CallsiteCollector(())(result)) == 4
        result = pt.concatenate_calls(
            result,
            lambda x: pt.tags.FunctionIdentifier("bar") in x.call.function.tags)
        assert len(CallsiteCollector(())(result)) == 2

    expect = pt.make_dict_of_named_arrays({"out1": call_bar(ref_tracer, x1, y1),
                                           "out2": call_bar(ref_tracer, x2, y2)}
                                          )

    _, result_out = pt.generate_loopy(result)(cq)
    _, expect_out = pt.generate_loopy(expect)(cq)

    assert result_out.keys() == expect_out.keys()
    for k in expect_out:
        np.testing.assert_allclose(result_out[k], expect_out[k])


def test_concatenate_calls_no_nested(ctx_factory):
    rng = np.random.default_rng(0)

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    def foo(x, y):
        return 3*x + 4*y + 42*pt.sin(x) + 1729*pt.tan(y)*pt.maximum(x, y)

    x1 = pt.make_placeholder("x1", (10, 4), np.float64)
    x2 = pt.make_placeholder("x2", (10, 4), np.float64)

    y1 = pt.make_placeholder("y1", (10, 4), np.float64)
    y2 = pt.make_placeholder("y2", (10, 4), np.float64)

    z1 = pt.make_placeholder("z1", (10, 4), np.float64)
    z2 = pt.make_placeholder("z2", (10, 4), np.float64)

    result = pt.make_dict_of_named_arrays({"out1": 2*pt.trace_call(foo, 2*x1, 3*x2),
                                           "out2": 4*pt.trace_call(foo, 4*y1, 9*y2),
                                           "out3": 6*pt.trace_call(foo, 7*z1, 8*z2)
                                           })

    concatenated_result = pt.concatenate_calls(
        result, lambda x: pt.tags.FunctionIdentifier("foo") in x.call.function.tags)

    result = pt.tag_all_calls_to_be_inlined(result)
    concatenated_result = pt.tag_all_calls_to_be_inlined(concatenated_result)

    assert (pt.analysis.get_num_nodes(pt.inline_calls(result))
            > pt.analysis.get_num_nodes(pt.inline_calls(concatenated_result)))

    x1_np, x2_np, y1_np, y2_np, z1_np, z2_np = rng.random((6, 10, 4))

    _, out_dict1 = pt.generate_loopy(result)(cq,
                                             x1=x1_np, x2=x2_np,
                                             y1=y1_np, y2=y2_np,
                                             z1=z1_np, z2=z2_np)

    _, out_dict2 = pt.generate_loopy(concatenated_result)(cq,
                                                          x1=x1_np, x2=x2_np,
                                                          y1=y1_np, y2=y2_np,
                                                          z1=z1_np, z2=z2_np)
    assert out_dict1.keys() == out_dict2.keys()

    for key in out_dict1:
        np.testing.assert_allclose(out_dict1[key], out_dict2[key])


def test_concatenation_via_constant_expressions(ctx_factory):

    from pytato.transform.calls import CallsiteCollector

    rng = np.random.default_rng(0)

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    def resampling(coords, iels):
        return coords[iels]

    n_el = 1000
    n_dof = 20
    n_dim = 3

    n_left_els = 17
    n_right_els = 29

    coords_dofs_np = rng.random((n_el, n_dim, n_dof), np.float64)
    left_bnd_iels_np = rng.integers(low=0, high=n_el, size=n_left_els)
    right_bnd_iels_np = rng.integers(low=0, high=n_el, size=n_right_els)

    coords_dofs = pt.make_data_wrapper(coords_dofs_np)
    left_bnd_iels = pt.make_data_wrapper(left_bnd_iels_np)
    right_bnd_iels = pt.make_data_wrapper(right_bnd_iels_np)

    lcoords = pt.trace_call(resampling, coords_dofs, left_bnd_iels)
    rcoords = pt.trace_call(resampling, coords_dofs, right_bnd_iels)

    result = pt.make_dict_of_named_arrays({"lcoords": lcoords,
                                           "rcoords": rcoords})
    result = pt.tag_all_calls_to_be_inlined(result)

    assert len(CallsiteCollector(())(result)) == 2
    concated_result = pt.concatenate_calls(
        result,
        lambda cs: pt.tags.FunctionIdentifier("resampling") in cs.call.function.tags
    )
    assert len(CallsiteCollector(())(concated_result)) == 1

    _, out_result = pt.generate_loopy(result)(cq)
    np.testing.assert_allclose(out_result["lcoords"],
                               coords_dofs_np[left_bnd_iels_np])
    np.testing.assert_allclose(out_result["rcoords"],
                               coords_dofs_np[right_bnd_iels_np])

    _, out_concated_result = pt.generate_loopy(result)(cq)
    np.testing.assert_allclose(out_concated_result["lcoords"],
                               coords_dofs_np[left_bnd_iels_np])
    np.testing.assert_allclose(out_concated_result["rcoords"],
                               coords_dofs_np[right_bnd_iels_np])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
