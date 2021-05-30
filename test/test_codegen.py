#!/usr/bin/env python

__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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

import pytato as pt
from pytato.array import Placeholder
from testlib import assert_allclose_to_numpy
import pymbolic.primitives as p


def test_basic_codegen(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x = Placeholder("x", (5,), np.int64)
    prog = pt.generate_loopy(x * x, cl_device=queue.device)
    x_in = np.array([1, 2, 3, 4, 5])
    _, (out,) = prog(queue, x=x_in)
    assert (out == x_in * x_in).all()


def test_scalar_placeholder(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x = Placeholder("x", (), np.int64)
    prog = pt.generate_loopy(x, cl_device=queue.device)
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
    prog = pt.generate_loopy(n, cl_device=queue.device)
    x_in = np.array([1, 2, 3, 4, 5])
    _, (n_out,) = prog(queue, x=x_in)
    assert n_out == 5


@pytest.mark.parametrize("x1_ndim", (1, 2))
@pytest.mark.parametrize("x2_ndim", (1, 2))
def test_matmul(ctx_factory, x1_ndim, x2_ndim):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    def get_array(ndim):
        arr = np.array([[1, 2], [3, 4]])
        return arr[(0,) * (arr.ndim - ndim)]

    x1_in = get_array(x1_ndim)
    x2_in = get_array(x2_ndim)

    x1 = pt.make_data_wrapper(x1_in)
    x2 = pt.make_data_wrapper(x2_in)
    prog = pt.generate_loopy(x1 @ x2, cl_device=queue.device)
    _, (out,) = prog(queue)

    assert (out == x1_in @ x2_in).all()


def test_data_wrapper(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # Without name/shape
    x_in = np.array([1, 2, 3, 4, 5])
    x = pt.make_data_wrapper(x_in)
    prog = pt.generate_loopy(x, cl_device=queue.device)
    _, (x_out,) = prog(queue)
    assert (x_out == x_in).all()

    # With name/shape
    x_in = np.array([[1, 2], [3, 4], [5, 6]])
    n = pt.make_size_param("n")
    x = pt.make_data_wrapper(x_in, name="x", shape=(n, 2))
    prog = pt.generate_loopy(x, cl_device=queue.device)
    _, (x_out,) = prog(queue)
    assert (x_out == x_in).all()


def test_codegen_with_DictOfNamedArrays(ctx_factory):  # noqa
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    x = Placeholder("x", (5,), np.int64)
    y = Placeholder("y", (5,), np.int64)
    x_in = np.array([1, 2, 3, 4, 5])
    y_in = np.array([6, 7, 8, 9, 10])

    result = pt.DictOfNamedArrays(dict(x_out=x, y_out=y))

    # Without return_dict.
    prog = pt.generate_loopy(result, cl_device=queue.device,
            options=lp.Options(return_dict=False))
    _, (x_out, y_out) = prog(queue, x=x_in, y=y_in)
    assert (x_out == x_in).all()
    assert (y_out == y_in).all()

    # With return_dict.
    prog = pt.generate_loopy(result, cl_device=queue.device)

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

    for first_dtype in (int, float, complex):
        x_in = first_dtype(x_orig)

        if first_dtype == complex and not_valid_in_complex:
            continue

        exprs = {}
        for dtype in ARITH_DTYPES:
            if dtype in "FDG" and not_valid_in_complex:
                continue
            y = pt.make_data_wrapper(
                    y_orig.astype(dtype), name=f"y{dtype}")
            exprs[dtype] = pt_op(x_in, y)

        prog = pt.generate_loopy(exprs, cl_device=queue.device)

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

        prog = pt.generate_loopy(exprs, cl_device=queue.device)

        _, outputs = prog(queue)

        for dtype in ARITH_DTYPES:
            if dtype in "FDG" and not_valid_in_complex:
                continue
            out = outputs[dtype]
            out_ref = np_op(x_in, y_orig.astype(dtype))

            assert out.dtype == out_ref.dtype, (out.dtype, out_ref.dtype)
            # In some cases ops are done in float32 in loopy but float64 in numpy.
            assert np.allclose(out, out_ref), (out, out_ref)


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

    prog = pt.generate_loopy(exprs, cl_device=queue.device)

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

    prog = pt.generate_loopy(outputs, cl_device=queue.device)

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
                                      (36, -1)])
def test_reshape(ctx_factory, oldshape, newshape):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=oldshape)

    x = pt.make_data_wrapper(x_in)

    assert_allclose_to_numpy(pt.reshape(x, newshape=newshape), queue)


def test_dict_of_named_array_codegen_avoids_recomputation():
    x = pt.make_placeholder(shape=(10, 4), dtype=float, name="x")
    y = 2*x
    z = y + 4*x

    yz = pt.DictOfNamedArrays({"y": y, "z": z})

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

    _, result_dict = pt.generate_loopy({"y": y, "z": z}, cl_device=queue.device)(
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


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("function_name", ("abs", "sin", "cos", "tan", "arcsin",
    "arccos", "arctan", "sinh", "cosh", "tanh", "exp", "log", "log10", "sqrt"))
def test_math_functions(ctx_factory, dtype, function_name):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=(10, 4)).astype(dtype)

    x = pt.make_data_wrapper(x_in)
    pt_func = getattr(pt, function_name)
    np_func = getattr(np, function_name)

    _, (y,) = pt.generate_loopy(pt_func(x),
            cl_device=queue.device)(queue)
    np.testing.assert_allclose(y, np_func(x_in), rtol=1e-6)


@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.float32, np.float64))
def test_full_zeros_ones(ctx_factory, dtype):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    _, (z,) = pt.generate_loopy(pt.zeros(10, dtype),
            cl_device=queue.device)(queue)
    _, (o,) = pt.generate_loopy(pt.ones(10, dtype),
            cl_device=queue.device)(queue)
    _, (t,) = pt.generate_loopy(pt.full(10, 2, dtype),
            cl_device=queue.device)(queue)

    for ary in (z, o, t):
        assert ary.dtype == dtype

    assert (z == 0).all()
    assert (o == 1).all()
    assert (t == 2).all()


def test_passsing_bound_arguments_raises(ctx_factory):
    queue = cl.CommandQueue(ctx_factory())

    x = pt.make_data_wrapper(np.ones(10), name="x")
    prg = pt.generate_loopy(42*x, cl_device=queue.device)

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

    prg = pt.generate_loopy(x+y, cl_device=queue.device)

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

    _, (y,) = pt.generate_loopy(pt_func(x1, x2),
                                cl_device=queue.device)(queue)
    np.testing.assert_allclose(y, np_func(x1_in, x2_in), rtol=1e-6)


@pytest.mark.parametrize("axis", (None, 1, 0))
@pytest.mark.parametrize("redn", ("sum", "amax", "amin", "prod"))
@pytest.mark.parametrize("shape", [(2, 2), (1, 2, 1), (3, 4, 5)])
def test_reductions(ctx_factory, axis, redn, shape):
    queue = cl.CommandQueue(ctx_factory())

    from numpy.random import default_rng
    rng = default_rng()
    x_in = rng.random(size=shape)

    x = pt.make_data_wrapper(x_in)
    np_func = getattr(np, redn)
    pt_func = getattr(pt, redn)
    prg = pt.generate_loopy(pt_func(x, axis=axis), cl_device=queue.device)

    evt, (out,) = prg(queue)

    assert np.all(abs(1 - out/np_func(x_in, axis)) < 1e-14)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
