from __future__ import annotations


__copyright__ = "Copyright (C) 2026 Kaushik Kulkarni"

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

import numpy as np
import pytest
from numpy.random import default_rng
from testlib import assert_allclose_to_ref
from typing_extensions import override

import pyopencl as cl
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,  # pyright: ignore[reportUnusedImport]
)

import pytato as pt


class AssertOnlyMaterializedIndexees(pt.transform.CachedWalkMapper[[]]):
    @override
    def get_cache_key(self, expr: pt.transform.ArrayOrNames) -> pt.transform.CacheKeyT:
        return (expr,)

    @override
    def _map_index_base(self, expr: pt.IndexBase) -> None:
        indexee = expr.array
        assert (
            isinstance(indexee, pt.InputArgumentBase)
            or pt.tags.ImplStored() in expr.tags
        )
        self.rec(indexee)  # do not recurse over indexes.


def assert_only_materialized_indexees(expr: pt.transform.ArrayOrNames) -> None:
    mapper = AssertOnlyMaterializedIndexees()
    mapper(expr)


def test_indirection_pusher_0():
    x = pt.make_placeholder("x", 10)
    idx = pt.make_placeholder("idx", 1729, np.int32)
    y = x[idx]
    assert pt.push_index_to_materialized_nodes(y) == y


def test_indirection_pusher_1(ctx_factory):
    x = pt.make_placeholder("x", 10)
    idx = pt.make_placeholder("idx", 1729, np.int32)
    y = (2 * x)[idx]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == 2 * (x[idx])

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {"x": rng.random(10), "idx": rng.integers(0, 10, 1729, np.int32)},
    )


def test_indirection_pusher_2(ctx_factory):
    x1 = pt.make_placeholder("x1", 10)
    x2 = pt.make_placeholder("x2", 10)
    idx = pt.make_placeholder("idx", 1729, np.int32)
    y = (x1 * x2)[idx]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == (x1[idx] * x2[idx])

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x1": rng.random(10),
            "x2": rng.random(10),
            "idx": rng.integers(0, 10, 1729, np.int32),
        },
    )


def test_indirection_pusher_3(ctx_factory):
    x = pt.make_placeholder("x", 10)
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 314, np.int32)
    y = x[idx1][idx2]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == x[idx1[idx2]]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random(10),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 1729, 314, np.int32),
        },
    )


def test_indirection_pusher_4(ctx_factory):
    x = pt.make_placeholder("x", 10)
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 314, np.int32)
    y = (2 * x[idx1])[idx2]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == 2 * x[idx1[idx2]]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random(10),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 1729, 314, np.int32),
        },
    )


def test_indirection_pusher_5(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    idx5 = pt.make_placeholder("idx5", 314, np.int32)

    y = x[:, idx1, idx2, :][idx3, idx4, idx5]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == x[idx3, idx1[idx4], idx2[idx4], idx5]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 10, 314, np.int32),
            "idx4": rng.integers(0, 1729, 314, np.int32),
            "idx5": rng.integers(0, 10, 314, np.int32),
        },
    )


def test_indirection_pusher_6(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    idx5 = pt.make_placeholder("idx5", 314, np.int32)

    y = x[::2, idx1, idx2, ::3][idx3, idx4, idx5]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == x[2 * idx3, idx1[idx4], idx2[idx4], 3 * idx5]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 5, 314, np.int32),
            "idx4": rng.integers(0, 1729, 314, np.int32),
            "idx5": rng.integers(0, 4, 314, np.int32),
        },
    )


def test_indirection_pusher_7(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)

    y = x[idx1, :, idx2][idx3, idx4]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == x[idx1[idx3], idx4, idx2[idx3]]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random((10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 1729, 314, np.int32),
            "idx4": rng.integers(0, 10, 314, np.int32),
        },
    )


def test_indirection_pusher_8(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)

    y = x[idx1, ::2, idx2][idx3, idx4]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == x[idx1[idx3], 2 * idx4, idx2[idx3]]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random((10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 1729, 314, np.int32),
            "idx4": rng.integers(0, 5, 314, np.int32),
        },
    )


def test_indirection_pusher_9(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)

    y = x[idx1, idx2, ::2, ::3][idx3, :, idx4]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == x[idx1[idx3], idx2[idx3], ::2, 3 * idx4]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 1729, 314, np.int32),
            "idx4": rng.integers(0, 4, 314, np.int32),
        },
    )


def test_indirection_pusher_10(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    # (_0, _1, _2) -> (idx1[_0], 2*_1, idx2[_0], _2)
    # (_0, _1)     -> (idx3[_0], 3*_1, idx4[_0])
    # Net:
    # (_0, _1) -> (idx1[idx3[_0]], 6*_1, idx2[idx3[_0]], idx4[_0])

    y = x[idx1, ::2, idx2][idx3, ::3, idx4]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == x[idx1[idx3], ::6, idx2[idx3], idx4]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 1729, 314, np.int32),
            "idx4": rng.integers(0, 10, 314, np.int32),
        },
    )


def test_indirection_pusher_11(ctx_factory):
    x1 = pt.make_placeholder("x1", (10, 1, 10, 1))
    x2 = pt.make_placeholder("x2", (1, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    y1 = (x1 + x2)[:, idx1, idx2, :]
    # (_0, _1, _2, _3) -> x1[_0, 0, _2, 0] + x2[0, _1, _2, _3]
    # (_0, _1, _2) -> (_0, idx1[_1], idx2[_1], _2])
    # Net ->
    # (_0, _1, _2) -> x1[_0, 0, idx2[_1], 0] + x2[0, idx1[_1], idx2[_1], _2]
    y2 = x1[:, 0, idx2, :] + x2[:, idx1, idx2, :]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x1": rng.random((10, 1, 10, 1)),
            "x2": rng.random((1, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
        },
    )


def test_indirection_pusher_12(ctx_factory):
    x1 = pt.make_placeholder("x1", (10, 1, 10, 1))
    x2 = pt.make_placeholder("x2", (1, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    y1 = (x1 + x2)[idx1, :, idx2, :]
    # (_0, _1, _2, _3) -> x1[_0, 0, _2, 0] + x2[0, _1, _2, _3]
    # (_0, _1, _2) -> (idx1[_0], _1, idx2[_0], _2)
    # Net->
    # (_0, _1, _2) -> x1[idx1[_0], 0, idx2[_0], 0] + x2[0, _1, idx2[_0], _2]

    y2 = x1[idx1, :, idx2, :] + x2[0, :, idx2, :]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x1": rng.random((10, 1, 10, 1)),
            "x2": rng.random((1, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
        },
    )


@pytest.mark.xfail(reason="axis permutation not yet supported.")
def test_indirection_pusher_13():
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    y1 = pt.transpose(x, (0, 2, 3, 1))[idx1, :idx2, :]
    # (_0, _1, _2, _3) -> (_0, _2, _3, _1)
    # (_0, _1, _2) -> (idx1[_0], _1, idx2[_0], _2)
    # Net->
    # (idx1[_0], idx2[_0], _2, _1)
    y2 = pt.transpose(x[idx1, idx2], (0, 1, 3, 2))
    assert pt.push_index_to_materialized_nodes(y1) == y2


@pytest.mark.xfail(reason="axis permutation not yet supported.")
def test_indirection_pusher_14():
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    y1 = pt.transpose(x, (0, 2, 3, 1))[idx1, :idx2, :]
    # (_0, _1, _2, _3) -> (_0, _2, _3, _1)
    # (_0, _1, _2) -> (idx1[_0], _1, idx2[_0], _2)
    # Net->
    # (idx1[_0], idx2[_0], _2, _1)
    y2 = pt.transpose(x[idx1, idx2], (0, 1, 3, 2))
    assert pt.push_index_to_materialized_nodes(y1) == y2


def test_indirection_pusher_15(ctx_factory):
    x = pt.make_placeholder("x", (10, 10))
    idx1 = pt.make_placeholder("idx1", 4, np.int32)
    idx2 = pt.make_placeholder("idx2", (10, 4), np.int32)
    idx3 = pt.make_placeholder("idx3", (1, 10, 10), np.int32)
    idx4 = pt.make_placeholder("idx4", (10, 10, 10), np.int32)

    y = x[idx1, idx2][idx3, idx4]
    y_prime = pt.push_index_to_materialized_nodes(y)
    assert y_prime == x[idx1[idx4], idx2[idx3, idx4]]

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(
        y_prime,
        y,
        cq,
        {
            "x": rng.random((10, 10)),
            "idx1": rng.integers(0, 10, 4, np.int32),
            "idx2": rng.integers(0, 10, (10, 4), np.int32),
            "idx3": rng.integers(0, 10, (1, 10, 10), np.int32),
            "idx4": rng.integers(0, 4, (10, 10, 10), np.int32),
        },
    )


def test_indirection_pusher_16(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10))
    idx1 = pt.make_placeholder("idx1", (4, 1, 4), np.int32)
    idx2 = pt.make_placeholder("idx2", (10, 4), np.int32)
    idx3 = pt.make_placeholder("idx3", (10, 4), np.int32)
    idx4 = pt.make_placeholder("idx4", (10, 1), np.int32)
    idx5 = pt.make_placeholder("idx5", (10, 10), np.int32)
    y1 = x[idx1, idx2, idx3][idx4, 2:5, idx5]
    y2 = x[
        idx1[idx4, :, idx5],
        pt.transpose(idx2[2:5, idx5], (1, 2, 0)),
        pt.transpose(idx3[2:5, idx5], (1, 2, 0)),
    ]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x": rng.random((10, 10, 10)),
            "idx1": rng.integers(0, 10, (4, 1, 4), np.int32),
            "idx2": rng.integers(0, 10, (10, 4), np.int32),
            "idx3": rng.integers(0, 10, (10, 4), np.int32),
            "idx4": rng.integers(0, 4, (10, 1), np.int32),
            "idx5": rng.integers(0, 4, (10, 10), np.int32),
        },
    )


def test_indirection_pusher_17(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    y1 = x[:, idx1, :, idx2][:, idx3, idx4]
    y2 = x[idx3, pt.expand_dims(idx1, 1), idx4, pt.expand_dims(idx2, 1)]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 10, 314, np.int32),
            "idx4": rng.integers(0, 10, 314, np.int32),
        },
    )


def test_indirection_pusher_18(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    y1 = x[:, idx1, idx2, :][:, idx3, idx4]
    y2 = x[:, idx1[idx3], idx2[idx3], idx4]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 1729, 314, np.int32),
            "idx4": rng.integers(0, 10, 314, np.int32),
        },
    )


def test_indirection_pusher_19(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    y1 = x[:, idx1, :, idx2, :][:, :, idx3, idx4]
    y2 = pt.transpose(
        x[:, pt.expand_dims(idx1, 1), idx3, pt.expand_dims(idx2, 1), idx4], (1, 0, 2)
    )
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x": rng.random((10, 10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 10, 314, np.int32),
            "idx4": rng.integers(0, 10, 314, np.int32),
        },
    )


def test_indirection_pusher_20(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", (271, 172, 31), np.int32)
    idx2 = pt.make_placeholder("idx2", (172, 31), np.int32)
    idx3 = pt.make_placeholder("idx3", 6, np.int32)
    idx4 = pt.make_placeholder("idx4", (10, 6), np.int32)
    y1 = x[:, idx1, :, idx2][:, :, :, idx3, idx4]
    y2 = x[idx3, pt.expand_dims(idx1, (3, 4)), idx4, pt.expand_dims(idx2, (2, 3))]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, (271, 172, 31), np.int32),
            "idx2": rng.integers(0, 10, (172, 31), np.int32),
            "idx3": rng.integers(0, 10, 6, np.int32),
            "idx4": rng.integers(0, 10, (10, 6), np.int32),
        },
    )


def test_indirection_pusher_21(ctx_factory):
    x1 = pt.make_placeholder("x1", (10, 1, 10))
    x2 = pt.make_placeholder("x2", (10, 10, 10))
    idx1 = pt.make_placeholder("idx1", (6, 1, 1), np.int32)
    idx2 = pt.make_placeholder("idx2", (6, 1), np.int32)
    idx3 = pt.make_placeholder("idx3", (6), np.int32)
    y1 = (x1 + x2)[idx1, idx2, idx3]
    y2 = x1[idx1, 0, idx3] + x2[idx1, idx2, idx3]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x1": rng.random((10, 1, 10)),
            "x2": rng.random((10, 10, 10)),
            "idx1": rng.integers(0, 10, (6, 1, 1), np.int32),
            "idx2": rng.integers(0, 10, (6, 1), np.int32),
            "idx3": rng.integers(0, 10, 6, np.int32),
        },
    )


def test_indirection_pusher_22(ctx_factory):
    x1 = pt.make_placeholder("x1", (10, 10))
    x2 = pt.make_placeholder("x2", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 1729, np.int32)
    y1 = (x1 + x2)[idx1, :, idx2, idx3]
    y2 = pt.expand_dims(x1[idx2, idx3], 1) + x2[idx1, :, idx2, idx3]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x1": rng.random((10, 10)),
            "x2": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 10, 1729, np.int32),
        },
    )


def test_indirection_pusher_23(ctx_factory):
    x1 = pt.make_placeholder("x1", (10, 10, 10))
    x2 = pt.make_placeholder("x2", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 1729, np.int32)
    y1 = (x1 + x2)[idx1, :, idx2, idx3]
    y2 = pt.transpose(x1[:, idx2, idx3], (1, 0)) + x2[idx1, :, idx2, idx3]
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x1": rng.random((10, 10, 10)),
            "x2": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 10, 1729, np.int32),
        },
    )


def test_indirection_pusher_24(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 314, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    y1 = x[:, :, idx1, :][:, idx2, :, idx3]
    y2 = pt.transpose(
        x[:, pt.expand_dims(idx2, 1), idx1, pt.expand_dims(idx3, 1)], (1, 0, 2)
    )
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 314, np.int32),
            "idx3": rng.integers(0, 10, 314, np.int32),
        },
    )


def test_indirection_pusher_25(ctx_factory):
    x1 = pt.make_placeholder("x1", (10, 10, 10))
    x2 = pt.make_placeholder("x2", (10, 10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 1729, np.int32)
    y1 = (x1 + x2)[:, idx1, :, idx2, idx3]
    y2 = (
        pt.transpose(pt.expand_dims(x1[:, idx2, idx3], 2), (1, 2, 0))
        + x2[:, idx1, :, idx2, idx3]
    )
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x1": rng.random((10, 10, 10)),
            "x2": rng.random((10, 10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, 1729, np.int32),
            "idx2": rng.integers(0, 10, 1729, np.int32),
            "idx3": rng.integers(0, 10, 1729, np.int32),
        },
    )


def test_indirection_pusher_26(ctx_factory):
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", (4, 1, 4), np.int32)
    idx2 = pt.make_placeholder("idx2", (10, 4), np.int32)
    idx3 = pt.make_placeholder("idx3", (10, 4), np.int32)
    idx4 = pt.make_placeholder("idx4", (10, 1), np.int32)
    idx5 = pt.make_placeholder("idx5", (10, 10), np.int32)
    y1 = x[:, idx1, idx2, idx3][:, idx4, 2:5, idx5]
    # In the computation of y1.
    # tmp1[_0, _1, _2, _3] = x[_0, idx1[_1, 0, _3], idx2[_2, _3], idx3[_2, _3]]
    # y1[_0, _1, _2, _3] = tmp1[_2, idx4[_0, 0], _3+2, idx5[_0, _1]]
    # Net
    # y1[_0, _1, _2, _3] =
    #  x[_2,
    #    idx1[idx4[_0, 0], 0, idx5[_0, _1]],
    #    idx2[_3+2, idx5[_0, _1]],
    #    idx3[_3 +2, idx5[_0, _1]]]
    y2 = pt.transpose(
        x[
            :,
            idx1[idx4, :, idx5],
            pt.transpose(idx2[2:5, idx5], (1, 2, 0)),
            pt.transpose(idx3[2:5, idx5], (1, 2, 0)),
        ],
        (1, 2, 0, 3),
    )
    assert pt.push_index_to_materialized_nodes(y1) == y2

    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)
    assert_only_materialized_indexees(y2)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y1)

    assert_allclose_to_ref(
        y2,
        y1,
        cq,
        {
            "x": rng.random((10, 10, 10, 10)),
            "idx1": rng.integers(0, 10, (4, 1, 4), np.int32),
            "idx2": rng.integers(0, 10, (10, 4), np.int32),
            "idx3": rng.integers(0, 10, (10, 4), np.int32),
            "idx4": rng.integers(0, 4, (10, 1), np.int32),
            "idx5": rng.integers(0, 4, (10, 10), np.int32),
        },
    )


def dgfem_flux(
    u,
    map_,
    map_0,
    map_1,
    map_2,
    map_3,
    map_4,
    map_5,
    map_6,
    map_7,
    map_8,
    map_9,
    map_10,
    map_11,
    map_12,
    map_13,
):
    tmp_3 = u[map_.reshape((192, 1)), map_0[map_1]]
    tmp_2 = tmp_3 - tmp_3
    tmp_1 = tmp_2[map_2.reshape((1536, 1)), map_3[map_4]]
    tmp_11 = u[map_5.reshape((1344, 1)), map_6[map_7]]
    tmp_9 = tmp_11[map_8.reshape((1344, 1)), map_9[map_10]] - tmp_11
    tmp_8 = tmp_9[map_11.reshape((1536, 1)), map_12[map_13]]
    tmp_0 = tmp_1 + tmp_8
    return tmp_0.reshape((4, 384, 15))


def test_indirection_pusher_27(ctx_factory):
    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)

    dgfem_flux_inputs = {
        "u": rng.random((384, 35), np.float64),
        "map_": rng.integers(0, 384, 192, np.int32),
        "map_0": rng.integers(0, 35, (4, 15), np.int64),
        "map_1": rng.integers(0, 4, 192, np.int8),
        "map_2": rng.integers(0, 192, 1536, np.int32),
        "map_3": rng.integers(0, 15, (1, 15), np.int64),
        "map_4": rng.integers(0, 1, 1536, np.int8),
        "map_5": rng.integers(0, 1536, 1344, np.int32),
        "map_6": rng.integers(0, 35, (4, 15), np.int64),
        "map_7": rng.integers(0, 4, 1344, np.int8),
        "map_8": rng.integers(0, 1344, 1344, np.int32),
        "map_9": rng.integers(0, 15, (3, 15), np.int64),
        "map_10": rng.integers(0, 3, 1344, np.int8),
        "map_11": rng.integers(0, 1344, 1536, np.int32),
        "map_12": rng.integers(0, 15, (1, 15), np.int64),
        "map_13": rng.integers(0, 1, 1536, np.int8),
    }

    y = dgfem_flux(
        **{
            k: pt.make_placeholder(k, v.shape, v.dtype)
            for k, v in dgfem_flux_inputs.items()
        }
    )

    y_prime = pt.push_index_to_materialized_nodes(y)

    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(y, y_prime, cq, dgfem_flux_inputs)


def wave_3d_p4_all_fluxes(
    *,
    u,
    v_0,
    v_1,
    v_2,
    map_,
    map_0,
    map_1,
    map_2,
    map_3,
    map_4,
    map_5,
    map_6,
    map_7,
    map_8,
    map_9,
    map_10,
    map_11,
    map_12,
    map_13,
    map_14,
    map_15,
    map_16,
    map_17,
    map_18,
    map_19,
    map_20,
    map_21,
):
    tmp_4 = pt.reshape(map_, (1536, 1))
    tmp_13 = pt.reshape(map_0, (192, 1))
    tmp_14 = map_1[map_2,]
    tmp_12 = u[tmp_13, tmp_14]
    tmp_11 = 0.0 + tmp_12
    tmp_15 = -1.0 * tmp_11
    tmp_10 = tmp_11 + tmp_15
    tmp_9 = 0.5 * tmp_10
    tmp_8 = tmp_9 * map_3
    tmp_23 = v_0[tmp_13, tmp_14]
    tmp_22 = 0.0 + tmp_23
    tmp_21 = tmp_22 - tmp_22
    tmp_20 = tmp_21 * map_4
    tmp_27 = v_1[tmp_13, tmp_14]
    tmp_26 = 0.0 + tmp_27
    tmp_25 = tmp_26 - tmp_26
    tmp_24 = tmp_25 * map_3
    tmp_19 = tmp_20 + tmp_24
    tmp_31 = v_2[tmp_13, tmp_14]
    tmp_30 = 0.0 + tmp_31
    tmp_29 = tmp_30 - tmp_30
    tmp_28 = tmp_29 * map_5
    tmp_18 = tmp_19 + tmp_28
    tmp_17 = 0.5 * tmp_18
    tmp_16 = tmp_17 * map_3
    tmp_7 = tmp_8 + tmp_16
    tmp_6 = 1.0 * tmp_7
    tmp_32 = pt.reshape(map_6, (1536, 1))
    tmp_33 = map_7[map_8,]
    tmp_5 = tmp_6[tmp_32, tmp_33]
    tmp_3 = pt.where(tmp_4, tmp_5, 0)
    tmp_2 = 0.0 + tmp_3
    tmp_37 = pt.reshape(map_9, (1536, 1))
    tmp_46 = pt.reshape(map_10, (1344, 1))
    tmp_47 = map_11[map_12,]
    tmp_45 = u[tmp_46, tmp_47]
    tmp_44 = 0.0 + tmp_45
    tmp_50 = pt.reshape(map_13, (1344, 1))
    tmp_51 = map_14[map_15,]
    tmp_49 = tmp_44[tmp_50, tmp_51]
    tmp_48 = 0.0 + tmp_49
    tmp_43 = tmp_44 + tmp_48
    tmp_42 = 0.5 * tmp_43
    tmp_41 = tmp_42 * map_16
    tmp_61 = v_0[tmp_46, tmp_47]
    tmp_60 = 0.0 + tmp_61
    tmp_59 = tmp_60[tmp_50, tmp_51]
    tmp_58 = 0.0 + tmp_59
    tmp_57 = tmp_58 - tmp_60
    tmp_56 = tmp_57 * map_17
    tmp_67 = v_1[tmp_46, tmp_47]
    tmp_66 = 0.0 + tmp_67
    tmp_65 = tmp_66[tmp_50, tmp_51]
    tmp_64 = 0.0 + tmp_65
    tmp_63 = tmp_64 - tmp_66
    tmp_62 = tmp_63 * map_16
    tmp_55 = tmp_56 + tmp_62
    tmp_73 = v_2[tmp_46, tmp_47]
    tmp_72 = 0.0 + tmp_73
    tmp_71 = tmp_72[tmp_50, tmp_51]
    tmp_70 = 0.0 + tmp_71
    tmp_69 = tmp_70 - tmp_72
    tmp_68 = tmp_69 * map_18
    tmp_54 = tmp_55 + tmp_68
    tmp_53 = 0.5 * tmp_54
    tmp_52 = tmp_53 * map_16
    tmp_40 = tmp_41 + tmp_52
    tmp_39 = 1.0 * tmp_40
    tmp_74 = pt.reshape(map_19, (1536, 1))
    tmp_75 = map_20[map_21,]
    tmp_38 = tmp_39[tmp_74, tmp_75]
    tmp_36 = pt.where(tmp_37, tmp_38, 0)
    tmp_35 = 0.0 + tmp_36
    tmp_34 = 0.0 + tmp_35
    tmp_1 = tmp_2 + tmp_34
    tmp_0 = pt.reshape(tmp_1, (4, 384, 15))
    tmp_87 = tmp_22 + tmp_22
    tmp_86 = 0.5 * tmp_87
    tmp_85 = tmp_86 * map_4
    tmp_90 = tmp_26 + tmp_26
    tmp_89 = 0.5 * tmp_90
    tmp_88 = tmp_89 * map_3
    tmp_84 = tmp_85 + tmp_88
    tmp_93 = tmp_30 + tmp_30
    tmp_92 = 0.5 * tmp_93
    tmp_91 = tmp_92 * map_5
    tmp_83 = tmp_84 + tmp_91
    tmp_95 = tmp_15 - tmp_11
    tmp_94 = 0.5 * tmp_95
    tmp_82 = tmp_83 + tmp_94
    tmp_81 = 1.0 * tmp_82
    tmp_80 = tmp_81[tmp_32, tmp_33]
    tmp_79 = pt.where(tmp_4, tmp_80, 0)
    tmp_78 = 0.0 + tmp_79
    tmp_106 = tmp_60 + tmp_58
    tmp_105 = 0.5 * tmp_106
    tmp_104 = tmp_105 * map_17
    tmp_109 = tmp_66 + tmp_64
    tmp_108 = 0.5 * tmp_109
    tmp_107 = tmp_108 * map_16
    tmp_103 = tmp_104 + tmp_107
    tmp_112 = tmp_72 + tmp_70
    tmp_111 = 0.5 * tmp_112
    tmp_110 = tmp_111 * map_18
    tmp_102 = tmp_103 + tmp_110
    tmp_114 = tmp_48 - tmp_44
    tmp_113 = 0.5 * tmp_114
    tmp_101 = tmp_102 + tmp_113
    tmp_100 = 1.0 * tmp_101
    tmp_99 = tmp_100[tmp_74, tmp_75]
    tmp_98 = pt.where(tmp_37, tmp_99, 0)
    tmp_97 = 0.0 + tmp_98
    tmp_96 = 0.0 + tmp_97
    tmp_77 = tmp_78 + tmp_96
    tmp_76 = pt.reshape(tmp_77, (4, 384, 15))
    tmp_122 = tmp_9 * map_5
    tmp_123 = tmp_17 * map_5
    tmp_121 = tmp_122 + tmp_123
    tmp_120 = 1.0 * tmp_121
    tmp_119 = tmp_120[tmp_32, tmp_33]
    tmp_118 = pt.where(tmp_4, tmp_119, 0)
    tmp_117 = 0.0 + tmp_118
    tmp_130 = tmp_42 * map_18
    tmp_131 = tmp_53 * map_18
    tmp_129 = tmp_130 + tmp_131
    tmp_128 = 1.0 * tmp_129
    tmp_127 = tmp_128[tmp_74, tmp_75]
    tmp_126 = pt.where(tmp_37, tmp_127, 0)
    tmp_125 = 0.0 + tmp_126
    tmp_124 = 0.0 + tmp_125
    tmp_116 = tmp_117 + tmp_124
    tmp_115 = pt.reshape(tmp_116, (4, 384, 15))
    tmp_139 = tmp_9 * map_4
    tmp_140 = tmp_17 * map_4
    tmp_138 = tmp_139 + tmp_140
    tmp_137 = 1.0 * tmp_138
    tmp_136 = tmp_137[tmp_32, tmp_33]
    tmp_135 = pt.where(tmp_4, tmp_136, 0)
    tmp_134 = 0.0 + tmp_135
    tmp_147 = tmp_42 * map_17
    tmp_148 = tmp_53 * map_17
    tmp_146 = tmp_147 + tmp_148
    tmp_145 = 1.0 * tmp_146
    tmp_144 = tmp_145[tmp_74, tmp_75]
    tmp_143 = pt.where(tmp_37, tmp_144, 0)
    tmp_142 = 0.0 + tmp_143
    tmp_141 = 0.0 + tmp_142
    tmp_133 = tmp_134 + tmp_141
    tmp_132 = pt.reshape(tmp_133, (4, 384, 15))
    return {"flux_0": tmp_0, "flux_1": tmp_76, "flux_2": tmp_115, "flux_3": tmp_132}


def test_indirection_pusher_28(ctx_factory):
    rng = default_rng(42)
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)

    np_fields = {
        "u": rng.random((384, 35)),
        "v_0": rng.random((384, 35)),
        "v_1": rng.random((384, 35)),
        "v_2": rng.random((384, 35)),
    }
    np_indirections = {
        "map_": rng.integers(0, 2, 1536, np.int8),
        "map_0": rng.integers(0, 384, 192, np.int32),
        "map_1": rng.integers(0, 35, (4, 15), np.int64),
        "map_2": rng.integers(0, 4, 192, np.int8),
        "map_3": rng.random((192, 1)),
        "map_4": rng.random((192, 1)),
        "map_5": rng.random((192, 1)),
        "map_6": rng.integers(0, 192, 1536, np.int32),
        "map_7": rng.integers(0, 15, (1, 15), np.int64),
        "map_8": rng.integers(0, 1, 1536, np.int8),
        "map_9": rng.integers(0, 2, 1536, np.int8),
        "map_10": rng.integers(0, 384, 1344, np.int32),
        "map_11": rng.integers(0, 35, (4, 15), np.int64),
        "map_12": rng.integers(0, 4, 1344, np.int8),
        "map_13": rng.integers(0, 1344, 1344, np.int32),
        "map_14": rng.integers(0, 15, (3, 15), np.int64),
        "map_15": rng.integers(0, 3, 1344, np.int8),
        "map_16": rng.random((1344, 1)),
        "map_17": rng.random((1344, 1)),
        "map_18": rng.random((1344, 1)),
        "map_19": rng.integers(0, 1344, 1536, np.int32),
        "map_20": rng.integers(0, 15, (1, 15), np.int64),
        "map_21": rng.integers(0, 1, 1536, np.int8),
    }
    pt_fields = {
        name: pt.make_placeholder(name, np_field.shape, np_field.dtype)
        for name, np_field in np_fields.items()
    }
    pt_indirections = {
        name: pt.make_data_wrapper(np_indirection)
        for name, np_indirection in np_indirections.items()
    }

    y = pt.make_dict_of_named_arrays(
        wave_3d_p4_all_fluxes(**pt_fields, **pt_indirections)
    )

    y_prime = pt.push_index_to_materialized_nodes(y)

    assert_only_materialized_indexees(y_prime)
    with pytest.raises(AssertionError):
        assert_only_materialized_indexees(y)

    assert_allclose_to_ref(y, y_prime, cq, np_fields)
