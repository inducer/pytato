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

import pytato as pt


def test_indirection_pusher_0():
    x = pt.make_placeholder("x", 10)
    idx = pt.make_placeholder("idx", 1729, np.int32)
    y = x[idx]
    assert pt.push_index_to_materialized_nodes(y) == y


def test_indirection_pusher_1():
    x = pt.make_placeholder("x", 10)
    idx = pt.make_placeholder("idx", 1729, np.int32)
    y = (2 * x)[idx]
    assert pt.push_index_to_materialized_nodes(y) == 2 * (x[idx])


def test_indirection_pusher_2():
    x1 = pt.make_placeholder("x1", 10)
    x2 = pt.make_placeholder("x2", 10)
    idx = pt.make_placeholder("idx", 1729, np.int32)
    y = (x1 * x2)[idx]
    assert pt.push_index_to_materialized_nodes(y) == (x1[idx] + x2[idx])


def test_indirection_pusher_3():
    x = pt.make_placeholder("x", 10)
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 314, np.int32)
    assert pt.push_index_to_materialized_nodes(x[idx1][idx2]) == x[idx1[idx2]]


def test_indirection_pusher_4():
    x = pt.make_placeholder("x", 10)
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 314, np.int32)
    assert pt.push_index_to_materialized_nodes((2 * x[idx1])[idx2]) == 2 * x[idx1[idx2]]


def test_indirection_pusher_5():
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    idx5 = pt.make_placeholder("idx5", 314, np.int32)

    assert (
        pt.push_index_to_materialized_nodes(x[:, idx1, idx2, :][idx3, idx4, idx5])
        == x[idx3, idx1[idx4], idx2[idx4], idx5]
    )


def test_indirection_pusher_6():
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    idx5 = pt.make_placeholder("idx5", 314, np.int32)

    assert (
        pt.push_index_to_materialized_nodes(x[::2, idx1, idx2, ::3][idx3, idx4, idx5])
        == x[2 * idx3, idx1[idx4], idx2[idx4], 3 * idx5]
    )


def test_indirection_pusher_7():
    x = pt.make_placeholder("x", (10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx1", 314, np.int32)
    idx4 = pt.make_placeholder("idx2", 314, np.int32)

    assert (
        pt.push_index_to_materialized_nodes(x[idx1, :, idx2][idx3, idx4])
        == x[idx1[idx3], idx4, idx2[idx3]]
    )


def test_indirection_pusher_8():
    x = pt.make_placeholder("x", (10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx1", 314, np.int32)
    idx4 = pt.make_placeholder("idx2", 314, np.int32)

    assert (
        pt.push_index_to_materialized_nodes(x[idx1, ::2, idx2][idx3, idx4])
        == x[idx1[idx3], 2 * idx4, idx2[idx3]]
    )


def test_indirection_pusher_9():
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)

    assert (
        pt.push_index_to_materialized_nodes(x[idx1, idx2, ::2, ::3][idx3, :, idx4])
        == x[idx1[idx3], idx2[idx3], ::2, 3 * idx4]
    )


def test_indirection_pusher_10():
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx3", 314, np.int32)
    idx4 = pt.make_placeholder("idx4", 314, np.int32)
    # (_0, _1, _2) -> (idx1[_0], 2*_1, idx2[_0], _2)
    # (_0, _1)     -> (idx3[_0], 3*_1, idx4[_0])
    # Net:
    # (_0, _1) -> (idx1[idx3[_0]], 6*_1, idx2[idx3[_0]], idx4[_0])

    assert (
        pt.push_index_to_materialized_nodes(x[idx1, ::2, idx2][idx3, ::3, idx4])
        == x[idx1[idx3], ::6, idx2[idx3], idx4]
    )


def test_indirection_pusher_11():
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


def test_indirection_pusher_12():
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


@pytest.mark.xfail("axis permutation not yet supported.")
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


@pytest.mark.xfail("axis permutation not yet supported.")
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


def test_indirection_pusher_15():
    x = pt.make_placeholder("x", (10, 10))
    idx1 = pt.make_placeholder("idx1", 4, np.int32)
    idx2 = pt.make_placeholder("idx2", (10, 4), np.int32)
    idx3 = pt.make_placeholder("idx3", (1, 10, 10), np.int32)
    idx4 = pt.make_placeholder("idx4", (10, 10, 10), np.int32)
    assert (
        pt.push_index_to_materialized_nodes(x[idx1, idx2][idx3, idx4])
        == x[idx1[idx4], idx2[idx3]]
    )


def test_indirection_pusher_16():
    x = pt.make_placeholder("x", (10, 10, 10))
    idx1 = pt.make_placeholder("idx1", (4, 1, 4), np.int32)
    idx2 = pt.make_placeholder("idx2", (10, 4), np.int32)
    idx3 = pt.make_placeholder("idx3", (10, 4), np.int32)
    idx4 = pt.make_placeholder("idx4", (10, 1), np.int32)
    idx5 = pt.make_placeholder("idx5", (10, 10), np.int32)
    assert (
        pt.push_index_to_materialized_nodes(x[idx1, idx2, idx3][idx4, 2:5, idx5])
        == x[idx1[idx4, :, idx5], idx2[2:5, idx5], idx3[idx5]]
    )


def test_indirection_pusher_17():
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx1", 314, np.int32)
    idx4 = pt.make_placeholder("idx2", 314, np.int32)
    y1 = x[:, idx1, :, idx2][:, idx3, idx4]
    y2 = x[idx3, idx1.reshape(-1, 1), idx4, idx2.reshape(-1, 1)]
    assert pt.push_index_to_materialized_nodes(y1) == y2


def test_indirection_pusher_18():
    x = pt.make_placeholder("x", (10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx1", 314, np.int32)
    idx4 = pt.make_placeholder("idx2", 314, np.int32)
    y1 = x[:, idx1, idx2, :][:, idx3, idx4]
    y2 = x[:, idx1[idx3], idx2[idx3], idx4]
    assert pt.push_index_to_materialized_nodes(y1) == y2


def test_indirection_pusher_19():
    x = pt.make_placeholder("x", (10, 10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", 1729, np.int32)
    idx2 = pt.make_placeholder("idx2", 1729, np.int32)
    idx3 = pt.make_placeholder("idx1", 314, np.int32)
    idx4 = pt.make_placeholder("idx2", 314, np.int32)
    y1 = x[:, idx1, :, idx2, :][:, :, idx3, idx4]
    y2 = pt.transpose(
        (1, 0, 2), x[:, idx1.reshape(-1, 1), idx3, idx2.reshape(-1, 1), idx4]
    )
    assert pt.push_index_to_materialized_nodes(y1) == y2


def test_indirection_pusher_20():
    x = pt.make_placeholder("x", (10, 10, 10, 10))
    idx1 = pt.make_placeholder("idx1", (2718, 1729, 314), np.int32)
    idx2 = pt.make_placeholder("idx2", (1729, 314), np.int32)
    idx3 = pt.make_placeholder("idx3", 6, np.int32)
    idx4 = pt.make_placeholder("idx4", (10, 6), np.int32)
    y1 = x[:, idx1, :, idx2][:, :, :, idx3, idx4]
    y2 = x[idx3, pt.expand_dims(idx1, (3, 4)), idx4, pt.expand_dims(idx2, (2, 3))]
    assert pt.push_index_to_materialized_nodes(y1) == y2
