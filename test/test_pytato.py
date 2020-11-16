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

import sys

import numpy as np
import pytest

import pytato as pt


def test_matmul_input_validation():
    namespace = pt.Namespace()

    a = pt.make_placeholder(namespace, name="a", shape=(10, 10), dtype=np.float)
    b = pt.make_placeholder(namespace, name="b", shape=(20, 10), dtype=np.float)

    with pytest.raises(ValueError):
        a @ b

    c = pt.make_placeholder(namespace, name="c", shape=(), dtype=np.float)
    with pytest.raises(ValueError):
        c @ c

    pt.make_size_param(namespace, "n")
    d = pt.make_placeholder(namespace, name="d", shape="(n, n)", dtype=np.float)
    d @ d


def test_roll_input_validation():
    namespace = pt.Namespace()

    a = pt.make_placeholder(namespace, name="a", shape=(10, 10), dtype=np.float)
    pt.roll(a, 1, axis=0)

    with pytest.raises(ValueError):
        pt.roll(a, 1, axis=2)

    with pytest.raises(ValueError):
        pt.roll(a, 1, axis=-1)


def test_transpose_input_validation():
    namespace = pt.Namespace()

    a = pt.make_placeholder(namespace, name="a", shape=(10, 10), dtype=np.float)
    pt.transpose(a)

    with pytest.raises(ValueError):
        pt.transpose(a, (2, 0, 1))

    with pytest.raises(ValueError):
        pt.transpose(a, (1, 1))

    with pytest.raises(ValueError):
        pt.transpose(a, (0,))


def test_slice_input_validation():
    namespace = pt.Namespace()

    a = pt.make_placeholder(namespace, name="a", shape=(10, 10, 10), dtype=np.float)

    a[0]
    a[0, 0]
    a[0, 0, 0]

    with pytest.raises(ValueError):
        a[0, 0, 0, 0]

    with pytest.raises(ValueError):
        a[10]


def test_stack_input_validation():
    namespace = pt.Namespace()

    x = pt.make_placeholder(namespace, name="x", shape=(10, 10), dtype=np.float)
    y = pt.make_placeholder(namespace, name="y", shape=(1, 10), dtype=np.float)

    assert pt.stack((x, x, x), axis=0).shape == (3, 10, 10)

    pt.stack((x,), axis=0)
    pt.stack((x,), axis=1)

    with pytest.raises(ValueError):
        pt.stack(())

    with pytest.raises(ValueError):
        pt.stack((x, y))

    with pytest.raises(ValueError):
        pt.stack((x, x), axis=3)


def test_make_placeholder_noname():
    ns = pt.Namespace()
    x = pt.make_placeholder(ns, shape=(10, 4), dtype=float)
    y = 2*x

    knl = pt.generate_loopy(y).program

    assert x.name in knl.arg_dict
    assert x.name in knl.get_read_variables()


def test_zero_length_arrays():
    ns = pt.Namespace()
    x = pt.make_placeholder(ns, shape=(0, 4), dtype=float)
    y = 2*x

    assert y.shape == (0, 4)

    knl = pt.generate_loopy(y).program
    assert all(dom.is_empty() for dom in knl.domains if dom.total_dim() != 0)


def test_concatenate_input_validation():
    namespace = pt.Namespace()

    x = pt.make_placeholder(namespace, name="x", shape=(10, 10), dtype=np.float)
    y = pt.make_placeholder(namespace, name="y", shape=(1, 10), dtype=np.float)

    assert pt.concatenate((x, x, x), axis=0).shape == (30, 10)
    assert pt.concatenate((x, y), axis=0).shape == (11, 10)

    pt.concatenate((x,), axis=0)
    pt.concatenate((x,), axis=1)

    with pytest.raises(ValueError):
        pt.concatenate(())

    with pytest.raises(ValueError):
        pt.concatenate((x, y), axis=1)

    with pytest.raises(ValueError):
        pt.concatenate((x, x), axis=3)


def test_reshape_input_validation():
    ns = pt.Namespace()

    x = pt.make_placeholder(ns, shape=(3, 3, 4), dtype=np.float)

    assert pt.reshape(x, (-1,)).shape == (36,)
    assert pt.reshape(x, (-1, 6)).shape == (6, 6)
    assert pt.reshape(x, (4, -1)).shape == (4, 9)
    assert pt.reshape(x, (36, -1)).shape == (36, 1)

    with pytest.raises(ValueError):
        # 36 not a multiple of 25
        pt.reshape(x, (5, 5))

    with pytest.raises(ValueError):
        # 2 unknown dimensions
        pt.reshape(x, (-1, -1, 3))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
