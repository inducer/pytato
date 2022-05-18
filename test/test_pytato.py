#!/usr/bin/env python

__copyright__ = """Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2021 Kaushik Kulkarni
Copyright (C) 2021 University of Illinois Board of Trustees
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

from typing import cast

import numpy as np
import pytest

import pytato as pt

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


def test_matmul_input_validation():
    a = pt.make_placeholder(name="a", shape=(10, 10), dtype=np.float64)
    b = pt.make_placeholder(name="b", shape=(20, 10), dtype=np.float64)

    with pytest.raises(ValueError):
        a @ b

    c = pt.make_placeholder(name="c", shape=(), dtype=np.float64)
    with pytest.raises(ValueError):
        c @ c

    n = pt.make_size_param("n")
    d = pt.make_placeholder(name="d", shape=(n, n), dtype=np.float64)
    d @ d


def test_roll_input_validation():
    a = pt.make_placeholder(name="a", shape=(10, 10), dtype=np.float64)
    pt.roll(a, 1, axis=0)

    with pytest.raises(ValueError):
        pt.roll(a, 1, axis=2)

    with pytest.raises(ValueError):
        pt.roll(a, 1, axis=-1)


def test_transpose_input_validation():
    a = pt.make_placeholder(name="a", shape=(10, 10), dtype=np.float64)
    pt.transpose(a)

    with pytest.raises(ValueError):
        pt.transpose(a, (2, 0, 1))

    with pytest.raises(ValueError):
        pt.transpose(a, (1, 1))

    with pytest.raises(ValueError):
        pt.transpose(a, (0,))


def test_slice_input_validation():
    a = pt.make_placeholder(name="a", shape=(10, 10, 10), dtype=np.float64)

    a[0]
    a[0, 0]
    a[0, 0, 0]

    with pytest.raises(IndexError):
        a[0, 0, 0, 0]

    with pytest.raises(IndexError):
        a[10]


def test_index_type_validation():
    a = pt.make_placeholder(name="a", shape=(10,), dtype=np.float64)

    idx = pt.make_placeholder(name="idx", shape=(5,), dtype=np.int8)
    a[idx]

    idx = pt.make_placeholder(name="idx", shape=(5,), dtype=np.uint8)
    a[idx]

    idx = pt.make_placeholder(name="idx", shape=(5,), dtype=np.float64)
    with pytest.raises(IndexError):
        a[idx]


def test_stack_input_validation():
    x = pt.make_placeholder(name="x", shape=(10, 10), dtype=np.float64)
    y = pt.make_placeholder(name="y", shape=(1, 10), dtype=np.float64)

    assert pt.stack((x, x, x), axis=0).shape == (3, 10, 10)

    pt.stack((x,), axis=0)
    pt.stack((x,), axis=1)

    with pytest.raises(ValueError):
        pt.stack(())

    with pytest.raises(ValueError):
        pt.stack((x, y))

    with pytest.raises(ValueError):
        pt.stack((x, x), axis=3)


@pytest.mark.xfail  # Unnamed placeholders should be used via pt.bind
def test_make_placeholder_noname():
    x = pt.make_placeholder("x", shape=(10, 4), dtype=float)
    y = 2*x

    knl = pt.generate_loopy(y).kernel

    assert x.name in knl.arg_dict
    assert x.name in knl.get_read_variables()


def test_zero_length_arrays():
    x = pt.make_placeholder("x", shape=(0, 4), dtype=float)
    y = 2*x

    assert y.shape == (0, 4)

    knl = pt.generate_loopy(y).kernel
    assert all(dom.is_empty() for dom in knl.domains if dom.total_dim() != 0)


def test_concatenate_input_validation():
    x = pt.make_placeholder(name="x", shape=(10, 10), dtype=np.float64)
    y = pt.make_placeholder(name="y", shape=(1, 10), dtype=np.float64)

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
    x = pt.make_placeholder("x", shape=(3, 3, 4), dtype=np.float64)

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

    # Reporter by alexfikl
    # See https://github.com/inducer/pytato/issues/157
    x = pt.make_placeholder("x", shape=(0,), dtype=np.float64)
    assert pt.reshape(x, (128, 0, 17)).shape == (128, 0, 17)


def test_binary_op_dispatch():
    class Foo:
        def __add__(self, other):
            if isinstance(other, pt.Array):
                return "bar"

            return NotImplemented

        def __radd__(self, other):
            if isinstance(other, pt.Array):
                return "baz"

            return NotImplemented

    x = pt.make_placeholder(name="x", shape=(10,), dtype=float)
    assert Foo() + x == "bar"
    assert x + Foo() == "baz"


def test_same_placeholder_name_raises():
    from pytato.diagnostic import NameClashError
    x = pt.make_placeholder(name="arr", shape=(10, 4), dtype=float)
    y = pt.make_placeholder(name="arr", shape=(10, 4), dtype=float)

    with pytest.raises(NameClashError):
        pt.generate_loopy(x+y)

    n1 = pt.make_size_param("n")
    n2 = pt.make_size_param("n")
    x = pt.make_placeholder(name="arr", shape=(n1, n2), dtype=float)
    with pytest.raises(NameClashError):
        pt.generate_loopy(2*x)


def test_einsum_error_handling():
    with pytest.raises(ValueError):
        # operands not enough
        pt.einsum("ij,j->j", pt.make_placeholder("x", (2, 2), float))

    with pytest.raises(ValueError):
        # double index use in the out spec.
        pt.einsum("ij,j->jj", ("a", "b"))


def test_accessing_dict_of_named_arrays_validation():
    x = pt.make_placeholder(name="x", shape=10, dtype=float)
    y1y2 = pt.make_dict_of_named_arrays({"y1": 2*x, "y2": 3*x})

    assert isinstance(y1y2["y1"], pt.array.NamedArray)
    assert y1y2["y1"].shape == (2*x).shape
    assert y1y2["y1"].dtype == (2*x).dtype


def test_call_loopy_shape_inference():
    from pytato.loopy import call_loopy
    from pytato.utils import are_shapes_equal
    import loopy as lp

    knl = lp.make_kernel(
            ["{[i, j]: 0<=i<(2*n + 3*m + 2) and 0<=j<(6*n + 4*m + 3)}",
             "{[ii, jj]: 0<=ii<m and 0<=jj<n}"],
            """
            <> tmp = sum([i, j], A[i, j])
            out[ii, jj] = tmp*(ii + jj)
            """, lang_version=(2018, 2))

    # {{{ variant 1

    A = pt.make_placeholder(name="x", shape=(20, 37), dtype=np.float64)  # noqa: N806
    y = call_loopy(knl, {"A": A})["out"]
    assert are_shapes_equal(y.shape, (4, 3))

    # }}}

    # {{{ variant 2

    n1 = pt.make_size_param("n1")
    n2 = pt.make_size_param("n2")
    A = pt.make_placeholder(name="x",  # noqa: N806
                            shape=(4*n1 + 6*n2 + 2, 12*n1 + 8*n2 + 3),
                            dtype=np.float64)

    y = call_loopy(knl, {"A": A})["out"]
    assert are_shapes_equal(y.shape, (2*n2, 2*n1))

    # }}}


def test_tagging_array():
    from pytools.tag import Tag

    class BestArrayTag(Tag):
        """
        Best array known to humankind.
        """

    x = pt.make_placeholder(shape=(42, 1729), dtype=float, name="x")
    y = x.tagged(BestArrayTag())
    assert any(isinstance(tag, BestArrayTag) for tag in y.tags)


def test_dict_of_named_arrays_comparison():
    # See https://github.com/inducer/pytato/pull/137
    x = pt.make_placeholder("x", (10, 4), float)
    dict1 = pt.make_dict_of_named_arrays({"out": 2 * x})
    dict2 = pt.make_dict_of_named_arrays({"out": 2 * x})
    dict3 = pt.make_dict_of_named_arrays({"not_out": 2 * x})
    dict4 = pt.make_dict_of_named_arrays({"out": 3 * x})
    assert dict1 == dict2
    assert dict1 != dict3
    assert dict1 != dict4


def test_toposortmapper():
    n = pt.make_size_param("n")
    array = pt.make_placeholder(name="array", shape=n, dtype=np.float64)
    stack = pt.stack([array, 2*array, array + 6])
    y = stack @ stack.T

    tm = pt.transform.TopoSortMapper()
    tm(y)

    from pytato.array import (AxisPermutation, IndexLambda,
                              Placeholder, Einsum, SizeParam, Stack)

    assert isinstance(tm.topological_order[0], SizeParam)
    assert isinstance(tm.topological_order[1], Placeholder)
    assert isinstance(tm.topological_order[2], IndexLambda)
    assert isinstance(tm.topological_order[3], IndexLambda)
    assert isinstance(tm.topological_order[4], Stack)
    assert isinstance(tm.topological_order[5], AxisPermutation)
    assert isinstance(tm.topological_order[6], Einsum)


def test_userscollector():
    from testlib import RandomDAGContext, make_random_dag
    from pytato.transform import UsersCollector, reverse_graph

    # Check that nodes without users are correctly reversed
    array = pt.make_placeholder(name="array", shape=1, dtype=np.int64)
    y = array+1

    uc = UsersCollector()
    uc(y)
    rev_graph = reverse_graph(uc.node_to_users)
    rev_graph2 = reverse_graph(reverse_graph(rev_graph))

    assert reverse_graph(rev_graph2) == uc.node_to_users

    # Test random DAGs
    axis_len = 5

    for i in range(100):
        rdagc = RandomDAGContext(np.random.default_rng(seed=i),
                axis_len=axis_len, use_numpy=False)

        dag = make_random_dag(rdagc)

        uc = UsersCollector()
        uc(dag)

        rev_graph = reverse_graph(uc.node_to_users)
        rev_graph2 = reverse_graph(reverse_graph(rev_graph))

        assert rev_graph2 == rev_graph


def test_asciidag():
    n = pt.make_size_param("n")
    array = pt.make_placeholder(name="array", shape=n, dtype=np.float64)
    stack = pt.stack([array, 2*array, array + 6])
    y = stack @ stack.T

    from pytato import get_ascii_graph

    res = get_ascii_graph(y, use_color=False)

    ref_str = r"""* Inputs
*-.   Placeholder
|\ \
* | | IndexLambda
| |/
|/|
| * IndexLambda
|/
*   Stack
|\
* | AxisPermutation
|/
* Einsum
* Outputs
"""

    assert res == ref_str


def test_linear_complexity_inequality():
    # See https://github.com/inducer/pytato/issues/163
    import pytato as pt
    from pytato.equality import EqualityComparer
    from numpy.random import default_rng

    def construct_intestine_graph(depth=90, seed=0):
        rng = default_rng(seed)
        x = pt.make_placeholder("x", shape=(10,), dtype=float)

        for _ in range(depth):
            coeff1, coeff2 = rng.integers(0, 10, 2)
            x = coeff1 * x + coeff2 * x

        return x

    graph1 = construct_intestine_graph()
    graph2 = construct_intestine_graph()
    graph3 = construct_intestine_graph(seed=3)

    assert EqualityComparer()(graph1, graph2)
    assert EqualityComparer()(graph2, graph1)
    assert not EqualityComparer()(graph1, graph3)
    assert not EqualityComparer()(graph2, graph3)


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
def test_einsum_is_similar_to_subscript(spec, argshapes):
    operands = [pt.make_placeholder(name=f"arg_{iarg}",
                                    shape=argshape,
                                    dtype=np.int32)
                for iarg, argshape in enumerate(argshapes)]
    expr = pt.einsum(spec, *operands)
    assert pt.analysis.is_einsum_similar_to_subscript(expr, spec)


def test_array_dot_repr():
    x = pt.make_placeholder("x", (10, 4), np.int64)
    y = pt.make_placeholder("y", (10, 4), np.int64)

    def _assert_stripped_repr(ary: pt.Array, expected_repr: str):
        from pytato.transform import remove_tags_of_type
        from pytato.tags import CreatedAt
        ary = cast(pt.Array, remove_tags_of_type(CreatedAt, ary))

        expected_str = "".join([c for c in repr(ary) if c not in [" ", "\n"]])
        result_str = "".join([c for c in expected_repr if c not in [" ", "\n"]])
        assert expected_str == result_str

    _assert_stripped_repr(
        3*x + 4*y,
        """
IndexLambda(
    expr=Sum((Subscript(Variable('_in0'),
                        (Variable('_0'), Variable('_1'))),
              Subscript(Variable('_in1'),
                        (Variable('_0'), Variable('_1'))))),
    shape=(10, 4),
    dtype='int64',
    bindings={'_in0': IndexLambda(expr=Product((3, Subscript(Variable('_in1'),
                                                             (Variable('_0'),
                                                              Variable('_1'))))),
                                  shape=(10, 4),
                                  dtype='int64',
                                  bindings={'_in1': Placeholder(shape=(10, 4),
                                                                dtype='int64',
                                                                name='x')}),
              '_in1': IndexLambda(expr=Product((4, Subscript(Variable('_in1'),
                                                             (Variable('_0'),
                                                              Variable('_1'))))),
                                  shape=(10, 4),
                                  dtype='int64',
                                  bindings={'_in1': Placeholder(shape=(10, 4),
                                                                dtype='int64',
                                                                name='y')})})""")

    _assert_stripped_repr(
        pt.roll(x.reshape(2, 20).reshape(-1), 3),
        """
Roll(
    array=Reshape(array=Reshape(array=Placeholder(shape=(10, 4),
                                                  dtype='int64',
                                                  name='x'),
                                newshape=(2, 20),
                                order='C'),
                  newshape=(40),
                  order='C'),
    shift=3, axis=0)""")
    _assert_stripped_repr(y * pt.not_equal(x, 3),
                          """
IndexLambda(
    expr=Product((Subscript(Variable('_in0'),
                            (Variable('_0'), Variable('_1'))),
                  Subscript(Variable('_in1'),
                            (Variable('_0'), Variable('_1'))))),
    shape=(10, 4),
    dtype='int64',
    bindings={'_in0': Placeholder(shape=(10, 4), dtype='int64', name='y'),
              '_in1': IndexLambda(
                  expr=Comparison(Subscript(Variable('_in0'),
                                            (Variable('_0'), Variable('_1'))),
                                  '!=',
                                  3),
                  shape=(10, 4),
                  dtype=<class 'numpy.bool_'>,
                  bindings={'_in0': Placeholder(shape=(10, 4),
                                                dtype='int64',
                                                name='x')})})""")
    _assert_stripped_repr(
        x[y[:, 2:3], x[2, :]],
        """
AdvancedIndexInContiguousAxes(
    array=Placeholder(shape=(10, 4), dtype='int64', name='x'),
    indices=(BasicIndex(array=Placeholder(shape=(10, 4),
                                          dtype='int64',
                                          name='y'),
                        indices=(NormalizedSlice(start=0, stop=10, step=1),
                                 NormalizedSlice(start=2, stop=3, step=1))),
             BasicIndex(array=Placeholder(shape=(10, 4),
                                          dtype='int64',
                                          name='x'),
                        indices=(2, NormalizedSlice(start=0, stop=4, step=1)))))""")

    _assert_stripped_repr(
        pt.stack([x[y[:, 2:3], x[2, :]].T, y[x[:, 2:3], y[2, :]].T]),
        """
Stack(
    arrays=(
        AxisPermutation(
            array=AdvancedIndexInContiguousAxes(
                array=Placeholder(shape=(10, 4),
                                  dtype='int64',
                                  name='x'),
                indices=(BasicIndex(array=(...),
                                    indices=(NormalizedSlice(start=0,
                                                             stop=10,
                                                             step=1),
                                             NormalizedSlice(start=2,
                                                             stop=3,
                                                             step=1))),
                         BasicIndex(array=(...),
                                    indices=(2,
                                             NormalizedSlice(start=0,
                                                             stop=4,
                                                             step=1))))),
            axis_permutation=(1, 0)),
        AxisPermutation(array=AdvancedIndexInContiguousAxes(
            array=Placeholder(shape=(10,
                                     4),
                              dtype='int64',
                              name='y'),
            indices=(BasicIndex(array=(...),
                                indices=(NormalizedSlice(start=0,
                                                         stop=10,
                                                         step=1),
                                         NormalizedSlice(start=2,
                                                         stop=3,
                                                         step=1))),
                     BasicIndex(array=(...),
                                indices=(2,
                                         NormalizedSlice(start=0,
                                                         stop=4,
                                                         step=1))))),
                        axis_permutation=(1, 0))), axis=0)
    """)


def test_repr_array_is_deterministic():

    from testlib import RandomDAGContext, make_random_dag

    axis_len = 5
    for i in range(50):
        rdagc = RandomDAGContext(np.random.default_rng(seed=i),
                                 axis_len=axis_len, use_numpy=False)
        dag = make_random_dag(rdagc)
        assert repr(dag) == repr(dag)


def test_nodecountmapper():
    from testlib import RandomDAGContext, make_random_dag
    from pytato.analysis import get_num_nodes

    axis_len = 5

    for i in range(10):
        rdagc = RandomDAGContext(np.random.default_rng(seed=i),
                                 axis_len=axis_len, use_numpy=False)
        dag = make_random_dag(rdagc)

        # Subtract 1 since NodeCountMapper counts an extra one for DictOfNamedArrays.
        assert get_num_nodes(dag)-1 == len(pt.transform.DependencyMapper()(dag))


def test_rec_get_user_nodes():
    x1 = pt.make_placeholder("x1", shape=(10, 4), dtype=np.float64)
    x2 = pt.make_placeholder("x2", shape=(10, 4), dtype=np.float64)

    expr = pt.make_dict_of_named_arrays({"out1": 2 * x1,
                                         "out2": 7 * x1 + 3 * x2})

    assert (pt.transform.rec_get_user_nodes(expr, x1)
            == frozenset({2 * x1, 7*x1, 7*x1 + 3 * x2, expr}))
    assert (pt.transform.rec_get_user_nodes(expr, x2)
            == frozenset({3 * x2, 7*x1 + 3 * x2, expr}))


def test_rec_get_user_nodes_linear_complexity():

    def construct_intestine_graph(depth=100, seed=0):
        from numpy.random import default_rng
        rng = default_rng(seed)
        x = pt.make_placeholder("x", shape=(10,), dtype=float)
        y = x

        for _ in range(depth):
            coeff1, coeff2 = rng.integers(0, 10, 2)
            y = coeff1 * y + coeff2 * y

        return y, x

    expected_result = set()

    class SubexprRecorder(pt.transform.CachedWalkMapper):
        def post_visit(self, expr):
            if not isinstance(expr, pt.Placeholder):
                expected_result.add(expr)
            else:
                assert expr.name == "x"

    expr, inp = construct_intestine_graph()
    result = pt.transform.rec_get_user_nodes(expr, inp)
    SubexprRecorder()(expr)

    assert (expected_result == result)


def test_tag_user_nodes_linear_complexity():
    from numpy.random import default_rng

    def construct_intestine_graph(depth=90, seed=0):
        rng = default_rng(seed)
        x = pt.make_placeholder("x", shape=(10,), dtype=float)
        y = x

        for _ in range(depth):
            coeff1, coeff2 = rng.integers(0, 10, 2)
            y = coeff1 * y + coeff2 * y

        return y, x

    expr, inp = construct_intestine_graph()
    user_collector = pt.transform.UsersCollector()
    user_collector(expr)

    expected_result = {}

    class ExpectedResultComputer(pt.transform.CachedWalkMapper):
        def post_visit(self, expr):
            expected_result[expr] = {"foo"}

    expr, inp = construct_intestine_graph()
    result = pt.transform.tag_user_nodes(user_collector.node_to_users, "foo", inp)
    ExpectedResultComputer()(expr)

    assert expected_result == result


def test_basic_index_equality_traverses_underlying_arrays():
    # to test bug in pytato which didn't account underlying arrays
    a = pt.make_placeholder("a", (10,), float)
    b = pt.make_placeholder("b", (10,), float)
    assert a[0] != b[0]


def test_idx_lambda_to_hlo():
    from pytato.raising import index_lambda_to_high_level_op
    from pytato.raising import (BinaryOp, BinaryOpType, FullOp, ReduceOp,
                                C99CallOp, BroadcastOp)
    from pyrsistent import pmap
    from pytato.reductions import (SumReductionOperation,
                                   ProductReductionOperation)

    a = pt.make_placeholder("a", (10, 4), dtype=np.float64)
    b = pt.make_placeholder("b", (10, 4), dtype=np.float64)

    assert index_lambda_to_high_level_op(a + b) == BinaryOp(BinaryOpType.ADD,
                                                            a, b)
    assert index_lambda_to_high_level_op(a / 42) == BinaryOp(BinaryOpType.TRUEDIV,
                                                             a, 42)
    assert index_lambda_to_high_level_op(42 * a) == BinaryOp(BinaryOpType.MULT,
                                                             42, a)
    assert index_lambda_to_high_level_op(a ** b) == BinaryOp(BinaryOpType.POWER,
                                                             a, b)
    assert index_lambda_to_high_level_op(a - b) == BinaryOp(BinaryOpType.SUB,
                                                            a, b)
    assert (index_lambda_to_high_level_op(a & b)
            == BinaryOp(BinaryOpType.BITWISE_AND, a, b))
    assert (index_lambda_to_high_level_op(a ^ b)
            == BinaryOp(BinaryOpType.BITWISE_XOR, a, b))
    assert (index_lambda_to_high_level_op(a | b)
            == BinaryOp(BinaryOpType.BITWISE_OR, a, b))
    assert (index_lambda_to_high_level_op(pt.equal(a, b))
            == BinaryOp(BinaryOpType.EQUAL, a, b))
    assert (index_lambda_to_high_level_op(pt.not_equal(a, b))
            == BinaryOp(BinaryOpType.NOT_EQUAL, a, b))
    assert (index_lambda_to_high_level_op(pt.less(a, b))
            == BinaryOp(BinaryOpType.LESS, a, b))
    assert (index_lambda_to_high_level_op(pt.less_equal(a, b))
            == BinaryOp(BinaryOpType.LESS_EQUAL, a, b))
    assert (index_lambda_to_high_level_op(pt.greater(a, b))
            == BinaryOp(BinaryOpType.GREATER, a, b))
    assert (index_lambda_to_high_level_op(pt.greater_equal(a, b))
            == BinaryOp(BinaryOpType.GREATER_EQUAL, a, b))

    assert index_lambda_to_high_level_op(pt.zeros(6)) == FullOp(0)
    assert (index_lambda_to_high_level_op(pt.sum(b, axis=1))
            == ReduceOp(SumReductionOperation(),
                        b,
                        pmap({1: "_r0"})))
    assert (index_lambda_to_high_level_op(pt.prod(a))
            == ReduceOp(ProductReductionOperation(),
                        a,
                        {0: "_r0",
                         1: "_r1"}))
    assert index_lambda_to_high_level_op(pt.sinh(a)) == C99CallOp("sinh", (a,))
    assert index_lambda_to_high_level_op(pt.arctan2(b, a)) == C99CallOp("atan2",
                                                                        (b, a))
    assert (index_lambda_to_high_level_op(pt.broadcast_to(a, (100, 10, 4)))
            == BroadcastOp(a))

    hlo = index_lambda_to_high_level_op(np.nan * a)
    assert isinstance(hlo, BinaryOp)
    assert hlo.binary_op == BinaryOpType.MULT
    assert np.isnan(hlo.x1)
    assert hlo.x2 is a


def test_deduplicate_data_wrappers():
    from pytato.transform import CachedWalkMapper, deduplicate_data_wrappers

    class DataWrapperCounter(CachedWalkMapper):
        def __init__(self):
            self.count = 0
            super().__init__()

        def map_data_wrapper(self, expr):
            self.count += 1
            return super().map_data_wrapper(expr)

    def count_data_wrappers(expr):
        dwc = DataWrapperCounter()
        dwc(expr)
        return dwc.count

    a = pt.make_data_wrapper(np.arange(27))
    b = pt.make_data_wrapper(np.arange(27))
    c = pt.make_data_wrapper(a.data.view())
    d = pt.make_data_wrapper(np.arange(1, 28))

    res = a+b+c+d

    assert count_data_wrappers(res) == 4

    dd_res = deduplicate_data_wrappers(res)

    assert count_data_wrappers(dd_res) == 3


def test_einsum_dot_axes_has_correct_dim():
    # before 'pytato@895bae5', this test would fail because of incorrect
    # default 'Einsum.axes' instantiation.
    a = pt.make_placeholder("a", (10, 10), "float64")
    b = pt.make_placeholder("b", (10, 10), "float64")
    einsum = pt.einsum("ij,jk   ->    ik", a, b)
    assert len(einsum.axes) == einsum.ndim


def test_created_at():
    a = pt.make_placeholder("a", (10, 10), "float64")
    b = pt.make_placeholder("b", (10, 10), "float64")

    res = a+b
    res2 = a+b

    # {{{ Check that CreatedAt tags are filtered correctly for equality
    from pytato.equality import preprocess_tags_for_equality

    assert res == res2

    assert res.tags != res2.tags
    assert (preprocess_tags_for_equality(res.tags)
            == preprocess_tags_for_equality(res2.tags))

    # }}}

    from pytato.tags import CreatedAt

    created_tag = res.tags_of_type(CreatedAt)

    assert len(created_tag) == 1

    # {{{ Make sure the function name appears in the traceback

    tag, = created_tag

    found = False

    stacksummary = tag.traceback.to_stacksummary()
    assert len(stacksummary) > 10

    for frame in tag.traceback.frames:
        if frame.name == "test_created_at":
            found = True
            break

    assert found

    # }}}


def test_pickling_and_unpickling_is_equal():
    from testlib import RandomDAGContext, make_random_dag
    import pickle
    from pytools import UniqueNameGenerator
    axis_len = 5

    for i in range(50):
        print(i)  # progress indicator

        seed = 120 + i
        rdagc_pt = RandomDAGContext(np.random.default_rng(seed=seed),
                                    axis_len=axis_len, use_numpy=False)

        dag = pt.make_dict_of_named_arrays({"out": make_random_dag(rdagc_pt)})

        # {{{ convert data-wrappers to placeholders

        vng = UniqueNameGenerator()

        def make_dws_placeholder(expr):
            if isinstance(expr, pt.DataWrapper):
                return pt.make_placeholder(vng("_pt_ph"),
                                           expr.shape, expr.dtype)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, make_dws_placeholder)

        # }}}

        assert pickle.loads(pickle.dumps(dag)) == dag

    # {{{ adds an example which guarantees NaN in expression tree

    # pytato<=d015f914 used IEEE-representation of NaN in its expression graphs
    # and since NaN != NaN the following assertions would fail.

    x = pt.make_placeholder("x", shape=(10, 4), dtype="float64")
    expr = pt.maximum(2*x, 3*x)

    assert pickle.loads(pickle.dumps(expr)) == expr

    expr = pt.full((10, 4), np.nan)
    assert pickle.loads(pickle.dumps(expr)) == expr

    # }}}


def test_adv_indexing_into_zero_long_axes():
    # See https://github.com/inducer/meshmode/issues/321#issuecomment-1105577180
    n = pt.make_size_param("n")

    with pytest.raises(IndexError):
        a = pt.make_placeholder("a", shape=(0, 10), dtype="float64")
        idx = pt.zeros(5, dtype=np.int64)
        a[idx]

    with pytest.raises(IndexError):
        a = pt.make_placeholder("a", shape=(n-n, 10), dtype="float64")
        idx = pt.zeros(5, dtype=np.int64)
        a[idx]

    with pytest.raises(IndexError):
        a = pt.make_placeholder("a", shape=(n-n-2, 10), dtype="float64")
        idx = pt.zeros(5, dtype=np.int64)
        a[idx]

    # {{{ no index error => sanity checks are working fine

    a = pt.make_placeholder("a", shape=(n-n+1, 10), dtype="float64")
    idx = pt.zeros(5, dtype=np.int64)
    a[idx]

    # }}}

    # {{{ indexer array is of zero size => should be fine

    a = pt.make_placeholder("a", shape=(n-n, 10), dtype="float64")
    idx = pt.zeros((0, 10), dtype=np.int64)
    a[idx]

    a = pt.make_placeholder("a", shape=(n-n, 10), dtype="float64")
    idx = pt.zeros((n-n, 10), dtype=np.int64)
    a[idx]

    # }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
