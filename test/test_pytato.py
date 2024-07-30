#!/usr/bin/env python
from __future__ import annotations


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
from typing import (
    Mapping,
)

import attrs
import numpy as np
import pytest
from testlib import RandomDAGContext, make_random_dag

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

import pytato as pt
from pytato.array import _SuppliedAxesAndTagsMixin
from pytato.transform.parameter_study import ParameterStudyAxisTag


def test_matmul_input_validation():
    a = pt.make_placeholder(name="a", shape=(10, 10))
    b = pt.make_placeholder(name="b", shape=(20, 10))

    with pytest.raises(ValueError):
        a @ b

    c = pt.make_placeholder(name="c", shape=())
    with pytest.raises(ValueError):
        c @ c

    n = pt.make_size_param("n")
    d = pt.make_placeholder(name="d", shape=(n, n))
    d @ d


def test_roll_input_validation():
    a = pt.make_placeholder(name="a", shape=(10, 10))
    pt.roll(a, 1, axis=0)

    with pytest.raises(ValueError):
        pt.roll(a, 1, axis=2)

    with pytest.raises(ValueError):
        pt.roll(a, 1, axis=-1)


def test_transpose_input_validation():
    a = pt.make_placeholder(name="a", shape=(10, 10))
    pt.transpose(a)

    with pytest.raises(ValueError):
        pt.transpose(a, (2, 0, 1))

    with pytest.raises(ValueError):
        pt.transpose(a, (1, 1))

    with pytest.raises(ValueError):
        pt.transpose(a, (0,))


def test_slice_input_validation():
    a = pt.make_placeholder(name="a", shape=(10, 10, 10))

    a[0]
    a[0, 0]
    a[0, 0, 0]

    with pytest.raises(IndexError):
        a[0, 0, 0, 0]

    with pytest.raises(IndexError):
        a[10]


def test_index_type_validation():
    a = pt.make_placeholder(name="a", shape=(10,))

    idx = pt.make_placeholder(name="idx", shape=(5,), dtype=np.int8)
    a[idx]

    idx = pt.make_placeholder(name="idx", shape=(5,), dtype=np.uint8)
    a[idx]

    idx = pt.make_placeholder(name="idx", shape=(5,))
    with pytest.raises(IndexError):
        a[idx]


def test_stack_input_validation():
    x = pt.make_placeholder(name="x", shape=(10, 10))
    y = pt.make_placeholder(name="y", shape=(1, 10))

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
    x = pt.make_placeholder("x", shape=(10, 4))
    y = 2*x

    knl = pt.generate_loopy(y).kernel

    assert x.name in knl.arg_dict
    assert x.name in knl.get_read_variables()


def test_zero_length_arrays():
    x = pt.make_placeholder("x", shape=(0, 4))
    y = 2*x

    assert y.shape == (0, 4)

    knl = pt.generate_loopy(y).kernel
    assert all(dom.is_empty() for dom in knl.domains if dom.total_dim() != 0)


def test_concatenate_input_validation():
    x = pt.make_placeholder(name="x", shape=(10, 10))
    y = pt.make_placeholder(name="y", shape=(1, 10))

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
    x = pt.make_placeholder("x", shape=(3, 3, 4))

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
    x = pt.make_placeholder("x", shape=(0,))
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

    x = pt.make_placeholder(name="x", shape=(10,))
    assert Foo() + x == "bar"
    assert x + Foo() == "baz"


def test_same_placeholder_name_raises():
    from pytato.diagnostic import NameClashError
    x = pt.make_placeholder(name="arr", shape=(10, 4))
    y = pt.make_placeholder(name="arr", shape=(10, 4))

    with pytest.raises(NameClashError):
        pt.generate_loopy(x+y)

    n1 = pt.make_size_param("n")
    n2 = pt.make_size_param("n")
    x = pt.make_placeholder(name="arr", shape=(n1, n2))
    with pytest.raises(NameClashError):
        pt.generate_loopy(2*x)


def test_einsum_error_handling():
    with pytest.raises(ValueError):
        # operands not enough
        pt.einsum("ij,j->j", pt.make_placeholder("x", (2, 2)))

    with pytest.raises(ValueError):
        # double index use in the out spec.
        pt.einsum("ij,j->jj", ("a", "b"))


def test_accessing_dict_of_named_arrays_validation():
    x = pt.make_placeholder(name="x", shape=10)
    y1y2 = pt.make_dict_of_named_arrays({"y1": 2*x, "y2": 3*x})

    assert isinstance(y1y2["y1"], pt.array.NamedArray)
    assert y1y2["y1"].shape == (2*x).shape
    assert y1y2["y1"].dtype == (2*x).dtype


def test_call_loopy_shape_inference():
    import loopy as lp

    from pytato.loopy import call_loopy
    from pytato.utils import are_shapes_equal

    knl = lp.make_kernel(
            ["{[i, j]: 0<=i<(2*n + 3*m + 2) and 0<=j<(6*n + 4*m + 3)}",
             "{[ii, jj]: 0<=ii<m and 0<=jj<n}"],
            """
            <> tmp = sum([i, j], A[i, j])
            out[ii, jj] = tmp*(ii + jj)
            """, lang_version=(2018, 2))

    # {{{ variant 1

    A = pt.make_placeholder(name="x", shape=(20, 37))  # noqa: N806
    y = call_loopy(knl, {"A": A})["out"]
    assert are_shapes_equal(y.shape, (4, 3))

    # }}}

    # {{{ variant 2

    n1 = pt.make_size_param("n1")
    n2 = pt.make_size_param("n2")
    A = pt.make_placeholder(name="x",  # noqa: N806
                            shape=(4*n1 + 6*n2 + 2, 12*n1 + 8*n2 + 3))

    y = call_loopy(knl, {"A": A})["out"]
    assert are_shapes_equal(y.shape, (2*n2, 2*n1))

    # }}}


def test_tagging_array():
    from pytools.tag import Tag

    class BestArrayTag(Tag):
        """
        Best array known to humankind.
        """

    x = pt.make_placeholder(shape=(42, 1729), name="x")
    y = x.tagged(BestArrayTag())
    assert any(isinstance(tag, BestArrayTag) for tag in y.tags)


def test_dict_of_named_arrays_comparison():
    # See https://github.com/inducer/pytato/pull/137
    x = pt.make_placeholder("x", (10, 4))
    dict1 = pt.make_dict_of_named_arrays({"out": 2 * x})
    dict2 = pt.make_dict_of_named_arrays({"out": 2 * x})
    dict3 = pt.make_dict_of_named_arrays({"not_out": 2 * x})
    dict4 = pt.make_dict_of_named_arrays({"out": 3 * x})
    assert dict1 == dict2
    assert dict1 != dict3
    assert dict1 != dict4


def test_toposortmapper():
    n = pt.make_size_param("n")
    array = pt.make_placeholder(name="array", shape=n)
    stack = pt.stack([array, 2*array, array + 6])
    y = stack @ stack.T

    tm = pt.transform.TopoSortMapper()
    tm(y)

    from pytato.array import (
        AxisPermutation,
        Einsum,
        IndexLambda,
        Placeholder,
        SizeParam,
        Stack,
    )

    assert isinstance(tm.topological_order[0], SizeParam)
    assert isinstance(tm.topological_order[1], Placeholder)
    assert isinstance(tm.topological_order[2], IndexLambda)
    assert isinstance(tm.topological_order[3], IndexLambda)
    assert isinstance(tm.topological_order[4], Stack)
    assert isinstance(tm.topological_order[5], AxisPermutation)
    assert isinstance(tm.topological_order[6], Einsum)


def test_userscollector():
    from testlib import RandomDAGContext, make_random_dag

    from pytools.graph import reverse_graph

    from pytato.analysis import get_nusers
    from pytato.transform import UsersCollector

    # Check that nodes without users are correctly reversed
    array = pt.make_placeholder(name="array", shape=1, dtype=np.int64)
    y = array+1

    uc = UsersCollector()
    uc(y)

    rev_graph = reverse_graph(uc.node_to_users)
    rev_graph2 = reverse_graph(reverse_graph(rev_graph))

    assert dict(reverse_graph(rev_graph2)) == uc.node_to_users

    assert len(uc.node_to_users) == 2
    assert uc.node_to_users[y] == set()
    assert uc.node_to_users[array].pop() == y
    assert len(uc.node_to_users[array]) == 0

    # Test random DAGs
    axis_len = 5

    for i in range(100):
        print(i)  # progress indicator
        rdagc = RandomDAGContext(np.random.default_rng(seed=i),
                axis_len=axis_len, use_numpy=False)

        dag = make_random_dag(rdagc)

        uc = UsersCollector()
        uc(dag)

        rev_graph = reverse_graph(uc.node_to_users)
        rev_graph2 = reverse_graph(reverse_graph(rev_graph))
        assert rev_graph2 == rev_graph

        nuc = get_nusers(dag)

        assert len(uc.node_to_users) == len(nuc)+1
        assert uc.node_to_users[dag] == set()
        assert nuc[dag] == 0


def test_linear_complexity_inequality():
    # See https://github.com/inducer/pytato/issues/163
    from numpy.random import default_rng

    import pytato as pt
    from pytato.equality import EqualityComparer

    def construct_intestine_graph(depth=100, seed=0):
        rng = default_rng(seed)
        x = pt.make_placeholder("x", shape=(10,))

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
        expected_str = "".join([c for c in expected_repr if c not in [" ", "\n"]])
        result_str = "".join([c for c in repr(ary)if c not in [" ", "\n"]])
        assert expected_str == result_str

    _assert_stripped_repr(
        3*x + 4*y,
        """
IndexLambda(
    shape=(10, 4),
    dtype='int64',
    expr=Sum((Subscript(Variable('_in0'),
                        (Variable('_0'), Variable('_1'))),
              Subscript(Variable('_in1'),
                        (Variable('_0'), Variable('_1'))))),
    bindings={'_in0': IndexLambda(
                                  shape=(10, 4),
                                  dtype='int64',
                                  expr=Product((3, Subscript(Variable('_in1'),
                                                             (Variable('_0'),
                                                              Variable('_1'))))),
                                  bindings={'_in1': Placeholder(shape=(10, 4),
                                                                dtype='int64',
                                                                name='x')}),
              '_in1': IndexLambda(
                                  shape=(10, 4),
                                  dtype='int64',
                                  expr=Product((4, Subscript(Variable('_in1'),
                                                             (Variable('_0'),
                                                              Variable('_1'))))),
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
    shape=(10, 4),
    dtype='int64',
    expr=Product((Subscript(Variable('_in0'),
                            (Variable('_0'), Variable('_1'))),
                  TypeCast(dtype('int64'), Subscript(Variable('_in1'),
                            (Variable('_0'), Variable('_1')))))),
    bindings={'_in0': Placeholder(shape=(10, 4), dtype='int64', name='y'),
              '_in1': IndexLambda(
                  shape=(10, 4),
                  dtype='bool',
                  expr=Comparison(Subscript(Variable('_in0'),
                                            (Variable('_0'), Variable('_1'))),
                                  '!=',
                                  3),
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
    x1 = pt.make_placeholder("x1", shape=(10, 4))
    x2 = pt.make_placeholder("x2", shape=(10, 4))

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
        x = pt.make_placeholder("x", shape=(10,))
        y = x

        for _ in range(depth):
            coeff1, coeff2 = rng.integers(0, 10, 2)
            y = coeff1 * y + coeff2 * y

        return y, x

    expected_result = set()

    class SubexprRecorder(pt.transform.CachedWalkMapper):
        def get_cache_key(self, expr: pt.transform.ArrayOrNames) -> int:
            return id(expr)

        def post_visit(self, expr):
            if not isinstance(expr, pt.Placeholder):
                expected_result.add(expr)
            else:
                assert expr.name == "x"

    expr, inp = construct_intestine_graph()
    result = pt.transform.rec_get_user_nodes(expr, inp)
    SubexprRecorder()(expr)

    assert (expected_result == result)


def test_basic_index_equality_traverses_underlying_arrays():
    # to test bug in pytato which didn't account underlying arrays
    a = pt.make_placeholder("a", (10,))
    b = pt.make_placeholder("b", (10,))
    assert a[0] != b[0]


def test_idx_lambda_to_hlo():
    from immutabledict import immutabledict

    from pytato.raising import (
        BinaryOp,
        BinaryOpType,
        BroadcastOp,
        C99CallOp,
        FullOp,
        ReduceOp,
        index_lambda_to_high_level_op,
    )
    from pytato.reductions import ProductReductionOperation, SumReductionOperation

    a = pt.make_placeholder("a", (10, 4))
    b = pt.make_placeholder("b", (10, 4))

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
                        immutabledict({1: "_r0"})))
    assert (index_lambda_to_high_level_op(pt.prod(a))
            == ReduceOp(ProductReductionOperation(),
                        a,
                        immutabledict({0: "_r0",
                             1: "_r1"})))
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

        def get_cache_key(self, expr):
            return id(expr)

        def map_data_wrapper(self, expr):
            self.count += 1
            return super().map_data_wrapper(expr)

    def count_data_wrappers(expr):
        dwc = DataWrapperCounter()
        dwc(expr)
        return dwc.count

    a = pt.make_data_wrapper(np.arange(27))
    b = pt.make_data_wrapper(np.arange(27))
    # pylint-disable-reason: pylint is correct, DataInterface doesn't declare a
    # view method, but for numpy-like arrays it should be OK.
    c = pt.make_data_wrapper(a.data.view())   # pylint: disable=E1101
    d = pt.make_data_wrapper(np.arange(1, 28))

    res = a+b+c+d

    assert count_data_wrappers(res) == 4

    dd_res = deduplicate_data_wrappers(res)

    assert count_data_wrappers(dd_res) == 3


def test_einsum_dot_axes_has_correct_dim():
    # before 'pytato@895bae5', this test would fail because of incorrect
    # default 'Einsum.axes' instantiation.
    a = pt.make_placeholder("a", (10, 10))
    b = pt.make_placeholder("b", (10, 10))
    einsum = pt.einsum("ij,jk   ->    ik", a, b)
    assert len(einsum.axes) == einsum.ndim


def test_created_at():
    pt.set_traceback_tag_enabled()

    a = pt.make_placeholder("a", (10, 10), "float64")
    b = pt.make_placeholder("b", (10, 10), "float64")

    # res1 and res2 are defined on different lines and should have different
    # CreatedAt tags.
    res1 = a+b
    res2 = a+b

    # res3 and res4 are defined on the same line and should have the same
    # CreatedAt tags.
    res3 = a+b; res4 = a+b  # noqa: E702

    # {{{ Check that CreatedAt tags are handled correctly for equality/hashing

    assert res1 == res2 == res3 == res4
    assert hash(res1) == hash(res2) == hash(res3) == hash(res4)

    assert res1.non_equality_tags != res2.non_equality_tags
    assert res3.non_equality_tags == res4.non_equality_tags
    assert hash(res1.non_equality_tags) != hash(res2.non_equality_tags)
    assert hash(res3.non_equality_tags) == hash(res4.non_equality_tags)

    assert res1.tags == res2.tags == res3.tags == res4.tags
    assert hash(res1.tags) == hash(res2.tags) == hash(res3.tags) == hash(res4.tags)

    # }}}

    from pytato.tags import CreatedAt

    created_tag = frozenset({tag
                         for tag in res1.non_equality_tags
                         if isinstance(tag, CreatedAt)})

    assert len(created_tag) == 1

    # {{{ Make sure the function name appears in the traceback

    tag, = created_tag

    found = False

    stacksummary = tag.traceback.to_stacksummary()
    assert len(stacksummary) > 10

    for frame in tag.traceback.frames:
        if frame.name == "test_created_at" and "a+b" in frame.line:
            found = True
            break

    assert found

    # }}}

    # {{{ Make sure that CreatedAt tags are in the visualization

    from pytato.visualization import get_dot_graph
    s = get_dot_graph(res1)
    assert "test_created_at" in s
    assert "a+b" in s

    # }}}

    # {{{ Make sure only a single CreatedAt tag is created

    old_tag = tag

    res1 = res1 + res2

    created_tag = frozenset({tag
                         for tag in res1.non_equality_tags
                         if isinstance(tag, CreatedAt)})

    assert len(created_tag) == 1

    tag, = created_tag

    # Tag should be recreated
    assert tag != old_tag

    # }}}

    # {{{ Make sure that copying preserves the tag

    old_tag = tag

    res1_new = pt.transform.map_and_copy(res1, lambda x: x)

    created_tag = frozenset({tag
                         for tag in res1_new.non_equality_tags
                         if isinstance(tag, CreatedAt)})

    assert len(created_tag) == 1

    tag, = created_tag

    assert old_tag == tag

    # }}}

    # {{{ Test disabling traceback creation

    pt.set_traceback_tag_enabled(False)

    a = pt.make_placeholder("a", (10, 10), "float64")

    created_tag = frozenset({tag
                         for tag in a.non_equality_tags
                         if isinstance(tag, CreatedAt)})

    assert len(created_tag) == 0

    # }}}


def test_pickling_and_unpickling_is_equal():
    import pickle

    from testlib import RandomDAGContext, make_random_dag

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
                return pt.make_placeholder(vng("_pt_ph"),  # noqa: B023
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
        a = pt.make_placeholder("a", shape=(0, 10))
        idx = pt.zeros(5, dtype=np.int64)
        a[idx]

    with pytest.raises(IndexError):
        a = pt.make_placeholder("a", shape=(n-n, 10))
        idx = pt.zeros(5, dtype=np.int64)
        a[idx]

    with pytest.raises(IndexError):
        a = pt.make_placeholder("a", shape=(n-n-2, 10))
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


def test_expand_dims_input_validate():
    a = pt.make_placeholder("x", (10, 4))

    assert pt.expand_dims(a, (0, 2, 4)).shape == (1, 10, 1, 4, 1)
    assert pt.expand_dims(a, (-5, -3, -1)).shape == (1, 10, 1, 4, 1)
    assert pt.expand_dims(a, (-3)).shape == (1, 10, 4)

    with pytest.raises(ValueError):
        pt.expand_dims(a, (3, 3))

    with pytest.raises(ValueError):
        pt.expand_dims(a, (0, 2, 5))

    with pytest.raises(ValueError):
        pt.expand_dims(a, -4)


def test_with_tagged_reduction():
    from testlib import FooRednTag

    from pytato.diagnostic import NotAReductionAxis
    from pytato.raising import index_lambda_to_high_level_op
    x = pt.make_placeholder("x", shape=(10, 10))
    x_sum = pt.sum(x)

    with pytest.raises(NotAReductionAxis):
        # axis='_0': not being reduced over.
        x_sum = x_sum.with_tagged_reduction("_0", FooRednTag())

    hlo = index_lambda_to_high_level_op(x_sum)
    x_sum = x_sum.with_tagged_reduction(hlo.axes[1], FooRednTag())
    assert x_sum.var_to_reduction_descr[hlo.axes[1]].tags_of_type(FooRednTag)
    assert not x_sum.var_to_reduction_descr[hlo.axes[0]].tags_of_type(FooRednTag)

    x_colsum = pt.einsum("ij->j", x)

    with pytest.raises(TypeError):
        # no longer support indexing by string.
        x_colsum.with_tagged_reduction("j", FooRednTag())

    my_descr = x_colsum.access_descriptors[0][0]
    x_colsum = x_colsum.with_tagged_reduction(my_descr,
                                              FooRednTag())

    assert (x_colsum
            .redn_axis_to_redn_descr[my_descr]
            .tags_of_type(FooRednTag))


def test_derived_class_uses_correct_array_eq():
    @attrs.define(frozen=True)
    class MyNewArrayT(_SuppliedAxesAndTagsMixin, pt.Array):
        pass

    with pytest.raises(AssertionError):
        MyNewArrayT(tags=frozenset(), axes=())

    @attrs.define(frozen=True, eq=False)
    class MyNewAndCorrectArrayT(_SuppliedAxesAndTagsMixin, pt.Array):
        pass

    MyNewAndCorrectArrayT(tags=frozenset(), axes=())


def test_lower_to_index_lambda():
    from pytato.array import IndexLambda, Reshape
    expr = pt.ones(12).reshape(6, 2).reshape(3, 4)
    idx_lambda = pt.to_index_lambda(expr)
    assert isinstance(idx_lambda, IndexLambda)
    binding, = idx_lambda.bindings.values()
    # test that it didn't recurse further
    assert isinstance(binding, Reshape)


# {{{ Expansion Mapper tests.
def test_expansion_mapper_placeholder():
    from pytato.transform.parameter_study import ExpansionMapper, ParameterStudyAxisTag

    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)
    assert expr.shape == (15, 5)
    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (15, 5, 10)

    for i, axis in enumerate(new_expr.axes):
        tags = axis.tags_of_type(ParameterStudyAxisTag)
        if i == 2:
            assert tags
        else:
            assert not tags


def test_expansion_mapper_basic_index():
    from pytato.transform.parameter_study import ExpansionMapper, ParameterStudyAxisTag

    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)[14, 0]

    assert expr.shape == ()

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (10,)
    assert new_expr.axes[0].tags_of_type(ParameterStudyAxisTag)


def test_expansion_mapper_advanced_index_contiguous_axes():
    from pytato.transform.parameter_study import ExpansionMapper, ParameterStudyAxisTag

    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)[pt.arange(10, dtype=int)]

    assert expr.shape == (10, 5)

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (10, 5, 10)
    assert new_expr.axes[2].tags_of_type(ParameterStudyAxisTag)

    assert isinstance(new_expr, pt.AdvancedIndexInContiguousAxes)
    assert isinstance(expr, type(new_expr))


def test_expansion_mapper_advanced_index_non_contiguous_axes():
    from pytato.transform.parameter_study import ExpansionMapper, ParameterStudyAxisTag

    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    ind0 = pt.arange(10, dtype=int).reshape(10, 1)
    ind1 = pt.arange(2, dtype=int).reshape(1, 2)
    expr = pt.make_placeholder(name, (15, 1000, 5), dtype=int)[ind0, :, ind1]

    assert isinstance(expr, pt.AdvancedIndexInNoncontiguousAxes)
    assert expr.shape == (10, 2, 1000)

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (10, 2, 1000, 10)
    assert new_expr.axes[3].tags_of_type(ParameterStudyAxisTag)

    assert isinstance(new_expr, pt.AdvancedIndexInNoncontiguousAxes)
    assert isinstance(expr, type(new_expr))


def test_expansion_mapper_index_lambda():
    from pytato.transform.parameter_study import ExpansionMapper, ParameterStudyAxisTag

    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)[14, 0] + pt.ones(100)

    assert expr.shape == (100,)

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (100, 10)
    assert isinstance(new_expr, pt.IndexLambda)

    scalar_expr = new_expr.expr

    assert len(scalar_expr.children) == len(expr.expr.children)
    assert scalar_expr != expr.expr
    # We modified it so that we have the new axis.
    assert new_expr.axes[1].tags_of_type(ParameterStudyAxisTag)


def test_expansion_mapper_roll():
    from pytato.transform.parameter_study import ExpansionMapper, ParameterStudyAxisTag

    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)[14, 0] + pt.ones(100)
    expr = pt.roll(expr, axis=0, shift=22)

    assert expr.shape == (100,)
    assert not any(axis.tags_of_type(ParameterStudyAxisTag) for axis in expr.axes)

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (100, 10,)
    assert isinstance(new_expr, pt.Roll)
    assert new_expr.axes[1].tags_of_type(ParameterStudyAxisTag)


def test_expansion_mapper_axis_permutation():
    from pytato.transform.parameter_study import ExpansionMapper, ParameterStudyAxisTag

    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.transpose(pt.make_placeholder(name, (15, 5), dtype=int))
    assert expr.shape == (5, 15)

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (5, 15, 10)
    assert isinstance(new_expr, pt.AxisPermutation)

    for i, axis in enumerate(new_expr.axes):
        tags = axis.tags_of_type(ParameterStudyAxisTag)
        if i == 2:
            assert tags
        else:
            assert not tags


def test_expansion_mapper_reshape():
    from pytato.transform.parameter_study import ExpansionMapper

    name_to_studies, studies, names = _set_up_expansion_mapper_tests()
    expr = pt.transpose(pt.make_placeholder(names[0],
                                            (15, 5), dtype=int))
    expr2 = pt.transpose(pt.make_placeholder(names[1],
                                             (15, 5), dtype=int))

    out_expr = pt.stack([expr, expr2], axis=0).reshape(10, 15)
    assert out_expr.shape == (10, 15)

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(out_expr)
    assert new_expr.shape == (10, 15, 10, 1000)
    assert isinstance(new_expr, pt.Reshape)

    for i, axis in enumerate(new_expr.axes):
        tags = axis.tags_of_type(ParameterStudyAxisTag)
        if i > 1:
            assert tags
        else:
            assert not tags

    assert not new_expr.axes[2].tags_of_type(studies[1])
    assert not new_expr.axes[3].tags_of_type(studies[0])


def test_expansion_mapper_stack():
    from pytato.transform.parameter_study import ExpansionMapper

    name_to_studies, studies, names = _set_up_expansion_mapper_tests()

    expr = pt.transpose(pt.make_placeholder(names[0],
                                            (15, 5), dtype=int))
    expr2 = pt.transpose(pt.make_placeholder(names[1],
                                             (15, 5), dtype=int))

    out_expr = pt.stack([expr, expr2], axis=0)
    assert out_expr.shape == (2, 5, 15)

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(out_expr)
    assert new_expr.shape == (2, 5, 15, 10, 1000)
    assert isinstance(new_expr, pt.Stack)

    for i, axis in enumerate(new_expr.axes):
        tags = axis.tags_of_type(ParameterStudyAxisTag)
        if i > 2:
            assert tags
        else:
            assert not tags

    assert not new_expr.axes[3].tags_of_type(studies[1])
    assert not new_expr.axes[4].tags_of_type(studies[0])


def _set_up_expansion_mapper_tests() -> tuple[Mapping[str,
                                                frozenset[ParameterStudyAxisTag]],
                                              tuple[ParameterStudyAxisTag, ...],
                                              tuple[str, ...]]:

    class Study2(ParameterStudyAxisTag):
        pass

    class Study1(ParameterStudyAxisTag):
        pass
    name = "a"
    study1 = Study1(10)
    arr2 = "b"
    study2 = Study2(1000)
    name_to_studies = {name: frozenset((study1,)), arr2: frozenset((study2,))}

    return name_to_studies, (Study1, Study2,), (name, arr2,)


def test_expansion_mapper_concatenate():
    from pytato.transform.parameter_study import ExpansionMapper

    name_to_studies, studies, names = _set_up_expansion_mapper_tests()

    expr = pt.transpose(pt.make_placeholder(names[0],
                                            (15, 5), dtype=int))
    expr2 = pt.transpose(pt.make_placeholder(names[1],
                                             (15, 5), dtype=int))

    out_expr = pt.concatenate([expr, expr2], axis=0)
    assert out_expr.shape == (10, 15)

    my_mapper = ExpansionMapper(name_to_studies)
    new_expr = my_mapper(out_expr)
    assert new_expr.shape == (10, 15, 10, 1000)
    assert isinstance(new_expr, pt.Concatenate)

    for i, axis in enumerate(new_expr.axes):
        tags = axis.tags_of_type(ParameterStudyAxisTag)
        if i > 1:
            assert tags
        else:
            assert not tags

    assert not new_expr.axes[2].tags_of_type(studies[1])
    assert not new_expr.axes[3].tags_of_type(studies[0])


def test_expansion_mapper_einsum_matmul():
    from pytato.transform.parameter_study import ExpansionMapper

    name_to_studies, _, names = _set_up_expansion_mapper_tests()

    # Matmul gets expanded correctly.
    a = pt.make_placeholder(names[0],
                            (47, 42), dtype=int)
    b = pt.make_placeholder(names[1],
                            (42, 5), dtype=int)

    c = pt.matmul(a, b)
    assert isinstance(c, pt.Einsum)
    assert c.shape == (47, 5)

    my_mapper = ExpansionMapper(name_to_studies)
    updated_c = my_mapper(c)

    assert updated_c.shape == (47, 5, 10, 1000)


def test_expansion_mapper_does_nothing_if_tags_not_there():
    from pytato.transform.parameter_study import ExpansionMapper

    name_to_studies, _, _ = _set_up_expansion_mapper_tests()

    from testlib import RandomDAGContext, make_random_dag

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
                return pt.make_placeholder(vng("_pt_ph"),  # noqa: B023
                                           expr.shape, expr.dtype)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, make_dws_placeholder)

        my_mapper = ExpansionMapper(name_to_studies)
        new_dag = my_mapper(dag)

        assert new_dag == dag

        # }}}
# }}}


def test_cached_walk_mapper_with_extra_args():
    from testlib import RandomDAGContext, make_random_dag

    class MyWalkMapper(pt.transform.CachedWalkMapper):
        def get_cache_key(self, expr, passed_number) -> int:
            return id(expr), passed_number

        def post_visit(self, expr, passed_number):
            assert passed_number == 42

    my_walk_mapper = MyWalkMapper()

    rdagc = RandomDAGContext(np.random.default_rng(seed=0),
                             axis_len=4, use_numpy=False)

    dag = make_random_dag(rdagc)

    my_walk_mapper(dag, 42)
    my_walk_mapper(dag, passed_number=42)

    with pytest.raises(AssertionError):
        my_walk_mapper(dag, 5)

    with pytest.raises(AssertionError):
        my_walk_mapper(dag, 7)

    with pytest.raises(TypeError):
        # passing incorrect argument should raise TypeError while calling post_visit
        my_walk_mapper(dag, bad_arg_name=7)


def test_unify_axes_tags():
    from testlib import BarTag, BazTag, FooTag, QuuxTag, TestlibTag

    from pytato.array import EinsumReductionAxis

    # {{{ 1. broadcasting + expand_dims

    x = pt.make_placeholder("x", (10, 4))
    x = x.with_tagged_axis(0, FooTag())
    x = x.with_tagged_axis(1, BarTag())

    y = pt.expand_dims(x, (2, 3)) + x

    y_unified = pt.unify_axes_tags(y)
    assert (y_unified.axes[0].tags_of_type(TestlibTag)
            == frozenset([FooTag()]))
    assert (y_unified.axes[2].tags_of_type(TestlibTag)
            == frozenset([FooTag()]))
    assert (y_unified.axes[1].tags_of_type(TestlibTag)
            == frozenset([BarTag()]))
    assert (y_unified.axes[3].tags_of_type(TestlibTag)
            == frozenset([BarTag()]))

    # }}}

    # {{{ 2. back-propagation + einsum

    x = pt.make_placeholder("x", (10, 4))
    x = x.with_tagged_axis(0, FooTag())

    y = pt.make_placeholder("y", (10, 4))
    y = y.with_tagged_axis(1, BarTag())

    z = pt.einsum("ij, ij -> i", x, y)
    z_unified = pt.unify_axes_tags(z)

    assert (z_unified.axes[0].tags_of_type(TestlibTag)
            == frozenset([FooTag()]))
    assert (z_unified.args[0].axes[1].tags_of_type(TestlibTag)
            == frozenset([BarTag()]))
    assert (z_unified.args[1].axes[0].tags_of_type(TestlibTag)
            == frozenset([FooTag()]))
    assert (z_unified.redn_axis_to_redn_descr[EinsumReductionAxis(0)]
            .tags_of_type(TestlibTag)
            == frozenset([BarTag()]))

    # }}}

    # {{{ 3. advanced indexing

    idx1 = pt.make_placeholder("idx1", (42, 1), "int32")
    idx1 = idx1.with_tagged_axis(0, FooTag())

    idx2 = pt.make_placeholder("idx2", (1, 1729), "int32")
    idx2 = idx2.with_tagged_axis(1, BarTag())

    u = pt.make_placeholder("u", (4, 5, 6, 7, 8, 9), "float32")
    u = u.with_tagged_axis(0, BazTag())
    u = u.with_tagged_axis(1, QuuxTag())
    u = u.with_tagged_axis(2, QuuxTag())
    u = u.with_tagged_axis(5, QuuxTag())

    y = u[:, 1:4, 2*idx1, 0, 3*idx2, :]

    y_unified = pt.unify_axes_tags(y)

    assert (y_unified.axes[0].tags_of_type(TestlibTag)
            == frozenset([BazTag()]))
    assert (y_unified.axes[1].tags_of_type(TestlibTag)
            == frozenset())
    assert (y_unified.axes[2].tags_of_type(TestlibTag)
            == frozenset([FooTag()]))
    assert (y_unified.axes[3].tags_of_type(TestlibTag)
            == frozenset([BarTag()]))
    assert (y_unified.axes[4].tags_of_type(TestlibTag)
            == frozenset([QuuxTag()]))

    # }}}


def test_rewrite_einsums_with_no_broadcasts():
    a = pt.make_placeholder("a", (10, 4, 1))
    b = pt.make_placeholder("b", (10, 1, 4))
    c = pt.einsum("ijk,ijk->ijk", a, b)
    expr = pt.einsum("ijk,ijk,ijk->i", a, b, c)

    new_expr = pt.rewrite_einsums_with_no_broadcasts(expr)
    assert pt.analysis.is_einsum_similar_to_subscript(new_expr, "ij,ik,ijk->i")
    assert pt.analysis.is_einsum_similar_to_subscript(new_expr.args[2], "ij,ik->ijk")


def test_dot_visualizers():
    a = pt.make_placeholder("A", shape=(10, 4))
    x1 = pt.make_placeholder("x1", shape=4)
    x2 = pt.make_placeholder("x2", shape=4)

    y = a @ (2*x1 + 3*x2)

    axis_len = 5

    graphs = [y]

    for i in range(100):
        rdagc = RandomDAGContext(np.random.default_rng(seed=i),
                axis_len=axis_len, use_numpy=False)
        graphs.append(make_random_dag(rdagc))

    # {{{ ensure that the generated output is valid dot-lang

    # TODO: Verify the soundness of the generated svg file

    for graph in graphs:
        # plot to .svg file to avoid dep on a webbrowser or X-window system
        pt.show_dot_graph(graph, output_to="svg")

    pt.show_fancy_placeholder_data_flow(y, output_to="svg")

    # }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
