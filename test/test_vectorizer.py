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
import numpy as np
import pytato as pt

from testlib import RandomDAGContext, make_random_dag
from pytato.transform.parameter_study import (
    ParameterStudyAxisTag,
    ParameterStudyVectorizer,
)

# {{{ Expansion Mapper tests.
def test_vectorize_mapper_placeholder():
    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)
    assert expr.shape == (15, 5)
    my_mapper = ParameterStudyVectorizer(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (15, 5, 10)

    for i, axis in enumerate(new_expr.axes):
        tags = axis.tags_of_type(ParameterStudyAxisTag)
        if i == 2:
            assert tags
        else:
            assert not tags


def test_vectorize_mapper_basic_index():
    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)[14, 0]

    assert expr.shape == ()

    my_mapper = ParameterStudyVectorizer(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (10,)
    assert new_expr.axes[0].tags_of_type(ParameterStudyAxisTag)


def test_vectorize_mapper_advanced_index_contiguous_axes():
    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)[pt.arange(10, dtype=int)]

    assert expr.shape == (10, 5)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (10, 5, 10)
    assert new_expr.axes[2].tags_of_type(ParameterStudyAxisTag)

    assert isinstance(new_expr, pt.AdvancedIndexInContiguousAxes)
    assert isinstance(expr, type(new_expr))


def test_vectorize_mapper_advanced_index_non_contiguous_axes():
    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    ind0 = pt.arange(10, dtype=int).reshape(10, 1)
    ind1 = pt.arange(2, dtype=int).reshape(1, 2)
    expr = pt.make_placeholder(name, (15, 1000, 5), dtype=int)[ind0, :, ind1]

    assert isinstance(expr, pt.AdvancedIndexInNoncontiguousAxes)
    assert expr.shape == (10, 2, 1000)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (10, 2, 1000, 10)
    assert new_expr.axes[3].tags_of_type(ParameterStudyAxisTag)

    assert isinstance(new_expr, pt.AdvancedIndexInNoncontiguousAxes)
    assert isinstance(expr, type(new_expr))


def test_vectorize_mapper_index_lambda():
    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)[14, 0] + pt.ones(100)

    assert expr.shape == (100,)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (100, 10)
    assert isinstance(new_expr, pt.IndexLambda)

    scalar_expr = new_expr.expr

    assert len(scalar_expr.children) == len(expr.expr.children)
    assert scalar_expr != expr.expr
    # We modified it so that we have the new axis.
    assert new_expr.axes[1].tags_of_type(ParameterStudyAxisTag)


def test_vectorize_mapper_roll():
    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.make_placeholder(name, (15, 5), dtype=int)[14, 0] + pt.ones(100)
    expr = pt.roll(expr, axis=0, shift=22)

    assert expr.shape == (100,)
    assert not any(axis.tags_of_type(ParameterStudyAxisTag) for axis in expr.axes)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (100, 10,)
    assert isinstance(new_expr, pt.Roll)
    assert new_expr.axes[1].tags_of_type(ParameterStudyAxisTag)


def test_vectorize_mapper_axis_permutation():
    name = "my_array"
    my_study = ParameterStudyAxisTag(10)
    name_to_studies = {name: frozenset((my_study,))}
    expr = pt.transpose(pt.make_placeholder(name, (15, 5), dtype=int))
    assert expr.shape == (5, 15)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
    new_expr = my_mapper(expr)
    assert new_expr.shape == (5, 15, 10)
    assert isinstance(new_expr, pt.AxisPermutation)

    for i, axis in enumerate(new_expr.axes):
        tags = axis.tags_of_type(ParameterStudyAxisTag)
        if i == 2:
            assert tags
        else:
            assert not tags


def test_vectorize_mapper_reshape():
    name_to_studies, studies, names = _set_up_vectorize_mapper_tests()
    expr = pt.transpose(pt.make_placeholder(names[0],
                                            (15, 5), dtype=int))
    expr2 = pt.transpose(pt.make_placeholder(names[1],
                                             (15, 5), dtype=int))

    out_expr = pt.stack([expr, expr2], axis=0).reshape(10, 15)
    assert out_expr.shape == (10, 15)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
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


def test_vectorize_mapper_stack():
    name_to_studies, studies, names = _set_up_vectorize_mapper_tests()

    expr = pt.transpose(pt.make_placeholder(names[0],
                                            (15, 5), dtype=int))
    expr2 = pt.transpose(pt.make_placeholder(names[1],
                                             (15, 5), dtype=int))

    out_expr = pt.stack([expr, expr2], axis=0)
    assert out_expr.shape == (2, 5, 15)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
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


def _set_up_vectorize_mapper_tests() -> tuple[Mapping[str,
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


def test_vectorize_mapper_concatenate():
    name_to_studies, studies, names = _set_up_vectorize_mapper_tests()

    expr = pt.transpose(pt.make_placeholder(names[0],
                                            (15, 5), dtype=int))
    expr2 = pt.transpose(pt.make_placeholder(names[1],
                                             (15, 5), dtype=int))

    out_expr = pt.concatenate([expr, expr2], axis=0)
    assert out_expr.shape == (10, 15)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
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


def test_vectorize_mapper_einsum_matmul():

    name_to_studies, _, names = _set_up_vectorize_mapper_tests()

    # Matmul gets expanded correctly.
    a = pt.make_placeholder(names[0],
                            (47, 42), dtype=int)
    b = pt.make_placeholder(names[1],
                            (42, 5), dtype=int)

    c = pt.matmul(a, b)
    assert isinstance(c, pt.Einsum)
    assert c.shape == (47, 5)

    my_mapper = ParameterStudyVectorizer(name_to_studies)
    updated_c = my_mapper(c)

    assert updated_c.shape == (47, 5, 10, 1000)


def test_vectorize_mapper_does_nothing_if_tags_not_there():
    name_to_studies, _, _ = _set_up_vectorize_mapper_tests()

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

        my_mapper = ParameterStudyVectorizer(name_to_studies)
        new_dag = my_mapper(dag)

        assert new_dag == dag

        # }}}
# }}}
