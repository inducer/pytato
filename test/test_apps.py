#!/usr/bin/env python
from __future__ import annotations


__copyright__ = """Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2022 Isuru Fernando
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

import numpy as np
import numpy.linalg as la

import pyopencl as cl
from pymbolic.mapper import IdentityMapper as PymbolicIdentityMapper
from pyopencl.tools import (  # noqa: F401
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
from pytools.tag import Tag, tag_dataclass

import pytato as pt
from pytato.transform import CopyMapper, WalkMapper, deduplicate


# {{{ Trace an FFT

# A failed-for-now experiment: Trace through an FFT to generate code.
# It might as well live on as a reasonably complex test.

@tag_dataclass
class FFTIntermediate(Tag):
    level: int


class FFTVectorGatherer(WalkMapper):
    def __init__(self, n):
        self.n = n
        self.level_to_arrays = {}
        super().__init__()

    def map_index_lambda(self, expr):
        tags = expr.tags_of_type(FFTIntermediate)
        if tags:
            ffti_tag, = tags
            self.level_to_arrays.setdefault(
                    ffti_tag.level, set()).add(expr)
        super().map_index_lambda(expr)


class ConstantSizer(PymbolicIdentityMapper):
    def map_constant(self, expr):
        if isinstance(expr, float):
            return np.float64(expr)
        elif isinstance(expr, complex):
            return np.complex128(expr)
        else:
            return expr


class FFTRealizationMapper(CopyMapper):
    def __init__(self, old_array_to_new_array):
        # Must use err_on_created_duplicate=False, because the use of ConstantSizer
        # in map_index_lambda creates IndexLambdas that differ only in the type of
        # their contained constants, which changes their identity but not their
        # equality
        super().__init__(err_on_created_duplicate=False)
        self.old_array_to_new_array = old_array_to_new_array

    def map_index_lambda(self, expr):
        tags = expr.tags_of_type(FFTIntermediate)
        if tags:
            try:
                return self.old_array_to_new_array[expr]
            except KeyError:
                pass

        return super().map_index_lambda(
                expr.copy(expr=ConstantSizer()(expr.expr)))

    def map_concatenate(self, expr):
        from pytato.tags import ImplStored, PrefixNamed
        return super().map_concatenate(expr).tagged(
                (ImplStored(), PrefixNamed("concat")))


def make_fft_realization_mapper(fft_vec_gatherer):
    old_array_to_new_array = {}
    levels = sorted(fft_vec_gatherer.level_to_arrays, reverse=True)

    for lev in levels:
        lev_mapper = FFTRealizationMapper(old_array_to_new_array)
        arrays = fft_vec_gatherer.level_to_arrays[lev]
        rec_arrays = [lev_mapper(ary) for ary in arrays]
        lev_array = pt.concatenate(rec_arrays, axis=0)
        assert lev_array.shape == (fft_vec_gatherer.n,)

        startidx = 0
        for array in arrays:
            size = array.shape[0]
            sub_array = lev_array[startidx:startidx+size]
            startidx += size
            old_array_to_new_array[array] = sub_array

        assert startidx == fft_vec_gatherer.n

    return FFTRealizationMapper(old_array_to_new_array)


def test_trace_fft(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 33
    x = pt.make_placeholder("x", n, dtype=np.complex128)

    from pymbolic.algorithm import fft
    result = fft(x, custom_np=pt,
            wrap_intermediate_with_level=(
                lambda level, ary: ary.tagged(FFTIntermediate(level))))

    result = deduplicate(result)
    fft_vec_gatherer = FFTVectorGatherer(n)
    fft_vec_gatherer(result)

    mapper = make_fft_realization_mapper(fft_vec_gatherer)

    result = mapper(result)

    prg = pt.generate_loopy(result).program

    x = np.random.randn(n).astype(np.complex128)
    _evt, (result,) = prg(queue, x=x)

    ref_result = fft(x)

    print(la.norm(result-ref_result))
    assert la.norm(result-ref_result) < 1e-14

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
