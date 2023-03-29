#!/usr/bin/env python

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
from pyopencl.tools import \
    pytest_generate_tests_for_pyopencl as pytest_generate_tests  # noqa: F401
from pytools.tag import Tag, tag_dataclass

import pytato as pt
from pytato.transform import CopyMapper, WalkMapper

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
    def __init__(self, fft_vec_gatherer):
        super().__init__()

        self.fft_vec_gatherer = fft_vec_gatherer

        self.old_array_to_new_array = {}
        levels = sorted(fft_vec_gatherer.level_to_arrays, reverse=True)

        lev = 0
        arrays = fft_vec_gatherer.level_to_arrays[lev]
        self.finalized = False

        for lev in levels:
            arrays = fft_vec_gatherer.level_to_arrays[lev]
            rec_arrays = [self.rec(ary) for ary in arrays]
            # reset cache so that the partial subs are not stored
            self._cache = {}
            lev_array = pt.concatenate(rec_arrays, axis=0)
            assert lev_array.shape == (fft_vec_gatherer.n,)

            startidx = 0
            for array in arrays:
                size = array.shape[0]
                sub_array = lev_array[startidx:startidx+size]
                startidx += size
                self.old_array_to_new_array[array] = sub_array

            assert startidx == fft_vec_gatherer.n
        self.finalized = True

    def map_index_lambda(self, expr):
        tags = expr.tags_of_type(FFTIntermediate)
        if tags:
            if self.finalized or expr in self.old_array_to_new_array:
                return self.old_array_to_new_array[expr]

        return super().map_index_lambda(
                expr.copy(expr=ConstantSizer()(expr.expr)))

    def map_concatenate(self, expr):
        from pytato.tags import ImplStored, PrefixNamed
        return super().map_concatenate(expr).tagged(
                (ImplStored(), PrefixNamed("concat")))


def test_trace_fft(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    n = 33
    x = pt.make_placeholder("x", n, dtype=np.complex128)

    from pymbolic.algorithm import fft
    result = fft(x, custom_np=pt,
            wrap_intermediate_with_level=(
                lambda level, ary: ary.tagged(FFTIntermediate(level))))

    fft_vec_gatherer = FFTVectorGatherer(n)
    fft_vec_gatherer(result)

    mapper = FFTRealizationMapper(fft_vec_gatherer)

    result = mapper(result)

    prg = pt.generate_loopy(result).program

    x = np.random.randn(n).astype(np.complex128)
    evt, (result,) = prg(queue, x=x)

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
