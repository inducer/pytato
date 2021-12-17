__copyright__ = """Copyright (C) 2021 Kaushik Kulkarni"""

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

import pytest
import numpy as np
import pytato as pt
from jax.config import config
config.update("jax_enable_x64", True)


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
@pytest.mark.parametrize("jit", ([False, True]))
def test_einsum(spec, argshapes, jit):

    np_operands = [np.random.rand(*argshape) for argshape in argshapes]
    pt_operands = [pt.make_data_wrapper(x_in) for x_in in np_operands]

    np_out = np.einsum(spec, *np_operands)
    pt_expr = pt.einsum(spec, *pt_operands)

    pt_out = pt.generate_jax(pt_expr, jit=jit)()
    assert np_out.shape == pt_out.shape
    np.testing.assert_allclose(np_out, pt_out)


@pytest.mark.parametrize("jit", ([False, True]))
def test_random_dag_against_numpy(jit):
    from testlib import RandomDAGContext, make_random_dag
    axis_len = 5
    from warnings import filterwarnings, catch_warnings
    with catch_warnings():
        # We'd like to know if Numpy divides by zero.
        filterwarnings("error")

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
            dict_named_arys = pt.DictOfNamedArrays({"result": dag})
            dict_named_arys = materialize_with_mpms(dict_named_arys)
            if 0:
                pt.show_dot_graph(dict_named_arys)

            pt_result = pt.generate_jax(dict_named_arys, jit=jit)()

            assert np.allclose(pt_result["result"], ref_result)


@pytest.mark.parametrize("jit", ([False, True]))
def test_placeholders_in_jax(jit):
    from numpy.random import default_rng
    rng = default_rng()

    img = pt.make_placeholder("img", (256, 256, 3), dtype=np.float32)
    scl = pt.make_placeholder("scl", 3, dtype=np.float32)

    img_in = rng.random(size=(256, 256, 3), dtype=np.float32)
    scl_in = rng.random(size=(3,), dtype=np.float32)

    pt_out = pt.generate_jax(img * scl, jit=jit)(img=img_in, scl=scl_in)
    np_out = img_in * scl_in

    np.testing.assert_allclose(pt_out, np_out, rtol=1e-6)
