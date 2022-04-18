__copyright__ = """Copyright (C) 2021 University of Illinois Board of Trustees
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

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
import pyopencl as cl
import numpy as np
import pytato as pt
import sys
import os

from pytato.distributed import (staple_distributed_send, make_distributed_recv,
    find_distributed_partition,
    execute_distributed_partition, number_distributed_tags)

from pytato.partition import generate_code_for_partition


# {{{ mpi test infrastructure

def run_test_with_mpi(num_ranks, f, *args):
    import pytest
    pytest.importorskip("mpi4py")

    from pickle import dumps
    from base64 import b64encode

    invocation_info = b64encode(dumps((f, args))).decode()
    from subprocess import check_call

    # NOTE: CI uses OpenMPI; -x to pass env vars. MPICH uses -env
    check_call([
        "mpiexec", "-np", str(num_ranks),
        "-x", "RUN_WITHIN_MPI=1",
        "-x", f"INVOCATION_INFO={invocation_info}",
        sys.executable, __file__])


def run_test_with_mpi_inner():
    from pickle import loads
    from base64 import b64decode
    f, args = loads(b64decode(os.environ["INVOCATION_INFO"].encode()))

    f(cl.create_some_context, *args)

# }}}


# {{{ "basic" test (similar to distributed example)

def test_distributed_execution_basic():
    run_test_with_mpi(2, _do_test_distributed_execution_basic)


def _do_test_distributed_execution_basic(ctx_factory):
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    rng = np.random.default_rng(seed=27)

    x_in = rng.integers(100, size=(4, 4))
    x = pt.make_data_wrapper(x_in)

    halo = staple_distributed_send(x, dest_rank=(rank-1) % size, comm_tag=42,
            stapled_to=make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=int))

    y = x+halo

    # Find the partition
    outputs = pt.DictOfNamedArrays({"out": y})
    distributed_parts = find_distributed_partition(outputs)
    prg_per_partition = generate_code_for_partition(distributed_parts)

    # Execute the distributed partition
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = execute_distributed_partition(distributed_parts, prg_per_partition,
                                             queue, comm)

    final_res = context["out"].get(queue)

    # All ranks generate the same random numbers (same seed).
    np.testing.assert_allclose(x_in*2, final_res)

# }}}


# {{{ test based on random dag

def test_distributed_execution_random_dag():
    run_test_with_mpi(2, _do_test_distributed_execution_random_dag)


class _RandomDAGTag:
    pass


def _do_test_distributed_execution_random_dag(ctx_factory):
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    rank = comm.Get_rank()
    size = comm.Get_size()

    from testlib import RandomDAGContext, make_random_dag

    axis_len = 4
    comm_fake_prob = 500

    gen_comm_called = False

    ntests = 10
    for i in range(ntests):
        seed = 120 + i
        print(f"Step {i} {seed}")

        # {{{ compute value with communication

        comm_tag = 17

        def gen_comm(rdagc):
            nonlocal gen_comm_called
            gen_comm_called = True

            nonlocal comm_tag
            comm_tag += 1
            tag = (comm_tag, _RandomDAGTag)

            inner = make_random_dag(rdagc)
            return staple_distributed_send(
                    inner, dest_rank=(rank-1) % size, comm_tag=tag,
                    stapled_to=make_distributed_recv(
                        src_rank=(rank+1) % size, comm_tag=tag,
                        shape=inner.shape, dtype=inner.dtype))

        rdagc_comm = RandomDAGContext(np.random.default_rng(seed=seed),
                axis_len=axis_len, use_numpy=False,
                additional_generators=[
                    (comm_fake_prob, gen_comm)
                    ])
        pt_dag = pt.DictOfNamedArrays({"result": make_random_dag(rdagc_comm)})
        x_comm = pt.transform.materialize_with_mpms(pt_dag)

        distributed_partition = find_distributed_partition(x_comm)

        # Transform symbolic tags into numeric ones for MPI
        distributed_partition, _new_mpi_base_tag = number_distributed_tags(
                comm,
                distributed_partition,
                base_tag=comm_tag)

        # Regression check for https://github.com/inducer/pytato/issues/307
        from pytato.distributed import PartIDTag
        for ary in distributed_partition.var_name_to_result.values():
            assert not ary.tags_of_type(PartIDTag)

        prg_per_partition = generate_code_for_partition(distributed_partition)

        context = execute_distributed_partition(
                    distributed_partition, prg_per_partition, queue, comm)

        res_comm = context["result"]

        # }}}

        # {{{ compute ref value without communication

        # compiled evaluation (i.e. use_numpy=False) fails for some of these
        # graphs, cf. https://github.com/inducer/pytato/pull/255
        rdagc_no_comm = RandomDAGContext(np.random.default_rng(seed=seed),
                axis_len=axis_len, use_numpy=True,
                additional_generators=[
                    (comm_fake_prob, lambda rdagc: make_random_dag(rdagc))
                    ])
        res_no_comm_numpy = make_random_dag(rdagc_no_comm)

        # }}}

        if not isinstance(res_comm, np.ndarray):
            res_comm = res_comm.get(queue=queue)

        np.testing.assert_allclose(res_comm, res_no_comm_numpy)

    assert gen_comm_called

# }}}


# {{{ test DAG with no comm nodes

def _test_dag_with_no_comm_nodes_inner(ctx_factory):
    from numpy.random import default_rng
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    rng = default_rng()
    x_np = rng.random((10, 4))
    x = pt.make_data_wrapper(x_np)

    # {{{ construct the DAG

    out1 = 2 * x
    out2 = 4 * out1
    dag = pt.make_dict_of_named_arrays({"out1": out1, "out2": out2})

    # }}}

    parts = find_distributed_partition(dag)
    assert len(parts.parts) == 1
    prg_per_partition = generate_code_for_partition(parts)
    out_dict = execute_distributed_partition(parts, prg_per_partition, queue, comm)

    np.testing.assert_allclose(out_dict["out1"], 2 * x_np)
    np.testing.assert_allclose(out_dict["out2"], 8 * x_np)


def test_dag_with_no_comm_nodes():
    run_test_with_mpi(2, _test_dag_with_no_comm_nodes_inner)

# }}}


if __name__ == "__main__":
    if "RUN_WITHIN_MPI" in os.environ:
        run_test_with_mpi_inner()
    elif len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
