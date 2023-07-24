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

import pytest
#from pytools.graph import CycleError
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
import pytest  # noqa
import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import pytato as pt
import sys
import os


# {{{ mpi test infrastructure

def run_test_with_mpi(num_ranks, f, *args, extra_env_vars=None):
    pytest.importorskip("mpi4py")

    if extra_env_vars is None:
        extra_env_vars = {}

    from pickle import dumps
    from base64 import b64encode

    from subprocess import check_call

    env_vars = {
            "RUN_WITHIN_MPI": "1",
            "INVOCATION_INFO": b64encode(dumps((f, args))).decode(),
            }
    env_vars.update(extra_env_vars)

    # NOTE: CI uses OpenMPI; -x to pass env vars. MPICH uses -env
    check_call([
        "mpiexec", "-np", str(num_ranks),
        "--oversubscribe",
        ] + [
            item
            for env_name, env_val in env_vars.items()
            for item in ["-x", f"{env_name}={env_val}"]
        ] + [sys.executable, "-m", "mpi4py", __file__])


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

    halo = pt.staple_distributed_send(x, dest_rank=(rank-1) % size, comm_tag=42,
            stapled_to=pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=int))

    y = x+halo

    # Find the partition
    outputs = pt.make_dict_of_named_arrays({"out": y})
    distributed_parts = pt.find_distributed_partition(comm, outputs)
    prg_per_partition = pt.generate_code_for_partition(distributed_parts)

    # Execute the distributed partition
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    context = pt.execute_distributed_partition(distributed_parts, prg_per_partition,
                                             queue, comm)

    final_res = context["out"].get(queue)

    # All ranks generate the same random numbers (same seed).
    np.testing.assert_allclose(x_in*2, final_res)

# }}}

def test_distributed_partioner_counts():
    _do_test_distributed_partioner_counts(1)

def _do_test_distributed_partioner_counts(max_size):
    from pytato.distributed.partition import schedule_wrapper 
    sizes = np.logspace(0,4,10,dtype=int)
    count_list = np.zeros(len(sizes))
    for i,tree_size in enumerate(sizes):
        counts = 0
        needed_ids = {i: set() for i in range(int(tree_size))}
        for key in needed_ids.keys():
            needed_ids[key] = set([key-1]) if key > 0 else set()
        comm_batches = schedule_wrapper(needed_ids,counts)
        count_list[i] = counts
        print(counts)

    # Now to do the fitting.
    print(sizes)
    print(count_list)
    coefficients = np.polyfit(sizes,count_list, 4)
    
    import numpy.linalg as la
    nonlinear_norm_frac = la.norm(coefficients[:-2], 2)/la.norm(coefficients, 2)
    print(nonlinear_norm_frac)
    assert nonlinear_norm_frac < 0.01

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
            tag = (comm_tag, _RandomDAGTag)  # noqa: B023

            inner = make_random_dag(rdagc)
            return pt.staple_distributed_send(
                    inner, dest_rank=(rank-1) % size, comm_tag=tag,
                    stapled_to=pt.make_distributed_recv(
                        src_rank=(rank+1) % size, comm_tag=tag,
                        shape=inner.shape, dtype=inner.dtype))

        rdagc_comm = RandomDAGContext(np.random.default_rng(seed=seed),
                axis_len=axis_len, use_numpy=False,
                additional_generators=[
                    (comm_fake_prob, gen_comm)
                    ])
        pt_dag = pt.make_dict_of_named_arrays(
            {"result": make_random_dag(rdagc_comm)})
        x_comm = pt.transform.materialize_with_mpms(pt_dag)

        distributed_partition = pt.find_distributed_partition(comm, x_comm)
        pt.verify_distributed_partition(comm, distributed_partition)

        # Transform symbolic tags into numeric ones for MPI
        distributed_partition, _new_mpi_base_tag = pt.number_distributed_tags(
                comm,
                distributed_partition,
                base_tag=comm_tag)

        prg_per_partition = pt.generate_code_for_partition(distributed_partition)

        context = pt.execute_distributed_partition(
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

    parts = pt.find_distributed_partition(comm, dag)
    assert len(parts.parts) == 1
    prg_per_partition = pt.generate_code_for_partition(parts)
    out_dict = pt.execute_distributed_partition(
            parts, prg_per_partition, queue, comm)

    np.testing.assert_allclose(out_dict["out1"], 2 * x_np)
    np.testing.assert_allclose(out_dict["out2"], 8 * x_np)


def test_dag_with_no_comm_nodes():
    run_test_with_mpi(2, _test_dag_with_no_comm_nodes_inner)

# }}}


# {{{ test DAG with duplicated output arrays

def _test_dag_with_duplicated_output_arrays_inner(ctx_factory):
    from numpy.random import default_rng
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    rng = default_rng()
    x_np = rng.random((10, 4))
    x = pt.make_data_wrapper(x_np)

    # {{{ construct the DAG

    y = 2 * x
    out = 4 * y
    dag = pt.make_dict_of_named_arrays({"out1": out, "out2": out})

    # }}}

    parts = pt.find_distributed_partition(comm, dag)
    prg_per_partition = pt.generate_code_for_partition(parts)
    out_dict = pt.execute_distributed_partition(
            parts, prg_per_partition, queue, comm)

    np.testing.assert_allclose(out_dict["out1"], 8 * x_np)
    np.testing.assert_allclose(out_dict["out2"], 8 * x_np)


def test_dag_with_duplicated_output_arrays():
    run_test_with_mpi(2, _test_dag_with_duplicated_output_arrays_inner)

# }}}


# {{{ test DAG with a receive as an output

def _test_dag_with_recv_as_output_inner(ctx_factory):
    from numpy.random import default_rng
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # {{{ construct the DAG

    if comm.rank == 0:
        rng = default_rng()
        x_np = rng.random((10, 4))
        x = pt.make_data_wrapper(cla.to_device(queue, x_np))
        y = 2 * x
        send = pt.staple_distributed_send(
            y, dest_rank=1, comm_tag=42,
            stapled_to=pt.ones(10))
        dag = pt.make_dict_of_named_arrays({"y": y, "send": send})
    else:
        y = pt.make_distributed_recv(
            src_rank=0, comm_tag=42,
            shape=(10, 4), dtype=np.float64)
        dag = pt.make_dict_of_named_arrays({"y": y})

    # }}}

    parts = pt.find_distributed_partition(comm, dag)
    prg_per_partition = pt.generate_code_for_partition(parts)
    out_dict = pt.execute_distributed_partition(
            parts, prg_per_partition, queue, comm)

    if comm.rank == 0:
        comm.bcast(x_np)
    else:
        x_np = comm.bcast(None)

    np.testing.assert_allclose(out_dict["y"].get(), 2 * x_np)


def test_dag_with_recv_as_output():
    run_test_with_mpi(2, _test_dag_with_recv_as_output_inner)

# }}}


# {{{ test DAG with a materialized array promoted to a part output

def _test_dag_with_materialized_array_promoted_to_part_output_inner(ctx_factory):
    from numpy.random import default_rng
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # {{{ construct the DAG

    if comm.rank == 0:
        rng = default_rng()
        x_np = rng.random((10, 4))
        x = pt.make_data_wrapper(cla.to_device(queue, x_np))
        y = 2 * x
        # Force y to be materialized
        from pytato.tags import ImplStored
        y = y.tagged(ImplStored())
        z = 2 * y
        send = pt.staple_distributed_send(
            z, dest_rank=1, comm_tag=42,
            stapled_to=pt.ones(10))
        w = pt.make_distributed_recv(
            src_rank=1, comm_tag=43,
            shape=(10, 4), dtype=np.float64)
        q = y + w
        dag = pt.make_dict_of_named_arrays({"q": q, "send": send})
    else:
        z = pt.make_distributed_recv(
            src_rank=0, comm_tag=42,
            shape=(10, 4), dtype=np.float64)
        w = 2 * z
        send = pt.staple_distributed_send(
            w, dest_rank=0, comm_tag=43,
            stapled_to=pt.ones(10))
        q = z/2 + w
        dag = pt.make_dict_of_named_arrays({"q": q, "send": send})

    # }}}

    parts = pt.find_distributed_partition(comm, dag)
    prg_per_partition = pt.generate_code_for_partition(parts)
    out_dict = pt.execute_distributed_partition(
            parts, prg_per_partition, queue, comm)

    if comm.rank == 0:
        # Is this too fragile?
        # part 0 should return a materialized array and a send
        assert len(parts.parts[0].output_names) == 2
        # part 1 should take a materialized array and a recv
        assert len(parts.parts[1].partition_input_names) == 2

    if comm.rank == 0:
        comm.bcast(x_np)
    else:
        x_np = comm.bcast(None)

    np.testing.assert_allclose(out_dict["q"].get(), 10 * x_np)


def test_dag_with_materialized_array_promoted_to_part_output():
    run_test_with_mpi(
        2, _test_dag_with_materialized_array_promoted_to_part_output_inner)

# }}}


# {{{ test DAG with multiple send nodes per sent array

def _test_dag_with_multiple_send_nodes_per_sent_array_inner(ctx_factory):
    from numpy.random import default_rng
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # {{{ construct the DAG

    if comm.rank == 0:
        rng = default_rng()
        x_np = rng.random((10, 4))
        x = pt.make_data_wrapper(cla.to_device(queue, x_np))
        y = 2 * x
        send1 = pt.staple_distributed_send(
            y, dest_rank=1, comm_tag=42,
            stapled_to=pt.ones(10))
        send2 = pt.staple_distributed_send(
            y, dest_rank=2, comm_tag=42,
            stapled_to=pt.ones(10))
        z = 4 * y
        dag = pt.make_dict_of_named_arrays({"z": z, "send1": send1, "send2": send2})
    else:
        y = pt.make_distributed_recv(
            src_rank=0, comm_tag=42,
            shape=(10, 4), dtype=np.float64)
        z = 4 * y
        dag = pt.make_dict_of_named_arrays({"z": z})

    # }}}

    parts = pt.find_distributed_partition(comm, dag)
    pt.verify_distributed_partition(comm, parts)
    prg_per_partition = pt.generate_code_for_partition(parts)
    out_dict = pt.execute_distributed_partition(
            parts, prg_per_partition, queue, comm)

    if comm.rank == 0:
        comm.bcast(x_np)
    else:
        x_np = comm.bcast(None)

    np.testing.assert_allclose(out_dict["z"].get(), 8 * x_np)


def test_dag_with_multiple_send_nodes_per_sent_array():
    run_test_with_mpi(3, _test_dag_with_multiple_send_nodes_per_sent_array_inner)

# }}}


# {{{ test DAG with periodic communication

def _test_dag_with_periodic_communication_inner(ctx_factory):
    from numpy.random import default_rng
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    # {{{ construct the DAG

    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = default_rng()
    x_np = rng.random((10, 4))
    x = pt.make_data_wrapper(cla.to_device(queue, x_np))

    x_plus = pt.staple_distributed_send(x, dest_rank=(rank-1) % size, comm_tag=42,
            stapled_to=pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(10, 4),
                dtype=np.float64))

    y = x + x_plus

    y_plus = pt.staple_distributed_send(y, dest_rank=(rank-1) % size, comm_tag=43,
            stapled_to=pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=43, shape=(10, 4),
                dtype=np.float64))

    z = y + y_plus

    dag = pt.make_dict_of_named_arrays({"z": z})

    # }}}

    parts = pt.find_distributed_partition(comm, dag)
    pt.verify_distributed_partition(comm, parts)
    prg_per_partition = pt.generate_code_for_partition(parts)
    out_dict = pt.execute_distributed_partition(
            parts, prg_per_partition, queue, comm)

    comm.isend(x_np, dest=(rank-1) % size, tag=44)
    ref_x_plus_np = comm.recv(source=(rank+1) % size, tag=44)

    ref_y_np = x_np + ref_x_plus_np

    comm.isend(ref_y_np, dest=(rank-1) % size, tag=45)
    ref_y_plus_np = comm.recv(source=(rank+1) % size, tag=45)

    ref_res = ref_y_np + ref_y_plus_np

    np.testing.assert_allclose(out_dict["z"].get(), ref_res)


@pytest.mark.parametrize("num_ranks", [2, 3])
def test_dag_with_periodic_communication(num_ranks):
    run_test_with_mpi(num_ranks, _test_dag_with_periodic_communication_inner)

# }}}


# {{{ test deterministic partitioning

def _gather_random_dist_partitions(ctx_factory):
    import mpi4py.MPI as MPI

    comm = MPI.COMM_WORLD

    seed = int(os.environ["PYTATO_DAG_SEED"])
    from testlib import get_random_pt_dag_with_send_recv_nodes
    dag = get_random_pt_dag_with_send_recv_nodes(
        seed, rank=comm.rank, size=comm.size,
        convert_dws_to_placeholders=True)

    my_partition = pt.find_distributed_partition(comm, dag)

    all_partitions = comm.gather(my_partition)

    from pickle import dump
    if comm.rank == 0:
        with open(os.environ["PYTATO_PARTITIONS_DUMP_FN"], "wb") as outf:
            dump(all_partitions, outf)


@pytest.mark.parametrize("seed", list(range(10)))
def test_deterministic_partitioning(seed):
    import os
    from pickle import load
    from pytools import is_single_valued

    partitions_across_seeds = []
    partitions_dump_fn = f"tmp-partitions-{os.getpid()}.pkl"

    for hashseed in [234, 241, 9222, 5]:
        run_test_with_mpi(2, _gather_random_dist_partitions, extra_env_vars={
            "PYTATO_DAG_SEED": str(seed),
            "PYTHONHASHSEED": str(hashseed),
            "PYTATO_PARTITIONS_DUMP_FN": partitions_dump_fn,
            })

        with open(partitions_dump_fn, "rb") as inf:
            partitions_across_seeds.append(load(inf))
            os.unlink(partitions_dump_fn)

    assert is_single_valued(partitions_across_seeds)

# }}}


# {{{ test Kaushik's MWE

def test_kaushik_mwe():
    run_test_with_mpi(2, _do_test_kaushik_mwe)


def _do_test_kaushik_mwe(ctx_factory):
    # from https://github.com/inducer/pytato/pull/393#issuecomment-1324642248
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        send_rank = 1
        recv_rank = 1

        recv = pt.make_distributed_recv(
                    src_rank=recv_rank, comm_tag=42,
                    shape=(10,), dtype=np.float64)
        y = 2*recv

        send = pt.staple_distributed_send(
            y, dest_rank=send_rank, comm_tag=43,
            stapled_to=pt.ones(10))
        out = pt.make_dict_of_named_arrays({"out": send})
    elif comm.rank == 1:
        send_rank = 0
        recv_rank = 0
        x = pt.make_data_wrapper(np.ones(10))

        send = pt.staple_distributed_send(
            2*x, dest_rank=send_rank, comm_tag=42,
            stapled_to=pt.zeros(10))
        recv = pt.make_distributed_recv(
                    src_rank=recv_rank, comm_tag=43,
                    shape=(10,), dtype=np.float64)
        out = pt.make_dict_of_named_arrays({"out1": send, "out2": recv})
    else:
        raise AssertionError()

    distributed_parts = pt.find_distributed_partition(comm, out)

    pt.verify_distributed_partition(comm, distributed_parts)
    prg_per_partition = pt.generate_code_for_partition(distributed_parts)

    # Execute the distributed partition
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    pt.execute_distributed_partition(distributed_parts, prg_per_partition,
                                            queue, comm,
                                            input_args={})

# }}}


# {{{ test verify_distributed_partition

def test_verify_distributed_partition():
    run_test_with_mpi(2, _do_verify_distributed_partition)


def _do_verify_distributed_partition(ctx_factory):
    from mpi4py import MPI  # pylint: disable=import-error
    comm = MPI.COMM_WORLD
    from pytato.distributed.verify import (DuplicateSendError,
                DuplicateRecvError, MissingSendError, MissingRecvError)

    rank = comm.Get_rank()
    size = comm.Get_size()

    # {{{ test unmatched recv

    y = pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=int)

    outputs = pt.make_dict_of_named_arrays({"out": y})
    with pytest.raises(MissingSendError):
        pt.find_distributed_partition(comm, outputs)

    # }}}

    comm.barrier()

    # {{{ test unmatched send

    x = pt.make_placeholder("x", (4, 4), int)
    send = pt.staple_distributed_send(x,
                dest_rank=(rank-1) % size, comm_tag=42, stapled_to=x)

    outputs = pt.make_dict_of_named_arrays({"out": send})
    with pytest.raises(MissingRecvError):
        pt.find_distributed_partition(comm, outputs)

    # }}}

    comm.barrier()

    # {{{ test duplicate recv

    recv2 = pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=float)

    send = pt.staple_distributed_send(recv2, dest_rank=(rank-1) % size, comm_tag=42,
            stapled_to=pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=int))

    outputs = pt.make_dict_of_named_arrays({"out": x+send})
    with pytest.raises(DuplicateRecvError):
        pt.find_distributed_partition(comm, outputs)

    # }}}

    comm.barrier()

    # {{{ test duplicate send

    send = pt.staple_distributed_send(x, dest_rank=(rank-1) % size, comm_tag=42,
            stapled_to=pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=int))

    send2 = pt.staple_distributed_send(x,
                dest_rank=(rank-1) % size, comm_tag=42, stapled_to=x)

    outputs = pt.make_dict_of_named_arrays({"out": send+send2})
    with pytest.raises(DuplicateSendError):
        pt.find_distributed_partition(comm, outputs)

    # }}}

    comm.barrier()

    # {{{ test cycle

    recv = pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=42, shape=(4, 4), dtype=int)

    recv2 = pt.make_distributed_recv(
                src_rank=(rank+1) % size, comm_tag=43, shape=(4, 4), dtype=int)

    send = pt.staple_distributed_send(recv2, dest_rank=(rank-1) % size, comm_tag=42,
            stapled_to=recv)

    send2 = pt.staple_distributed_send(recv2, dest_rank=(rank-1) % size, comm_tag=43,
            stapled_to=recv)

    outputs = pt.make_dict_of_named_arrays({"out": send+send2})

    print(f"BEGIN {comm.rank}")
    with pytest.raises(CycleError):
        pt.find_distributed_partition(comm, outputs)

    # }}}

# }}}


if __name__ == "__main__":
    #test_distributed_partioner_counts()
    if "RUN_WITHIN_MPI" in os.environ:
        run_test_with_mpi_inner()
    elif len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
