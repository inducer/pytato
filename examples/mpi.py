from mpi4py import MPI
comm = MPI.COMM_WORLD
import pytato as pt


def advect(x, halo, bnd):
    return 4*x


ns = pt.Namespace()

rank = comm.Get_rank()
size = comm.Get_size()
x = pt.make_placeholder(ns, shape=(10,), dtype=float)
bnd = pt.make_distributed_send(x[0], dest_rank=(rank-1) % size, comm_tag="halo")

halo = pt.make_distributed_recv(src_rank=(rank+1) % size, comm_tag="halo", shape=(),
                          dtype=float, tags=pt.AdditionalOutput(bnd, prefix="send"))

y = advect(x, halo, bnd)
print(pt.generate_loopy({"y": y}).program)

dot_code = pt.get_dot_graph(y)

from graphviz import Source

src = Source(dot_code)

src.view()
