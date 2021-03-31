from mpi4py import MPI
comm = MPI.COMM_WORLD
import pytato as pt


def advect(x, halo):
    return x


ns = pt.Namespace()

rank = comm.Get_rank()
size = comm.Get_size()
x = pt.make_placeholder(ns, shape=(10,), dtype=float)
bnd = pt.DistributedSend(x[0], to_rank=(rank-1) % size, tag="halo")

halo = pt.DistributedRecv(src_rank=(rank+1) % size, tag="halo", shape=(),
                          dtype=float, tags=pt.AdditionalOutput(bnd, prefix="send"))

y = advect(x, halo)
print(pt.generate_loopy({"y": y}).program)
