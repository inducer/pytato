from mpi4py import MPI
comm = MPI.COMM_WORLD
import pytato as pt

ns = pt.Namespace()

rank = comm.Get_rank()
size = comm.Get_size()
x = pt.make_placeholder(ns, shape=(10,), dtype=float)
bnd = pt.DistributedSend(x[0], to=(rank-1)%size, tag="halo")


halo = pt.DistributedRecv(src=(rank+1)%size, tag="halo", shape=(), dtype=float, tags=pt.AdditionalOutput(bnd, prefix="send"))

# left_bnd = pt.tag(x[0], pt.MPISendTag(to=(rank-1)//size, tag="left"))
# rght_bnd = pt.tag(x[9], pt.MPISendTag(to=(rank+1)//size, tag="right"))
# left_halo = pt.mpi_recv(src=(rank-1)//size, tag="right", shape=(), dtype=float)
# rght_halo = pt.mpi_recv(src=(rank+1)//size, tag="left", shape=(), dtype=float)
# y = advect(x, left_halo, rght_halo)
pt.generate_loopy({"left_bnd": left_bnd, "rght_bnd": rght_bnd, "y": y})  # send nodes are "outputs"
