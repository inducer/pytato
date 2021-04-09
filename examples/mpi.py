from mpi4py import MPI
comm = MPI.COMM_WORLD
import pytato as pt
from pytato.transform import WalkMapper


def advect(x, halo, bnd):
    return bnd*x*halo


class SendFeederFinder(WalkMapper):
    """
    .. attribute:: node_to_fed_sends

        :class:`dict`, maps each node in the graph to the set of sends that
        depend on it
    """
    def __init__(self):
        self.node_to_fed_sends = {}

    def visit(self, expr, fed_sends):
        self.node_to_fed_sends[expr] = fed_sends
        return True

    def map_distributed_send(self, expr, fed_sends):
        fed_sends = fed_sends | {expr}
        super().map_distributed_send(expr, fed_sends)

    def __call__(self, expr):
        return self.rec(expr, set())


def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    x = pt.make_placeholder(shape=(10,), dtype=float)
    bnd = pt.make_distributed_send(x[0], dest_rank=(rank-1) % size, comm_tag="halo")

    halo = pt.make_distributed_recv(src_rank=(rank+1) % size, comm_tag="halo",
            shape=(), dtype=float,
            tags=frozenset({pt.AdditionalOutput(bnd, prefix="send")}))

    y = advect(x, halo, bnd)
    #print(pt.generate_loopy({"y": y}).program)

    sff = SendFeederFinder()
    sff(y)

    print(sff.node_to_fed_sends)
    1/0
    dot_code = pt.get_dot_graph(y)

    from graphviz import Source
    src = Source(dot_code)
    src.view()


if __name__ == "__main__":
    main()
