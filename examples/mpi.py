from mpi4py import MPI
comm = MPI.COMM_WORLD
import pytato as pt
from pytato.transform import WalkMapper


def advect(*args):
    return sum([a for a in args])


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

    def map_distributed_recv(self, expr, fed_sends):
        pass
        # fed_sends = fed_sends | {expr}
        # super().map_distributed_recv(expr, fed_sends)

    def __call__(self, expr):
        return self.rec(expr, set())


def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    x = pt.make_placeholder(shape=(10,), dtype=float)
    bnd = pt.make_distributed_send(x[0], dest_rank=(rank-1) % size, comm_tag="halo")

    halo = pt.make_distributed_recv(src_rank=(rank+1) % size, comm_tag="halo",
            shape=(), dtype=float)
#            # tags=frozenset({pt.AdditionalOutput(bnd, prefix="send")}))

    y = x+bnd+halo

    bnd2 = pt.make_distributed_send(y[0], dest_rank=(rank-1) % size, comm_tag="halo")

    halo2 = pt.make_distributed_recv(src_rank=(rank+1) % size, comm_tag="halo",
            shape=(), dtype=float)
    y += bnd2 + halo2

    # y = advect(x, halo, bnd)
    #print(pt.generate_loopy({"y": y}).program)

    sff = SendFeederFinder()
    sff(y)

    # l_sends = set()

    # for x in sff.node_to_fed_sends:
    #     print(x, sff.node_to_fed_sends[x])
    #     if len(sff.node_to_fed_sends[x]) > 0:
    #         l_sends.add(sff.node_to_fed_sends[x].pop())

    def partition_sends(node_to_fed_sends):
        sends_to_nodes = {}

        for x in node_to_fed_sends:
            if len(sff.node_to_fed_sends[x]) > 0:
                send = sff.node_to_fed_sends[x].pop()
                sends_to_nodes.setdefault(send, []).append(x)

        return sends_to_nodes
        # print(sends_to_nodes)

    print("========")
    # print(l_sends)

    print(partition_sends(sff.node_to_fed_sends))

    dot_code = pt.get_dot_graph(y)

    from graphviz import Source
    src = Source(dot_code)
    src.view()


if __name__ == "__main__":
    main()
