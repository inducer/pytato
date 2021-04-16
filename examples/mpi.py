from mpi4py import MPI
comm = MPI.COMM_WORLD
import pytato as pt
from pytato.transform import Mapper, WalkMapper

from pytato.array import Array


def advect(*args):
    return sum([a for a in args])


class GraphToDictMapper(WalkMapper):
    """
    .. attribute:: graph_dict

        :class:`dict`, maps each node in the graph to the set of sends that
        depend on it
    """
    def __init__(self):
        self.graph_dict = {}

    def visit(self, expr, children):
        self.graph_dict[expr] = children
        return True

    def map_placeholder(self, expr, children):
        children = children | {expr}
        super().map_placeholder(expr, children)

    def map_slice(self, expr, children):
        children = children | {expr}
        super().map_slice(expr, children)

    def map_index_lambda(self, expr, children):
        children = children | {expr}
        super().map_index_lambda(expr, children)

    def map_distributed_send(self, expr, children):
        children = children | {expr}
        super().map_distributed_send(expr, children)

    def map_distributed_recv(self, expr, children):
        children = children | {expr}
        super().map_distributed_recv(expr, children)

    def __call__(self, expr):
        return self.rec(expr, set())  # Why can't we use list here?


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
        self.rec(expr.data, fed_sends)
        # super().map_distributed_send(expr, fed_sends)

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

    gdm = GraphToDictMapper()
    gdm(y)

    for k, v in gdm.graph_dict.items():
        print(k, v)
    # print(gdm.graph_dict)

    def partition_sends(node_to_fed_sends):
        sends_to_nodes = {}

        for x in node_to_fed_sends:
            if len(sff.node_to_fed_sends[x]) > 0:
                send = next(iter(sff.node_to_fed_sends[x]))
                sends_to_nodes.setdefault(send, []).append(x)

        return sends_to_nodes

    # print("========")
    # print(partition_sends(sff.node_to_fed_sends))

    dot_code = pt.get_dot_graph(y)

    from graphviz import Source
    src = Source(dot_code)
    # src.view()


if __name__ == "__main__":
    main()
