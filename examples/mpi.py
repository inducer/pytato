from mpi4py import MPI
comm = MPI.COMM_WORLD
import pytato as pt
from pytato.transform import Mapper, WalkMapper

from pytato.array import Array, IndexLambda


def advect(*args):
    return sum([a for a in args])


class GraphToDictMapper(Mapper):
    """
    .. attribute:: graph_dict

        :class:`dict`, maps each node in the graph to the set of directly connected
        nodes.
    """
    def __init__(self):
        self.graph_dict = {}

    def map_placeholder(self, expr, *args):
        children = set()

        for dim in expr.shape:
            if isinstance(dim, Array):
                children = children | {dim}
                self.rec(dim, *args)

        for c in children:
            self.graph_dict.setdefault(c, set()).add(expr)

    def map_slice(self, expr, *args):
        self.graph_dict.setdefault(expr.array, set()).add(expr)
        self.rec(expr.array)

    def map_index_lambda(self, expr: IndexLambda, *args) -> None:
        children = set()

        for child in expr.bindings.values():
            children = children | {child}
            self.rec(child)

        for dim in expr.shape:
            if isinstance(dim, Array):
                children = children | {dim}
                self.rec(dim)

        for c in children:
            self.graph_dict.setdefault(c, set()).add(expr)

    def map_distributed_send(self, expr, *args):
        self.graph_dict.setdefault(expr.data, set()).add(expr)
        self.rec(expr.data)

    def map_distributed_recv(self, expr, *args):
        self.graph_dict.setdefault(expr.data, set()).add(expr)
        self.rec(expr.data)

    def __call__(self, expr):
        return self.rec(expr)


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


class RecvFeedingFinder(WalkMapper):
    """
    .. attribute:: node_to_feeding_recvs

        :class:`dict`, maps each node in the graph to the set of recv's that
        feed it
    """
    def __init__(self):
        self.node_to_feeding_recvs = {}

    def visit(self, expr, receiving_nodes):
        self.node_to_feeding_recvs[expr] = receiving_nodes
        return True

    def map_distributed_recv(self, expr, receiving_nodes):
        receiving_nodes = receiving_nodes | {expr}
        super().map_distributed_recv(expr, receiving_nodes)

    def __call__(self, expr):
        return self.rec(expr, set())


class PartitionFinder(Mapper):
    """
    .. attribute:: node_to_feeding_recvs

        :class:`dict`, maps each node in the graph to the set of recv's that
        feed it
    """

    def __init__(self, node_to_fed_sends, node_to_feeding_recvs):
        self.node_to_feeding_recvs = node_to_feeding_recvs
        self.node_to_fed_sends = node_to_fed_sends

    def check_partition(self, expr1, expr2):
        if (self.node_to_fed_sends[expr1] != self.node_to_fed_sends[expr2] or
             self.node_to_feeding_recvs[expr1] != self.node_to_feeding_recvs[expr2]):
            print("partition", expr1, expr2)
            return True
        print("NO partition", expr1, expr2)
        return False

    def map_distributed_send(self, expr, *args):
        self.check_partition(expr, expr.data)

        self.rec(expr.data)

    def map_distributed_recv(self, expr, *args):
        self.check_partition(expr, expr.data)
        self.rec(expr.data)

    def map_slice(self, expr, *args):
        self.check_partition(expr, expr.array)
        self.rec(expr.array)

    def map_placeholder(self, expr, *args):
        for dim in expr.shape:
            if isinstance(dim, Array):
                self.check_partition(expr, dim)
                self.rec(dim, *args)

    def map_index_lambda(self, expr: IndexLambda, *args) -> None:
        for child in expr.bindings.values():
            self.check_partition(expr, child)
            self.rec(child)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.check_partition(expr, dim)
                self.rec(dim)

    def __call__(self, expr):
        return self.rec(expr, set())

# Partition the pytato graph whenever an edge (node1, node2) satisfies
# node_to_fed_sends[node1] != node_to_fed_sends[node2] or
# node_to_feeding_recvs[node1] != node_to_feeding_recvs[node2] If an edge
# crosses a partition boundary, replace the depended-upon node (that nominally
# lives in the other partition) with a Placeholder that lives in the current
# partition. For each partition, collect the placeholder names that it’s
# supposed to compute.


def main():
    rank = comm.Get_rank()
    size = comm.Get_size()
    x = pt.make_placeholder(shape=(10,), dtype=float)
    bnd = pt.make_distributed_send(x[0], dest_rank=(rank-1) % size, comm_tag="halo")

    halo = pt.make_distributed_recv(x[9], src_rank=(rank+1) % size, comm_tag="halo",
            shape=(), dtype=float)

    y = x+bnd+halo

    bnd2 = pt.make_distributed_send(y[0], dest_rank=(rank-1) % size, comm_tag="halo")

    halo2 = pt.make_distributed_recv(y[9], src_rank=(rank+1) % size, comm_tag="halo",
            shape=(), dtype=float)
    y += bnd2 + halo2

    sff = SendFeederFinder()
    sff(y)

    for k, v in sff.node_to_fed_sends.items():
        print(k, v)

    print("========")

    rff = RecvFeedingFinder()
    rff(y)

    for k, v in rff.node_to_feeding_recvs.items():
        print(k, v)

    # gdm = GraphToDictMapper()
    # gdm(y)

    print("========")
    # for k, v in gdm.graph_dict.items():
    #     print(k, v)
    # print(gdm.graph_dict)

    pf = PartitionFinder(sff.node_to_fed_sends, rff.node_to_feeding_recvs)
    pf(y)

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
    src.view()


if __name__ == "__main__":
    main()
