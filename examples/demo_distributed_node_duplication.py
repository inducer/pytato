"""
An example to demonstrate the behavior of
:func:`pytato.find_distrbuted_partition`. One of the key characteristic of the
partitioning routine is to recompute expressions that appear in the multiple
partitions but are not materialized.
"""
import numpy as np

import pytato as pt

size = 2
rank = 0

x1 = pt.make_placeholder("x1", shape=(10, 4), dtype=np.float64)
x2 = pt.make_placeholder("x2", shape=(10, 4), dtype=np.float64)
x3 = pt.make_placeholder("x3", shape=(10, 4), dtype=np.float64)
x4 = pt.make_placeholder("x4", shape=(10, 4), dtype=np.float64)


tmp1 = (x1 + x2).tagged(pt.tags.ImplStored())
tmp2 = tmp1 + x3
# "marking" *tmp2* so that its duplication can be clearly visualized.
tmp2 = tmp2.tagged(pt.tags.Named("tmp2"))
tmp3 = (2 * x4).tagged(pt.tags.ImplStored())
tmp4 = tmp2 + tmp3

recv = pt.staple_distributed_send(tmp4, dest_rank=(rank-1) % size, comm_tag=10,
        stapled_to=pt.make_distributed_recv(
            src_rank=(rank+1) % size, comm_tag=10, shape=(10, 4), dtype=int))

out = tmp2 + recv
result = pt.make_dict_of_named_arrays({"out": out})

partitions = pt.find_distributed_partition(result)

# Visualize *partitions* to see that each of the two partitions contains a node
# named 'tmp2'.
pt.show_dot_graph(partitions)
