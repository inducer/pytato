import numpy as np
import pytato as pt
import pyopencl as cl


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

spec = "ij,ij->ij"
n = 4
a_in = np.random.rand(n, n)
b_in = np.random.rand(n, n)

a = pt.make_data_wrapper(a_in)
b = pt.make_data_wrapper(b_in)

prg = pt.generate_loopy(pt.einsum(spec, a, b), cl_device=queue.device)

evt, (out,) = prg(queue)
ans = np.einsum(spec, a_in, b_in)

assert np.linalg.norm(out - ans) <= 1e-15
