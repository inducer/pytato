import numpy as np
import pytato as pt
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

for axis in (None, 1, 0):
    for redn in ("sum", "amax", "amin", "prod"):
        x_in = np.random.randn(20, 10)

        x = pt.make_data_wrapper(x_in)

        x2 = x.T @ x

        np_func = getattr(np, redn)
        pt_func = getattr(pt, redn)
        prg = pt.generate_loopy(pt_func(x2, axis=axis), cl_device=queue.device)

        evt, (out,) = prg(queue)
        evt.wait()

        print("redn =", redn, ", axis =", axis, ", max error =",
              np.amax(abs(out - np_func(x_in.T @ x_in, axis))))
