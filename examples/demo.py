#!/usr/bin/env python

import numpy as np
import pytato as pt

n = pt.make_size_param("n")
a = pt.make_placeholder(name="a", shape=(n, n), dtype=np.float64)

a2a = a@(2*a)
aat = a@a.T
result = pt.DictOfNamedArrays({"a2a": a2a, "aat": aat})


# {{{ execute

import pyopencl as cl
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
prg = pt.generate_loopy(result, cl_device=queue.device)
a = np.random.randn(20, 20)
_, out = prg(queue, a=a)
assert np.allclose(out["a2a"], a@(2*a))
assert np.allclose(out["aat"], a@a.T)

# }}}

# {{{ print generated Loopy and OpenCL code

prg = pt.generate_loopy(result)

print("=============== Loopy code =================")
print(prg.program)
print("============= End Loopy code ===============")

import loopy as lp
print("============== OpenCL code =================")
print(lp.generate_code_v2(prg.program).device_code())
print("============ End OpenCL code ===============")

# }}}
