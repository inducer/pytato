#!/usr/bin/env python

import numpy as np

import pytato as pt


n = pt.make_size_param("n")
a = pt.make_placeholder(name="a", shape=(n, n), dtype=np.float64)

a2a = a@(2*a)
aat = a@a.T
result = pt.make_dict_of_named_arrays({"a2a": a2a, "aat": aat})


# {{{ execute with loopy/pyopencl

import pyopencl as cl
import pyopencl.array as cla


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# simple interface
prg = pt.generate_loopy(result)
a = np.random.randn(20, 20)
_, out = prg(queue, a=a)
assert np.allclose(out["a2a"], a@(2*a))
assert np.allclose(out["aat"], a@a.T)

# interface for efficient execution
import loopy as lp


prg = pt.generate_loopy(
        result, options=lp.Options(no_numpy=True, return_dict=True)
        ).bind_to_context(ctx)
a = np.random.randn(20, 20)
a_dev = cla.to_device(queue, a)
_, out = prg(queue, a=a_dev)
assert np.allclose(out["a2a"].get(), a@(2*a))
assert np.allclose(out["aat"].get(), a@a.T)

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
