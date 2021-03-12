import numpy as np
import pytato as pt

ns = pt.Namespace()
pt.make_size_param(ns, "n")
a = pt.make_placeholder(ns, name="a", shape="n,n", dtype=np.float64)

a2a = a@(2*a)
aat = a@a.T
result = pt.DictOfNamedArrays({"a2a": a2a, "aat": aat})


# {{{ execute

import pyopencl as cl
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
prg = pt.generate_loopy(result, target=pt.LoopyPyOpenCLTarget(queue))
a = np.random.randn(20, 20)
_, (a2a, aat) = prg(a=a)
assert np.allclose(a2a, a@(2*a))
assert np.allclose(aat, a@a.T)

# }}}

# {{{ generate OpenCL code

prg = pt.generate_loopy(result)

import loopy as lp
print(lp.generate_code_v2(prg.program).device_code())

# }}}
