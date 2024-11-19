from __future__ import annotations


__copyright__ = """
Copyright (C) 2022 Kaushik Kulkarni
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import ast
from collections.abc import Mapping

from pytato.array import Array, DictOfNamedArrays
from pytato.target.python import BoundJAXPythonProgram, JAXPythonTarget
from pytato.target.python.numpy_like import generate_numpy_like


__doc__ = """
.. autofunction:: generate_jax
"""


def generate_jax(expr: Array | Mapping[str, Array] | DictOfNamedArrays,
                 *,
                 target: JAXPythonTarget | None = None,
                 jit: bool = False,
                 function_name: str = "_pt_kernel",
                 show_code: bool = False,
                 colorize_show_code: bool = True,
                 ) -> BoundJAXPythonProgram:
    """
    Returns a :class:`pytato.target.python.BoundJAXPythonProgram` for the array
    expressions in *expr*.

    :arg jit: If *True*, the generated function is decorated with
        :func:`jax.jit`.
    :arg function: Name of the entrypoint function in the generated code.
    :arg show_code: If *True*, the generated code is printed to ``stdout``.
    """
    if target is None:
        target = JAXPythonTarget()

    extra_preambles = []
    decorators = []

    if jit:
        extra_preambles.append(ast.ImportFrom(module="jax",
                                              names=[ast.alias(
                                                  "jit",
                                                  asname="_pt_jax_jit")],
                                              level=0))
        decorators.append("_pt_jax_jit")

    return generate_numpy_like(expr, target=target,  # type: ignore[return-value]
                               function_name=function_name,
                               show_code=show_code,
                               extra_preambles=tuple(extra_preambles),
                               entrypoint_decorators=tuple(decorators),
                               colorize_show_code=colorize_show_code)
