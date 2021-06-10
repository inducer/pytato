from __future__ import annotations

__copyright__ = """Copyright (C) 2020 Matt Wala"""

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

__doc__ = """
.. currentmodule:: pytato

.. autofunction:: generate_loopy

.. currentmodule:: pytato.target.loopy

.. autoclass:: LoopyTarget
.. autoclass:: LoopyPyOpenCLTarget
.. autoclass:: BoundPyOpenCLProgram
"""

import sys
from dataclasses import dataclass

from typing import Any, Mapping, Optional, Union

from pytato.target import Target, BoundProgram

import loopy


# set in doc/conf.py
if getattr(sys, "PYTATO_BUILDING_SPHINX_DOCS", False):
    # Avoid import unless building docs to avoid creating a hard
    # dependency on pyopencl, when Loopy can run fine without.
    import pyopencl


class LoopyTarget(Target):
    """An :mod:`loopy` target.

    .. automethod:: get_loopy_target
    """

    def get_loopy_target(self) -> "loopy.TargetBase":
        """Return the corresponding :mod:`loopy` target."""
        raise NotImplementedError


class LoopyPyOpenCLTarget(LoopyTarget):
    """A :mod:`pyopencl` code generation target.

    .. attribute:: device

        The :mod:`pyopencl` device used to construct the
        :class:`loopy.PyOpenCLTarget`, or *None*.
    """

    def __init__(self, device: Optional["pyopencl.Device"] = None):
        import pyopencl as cl
        if device is not None and not isinstance(device, cl.Device):
            raise TypeError("device must be cl.Device or None")
        self.device = device

    def get_loopy_target(self) -> "loopy.LoopyPyOpenCLTarget":
        import loopy as lp
        return lp.PyOpenCLTarget(self.device)

    def bind_program(self, program: Union["loopy.Program", "loopy.LoopKernel"],
            bound_arguments: Mapping[str, Any]) -> BoundProgram:
        return BoundPyOpenCLProgram(program=program,
                bound_arguments=bound_arguments,
                target=self)


@dataclass(init=True, repr=False, eq=False)
class BoundPyOpenCLProgram(BoundProgram):
    """A wrapper around a :mod:`loopy` kernel for execution with :mod:`pyopencl`.

    .. automethod:: __call__
    """
    def __call__(self, queue: "pyopencl.CommandQueue",
            *args: Any, **kwargs: Any) -> Any:
        """Convenience function for launching a :mod:`pyopencl` computation."""

        if set(kwargs.keys()) & set(self.bound_arguments.keys()):
            raise ValueError("Got arguments that were previously bound: "
                    f"{set(kwargs.keys()) & set(self.bound_arguments.keys())}.")

        updated_kwargs = dict(self.bound_arguments)
        updated_kwargs.update(kwargs)
        if not isinstance(self. program, loopy.LoopKernel):
            updated_kwargs.setdefault("entrypoint", "_pt_kernel")

        return self.program(queue, *args, **updated_kwargs)

    @property
    def kernel(self) -> "loopy.LoopKernel":
        if isinstance(self.program, loopy.LoopKernel):
            return self.program
        else:
            return self.program["_pt_kernel"]


# vim: foldmethod=marker
