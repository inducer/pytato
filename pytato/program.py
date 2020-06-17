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
.. currentmodule:: pytato.program

Code Generation Targets
-----------------------

.. autoclass:: Target
.. autoclass:: PyOpenCLTarget

Generated Executable Programs
-----------------------------

.. autoclass:: BoundProgram
.. autoclass:: BoundPyOpenCLProgram
"""

from dataclasses import dataclass
import typing
from typing import Any, Mapping, Optional

if typing.TYPE_CHECKING:
    # Skip imports for efficiency.  FIXME: Neither of these work as type stubs
    # are not present. Types are here only as documentation.
    import pyopencl as cl
    import loopy as lp


class Target:
    """An abstract code generation target.

    .. automethod:: get_loopy_target
    .. automethod:: bind_program
    """

    def get_loopy_target(self) -> "lp.TargetBase":
        """Return the corresponding :mod:`loopy` target."""
        raise NotImplementedError

    def bind_program(self, program: "lp.LoopKernel",
            bound_arguments: Mapping[str, Any]) -> BoundProgram:
        """Create a :class:`BoundProgram` for this code generation target.

        :param program: the :mod:`loopy` kernel
        :param bound_arguments: a mapping from argument names to outputs
        """
        raise NotImplementedError


class PyOpenCLTarget(Target):
    """A :mod:`pyopencl` code generation target.

    .. attribute:: queue

        The :mod:`pyopencl` command queue, or *None*.
    """

    def __init__(self, queue: Optional["cl.CommandQueue"] = None):
        self.queue = queue

    def get_loopy_target(self) -> "lp.PyOpenCLTarget":
        import loopy as lp
        device = None
        if self.queue is not None:
            device = self.queue.device
        return lp.PyOpenCLTarget(device)

    def bind_program(self, program: "lp.LoopKernel",
            bound_arguments: Mapping[str, Any]) -> BoundProgram:
        return BoundPyOpenCLProgram(program=program,
                queue=self.queue,
                bound_arguments=bound_arguments,
                target=self)


@dataclass(init=True, repr=False, eq=False)
class BoundProgram:
    """A wrapper around a :mod:`loopy` kernel for execution.

    .. attribute:: program

        The underlying :class:`loopy.LoopKernel`.

    .. attribute:: target

       The code generation target.

    .. attribute:: bound_arguments

        A map from names to pre-bound kernel arguments.

    .. automethod:: __call__
    """

    program: "lp.LoopKernel"
    bound_arguments: Mapping[str, Any]
    target: Target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@dataclass(init=True, repr=False, eq=False)
class BoundPyOpenCLProgram(BoundProgram):
    """A wrapper around a :mod:`loopy` kernel for execution with :mod:`pyopencl`.

    .. attribute:: queue

        A :mod:`pyopencl` command queue.

    .. automethod:: __call__
    """
    queue: Optional["cl.CommandQueue"]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Convenience function for launching a :mod:`pyopencl` computation."""
        if not self.queue:
            raise ValueError("queue must be specified")

        updated_kwargs = dict(self.bound_arguments)
        updated_kwargs.update(kwargs)
        return self.program(self.queue, *args, **updated_kwargs)

# vim: foldmethod=marker
