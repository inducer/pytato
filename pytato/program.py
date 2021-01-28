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

Generated Executable Programs
-----------------------------

.. autoclass:: BoundProgram
.. autoclass:: BoundPyOpenCLProgram
"""

from dataclasses import dataclass
import typing
from typing import Any, Mapping, Optional

import loopy

if typing.TYPE_CHECKING:
    # Imports skipped for efficiency.  FIXME: Neither of these work as type
    # stubs are not present. Types are here only as documentation.
    import pyopencl
    # Imports skipped to avoid circular dependencies.
    import pytato.target


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

    program: "loopy.LoopKernel"
    bound_arguments: Mapping[str, Any]
    target: "pytato.target.Target"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@dataclass(init=True, repr=False, eq=False)
class BoundPyOpenCLProgram(BoundProgram):
    """A wrapper around a :mod:`loopy` kernel for execution with :mod:`pyopencl`.

    .. attribute:: queue

        A :mod:`pyopencl` command queue.

    .. automethod:: __call__
    """
    queue: Optional["pyopencl.CommandQueue"]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Convenience function for launching a :mod:`pyopencl` computation."""
        if not self.queue:
            raise ValueError("queue must be specified")

        if set(kwargs.keys()) & set(self.bound_arguments.keys()):
            raise ValueError("Got arguments that were previously bound: "
                    f"{set(kwargs.keys()) & set(self.bound_arguments.keys())}.")

        updated_kwargs = dict(self.bound_arguments)
        updated_kwargs.update(kwargs)
        return self.program(self.queue, *args, **updated_kwargs)

# vim: foldmethod=marker
