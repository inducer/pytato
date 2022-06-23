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

from typing import Any, Mapping, Optional, Callable

from pytato.target import Target, BoundProgram

import loopy


# set in doc/conf.py
if getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    # Avoid import unless building docs to avoid creating a hard
    # dependency on pyopencl, when Loopy can run fine without.
    import pyopencl


class LoopyTarget(Target):
    """An :mod:`loopy` target.

    .. automethod:: get_loopy_target

    .. automethod:: bind_program
    """

    def get_loopy_target(self) -> "loopy.TargetBase":
        """Return the corresponding :mod:`loopy` target."""
        raise NotImplementedError

    def bind_program(self, program: loopy.TranslationUnit,
                     bound_arguments: Mapping[str, Any]) -> BoundProgram:
        """
        Create a :class:`pytato.target.BoundProgram` for this code generation
        target.

        :param program: the :mod:`loopy` program
        :param bound_arguments: a mapping from argument names to outputs
        """
        raise NotImplementedError


class LoopyPyOpenCLTarget(LoopyTarget):
    """A :mod:`pyopencl` code generation target.

    .. attribute:: device

        The :mod:`pyopencl` device used to construct the
        :class:`loopy.PyOpenCLTarget`, or *None*.
    """

    def __init__(self, device: Optional["pyopencl.Device"] = None):
        if device is not None:
            from warnings import warn
            warn("Passing 'device' is deprecated and will stop working in 2023.",
                    DeprecationWarning, stacklevel=2)

    def get_loopy_target(self) -> "loopy.LoopyPyOpenCLTarget":
        import loopy as lp
        return lp.PyOpenCLTarget()

    def bind_program(self, program: loopy.TranslationUnit,
                     bound_arguments: Mapping[str, Any]) -> BoundProgram:
        return BoundPyOpenCLProgram(program=program,
                                    bound_arguments=bound_arguments,
                                    target=self)


@dataclass(init=True, repr=False, eq=False)
class BoundPyOpenCLProgram(BoundProgram):
    """A wrapper around a :mod:`loopy` kernel for execution with :mod:`pyopencl`.

    .. automethod:: __call__
    .. automethod:: copy
    .. automethod:: with_transformed_program
    """

    def copy(self, *,
             program: Optional[loopy.TranslationUnit] = None,
             bound_arguments: Optional[Mapping[str, Any]] = None,
             target: Optional[Target] = None
             ) -> BoundPyOpenCLProgram:
        if program is None:
            program = self.program

        if bound_arguments is None:
            bound_arguments = self.bound_arguments

        if target is None:
            target = self.target

        return BoundPyOpenCLProgram(program=program,
                                    bound_arguments=bound_arguments,
                                    target=target)

    def with_transformed_program(self, f: Callable[[loopy.TranslationUnit],
                                                   loopy.TranslationUnit]
                                 ) -> BoundPyOpenCLProgram:
        """
        Returns a copy of *self* with an *f*-transformed loopy translation unit.
        """
        return self.copy(program=f(self.program))

    def __call__(self, queue: "pyopencl.CommandQueue",  # type: ignore
                 allocator=None, wait_for=None, out_host=None,
                 **kwargs: Any) -> Any:
        """Convenience function for launching a :mod:`pyopencl` computation."""

        if set(kwargs.keys()) & set(self.bound_arguments.keys()):
            raise ValueError("Got arguments that were previously bound: "
                    f"{set(kwargs.keys()) & set(self.bound_arguments.keys())}.")

        updated_kwargs = dict(self.bound_arguments)
        updated_kwargs.update(kwargs)

        # final DAG might be independent of certain placeholders, for ex.
        # '0 * x' results in a final loopy t-unit that is independent of the
        # array 'x', do not pass such inputs
        updated_kwargs = {kw: arg
                          for kw, arg in updated_kwargs.items()
                          if kw in self.program.default_entrypoint.arg_dict}

        return self.program(queue,
                            allocator=allocator, wait_for=wait_for,
                            out_host=out_host,
                            **updated_kwargs)

    @property
    def kernel(self) -> "loopy.LoopKernel":
        if isinstance(self.program, loopy.LoopKernel):
            return self.program
        else:
            return self.program.default_entrypoint


# vim: foldmethod=marker
