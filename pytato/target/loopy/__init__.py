from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Matt Wala
Copyright (C) 2023 University of Illinois Board of Trustees
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

__doc__ = """
.. currentmodule:: pytato

.. autofunction:: generate_loopy

.. currentmodule:: pytato.target.loopy

.. autoclass:: LoopyTarget
.. autoclass:: LoopyPyOpenCLTarget
.. autoclass:: BoundPyOpenCLProgram
.. autoclass:: BoundPyOpenCLExecutable
.. autoclass:: ImplSubstitution

Stuff that's only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyopencl._cl

.. autoclass:: MemoryObject

    See :class:`pyopencl.MemoryObject`.
"""

import sys
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from immutabledict import immutabledict

import loopy

from pytato.tags import ImplementationStrategy
from pytato.target import BoundProgram, Target


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pyopencl


class ImplSubstitution(ImplementationStrategy):
    """
    An :class:`~pytato.tags.ImplementationStrategy` that lowers the array
    expression as a :class:`loopy.SubstitutionRule` invocation.
    """


# set in doc/conf.py
if getattr(sys, "_BUILDING_SPHINX_DOCS", False) or TYPE_CHECKING:
    # Avoid import unless building docs to avoid creating a hard
    # dependency on pyopencl, when Loopy can run fine without.
    pass


class LoopyTarget(Target):
    """An :mod:`loopy` target.

    .. automethod:: get_loopy_target

    .. automethod:: bind_program
    """

    def get_loopy_target(self) -> loopy.TargetBase:
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

    def __init__(self, device: pyopencl.Device | None = None):
        if device is not None:
            from warnings import warn
            warn("Passing 'device' is deprecated and will stop working in 2023.",
                    DeprecationWarning, stacklevel=2)

    def get_loopy_target(self) -> loopy.PyOpenCLTarget:
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
    .. automethod:: bind_to_context
    """
    program: loopy.TranslationUnit
    _processed_bound_args_cache: dict[pyopencl.Context,
                                      Mapping[str, Any]] = \
                                        field(default_factory=dict)

    def copy(self, *,
             program: loopy.TranslationUnit | None = None,
             bound_arguments: Mapping[str, Any] | None = None,
             target: Target | None = None
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

    def _get_processed_bound_arguments(
                self,
                queue: pyopencl.CommandQueue,
                allocator: Callable[[int], pyopencl.MemoryObject] | None,
            ) -> Mapping[str, Any]:
        import pyopencl.array as cla

        cache_key = queue.context
        try:
            return self._processed_bound_args_cache[cache_key]
        except KeyError:
            proc_bnd_args: dict[str, Any] = {}
            for name, bnd_arg in self.bound_arguments.items():
                if np.isscalar(bnd_arg):
                    proc_bnd_args[name] = bnd_arg
                elif isinstance(bnd_arg, np.ndarray):
                    if self.program.default_entrypoint.options.no_numpy:
                        raise TypeError(f"Got numpy array for the DataWrapper {name}"
                                        ", in no_numpy=True mode. Expects a"
                                        " pyopencl.array.Array.") from None
                    proc_bnd_args[name] = cla.to_device(queue, bnd_arg, allocator)
                elif isinstance(bnd_arg, cla.Array):
                    proc_bnd_args[name] = bnd_arg
                else:
                    raise TypeError("Data in a bound argument can be one of"
                                    " numpy array, pyopencl array or scalar."
                                    f" Got {type(bnd_arg).__name__} for '{name}'."
                                ) from None

            result: Mapping[str, Any] = immutabledict(proc_bnd_args)
            assert set(result.keys()) == set(self.bound_arguments.keys())
            self._processed_bound_args_cache[cache_key] = result
            return result

    @cached_property
    def all_bound_args_on_host(self) -> bool:
        """
        Returns *True* only if all bound arguments were on the host.
        """
        return all(isinstance(arg, np.ndarray) or np.isscalar(arg)
                   for arg in self.bound_arguments.values())

    def __call__(self, queue: pyopencl.CommandQueue,  # type: ignore[no-untyped-def,no-any-unimported]
                 allocator=None, wait_for=None, out_host: bool | None = None,
                 **kwargs: Any) -> Any:
        """Convenience function for launching a :mod:`pyopencl` computation."""

        if __debug__:  # noqa: SIM102
            if set(kwargs.keys()) & set(self.bound_arguments.keys()):
                raise ValueError("Got arguments that were previously bound: "
                        f"{set(kwargs.keys()) & set(self.bound_arguments.keys())}.")

        updated_kwargs = dict(self._get_processed_bound_arguments(queue, allocator))
        updated_kwargs.update(kwargs)

        # final DAG might be independent of certain placeholders, for ex.
        # '0 * x' results in a final loopy t-unit that is independent of the
        # array 'x', do not pass such inputs
        updated_kwargs = {kw: arg
                          for kw, arg in updated_kwargs.items()
                          if kw in self.program.default_entrypoint.arg_dict}

        if out_host is None and not self.program.default_entrypoint.options.no_numpy:
            # follow loopy's device->host transfer semantics here i.e. if all
            # the arguments to the kernel are on the host then transfer the
            # result to the host.
            out_host = self.all_bound_args_on_host and all(
                isinstance(arg, np.ndarray) or np.isscalar(arg)
                for arg in kwargs.values())

        return self.program(queue,
                            allocator=allocator, wait_for=wait_for,
                            out_host=out_host,
                            **updated_kwargs)

    @property
    def kernel(self) -> loopy.LoopKernel:
        if isinstance(self.program, loopy.LoopKernel):
            return self.program
        else:
            return self.program.default_entrypoint

    def bind_to_context(self, context: pyopencl.Context,
                        allocator: Callable[[int], pyopencl.MemoryObject] | None = None
                        ) -> BoundPyOpenCLExecutable:
        if not self.program.default_entrypoint.options.no_numpy:
            raise ValueError("numpy compatibility for arguments is not supported "
                             "for bound-to-context bound programs")

        from pyopencl import CommandQueue
        with CommandQueue(context) as queue:
            args = self._get_processed_bound_arguments(queue, allocator=allocator)
        return BoundPyOpenCLExecutable(
                program=self.program.executor(context),
                bound_arguments=args,
                target=self.target,
                cl_context=context)


@dataclass(init=True, repr=False, eq=False)
class BoundPyOpenCLExecutable(BoundProgram):
    """A wrapper around a :mod:`loopy` kernel for execution with
    :mod:`pyopencl`.  In contrast to :class:`BoundPyOpenCLProgram`, this object
    is specific to a given :class:`pyopencl.Context`, allowing it to store a
    :class:`loopy.ExecutorBase` instead of a :class:`loopy.TranslationUnit`, as
    well as retrieving pre-transferred bound arguments without a cache lookup,
    permitting more efficient invocation.

    Create these objects using :meth:`BoundPyOpenCLProgram.bind_to_context`.

    .. automethod:: __call__
    .. automethod:: with_transformed_translation_unit
    """
    program: loopy.ExecutorBase
    cl_context: pyopencl.Context

    def with_transformed_translation_unit(
            self, f: Callable[[loopy.TranslationUnit],
                              loopy.TranslationUnit]
            ) -> BoundPyOpenCLExecutable:
        """
        Returns a copy of *self* with an *f*-transformed loopy translation unit.
        """
        return BoundPyOpenCLExecutable(
                program=f(self.program.t_unit).executor(self.cl_context),
                bound_arguments=self.bound_arguments,
                target=self.target,
                cl_context=self.cl_context)

    @cached_property
    def all_bound_args_on_host(self) -> bool:
        """
        Returns *True* only if all bound arguments were on the host.
        """
        return all(np.isscalar(arg) for arg in self.bound_arguments.values())

    def __call__(self, queue: pyopencl.CommandQueue,  # type: ignore[no-untyped-def,no-any-unimported]
                 allocator=None, wait_for=None,
                 **kwargs: Any) -> Any:
        """Convenience function for launching a :mod:`pyopencl` computation."""

        if __debug__:  # noqa: SIM102
            if set(kwargs.keys()) & set(self.bound_arguments.keys()):
                raise ValueError("Got arguments that were previously bound: "
                        f"{set(kwargs.keys()) & set(self.bound_arguments.keys())}.")

        updated_kwargs = dict(self.bound_arguments)
        updated_kwargs.update(kwargs)

        # final DAG might be independent of certain placeholders, for ex.
        # '0 * x' results in a final loopy t-unit that is independent of the
        # array 'x', do not pass such inputs
        arg_dict = self.program.t_unit.default_entrypoint.arg_dict
        updated_kwargs = {kw: arg
                          for kw, arg in updated_kwargs.items()
                          if kw in arg_dict}

        return self.program(queue,
                            allocator=allocator, wait_for=wait_for,
                            **updated_kwargs)

    @property
    def kernel(self) -> loopy.LoopKernel:
        return self.program.t_unit.default_entrypoint


# vim: foldmethod=marker
