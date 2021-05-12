from __future__ import annotations

__copyright__ = """
Copyright (C) 2021 Kaushik Kulkarni
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


import numpy as np
import loopy as lp
import pymbolic.primitives as prim
from loopy.types import NumpyType
from typing import Dict, Optional, Any, Union
from numbers import Number
from pytato.array import DictOfNamedArrays, Array, ShapeType, NamedArray
from pytato.scalar_expr import SubstitutionMapper, ScalarExpression
from pytools import memoize_method
from pytools.tag import TagsType

__doc__ = """
.. currentmodule:: pytato.loopy

.. autoclass:: LoopyFunction

.. autoclass:: LoopyFunctionResult

.. autofunction:: call_loopy
"""


class LoopyFunction(DictOfNamedArrays):
    """
    Call to a :mod:`loopy` program.
    """
    _mapper_method = "map_loopy_function"

    def __init__(self,
            program: "lp.Program",
            bindings: Dict[str, Union[Array, Number]],
            entrypoint: str):
        super().__init__({})

        self.program = program
        self.bindings = bindings
        self.entrypoint = entrypoint

        entry_kernel = program[entrypoint]

        self._named_arrays = {name: LoopyFunctionResult(self, name)
                              for name, lp_arg in entry_kernel.arg_dict.items()
                              if lp_arg.is_output}

    @memoize_method
    def to_pytato(self, expr: ScalarExpression) -> ScalarExpression:
        from pymbolic.mapper.substitutor import make_subst_func
        return SubstitutionMapper(make_subst_func(self.bindings))(expr)

    @property
    def entry_kernel(self) -> lp.LoopKernel:
        return self.program[self.entrypoint]

    def __hash__(self) -> int:
        return hash((self.program, tuple(self.bindings.items()), self.entrypoint))

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True

        if not isinstance(other, LoopyFunction):
            return False

        if ((self.entrypoint == other.entrypoint)
             and (self.bindings == other.bindings)
             and (self.program == other.program)):
            return True
        return False


class LoopyFunctionResult(NamedArray):
    """
    Named array for :class:`LoopyFunction`'s result.
    """
    def __init__(self,
            dict_of_named_arrays: LoopyFunction,
            name: str,
            tags: TagsType = frozenset()) -> None:
        super().__init__(dict_of_named_arrays, name, tags=tags)

    def expr(self) -> Array:
        raise ValueError("Expressions for results of loopy functions aren't defined")

    @property
    def shape(self) -> ShapeType:
        loopy_arg = self.dict_of_named_arrays.entry_kernel.arg_dict[  # type:ignore
                self.name]
        shape: ShapeType = self.dict_of_named_arrays.to_pytato(  # type:ignore
                loopy_arg.shape)
        return shape

    @property
    def dtype(self) -> np.dtype[Any]:
        loopy_arg = self.dict_of_named_arrays.entry_kernel.arg_dict[  # type:ignore
                self.name]
        dtype = loopy_arg.dtype

        if isinstance(dtype, np.dtype):
            return dtype
        elif isinstance(dtype, NumpyType):
            return np.dtype(dtype.numpy_dtype)
        else:
            raise NotImplementedError(f"Unknown dtype type '{dtype}'")


def call_loopy(program: "lp.Program",
               bindings: Dict[str, Union[Array, Number]],
               entrypoint: Optional[str] = None) -> LoopyFunction:
    """
    Operates a general :class:`loopy.Program` on the array inputs as specified
    by *bindings*.

    Restrictions on the structure of ``program[entrypoint]``:

    * array arguments of ``program[entrypoint]`` should either be either
      input-only or output-only.
    * all input-only arguments of ``program[entrypoint]`` must appear in
      *bindings*.
    * all output-only arguments of ``program[entrypoint]`` must appear in
      *bindings*.
    * if *program* has been declared with multiple entrypoints, *entrypoint*
      can not be *None*.

    :arg bindings: mapping from argument names of ``program[entrypoint]`` to
        :class:`pytato.array.Array`.
    :arg results: names of ``program[entrypoint]`` argument names that have to
        be returned from the call.
    """
    if entrypoint is None:
        if len(program.entrypoints) != 1:
            raise ValueError("cannot infer entrypoint")

        entrypoint, = program.entrypoints

    program = program.with_entrypoints(entrypoint)

    # {{{ sanity checks

    if any([arg.is_input and arg.is_output
            for arg in program[entrypoint].args]):
        # Pytato DAG cannot have stateful nodes.
        raise ValueError("Cannot call a kernel with side-effects.")

    for name in bindings:
        if name not in program[entrypoint].arg_dict:
            raise ValueError(f"Kernel '{entrypoint}' got an unexpected input: "
                    f"'{name}'.")
        if program[entrypoint].arg_dict[name].is_output:
            raise ValueError(f"Kernel '{entrypoint}' got an output arg '{name}' "
                    f"as input.")

    for arg in program[entrypoint].args:
        if arg.is_input:
            if arg.name not in bindings:
                raise ValueError(f"Kernel '{entrypoint}' expects an input"
                        f" '{arg.name}'")
            if isinstance(arg, (lp.ArrayArg, lp.ConstantArg)):
                if not isinstance(bindings[arg.name], Array):
                    raise ValueError(f"Argument '{arg.name}' expected to be a "
                            f"pytato.Array, got {type(bindings[arg.name])}.")
            else:
                assert isinstance(arg, lp.ValueArg)
                if not (isinstance(bindings[arg.name], Number)
                        or (isinstance(bindings[arg.name], Array)
                            and bindings[arg.name].shape == ())):  # type: ignore
                    raise ValueError(f"Argument '{arg.name}' expected to be a "
                            " number or a scalar expression, got "
                            f"{type(bindings[arg.name])}.")

    # }}}

    # {{{ infer types of the program

    for name, ary in bindings.items():
        if isinstance(ary, Array):
            program = lp.add_dtypes(program, {name: ary.dtype})
        elif isinstance(ary, prim.Expression):
            program = lp.add_dtypes(program, {name: np.intp})
        else:
            assert isinstance(ary, Number)
            program = lp.add_dtypes(program, {name: type(ary)})

    program = lp.infer_unknown_types(program)

    # }}}

    # {{{ infer shapes of the program

    program = lp.infer_arg_descr(program)

    # }}}

    program = program.with_entrypoints(frozenset())

    return LoopyFunction(program, bindings, entrypoint)


# vim: fdm=marker
