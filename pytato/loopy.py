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
from typing import Dict, Optional, Any, Union
from numbers import Number
from pytato.array import DictOfNamedArrays, Array, ShapeType, NamedArray
from pytato.scalar_expr import SubstitutionMapper, ScalarExpression
from pytools import memoize_method
from pytools.tag import TagsType

__doc__ = """
.. currentmodule:: pytato.loopy

.. autoclass:: LoopyCall

.. autoclass:: LoopyCallResult

.. autofunction:: call_loopy
"""


class LoopyCall(DictOfNamedArrays):
    """
    An array expression node representing a call to an entrypoint in a
    :mod:`loopy` translation unit.
    """
    _mapper_method = "map_loopy_function"

    def __init__(self,
            translation_unit: "lp.TranslationUnit",
            bindings: Dict[str, Union[Array, Number]],
            entrypoint: str):
        super().__init__({})

        self.translation_unit = translation_unit
        self.bindings = bindings
        self.entrypoint = entrypoint

        entry_kernel = translation_unit[entrypoint]

        self._named_arrays = {name: LoopyCallResult(self, name)
                              for name, lp_arg in entry_kernel.arg_dict.items()
                              if lp_arg.is_output}

    @memoize_method
    def _to_pytato(self, expr: ScalarExpression) -> ScalarExpression:
        from pymbolic.mapper.substitutor import make_subst_func
        return SubstitutionMapper(make_subst_func(self.bindings))(expr)

    @property
    def _entry_kernel(self) -> lp.LoopKernel:
        return self.translation_unit[self.entrypoint]

    def __hash__(self) -> int:
        return hash((self.translation_unit, tuple(self.bindings.items()),
                     self.entrypoint))

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True

        if not isinstance(other, LoopyCall):
            return False

        if ((self.entrypoint == other.entrypoint)
             and (self.bindings == other.bindings)
             and (self.translation_unit == other.translation_unit)):
            return True
        return False


class LoopyCallResult(NamedArray):
    """
    Named array for :class:`LoopyCall`'s result.
    Inherits from :class:`~pytato.array.NamedArray`.
    """
    def __init__(self,
            dict_of_named_arrays: LoopyCall,
            name: str,
            tags: TagsType = frozenset()) -> None:
        super().__init__(dict_of_named_arrays, name, tags=tags)

    def expr(self) -> Array:
        raise ValueError("Expressions for results of loopy functions aren't defined")

    @property
    def shape(self) -> ShapeType:
        loopy_arg = self.dict_of_named_arrays._entry_kernel.arg_dict[  # type:ignore
                self.name]
        shape: ShapeType = self.dict_of_named_arrays._to_pytato(  # type:ignore
                loopy_arg.shape)
        return shape

    @property
    def dtype(self) -> Any:
        loopy_arg = self.dict_of_named_arrays._entry_kernel.arg_dict[  # type:ignore
                self.name]
        return loopy_arg.dtype.numpy_dtype


def call_loopy(translation_unit: "lp.TranslationUnit",
               bindings: Dict[str, Union[Array, Number]],
               entrypoint: Optional[str] = None) -> LoopyCall:
    """
    Operates a general :class:`loopy.TranslationUnit` on the array inputs as
    specified by *bindings*.

    Restrictions on the structure of ``translation_unit[entrypoint]``:

    * array arguments of ``translation_unit[entrypoint]`` must either be either
      input-only or output-only.
    * all input-only arguments of ``translation_unit[entrypoint]`` must appear in
      *bindings*.
    * all output-only arguments of ``translation_unit[entrypoint]`` must appear
      in *bindings*.
    * if *translation_unit* has been declared with multiple entrypoints,
      *entrypoint* can not be *None*.

    :arg bindings: mapping from argument names of ``translation_unit[entrypoint]``
    to :class:`pytato.array.Array`.
    :arg results: names of ``translation_unit[entrypoint]`` argument names that
    have to be returned from the call.
    """
    if entrypoint is None:
        if len(translation_unit.entrypoints) != 1:
            raise ValueError("cannot infer entrypoint")

        entrypoint, = translation_unit.entrypoints

    translation_unit = translation_unit.with_entrypoints(entrypoint)

    # {{{ sanity checks

    if any([arg.is_input and arg.is_output
            for arg in translation_unit[entrypoint].args]):
        # Pytato DAG cannot have stateful nodes.
        raise ValueError("Cannot call a kernel with side-effects.")

    for name in bindings:
        if name not in translation_unit[entrypoint].arg_dict:
            raise ValueError(f"Kernel '{entrypoint}' got an unexpected input: "
                    f"'{name}'.")
        if translation_unit[entrypoint].arg_dict[name].is_output:
            raise ValueError(f"Kernel '{entrypoint}' got an output arg '{name}' "
                    f"as input.")

    for arg in translation_unit[entrypoint].args:
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

    # {{{ infer types of the translation_unit

    for name, ary in bindings.items():
        if isinstance(ary, Array):
            translation_unit = lp.add_dtypes(translation_unit, {name: ary.dtype})
        elif isinstance(ary, prim.Expression):
            translation_unit = lp.add_dtypes(translation_unit, {name: np.intp})
        else:
            assert isinstance(ary, Number)
            translation_unit = lp.add_dtypes(translation_unit, {name: type(ary)})

    translation_unit = lp.infer_unknown_types(translation_unit)

    # }}}

    # {{{ infer shapes of the translation_unit

    translation_unit = lp.infer_arg_descr(translation_unit)

    # }}}

    translation_unit = translation_unit.with_entrypoints(frozenset())

    return LoopyCall(translation_unit, bindings, entrypoint)


# vim: fdm=marker
