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

from pytools import memoize_method

from typing import Any, Dict, Tuple, cast
from pytato.transform import Mapper
from pytato.array import (Array, DataWrapper, DictOfNamedArrays, Axis,
                          IndexLambda, ReductionDescriptor)
from pytato.function import FunctionDefinition, Call
from pytato.loopy import LoopyCall
from immutabledict import immutabledict
import dataclasses


__doc__ = """
.. currentmodule:: pytato.stringifier

.. autoclass:: Reprifier
"""


# {{{ Reprifier

class Reprifier(Mapper):
    """
    Stringifies :mod:`pytato`-types to closely resemble CPython's implementation
    of :func:`repr` for its builtin datatypes.
    """

    def __init__(self,
                 truncation_depth: int = 3,
                 truncation_string: str = "(...)") -> None:
        super().__init__()
        self.truncation_depth = truncation_depth
        self.truncation_string = truncation_string

        self._cache: Dict[Tuple[int, int], str] = {}

    def rec(self, expr: Any, depth: int) -> str:
        cache_key = (id(expr), depth)
        try:
            return self._cache[cache_key]
        except KeyError:
            result = super().rec(expr, depth)
            self._cache[cache_key] = result
            return result  # type: ignore[no-any-return]

    def __call__(self, expr: Any, depth: int = 0) -> str:
        return self.rec(expr, depth)

    def map_foreign(self, expr: Any, depth: int) -> str:
        if isinstance(expr, tuple):
            return "(" + ", ".join(self.rec(el, depth) for el in expr) + ")"
        elif isinstance(expr, (dict, immutabledict)):
            return ("{"
                    + ", ".join(f"{key!r}: {self.rec(val, depth)}"
                                for key, val
                                in sorted(expr.items(),
                                          key=lambda k_x_v: cast(str, k_x_v[0])))
                    + "}")
        elif isinstance(expr, (frozenset, set)):
            return "{" + ", ".join(self.rec(el, depth) for el in expr) + "}"
        elif isinstance(expr, np.dtype):
            return f"'{expr.name}'"
        else:
            return repr(expr)

    def _map_generic_array(self, expr: Array, depth: int) -> str:
        if depth > self.truncation_depth:
            return self.truncation_string

        # pylint: disable=not-an-iterable
        fields = tuple(field.name for field in dataclasses.fields(type(expr)))

        fields = tuple(field for field in fields if field != "non_equality_tags")

        if expr.ndim <= 1:
            # prettify: if ndim <=1 'expr.axes' would be trivial,
            # => don't print.
            fields = tuple(field for field in fields if field != "axes")

        if not expr.tags:
            # prettify: if empty 'expr.tags' => don't print.
            fields = tuple(field for field in fields if field != "tags")

        if all(axis == Axis(frozenset()) for axis in expr.axes):
            # prettify: if trivial 'expr.axes' => don't print.
            fields = tuple(field for field in fields if field != "axes")

        if (isinstance(expr, IndexLambda)
                and all(redn_descr == ReductionDescriptor(frozenset())
                        for redn_descr in expr.var_to_reduction_descr.values())):
            # prettify: if trivial 'expr.var_to_reduction_descr' => don't print.
            fields = tuple(field
                           for field in fields
                           if field != "var_to_reduction_descr")

        return (f"{type(expr).__name__}("
                + ", ".join(f"{field}="
                            f"{self.rec(getattr(expr, field), depth+1)}"
                            for field in fields)
                + ")")

    map_placeholder = _map_generic_array
    map_size_param = _map_generic_array
    map_named_array = _map_generic_array
    map_index_lambda = _map_generic_array
    map_matrix_product = _map_generic_array
    map_stack = _map_generic_array
    map_concatenate = _map_generic_array
    map_roll = _map_generic_array
    map_axis_permutation = _map_generic_array
    map_basic_index = _map_generic_array
    map_contiguous_advanced_index = _map_generic_array
    map_non_contiguous_advanced_index = _map_generic_array
    map_reshape = _map_generic_array
    map_einsum = _map_generic_array
    map_distributed_recv = _map_generic_array
    map_distributed_send_ref_holder = _map_generic_array

    def map_data_wrapper(self, expr: DataWrapper, depth: int) -> str:
        if depth > self.truncation_depth:
            return self.truncation_string

        def _get_field_val(field: str) -> str:
            if field == "data":
                return object.__repr__(expr.data)
            else:
                return self.rec(getattr(expr, field), depth+1)

        # pylint: disable=not-an-iterable
        return (f"{type(expr).__name__}("
                + ", ".join(f"{field.name}={_get_field_val(field.name)}"
                        for field in dataclasses.fields(type(expr)))
                + ")")

    @memoize_method
    def map_function_definition(self, expr: FunctionDefinition, depth: int) -> str:
        if depth > self.truncation_depth:
            return self.truncation_string

        def _get_field_val(field: str) -> str:
            if field == "returns":
                return self.rec(getattr(expr, field), depth+1)
            else:
                return repr(getattr(expr, field))

        # pylint: disable=not-an-iterable
        return (f"{type(expr).__name__}("
                + ", ".join(f"{field.name}={_get_field_val(field.name)}"
                        for field in dataclasses.fields(type(expr)))
                + ")")

    def map_call(self, expr: Call, depth: int) -> str:
        if depth > self.truncation_depth:
            return self.truncation_string

        def _get_field_val(field: str) -> str:
            if field == "function":
                return self.map_function_definition(expr.function, depth+1)
            else:
                return self.rec(getattr(expr, field), depth+1)

        return (f"{type(expr).__name__}("
                + ", ".join(f"{field}={_get_field_val(field)}"
                            for field in ["function",
                                          "bindings"])
                + ")")

    def map_loopy_call(self, expr: LoopyCall, depth: int) -> str:
        if depth > self.truncation_depth:
            return self.truncation_string

        def _get_field_val(field: str) -> str:
            if field == "translation_unit":
                return object.__repr__(expr.translation_unit)
            else:
                return self.rec(getattr(expr, field), depth+1)

        return (f"{type(expr).__name__}("
                + ", ".join(f"{field}={_get_field_val(field)}"
                            for field in ["translation_unit",
                                          "bindings",
                                          "entrypoint"])
                + ")")

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays, depth: int) -> str:
        if depth > self.truncation_depth:
            return self.truncation_string

        return (f"{type(expr).__name__}("
                + self.rec(expr._data, depth)
                + ")")

# }}}

# vim: fdm=marker
