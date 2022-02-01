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

from typing import Any, Callable, Dict, TYPE_CHECKING, Tuple, Union
from pytato.array import (AdvancedIndexInContiguousAxes,
                          AdvancedIndexInNoncontiguousAxes, AxisPermutation,
                          BasicIndex, Concatenate, DataWrapper, Einsum,
                          IndexBase, IndexLambda, NamedArray,
                          Reshape, Roll, Stack, AbstractResultWithNamedArrays,
                          Array, DictOfNamedArrays, Placeholder, SizeParam)

if TYPE_CHECKING:
    from pytato.loopy import LoopyCall, LoopyCallResult

__doc__ = """
.. autoclass:: EqualityComparer
"""


ArrayOrNames = Union[Array, AbstractResultWithNamedArrays]


# {{{ EqualityComparer

class EqualityComparer:
    """
    A :class:`pytato.array.Array` visitor to check equality between two
    expression DAGs.

    .. note::

        - Compares two expression graphs ``expr1``, ``expr2`` in :math:`O(N)`
          comparisons, where :math:`N` is the number of nodes in ``expr1``.
        - This visitor was introduced to memoize the sub-expression comparisons
          of the expressions to be compared. Not memoizing the sub-expression
          comparisons results in :math:`O(2^N)` complexity for the comparison
          operation, where :math:`N` is the number of nodes in expressions. See
          `GH-Issue-163 <https://github.com/inducer/pytato/issues/163>` for
          more on this.
    """
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int], bool] = {}

    def rec(self, expr1: ArrayOrNames, expr2: Any) -> bool:
        cache_key = id(expr1), id(expr2)
        try:
            return self._cache[cache_key]
        except KeyError:

            method: Callable[[Union[Array, AbstractResultWithNamedArrays], Any],
                             bool]

            try:
                method = getattr(self, expr1._mapper_method)
            except AttributeError:
                if isinstance(expr1, Array):
                    result = self.handle_unsupported_array(expr1, expr2)
                else:
                    result = self.map_foreign(expr1, expr2)
            else:
                result = method(expr1, expr2)

            self._cache[cache_key] = result
            return result

    def __call__(self, expr1: ArrayOrNames, expr2: Any
                 ) -> bool:
        return self.rec(expr1, expr2)

    def handle_unsupported_array(self, expr1: Array,
                                 expr2: Any) -> bool:
        raise NotImplementedError(type(expr1).__name__)

    def map_foreign(self, expr1: Any, expr2: Any) -> bool:
        raise NotImplementedError(type(expr1).__name__)

    def map_placeholder(self, expr1: Placeholder, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.name == expr2.name
                and expr1.shape == expr2.shape
                and expr1.dtype == expr2.dtype
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_size_param(self, expr1: SizeParam, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.name == expr2.name
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_data_wrapper(self, expr1: DataWrapper, expr2: Any) -> bool:
        return expr1 is expr2

    def map_index_lambda(self, expr1: IndexLambda, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.expr == expr2.expr
                and (frozenset(expr1.bindings.keys())
                     == frozenset(expr2.bindings.keys()))
                and all(self.rec(expr1.bindings[name], expr2.bindings[name])
                        for name in expr1.bindings)
                and len(expr1.shape) == len(expr2.shape)
                and all(self.rec(dim1, dim2)
                        if isinstance(dim1, Array)
                        else dim1 == dim2
                        for dim1, dim2 in zip(expr1.shape, expr2.shape))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes)

    def map_stack(self, expr1: Stack, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.axis == expr2.axis
                and len(expr1.arrays) == len(expr2.arrays)
                and all(self.rec(ary1, ary2)
                        for ary1, ary2 in zip(expr1.arrays, expr2.arrays))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_concatenate(self, expr1: Concatenate, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.axis == expr2.axis
                and len(expr1.arrays) == len(expr2.arrays)
                and all(self.rec(ary1, ary2)
                        for ary1, ary2 in zip(expr1.arrays, expr2.arrays))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_roll(self, expr1: Roll, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.axis == expr2.axis
                and self.rec(expr1.array, expr2.array)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_axis_permutation(self, expr1: AxisPermutation, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.axis_permutation == expr2.axis_permutation
                and self.rec(expr1.array, expr2.array)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def _map_index_base(self, expr1: IndexBase, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and self.rec(expr1.array, expr2.array)
                and len(expr1.indices) == len(expr2.indices)
                and all(self.rec(idx1, idx2)
                        if (isinstance(idx1, Array)
                            and isinstance(idx2, Array))
                        else idx1 == idx2
                        for idx1, idx2 in zip(expr1.indices, expr2.indices))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_basic_index(self, expr1: BasicIndex, expr2: Any) -> bool:
        return self._map_index_base(expr1, expr2)

    def map_contiguous_advanced_index(self,
                                      expr1: AdvancedIndexInContiguousAxes,
                                      expr2: Any
                                      ) -> bool:
        return self._map_index_base(expr1, expr2)

    def map_non_contiguous_advanced_index(self,
                                          expr1: AdvancedIndexInNoncontiguousAxes,
                                          expr2: Any
                                          ) -> bool:
        return self._map_index_base(expr1, expr2)

    def map_reshape(self, expr1: Reshape, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.newshape == expr2.newshape
                and self.rec(expr1.array, expr2.array)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_einsum(self, expr1: Einsum, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.access_descriptors == expr2.access_descriptors
                and all(self.rec(ary1, ary2)
                        for ary1, ary2 in zip(expr1.args,
                                              expr2.args))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_named_array(self, expr1: NamedArray, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and self.rec(expr1._container, expr2._container)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                and expr1.name == expr2.name)

    def map_loopy_call(self, expr1: LoopyCall, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and expr1.translation_unit == expr2.translation_unit
                and expr1.entrypoint == expr2.entrypoint
                and frozenset(expr1.bindings) == frozenset(expr2.bindings)
                and all(self.rec(bnd,
                                 expr2.bindings[name])
                        if isinstance(bnd, Array)
                        else bnd == expr2.bindings[name]
                        for name, bnd in expr1.bindings.items())
                )

    def map_loopy_call_result(self, expr1: LoopyCallResult, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and self.rec(expr1._container, expr2._container)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                and expr1.name == expr2.name)

    def map_dict_of_named_arrays(self, expr1: DictOfNamedArrays, expr2: Any) -> bool:
        return (expr1.__class__ is expr2.__class__
                and frozenset(expr1._data.keys()) == frozenset(expr2._data.keys())
                and all(self.rec(expr1._data[name], expr2._data[name])
                        for name in expr1._data))

# }}}

# vim: fdm=marker
