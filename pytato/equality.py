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

from typing import TYPE_CHECKING

from pytato.array import (
    AbstractResultWithNamedArrays,
    AdvancedIndexInContiguousAxes,
    AdvancedIndexInNoncontiguousAxes,
    Array,
    AxisPermutation,
    BasicIndex,
    Concatenate,
    CSRMatmul,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexLambda,
    NamedArray,
    Placeholder,
    Reshape,
    Roll,
    SizeParam,
    Stack,
)
from pytato.function import FunctionDefinition


if TYPE_CHECKING:
    from collections.abc import Callable

    from pytato.distributed.nodes import DistributedRecv, DistributedSendRefHolder
    from pytato.function import Call, NamedCallResult
    from pytato.loopy import LoopyCall, LoopyCallResult

__doc__ = """
.. autoclass:: EqualityComparer
"""


ArrayOrNames = Array | AbstractResultWithNamedArrays


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
        # Uses the same cache for both arrays and functions
        self._cache: dict[tuple[int, int], bool] = {}

    def rec(self, expr1: ArrayOrNames | FunctionDefinition, expr2: object) -> bool:
        # These cases are simple enough that they don't need to be cached
        if expr1 is expr2:
            return True
        if expr1.__class__ is not expr2.__class__:
            return False

        cache_key = id(expr1), id(expr2)
        try:
            return self._cache[cache_key]
        except KeyError:
            if isinstance(expr1, ArrayOrNames):
                assert isinstance(expr2, ArrayOrNames)
                method: Callable[[ArrayOrNames, ArrayOrNames], bool]
                try:
                    method = getattr(self, expr1._mapper_method)
                except AttributeError:
                    if isinstance(expr1, Array):
                        result = self.handle_unsupported_array(expr1, expr2)
                    else:
                        result = self.map_foreign(expr1, expr2)
                else:
                    result = method(expr1, expr2)
            elif isinstance(expr1, FunctionDefinition):
                assert isinstance(expr2, FunctionDefinition)
                result = self.map_function_definition(expr1, expr2)
            else:
                result = self.map_foreign(expr1, expr2)

            self._cache[cache_key] = result
            return result

    def __call__(self, expr1: ArrayOrNames | FunctionDefinition, expr2: object) -> bool:
        return self.rec(expr1, expr2)

    def handle_unsupported_array(self, expr1: Array,
                                 expr2: object) -> bool:
        raise NotImplementedError(type(expr1).__name__)

    def map_foreign(self, expr1: object, expr2: object) -> bool:
        raise NotImplementedError(type(expr1).__name__)

    def map_placeholder(self, expr1: Placeholder, expr2: Placeholder) -> bool:
        return (expr1.name == expr2.name
                and expr1.shape == expr2.shape
                and expr1.dtype == expr2.dtype
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_size_param(self, expr1: SizeParam, expr2: SizeParam) -> bool:
        return (expr1.name == expr2.name
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_data_wrapper(self, expr1: DataWrapper, expr2: DataWrapper) -> bool:
        return expr1 is expr2

    def map_index_lambda(self, expr1: IndexLambda, expr2: IndexLambda) -> bool:
        return (expr1.expr == expr2.expr
                and (frozenset(expr1.bindings.keys())
                     == frozenset(expr2.bindings.keys()))
                and all(self.rec(expr1.bindings[name], expr2.bindings[name])
                        for name in expr1.bindings)
                and len(expr1.shape) == len(expr2.shape)
                and all(self.rec(dim1, dim2)
                        if isinstance(dim1, Array)
                        else dim1 == dim2
                        for dim1, dim2 in zip(expr1.shape, expr2.shape, strict=True))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                and expr1.var_to_reduction_descr == expr2.var_to_reduction_descr
                )

    def map_stack(self, expr1: Stack, expr2: Stack) -> bool:
        return (expr1.axis == expr2.axis
                and len(expr1.arrays) == len(expr2.arrays)
                and all(self.rec(ary1, ary2)
                        for ary1, ary2 in zip(expr1.arrays, expr2.arrays, strict=True))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_concatenate(self, expr1: Concatenate, expr2: Concatenate) -> bool:
        return (expr1.axis == expr2.axis
                and len(expr1.arrays) == len(expr2.arrays)
                and all(self.rec(ary1, ary2)
                        for ary1, ary2 in zip(expr1.arrays, expr2.arrays, strict=True))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_roll(self, expr1: Roll, expr2: Roll) -> bool:
        return (expr1.axis == expr2.axis
                and expr1.shift == expr2.shift
                and self.rec(expr1.array, expr2.array)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_axis_permutation(
            self, expr1: AxisPermutation, expr2: AxisPermutation) -> bool:
        return (expr1.axis_permutation == expr2.axis_permutation
                and self.rec(expr1.array, expr2.array)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def _map_index_base(self, expr1: IndexBase, expr2: IndexBase) -> bool:
        return (self.rec(expr1.array, expr2.array)
                and len(expr1.indices) == len(expr2.indices)
                and all(self.rec(idx1, idx2)
                        if (isinstance(idx1, Array)
                            and isinstance(idx2, Array))
                        else idx1 == idx2
                        for idx1, idx2 in zip(
                            expr1.indices, expr2.indices, strict=True))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_basic_index(self, expr1: BasicIndex, expr2: BasicIndex) -> bool:
        return self._map_index_base(expr1, expr2)

    def map_contiguous_advanced_index(self,
                                      expr1: AdvancedIndexInContiguousAxes,
                                      expr2: AdvancedIndexInContiguousAxes
                                      ) -> bool:
        return self._map_index_base(expr1, expr2)

    def map_non_contiguous_advanced_index(self,
                                          expr1: AdvancedIndexInNoncontiguousAxes,
                                          expr2: AdvancedIndexInNoncontiguousAxes
                                          ) -> bool:
        return self._map_index_base(expr1, expr2)

    def map_reshape(self, expr1: Reshape, expr2: Reshape) -> bool:
        return (expr1.newshape == expr2.newshape
                and self.rec(expr1.array, expr2.array)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                )

    def map_einsum(self, expr1: Einsum, expr2: Einsum) -> bool:
        return (expr1.access_descriptors == expr2.access_descriptors
                and all(self.rec(ary1, ary2)
                        for ary1, ary2 in zip(expr1.args,
                                              expr2.args, strict=True))
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                and expr1.redn_axis_to_redn_descr == expr2.redn_axis_to_redn_descr
                )

    def map_csr_matmul(self, expr1: CSRMatmul, expr2: CSRMatmul) -> bool:
        return (self.rec(expr1.matrix.elem_values, expr2.matrix.elem_values)
                and self.rec(
                    expr1.matrix.elem_col_indices, expr2.matrix.elem_col_indices)
                and self.rec(expr1.matrix.row_starts, expr2.matrix.row_starts)
                and self.rec(expr1.array, expr2.array)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes)

    def map_named_array(self, expr1: NamedArray, expr2: NamedArray) -> bool:
        return (self.rec(expr1._container, expr2._container)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                and expr1.name == expr2.name)

    def map_loopy_call(self, expr1: LoopyCall, expr2: LoopyCall) -> bool:
        return (expr1.translation_unit == expr2.translation_unit
                and expr1.entrypoint == expr2.entrypoint
                and frozenset(expr1.bindings) == frozenset(expr2.bindings)
                and all(self.rec(bnd,
                                 expr2.bindings[name])
                        if isinstance(bnd, Array)
                        else bnd == expr2.bindings[name]
                        for name, bnd in expr1.bindings.items())
                and expr1.tags == expr2.tags
                )

    def map_loopy_call_result(
            self, expr1: LoopyCallResult, expr2: LoopyCallResult) -> bool:
        return (self.rec(expr1._container, expr2._container)
                and expr1.tags == expr2.tags
                and expr1.axes == expr2.axes
                and expr1.name == expr2.name)

    def map_dict_of_named_arrays(
            self, expr1: DictOfNamedArrays, expr2: DictOfNamedArrays) -> bool:
        return (frozenset(expr1._data.keys()) == frozenset(expr2._data.keys())
                and all(self.rec(expr1._data[name], expr2._data[name])
                        for name in expr1._data)
                and expr1.tags == expr2.tags
                )

    def map_distributed_send_ref_holder(
            self, expr1: DistributedSendRefHolder, expr2: DistributedSendRefHolder
            ) -> bool:
        return (self.rec(expr1.send.data, expr2.send.data)
                and self.rec(expr1.passthrough_data, expr2.passthrough_data)
                and expr1.send.dest_rank == expr2.send.dest_rank
                and expr1.send.comm_tag == expr2.send.comm_tag
                and expr1.send.tags == expr2.send.tags
                and expr1.tags == expr2.tags
                )

    def map_distributed_recv(
            self, expr1: DistributedRecv, expr2: DistributedRecv) -> bool:
        return (expr1.src_rank == expr2.src_rank
                and expr1.comm_tag == expr2.comm_tag
                and expr1.shape == expr2.shape
                and expr1.dtype == expr2.dtype
                and expr1.tags == expr2.tags
                )

    def map_function_definition(
            self, expr1: FunctionDefinition, expr2: FunctionDefinition) -> bool:
        return (expr1.parameters == expr2.parameters
                and expr1.return_type == expr2.return_type
                and (set(expr1.returns.keys()) == set(expr2.returns.keys()))
                and all(self.rec(expr1.returns[k], expr2.returns[k])
                        for k in expr1.returns)
                and expr1.tags == expr2.tags
                )

    def map_call(self, expr1: Call, expr2: Call) -> bool:
        return (self.rec(expr1.function, expr2.function)
                and frozenset(expr1.bindings) == frozenset(expr2.bindings)
                and all(self.rec(bnd,
                                 expr2.bindings[name])
                        for name, bnd in expr1.bindings.items())
                and expr1.tags == expr2.tags
                )

    def map_named_call_result(
            self, expr1: NamedCallResult, expr2: NamedCallResult) -> bool:
        return (expr1.name == expr2.name
                and self.rec(expr1._container, expr2._container))

# }}}

# vim: fdm=marker
