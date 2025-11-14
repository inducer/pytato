from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Matt Wala
Copyright (C) 2020-21 Kaushik Kulkarni
Copyright (C) 2020-21 University of Illinois Board of Trustees
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
import dataclasses
import logging
from typing import (
    TYPE_CHECKING,
    cast,
)

from immutabledict import immutabledict
from typing_extensions import Never, Self, override

from pytato.array import (
    AdvancedIndexInContiguousAxes,
    AdvancedIndexInNoncontiguousAxes,
    Array,
    AxisPermutation,
    BasicIndex,
    Concatenate,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexLambda,
    InputArgumentBase,
    NamedArray,
    Placeholder,
    Reshape,
    Roll,
    SizeParam,
    Stack,
    _entries_are_identical,
)
from pytato.equality import EqualityComparer
from pytato.tags import ImplStored
from pytato.transform import (
    ArrayOrNames,
    ArrayOrNamesTc,
    CachedMapper,
    CachedMapperCache,
    CacheInputsWithKey,
    MapperCreatedDuplicateError,
    _is_mapper_created_duplicate,
    _verify_is_array,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from pytato.distributed.nodes import (
        DistributedRecv,
        DistributedSendRefHolder,
    )
    from pytato.function import FunctionDefinition, NamedCallResult


__doc__ = """
.. currentmodule:: pytato.transform.materialize

.. autofunction:: materialize_with_mpms
"""

# {{{ MPMS


@dataclasses.dataclass(frozen=True, eq=True)
class MPMSMaterializerAccumulator:
    """This class serves as the return value of :class:`MPMSMaterializer`. It
    contains the set of materialized predecessors and the rewritten expression
    (i.e. the expression with tags for materialization applied).
    """
    materialized_predecessors: frozenset[Array]
    expr: Array


class MPMSMaterializerCache(
        CachedMapperCache[ArrayOrNames, MPMSMaterializerAccumulator, []]):
    """
    Cache for :class:`MPMSMaterializer`.

    .. automethod:: __init__
    .. automethod:: add
    """
    def __init__(
            self,
            err_on_collision: bool,
            err_on_created_duplicate: bool) -> None:
        """
        Initialize the cache.

        :arg err_on_collision: Raise an exception if two distinct input expression
            instances have the same key.
        :arg err_on_created_duplicate: Raise an exception if mapping produces a new
            array instance that has the same key as the input array.
        """
        super().__init__(err_on_collision=err_on_collision)

        self.err_on_created_duplicate: bool = err_on_created_duplicate

        self._result_key_to_result: dict[
            ArrayOrNames, MPMSMaterializerAccumulator] = {}

        self._equality_comparer: EqualityComparer = EqualityComparer()

    @override
    def add(
            self,
            inputs: CacheInputsWithKey[ArrayOrNames, []],
            result: MPMSMaterializerAccumulator) -> MPMSMaterializerAccumulator:
        """
        Cache a mapping result.

        Returns the cached result (which may not be identical to *result* if a
        result was already cached with the same result key).
        """
        key = inputs.key

        assert key not in self._input_key_to_result, \
            f"Cache entry is already present for key '{key}'."

        try:
            # The first encountered instance of each distinct result (in terms of
            # "==" of result.expr) gets cached, and subsequent mappings with results
            # that are equal to prior cached results are replaced with the original
            # instance
            result = self._result_key_to_result[result.expr]
        except KeyError:
            if (
                    self.err_on_created_duplicate
                    and _is_mapper_created_duplicate(
                        inputs.expr, result.expr,
                        equality_comparer=self._equality_comparer)):
                raise MapperCreatedDuplicateError from None

            self._result_key_to_result[result.expr] = result

        self._input_key_to_result[key] = result
        if self.err_on_collision:
            self._input_key_to_expr[key] = inputs.expr

        return result


def _materialize_if_mpms(expr: Array,
                         successors: list[ArrayOrNames],
                         predecessors: Iterable[MPMSMaterializerAccumulator]
                         ) -> MPMSMaterializerAccumulator:
    """
    Returns an instance of :class:`MPMSMaterializerAccumulator`, that
    materializes *expr* if it has more than 1 successor and more than 1
    materialized predecessor.
    """
    from functools import reduce

    materialized_predecessors: frozenset[Array] = reduce(
        cast(
            "Callable[[frozenset[Array], frozenset[Array]], frozenset[Array]]",
            frozenset.union),
        (pred.materialized_predecessors for pred in predecessors),
        cast("frozenset[Array]", frozenset()))

    nsuccessors = 0
    for successor in successors:
        # Handle indexing with heavy reuse, if the sizes are known ahead of time.
        # This can occur when the elements of a smaller array are used repeatedly to
        # compute the elements of a larger array. (Example: In meshmode's direct
        # connection code, this happens when injecting data from a smaller
        # discretization into a larger one, such as BTAG_ALL -> FACE_RESTR_ALL.)
        #
        # In this case, we would like to bias towards materialization by
        # making one successor seem like n of them, if it is n times bigger.
        if (
                isinstance(successor, IndexBase)
                and isinstance(successor.size, int)
                and isinstance(expr.size, int)):
            nsuccessors += (successor.size // expr.size) if expr.size else 0
        else:
            nsuccessors += 1

    if nsuccessors > 1 and len(materialized_predecessors) > 1:
        new_expr = expr.tagged(ImplStored())
        return MPMSMaterializerAccumulator(frozenset([new_expr]), new_expr)
    else:
        return MPMSMaterializerAccumulator(materialized_predecessors, expr)


class MPMSMaterializer(
        CachedMapper[MPMSMaterializerAccumulator, Never, []]):
    """
    See :func:`materialize_with_mpms` for an explanation.

    .. attribute:: successors

        A mapping from a node in the expression graph (i.e. an
        :class:`~pytato.Array`) to a list of its successors (possibly including
        multiple references to the same successor if it uses the node multiple times).
    """
    def __init__(
            self,
            successors: Mapping[Array, list[ArrayOrNames]],
            _cache: MPMSMaterializerCache | None = None):
        err_on_collision = __debug__
        err_on_created_duplicate = __debug__

        if _cache is None:
            _cache = MPMSMaterializerCache(
                err_on_collision=err_on_collision,
                err_on_created_duplicate=err_on_created_duplicate)

        # Does not support functions, so function_cache is ignored
        super().__init__(err_on_collision=err_on_collision, _cache=_cache)

        self.successors: Mapping[Array, list[ArrayOrNames]] = successors

    @override
    def _cache_add(
            self,
            inputs: CacheInputsWithKey[ArrayOrNames, []],
            result: MPMSMaterializerAccumulator) -> MPMSMaterializerAccumulator:
        try:
            return self._cache.add(inputs, result)
        except MapperCreatedDuplicateError as e:
            raise ValueError(
                f"no-op duplication detected on {type(inputs.expr)} in "
                f"{type(self)}.") from e

    @override
    def clone_for_callee(
            self, function: FunctionDefinition) -> Self:
        """
        Called to clone *self* before starting traversal of a
        :class:`pytato.function.FunctionDefinition`.
        """
        raise AssertionError("Control shouldn't reach this point.")

    def _map_input_base(self, expr: InputArgumentBase
                        ) -> MPMSMaterializerAccumulator:
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_placeholder(self, expr: Placeholder) -> MPMSMaterializerAccumulator:
        return self._map_input_base(expr)

    def map_data_wrapper(self, expr: DataWrapper) -> MPMSMaterializerAccumulator:
        return self._map_input_base(expr)

    def map_size_param(self, expr: SizeParam) -> MPMSMaterializerAccumulator:
        return self._map_input_base(expr)

    def map_named_array(self, expr: NamedArray) -> MPMSMaterializerAccumulator:
        raise NotImplementedError("only LoopyCallResult named array"
                                  " supported for now.")

    def map_index_lambda(self, expr: IndexLambda) -> MPMSMaterializerAccumulator:
        children_rec = {bnd_name: self.rec(bnd)
                        for bnd_name, bnd in sorted(expr.bindings.items())}
        new_children: Mapping[str, Array] = immutabledict({
            bnd_name: bnd.expr
            for bnd_name, bnd in children_rec.items()})
        return _materialize_if_mpms(
            expr.replace_if_different(bindings=new_children),
            self.successors[expr],
            children_rec.values())

    def map_stack(self, expr: Stack) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_arrays = tuple(ary.expr for ary in rec_arrays)
        return _materialize_if_mpms(
            expr.replace_if_different(arrays=new_arrays),
            self.successors[expr],
            rec_arrays)

    def map_concatenate(self, expr: Concatenate) -> MPMSMaterializerAccumulator:
        rec_arrays = [self.rec(ary) for ary in expr.arrays]
        new_arrays = tuple(ary.expr for ary in rec_arrays)
        return _materialize_if_mpms(
            expr.replace_if_different(arrays=new_arrays),
            self.successors[expr],
            rec_arrays)

    def map_roll(self, expr: Roll) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        return _materialize_if_mpms(
            expr.replace_if_different(array=rec_array.expr),
            self.successors[expr],
            (rec_array,))

    def map_axis_permutation(self, expr: AxisPermutation
                             ) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        return _materialize_if_mpms(
            expr.replace_if_different(array=rec_array.expr),
            self.successors[expr],
            (rec_array,))

    def _map_index_base(self, expr: IndexBase) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        rec_indices = {i: self.rec(idx)
                       for i, idx in enumerate(expr.indices)
                       if isinstance(idx, Array)}
        new_indices = tuple(rec_indices[i].expr
                            if i in rec_indices
                            else expr.indices[i]
                            for i in range(
                                len(expr.indices)))
        new_indices = (
            expr.indices
            if _entries_are_identical(new_indices, expr.indices)
            else new_indices)
        return _materialize_if_mpms(
            expr.replace_if_different(array=rec_array.expr, indices=new_indices),
            self.successors[expr],
            (rec_array, *tuple(rec_indices.values())))

    def map_basic_index(self, expr: BasicIndex) -> MPMSMaterializerAccumulator:
        return self._map_index_base(expr)

    def map_contiguous_advanced_index(
            self, expr: AdvancedIndexInContiguousAxes) -> MPMSMaterializerAccumulator:
        return self._map_index_base(expr)

    def map_non_contiguous_advanced_index(
            self, expr: AdvancedIndexInNoncontiguousAxes
            ) -> MPMSMaterializerAccumulator:
        return self._map_index_base(expr)

    def map_reshape(self, expr: Reshape) -> MPMSMaterializerAccumulator:
        rec_array = self.rec(expr.array)
        return _materialize_if_mpms(
            expr.replace_if_different(array=rec_array.expr),
            self.successors[expr],
            (rec_array,))

    def map_einsum(self, expr: Einsum) -> MPMSMaterializerAccumulator:
        rec_args = [self.rec(ary) for ary in expr.args]
        new_args = tuple(ary.expr for ary in rec_args)
        return _materialize_if_mpms(
            expr.replace_if_different(args=new_args),
            self.successors[expr],
            rec_args)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays
                                 ) -> MPMSMaterializerAccumulator:
        raise NotImplementedError

    def map_loopy_call_result(self, expr: NamedArray) -> MPMSMaterializerAccumulator:
        # loopy call result is always materialized
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> MPMSMaterializerAccumulator:
        rec_send_data = self.rec(expr.send.data)
        rec_passthrough = self.rec(expr.passthrough_data)
        return MPMSMaterializerAccumulator(
            rec_passthrough.materialized_predecessors,
            expr.replace_if_different(
                send=expr.send.replace_if_different(data=rec_send_data.expr),
                passthrough_data=rec_passthrough.expr))

    def map_distributed_recv(self, expr: DistributedRecv
                             ) -> MPMSMaterializerAccumulator:
        return MPMSMaterializerAccumulator(frozenset([expr]), expr)

    def map_named_call_result(self, expr: NamedCallResult
                              ) -> MPMSMaterializerAccumulator:
        raise NotImplementedError("MPMSMaterializer does not support functions.")


def materialize_with_mpms(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    r"""
    Materialize nodes in *expr* with MPMS materialization strategy.
    MPMS stands for Multiple-Predecessors, Multiple-Successors.

    .. note::

        - MPMS materialization strategy is a greedy materialization algorithm in
          which any node with more than 1 materialized predecessor and more than
          1 successor is materialized.
        - Materializing here corresponds to tagging a node with
          :class:`~pytato.tags.ImplStored`.
        - Does not attempt to materialize sub-expressions in
          :attr:`pytato.Array.shape`.

    .. warning::

        This is a greedy materialization algorithm and thereby this algorithm
        might be too eager to materialize. Consider the graph below:

        ::

                           I1          I2
                            \         /
                             \       /
                              \     /
                               🡦   🡧
                                 T
                                / \
                               /   \
                              /     \
                             🡧       🡦
                            O1        O2

        where, 'I1', 'I2' correspond to instances of
        :class:`pytato.array.InputArgumentBase`, and, 'O1' and 'O2' are the outputs
        required to be evaluated in the computation graph. MPMS materialization
        algorithm will materialize the intermediate node 'T' as it has 2
        predecessors and 2 successors. However, the total number of memory
        accesses after applying MPMS goes up as shown by the table below.

        ======  ========  =======
        ..        Before    After
        ======  ========  =======
        Reads          4        4
        Writes         2        3
        Total          6        7
        ======  ========  =======

    """
    from pytato.analysis import get_list_of_users, get_num_nodes, get_num_tags_of_type
    materializer = MPMSMaterializer(get_list_of_users(expr))

    if isinstance(expr, Array):
        res = materializer(expr).expr
        assert isinstance(res, Array)
    elif isinstance(expr, DictOfNamedArrays):
        res = expr.replace_if_different(
            data={
                name: _verify_is_array(materializer(ary).expr)
                for name, ary, in expr._data.items()})
        assert isinstance(res, DictOfNamedArrays)
    else:
        raise NotImplementedError("not implemented for {type(expr).__name__}.")

    from pytato import DEBUG_ENABLED
    if DEBUG_ENABLED:
        logger.info("materialize_with_mpms: materialized "
            f"{get_num_tags_of_type(res, ImplStored)} out of "
            f"{get_num_nodes(res)} nodes")

    return res

# }}}

# vim: foldmethod=marker
