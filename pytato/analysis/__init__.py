from __future__ import annotations


__copyright__ = """
Copyright (C) 2021 Kaushik Kulkarni
Copyright (C) 2022 University of Illinois Board of Trustees
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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, overload

from orderedsets import FrozenOrderedSet
from typing_extensions import Never, Self, override

from loopy.tools import LoopyKeyBuilder
from pymbolic.mapper.optimize import optimize_mapper
from pytools import product

from pytato.array import (
    Array,
    ArrayOrScalar,
    Concatenate,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexLambda,
    IndexRemappingBase,
    InputArgumentBase,
    NamedArray,
    Placeholder,
    ShapeType,
    Stack,
)
from pytato.distributed.nodes import DistributedRecv, DistributedSendRefHolder
from pytato.function import Call, FunctionDefinition, NamedCallResult
from pytato.scalar_expr import (
    FlopCounter as ScalarFlopCounter,
)
from pytato.tags import ImplStored
from pytato.transform import (
    ArrayOrNames,
    ArrayOrNamesTc,
    CachedWalkMapper,
    CombineMapper,
    Mapper,
    VisitKeyT,
    map_and_copy,
)
from pytato.transform.lower_to_index_lambda import to_index_lambda
from pytato.utils import has_taggable_materialization, is_materialized


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pytools.tag

    from pytato.loopy import LoopyCall

__doc__ = """
.. currentmodule:: pytato.analysis

.. autofunction:: get_nusers
.. autofunction:: get_list_of_users

.. autofunction:: is_einsum_similar_to_subscript

.. autofunction:: get_num_nodes

.. autofunction:: get_node_type_counts

.. autofunction:: get_node_multiplicities

.. autofunction:: get_num_call_sites

.. autoclass:: DirectPredecessorsGetter
.. autoclass:: ListOfDirectPredecessorsGetter

.. autoclass:: TagCountMapper
.. autofunction:: get_num_tags_of_type

.. autoclass:: UndefinedOpFlopCountError
.. autofunction:: get_default_op_name_to_num_flops
.. autofunction:: get_num_flops
.. autofunction:: get_materialized_node_flop_counts
.. autoclass:: UnmaterializedNodeFlopCounts
.. autofunction:: get_unmaterialized_node_flop_counts
"""


# {{{ ListOfUsersCollector

class ListOfUsersCollector(Mapper[None, Never, []]):
    """
    A :class:`pytato.transform.CachedWalkMapper` that records, for each array
    expression, the nodes that directly depend on it.

    .. note::

        - We do not consider the :class:`pytato.DistributedSendRefHolder`
          a user of :attr:`pytato.DistributedSendRefHolder.send`. This is
          because in a data flow sense, the send-ref holder does not use the
          send's data.
    """
    def __init__(self) -> None:
        super().__init__()
        self._visited_ids: set[int] = set()
        self.array_to_users: dict[Array, list[ArrayOrNames]] = defaultdict(list)

    @override
    def rec(self, expr: ArrayOrNames) -> None:
        # See CachedWalkMapper.rec on why we chose id(x) as the cache key.

        if id(expr) in self._visited_ids:
            return

        super().rec(expr)
        self._visited_ids.add(id(expr))

    def map_index_lambda(self, expr: IndexLambda) -> None:
        for ary in expr.bindings.values():
            self.array_to_users[ary].append(expr)
            self.rec(ary)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.array_to_users[dim].append(expr)
                self.rec(dim)

    def map_stack(self, expr: Stack) -> None:
        for ary in expr.arrays:
            self.array_to_users[ary].append(expr)
            self.rec(ary)

    def map_concatenate(self, expr: Concatenate) -> None:
        for ary in expr.arrays:
            self.array_to_users[ary].append(expr)
            self.rec(ary)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        for ary in expr.bindings.values():
            if isinstance(ary, Array):
                self.array_to_users[ary].append(expr)
                self.rec(ary)

    def map_einsum(self, expr: Einsum) -> None:
        for ary in expr.args:
            self.array_to_users[ary].append(expr)
            self.rec(ary)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.array_to_users[dim].append(expr)
                self.rec(dim)

    def map_named_array(self, expr: NamedArray) -> None:
        self.rec(expr._container)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        for child in expr._data.values():
            self.rec(child)

    def _map_index_base(self, expr: IndexBase) -> None:
        self.array_to_users[expr.array].append(expr)
        self.rec(expr.array)

        for idx in expr.indices:
            if isinstance(idx, Array):
                self.array_to_users[idx].append(expr)
                self.rec(idx)

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def _map_index_remapping_base(self, expr: IndexRemappingBase) -> None:
        self.array_to_users[expr.array].append(expr)
        self.rec(expr.array)

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_input_base(self, expr: InputArgumentBase) -> None:
        for dim in expr.shape:
            if isinstance(dim, Array):
                self.array_to_users[dim].append(expr)
                self.rec(dim)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_distributed_send_ref_holder(self, expr: DistributedSendRefHolder
                                        ) -> None:
        # Note: We do not consider 'expr.send.data' as a predecessor of *expr*,
        # as there is no dataflow from *expr.send.data* to *expr*
        self.array_to_users[expr.passthrough_data].append(expr)
        self.rec(expr.passthrough_data)
        self.rec(expr.send.data)

    def map_distributed_recv(self, expr: DistributedRecv) -> None:
        for dim in expr.shape:
            if isinstance(dim, Array):
                self.array_to_users[dim].append(expr)
                self.rec(dim)

    def map_call(self, expr: Call) -> None:
        for ary in expr.bindings.values():
            if isinstance(ary, Array):
                self.array_to_users[ary].append(expr)
                self.rec(ary)

    def map_named_call_result(self, expr: NamedCallResult) -> None:
        self.rec(expr._container)

# }}}


def get_nusers(outputs: ArrayOrNames) -> Mapping[Array, int]:
    """
    For the DAG *outputs*, returns the mapping from each array node to the number of
    nodes using its value within the DAG given by *outputs*.
    """
    list_of_users_collector = ListOfUsersCollector()
    list_of_users_collector(outputs)
    return defaultdict(int, {
        ary: len(users)
        for ary, users in list_of_users_collector.array_to_users.items()})


def get_list_of_users(outputs: ArrayOrNames) -> Mapping[Array, list[ArrayOrNames]]:
    """
    For the DAG *outputs*, returns the mapping from each array node to the list of
    nodes using its value within the DAG given by *outputs*.
    """
    list_of_users_collector = ListOfUsersCollector()
    list_of_users_collector(outputs)
    return list_of_users_collector.array_to_users


# {{{ is_einsum_similar_to_subscript

def _get_indices_from_input_subscript(subscript: str,
                                      is_output: bool,
                                      ) -> tuple[str, ...]:
    from pytato.array import EINSUM_FIRST_INDEX

    acc = subscript.strip()
    normalized_indices = []

    while acc:
        # {{{ consume indices of in_subscript.

        match = EINSUM_FIRST_INDEX.match(acc)
        if match:
            if "alpha" in match.groupdict():
                normalized_indices.append(match.groupdict()["alpha"])
            else:
                assert "ellipsis" in match.groupdict()
                raise NotImplementedError("Checking against einsum specs"
                                            " with ellipses: not yet supported.")
            assert match.span()[0] == 0
            acc = acc[match.span()[-1]:]
        else:
            raise ValueError(f"Cannot parse '{acc}' in provided einsum"
                             f" '{subscript}'.")

        # }}}

    if is_output and len(normalized_indices) != len(set(normalized_indices)):
        repeated_idx = next(idx
                            for idx in normalized_indices
                            if normalized_indices.count(idx) > 1)
        raise ValueError(f"Output subscript '{subscript}' contains "
                         f"'{repeated_idx}' multiple times.")

    return tuple(normalized_indices)


def is_einsum_similar_to_subscript(expr: Einsum, subscripts: str) -> bool:
    """
    Returns *True* if and only if an einsum with the subscript descriptor
    string *subscripts* operated on *expr*'s :attr:`pytato.Einsum.args`
    would compute the same result as *expr*.
    """

    from pytato.array import (
        EinsumAxisDescriptor,
        EinsumElementwiseAxis,
        EinsumReductionAxis,
    )

    if not isinstance(expr, Einsum):
        raise TypeError(f"{expr} expected to be Einsum, got {type(expr)}.")

    if "->" not in subscripts:
        raise NotImplementedError("Comparing against implicit mode einsums:"
                                  " not supported.")

    in_spec, out_spec = subscripts.split("->")

    # build up a mapping from index names to axis descriptors
    index_to_descrs: dict[str, EinsumAxisDescriptor] = {}

    for idim, idx in enumerate(_get_indices_from_input_subscript(out_spec,
                                                                 is_output=True)):
        index_to_descrs[idx] = EinsumElementwiseAxis(idim)

    if len(in_spec.split(",")) != len(expr.args):
        return False

    for in_subscript, access_descrs in zip(in_spec.split(","),
                                           expr.access_descriptors, strict=True):
        indices = _get_indices_from_input_subscript(in_subscript,
                                                    is_output=False)
        if len(indices) != len(access_descrs):
            return False

        # {{{ add reduction dims to 'index_to_descr', check for any inconsistencies

        for idx, access_descr in zip(indices, access_descrs, strict=True):

            try:
                if index_to_descrs[idx] != access_descr:
                    return False
            except KeyError:
                if not isinstance(access_descr, EinsumReductionAxis):
                    return False
                index_to_descrs[idx] = access_descr

        # }}}

    return True

# }}}


# {{{ ListOfDirectPredecessorsGetter

class ListOfDirectPredecessorsGetter(
        Mapper[
            list[ArrayOrNames | FunctionDefinition],
            list[ArrayOrNames],
            []]):
    """
    Helper to get the
    `direct predecessors
    <https://en.wikipedia.org/wiki/Glossary_of_graph_theory#direct_predecessor>`__
    of a node.

    .. note::

        We only consider the predecessors of a node in a data-flow sense.
    """
    def __init__(self, *, include_functions: bool = False) -> None:
        super().__init__()
        self.include_functions = include_functions

    def _get_preds_from_shape(self, shape: ShapeType) -> list[ArrayOrNames]:
        return [dim for dim in shape if isinstance(dim, Array)]

    def map_dict_of_named_arrays(
            self, expr: DictOfNamedArrays) -> list[ArrayOrNames]:
        return list(expr._data.values())

    def map_index_lambda(self, expr: IndexLambda) -> list[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape) + list(expr.bindings.values())

    def map_stack(self, expr: Stack) -> list[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape) + list(expr.arrays)

    def map_concatenate(self, expr: Concatenate) -> list[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape) + list(expr.arrays)

    def map_einsum(self, expr: Einsum) -> list[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape) + list(expr.args)

    def map_loopy_call(self, expr: LoopyCall) -> list[ArrayOrNames]:
        return [ary for ary in expr.bindings.values() if isinstance(ary, Array)]

    def map_loopy_call_result(self, expr: NamedArray) -> list[ArrayOrNames]:
        from pytato.loopy import LoopyCall, LoopyCallResult
        assert isinstance(expr, LoopyCallResult)
        assert isinstance(expr._container, LoopyCall)
        return [
            *self._get_preds_from_shape(expr.shape),
            expr._container]

    def _map_index_base(self, expr: IndexBase) -> list[ArrayOrNames]:
        return (
            self._get_preds_from_shape(expr.shape)
            + [expr.array]
            + [idx for idx in expr.indices if isinstance(idx, Array)])

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def _map_index_remapping_base(self, expr: IndexRemappingBase
                                  ) -> list[ArrayOrNames]:
        return [expr.array]

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_input_base(self, expr: InputArgumentBase) \
            -> list[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_distributed_recv(self,
                             expr: DistributedRecv) -> list[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> list[ArrayOrNames]:
        return [expr.send.data, expr.passthrough_data]

    def map_call(
            self, expr: Call) -> list[ArrayOrNames | FunctionDefinition]:
        result: list[ArrayOrNames | FunctionDefinition] = []
        if self.include_functions:
            result.append(expr.function)
        result += list(expr.bindings.values())
        return result

    def map_function_definition(
            self, expr: FunctionDefinition) -> list[ArrayOrNames]:
        return list(expr.returns.values())

    def map_named_call_result(
            self, expr: NamedCallResult) -> list[ArrayOrNames]:
        return [expr._container]

# }}}


# {{{ DirectPredecessorsGetter

class DirectPredecessorsGetter:
    """
    Helper to get the
    `direct predecessors
    <https://en.wikipedia.org/wiki/Glossary_of_graph_theory#direct_predecessor>`__
    of a node.

    .. note::

        We only consider the predecessors of a node in a data-flow sense.
    """
    def __init__(self, *, include_functions: bool = False) -> None:
        self._pred_getter = \
            ListOfDirectPredecessorsGetter(include_functions=include_functions)

    @overload
    def __call__(
            self, expr: ArrayOrNames
            ) -> FrozenOrderedSet[ArrayOrNames | FunctionDefinition]:
        ...

    @overload
    def __call__(self, expr: FunctionDefinition) -> FrozenOrderedSet[ArrayOrNames]:
        ...

    def __call__(
            self,
            expr: ArrayOrNames | FunctionDefinition,
            ) -> (
                FrozenOrderedSet[ArrayOrNames | FunctionDefinition]
                | FrozenOrderedSet[ArrayOrNames]):
        """Get the direct predecessors of *expr*."""
        return FrozenOrderedSet(self._pred_getter(expr))

# }}}


# {{{ NodeCountMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class NodeCountMapper(CachedWalkMapper[[]]):
    """
    Counts the number of nodes of a given type in a DAG.

    .. autoattribute:: expr_type_counts
    .. autoattribute:: count_duplicates

       Dictionary mapping node types to number of nodes of that type.
    """

    def __init__(
            self,
            count_duplicates: bool = False,
            _visited_functions: set[VisitKeyT] | None = None,
            ) -> None:
        super().__init__(_visited_functions=_visited_functions)

        self.expr_type_counts: dict[type[Any], int] = defaultdict(int)
        self.count_duplicates: bool = count_duplicates

    @override
    def get_cache_key(self, expr: ArrayOrNames) -> int | ArrayOrNames:
        # Returns unique nodes only if count_duplicates is False
        return id(expr) if self.count_duplicates else expr

    @override
    def get_function_definition_cache_key(
            self, expr: FunctionDefinition) -> int | FunctionDefinition:
        # Returns unique nodes only if count_duplicates is False
        return id(expr) if self.count_duplicates else expr

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            count_duplicates=self.count_duplicates,
            _visited_functions=self._visited_functions)

    @override
    def post_visit(self, expr: ArrayOrNames | FunctionDefinition) -> None:
        if not isinstance(expr, DictOfNamedArrays):
            self.expr_type_counts[type(expr)] += 1


def get_node_type_counts(
        outputs: ArrayOrNames,
        count_duplicates: bool = False
        ) -> dict[type[Any], int]:
    """
    Returns a dictionary mapping node types to node count for that type
    in DAG *outputs*.

    Instances of `DictOfNamedArrays` are excluded from counting.
    """

    ncm = NodeCountMapper(count_duplicates)
    ncm(outputs)

    return ncm.expr_type_counts


def get_num_nodes(
        outputs: ArrayOrNames,
        count_duplicates: bool | None = None
        ) -> int:
    """
    Returns the number of nodes in DAG *outputs*.
    Instances of `DictOfNamedArrays` are excluded from counting.
    """
    if count_duplicates is None:
        from warnings import warn
        warn(
            "The default value of 'count_duplicates' will change "
            "from True to False in 2025. "
            "For now, pass the desired value explicitly.",
            DeprecationWarning, stacklevel=2)
        count_duplicates = True

    ncm = NodeCountMapper(count_duplicates)
    ncm(outputs)

    return sum(ncm.expr_type_counts.values())

# }}}


# {{{ NodeMultiplicityMapper


class NodeMultiplicityMapper(CachedWalkMapper[[]]):
    """
    Computes the multiplicity of each unique node in a DAG.

    The multiplicity of a node `x` is the number of nodes with distinct `id()`\\ s
    that equal `x`.

    .. autoattribute:: expr_multiplicity_counts
    """
    def __init__(self, _visited_functions: set[Any] | None = None) -> None:
        super().__init__(_visited_functions=_visited_functions)

        self.expr_multiplicity_counts: \
            dict[ArrayOrNames | FunctionDefinition, int] = defaultdict(int)

    @override
    def get_cache_key(self, expr: ArrayOrNames) -> int:
        # Returns each node, including nodes that are duplicates
        return id(expr)

    @override
    def get_function_definition_cache_key(self, expr: FunctionDefinition) -> int:
        # Returns each node, including nodes that are duplicates
        return id(expr)

    @override
    def post_visit(self, expr: ArrayOrNames | FunctionDefinition) -> None:
        if not isinstance(expr, DictOfNamedArrays):
            self.expr_multiplicity_counts[expr] += 1


def get_node_multiplicities(
        outputs: ArrayOrNames) -> dict[ArrayOrNames | FunctionDefinition, int]:
    """
    Returns the multiplicity per `expr`.
    """
    nmm = NodeMultiplicityMapper()
    nmm(outputs)

    return nmm.expr_multiplicity_counts

# }}}


# {{{ CallSiteCountMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class CallSiteCountMapper(CachedWalkMapper[[]]):
    """
    Counts the number of :class:`~pytato.Call` nodes in a DAG.

    .. attribute:: count

       The number of nodes.
    """

    def __init__(self, _visited_functions: set[VisitKeyT] | None = None) -> None:
        super().__init__(_visited_functions=_visited_functions)
        self.count = 0

    @override
    def get_cache_key(self, expr: ArrayOrNames) -> int:
        return id(expr)

    @override
    def get_function_definition_cache_key(self, expr: FunctionDefinition) -> int:
        return id(expr)

    @override
    def post_visit(self, expr: ArrayOrNames | FunctionDefinition) -> None:
        if isinstance(expr, Call):
            self.count += 1

    @override
    def map_function_definition(self, expr: FunctionDefinition) -> None:
        if not self.visit(expr):
            return

        new_mapper = self.clone_for_callee(expr)
        for subexpr in expr.returns.values():
            new_mapper(subexpr)
        self.count += new_mapper.count

        self.post_visit(expr)


def get_num_call_sites(outputs: ArrayOrNames) -> int:
    """Returns the number of nodes in DAG *outputs*."""
    cscm = CallSiteCountMapper()
    cscm(outputs)

    return cscm.count

# }}}


# {{{ TagCountMapper

class TagCountMapper(CombineMapper[int, Never]):
    """
    Returns the number of nodes in a DAG that are tagged with all the tag types in
    *tag_types*.
    """

    def __init__(
            self,
            tag_types:
                type[pytools.tag.Tag]
                | Iterable[type[pytools.tag.Tag]]) -> None:
        super().__init__()
        if isinstance(tag_types, type):
            tag_types = frozenset((tag_types,))
        elif not isinstance(tag_types, frozenset):
            tag_types = frozenset(tag_types)
        self._tag_types = tag_types

    def combine(self, *args: int) -> int:
        return sum(args)

    def rec(self, expr: ArrayOrNames) -> int:
        inputs = self._make_cache_inputs(expr)
        try:
            return self._cache_retrieve(inputs)
        except KeyError:
            # Intentionally going to Mapper instead of super() to avoid
            # double caching when subclasses of CachedMapper override rec,
            # see https://github.com/inducer/pytato/pull/585
            s = Mapper.rec(self, expr)
            if (
                    isinstance(expr, Array)
                    and (
                        self._tag_types
                        <= frozenset(type(tag) for tag in expr.tags))):
                result = 1 + s
            else:
                result = 0 + s

            self._cache_add(inputs, 0)
            return result


def get_num_tags_of_type(
        outputs: ArrayOrNames,
        tag_types: type[pytools.tag.Tag] | Iterable[type[pytools.tag.Tag]]) -> int:
    """Returns the number of nodes in DAG *outputs* that are tagged with
    all the tag types in *tag_types*."""

    tcm = TagCountMapper(tag_types)

    return tcm(outputs)

# }}}


# {{{ PytatoKeyBuilder

class PytatoKeyBuilder(LoopyKeyBuilder):
    """A custom :class:`pytools.persistent_dict.KeyBuilder` subclass
    for objects within :mod:`pytato`.
    """
    # The types below aren't immutable in general, but in the context of
    # pytato, they are used as such.

    def update_for_ndarray(self, key_hash: Any, key: Any) -> None:
        import numpy as np
        assert isinstance(key, np.ndarray)
        self.rec(key_hash, key.data.tobytes())

    def update_for_TaggableCLArray(self, key_hash: Any, key: Any) -> None:
        from arraycontext.impl.pyopencl.taggable_cl_array import (  # pylint: disable=import-error
            TaggableCLArray,
        )
        assert isinstance(key, TaggableCLArray)
        self.rec(key_hash, key.get())

    def update_for_Array(self, key_hash: Any, key: Any) -> None:
        from pyopencl.array import Array
        assert isinstance(key, Array)
        self.rec(key_hash, key.get())

# }}}


# {{{ flop counting

@dataclass
class UndefinedOpFlopCountError(ValueError):
    op_name: str


class _PerEntryFlopCounter(CombineMapper[int, Never]):
    def __init__(self, op_name_to_num_flops: Mapping[str, int]) -> None:
        super().__init__()
        self.scalar_flop_counter: ScalarFlopCounter = ScalarFlopCounter(
            op_name_to_num_flops)
        self.node_to_nflops: dict[Array, int] = {}

    @override
    def combine(self, *args: int) -> int:
        return sum(args)

    def _get_own_flop_count(self, expr: Array) -> int:
        if isinstance(
                expr,
                (
                    DataWrapper,
                    Placeholder,
                    NamedArray,
                    DistributedRecv,
                    DistributedSendRefHolder)):
            return 0
        nflops = self.scalar_flop_counter(to_index_lambda(expr).expr)
        if not isinstance(nflops, int):
            # Restricting to numerical result here because the flop counters that use
            # this mapper subsequently multiply the result by things that are
            # potentially arrays (e.g., shape components), and arrays and scalar
            # expressions are not interoperable
            from pytato.scalar_expr import OpFlops, OpFlopsCollector
            op_flops: frozenset[OpFlops] = OpFlopsCollector()(nflops)
            if op_flops:
                raise UndefinedOpFlopCountError(next(iter(op_flops)).op)
            else:
                raise AssertionError
        return nflops

    @override
    def rec(self, expr: ArrayOrNames) -> int:
        inputs = self._make_cache_inputs(expr)
        try:
            return self._cache_retrieve(inputs)
        except KeyError:
            result: int
            if isinstance(expr, Array) and not is_materialized(expr):
                result = (
                    self._get_own_flop_count(expr)
                    # Intentionally going to Mapper instead of super() to avoid
                    # double caching when subclasses of CachedMapper override rec,
                    # see https://github.com/inducer/pytato/pull/585
                    + cast("int", Mapper.rec(self, expr)))
            else:
                result = 0
            if isinstance(expr, Array):
                self.node_to_nflops[expr] = result
            return self._cache_add(inputs, result)


class MaterializedNodeFlopCounter(CachedWalkMapper[[]]):
    """
    Mapper that counts the number of floating point operations of each materialized
    expression in a DAG.

    .. note::

        Flops from nodes inside function calls are accumulated onto the corresponding
        call node.
    """
    def __init__(
            self,
            op_name_to_num_flops: Mapping[str, int],
            ) -> None:
        super().__init__()
        self.op_name_to_num_flops: Mapping[str, int] = op_name_to_num_flops
        self.materialized_node_to_nflops: dict[Array, ArrayOrScalar] = {}
        self._per_entry_flop_counter: _PerEntryFlopCounter = _PerEntryFlopCounter(
            self.op_name_to_num_flops)

    @override
    def get_cache_key(self, expr: ArrayOrNames) -> VisitKeyT:
        return expr

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        raise AssertionError("Control shouldn't reach this point.")

    @override
    def map_function_definition(self, expr: FunctionDefinition) -> None:
        if not self.visit(expr):
            return

        raise NotImplementedError(
            f"{type(self).__name__} does not support functions.")

    @override
    def map_call(self, expr: Call) -> None:
        if not self.visit(expr):
            return

        raise NotImplementedError(
            f"{type(self).__name__} does not support functions.")

    @override
    def post_visit(self, expr: ArrayOrNames | FunctionDefinition) -> None:
        if not is_materialized(expr):
            return
        assert isinstance(expr, Array)
        if has_taggable_materialization(expr):
            unmaterialized_expr = expr.without_tags(ImplStored())
            self.materialized_node_to_nflops[expr] = (
                product(expr.shape)
                * self._per_entry_flop_counter(unmaterialized_expr))
        else:
            self.materialized_node_to_nflops[expr] = 0


class _UnmaterializedSubexpressionUseCounter(CombineMapper[dict[Array, int], Never]):
    @override
    def combine(self, *args: dict[Array, int]) -> dict[Array, int]:
        result: dict[Array, int] = defaultdict(int)
        for arg in args:
            for ary, nuses in arg.items():
                result[ary] += nuses
        return result

    @override
    def rec(self, expr: ArrayOrNames) -> dict[Array, int]:
        inputs = self._make_cache_inputs(expr)
        try:
            return self._cache_retrieve(inputs)
        except KeyError:
            result: dict[Array, int]
            if isinstance(expr, Array) and not is_materialized(expr):
                # Intentionally going to Mapper instead of super() to avoid
                # double caching when subclasses of CachedMapper override rec,
                # see https://github.com/inducer/pytato/pull/585
                result = self.combine(
                    {expr: 1}, cast("dict[Array, int]", Mapper.rec(self, expr)))
            else:
                result = {}
            return self._cache_add(inputs, result)


@dataclass
class UnmaterializedNodeFlopCounts:
    """
    Floating point operation counts for an unmaterialized node. See
    :func:`get_unmaterialized_node_flop_counts` for details.
    """
    materialized_successor_to_contrib_nflops: dict[Array, ArrayOrScalar]
    nflops_if_materialized: ArrayOrScalar


class UnmaterializedNodeFlopCounter(CachedWalkMapper[[]]):
    """
    Mapper that counts the accumulated number of floating point operations that each
    unmaterialized expression contributes to materialized expressions in the DAG.

    .. note::

        This mapper does not descend into functions.
    """
    def __init__(
            self,
            op_name_to_num_flops: Mapping[str, int]) -> None:
        super().__init__()
        self.op_name_to_num_flops: Mapping[str, int] = op_name_to_num_flops
        self.unmaterialized_node_to_flop_counts: \
            dict[Array, UnmaterializedNodeFlopCounts] = {}
        self._per_entry_flop_counter: _PerEntryFlopCounter = _PerEntryFlopCounter(
            self.op_name_to_num_flops)

    @override
    def get_cache_key(self, expr: ArrayOrNames) -> VisitKeyT:
        return expr

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        raise AssertionError("Control shouldn't reach this point.")

    @override
    def map_function_definition(self, expr: FunctionDefinition) -> None:
        if not self.visit(expr):
            return

        raise NotImplementedError(
            f"{type(self).__name__} does not support functions.")

    @override
    def map_call(self, expr: Call) -> None:
        if not self.visit(expr):
            return

        raise NotImplementedError(
            f"{type(self).__name__} does not support functions.")

    @override
    def post_visit(self, expr: ArrayOrNames | FunctionDefinition) -> None:
        if not is_materialized(expr) or not has_taggable_materialization(expr):
            return
        assert isinstance(expr, Array)
        unmaterialized_expr = expr.without_tags(ImplStored())
        subexpr_to_nuses = _UnmaterializedSubexpressionUseCounter()(
            unmaterialized_expr)
        del subexpr_to_nuses[unmaterialized_expr]
        self._per_entry_flop_counter(unmaterialized_expr)
        for subexpr, nuses in subexpr_to_nuses.items():
            per_entry_nflops = self._per_entry_flop_counter.node_to_nflops[subexpr]
            if subexpr not in self.unmaterialized_node_to_flop_counts:
                nflops_if_materialized = product(subexpr.shape) * per_entry_nflops
                flop_counts = UnmaterializedNodeFlopCounts({}, nflops_if_materialized)
                self.unmaterialized_node_to_flop_counts[subexpr] = flop_counts
            else:
                flop_counts = self.unmaterialized_node_to_flop_counts[subexpr]
            assert expr not in flop_counts.materialized_successor_to_contrib_nflops
            flop_counts.materialized_successor_to_contrib_nflops[expr] = (
                nuses * product(expr.shape) * per_entry_nflops)


# FIXME: Should this be added to normalize_outputs?
def _normalize_materialization(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    # Make sure outputs are materialized
    if isinstance(expr, DictOfNamedArrays):
        output_to_materialized_output: dict[Array, Array] = {
            ary: (
                ary.tagged(ImplStored())
                if has_taggable_materialization(ary)
                else ary)
            for ary in expr._data.values()}

        def replace_with_materialized(ary: ArrayOrNames) -> ArrayOrNames:
            if not isinstance(ary, Array):
                return ary
            try:
                return output_to_materialized_output[ary]
            except KeyError:
                return ary

        expr = map_and_copy(expr, replace_with_materialized)

    return expr


def get_default_op_name_to_num_flops() -> dict[str, int]:
    """
    Returns a mapping from operator name to floating point operation count for
    operators that are almost always a single flop.
    """
    return {
        "+": 1,
        "*": 1,
        "==": 1,
        "!=": 1,
        "<": 1,
        ">": 1,
        "<=": 1,
        ">=": 1,
        "min": 1,
        "max": 1}


def get_num_flops(
        expr: ArrayOrNames,
        op_name_to_num_flops: Mapping[str, int] | None = None,
    ) -> ArrayOrScalar:
    """
    Count the total number of floating point operations in the DAG *expr*.

    Counts flops as if emitting a statement at each materialized node (i.e., a node
    tagged with :class:`pytato.tags.ImplStored`) that computes everything up to
    (not including) its materialized predecessors. The total flop count is the sum
    over all materialized nodes.

    .. note::

        For arrays whose index lambda form contains :class:`pymbolic.primitives.If`,
        this function assumes a SIMT-like model of computation in which the per-entry
        cost is the sum of the costs of the two branches.

    .. note::

        Does not support functions. Function calls must be inlined before calling.
    """
    from pytato.codegen import normalize_outputs
    expr = normalize_outputs(expr)
    expr = _normalize_materialization(expr)

    if op_name_to_num_flops is None:
        op_name_to_num_flops = get_default_op_name_to_num_flops()

    fc = MaterializedNodeFlopCounter(op_name_to_num_flops)
    fc(expr)

    return sum(fc.materialized_node_to_nflops.values())


def get_materialized_node_flop_counts(
        expr: ArrayOrNames,
        op_name_to_num_flops: Mapping[str, int] | None = None,
    ) -> dict[Array, ArrayOrScalar]:
    """
    Returns a dictionary mapping materialized nodes in DAG *expr* to their floating
    point operation count.

    Counts flops as if emitting a statement at each materialized node (i.e., a node
    tagged with :class:`pytato.tags.ImplStored`) that computes everything up to
    (not including) its materialized predecessors.

    .. note::

        For arrays whose index lambda form contains :class:`pymbolic.primitives.If`,
        this function assumes a SIMT-like model of computation in which the per-entry
        cost is the sum of the costs of the two branches.

    .. note::

        Does not support functions. Function calls must be inlined before calling.
    """
    from pytato.codegen import normalize_outputs
    expr = normalize_outputs(expr)
    expr = _normalize_materialization(expr)

    if op_name_to_num_flops is None:
        op_name_to_num_flops = get_default_op_name_to_num_flops()

    fc = MaterializedNodeFlopCounter(op_name_to_num_flops)
    fc(expr)

    return fc.materialized_node_to_nflops


def get_unmaterialized_node_flop_counts(
        expr: ArrayOrNames,
        op_name_to_num_flops: Mapping[str, int] | None = None,
    ) -> dict[Array, UnmaterializedNodeFlopCounts]:
    """
    Returns a dictionary mapping unmaterialized nodes in DAG *expr* to a
    :class:`UnmaterializedNodeFlopCounts` containing floating-point operation count
    information.

    The :class:`UnmaterializedNodeFlopCounts` instance for each unmaterialized node
    (i.e., a node that can be tagged with :class:`pytato.tags.ImplStored` but isn't)
    contains `materialized_successor_to_contrib_nflops` and `nflops_if_materialized`
    attributes. The former is a mapping from each materialized successor of the
    unmaterialized node to the number of flops the node contributes to evaluating
    that successor (this includes flops from the predecessors of the unmaterialized
    node). The latter is the number of flops that would be required to evaluate the
    unmaterialized node if it was materialized instead.

    .. note::

        For arrays whose index lambda form contains :class:`pymbolic.primitives.If`,
        this function assumes a SIMT-like model of computation in which the per-entry
        cost is the sum of the costs of the two branches.

    .. note::

        Does not support functions. Function calls must be inlined before calling.
    """
    from pytato.codegen import normalize_outputs
    expr = normalize_outputs(expr)
    expr = _normalize_materialization(expr)

    if op_name_to_num_flops is None:
        op_name_to_num_flops = get_default_op_name_to_num_flops()

    fc = UnmaterializedNodeFlopCounter(op_name_to_num_flops)
    fc(expr)

    return fc.unmaterialized_node_to_flop_counts

# }}}


# vim: fdm=marker
