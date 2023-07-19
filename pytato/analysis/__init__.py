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

from typing import TYPE_CHECKING, Any, Never

from orderedsets import FrozenOrderedSet
from typing_extensions import Self

import pytools
from loopy.tools import LoopyKeyBuilder
from pymbolic.mapper.optimize import optimize_mapper

from pytato.array import (
    Array,
    Concatenate,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexLambda,
    IndexRemappingBase,
    InputArgumentBase,
    NamedArray,
    ShapeType,
    Stack,
)
from pytato.function import Call, FunctionDefinition, NamedCallResult
from pytato.transform import ArrayOrNames, CachedWalkMapper, CombineMapper, Mapper, P


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pytato.distributed.nodes import DistributedRecv, DistributedSendRefHolder
    from pytato.loopy import LoopyCall

__doc__ = """
.. currentmodule:: pytato.analysis

.. autofunction:: get_nusers

.. autofunction:: is_einsum_similar_to_subscript

.. autofunction:: get_num_nodes
.. autofunction:: get_max_node_depth

.. autofunction:: get_node_type_counts

.. autofunction:: get_node_multiplicities

.. autofunction:: get_num_call_sites

.. autoclass:: DirectPredecessorsGetter

.. autoclass:: TagCountMapper
.. autofunction:: get_num_tags_of_type
"""


# {{{ NUserCollector

class NUserCollector(Mapper[None, None, []]):
    """
    A :class:`pytato.transform.CachedWalkMapper` that records the number of
    times an array expression is a direct dependency of other nodes.

    .. note::

        - We do not consider the :class:`pytato.DistributedSendRefHolder`
          a user of :attr:`pytato.DistributedSendRefHolder.send`. This is
          because in a data flow sense, the send-ref holder does not use the
          send's data.
    """
    def __init__(self) -> None:
        from collections import defaultdict
        super().__init__()
        self._visited_ids: set[int] = set()
        self.nusers: dict[Array, int] = defaultdict(lambda: 0)

    # type-ignore reason: NUserCollector.rec's type does not match
    # Mapper.rec's type
    def rec(self, expr: ArrayOrNames) -> None:  # type: ignore
        # See CachedWalkMapper.rec on why we chose id(x) as the cache key.

        if id(expr) in self._visited_ids:
            return

        super().rec(expr)
        self._visited_ids.add(id(expr))

    def map_index_lambda(self, expr: IndexLambda) -> None:
        for ary in expr.bindings.values():
            self.nusers[ary] += 1
            self.rec(ary)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.nusers[dim] += 1
                self.rec(dim)

    def map_stack(self, expr: Stack) -> None:
        for ary in expr.arrays:
            self.nusers[ary] += 1
            self.rec(ary)

    def map_concatenate(self, expr: Concatenate) -> None:
        for ary in expr.arrays:
            self.nusers[ary] += 1
            self.rec(ary)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        for ary in expr.bindings.values():
            if isinstance(ary, Array):
                self.nusers[ary] += 1
                self.rec(ary)

    def map_einsum(self, expr: Einsum) -> None:
        for ary in expr.args:
            self.nusers[ary] += 1
            self.rec(ary)

        for dim in expr.shape:
            if isinstance(dim, Array):
                self.nusers[dim] += 1
                self.rec(dim)

    def map_named_array(self, expr: NamedArray) -> None:
        self.rec(expr._container)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        for child in expr._data.values():
            self.rec(child)

    def _map_index_base(self, expr: IndexBase) -> None:
        self.nusers[expr.array] += 1
        self.rec(expr.array)

        for idx in expr.indices:
            if isinstance(idx, Array):
                self.nusers[idx] += 1
                self.rec(idx)

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def _map_index_remapping_base(self, expr: IndexRemappingBase) -> None:
        self.nusers[expr.array] += 1
        self.rec(expr.array)

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_input_base(self, expr: InputArgumentBase) -> None:
        for dim in expr.shape:
            if isinstance(dim, Array):
                self.nusers[dim] += 1
                self.rec(dim)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_distributed_send_ref_holder(self, expr: DistributedSendRefHolder
                                        ) -> None:
        # Note: We do not consider 'expr.send.data' as a predecessor of *expr*,
        # as there is no dataflow from *expr.send.data* to *expr*
        self.nusers[expr.passthrough_data] += 1
        self.rec(expr.passthrough_data)
        self.rec(expr.send.data)

    def map_distributed_recv(self, expr: DistributedRecv) -> None:
        for dim in expr.shape:
            if isinstance(dim, Array):
                self.nusers[dim] += 1
                self.rec(dim)

    def map_call(self, expr: Call) -> None:
        for ary in expr.bindings.values():
            if isinstance(ary, Array):
                self.nusers[ary] += 1
                self.rec(ary)

    def map_named_call_result(self, expr: NamedCallResult) -> None:
        self.rec(expr._container)

# }}}


def get_nusers(outputs: Array | DictOfNamedArrays) -> Mapping[Array, int]:
    """
    For the DAG *outputs*, returns the mapping from each node to the number of
    nodes using its value within the DAG given by *outputs*.
    """
    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)
    nuser_collector = NUserCollector()
    nuser_collector(outputs)
    return nuser_collector.nusers


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
    string *subscripts* operated on *expr*'s :attr:`pytato.array.Einsum.args`
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


# {{{ DirectPredecessorsGetter

class DirectPredecessorsGetter(Mapper[frozenset[ArrayOrNames], Never, []]):
    """
    Mapper to get the
    `direct predecessors
    <https://en.wikipedia.org/wiki/Glossary_of_graph_theory#direct_predecessor>`__
    of a node.

    .. note::

        We only consider the predecessors of a nodes in a data-flow sense.
    """
    def _get_preds_from_shape(self, shape: ShapeType) -> FrozenOrderedSet[ArrayOrNames]:
        return FrozenOrderedSet(dim for dim in shape if isinstance(dim, Array))

    def map_index_lambda(self, expr: IndexLambda) -> FrozenOrderedSet[ArrayOrNames]:
        return (FrozenOrderedSet(expr.bindings.values())
                | self._get_preds_from_shape(expr.shape))

    def map_stack(self, expr: Stack) -> FrozenOrderedSet[ArrayOrNames]:
        return (FrozenOrderedSet(expr.arrays)
                | self._get_preds_from_shape(expr.shape))

    def map_concatenate(self, expr: Concatenate) -> FrozenOrderedSet[ArrayOrNames]:
        return (FrozenOrderedSet(expr.arrays)
                | self._get_preds_from_shape(expr.shape))

    def map_einsum(self, expr: Einsum) -> FrozenOrderedSet[ArrayOrNames]:
        return (FrozenOrderedSet(expr.args)
                | self._get_preds_from_shape(expr.shape))

    def map_loopy_call_result(self, expr: NamedArray) -> FrozenOrderedSet[ArrayOrNames]:
        from pytato.loopy import LoopyCall, LoopyCallResult
        assert isinstance(expr, LoopyCallResult)
        assert isinstance(expr._container, LoopyCall)
        return (FrozenOrderedSet(ary
                          for ary in expr._container.bindings.values()
                          if isinstance(ary, Array))
                | self._get_preds_from_shape(expr.shape))

    def _map_index_base(self, expr: IndexBase) -> FrozenOrderedSet[ArrayOrNames]:
        return (FrozenOrderedSet([expr.array])
                | FrozenOrderedSet(idx for idx in expr.indices
                            if isinstance(idx, Array))
                | self._get_preds_from_shape(expr.shape))

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def _map_index_remapping_base(self, expr: IndexRemappingBase
                                  ) -> FrozenOrderedSet[ArrayOrNames]:
        return FrozenOrderedSet([expr.array])

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_input_base(self, expr: InputArgumentBase) \
            -> FrozenOrderedSet[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_distributed_recv(self,
                             expr: DistributedRecv) -> FrozenOrderedSet[ArrayOrNames]:
        return self._get_preds_from_shape(expr.shape)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> FrozenOrderedSet[ArrayOrNames]:
        return FrozenOrderedSet([expr.passthrough_data])

    def map_call(self, expr: Call) -> FrozenOrderedSet[ArrayOrNames]:
        return FrozenOrderedSet(expr.bindings.values())

    def map_named_call_result(
            self, expr: NamedCallResult) -> FrozenOrderedSet[ArrayOrNames]:
        return FrozenOrderedSet([expr._container])


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
            _visited_functions: set[Any] | None = None,
            ) -> None:
        super().__init__(_visited_functions=_visited_functions)

        from collections import defaultdict
        self.expr_type_counts: dict[type[Any], int] = defaultdict(int)
        self.count_duplicates = count_duplicates

    def get_cache_key(self, expr: ArrayOrNames) -> int | ArrayOrNames:
        # Returns unique nodes only if count_duplicates is False
        return id(expr) if self.count_duplicates else expr

    def get_function_definition_cache_key(
            self, expr: FunctionDefinition) -> int | FunctionDefinition:
        # Returns unique nodes only if count_duplicates is False
        return id(expr) if self.count_duplicates else expr

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            count_duplicates=self.count_duplicates,
            _visited_functions=self._visited_functions)

    def post_visit(self, expr: Any) -> None:
        if not isinstance(expr, DictOfNamedArrays):
            self.expr_type_counts[type(expr)] += 1


def get_node_type_counts(
        outputs: Array | DictOfNamedArrays,
        count_duplicates: bool = False
        ) -> dict[type[Any], int]:
    """
    Returns a dictionary mapping node types to node count for that type
    in DAG *outputs*.

    Instances of `DictOfNamedArrays` are excluded from counting.
    """

    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)

    ncm = NodeCountMapper(count_duplicates)
    ncm(outputs)

    return ncm.expr_type_counts


def get_num_nodes(
        outputs: Array | DictOfNamedArrays,
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

    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)

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

        from collections import defaultdict
        self.expr_multiplicity_counts: dict[Array, int] = defaultdict(int)

    def get_cache_key(self, expr: ArrayOrNames) -> int:
        # Returns each node, including nodes that are duplicates
        return id(expr)

    def get_function_definition_cache_key(self, expr: FunctionDefinition) -> int:
        # Returns each node, including nodes that are duplicates
        return id(expr)

    def post_visit(self, expr: Any) -> None:
        if not isinstance(expr, DictOfNamedArrays):
            self.expr_multiplicity_counts[expr] += 1


def get_node_multiplicities(
        outputs: Array | DictOfNamedArrays) -> dict[Array, int]:
    """
    Returns the multiplicity per `expr`.
    """
    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)

    nmm = NodeMultiplicityMapper()
    nmm(outputs)

    return nmm.expr_multiplicity_counts

# }}}


# {{{ NodeMaxDepthMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class NodeMaxDepthMapper(CachedWalkMapper):
    """
    Finds the maximum depth of a node in a DAG.

    .. attribute:: max_depth

       The depth of the deepest node.
    """

    def __init__(self) -> None:
        super().__init__()
        # Want the first rec() call to increment to 0, so start at -1
        self.depth = -1
        self.max_depth = -1

    # FIXME: Do I need this?
    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self, expr: ArrayOrNames) -> int:  # type: ignore[override]
        return id(expr)

    def rec(self, expr: ArrayOrNames, *args: Any, **kwargs: Any) -> None:
        """Call the mapper method of *expr* and return the result."""
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)

        try:
            super().rec(expr, *args, **kwargs)
        finally:
            self.depth -= 1


def get_max_node_depth(outputs: Union[Array, DictOfNamedArrays]) -> int:
    """Finds the maximum depth of a node in *outputs*."""

    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)

    nmdm = NodeMaxDepthMapper()
    nmdm(outputs)

    return nmdm.max_depth

# }}}


# {{{ CallSiteCountMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class CallSiteCountMapper(CachedWalkMapper[[]]):
    """
    Counts the number of :class:`~pytato.Call` nodes in a DAG.

    .. attribute:: count

       The number of nodes.
    """

    def __init__(self, _visited_functions: set[Any] | None = None) -> None:
        super().__init__(_visited_functions=_visited_functions)
        self.count = 0

    def get_cache_key(self, expr: ArrayOrNames) -> int:
        return id(expr)

    def get_function_definition_cache_key(self, expr: FunctionDefinition) -> int:
        return id(expr)

    def map_function_definition(self, expr: FunctionDefinition) -> None:
        if not self.visit(expr):
            return

        new_mapper = self.clone_for_callee(expr)
        for subexpr in expr.returns.values():
            new_mapper(subexpr)
        self.count += new_mapper.count

        self.post_visit(expr)

    def post_visit(self, expr: Any) -> None:
        if isinstance(expr, Call):
            self.count += 1


def get_num_call_sites(outputs: Array | DictOfNamedArrays) -> int:
    """Returns the number of nodes in DAG *outputs*."""

    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)

    cscm = CallSiteCountMapper()
    cscm(outputs)

    return cscm.count

# }}}


# {{{ TagCountMapper

class TagCountMapper(CombineMapper[int, Never]):
    """
    Returns the number of nodes in a DAG that are tagged with all the tags in *tags*.
    """

    def __init__(self, tags: pytools.tag.Tag | Iterable[pytools.tag.Tag]) -> None:
        super().__init__()
        if isinstance(tags, pytools.tag.Tag):
            tags = frozenset((tags,))
        elif not isinstance(tags, frozenset):
            tags = frozenset(tags)
        self._tags = tags

    def combine(self, *args: int) -> int:
        return sum(args)

    def rec(self, expr: ArrayOrNames, *args: P.args, **kwargs: P.kwargs) -> int:
        key = self._cache.get_key(expr, *args, **kwargs)
        try:
            return self._cache.retrieve((expr, args, kwargs), key=key)
        except KeyError:
            s = super().rec(expr, *args, **kwargs)
            if isinstance(expr, Array) and self._tags <= expr.tags:
                result = 1 + s
            else:
                result = 0 + s

            self._cache.add((expr, args, kwargs),
                0,
                key=key)
            return result


def get_num_tags_of_type(
        outputs: Array | DictOfNamedArrays,
        tags: pytools.tag.Tag | Iterable[pytools.tag.Tag]) -> int:
    """Returns the number of nodes in DAG *outputs* that are tagged with
    all the tags in *tags*."""

    tcm = TagCountMapper(tags)

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

# vim: fdm=marker
