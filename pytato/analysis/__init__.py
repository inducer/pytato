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

from typing import (Mapping, Dict, Union, Set, Tuple, Any, FrozenSet,
                    TYPE_CHECKING)
from pytato.array import (Array, IndexLambda, Stack, Concatenate, Einsum,
                          DictOfNamedArrays, NamedArray,
                          IndexBase, IndexRemappingBase, InputArgumentBase,
                          ShapeType)
from pytato.function import FunctionDefinition, Call
from pytato.transform import Mapper, ArrayOrNames, CachedWalkMapper
from pytato.loopy import LoopyCall
from pymbolic.mapper.optimize import optimize_mapper
from pytools import memoize_method
from pytools.persistent_dict import KeyBuilder

from pymbolic.mapper.persistent_hash import PersistentHashWalkMapper

from immutables import Map

if TYPE_CHECKING:
    from pytato.distributed.nodes import DistributedRecv, DistributedSendRefHolder

__doc__ = """
.. currentmodule:: pytato.analysis

.. autofunction:: get_nusers

.. autofunction:: is_einsum_similar_to_subscript

.. autofunction:: get_num_nodes

.. autofunction:: get_num_call_sites

.. autofunction:: get_hash

.. autoclass:: DirectPredecessorsGetter
"""


# {{{ NUserCollector

class NUserCollector(Mapper):
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
        self._visited_ids: Set[int] = set()
        self.nusers: Dict[Array, int] = defaultdict(lambda: 0)

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

# }}}


def get_nusers(outputs: Union[Array, DictOfNamedArrays]) -> Mapping[Array, int]:
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
                                      ) -> Tuple[str, ...]:
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

    if is_output:
        if len(normalized_indices) != len(set(normalized_indices)):
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

    from pytato.array import (EinsumElementwiseAxis, EinsumReductionAxis,
                              EinsumAxisDescriptor)

    if not isinstance(expr, Einsum):
        raise TypeError(f"{expr} expected to be Einsum, got {type(expr)}.")

    if "->" not in subscripts:
        raise NotImplementedError("Comparing against implicit mode einsums:"
                                  " not supported.")

    in_spec, out_spec = subscripts.split("->")

    # build up a mapping from index names to axis descriptors
    index_to_descrs: Dict[str, EinsumAxisDescriptor] = {}

    for idim, idx in enumerate(_get_indices_from_input_subscript(out_spec,
                                                                 is_output=True)):
        index_to_descrs[idx] = EinsumElementwiseAxis(idim)

    if len(in_spec.split(",")) != len(expr.args):
        return False

    for in_subscript, access_descrs in zip(in_spec.split(","),
                                           expr.access_descriptors):
        indices = _get_indices_from_input_subscript(in_subscript,
                                                    is_output=False)
        if len(indices) != len(access_descrs):
            return False

        # {{{ add reduction dims to 'index_to_descr', check for any inconsistencies

        for idx, access_descr in zip(indices, access_descrs):

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

class DirectPredecessorsGetter(Mapper):
    """
    Mapper to get the
    `direct predecessors
    <https://en.wikipedia.org/wiki/Glossary_of_graph_theory#direct_predecessor>`__
    of a node.

    .. note::

        We only consider the predecessors of a nodes in a data-flow sense.
    """
    def _get_preds_from_shape(self, shape: ShapeType) -> FrozenSet[Array]:
        return frozenset({dim for dim in shape if isinstance(dim, Array)})

    def map_index_lambda(self, expr: IndexLambda) -> FrozenSet[Array]:
        return (frozenset(expr.bindings.values())
                | self._get_preds_from_shape(expr.shape))

    def map_stack(self, expr: Stack) -> FrozenSet[Array]:
        return (frozenset(expr.arrays)
                | self._get_preds_from_shape(expr.shape))

    def map_concatenate(self, expr: Concatenate) -> FrozenSet[Array]:
        return (frozenset(expr.arrays)
                | self._get_preds_from_shape(expr.shape))

    def map_einsum(self, expr: Einsum) -> FrozenSet[Array]:
        return (frozenset(expr.args)
                | self._get_preds_from_shape(expr.shape))

    def map_loopy_call_result(self, expr: NamedArray) -> FrozenSet[Array]:
        from pytato.loopy import LoopyCallResult, LoopyCall
        assert isinstance(expr, LoopyCallResult)
        assert isinstance(expr._container, LoopyCall)
        return (frozenset(ary
                          for ary in expr._container.bindings.values()
                          if isinstance(ary, Array))
                | self._get_preds_from_shape(expr.shape))

    def _map_index_base(self, expr: IndexBase) -> FrozenSet[Array]:
        return (frozenset([expr.array])
                | frozenset(idx for idx in expr.indices
                            if isinstance(idx, Array))
                | self._get_preds_from_shape(expr.shape))

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def _map_index_remapping_base(self, expr: IndexRemappingBase
                                  ) -> FrozenSet[Array]:
        return frozenset([expr.array])

    map_roll = _map_index_remapping_base
    map_axis_permutation = _map_index_remapping_base
    map_reshape = _map_index_remapping_base

    def _map_input_base(self, expr: InputArgumentBase) -> FrozenSet[Array]:
        return self._get_preds_from_shape(expr.shape)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_distributed_recv(self, expr: DistributedRecv) -> FrozenSet[Array]:
        return self._get_preds_from_shape(expr.shape)

    def map_distributed_send_ref_holder(self,
                                        expr: DistributedSendRefHolder
                                        ) -> FrozenSet[Array]:
        return frozenset([expr.passthrough_data])

# }}}


# {{{ NodeCountMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class NodeCountMapper(CachedWalkMapper):
    """
    Counts the number of nodes in a DAG.

    .. attribute:: count

       The number of nodes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self, expr: ArrayOrNames) -> int:  # type: ignore[override]
        return id(expr)

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def post_visit(self, expr: Any) -> None:  # type: ignore[override]
        self.count += 1


def get_num_nodes(outputs: Union[Array, DictOfNamedArrays]) -> int:
    """Returns the number of nodes in DAG *outputs*."""

    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)

    ncm = NodeCountMapper()
    ncm(outputs)

    return ncm.count

# }}}


# {{{ CallSiteCountMapper

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class CallSiteCountMapper(CachedWalkMapper):
    """
    Counts the number of :class:`~pytato.Call` nodes in a DAG.

    .. attribute:: count

       The number of nodes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self, expr: ArrayOrNames) -> int:  # type: ignore[override]
        return id(expr)

    @memoize_method
    def map_function_definition(self, /, expr: FunctionDefinition,
                                *args: Any, **kwargs: Any) -> None:
        if not self.visit(expr):
            return

        new_mapper = self.clone_for_callee()
        for subexpr in expr.returns.values():
            new_mapper(subexpr, *args, **kwargs)

        self.count += new_mapper.count

        self.post_visit(expr, *args, **kwargs)

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def post_visit(self, expr: Any) -> None:  # type: ignore[override]
        if isinstance(expr, Call):
            self.count += 1


def get_num_call_sites(outputs: Union[Array, DictOfNamedArrays]) -> int:
    """Returns the number of nodes in DAG *outputs*."""

    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)

    cscm = CallSiteCountMapper()
    cscm(outputs)

    return cscm.count

# }}}


# {{{ get_hash

class PytatoKeyBuilder(KeyBuilder):
    """A custom :class:`pytools.persistent_dict.KeyBuilder` subclass
    for objects within :mod:`pytato`.
    """

    update_for_list = KeyBuilder.update_for_tuple
    update_for_set = KeyBuilder.update_for_frozenset

    def update_for_dict(self, key_hash, key):
        from pytools import unordered_hash
        unordered_hash(
            key_hash,
            (self.rec(self.new_hash(), (k, v)).digest()
                for k, v in key.items()))

    update_for_defaultdict = update_for_dict

    def update_for_ndarray(self, key_hash, key):
        self.rec(key_hash, hash(key.data.tobytes()))

    def update_for_frozenset(self, key_hash, key):
        for set_key in sorted(key,
                key=lambda obj: type(obj).__name__ + str(obj)):
            self.rec(key_hash, set_key)

    def update_for_BasicSet(self, key_hash, key):  # noqa
        from islpy import Printer
        prn = Printer.to_str(key.get_ctx())
        getattr(prn, "print_"+key._base_name)(key)
        key_hash.update(prn.get_str().encode("utf8"))

    def update_for_Map(self, key_hash, key):  # noqa
        import islpy as isl
        if isinstance(key, Map):
            self.update_for_dict(key_hash, key)
        elif isinstance(key, isl.Map):
            self.update_for_BasicSet(key_hash, key)
        else:
            raise AssertionError()

    def update_for_pymbolic_expression(self, key_hash, key):
        if key is None:
            self.update_for_NoneType(key_hash, key)
        else:
            PersistentHashWalkMapper(key_hash)(key)

    def update_for_Product(self, key_hash, key):
        PersistentHashWalkMapper(key_hash)(key)

    def update_for_Array(self, key_hash, key):
        self.update_for_ndarray(key_hash, key.get())

    def update_for_int64(self, key_hash, key):
        self.rec(key_hash, int(key))

    def update_for_Reduce(self, key_hash, key):
        self.rec(key_hash, hash(key))

    update_for_Sum = update_for_Product
    update_for_If = update_for_Product
    update_for_LogicalOr = update_for_Product
    update_for_Call = update_for_Product
    update_for_Comparison = update_for_Product
    update_for_Quotient = update_for_Product
    update_for_Power = update_for_Product
    update_for_PMap = update_for_dict  # noqa: N815


def get_hash(outputs: Union[Array, DictOfNamedArrays]) -> str:
    """Returns a hash of the DAG *outputs*."""

    from pytato.codegen import normalize_outputs
    outputs = normalize_outputs(outputs)

    hm = PytatoKeyBuilder()

    return hm(outputs)

# }}}

# vim: fdm=marker
