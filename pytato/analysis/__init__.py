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
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, overload

from orderedsets import FrozenOrderedSet
from typing_extensions import Never, Self, override

from loopy.tools import LoopyKeyBuilder

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
from pytato.transform import (
    ArrayOrNames,
    MapAndReduceMapper,
    Mapper,
    VisitKeyT,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import pytools.tag

    from pytato.distributed.nodes import DistributedRecv, DistributedSendRefHolder
    from pytato.loopy import LoopyCall

__doc__ = """
.. currentmodule:: pytato.analysis

.. autofunction:: get_nusers
.. autofunction:: get_list_of_users

.. autofunction:: is_einsum_similar_to_subscript

.. autoclass:: DirectPredecessorsGetter
.. autoclass:: ListOfDirectPredecessorsGetter

.. autofunction:: get_node_type_counts
.. autofunction:: get_num_nodes
.. autofunction:: get_node_multiplicities
.. autofunction:: get_num_node_instances_of
.. autofunction:: get_num_call_sites
.. autofunction:: get_num_tags_of_type
"""


# {{{ reduce_dicts

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


def reduce_dicts(
        # FIXME: Is there a way to make argument type annotation more specific?
        function: Callable[..., ValueT],
        iterable: Iterable[dict[KeyT, ValueT]]) -> dict[KeyT, ValueT]:
    """
    Apply *function* to the collection of values corresponding to each unique key in
    *iterable*.
    """
    key_to_list_of_values: dict[KeyT, list[ValueT]] = defaultdict(list)
    for d in iterable:
        for key, value in d.items():
            key_to_list_of_values[key].append(value)
    return {
        key: function(*list_of_values)
        for key, list_of_values in key_to_list_of_values.items()}

# }}}


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


# {{{ get_node_type_counts

NodeTypeCountDict: TypeAlias = dict[type[ArrayOrNames | FunctionDefinition], int]


# FIXME: I'm on the fence about whether these mapper classes should be kept around
# if they can be replaced with a call to map_and_reduce (which will be the case
# for most of these mappers once count_dict is removed). AFAIK the only real use
# case for using the mapper directly is if you want to compute a result for a
# collection of subexpressions by calling it multiple times while accumulating the set
# of visited nodes. But in most cases you could do that by putting them in a
# DictOfNamedArrays first and then call it once instead? *shrug*
# FIXME: optimize_mapper?
class NodeTypeCountMapper(MapAndReduceMapper[NodeTypeCountDict]):
    """Count the number of nodes of each type in a DAG."""
    def __init__(
            self,
            traverse_functions: bool = True,
            map_duplicates: bool = False,
            map_in_different_functions: bool = True,
            map_dict: bool = False) -> None:
        super().__init__(
            map_fn=lambda expr: {type(expr): 1},
            reduce_fn=lambda *args: reduce_dicts(
                lambda *values: sum(values, 0), args),
            traverse_functions=traverse_functions,
            map_duplicates=map_duplicates,
            map_in_different_functions=map_in_different_functions)

        # FIXME: Remove this once count_dict argument has been eliminated from
        # get_node_type_counts
        self.map_dict: bool = map_dict

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            traverse_functions=self.traverse_functions,
            map_duplicates=self.map_duplicates,
            map_in_different_functions=self.map_in_different_functions,
            map_dict=self.map_dict)

    # FIXME: Remove this once count_dict argument has been eliminated from
    # get_node_type_counts
    @override
    def map_dict_of_named_arrays(
            self,
            expr: DictOfNamedArrays,
            visited_node_keys: set[VisitKeyT] | None) -> NodeTypeCountDict:
        if self.map_dict:
            return self.reduce_fn(
                self.map_fn(expr),
                *(self.rec(val.expr, visited_node_keys) for val in expr.values()))
        else:
            return self.reduce_fn(
                *(self.rec(val.expr, visited_node_keys) for val in expr.values()))


def get_node_type_counts(
        outputs: ArrayOrNames | FunctionDefinition, *,
        traverse_functions: bool = True,
        count_duplicates: bool = False,
        count_in_different_functions: bool | None = None,
        count_dict: bool | None = None) -> NodeTypeCountDict:
    """
    Returns a dictionary mapping node types to node count for that type
    in DAG *outputs*.
    """
    if count_in_different_functions is None:
        from warnings import warn
        warn(
            "The default value of 'count_in_different_functions' will change "
            "from False to True in Q3 2026. "
            "For now, pass the desired value explicitly.",
            DeprecationWarning, stacklevel=2)
        count_in_different_functions = False

    # FIXME: Deprecate/remove count_dict argument entirely after default value is
    # changed
    if count_dict is None:
        from warnings import warn
        warn(
            "The default value of 'count_dict' will change "
            "from False to True in Q3 2026. "
            "For now, pass the desired value explicitly.",
            DeprecationWarning, stacklevel=2)
        count_dict = False

    ntcm = NodeTypeCountMapper(
        traverse_functions=traverse_functions,
        map_duplicates=count_duplicates,
        map_in_different_functions=count_in_different_functions,
        map_dict=count_dict)
    return ntcm(outputs)

# }}}


# {{{ get_num_nodes

# FIXME: optimize_mapper?
class NodeCountMapper(MapAndReduceMapper[int]):
    """Count the total number of nodes in a DAG."""
    def __init__(
            self,
            traverse_functions: bool = True,
            map_duplicates: bool = False,
            map_in_different_functions: bool = True,
            map_dict: bool = False) -> None:
        super().__init__(
            map_fn=lambda _: 1,
            reduce_fn=lambda *args: sum(args, 0),
            traverse_functions=traverse_functions,
            map_duplicates=map_duplicates,
            map_in_different_functions=map_in_different_functions)

        # FIXME: Remove this once count_dict argument has been eliminated from
        # get_num_nodes
        self.map_dict: bool = map_dict

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            traverse_functions=self.traverse_functions,
            map_duplicates=self.map_duplicates,
            map_in_different_functions=self.map_in_different_functions,
            map_dict=self.map_dict)

    # FIXME: Remove this once count_dict argument has been eliminated from
    # get_num_nodes
    @override
    def map_dict_of_named_arrays(
            self,
            expr: DictOfNamedArrays,
            visited_node_keys: set[VisitKeyT] | None) -> int:
        if self.map_dict:
            return self.reduce_fn(
                self.map_fn(expr),
                *(self.rec(val.expr, visited_node_keys) for val in expr.values()))
        else:
            return self.reduce_fn(
                *(self.rec(val.expr, visited_node_keys) for val in expr.values()))


def get_num_nodes(
        outputs: ArrayOrNames | FunctionDefinition, *,
        traverse_functions: bool = True,
        count_duplicates: bool = False,
        count_in_different_functions: bool | None = None,
        count_dict: bool | None = None) -> int:
    """
    Returns the number of nodes in DAG *outputs*.
    """
    if count_in_different_functions is None:
        from warnings import warn
        warn(
            "The default value of 'count_in_different_functions' will change "
            "from False to True in Q3 2026. "
            "For now, pass the desired value explicitly.",
            DeprecationWarning, stacklevel=2)
        count_in_different_functions = False

    # FIXME: Deprecate/remove count_dict argument entirely after default value is
    # changed
    if count_dict is None:
        from warnings import warn
        warn(
            "The default value of 'count_dict' will change "
            "from False to True in Q3 2026. "
            "For now, pass the desired value explicitly.",
            DeprecationWarning, stacklevel=2)
        count_dict = False

    ncm = NodeCountMapper(
        traverse_functions=traverse_functions,
        map_duplicates=count_duplicates,
        map_in_different_functions=count_in_different_functions,
        map_dict=count_dict)
    return ncm(outputs)

# }}}


# {{{ get_node_multiplicities

NodeMultiplicityDict: TypeAlias = dict[ArrayOrNames | FunctionDefinition, int]


# FIXME: optimize_mapper?
class NodeMultiplicityMapper(MapAndReduceMapper[NodeMultiplicityDict]):
    """
    Computes the multiplicity of each unique node in a DAG.

    See :func:`get_node_multiplicities` for details.
    """
    def __init__(
            self,
            traverse_functions: bool = True,
            map_duplicates: bool = True,
            map_in_different_functions: bool = True,
            map_dict: bool = False) -> None:
        super().__init__(
            map_fn=lambda expr: {expr: 1},
            reduce_fn=lambda *args: reduce_dicts(
                lambda *values: sum(values, 0), args),
            traverse_functions=traverse_functions,
            map_duplicates=map_duplicates,
            map_in_different_functions=map_in_different_functions)

        # FIXME: Remove this once count_dict argument has been eliminated from
        # get_node_multiplicities
        self.map_dict: bool = map_dict

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            traverse_functions=self.traverse_functions,
            map_duplicates=self.map_duplicates,
            map_in_different_functions=self.map_in_different_functions,
            map_dict=self.map_dict)

    # FIXME: Remove this once count_dict argument has been eliminated from
    # get_node_multiplicities
    @override
    def map_dict_of_named_arrays(
            self,
            expr: DictOfNamedArrays,
            visited_node_keys: set[VisitKeyT] | None) -> NodeMultiplicityDict:
        if self.map_dict:
            return self.reduce_fn(
                self.map_fn(expr),
                *(self.rec(val.expr, visited_node_keys) for val in expr.values()))
        else:
            return self.reduce_fn(
                *(self.rec(val.expr, visited_node_keys) for val in expr.values()))


def get_node_multiplicities(
        outputs: ArrayOrNames | FunctionDefinition,
        traverse_functions: bool = True,
        count_duplicates: bool = True,
        count_in_different_functions: bool = True,
        count_dict: bool | None = None) -> NodeMultiplicityDict:
    """
    Computes the multiplicity of each unique node in a DAG.

    The multiplicity of a node `x` is the number of times an object equal to `x` will
    be mapped during a cached DAG traversal. This varies depending on the combination
    of options used.

    :param count_duplicates: If *True*, distinct node instances equal to `x` will be
        counted.
    :param count_in_different_functions: If *True*, instances equal to `x` in
        different functions will be counted.
    """
    # FIXME: Deprecate/remove count_dict argument entirely after default value is
    # changed
    if count_dict is None:
        from warnings import warn
        warn(
            "The default value of 'count_dict' will change "
            "from False to True in Q3 2026. "
            "For now, pass the desired value explicitly.",
            DeprecationWarning, stacklevel=2)
        count_dict = False

    nmm = NodeMultiplicityMapper(
        traverse_functions=traverse_functions,
        map_duplicates=count_duplicates,
        map_in_different_functions=count_in_different_functions,
        map_dict=count_dict)
    return nmm(outputs)

# }}}


# {{{ get_num_node_instances_of

# FIXME: optimize_mapper?
class NodeInstanceCountMapper(MapAndReduceMapper[int]):
    """Count the number of nodes in a DAG that are instances of *node_type*."""
    def __init__(
            self,
            node_type:
                type[ArrayOrNames | FunctionDefinition]
                | tuple[type[ArrayOrNames | FunctionDefinition], ...],
            traverse_functions: bool = True,
            map_duplicates: bool = False,
            map_in_different_functions: bool = True) -> None:
        super().__init__(
            map_fn=lambda expr: int(isinstance(expr, node_type)),
            reduce_fn=lambda *args: sum(args, 0),
            traverse_functions=traverse_functions,
            map_duplicates=map_duplicates,
            map_in_different_functions=map_in_different_functions)

        self.node_type: \
            type[ArrayOrNames | FunctionDefinition] \
            | tuple[type[ArrayOrNames | FunctionDefinition], ...] = node_type

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            node_type=self.node_type,
            traverse_functions=self.traverse_functions,
            map_duplicates=self.map_duplicates,
            map_in_different_functions=self.map_in_different_functions)


def get_num_node_instances_of(
        outputs: ArrayOrNames | FunctionDefinition,
        node_type:
            type[ArrayOrNames | FunctionDefinition]
            | tuple[type[ArrayOrNames | FunctionDefinition], ...],
        traverse_functions: bool = True,
        count_duplicates: bool = False,
        count_in_different_functions: bool = True) -> int:
    """
    Returns the number of nodes in DAG *outputs* that are instances of *node_type*.
    """
    nicm = NodeInstanceCountMapper(
        node_type=node_type,
        traverse_functions=traverse_functions,
        map_duplicates=count_duplicates,
        map_in_different_functions=count_in_different_functions)
    return nicm(outputs)

# }}}


# {{{ get_num_call_sites

# FIXME: optimize_mapper?
class CallSiteCountMapper(MapAndReduceMapper[int]):
    """Count the number of :class:`~pytato.Call` nodes in a DAG."""
    def __init__(
            self,
            traverse_functions: bool = True,
            map_duplicates: bool = False,
            map_in_different_functions: bool = True) -> None:
        super().__init__(
            map_fn=lambda expr: int(isinstance(expr, Call)),
            reduce_fn=lambda *args: sum(args, 0),
            traverse_functions=traverse_functions,
            map_duplicates=map_duplicates,
            map_in_different_functions=map_in_different_functions)

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            traverse_functions=self.traverse_functions,
            map_duplicates=self.map_duplicates,
            map_in_different_functions=self.map_in_different_functions)


def get_num_call_sites(
        outputs: ArrayOrNames | FunctionDefinition,
        traverse_functions: bool = True,
        count_duplicates: bool | None = None,
        count_in_different_functions: bool = True) -> int:
    """Returns the number of :class:`pytato.function.Call` nodes in DAG *outputs*."""
    if count_duplicates is None:
        from warnings import warn
        warn(
            "The default value of 'count_duplicates' will change "
            "from True to False in Q3 2026. "
            "For now, pass the desired value explicitly.",
            DeprecationWarning, stacklevel=2)
        count_duplicates = True

    cscm = CallSiteCountMapper(
        traverse_functions=traverse_functions,
        map_duplicates=count_duplicates,
        map_in_different_functions=count_in_different_functions)
    return cscm(outputs)

# }}}


# {{{ get_num_tags_of_type

# FIXME: optimize_mapper?
class TagCountMapper(MapAndReduceMapper[int]):
    """
    Count the number of nodes in a DAG that are tagged with all the tag types in
    *tag_types*.
    """
    def __init__(
            self,
            tag_types:
                type[pytools.tag.Tag]
                | Iterable[type[pytools.tag.Tag]],
            traverse_functions: bool = False,
            map_duplicates: bool = False,
            map_in_different_functions: bool = True) -> None:
        super().__init__(
            map_fn=lambda expr: int(
                isinstance(expr, Array)
                and (
                    self.tag_types
                    <= frozenset(type(tag) for tag in expr.tags))),
            reduce_fn=lambda *args: sum(args, 0),
            traverse_functions=traverse_functions,
            map_duplicates=map_duplicates,
            map_in_different_functions=map_in_different_functions)

        if isinstance(tag_types, type):
            tag_types = frozenset((tag_types,))
        elif not isinstance(tag_types, frozenset):
            tag_types = frozenset(tag_types)
        self.tag_types: frozenset[type[pytools.tag.Tag]] = tag_types

    @override
    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            tag_types=self.tag_types,
            traverse_functions=self.traverse_functions,
            map_duplicates=self.map_duplicates,
            map_in_different_functions=self.map_in_different_functions)


def get_num_tags_of_type(
        outputs: ArrayOrNames | FunctionDefinition,
        tag_types: type[pytools.tag.Tag] | Iterable[type[pytools.tag.Tag]],
        traverse_functions: bool | None = None,
        count_duplicates: bool = False,
        count_in_different_functions: bool = True) -> int:
    """Returns the number of nodes in DAG *outputs* that are tagged with
    all the tag types in *tag_types*."""
    if traverse_functions is None:
        from warnings import warn
        warn(
            "The default value of 'traverse_functions' will change "
            "from False to True in Q3 2026. "
            "For now, pass the desired value explicitly.",
            DeprecationWarning, stacklevel=2)
        traverse_functions = False

    tcm = TagCountMapper(
        tag_types=tag_types,
        traverse_functions=traverse_functions,
        map_duplicates=count_duplicates,
        map_in_different_functions=count_in_different_functions)
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
