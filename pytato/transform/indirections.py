__copyright__ = "Copyright (C) 2023 Kaushik Kulkarni"

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
from typing import (Any, Dict, Mapping, Tuple, TypeAlias, Iterable,
                    FrozenSet, Union, Set, List, Optional, Callable)
from pytato.array import (Array, InputArgumentBase, DictOfNamedArrays,
                          IndexLambda, ShapeComponent,
                          NormalizedSlice,
                          AdvancedIndexInContiguousAxes,
                          AdvancedIndexInNoncontiguousAxes,
                          BasicIndex, Reshape, Roll, Einsum, AxisPermutation,
                          Stack, Concatenate, DataWrapper, IndexBase, Placeholder)

from pytato.tags import ImplStored
from pytato.transform import (CachedMapper, CopyMapper, Mapper, MappedT,
                              ArrayOrNames, CombineMapper)
from dataclasses import dataclass
import pytato.scalar_expr as scalar_expr
import pymbolic.primitives as prim
from immutables import Map
from pytato.utils import are_shape_components_equal

_ComposedIndirectionT: TypeAlias = Tuple[Array, ...]
IndexT: TypeAlias = Union[Array, NormalizedSlice]
IndexStackT: TypeAlias = Tuple[IndexT, ...]


def _is_materialized(expr: Array) -> bool:
    """
    Returns true if an array is materialized. An array is considered to be
    materialized if it is either a :class:`pytato.array.InputArgumentBase` or
    is tagged with :class:`pytato.tags.ImplStored`.
    """
    return (isinstance(expr, InputArgumentBase)
            or bool(expr.tags_of_type(ImplStored)))


def _is_trivial_slice(dim: ShapeComponent, slice_: NormalizedSlice) -> bool:
    """
    Returns *True* only if *slice_* indexes an entire axis of shape *dim* with
    a step of 1.
    """
    return (slice_.step == 1
            and are_shape_components_equal(slice_.start, 0)
            and are_shape_components_equal(slice_.stop, dim)
            )


def _take_along_axis(ary: Array, iaxis: int, idxs: IndexStackT) -> Array:
    """
    Returns an indexed version of *ary* with *iaxis*-th axis indexed with *idxs*.
    """
    # {{{ compose the slices

    composed_slice: Union[Array, slice] = slice(0, ary.shape[iaxis], 1)

    for idx in idxs[::-1]:
        if isinstance(composed_slice, slice):
            if isinstance(idx, NormalizedSlice):
                new_start = (composed_slice.start
                                + composed_slice.step * idx.start)
                if composed_slice.step > 0:
                    new_stop = (composed_slice.start
                                + composed_slice.step * (idx.stop-1)
                                + 1)
                else:
                    new_stop = (composed_slice.start
                                + composed_slice.step * (idx.stop+1)
                                - 1)

                new_step = composed_slice.step * idx.step

                composed_slice = slice(new_start, new_stop, new_step)
            else:
                assert isinstance(idx, Array)
                if composed_slice.step > 0:
                    if (composed_slice.step == 1
                            and are_shape_components_equal(composed_slice.start, 0)):
                        # minor optimization to emit cleaner DAGs when possible
                        composed_slice = idx
                    else:
                        composed_slice = (composed_slice.step * idx
                                          + composed_slice.start)
                else:
                    composed_slice = ((composed_slice.stop - 1)
                                      + composed_slice.step * idx)
        else:
            assert isinstance(composed_slice, Array)
            if isinstance(idx, NormalizedSlice):
                composed_slice = composed_slice[slice(idx.start, idx.stop, idx.step)]
            else:
                assert isinstance(idx, Array)
                composed_slice = composed_slice[idx]

    # }}}

    if (isinstance(composed_slice, slice)
        and _is_trivial_slice(ary.shape[iaxis],
                              NormalizedSlice(composed_slice.start,
                                              composed_slice.stop,
                                              composed_slice.step))):
        return ary
    else:
        return ary[(slice(None), )*iaxis + (composed_slice, )]


@dataclass(frozen=True)
class _BindingAxisGetterAcc:
    r"""
    Return type of :class:`_BindingAxisGetter` recording how a particular axis
    is indexed in a :class:`pytato.array.IndexLambda` for a particular binding.
    """


@dataclass(frozen=True)
class _InvariantAxis(_BindingAxisGetterAcc):
    r"""
    Records that the array :attr:`_BindingAxisGetter.bnd_name`\ 's access in a
    :class:`~pytato.scalar_expr.ScalarExpression` is invariant along the
    :attr:`_BindingAxisGetter.iout_axis` axis.
    """


@dataclass(frozen=True)
class _BindingNotAccessed(_BindingAxisGetterAcc):
    """
    Records that the array, :attr:`_BindingAxisGetterAcc.bnd_name`, is not
    accessed in a :class:`~pytato.scalar_expr.ScalarExpression`.
    """
    pass


@dataclass(frozen=True)
class _SingleAxisDependentAccess(_BindingAxisGetterAcc):
    """
    Records that the array's *iaxis*-th index is dependent only a single output
    axis of an :class:`~pytato.array.IndexLambda`.
    """
    iaxis: int


@dataclass(frozen=True)
class _IllegalAxisAccess(_BindingAxisGetterAcc):
    """
    Records that the access :attr:`_BindingAxisGetter.iout_axis` does not allow
    reindexing without modifying :class:`pytato.array.IndexLambda.expr`.
    """


class _BindingAxisGetter(scalar_expr.CombineMapper):
    """
    Mapper that returns how the binding named :attr:`bnd_name` is dependent on
    the index :attr:`iout_axis`.
    """
    def __init__(self, iout_axis: int, bnd_name: str):
        self.iout_axis = iout_axis
        self.bnd_name = bnd_name
        super().__init__()

    def combine(self,
                values: Iterable[_BindingAxisGetterAcc]) -> _BindingAxisGetterAcc:

        values = list(values)  # avoid running into generators
        if any(isinstance(val, _IllegalAxisAccess) for val in values):
            return _IllegalAxisAccess()

        axis_dependent_values = {val
                                 for val in values
                                 if isinstance(val, _SingleAxisDependentAccess)}
        invariant_axis_values = {val
                                 for val in values
                                 if isinstance(val, _InvariantAxis)}

        if len(invariant_axis_values | axis_dependent_values) == 0:
            return _BindingNotAccessed()
        elif len(invariant_axis_values | axis_dependent_values) == 1:
            combined_value, = invariant_axis_values | axis_dependent_values
            return combined_value
        else:
            return _IllegalAxisAccess()

    def map_subscript(self, expr: prim.Subscript) -> _BindingAxisGetterAcc:
        from pytato.scalar_expr import get_dependencies

        if expr.aggregate == prim.Variable(self.bnd_name):
            if f"_{self.iout_axis}" not in get_dependencies(expr.index_tuple):
                return _InvariantAxis()

            values = []
            for i_idx, idx in enumerate(expr.index_tuple):
                if get_dependencies(idx) == frozenset([f"_{self.iout_axis}"]):
                    values.append(_SingleAxisDependentAccess(i_idx))
                else:
                    values.append(self.rec(idx))
            return self.combine(values)
        else:
            return self.combine([self.rec(idx)
                                 for idx in expr.index_tuple
                                 if self.bnd_name in get_dependencies(idx)])

    def map_variable(self, expr: prim.Variable) -> _BindingAxisGetterAcc:
        if expr.name == f"_{self.iout_axis}":
            return _IllegalAxisAccess()

        return _BindingNotAccessed()

    def map_constant(self,
                     expr: scalar_expr.ScalarExpression) -> _BindingAxisGetterAcc:
        return _BindingNotAccessed()

    map_nan = map_constant


def _get_iaxis_in_binding(expr: scalar_expr.ScalarExpression,
                          iaxis: int,
                          bnd_name: str) -> _BindingAxisGetterAcc:
    mapper = _BindingAxisGetter(iaxis, bnd_name)
    # type-ignore-reason: pymbolic mapper types are imprecise.
    return mapper(expr)  # type: ignore[no-any-return]


class _LegallyAxisReorderingFinder(CachedMapper[FrozenSet[int]]):
    """
    Maps a :class:`pytato.array` to it's set of axes along which indirections
    can be propagated. We use the following rules to get the legally
    reorderable axes of an array:

    - All axes of a materialized array are reorderable.
    - The i-th axis of an :class:`~pytato.array.IndexLambda` is reorderable
      only if all its bindings either do not index using the i-th index, OR,
      every binding has a unique axis which indexes using the i-th index and
      that axis in binding is reorderable.

    These rules legally allow propagating indirections that are applied to an
    index lambda's to its bindings axes without altering the index lambda's
    scalar expression.
    """
    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> Any:
        raise ValueError("_LegallyAxisReorderingFinder is a valid operation"
                         " only for arrays")

    def _map_materialized(self, expr: Array) -> FrozenSet[int]:
        return frozenset(range(expr.ndim))

    map_placeholder = _map_materialized
    map_data_wrapper = _map_materialized

    def _map_index_lambda_like(self, expr: IndexLambda) -> FrozenSet[int]:
        if _is_materialized(expr):
            return self._map_materialized(expr)

        from pytato.transform.lower_to_index_lambda import to_index_lambda
        idx_lambda = to_index_lambda(expr)
        legal_orderings: Set[int] = set()
        rec_bindings = {name: self.rec(bnd)
                        for name, bnd in idx_lambda.bindings.items()}

        for idim in range(idx_lambda.ndim):
            bnd_name_to_iaxis = {name: _get_iaxis_in_binding(idx_lambda.expr,
                                                             idim,
                                                             name)
                                 for name in idx_lambda.bindings}
            is_reordering_idim_legal = all(
                ((isinstance(ibnd_axis, _SingleAxisDependentAccess)
                  and ibnd_axis.iaxis in rec_bindings[name])
                 or isinstance(ibnd_axis, (_InvariantAxis, _BindingNotAccessed)))
                for name, ibnd_axis in bnd_name_to_iaxis.items()
            )
            if is_reordering_idim_legal:
                legal_orderings.add(idim)

        return frozenset(legal_orderings)

    map_index_lambda = _map_index_lambda_like
    map_stack = _map_index_lambda_like
    map_concatenate = _map_index_lambda_like
    map_einsum = _map_index_lambda_like
    map_roll = _map_index_lambda_like
    map_basic_index = _map_index_lambda_like
    map_reshape = _map_index_lambda_like
    map_axis_permutation = _map_index_lambda_like
    map_basic_index = _map_index_lambda_like

    # {{{ advanced indexing nodes are special -> requires additional checks
    # on the indexers such as single-axis reordering

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> FrozenSet[int]:
        if _is_materialized(expr):
            return self._map_materialized(expr)

        from pytato.utils import (get_shape_after_broadcasting,
                                  partition)
        legal_orderings: Set[int] = set()
        array_legal_orderings = self.rec(expr.array)

        i_adv_indices, i_basic_indices = partition(lambda idx: isinstance(
                                                                expr.indices[idx],
                                                                NormalizedSlice),
                                                   range(len(expr.indices)))

        pre_basic_indices, post_basic_indices = partition(
            lambda idx: idx > i_adv_indices[0],
            i_basic_indices,
        )
        ary_indices = tuple(i_idx
                            for i_idx, idx in enumerate(expr.indices)
                            if isinstance(idx, Array))

        assert all(i_adv_indices[0] > i_basic_idx
                   for i_basic_idx in pre_basic_indices)
        assert all(i_adv_indices[-1] < i_basic_idx
                   for i_basic_idx in post_basic_indices)

        adv_idx_shape = get_shape_after_broadcasting([expr.indices[i_idx]
                                                      for i_idx in i_adv_indices])

        iout_axis = 0

        for i_idx in pre_basic_indices:
            if i_idx in array_legal_orderings:
                legal_orderings.add(iout_axis)
            iout_axis += 1

        if len(adv_idx_shape) != 1 or len(ary_indices) != 1:
            # cannot reorder these axes
            iout_axis += len(adv_idx_shape)
        else:
            if ary_indices[0] in array_legal_orderings:
                legal_orderings.add(iout_axis)
            iout_axis += 1

        for i_idx in post_basic_indices:
            if i_idx in array_legal_orderings:
                legal_orderings.add(iout_axis)
            iout_axis += 1

        return frozenset(legal_orderings)

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes
                                          ) -> FrozenSet[int]:
        if _is_materialized(expr):
            return self._map_materialized(expr)

        from pytato.utils import (get_shape_after_broadcasting,
                                  partition)
        legal_orderings: Set[int] = set()
        array_legal_orderings = self.rec(expr.array)

        i_adv_indices, i_basic_indices = partition(lambda idx: isinstance(
                                                                expr.indices[idx],
                                                                NormalizedSlice),
                                                   range(len(expr.indices)))

        ary_indices = tuple(i_idx
                            for i_idx, idx in enumerate(expr.indices)
                            if isinstance(idx, Array))

        adv_idx_shape = get_shape_after_broadcasting([expr.indices[i_idx]
                                                      for i_idx in i_adv_indices])

        iout_axis = 0

        if len(adv_idx_shape) != 1 or len(ary_indices) != 1:
            # cannot reorder these axes
            iout_axis += len(adv_idx_shape)
        else:
            if ary_indices[0] in array_legal_orderings:
                legal_orderings.add(iout_axis)
            iout_axis += 1

        for i_idx in i_basic_indices:
            if i_idx in array_legal_orderings:
                legal_orderings.add(iout_axis)
            iout_axis += 1

        return frozenset(legal_orderings)

    # }}}


def _get_iout_axis_to_binding_axis(
        expr: Array) -> Map[Array, Map[int, int]]:
    from pytato.transform.lower_to_index_lambda import to_index_lambda
    idx_lambda = to_index_lambda(expr)

    result: Dict[Array, Dict[int, int]] = {
        bnd: {}
        for bnd in idx_lambda.bindings.values()
    }

    for name, bnd in idx_lambda.bindings.items():
        for iout_axis in range(expr.ndim):
            ibnd_axis = _get_iaxis_in_binding(idx_lambda.expr, iout_axis, name)
            if isinstance(ibnd_axis, _SingleAxisDependentAccess):
                result[bnd][iout_axis] = ibnd_axis.iaxis

    return Map({k: Map(v) for k, v in result.items()})


class _IndirectionPusher(Mapper):
    """
    Mapper to move the indirections in the array expression closer to the
    materialized nodes of the graph. The logic implemented in the mapper
    complements the implementation in :class:`_LegallyAxisReorderingFinder`.
    """

    def __init__(self) -> None:
        self.get_reordarable_axes = _LegallyAxisReorderingFinder()
        self._cache: Dict[Tuple[ArrayOrNames, Map[int, IndexStackT]],
                          ArrayOrNames] = {}
        super().__init__()

    def rec(self,  # type: ignore[override]
            expr: MappedT,
            index_stacks: Map[int, IndexStackT]) -> MappedT:
        key = (expr, index_stacks)
        try:
            # type-ignore-reason: parametric mapping types aren't a thing in 'typing'
            return self._cache[key]  # type: ignore[return-value]
        except KeyError:
            result = Mapper.rec(self, expr, index_stacks)
            self._cache[key] = result
            return result  # type: ignore[no-any-return]

    def __call__(self,  # type: ignore[override]
                 expr: MappedT,
                 index_stacks: Map[int, IndexStackT]) -> MappedT:
        return self.rec(expr, index_stacks)

    def _map_materialized(self,
                          expr: Array,
                          index_stacks: Map[int, IndexStackT]) -> Array:
        result = expr
        for iaxis, idxs in index_stacks.items():
            result = _take_along_axis(result, iaxis, idxs)

        return result

    def map_dict_of_named_arrays(self,
                                 expr: DictOfNamedArrays,
                                 *args: Any, **kwargs: Any) -> Any:
        raise ValueError("_IndirectionPusher cannot map AbstractResultOfNamedArrays")

    map_placeholder = _map_materialized
    map_data_wrapper = _map_materialized

    def map_index_lambda(self,
                         expr: IndexLambda,
                         index_stacks: Map[int, IndexStackT]
                         ) -> Array:
        if _is_materialized(expr):
            # do not propagate the indexings to the bindings.
            expr = IndexLambda(expr.expr,
                               expr.shape,
                               expr.dtype,
                               Map({name: self.rec(bnd, Map())
                                    for name, bnd in expr.bindings.items()}),
                               expr.var_to_reduction_descr,
                               tags=expr.tags,
                               axes=expr.axes,
                               )
            return self._map_materialized(expr, index_stacks)

        iout_axis_to_bnd_axis = _get_iout_axis_to_binding_axis(expr)

        new_bindings = {
            name: self.rec(bnd,
                           Map({iout_axis_to_bnd_axis[bnd][iout_axis]: index_stack
                                for iout_axis, index_stack in index_stacks.items()
                                if iout_axis in iout_axis_to_bnd_axis[bnd]})
                           )
            for name, bnd in expr.bindings.items()
        }

        # {{{ compute the new shape after propagating the indirections

        iaxis_to_new_shape: Dict[int, ShapeComponent] = {
            idim: axis_len
            for idim, axis_len in enumerate(expr.shape)
            if idim not in index_stacks
        }

        for iaxis in index_stacks:
            for bnd_name in expr.bindings:
                ibnd_axis = _get_iaxis_in_binding(expr.expr, iaxis, bnd_name)
                assert isinstance(ibnd_axis, _SingleAxisDependentAccess)
                new_bnd_axis_len = new_bindings[bnd_name].shape[ibnd_axis.iaxis]
                assert are_shape_components_equal(
                    iaxis_to_new_shape.setdefault(iaxis, new_bnd_axis_len),
                    new_bnd_axis_len)

        # }}}

        assert len(iaxis_to_new_shape) == expr.ndim
        return IndexLambda(expr=expr.expr,
                           bindings=Map(new_bindings),
                           dtype=expr.dtype,
                           shape=tuple(iaxis_to_new_shape[idim]
                                       for idim in range(expr.ndim)),
                           var_to_reduction_descr=expr.var_to_reduction_descr,
                           tags=expr.tags,
                           axes=expr.axes)

    def map_basic_index(self,
                        expr: BasicIndex,
                        index_stacks: Map[int, IndexStackT]) -> Array:

        if _is_materialized(expr):
            # do not propagate the indexings to the indexee.
            expr = BasicIndex(self.rec(expr.array, Map()),
                              expr.indices,
                              tags=expr.tags,
                              axes=expr.axes)
            return self._map_materialized(expr, index_stacks)

        iout_axis_to_iarray_axis: Dict[int, int] = {}
        iout_axis = 0

        for iarray_axis, idx in enumerate(expr.indices):
            if isinstance(idx, NormalizedSlice):
                iout_axis_to_iarray_axis[iout_axis] = iarray_axis
                iout_axis += 1

        assert iout_axis == expr.ndim
        # initialize from previous indexing operations
        array_index_stacks = {
            iout_axis_to_iarray_axis[iout_axis]: index_stack
            for iout_axis, index_stack in index_stacks.items()
        }

        # indices that cannot be propagated to expr.array
        unreordered_i_indices: List[int] = []

        for iarray_axis, idx in enumerate(expr.indices):
            if isinstance(idx, NormalizedSlice):
                if iarray_axis in self.get_reordarable_axes(expr.array):
                    array_index_stacks[iarray_axis] = (
                        array_index_stacks.get(iarray_axis, ()) + (idx,)
                    )
                else:
                    if not _is_trivial_slice(expr.array.shape[iarray_axis],
                                             idx):
                        unreordered_i_indices.append(iarray_axis)

        if unreordered_i_indices:
            raise NotImplementedError("Partially pushing the indexers is not yet"
                                      " implemented.")

        # FIXME: Think about metadata preservation??
        return self.rec(expr.array, Map(array_index_stacks))

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      index_stacks: Map[int, IndexStackT]
                                      ) -> Array:
        from pytato.utils import partition, get_shape_after_broadcasting
        if _is_materialized(expr):
            # do not propagate the indexings to the indexee.
            expr = AdvancedIndexInContiguousAxes(self.rec(expr.array, Map()),
                                                 expr.indices,
                                                 tags=expr.tags,
                                                 axes=expr.axes)
            return self._map_materialized(expr, index_stacks)

        array_index_stacks: Dict[int, IndexStackT] = {}
        unreorderable_axes: List[int] = []
        iout_axis = 0

        i_adv_indices, i_basic_indices = partition(lambda idx: isinstance(
                                                                expr.indices[idx],
                                                                NormalizedSlice),
                                                   range(len(expr.indices)))

        pre_basic_indices, post_basic_indices = partition(
            lambda idx: idx > i_adv_indices[0],
            i_basic_indices,
        )
        ary_indices = tuple(i_idx
                            for i_idx, idx in enumerate(expr.indices)
                            if isinstance(idx, Array))
        adv_idx_shape = get_shape_after_broadcasting([expr.indices[i_idx]
                                                      for i_idx in i_adv_indices])

        for iarray_axis in pre_basic_indices:
            if iarray_axis in self.get_reordarable_axes(expr.array):
                # type-ignore-reason: mypy cannot infer iarray_axis corresponds
                # to a slice
                array_index_stacks[iarray_axis] = (
                    index_stacks.get(iout_axis, ())
                    + (expr.indices[iarray_axis],))  # type: ignore[operator]
            else:
                # type-ignore-reason: mypy cannot infer iarray_axis corresponds
                # to a slice
                if not _is_trivial_slice(
                        expr.array.shape[iarray_axis],
                        expr.indices[iarray_axis]  # type: ignore[arg-type]
                ):
                    unreorderable_axes.append(iarray_axis)
            iout_axis += 1

        # type-ignore-reason: mypy cannot infer ary_indices corresponds
        # to indirections
        if (len(ary_indices) == 1
                and (expr.indices[ary_indices[0]].ndim  # type: ignore[union-attr]
                     == 1)
                and ary_indices[0] in self.get_reordarable_axes(expr.array)):

            array_index_stacks[ary_indices[0]] = (
                index_stacks.get(iout_axis, ())
                + (expr.indices[ary_indices[0]],))  # type: ignore[operator]
            iout_axis += 1
        else:
            iout_axis += len(adv_idx_shape)
            unreorderable_axes.extend(ary_indices)

        for iarray_axis in post_basic_indices:
            if iarray_axis in self.get_reordarable_axes(expr.array):
                # type-ignore-reason: mypy cannot infer post_basic_indices
                # corresponds to slices
                array_index_stacks[iarray_axis] = (
                    index_stacks.get(iout_axis, ())
                    + (expr.indices[iarray_axis],))  # type: ignore[operator]
            else:
                if not _is_trivial_slice(
                        expr.array.shape[iarray_axis],
                        expr.indices[iarray_axis]):  # type: ignore[arg-type]
                    unreorderable_axes.append(iarray_axis)
            iout_axis += 1

        assert iout_axis == expr.ndim

        if unreorderable_axes:
            raise NotImplementedError("Partially pushing the indexers is not yet"
                                      " implemented.")

        return self.rec(expr.array, Map(array_index_stacks))

    def map_non_contiguous_advanced_index(
            self,
            expr: AdvancedIndexInNoncontiguousAxes,
            index_stacks: Map[int, IndexStackT]) -> Array:
        from pytato.utils import partition, get_shape_after_broadcasting
        if _is_materialized(expr):
            # do not propagate the indexings to the indexee.
            expr = AdvancedIndexInNoncontiguousAxes(self.rec(expr.array, Map()),
                                                    expr.indices,
                                                    tags=expr.tags,
                                                    axes=expr.axes)
            return self._map_materialized(expr, index_stacks)

        array_index_stacks: Dict[int, IndexStackT] = {}
        unreorderable_axes: List[int] = []
        iout_axis = 0

        i_adv_indices, i_basic_indices = partition(lambda idx: isinstance(
                                                                expr.indices[idx],
                                                                NormalizedSlice),
                                                   range(len(expr.indices)))

        ary_indices = tuple(i_idx
                            for i_idx, idx in enumerate(expr.indices)
                            if isinstance(idx, Array))
        adv_idx_shape = get_shape_after_broadcasting([expr.indices[i_idx]
                                                      for i_idx in i_adv_indices])

        if (len(ary_indices) == 1
                and (expr.indices[ary_indices[0]].ndim  # type: ignore[union-attr]
                     == 1)
                and ary_indices[0] in self.get_reordarable_axes(expr.array)):

            # type-ignore-reason: mypy cannot infer ary_indices correspond to
            # indirections
            array_index_stacks[ary_indices[0]] = (
                index_stacks.get(iout_axis, ())
                + (expr.indices[ary_indices[0]],))  # type: ignore[operator]
            iout_axis += 1
        else:
            iout_axis += len(adv_idx_shape)
            unreorderable_axes.extend(ary_indices)

        for iarray_axis in i_basic_indices:
            if iarray_axis in self.get_reordarable_axes(expr.array):
                # type-ignore-reason: mypy cannot infer ary_indices correspond to
                # slices
                array_index_stacks[iarray_axis] = (
                    index_stacks.get(iout_axis, ())
                    + (expr.indices[iarray_axis],))  # type: ignore[operator]
            else:
                if not _is_trivial_slice(
                        expr.array.shape[iarray_axis],
                        expr.indices[iarray_axis]):  # type: ignore[arg-type]
                    unreorderable_axes.append(iarray_axis)
            iout_axis += 1

        assert iout_axis == expr.ndim

        if unreorderable_axes:
            raise NotImplementedError("Partially pushing the indexers is not yet"
                                      " implemented.")

        return self.rec(expr.array, Map(array_index_stacks))

    def map_stack(self,
                  expr: Stack,
                  index_stacks: Map[int, IndexStackT]) -> Array:
        if _is_materialized(expr):
            # do not propagate the indexings to the bindings.
            expr = Stack(
                arrays=tuple(self.rec(ary, Map()) for ary in expr.arrays),
                axis=expr.axis,
                tags=expr.tags,
                axes=expr.axes,
            )
            return self._map_materialized(expr, index_stacks)

        iout_axis_to_bnd_axis = _get_iout_axis_to_binding_axis(expr)
        assert expr.axis not in index_stacks
        return Stack(
            arrays=tuple(
                self.rec(ary,
                         Map({iout_axis_to_bnd_axis[ary][iout_axis]: index_stack
                              for iout_axis, index_stack in index_stacks.items()}))
                for ary in expr.arrays),
            axis=expr.axis,
            tags=expr.tags,
            axes=expr.axes,
        )

    def map_concatenate(self,
                        expr: Concatenate,
                        index_stacks: Map[int, IndexStackT]) -> Array:
        if _is_materialized(expr):
            # do not propagate the indexings to the bindings.
            expr = Concatenate(
                arrays=tuple(self.rec(ary, Map()) for ary in expr.arrays),
                axis=expr.axis,
                tags=expr.tags,
                axes=expr.axes,
            )
            return self._map_materialized(expr, index_stacks)

        iout_axis_to_bnd_axis = _get_iout_axis_to_binding_axis(expr)
        assert expr.axis not in index_stacks
        return Concatenate(
            arrays=tuple(
                self.rec(ary,
                         Map({iout_axis_to_bnd_axis[ary][iout_axis]: index_stack
                              for iout_axis, index_stack in index_stacks.items()}))
                for ary in expr.arrays),
            axis=expr.axis,
            tags=expr.tags,
            axes=expr.axes,
        )

    def map_einsum(self,
                   expr: Einsum,
                   index_stacks: Map[int, IndexStackT]) -> Array:

        if _is_materialized(expr):
            # do not propagate the indexings to the bindings.
            expr = Einsum(
                expr.access_descriptors,
                args=tuple(self.rec(arg, Map()) for arg in expr.args),
                index_to_access_descr=expr.index_to_access_descr,
                redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                tags=expr.tags,
                axes=expr.axes,
            )
            return self._map_materialized(expr, index_stacks)

        iout_axis_to_bnd_axis = _get_iout_axis_to_binding_axis(expr)
        return Einsum(
            expr.access_descriptors,
            args=tuple(
                self.rec(arg,
                         Map({iout_axis_to_bnd_axis[arg][iout_axis]: index_stack
                              for iout_axis, index_stack in index_stacks.items()
                              if iout_axis in iout_axis_to_bnd_axis[arg]})
                         )
                for arg in expr.args),
            index_to_access_descr=expr.index_to_access_descr,
            redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
            tags=expr.tags,
            axes=expr.axes,
        )

    def map_roll(self,
                 expr: Roll,
                 index_stacks: Map[int, IndexStackT]) -> Array:
        if _is_materialized(expr):
            # do not propagate the indexings to the bindings.
            expr = Roll(
                self.rec(expr.array, Map()),
                expr.shift,
                expr.axis,
                tags=expr.tags,
                axes=expr.axes,)
            return self._map_materialized(expr, index_stacks)

        iout_axis_to_bnd_axis = _get_iout_axis_to_binding_axis(expr)
        return Roll(
            self.rec(expr.array,
                     Map({iout_axis_to_bnd_axis[expr.array][iout_axis]: index_stack
                          for iout_axis, index_stack in index_stacks.items()})
                     ),
            expr.shift,
            expr.axis,
            tags=expr.tags,
            axes=expr.axes,)

    def map_reshape(self,
                    expr: Reshape,
                    index_stacks: Map[int, IndexStackT]) -> Array:

        if _is_materialized(expr):
            # do not propagate the indexings to the bindings.
            expr = Reshape(
                self.rec(expr.array, Map()),
                expr.newshape,
                expr.order,
                tags=expr.tags,
                axes=expr.axes,)
            return self._map_materialized(expr, index_stacks)

        iout_axis_to_bnd_axis = _get_iout_axis_to_binding_axis(expr)
        return Reshape(
            self.rec(
                expr.array,
                Map({iout_axis_to_bnd_axis[expr.array][iout_axis]: index_stack
                                for iout_axis, index_stack in index_stacks.items()})
            ),
            expr.newshape,
            expr.order,
            tags=expr.tags,
            axes=expr.axes,)

    def map_axis_permutation(self,
                             expr: AxisPermutation,
                             index_stacks: Map[int, IndexStackT]) -> Array:

        if _is_materialized(expr):
            # do not propagate the indexings to the bindings.
            expr = AxisPermutation(
                self.rec(expr.array, Map()),
                expr.axis_permutation,
                tags=expr.tags,
                axes=expr.axes,
            )
            return self._map_materialized(expr, index_stacks)

        iout_axis_to_bnd_axis = _get_iout_axis_to_binding_axis(expr)
        return AxisPermutation(
            self.rec(
                expr.array,
                Map({iout_axis_to_bnd_axis[expr.array][iout_axis]: index_stack
                     for iout_axis, index_stack in index_stacks.items()})
            ),
            expr.axis_permutation,
            tags=expr.tags,
            axes=expr.axes,
        )


def push_axis_indirections_towards_materialized_nodes(expr: MappedT
                                                      ) -> MappedT:
    """
    Returns a copy of *expr* with the indirections propagated closer to the
    materialized nodes. We propagate an indirections only if the indirection in
    an :class:`~pytato.array.AdvancedIndexInContiguousAxes` or
    :class:`~pytato.array.AdvancedIndexInNoncontiguousAxes` is an indirection
    over a single axis.
    """
    mapper = _IndirectionPusher()

    return mapper(expr, Map())


def _get_unbroadcasted_axis_in_indirections(
        expr: AdvancedIndexInContiguousAxes) -> Optional[Mapping[int, int]]:
    """
    Returns a mapping from the index of an indirection to its *only*
    unbroadcasted axis as required by the logic. Returns *None* if no such
    mapping exists.
    """
    from pytato.utils import partition, get_shape_after_broadcasting
    adv_indices, _ = partition(lambda i: isinstance(expr.indices[i],
                                                    NormalizedSlice),
                               range(expr.array.ndim))
    i_ary_indices = [i_idx
                     for i_idx, idx in enumerate(expr.indices)
                     if isinstance(idx, Array)]

    adv_idx_shape = get_shape_after_broadcasting([expr.indices[i_idx]
                                                  for i_idx in adv_indices])

    if len(adv_idx_shape) != len(i_ary_indices):
        return None

    i_adv_out_axis_to_candidate_i_arys: Dict[int, Set[int]] = {
        idim: set()
        for idim, _ in enumerate(adv_idx_shape)
    }

    for i_ary_idx in i_ary_indices:
        ary = expr.indices[i_ary_idx]
        assert isinstance(ary, Array)
        for iadv_out_axis, i_ary_axis in zip(range(len(adv_idx_shape)-1, -1, -1),
                                             range(ary.ndim-1, -1, -1)):
            if are_shape_components_equal(adv_idx_shape[iadv_out_axis],
                                          ary.shape[i_ary_axis]):
                i_adv_out_axis_to_candidate_i_arys[iadv_out_axis].add(i_ary_idx)

    from itertools import permutations
    # FIXME: O(expr.ndim!) complexity, typically ndim <= 4 so this should be fine.
    for guess_i_adv_out_axis_to_i_ary in permutations(range(len(i_ary_indices))):
        if all(i_ary in i_adv_out_axis_to_candidate_i_arys[i_adv_out]
               for i_adv_out, i_ary in enumerate(guess_i_adv_out_axis_to_i_ary)):
            # TODO: Return the mapping here...
            i_ary_to_unbroadcasted_axis: Dict[int, int] = {}
            for guess_i_adv_out_axis, i_ary_idx in enumerate(
                    guess_i_adv_out_axis_to_i_ary):
                ary = expr.indices[i_ary_idx]
                assert isinstance(ary, Array)
                iunbroadcasted_axis, = [
                    i_ary_axis
                    for i_adv_out_axis, i_ary_axis in zip(
                            range(len(adv_idx_shape)-1, -1, -1),
                            range(ary.ndim-1, -1, -1))
                    if i_adv_out_axis == guess_i_adv_out_axis
                ]
                i_ary_to_unbroadcasted_axis[i_ary_idx] = iunbroadcasted_axis

            return Map(i_ary_to_unbroadcasted_axis)

    return None


class MultiAxisIndirectionsDecoupler(CopyMapper):
    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes
                                      ) -> Array:
        i_ary_idx_to_unbroadcasted_axis = _get_unbroadcasted_axis_in_indirections(
            expr)

        if i_ary_idx_to_unbroadcasted_axis is not None:
            from pytato.utils import partition
            i_adv_indices, _ = partition(lambda idx: isinstance(expr.indices[idx],
                                                                NormalizedSlice),
                                         range(len(expr.indices)))

            result = self.rec(expr.array)

            for iaxis, idx in enumerate(expr.indices):
                if isinstance(idx, Array):
                    from pytato.array import squeeze
                    axes_to_squeeze = [
                        idim
                        for idim in range(expr
                                          .indices[iaxis]  # type: ignore[union-attr]
                                          .ndim)
                        if idim != i_ary_idx_to_unbroadcasted_axis[iaxis]]
                    if axes_to_squeeze:
                        idx = squeeze(idx, axis=axes_to_squeeze)
                if not (isinstance(idx, NormalizedSlice)
                        and _is_trivial_slice(expr.array.shape[iaxis], idx)):
                    result = result[
                        (slice(None),) * iaxis + (idx, )]  # type: ignore[operator]

            return result
        else:
            return super().map_contiguous_advanced_index(expr)


def decouple_multi_axis_indirections_into_single_axis_indirections(
        expr: MappedT) -> MappedT:
    """
    Returns a copy of *expr* with multiple indirections in an
    :class:`~pytato.array.AdvancedIndexInContiguousAxes` decoupled as a
    composition of indexing nodes with single-axis indirections.

    .. note::

        This is a dependency preserving transformation. If a decoupling an
        advanced indexing node is not legal, we leave the node unmodified.
    """
    mapper = MultiAxisIndirectionsDecoupler()
    return mapper(expr)


# {{{ fold indirection constants

class _ConstantIndirectionArrayCollector(CombineMapper[FrozenSet[Array]]):
    def __init__(self) -> None:
        from pytato.transform import InputGatherer
        super().__init__()
        self.get_inputs = InputGatherer()

    def combine(self, *args: FrozenSet[Array]) -> FrozenSet[Array]:
        from functools import reduce
        return reduce(frozenset.union, args, frozenset())

    def _map_input_base(self, expr: InputArgumentBase) -> FrozenSet[Array]:
        return frozenset()

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def _map_index_base(self, expr: IndexBase) -> FrozenSet[Array]:
        rec_results: List[FrozenSet[Array]] = []

        rec_results.append(self.rec(expr.array))

        for idx in expr.indices:
            if isinstance(idx, Array):
                if any(isinstance(inp, Placeholder)
                       for inp in self.get_inputs(idx)):
                    rec_results.append(self.rec(idx))
                else:
                    rec_results.append(frozenset([idx]))

        return self.combine(*rec_results)


def fold_constant_indirections(
        expr: MappedT,
        evaluator: Callable[[DictOfNamedArrays], Mapping[str, DataWrapper]]
) -> MappedT:
    """
    Returns a copy of *expr* with constant indirection expressions frozen.

    :arg evaluator: A callable that takes in a
        :class:`~pytato.array.DictOfNamedArrays` and returns a mapping from the
        name of every named array to it's corresponding evaluated array as an
        instance of :class:`~pytato.array.DataWrapper`.
    """
    from pytools import UniqueNameGenerator
    from pytato.array import make_dict_of_named_arrays
    import collections.abc as abc
    from pytato.transform import map_and_copy

    vng = UniqueNameGenerator()
    arys_to_evaluate = _ConstantIndirectionArrayCollector()(expr)
    dict_of_named_arrays = make_dict_of_named_arrays(
        {vng("_pt_folded_cnst"): ary for ary in arys_to_evaluate}
    )
    del arys_to_evaluate
    evaluated_arys = evaluator(dict_of_named_arrays)

    if not isinstance(evaluated_arys, abc.Mapping):
        raise TypeError("evaluator did not return a mapping")

    if set(evaluated_arys.keys()) != set(dict_of_named_arrays.keys()):
        raise ValueError("evaluator must return a mapping with "
                         f"the keys: '{set(dict_of_named_arrays.keys())}'.")

    for key, ary in evaluated_arys.items():
        if not isinstance(ary, DataWrapper):
            raise TypeError(f"evaluated array for '{key}' not a DataWrapper")

    before_to_after_subst = {
        dict_of_named_arrays._data[name]: evaluated_ary
        for name, evaluated_ary in evaluated_arys.items()
    }

    def _replace_with_folded_constants(subexpr: ArrayOrNames) -> ArrayOrNames:
        if isinstance(subexpr, Array):
            return before_to_after_subst.get(subexpr, subexpr)
        else:
            return subexpr

    return map_and_copy(expr, _replace_with_folded_constants)

# }}}

# vim: foldmethod=marker
