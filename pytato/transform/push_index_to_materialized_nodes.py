from __future__ import annotations

from pytato.scalar_expr import INT_CLASSES


__copyright__ = """Copyright (C) 2026 Kaushik Kulkarni
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

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from immutabledict import immutabledict

from pytato.array import (
    Array,
    DictOfNamedArrays,
    IndexBase,
    IndexExpr,
    InputArgumentBase,
    NormalizedSlice,
    expand_dims,
    transpose,
)
from pytato.tags import ImplStored
from pytato.transform import ArrayOrNames, CacheKeyT, TransformMapperWithExtraArgs
from pytato.utils import are_shape_components_equal, get_shape_after_broadcasting


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from pymbolic.typing import Integer

    from pytato.array import (
        AxisPermutation,
        Concatenate,
        Einsum,
        IndexBase,
        IndexLambda,
        Reshape,
        Roll,
        ShapeComponent,
        SizeParam,
        Stack,
    )
    from pytato.function import NamedCallResult
    from pytato.loopy import LoopyCall, LoopyCallResult
    from pytato.transform import ArrayOrNamesTc

IndexesT: TypeAlias = tuple[IndexExpr, ...]


def _is_trivial_slice(dim: ShapeComponent, slice_: IndexExpr) -> bool:
    return (
        isinstance(slice_, NormalizedSlice)
        and slice_.start == 0
        and slice_.step == 1
        and slice_.stop == dim
    )


def _is_materialized(x: Array) -> bool:
    return isinstance(x, InputArgumentBase) or len(x.tags_of_type(ImplStored)) != 0


def get_indexing_kind(
    indexes: IndexesT,
) -> Literal["basic", "contiguous_advanced", "non_contiguous_advanced"]:
    from pytato.utils import partition

    i_adv_indices, i_basic_indices = partition(
        lambda idx: isinstance(indexes[idx], NormalizedSlice), range(len(indexes))
    )
    if all(
        isinstance(indexes[i_adv_idx], INT_CLASSES)
        or (isinstance(indexes[i_adv_idx], Array) and indexes[i_adv_idx].shape == ())
        for i_adv_idx in i_adv_indices
    ):
        return "basic"
    elif any(
        i_adv_indices[0] < i_basic_idx < i_adv_indices[-1]
        for i_basic_idx in i_basic_indices
    ):
        return "non_contiguous_advanced"
    else:
        return "contiguous_advanced"


def _get_indices_ndim(indices: Sequence[IndexExpr]) -> int:
    return len([idx for idx in indices if isinstance(idx, NormalizedSlice)]) + len(
        get_shape_after_broadcasting([idx for idx in indices if isinstance(idx, Array)])
    )


def _partition_into_adv_and_basic_indices(
    indexes: IndexesT,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    from pytato.utils import partition

    i_adv_indices, i_basic_indices = partition(
        lambda idx: isinstance(indexes[idx], NormalizedSlice), range(len(indexes))
    )
    return tuple(i_adv_indices), tuple(i_basic_indices)


@dataclass(frozen=True)
class AxisAccess(ABC):  # noqa: B024
    ...


@dataclass(frozen=True)
class PointAccess(AxisAccess):
    point: Integer


@dataclass(frozen=True)
class SliceAccess(AxisAccess):
    tgt_axis: int
    slice_: NormalizedSlice

    def __post_init__(self) -> None:
        assert isinstance(self.tgt_axis, int)
        assert isinstance(self.slice_, NormalizedSlice)


@dataclass(frozen=True)
class ArrayIndexAccess(AxisAccess):
    tgt_axes: tuple[int, ...]
    ary: Array

    def __post_init__(self) -> None:
        assert isinstance(self.ary, Array)
        assert self.ary.ndim == len(self.tgt_axes)


def _get_axis_accesses(indices: IndexesT) -> tuple[AxisAccess, ...]:
    accesses: list[AxisAccess] = []
    kind = get_indexing_kind(indices)
    i_adv_indices, i_basic_indices = _partition_into_adv_and_basic_indices(indices)
    adv_ndim = _get_indices_ndim([indices[i_idx] for i_idx in i_adv_indices])

    if kind == "basic":
        tgt_axis = 0
        for idx in indices:
            if isinstance(idx, INT_CLASSES):
                accesses.append(PointAccess(idx))
            else:
                assert isinstance(idx, NormalizedSlice)
                accesses.append(SliceAccess(tgt_axis, idx))
                tgt_axis += 1
    elif kind == "contiguous_advanced":
        tgt_axis = 0
        for i_idx in range(i_adv_indices[0]):
            idx = indices[i_idx]
            assert isinstance(idx, NormalizedSlice)
            accesses.append(SliceAccess(tgt_axis, idx))
            tgt_axis += 1

        for i_idx in i_adv_indices:
            idx = indices[i_idx]
            if isinstance(idx, INT_CLASSES):
                accesses.append(PointAccess(idx))
            else:
                assert isinstance(idx, Array)
                tgt_axes = tuple(
                    range(tgt_axis + adv_ndim - idx.ndim, tgt_axis + adv_ndim)
                )
                accesses.append(ArrayIndexAccess(tgt_axes, idx))
        tgt_axis += adv_ndim
        for i_idx in range(i_adv_indices[-1] + 1, len(indices)):
            idx = indices[i_idx]
            assert isinstance(idx, NormalizedSlice)
            accesses.append(SliceAccess(tgt_axis, idx))
            tgt_axis += 1
    else:
        assert kind == "non_contiguous_advanced"
        for i_idx in i_adv_indices:
            idx = indices[i_idx]
            if isinstance(idx, INT_CLASSES):
                accesses.append(PointAccess(idx))
            else:
                assert isinstance(idx, Array)
                tgt_axes = tuple(range(adv_ndim - idx.ndim, adv_ndim))
                accesses.append(ArrayIndexAccess(idx, tgt_axes))
        tgt_axis = adv_ndim
        for i_idx in i_basic_indices:
            idx = indices[i_idx]
            assert isinstance(idx, NormalizedSlice)
            accesses.append(SliceAccess(tgt_axis, idx))
            tgt_axis += 1

    assert len(accesses) == len(indices)
    return tuple(accesses)


def _get_array_tgt_axes(accesses: tuple[AxisAccess, ...]) -> tuple[int, ...]:
    from functools import reduce

    return tuple(
        sorted(
            reduce(
                frozenset.union,
                (
                    frozenset(access.tgt_axes)
                    for access in accesses
                    if isinstance(access, ArrayIndexAccess)
                ),
                frozenset(),
            )
        )
    )


def _permute_tgt_axes(
    accesses: tuple[AxisAccess, ...], perm: tuple[int, ...]
) -> tuple[AxisAccess, ...]:
    new_accesses: list[AxisAccess] = []
    for access in accesses:
        if isinstance(access, PointAccess):
            new_accesses.add(access)
        elif isinstance(access, SliceAccess):
            new_accesses.append(SliceAccess(perm[access.tgt_axis], access.slice_))
        else:
            assert isinstance(access, AxisAccess)
            new_tgt_axes = tuple(perm[tgt_axis] for tgt_axis in access.tgt_axes)
            new_accesses.append(ArrayIndexAccess(new_tgt_axes, access.ary))

    assert len(new_accesses) == len(accesses)
    return tuple(new_accesses)


def _get_resulting_target_axes(accesses: Sequence[AxisAccess]) -> tuple[int, ...]:
    from pytato.utils import partition

    i_adv_indices, i_basic_indices = partition(
        lambda idx: isinstance(accesses[idx], SliceAccess), range(len(accesses))
    )

    if all(
        isinstance(accesses[i_adv_idx], PointAccess)
        or (
            isinstance(accesses[i_adv_idx], ArrayIndexAccess)
            and accesses[i_adv_idx].ary.ndim == 0
        )
        for i_adv_idx in i_adv_indices
    ):
        kind = "basic"
    elif any(
        i_adv_indices[0] < i_basic_idx < i_adv_indices[-1]
        for i_basic_idx in i_basic_indices
    ):
        kind = "non_contiguous_advanced"
    else:
        kind = "contiguous_advanced"

    if kind == "basic":
        return tuple(access.tgt_axis for access in accesses)
    elif kind == "contiguous_advanced":
        pre_basic_tgts = [
            access.tgt_axis
            for access in accesses[: i_adv_indices[0]]
            if isinstance(access, SliceAccess)
        ]
        advanced_tgts = max(
            [
                accesses[i_adv_idx]
                for i_adv_idx in i_adv_indices
                if isinstance(accesses[i_adv_idx], AxisAccess)
            ],
            key=lambda x: x.ary.ndim,
        ).tgt_axes

        assert all(
            accesses[i_adv_idx].tgt_axes
            == advanced_tgts[-len(accesses[i_adv_idx].ary.ndim)]
            for i_adv_idx in i_adv_indices
            if isinstance(accesses[i_adv_idx], AxisAccess)
        )
        post_basic_tgts = [
            access.tgt_axis
            for access in accesses[i_adv_indices[-1] + 1 :]
            if isinstance(access, SliceAccess)
        ]
        return tuple(pre_basic_tgts + advanced_tgts + post_basic_tgts)
    else:
        assert kind == "non_contiguous_advanced"
        basic_tgts = [
            accesses[i_adv_idx].tgt_axis
            for i_adv_idx in i_adv_indices
            if isinstance(accesses[i_adv_idx], SliceAccess)
        ]
        advanced_tgts = max(
            [
                accesses[i_adv_idx]
                for i_adv_idx in i_adv_indices
                if isinstance(accesses[i_adv_idx], AxisAccess)
            ],
            key=lambda x: x.ary.ndim,
        ).tgt_axes

        assert all(
            accesses[i_adv_idx].tgt_axes
            == advanced_tgts[-len(accesses[i_adv_idx].ary.ndim)]
            for i_adv_idx in i_adv_indices
            if isinstance(accesses[i_adv_idx], AxisAccess)
        )
        return tuple(advanced_tgts + basic_tgts)


def _compose_indices(
    *, inner_indices: IndexesT, outer_indices: IndexesT
) -> tuple[tuple[int, ...], IndexesT]:
    """
    Returns ``(axis_perm, indices)`` such that ``pt.transpose(x[indices],
    axis_perm) == x[inner_indices][outer_indices]``.
    """
    accesses_to_x = _get_axis_accesses(inner_indices)
    accesses_to_x_inner = _get_axis_accesses(outer_indices)
    adv_idx_shape_of_innner_indices = get_shape_after_broadcasting(
        [idx for idx in inner_indices if isinstance(idx, Array)]
    )
    adv_tgt_axes_of_x = _get_array_tgt_axes(accesses_to_x)
    adv_tgt_axes_of_x_inner = _get_array_tgt_axes(accesses_to_x_inner)
    additional_adv_tgt_axes = tuple(
        access_to_x_inner.tgt_axis
        for iaxis, access_to_x_inner in enumerate(outer_indices)
        if (isinstance(access_to_x_inner, SliceAccess) and iaxis in adv_tgt_axes_of_x)
    )

    if not additional_adv_tgt_axes:
        axis_perm = tuple(range(_get_indices_ndim(outer_indices)))
    else:
        if get_indexing_kind(outer_indices) in {"contiguous_advanced", "basic"}:
            first_adv_tgt_axis_of_x_inner = (
                adv_tgt_axes_of_x_inner[0]
                if adv_tgt_axes_of_x_inner
                else additional_adv_tgt_axes[0]
            )
            last_adv_tgt_axis_of_x_inner = (
                adv_tgt_axes_of_x_inner[-1]
                if adv_tgt_axes_of_x_inner
                else additional_adv_tgt_axes[0]
            )

            additional_adv_tgt_axes_before = [
                tgt_axis
                for tgt_axis in additional_adv_tgt_axes
                if tgt_axis < first_adv_tgt_axis_of_x_inner
            ]
            additional_adv_tgt_axes_after = [
                tgt_axis
                for tgt_axis in additional_adv_tgt_axes
                if tgt_axis > last_adv_tgt_axis_of_x_inner
            ]
            axis_perm_inv = (
                [
                    iaxis
                    for iaxis in range(first_adv_tgt_axis_of_x_inner)
                    if iaxis not in additional_adv_tgt_axes
                ]
                + additional_adv_tgt_axes_before
                + adv_tgt_axes_of_x_inner
                + additional_adv_tgt_axes_after
                + [
                    iaxis
                    for iaxis in range(
                        last_adv_tgt_axis_of_x_inner + 1,
                        _get_indices_ndim(outer_indices),
                    )
                    if iaxis not in additional_adv_tgt_axes
                ]
            )

            axis_perm = tuple(np.argsort(axis_perm_inv).tolist())
            del axis_perm_inv
        else:
            assert get_indexing_kind(outer_indices) == "non_contiguous_advanced"
            axis_perm_inv = (
                adv_tgt_axes_of_x_inner
                + additional_adv_tgt_axes
                + tuple(
                    i
                    for i in range(
                        len(adv_tgt_axes_of_x_inner), _get_indices_ndim(outer_indices)
                    )
                    if i not in additional_adv_tgt_axes
                )
            )

            axis_perm = tuple(np.argsort(axis_perm_inv).tolist())
            del axis_perm_inv

    accesses_to_x_inner = _permute_tgt_axes(
        accesses_to_x_inner, tuple(np.argsort(axis_perm).tolist())
    )
    adv_tgt_axes_of_x_inner = _get_array_tgt_axes(accesses_to_x_inner)

    composed_indices_to_x: list[IndexExpr] = []

    for access_to_x in accesses_to_x:
        if isinstance(access_to_x, PointAccess):
            composed_indices_to_x.append(access_to_x.point)
        elif isinstance(access_to_x, SliceAccess):
            inner_slice = access_to_x.slice_
            access_to_x_inner = accesses_to_x_inner[access_to_x.tgt_axis]
            if isinstance(access_to_x_inner, PointAccess):
                composed_indices_to_x.append(
                    inner_slice.start + inner_slice.step * access_to_x_inner.point
                )
            elif isinstance(access_to_x_inner, SliceAccess):
                outer_slice = access_to_x_inner.slice_
                composed_indices_to_x.append(
                    NormalizedSlice(
                        inner_slice.start + inner_slice.step * outer_slice.start,
                        inner_slice.start + inner_slice.step * outer_slice.end,
                        inner_slice.step * outer_slice.step,
                    )
                )
            else:
                assert isinstance(access_to_x_inner, ArrayIndexAccess)
                composed_indices_to_x.append(
                    inner_slice.start + inner_slice.step * access_to_x_inner.ary,
                )
        else:
            assert isinstance(access_to_x, ArrayIndexAccess)
            tgt_axes = access_to_x.tgt_axes
            resulting_tgt_axes = _get_resulting_target_axes(
                [
                    (
                        accesses_to_x_inner[tgt_axis]
                        if are_shape_components_equal(
                            adv_idx_shape_of_innner_indices[
                                -access_to_x.ary.ndim + src_axis
                            ],
                            access_to_x.ary.shape[src_axis],
                        )
                        else PointAccess(0)
                    )
                    for src_axis, tgt_axis in enumerate(tgt_axes)
                ]
            )
            new_indices = []
            for src_axis, tgt_axis in enumerate(tgt_axes):
                access_to_x_inner = accesses_to_x_inner[tgt_axis]
                if isinstance(access_to_x_inner, PointAccess):
                    new_indices.append(access_to_x_inner.point)
                elif isinstance(access_to_x_inner, SliceAccess):
                    slice_ = access_to_x_inner.slice_
                    new_indices.append(slice(slice_.start, slice_.stop, slice_.step))
                else:
                    assert isinstance(access_to_x_inner, Array)
                    if not are_shape_components_equal(
                        adv_idx_shape_of_innner_indices[-access_to_x.ndim + src_axis],
                        access_to_x.ary.shape[src_axis],
                    ):
                        assert are_shape_components_equal(
                            access_to_x.ary.shape[src_axis], 1
                        )
                        new_indices.append(0)
                    else:
                        new_indices.append(access_to_x_inner.ary)
            axis_perm = tuple(np.argsort(resulting_tgt_axes).tolist())
            dims_to_expand = (
                [
                    iaxis
                    for iaxis, tgt_axis in enumerate(
                        adv_tgt_axes_of_x_inner[
                            adv_tgt_axes_of_x_inner.index(min(resulting_tgt_axes)) :
                        ]
                    )
                    if tgt_axis not in resulting_tgt_axes
                ]
                if adv_tgt_axes_of_x_inner
                else []
            )

            new_ary = access_to_x.ary
            if not all(
                (not isinstance(axis_len, slice))
                or _is_trivial_slice(
                    axis_len, NormalizedSlice(idx.start, idx.stop, idx.step)
                )
                for idx, axis_len in zip(
                    new_indices, access_to_x.ary.shape, strict=True
                )
            ):
                new_ary = new_ary[tuple(new_indices)]

            assert len(axis_perm) == new_ary.ndim

            if axis_perm != tuple(range(new_ary.ndim)):
                new_ary = transpose(new_ary, axis_perm)

            if dims_to_expand:
                new_ary = expand_dims(new_ary, dims_to_expand)
            composed_indices_to_x.append(new_ary)

    assert len(composed_indices_to_x) == len(inner_indices)

    return axis_perm, tuple(composed_indices_to_x)


class IndexPusher(TransformMapperWithExtraArgs[[IndexesT]]):
    def get_cache_key(self, expr: ArrayOrNames, indexes: IndexesT) -> CacheKeyT:
        return (expr, indexes)

    def rec_ary(self, expr: Array, indexes: IndexesT) -> Array:
        result = self.rec(expr, indexes)
        assert isinstance(result, Array)
        return result

    def rec_w_passthru_indices(self, expr: Array) -> Array:
        """
        Recurse over *expr* with all indices being the trivial slices, i.e.
        ``slice()``.
        """
        indexes = tuple(NormalizedSlice(0, axis_len, 1) for axis_len in expr.shape)
        return self.rec_ary(expr, indexes)

    def _eagerly_index(self, expr: Array, indexes: IndexesT) -> Array:
        """
        Returns *expr* index with *indexes*, i.e. returns ``expr[indices]``.


        .. note::

            No further attempts at propagating
            *indexes* to the predecessors of *expr* is done.
        """
        assert expr.ndim == len(indexes)
        if all(
            _is_trivial_slice(dim, idx)
            for dim, idx in zip(expr.shape, indexes, strict=True)
        ):
            return expr
        else:
            # <https://github.com/inducer/pytato/issues/633> explains why this
            # is needed.
            new_indexes = tuple(
                (
                    slice(idx.start, idx.stop, idx.step)
                    if isinstance(idx, NormalizedSlice)
                    else idx
                )
                for idx in indexes
            )
            return expr[new_indexes]

    map_placeholder = _eagerly_index
    map_data_wrapper = _eagerly_index

    def _map_index_base(self, expr: IndexBase, indexes: IndexesT) -> Array:
        assert len(indexes) == expr.ndim
        if _is_materialized(expr):
            return self._eagerly_index(self.rec_ary(expr.array, expr.indices), indexes)
        else:
            axis_perm, new_indices = _compose_indices(
                outer_indices=indexes, inner_indices=expr.indices
            )
            reced_ary = self.rec_ary(expr.array, new_indices)
            if axis_perm == tuple(range(len(axis_perm))):
                return reced_ary
            else:
                return transpose(reced_ary, axis_perm)

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_size_param(self, expr: SizeParam, indexes: IndexesT) -> Array:
        raise NotImplementedError

    def map_index_lambda(self, expr: IndexLambda, indexes: IndexesT) -> Array:
        if _is_materialized(expr):
            new_expr = expr.copy(
                bindings=immutabledict(
                    {
                        name: self.rec_w_passthru_indices(bnd)
                        for name, bnd in expr.bindings.items()
                    }
                )
            )
            return self._eagerly_index(new_expr, indexes)
        else:
            raise NotImplementedError

    def map_concatenate(self, expr: Concatenate, indexes: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(
                    arrays=tuple(
                        self.rec_w_passthru_indices(ary) for ary in expr.arrays
                    )
                ),
                indexes,
            )
        else:
            # TODO: Skipping for now. (Should be doable with some swizzling
            # of expr.arrays, don't see any need for it now.)
            raise NotImplementedError

    def map_stack(self, expr: Stack, indexes: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(
                    arrays=tuple(
                        self.rec_w_passthru_indices(ary) for ary in expr.arrays
                    )
                ),
                indexes,
            )
        else:
            # TODO: Skipping for now. (Should be doable with some swizzling
            # of expr.arrays, don't see any need for it now.)
            raise NotImplementedError

    def map_roll(self, expr: Roll, indexes: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(array=self.rec_w_passthru_indices(expr.array)), indexes
            )
        else:
            # TODO: Skipping for now. (Should be doable with a modulo operation,
            # don't see any need for it now.)
            raise NotImplementedError

    def map_axis_permutation(self, expr: AxisPermutation, indexes: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(array=self.rec_w_passthru_indices(expr.array)), indexes
            )
        else:
            # TODO: Skipping for now. The permutation axes have to be changed to
            # play with non-contiguous advanced indexing. Not needed for now.
            raise NotImplementedError

    def map_reshape(self, expr: Reshape, indexes: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(array=self.rec_w_passthru_indices(expr.array)), indexes
            )
        else:
            # TODO: Skipping for now. For certain cases, like the expand_dims
            # reshape, propagating the indices should be doable. Not needed for now.
            raise NotImplementedError

    def map_einsum(self, expr: Einsum, indexes: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(
                    args=tuple(self.rec_w_passthru_indices(arg) for arg in expr.args)
                ),
                indexes,
            )
        else:
            # TODO: Skipping for now. Involves transmitting the indices to the
            # operands as well as changing the subscript itself.
            raise NotImplementedError

    def map_loopy_call(self, expr: LoopyCall, indexes: IndexesT) -> LoopyCall:
        if indexes != ():
            raise ValueError("map_loopy_call must be called with outer indexes = ().")
        return expr.copy(
            bindings=immutabledict(
                {
                    name: self.rec_w_passthru_indices(bnd)
                    for name, bnd in expr.bindings.items()
                }
            )
        )

    def map_loopy_call_result(self, expr: LoopyCallResult, indexes: IndexesT) -> Array:
        new_expr = self.rec(expr, ())[expr.name]
        return self._eagerly_index(new_expr, indexes)

    def map_named_call_result(self, expr: NamedCallResult, indexes: IndexesT) -> Array:
        # TODO: Maybe we should propagate indexes to the function definition itself?
        raise NotImplementedError(
            "NamedCall results currently not supported in"
            " push_index_to_materialized_nodes."
        )

    def map_dict_of_named_arrays(
        self, expr: DictOfNamedArrays, indexes: IndexesT
    ) -> DictOfNamedArrays:
        if indexes != ():
            raise ValueError(
                "map_dict_of_named_arrays must be called with outer indexes = ()."
            )
        from pytato.array import make_dict_of_named_arrays

        return make_dict_of_named_arrays(
            {
                name: self.rec_w_passthru_indices(subexpr)
                for name, subexpr in expr._data.items()
            }
        )


def push_index_to_materialized_nodes(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    mapper = IndexPusher()
    if isinstance(expr, Array):
        return mapper(
            expr, tuple(NormalizedSlice(0, axis_len, 1) for axis_len in expr.shape)
        )
    else:
        return mapper(expr, ())
