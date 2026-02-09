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
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from constantdict import constantdict
from typing_extensions import override

from pytato.array import (
    Array,
    DataWrapper,
    DictOfNamedArrays,
    IndexBase,
    IndexExpr,
    InputArgumentBase,
    NormalizedSlice,
    Placeholder,
    ShapeComponent,
    ShapeType,
    expand_dims,
    transpose,
    zeros,
)
from pytato.loopy import LoopyCall
from pytato.raising import (
    BinaryOp,
    BinaryOpType,
    BroadcastOp,
    C99CallOp,
    LogicalNotOp,
    ReduceOp,
    WhereOp,
    ZerosLikeOp,
    index_lambda_to_high_level_op,
)
from pytato.tags import ImplStored
from pytato.transform import ArrayOrNames, CacheKeyT, TransformMapperWithExtraArgs
from pytato.utils import are_shape_components_equal, get_shape_after_broadcasting


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TypeAlias

    from pymbolic import Scalar
    from pymbolic.typing import Integer

    from pytato.array import (
        ArrayOrScalar,
        AxisPermutation,
        Concatenate,
        Einsum,
        IndexBase,
        IndexLambda,
        Reshape,
        Roll,
        SizeParam,
        Stack,
    )
    from pytato.function import NamedCallResult
    from pytato.loopy import LoopyCallResult
    from pytato.transform import ArrayOrNamesTc

IndexesT: TypeAlias = tuple[IndexExpr, ...]


def _lower_binary_op_hlo(hlo: BinaryOp) -> Array:
    """
    Returns a :class:`pytato.Array` corresponding to a binary operation
    high-level op.
    """
    from pytato.array import (
        equal,
        greater,
        greater_equal,
        less,
        less_equal,
        logical_and,
        logical_or,
        not_equal,
    )
    assert isinstance(hlo.x1, Array) or isinstance(hlo.x2, Array)
    # Note: We have a bunch of "pyright reportOperatorIssue" below, it does not
    # respect the above runtime guard that at least one of x1 and x2 are of type
    # Array.

    match hlo.binary_op:
        case BinaryOpType.ADD:
            return cast("Array", hlo.x1 + hlo.x2)
        case BinaryOpType.SUB:
            return cast("Array", hlo.x1 - hlo.x2)  # pyright: ignore[reportOperatorIssue]
        case BinaryOpType.MULT:
            return cast("Array", hlo.x1 * hlo.x2)
        case BinaryOpType.LOGICAL_OR:
            return cast("Array", logical_or(hlo.x1, hlo.x2))
        case BinaryOpType.LOGICAL_AND:
            return cast("Array", logical_and(hlo.x1, hlo.x2))
        case BinaryOpType.BITWISE_OR:
            return cast("Array", hlo.x1 | hlo.x2)  # pyright: ignore[reportOperatorIssue]
        case BinaryOpType.BITWISE_AND:
            return cast("Array", hlo.x1 & hlo.x2)  # pyright: ignore[reportOperatorIssue]
        case BinaryOpType.BITWISE_XOR:
            return cast("Array", hlo.x1 ^ hlo.x2)  # pyright: ignore[reportOperatorIssue]
        case BinaryOpType.TRUEDIV:
            return cast("Array", hlo.x1 / hlo.x2)
        case BinaryOpType.FLOORDIV:
            return cast("Array", hlo.x1 // hlo.x2)  # pyright: ignore[reportOperatorIssue]
        case BinaryOpType.POWER:
            return cast("Array", hlo.x1**hlo.x2)
        case BinaryOpType.MOD:
            return cast("Array", hlo.x1 % hlo.x2)  # pyright: ignore[reportOperatorIssue]
        case BinaryOpType.LESS:
            return cast("Array", less(hlo.x1, hlo.x2))
        case BinaryOpType.LESS_EQUAL:
            return cast("Array", less_equal(hlo.x1, hlo.x2))
        case BinaryOpType.GREATER:
            return cast("Array", greater(hlo.x1, hlo.x2))
        case BinaryOpType.GREATER_EQUAL:
            return cast("Array", greater_equal(hlo.x1, hlo.x2))
        case BinaryOpType.EQUAL:
            return cast("Array", equal(hlo.x1, hlo.x2))
        case BinaryOpType.NOT_EQUAL:
            return cast("Array", not_equal(hlo.x1, hlo.x2))


def _lower_call_op_hlo(hlo: C99CallOp) -> Array:
    """
    Returns a :class:`pytato.Array` corresponding to a function high level op.
    """

    import pytato.cmath
    from pytato.raising import PT_C99BINARY_FUNCS, PT_C99UNARY_FUNCS

    function = hlo.function
    if function in {"asin", "acos", "atan", "atan2"}:
        # these functions have different names on the numpy side vs on the C99
        # side.
        function = "arc" + function[1:]
    if hlo.function in PT_C99UNARY_FUNCS:
        unary_mathfn = cast(
            "Callable[[ArrayOrScalar], ArrayOrScalar]",
            getattr(pytato.cmath, function),
        )
        return cast("Array", unary_mathfn(hlo.args[0]))
    else:
        assert hlo.function in PT_C99BINARY_FUNCS
        binary_mathfn = cast(
            "Callable[[ArrayOrScalar, ArrayOrScalar], ArrayOrScalar]",
            getattr(pytato.cmath, function),
        )
        return cast("Array", binary_mathfn(hlo.args[0], hlo.args[1]))


def _is_trivial_slice(dim: ShapeComponent, slice_: IndexExpr) -> bool:
    """
    Returns *True* only if *slice_* represents the ``[:]`` index for an array's
    axis of length *dim*.
    """
    return (
        isinstance(slice_, NormalizedSlice)
        and slice_.start == 0
        and slice_.step == 1
        and slice_.stop == dim
    )


def _is_materialized(x: Array) -> bool:
    # TODO: Maybe in the later versions, think about LoopyCallResult, etc.
    return isinstance(x, InputArgumentBase) or len(x.tags_of_type(ImplStored)) != 0


def get_indexing_kind(
    indices: Sequence[IndexExpr],
) -> Literal["basic", "contiguous_advanced", "non_contiguous_advanced"]:
    """
    Returns what kind of :mod:`numpy` indexing does *indices* correspond to.
    """
    from pytato.utils import partition

    i_adv_indices, i_basic_indices = partition(
        lambda idx: isinstance(indices[idx], NormalizedSlice), range(len(indices))
    )
    if all(
        isinstance(idx := indices[i_adv_idx], INT_CLASSES)
        or (isinstance(idx, Array) and idx.ndim == 0)
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


def _partition_into_adv_and_basic_indices(
    indices: Sequence[IndexExpr],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Returns the tuple ``(ia, ib)``, such that, for every ``i`` in ``ib``,
    ``indices[i]`` is an instance of :class:`pytato.NormalizedSlice`, and, for
    every ``j`` in ``ia``, ``indices[j]`` is an instance of
    :class:`pytato.Array` or an integer.
    """
    from pytato.utils import partition

    i_adv_indices, i_basic_indices = partition(
        lambda idx: isinstance(indices[idx], NormalizedSlice), range(len(indices))
    )
    return tuple(i_adv_indices), tuple(i_basic_indices)


def _get_indices_shape(indices: Sequence[IndexExpr]) -> ShapeType:
    """
    Returns the shape of the array constructed by the :mod:`numpy` styled
    indexing: ``x[indices[0], indices[1], ...]``.
    """
    from pytato.utils import _normalized_slice_len

    kind = get_indexing_kind(indices)
    i_adv_indices, i_basic_indices = _partition_into_adv_and_basic_indices(indices)
    if kind == "basic":
        return tuple(
            _normalized_slice_len(cast("NormalizedSlice", indices[i_idx]))
            for i_idx in i_basic_indices
        )
    elif kind == "contiguous_advanced":
        return (
            tuple(
                _normalized_slice_len(cast("NormalizedSlice", indices[i_idx]))
                for i_idx in i_basic_indices
                if i_idx < i_adv_indices[0]
            )
            + get_shape_after_broadcasting(
                [cast("Array | Scalar", indices[i_idx]) for i_idx in i_adv_indices]
            )
            + tuple(
                _normalized_slice_len(cast("NormalizedSlice", indices[i_idx]))
                for i_idx in i_basic_indices
                if i_idx > i_adv_indices[-1]
            )
        )
    else:
        assert kind == "non_contiguous_advanced"
        return get_shape_after_broadcasting(
            [cast("Array | Scalar", indices[i_idx]) for i_idx in i_adv_indices]
        ) + tuple(
            _normalized_slice_len(cast("NormalizedSlice", indices[i_idx]))
            for i_idx in i_basic_indices
        )


def _get_indices_ndim(indices: Sequence[IndexExpr]) -> int:
    """
    Returns the dimensionality of the array ``x[indices[0], indices[1], ...]``.
    """
    return len([idx for idx in indices if isinstance(idx, NormalizedSlice)]) + len(
        get_shape_after_broadcasting([idx for idx in indices if isinstance(idx, Array)])
    )


@dataclass(frozen=True)
class AxisAccess(ABC):  # noqa: B024
    """
    Records an index access expression along an array's axis.
    """


@dataclass(frozen=True)
class PointAccess(AxisAccess):
    """
    Represents a single point access into an array's access.
    """
    point: Integer


@dataclass(frozen=True)
class SliceAccess(AxisAccess):
    """
    Records a slice access of an axis that targets the axis :attr:`tgt_axis` in
    the output.

    Consider X an array of shape ``(10, 10, 10, 10)`` which is indexed with
    non-contiguous advanced indices as ``Y[_0, _1] = X[idx1[_0], 3, _1, idx2[_0]]``. In
    this expression, the access to the 3rd axis from left is modeled as --
    ``SliceAccess(tgt_axis=1, slice_=NormalizedSlice(0, 10, 1))``.
    """
    tgt_axis: int
    slice_: NormalizedSlice

    def __post_init__(self) -> None:
        assert isinstance(self.tgt_axis, int)
        assert isinstance(self.slice_, NormalizedSlice)


@dataclass(frozen=True)
class ArrayIndexAccess(AxisAccess):
    """
    Records an array access of an axis that targets the axes :attr:`tgt_axes` in
    the output.

    Consider X an array of shape ``(10, 10, 10, 10)`` which is indexed with
    non-contiguous advanced indices as ``Y[_0, _1] = X[idx1[_0], 3, _1, idx2[_0,
    _1]]``. In this expression, the access to the 4th axis from left is modeled
    as -- ``ArrayIndexAccess(tgt_axes=(0, 1), ary=idx2)``.
    """
    tgt_axes: tuple[int, ...]
    ary: Array

    def __post_init__(self) -> None:
        assert isinstance(self.ary, Array)
        assert self.ary.ndim == len(self.tgt_axes)


def _get_axis_accesses(indices: IndexesT) -> tuple[AxisAccess, ...]:
    accesses: list[AxisAccess] = []
    kind = get_indexing_kind(indices)
    i_adv_indices, _ = _partition_into_adv_and_basic_indices(indices)
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
                accesses.append(
                    ArrayIndexAccess(
                        tuple(
                            range(tgt_axis + adv_ndim - idx.ndim, tgt_axis + adv_ndim)
                        ),
                        idx,
                    )
                )
        tgt_axis += adv_ndim
        for i_idx in range(i_adv_indices[-1] + 1, len(indices)):
            idx = indices[i_idx]
            assert isinstance(idx, NormalizedSlice)
            accesses.append(SliceAccess(tgt_axis, idx))
            tgt_axis += 1
    else:
        assert kind == "non_contiguous_advanced"
        slice_tgt_axis = adv_ndim
        for idx in indices:
            if isinstance(idx, INT_CLASSES):
                accesses.append(PointAccess(idx))
            elif isinstance(idx, NormalizedSlice):
                accesses.append(SliceAccess(slice_tgt_axis, idx))
                slice_tgt_axis += 1
            else:
                assert isinstance(idx, Array)
                accesses.append(
                    ArrayIndexAccess(tuple(range(adv_ndim - idx.ndim, adv_ndim)), idx)
                )

    assert len(accesses) == len(indices)
    return tuple(accesses)


def _get_array_tgt_axes(accesses: tuple[AxisAccess, ...]) -> tuple[int, ...]:
    """
    Returns all the target axes that contributions from an indirection access in
    *accesses*.
    """
    from functools import reduce

    return tuple(
        sorted(
            reduce(
                lambda x1, x2: x1 | x2,
                (
                    frozenset(access.tgt_axes)
                    for access in accesses
                    if isinstance(access, ArrayIndexAccess)
                ),
                cast("frozenset[int]", frozenset()),
            )
        )
    )


def _permute_tgt_axes(
    accesses: tuple[AxisAccess, ...], perm: tuple[int, ...]
) -> tuple[AxisAccess, ...]:
    """
    Returns a transformed version of *accesses* such that the target axes are
    permuted with the permutation *perm*.
    """
    new_accesses: list[AxisAccess] = []
    for access in accesses:
        if isinstance(access, PointAccess):
            new_accesses.append(access)
        elif isinstance(access, SliceAccess):
            new_accesses.append(SliceAccess(perm[access.tgt_axis], access.slice_))
        else:
            assert isinstance(access, ArrayIndexAccess)
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
            and cast("ArrayIndexAccess", accesses[i_adv_idx]).ary.ndim == 0
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
        return tuple(
            access.tgt_axis for access in accesses if isinstance(access, SliceAccess)
        )
    elif kind == "contiguous_advanced":
        pre_basic_tgts = tuple(
            access.tgt_axis
            for access in accesses[: i_adv_indices[0]]
            if isinstance(access, SliceAccess)
        )
        advanced_tgts = max(
            [
                cast("ArrayIndexAccess", accesses[i_adv_idx])
                for i_adv_idx in i_adv_indices
                if isinstance(accesses[i_adv_idx], ArrayIndexAccess)
            ],
            default=ArrayIndexAccess((), zeros(())),
            key=lambda x: x.ary.ndim,
        ).tgt_axes

        assert all(
            cast("ArrayIndexAccess", accesses[i_adv_idx]).tgt_axes
            == advanced_tgts[-cast("ArrayIndexAccess", accesses[i_adv_idx]).ary.ndim :]
            for i_adv_idx in i_adv_indices
            if isinstance(accesses[i_adv_idx], ArrayIndexAccess)
        )
        post_basic_tgts = tuple(
            access.tgt_axis
            for access in accesses[i_adv_indices[-1] + 1 :]
            if isinstance(access, SliceAccess)
        )
        return pre_basic_tgts + advanced_tgts + post_basic_tgts
    else:
        assert kind == "non_contiguous_advanced"
        basic_tgts = tuple(
            cast("SliceAccess", accesses[i_idx]).tgt_axis for i_idx in i_basic_indices
        )
        advanced_tgts = max(
            [
                cast("ArrayIndexAccess", accesses[i_adv_idx])
                for i_adv_idx in i_adv_indices
                if isinstance(accesses[i_adv_idx], ArrayIndexAccess)
            ],
            default=ArrayIndexAccess((), zeros(())),
            key=lambda x: x.ary.ndim,
        ).tgt_axes

        assert all(
            cast("ArrayIndexAccess", accesses[i_adv_idx]).tgt_axes
            == advanced_tgts[-cast("ArrayIndexAccess", accesses[i_adv_idx]).ary.ndim :]
            for i_adv_idx in i_adv_indices
            if isinstance(accesses[i_adv_idx], ArrayIndexAccess)
        )
        return advanced_tgts + basic_tgts


def _compose_axis_transposes(
    inner_perm: tuple[int, ...], outer_perm: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Returns ``axis_perm`` such that ``pt.transpose(x, axis_perm) ==
    pt.transpose(pt.transpose(x, inner_perm), outer_perm)``.
    """
    n = len(inner_perm)
    assert n == len(outer_perm)
    return tuple(inner_perm[outer_perm[i]] for i in range(n))


def _compose_indices(
    *, inner_indices: IndexesT, outer_indices: IndexesT
) -> tuple[tuple[int, ...], IndexesT]:
    """
    Returns ``(axis_perm, indices)`` such that ``pt.transpose(x[indices],
    axis_perm) == x[inner_indices][outer_indices]``.
    """
    accesses_to_x = _get_axis_accesses(inner_indices)
    accesses_to_x_inner = _get_axis_accesses(outer_indices)
    adv_idx_shape_of_inner_indices = get_shape_after_broadcasting(
        [idx for idx in inner_indices if isinstance(idx, Array)]
    )
    adv_tgt_axes_of_x = _get_array_tgt_axes(accesses_to_x)
    adv_tgt_axes_of_x_inner = _get_array_tgt_axes(accesses_to_x_inner)

    # {{{ identify additional advanced target axes.

    # These additional indirections in the composed indices force an axis
    # permutation on the output.

    additional_adv_tgt_axes = tuple(
        access_to_x_inner.tgt_axis
        for iaxis, access_to_x_inner in enumerate(accesses_to_x_inner)
        if (isinstance(access_to_x_inner, SliceAccess) and iaxis in adv_tgt_axes_of_x)
    )

    if not additional_adv_tgt_axes:
        axis_perm_inv = tuple(range(_get_indices_ndim(outer_indices)))
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

            additional_adv_tgt_axes_before = tuple(
                tgt_axis
                for tgt_axis in additional_adv_tgt_axes
                if tgt_axis < first_adv_tgt_axis_of_x_inner
            )
            additional_adv_tgt_axes_after = tuple(
                tgt_axis
                for tgt_axis in additional_adv_tgt_axes
                if tgt_axis > last_adv_tgt_axis_of_x_inner
            )
            axis_perm_inv = (
                tuple(
                    iaxis
                    for iaxis in range(first_adv_tgt_axis_of_x_inner)
                    if iaxis not in additional_adv_tgt_axes
                )
                + (() if adv_tgt_axes_of_x_inner else (first_adv_tgt_axis_of_x_inner,))
                + additional_adv_tgt_axes_before
                + adv_tgt_axes_of_x_inner
                + additional_adv_tgt_axes_after
                + tuple(
                    iaxis
                    for iaxis in range(
                        last_adv_tgt_axis_of_x_inner + 1,
                        _get_indices_ndim(outer_indices),
                    )
                    if iaxis not in additional_adv_tgt_axes
                )
            )
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

    # }}}

    axis_perm = tuple(cast("list[int]", np.argsort(axis_perm_inv).tolist()))
    accesses_to_x_inner = _permute_tgt_axes(
        accesses_to_x_inner, tuple(cast("list[int]", np.argsort(axis_perm).tolist()))
    )
    adv_tgt_axes_of_x_inner = tuple(
        sorted(
            axis_perm_inv[iaxis]
            for iaxis in (adv_tgt_axes_of_x_inner + additional_adv_tgt_axes)
        )
    )
    del axis_perm_inv

    composed_indices_to_x: list[IndexExpr] = []
    composed_accesses_to_x: list[AxisAccess] = []

    for access_to_x in accesses_to_x:
        if isinstance(access_to_x, PointAccess):
            composed_indices_to_x.append(access_to_x.point)
            composed_accesses_to_x.append(access_to_x)
        elif isinstance(access_to_x, SliceAccess):
            inner_slice = access_to_x.slice_
            access_to_x_inner = accesses_to_x_inner[access_to_x.tgt_axis]
            if isinstance(access_to_x_inner, PointAccess):
                composed_pt = (
                    inner_slice.start + inner_slice.step * access_to_x_inner.point
                )
                composed_indices_to_x.append(composed_pt)
                if isinstance(composed_pt, Array):
                    composed_accesses_to_x.append(ArrayIndexAccess((), composed_pt))
                else:
                    composed_accesses_to_x.append(PointAccess(composed_pt))
            elif isinstance(access_to_x_inner, SliceAccess):
                outer_slice = access_to_x_inner.slice_
                composed_slice = NormalizedSlice(
                    inner_slice.start + inner_slice.step * outer_slice.start,
                    inner_slice.start + inner_slice.step * outer_slice.stop,
                    inner_slice.step * outer_slice.step,
                )
                composed_indices_to_x.append(composed_slice)
                composed_accesses_to_x.append(
                    SliceAccess(access_to_x_inner.tgt_axis, composed_slice)
                )
            else:
                assert isinstance(access_to_x_inner, ArrayIndexAccess)
                idx_ary = access_to_x_inner.ary
                if inner_slice.step != 1:
                    idx_ary = inner_slice.step * idx_ary
                if not are_shape_components_equal(inner_slice.start, 0):
                    idx_ary = idx_ary + inner_slice.start
                dims_to_expand = tuple(
                    iaxis
                    for iaxis, tgt_axis in enumerate(
                        adv_tgt_axes_of_x_inner[
                            adv_tgt_axes_of_x_inner.index(
                                min(access_to_x_inner.tgt_axes)
                            ) :
                        ]
                    )
                    if tgt_axis not in access_to_x_inner.tgt_axes
                )
                if dims_to_expand:
                    idx_ary = expand_dims(idx_ary, dims_to_expand)

                composed_indices_to_x.append(idx_ary)
                composed_accesses_to_x.append(
                    ArrayIndexAccess(adv_tgt_axes_of_x_inner[-idx_ary.ndim :], idx_ary)
                )
        else:
            assert isinstance(access_to_x, ArrayIndexAccess)
            tgt_axes = access_to_x.tgt_axes
            resulting_tgt_axes = _get_resulting_target_axes(
                [
                    (
                        PointAccess(0)
                        if isinstance(accesses_to_x_inner[tgt_axis], ArrayIndexAccess)
                        and not are_shape_components_equal(
                            adv_idx_shape_of_inner_indices[
                                -access_to_x.ary.ndim + src_axis
                            ],
                            access_to_x.ary.shape[src_axis],
                        )
                        else accesses_to_x_inner[tgt_axis]
                    )
                    for src_axis, tgt_axis in enumerate(tgt_axes)
                ]
            )
            new_indices: list[Integer | slice | Array] = []
            for src_axis, tgt_axis in enumerate(tgt_axes):
                access_to_x_inner = accesses_to_x_inner[tgt_axis]
                if isinstance(access_to_x_inner, PointAccess):
                    new_indices.append(access_to_x_inner.point)
                elif isinstance(access_to_x_inner, SliceAccess):
                    slice_ = access_to_x_inner.slice_
                    if not are_shape_components_equal(
                        adv_idx_shape_of_inner_indices[
                            -access_to_x.ary.ndim + src_axis
                        ],
                        access_to_x.ary.shape[src_axis],
                    ):
                        assert are_shape_components_equal(
                            access_to_x.ary.shape[src_axis], 1
                        )
                        new_indices.append(slice(0, 1, 1))
                    else:
                        new_indices.append(
                            slice(slice_.start, slice_.stop, slice_.step)
                        )
                else:
                    assert isinstance(access_to_x_inner, ArrayIndexAccess)
                    if not are_shape_components_equal(
                        adv_idx_shape_of_inner_indices[
                            -access_to_x.ary.ndim + src_axis
                        ],
                        access_to_x.ary.shape[src_axis],
                    ):
                        assert are_shape_components_equal(
                            access_to_x.ary.shape[src_axis], 1
                        )
                        new_indices.append(0)
                    else:
                        new_indices.append(access_to_x_inner.ary)

            axis_perm_inner = tuple(
                cast("list[int]", np.argsort(resulting_tgt_axes).tolist())
            )
            dims_to_expand = (
                tuple(
                    iaxis
                    for iaxis, tgt_axis in enumerate(
                        adv_tgt_axes_of_x_inner[
                            adv_tgt_axes_of_x_inner.index(min(resulting_tgt_axes)) :
                        ]
                    )
                    if tgt_axis not in resulting_tgt_axes
                )
                if adv_tgt_axes_of_x_inner
                else ()
            )

            new_ary = access_to_x.ary
            if not all(
                isinstance(idx, slice)
                and _is_trivial_slice(
                    axis_len,
                    NormalizedSlice(
                        cast("ShapeComponent", idx.start),
                        cast("ShapeComponent", idx.stop),
                        cast("int", idx.step),
                    ),
                )
                for idx, axis_len in zip(
                    new_indices, access_to_x.ary.shape, strict=True
                )
            ):
                new_ary = new_ary[tuple(new_indices)]

            if axis_perm_inner != tuple(range(new_ary.ndim)):
                new_ary = transpose(new_ary, axis_perm_inner)

            if dims_to_expand:
                new_ary = expand_dims(new_ary, dims_to_expand)
            composed_indices_to_x.append(new_ary)
            composed_accesses_to_x.append(
                ArrayIndexAccess(
                    adv_tgt_axes_of_x_inner[-new_ary.ndim :],
                    new_ary,
                )
            )

    additional_axis_perm = tuple(
        cast(
            "list[int]",
            np.argsort(_get_resulting_target_axes(composed_accesses_to_x)).tolist(),
        )
    )

    assert len(composed_indices_to_x) == len(inner_indices)
    return _compose_axis_transposes(additional_axis_perm, axis_perm), tuple(
        composed_indices_to_x
    )


def _get_indices_for_broadcast(
    from_shape: ShapeType, to_shape: ShapeType, indices: IndexesT
) -> tuple[tuple[int, ...], tuple[int, ...], IndexesT]:
    """
    Returns ``(axis_perm, dims_to_expand, new_indices)`` such that
    ``pt.broadcast_to(x, to_shape)[indices] ==
    pt.transpose(pt.expand_dims(pt.broadcast_to(x[new_idxs],
    _get_indexing_shape(indices)), dims_to_expand), axis_perm)``, where
    ``x.shape == from_shape``.
    """
    assert len(to_shape) == len(indices)
    new_indices: list[IndexExpr] = []

    for from_axis_len, to_axis_len, idx in zip(
        from_shape,
        to_shape[-len(from_shape) :],
        indices[-len(from_shape) :],
        strict=True,
    ):
        if are_shape_components_equal(from_axis_len, to_axis_len):
            new_indices.append(idx)
        else:
            assert are_shape_components_equal(from_axis_len, 1)
            if isinstance(idx, NormalizedSlice):
                new_indices.append(NormalizedSlice(0, 1, 1))
            else:
                new_indices.append(0)

    i_adv_axes, i_basic_axes = _partition_into_adv_and_basic_indices(indices)
    i_new_adv_axes, i_new_basic_axes = _partition_into_adv_and_basic_indices(
        new_indices
    )
    adv_ndim = _get_indices_ndim([indices[iaxis] for iaxis in i_adv_axes])
    new_adv_ndim = _get_indices_ndim([new_indices[iaxis] for iaxis in i_new_adv_axes])

    if get_indexing_kind(indices) == "basic":
        assert get_indexing_kind(new_indices) == "basic"
        dims_to_expand: tuple[int, ...] = ()
        axis_perm = tuple(range(_get_indices_ndim(new_indices)))
    elif get_indexing_kind(indices) == "contiguous_advanced":
        if get_indexing_kind(new_indices) == "basic":
            n_new_pre_basic_axes = len(
                [iaxis for iaxis in i_new_basic_axes if iaxis < i_new_adv_axes[0]]
            )
            dims_to_expand = (
                tuple(
                    range(
                        n_new_pre_basic_axes,
                        n_new_pre_basic_axes + adv_ndim,
                    )
                )
                if n_new_pre_basic_axes
                else ()
            )
            axis_perm = tuple(range(_get_indices_ndim(new_indices)))
        else:
            assert get_indexing_kind(new_indices) == "contiguous_advanced"
            assert len(
                [iaxis for iaxis in i_basic_axes if iaxis > i_adv_axes[-1]]
            ) == len(
                [iaxis for iaxis in i_new_basic_axes if iaxis > i_new_adv_axes[-1]]
            )
            n_new_pre_basic_axes = len(
                [iaxis for iaxis in i_new_basic_axes if iaxis < i_new_adv_axes[0]]
            )

            dims_to_expand = tuple(
                range(
                    n_new_pre_basic_axes, n_new_pre_basic_axes + adv_ndim - new_adv_ndim
                )
            )
            axis_perm = tuple(range(_get_indices_ndim(new_indices)))
    else:
        assert get_indexing_kind(indices) == "non_contiguous_advanced"
        if get_indexing_kind(new_indices) == "basic":
            dims_to_expand = ()
            axis_perm = tuple(range(_get_indices_ndim(new_indices)))
        elif get_indexing_kind(new_indices) == "contiguous_advanced":
            n_missing_basic_axes = len(i_basic_axes) - len(i_new_basic_axes)
            dims_to_expand = tuple(
                range(
                    _get_indices_ndim(new_indices),
                    _get_indices_ndim(new_indices) + n_missing_basic_axes,
                )
            )
            n_new_pre_basic_axes = len(
                [iaxis for iaxis in i_new_basic_axes if iaxis < i_new_adv_axes[0]]
            )
            n_new_post_basic_axes = len(
                [iaxis for iaxis in i_new_basic_axes if iaxis > i_new_adv_axes[-1]]
            )
            axis_perm = (
                tuple(range(n_new_pre_basic_axes, n_new_pre_basic_axes + new_adv_ndim))
                + tuple(
                    range(
                        n_new_pre_basic_axes + new_adv_ndim + n_new_post_basic_axes,
                        n_missing_basic_axes
                        + n_new_pre_basic_axes
                        + new_adv_ndim
                        + n_new_post_basic_axes,
                    )
                )
                + tuple(range(n_new_pre_basic_axes))
                + tuple(
                    range(
                        n_new_pre_basic_axes + new_adv_ndim,
                        n_new_pre_basic_axes + new_adv_ndim + n_new_post_basic_axes,
                    )
                )
            )
        else:
            assert get_indexing_kind(new_indices) == "non_contiguous_advanced"
            n_missing_basic_axes = len(i_basic_axes) - len(i_new_basic_axes)
            dims_to_expand = tuple(
                range(new_adv_ndim, new_adv_ndim + n_missing_basic_axes)
            )
            axis_perm = tuple(
                range(_get_indices_ndim(new_indices) + n_missing_basic_axes)
            )

    return axis_perm, dims_to_expand, tuple(new_indices)


class IndexPusher(TransformMapperWithExtraArgs[[IndexesT]]):
    @override
    def get_cache_key(self, expr: ArrayOrNames, indices: IndexesT) -> CacheKeyT:
        return (expr, indices)

    def rec_ary(self, expr: Array, indices: IndexesT) -> Array:
        result = self.rec(expr, indices)
        assert isinstance(result, Array)
        return result

    def rec_w_passthru_indices(self, expr: Array) -> Array:
        """
        Recurse over *expr* with all indices being the trivial slices, i.e.
        ``slice()``.
        """
        indices = tuple(NormalizedSlice(0, axis_len, 1) for axis_len in expr.shape)
        return self.rec_ary(expr, indices)

    def rec_w_broadcast(
        self, expr: Array, to_shape: ShapeType, indices: IndexesT
    ) -> Array:
        axis_perm, new_dims, new_indices = _get_indices_for_broadcast(
            expr.shape, to_shape, indices
        )
        expr = self.rec_ary(expr, new_indices)
        expr = expand_dims(expr, new_dims) if new_dims else expr
        return (
            transpose(expr, axis_perm) if axis_perm != tuple(range(expr.ndim)) else expr
        )

    def _eagerly_index(self, expr: Array, indices: IndexesT) -> Array:
        """
        Returns *expr* index with *indices*, i.e. returns ``expr[indices]``.

        .. note::

            No further attempts at propagating
            *indices* to the predecessors of *expr* is done.
        """
        assert expr.ndim == len(indices)
        if all(
            _is_trivial_slice(dim, idx)
            for dim, idx in zip(expr.shape, indices, strict=True)
        ):
            return expr
        else:
            # <https://github.com/inducer/pytato/issues/633> explains why this
            # is needed.
            new_indices = tuple(
                (
                    slice(idx.start, idx.stop, idx.step)
                    if isinstance(idx, NormalizedSlice)
                    else idx
                )
                for idx in indices
            )
            return expr[new_indices]

    def map_placeholder(self, expr: Placeholder, indices: IndexesT) -> Array:
        return self._eagerly_index(expr, indices)

    def map_data_wrapper(self, expr: DataWrapper, indices: IndexesT) -> Array:
        return self._eagerly_index(expr, indices)

    def _map_index_base(self, expr: IndexBase, indices: IndexesT) -> Array:
        assert len(indices) == expr.ndim
        if _is_materialized(expr):
            return self._eagerly_index(self.rec_ary(expr.array, expr.indices), indices)
        else:
            axis_perm, new_indices = _compose_indices(
                outer_indices=indices, inner_indices=expr.indices
            )
            reced_ary = self.rec_ary(expr.array, new_indices)
            if axis_perm == tuple(range(len(axis_perm))):
                return reced_ary
            else:
                return transpose(reced_ary, axis_perm)

    def map_basic_index(self, expr: IndexBase, indices: IndexesT) -> Array:
        return self._map_index_base(expr, indices)

    def map_contiguous_advanced_index(
        self, expr: IndexBase, indices: IndexesT
    ) -> Array:
        return self._map_index_base(expr, indices)

    def map_non_contiguous_advanced_index(
        self, expr: IndexBase, indices: IndexesT
    ) -> Array:
        return self._map_index_base(expr, indices)

    def map_size_param(self, expr: SizeParam, indices: IndexesT) -> Array:
        raise NotImplementedError

    def map_index_lambda(self, expr: IndexLambda, indices: IndexesT) -> Array:
        if _is_materialized(expr):
            new_expr = expr.copy(
                bindings=constantdict(
                    {
                        name: self.rec_w_passthru_indices(bnd)
                        for name, bnd in expr.bindings.items()
                    }
                )
            )
            return self._eagerly_index(new_expr, indices)
        else:
            hlo = index_lambda_to_high_level_op(expr)

            if isinstance(hlo, BinaryOp):
                from dataclasses import replace

                x1, x2 = hlo.x1, hlo.x2
                if isinstance(x1, Array):
                    x1 = self.rec_w_broadcast(x1, expr.shape, indices)
                if isinstance(x2, Array):
                    x2 = self.rec_w_broadcast(x2, expr.shape, indices)
                return _lower_binary_op_hlo(replace(hlo, x1=x1, x2=x2))
            elif isinstance(hlo, BroadcastOp):
                from pytato.array import broadcast_to

                x = hlo.x
                if isinstance(x, Array):
                    x = self.rec_w_broadcast(x, expr.shape, indices)

                return broadcast_to(
                    x,
                    _get_indices_shape(indices),
                )
            elif isinstance(hlo, C99CallOp):
                new_args = tuple(
                    (
                        self.rec_w_broadcast(ary_arg, expr.shape, indices)
                        if isinstance(ary_arg, Array)
                        else ary_arg
                    )
                    for ary_arg in hlo.args
                )
                return _lower_call_op_hlo(
                    replace(  # pyright: ignore[reportUnboundVariable,reportUnknownArgumentType]
                        hlo, args=new_args
                    )
                )
            elif isinstance(hlo, LogicalNotOp):
                from pytato.array import logical_not

                return cast("Array", logical_not(self.rec_ary(hlo.x, indices)))
            elif isinstance(hlo, ReduceOp):
                # TODO: Skipping for now. (Should be doable after figure out the
                # appropriate transpose of the result.)
                raise NotImplementedError
            elif isinstance(hlo, WhereOp):
                from pytato.array import where

                (cond, then, else_) = [
                    (
                        self.rec_w_broadcast(ary_arg, expr.shape, indices)
                        if isinstance(ary_arg, Array)
                        else ary_arg
                    )
                    for ary_arg in [hlo.condition, hlo.then, hlo.else_]
                ]
                return cast("Array", where(cond, then, else_))
            elif isinstance(hlo, ZerosLikeOp):
                from pytato.cmath import zeros_like
                return cast("Array", zeros_like(
                    (
                        self.rec_ary(hlo.x, indices)
                    ),
                    expr.dtype,
                ))
            else:
                raise NotImplementedError(type(hlo))

    def map_concatenate(self, expr: Concatenate, indices: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(
                    arrays=tuple(
                        self.rec_w_passthru_indices(ary) for ary in expr.arrays
                    )
                ),
                indices,
            )
        else:
            # TODO: Skipping for now. (Should be doable with some swizzling
            # of expr.arrays, don't see any need for it now.)
            raise NotImplementedError

    def map_stack(self, expr: Stack, indices: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(
                    arrays=tuple(
                        self.rec_w_passthru_indices(ary) for ary in expr.arrays
                    )
                ),
                indices,
            )
        else:
            # TODO: Skipping for now. (Should be doable with some swizzling
            # of expr.arrays, don't see any need for it now.)
            raise NotImplementedError

    def map_roll(self, expr: Roll, indices: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(array=self.rec_w_passthru_indices(expr.array)), indices
            )
        else:
            # TODO: Skipping for now. (Should be doable with a modulo operation,
            # don't see any need for it now.)
            raise NotImplementedError

    def map_axis_permutation(self, expr: AxisPermutation, indices: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(array=self.rec_w_passthru_indices(expr.array)), indices
            )
        else:
            # TODO: Skipping for now. The permutation axes have to be changed to
            # play with non-contiguous advanced indexing. Not needed for now.
            raise NotImplementedError

    def map_reshape(self, expr: Reshape, indices: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(array=self.rec_w_passthru_indices(expr.array)), indices
            )
        else:
            # TODO: Skipping for now. For certain cases, like the expand_dims
            # reshape, propagating the indices should be doable. Not needed for now.
            if all(
                _is_trivial_slice(axis_len, idx)
                for axis_len, idx in zip(expr.shape, indices, strict=True)
            ):
                # handling a special case needed in grudge expressions
                result = expr.copy(array=self.rec_w_passthru_indices(expr.array))
                # do not create duplicates in the expression graph.
                return result if result != expr else expr
            raise NotImplementedError

    def map_einsum(self, expr: Einsum, indices: IndexesT) -> Array:
        if _is_materialized(expr):
            return self._eagerly_index(
                expr.copy(
                    args=tuple(self.rec_w_passthru_indices(arg) for arg in expr.args)
                ),
                indices,
            )
        else:
            # TODO: Skipping for now. Involves transmitting the indices to the
            # operands as well as changing the subscript itself.
            raise NotImplementedError

    def map_loopy_call(self, expr: LoopyCall, indices: IndexesT) -> LoopyCall:
        if indices != ():
            raise ValueError("map_loopy_call must be called with outer indices = ().")
        return expr.copy(
            bindings=constantdict(
                {
                    name: (
                        self.rec_w_passthru_indices(bnd)
                        if isinstance(bnd, Array)
                        else bnd
                    )
                    for name, bnd in expr.bindings.items()
                }
            )
        )

    def map_loopy_call_result(self, expr: LoopyCallResult, indices: IndexesT) -> Array:
        new_loopy_call = self.rec(expr._container, ())
        assert isinstance(new_loopy_call, LoopyCall)
        return self._eagerly_index(new_loopy_call[expr.name], indices)

    def map_named_call_result(self, expr: NamedCallResult, indices: IndexesT) -> Array:
        # TODO: Maybe we should propagate indices to the function definition itself?
        raise NotImplementedError(
            "NamedCall results currently not supported in"
            " push_index_to_materialized_nodes."
        )

    def map_dict_of_named_arrays(
        self, expr: DictOfNamedArrays, indices: IndexesT
    ) -> DictOfNamedArrays:
        if indices != ():
            raise ValueError(
                "map_dict_of_named_arrays must be called with outer indices = ()."
            )
        from pytato.array import make_dict_of_named_arrays

        return make_dict_of_named_arrays(
            {
                name: self.rec_w_passthru_indices(subexpr)
                for name, subexpr in expr._data.items()
            }
        )


def push_index_to_materialized_nodes(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    """
    Returns a transformed version of *expr* such that all indexing nodes in
    *expr* are propagated towards materialized arrays. Consequently, the
    transformed *expr* will contain expressions of the form
    ``x[idx1, idx2, ..., idxn]`` only if ``x`` is a materialized array.

    We consider an array as materialized when it is either an instance of
    :class:`pytato.InputArgumentBase` or has been tagged with
    :class:`pytato.tags.ImplStored`.
    """
    from pytato.transform import deduplicate
    mapper = IndexPusher()
    if isinstance(expr, Array):
        return deduplicate(mapper.rec_ary(
            expr, tuple(NormalizedSlice(0, axis_len, 1) for axis_len in expr.shape)
        ))
    else:
        assert isinstance(expr, DictOfNamedArrays)
        result_w_dups = mapper(expr, ())
        assert isinstance(result_w_dups, DictOfNamedArrays)
        return deduplicate(result_w_dups)
