"""
.. currentmodule:: pytato.transform.einsum_distributive_law


.. autoclass:: EinsumDistributiveLawDescriptor
.. autoclass:: DoNotDistribute
.. autoclass:: DoDistribute

.. autofunction:: apply_distributive_property_to_einsums
"""
from __future__ import annotations


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


import dataclasses
from typing import TYPE_CHECKING, cast

import numpy as np
from immutabledict import immutabledict

from pytato.array import (
    Array,
    AxesT,
    AxisPermutation,
    Concatenate,
    Einsum,
    EinsumAxisDescriptor,
    EinsumReductionAxis,
    IndexBase,
    IndexLambda,
    InputArgumentBase,
    ReductionDescriptor,
    Reshape,
    Roll,
    Stack,
)
from pytato.transform import (
    ArrayOrNames,
    CacheKeyT,
    MappedT,
    TransformMapperWithExtraArgs,
    _verify_is_array,
)
from pytato.utils import are_shapes_equal


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pytools.tag import Tag

    from pytato.raising import HighLevelOp


class EinsumDistributiveLawDescriptor:
    """
    Abstract-type that informs :func:`apply_distributive_property_to_einsums`
    how should the distributive law be applied along an einsum's operands.
    """


@dataclasses.dataclass(frozen=True)
class DoNotDistribute(EinsumDistributiveLawDescriptor):
    """
    Tells :func:`apply_distributive_property_to_einsums` to not apply
    distributive law along any operands of the :class:`~pytato.array.Einsum`.
    """


@dataclasses.dataclass(frozen=True)
class DoDistribute(EinsumDistributiveLawDescriptor):
    """
    Tells :func:`apply_distributive_property_to_einsums` to apply distributive
    law along the *ioperand*-th operands of the :class:`~pytato.array.Einsum`.
    """
    ioperand: int


@dataclasses.dataclass(frozen=True)
class _EinsumDistributiveLawMapperContext:
    access_descriptors: tuple[tuple[EinsumAxisDescriptor, ...], ...]
    surrounding_args: Mapping[int, Array]
    redn_axis_to_redn_descr: Mapping[EinsumReductionAxis,
                                 ReductionDescriptor]
    axes: AxesT = dataclasses.field(kw_only=True)
    tags: frozenset[Tag] = dataclasses.field(kw_only=True)

    def __post_init__(self) -> None:
        # {{{ check that exactly one of the args is missing

        assert len(self.surrounding_args) == (
            len(self.access_descriptors) - 1)
        assert all(0 <= iarg < len(self.access_descriptors)
                   for iarg in self.surrounding_args)

        # }}}


def _wrap_einsum_from_ctx(expr: Array,
                          ctx: _EinsumDistributiveLawMapperContext | None
                          ) -> Array:
    if ctx is None:
        return expr
    else:
        new_args = tuple(
            ctx.surrounding_args.get(iarg, expr)
            for iarg in range(len(ctx.access_descriptors))
        )
        return Einsum(
            ctx.access_descriptors,
            new_args,
            ctx.redn_axis_to_redn_descr,
            tags=ctx.tags,
            axes=ctx.axes
        )


def _can_hlo_be_distributed(hlo: HighLevelOp) -> bool:
    from pytato.raising import BinaryOp, BinaryOpType
    return (isinstance(hlo, BinaryOp)
            and ((hlo.binary_op in [BinaryOpType.MULT, BinaryOpType.TRUEDIV]
                  and (np.isscalar(hlo.x1) or np.isscalar(hlo.x2)))
                 or (hlo.binary_op in [BinaryOpType.ADD, BinaryOpType.SUB]
                     and isinstance(hlo.x1, Array)
                     and isinstance(hlo.x2, Array)
                     and are_shapes_equal(hlo.x1.shape,
                                          hlo.x2.shape))))


class EinsumDistributiveLawMapper(
                TransformMapperWithExtraArgs[
                    [_EinsumDistributiveLawMapperContext | None]]):
    """
    Primary mapper for :func:`apply_distributive_property_to_einsums`.
    """
    def __init__(self,
                 how_to_distribute: Callable[[Array],
                                             EinsumDistributiveLawDescriptor]
                 ) -> None:
        super().__init__()
        self.how_to_distribute = how_to_distribute

    def get_cache_key(
                self,
                expr: ArrayOrNames,
                ctx: _EinsumDistributiveLawMapperContext | None
            ) -> CacheKeyT:
        return (expr, ctx)

    def _map_input_base(self,
                        expr: InputArgumentBase,
                        ctx: _EinsumDistributiveLawMapperContext | None,
                        ) -> Array:
        return _wrap_einsum_from_ctx(expr, ctx)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_index_lambda(self,
                         expr: IndexLambda,
                         ctx: _EinsumDistributiveLawMapperContext | None,
                         ) -> Array:
        from pytato.raising import BinaryOp, BinaryOpType, index_lambda_to_high_level_op

        hlo = index_lambda_to_high_level_op(expr)

        if _can_hlo_be_distributed(hlo):
            assert isinstance(hlo, BinaryOp)
            # /!\ Warning: Loses metadata.

            if hlo.binary_op == BinaryOpType.ADD:
                assert (isinstance(hlo.x1, Array)
                        and isinstance(hlo.x2, Array)
                        and are_shapes_equal(hlo.x1.shape, hlo.x2.shape))
                # https://github.com/python/mypy/issues/16499
                return (  # type: ignore[no-any-return]
                    _verify_is_array(self.rec(hlo.x1, ctx))
                    + _verify_is_array(self.rec(hlo.x2, ctx)))
            elif hlo.binary_op == BinaryOpType.SUB:
                assert (isinstance(hlo.x1, Array)
                        and isinstance(hlo.x2, Array)
                        and are_shapes_equal(hlo.x1.shape, hlo.x2.shape))
                assert are_shapes_equal(hlo.x1.shape, hlo.x2.shape)
                # https://github.com/python/mypy/issues/16499
                return (  # type: ignore[no-any-return]
                    _verify_is_array(self.rec(hlo.x1, ctx))
                    - _verify_is_array(self.rec(hlo.x2, ctx)))
            elif hlo.binary_op == BinaryOpType.MULT:
                if isinstance(hlo.x1, Array) and not isinstance(hlo.x2, Array):
                    # https://github.com/python/mypy/issues/16499
                    return _verify_is_array(self.rec(hlo.x1, ctx)) * hlo.x2  # type: ignore[no-any-return]
                else:
                    assert isinstance(hlo.x2, Array) and not isinstance(hlo.x1, Array)
                    # https://github.com/python/mypy/issues/16499
                    return hlo.x1 * _verify_is_array(self.rec(hlo.x2, ctx))  # type: ignore[no-any-return]
            elif hlo.binary_op == BinaryOpType.TRUEDIV:
                if isinstance(hlo.x1, Array) and not isinstance(hlo.x2, Array):
                    # https://github.com/python/mypy/issues/16499
                    return _verify_is_array(self.rec(hlo.x1, ctx)) / hlo.x2  # type: ignore[no-any-return]
                else:
                    assert isinstance(hlo.x2, Array) and not isinstance(hlo.x1, Array)
                    # https://github.com/python/mypy/issues/16499
                    return hlo.x1 / _verify_is_array(self.rec(hlo.x2, ctx))  # type: ignore[no-any-return]
            else:
                raise NotImplementedError(hlo)
        else:
            rec_expr = IndexLambda(
                expr=expr.expr,
                shape=expr.shape,
                dtype=expr.dtype,
                bindings=immutabledict({name: _verify_is_array(self.rec(bnd, None))
                              for name, bnd in sorted(expr.bindings.items())}),
                var_to_reduction_descr=expr.var_to_reduction_descr,
                tags=expr.tags,
                axes=expr.axes,
            )
            return _wrap_einsum_from_ctx(rec_expr, ctx)

    def map_einsum(self,
                   expr: Einsum,
                   ctx: _EinsumDistributiveLawMapperContext | None,
                   ) -> Array:
        distributive_law_descr = self.how_to_distribute(expr)

        if isinstance(distributive_law_descr, DoDistribute):
            if ctx is not None:
                raise RuntimeError("Cannot distribute composed einsums.")
            else:
                ctx = _EinsumDistributiveLawMapperContext(
                    expr.access_descriptors,
                    immutabledict({iarg: arg
                         for iarg, arg in enumerate(expr.args)
                         if iarg != distributive_law_descr.ioperand}),
                    immutabledict(expr.redn_axis_to_redn_descr),
                    tags=expr.tags,
                    axes=expr.axes,
                )
                return _verify_is_array(
                    self.rec(expr.args[distributive_law_descr.ioperand], ctx))
        else:
            assert isinstance(distributive_law_descr, DoNotDistribute)
            rec_expr = Einsum(
                expr.access_descriptors,
                tuple(_verify_is_array(self.rec(arg, None)) for arg in expr.args),
                expr.redn_axis_to_redn_descr,
                tags=expr.tags,
                axes=expr.axes
            )

            return _wrap_einsum_from_ctx(rec_expr, ctx)

    def map_stack(self,
                  expr: Stack,
                  ctx: _EinsumDistributiveLawMapperContext | None) -> Array:
        rec_expr = Stack(tuple(_verify_is_array(self.rec(ary, None))
                               for ary in expr.arrays),
                         expr.axis,
                         tags=expr.tags,
                         axes=expr.axes)
        return _wrap_einsum_from_ctx(rec_expr, ctx)

    def map_concatenate(self,
                        expr: Concatenate,
                        ctx: _EinsumDistributiveLawMapperContext | None
                        ) -> Array:
        rec_expr = Concatenate(tuple(_verify_is_array(self.rec(ary, None))
                                     for ary in expr.arrays),
                               expr.axis,
                               tags=expr.tags,
                               axes=expr.axes)
        return _wrap_einsum_from_ctx(rec_expr, ctx)

    def map_roll(self,
                 expr: Roll,
                 ctx: _EinsumDistributiveLawMapperContext | None
                 ) -> Array:
        rec_expr = Roll(_verify_is_array(self.rec(expr.array, None)),
                        expr.shift,
                        expr.axis,
                        tags=expr.tags,
                        axes=expr.axes)
        return _wrap_einsum_from_ctx(rec_expr, ctx)

    def map_axis_permutation(self,
                             expr: AxisPermutation,
                             ctx: _EinsumDistributiveLawMapperContext | None
                             ) -> Array:
        rec_expr = AxisPermutation(_verify_is_array(self.rec(expr.array, None)),
                                   expr.axis_permutation,
                                   tags=expr.tags,
                                   axes=expr.axes)
        return _wrap_einsum_from_ctx(rec_expr, ctx)

    def _map_index_base(self,
                        expr: IndexBase,
                        ctx: _EinsumDistributiveLawMapperContext | None
                        ) -> Array:
        rec_expr = type(expr)(_verify_is_array(self.rec(expr.array, None)),
                              expr.indices,
                              tags=expr.tags,
                              axes=expr.axes)
        return _wrap_einsum_from_ctx(rec_expr, ctx)

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_reshape(self,
                    expr: Reshape,
                    ctx: _EinsumDistributiveLawMapperContext | None
                    ) -> Array:
        rec_expr = Reshape(_verify_is_array(self.rec(expr.array, None)),
                           expr.newshape,
                           expr.order,
                           tags=expr.tags,
                           axes=expr.axes)
        return _wrap_einsum_from_ctx(rec_expr, ctx)


def apply_distributive_property_to_einsums(
    expr: MappedT,
    how_to_distribute: Callable[[Array], EinsumDistributiveLawDescriptor]
) -> MappedT:
    """
    Returns a copy of *expr* after applying distributive law for einstein
    summation nodes in the expression graph.

    .. testsetup::

        >>> from pytato.transform.einsum_distributive_law import (
        ...     EinsumDistributiveLawDescriptor,
        ...     DoDistribute, DoNotDistribute,
        ...     apply_distributive_property_to_einsums)

    .. doctest::

        >>> import pytato as pt
        >>> x1 = pt.make_placeholder("x1", 4, np.float64)
        >>> x2 = pt.make_placeholder("x2", 4, np.float64)
        >>> A = pt.make_placeholder("A", (10, 4), np.float64)
        >>> y = A @ (x1 + x2)

        >>> def how_to_distribute(expr):
        ...     if pt.analysis.is_einsum_similar_to_subscript(
        ...         expr, "ij,j->i"):
        ...         return DoDistribute(ioperand=1)
        ...     else:
        ...         return DoNotDistribute()

        >>> y_transformed = apply_distributive_property_to_einsums(y,
        ...                     how_to_distribute)

        >>> y_transformed == A @ x1 + A @ x2
        True
    """
    mapper = EinsumDistributiveLawMapper(how_to_distribute)
    return cast("MappedT", mapper(expr, None))
