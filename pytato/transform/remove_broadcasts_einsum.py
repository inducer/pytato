"""
.. currentmodule:: pytato.transform.remove_broadcasts_einsum

.. autofunction:: rewrite_einsums_with_no_broadcasts
"""
from __future__ import annotations


__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

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

from typing import TYPE_CHECKING, cast

from pytato.array import Array, Einsum, EinsumAxisDescriptor
from pytato.transform import (
    ArrayOrNames,
    ArrayOrNamesTc,
    CacheKeyT,
    CopyMapperWithExtraArgs,
    Mapper,
    _verify_is_array,
)
from pytato.utils import are_shape_components_equal


if TYPE_CHECKING:
    from pytato.function import FunctionDefinition


class EinsumWithNoBroadcastsRewriter(CopyMapperWithExtraArgs[[tuple[int, ...]]]):
    def get_cache_key(
                self,
                expr: ArrayOrNames,
                axes_to_squeeze: tuple[int, ...]
            ) -> CacheKeyT:
        return (expr, axes_to_squeeze)

    def get_function_definition_cache_key(
                self,
                expr: FunctionDefinition,
                axes_to_squeeze: tuple[int, ...]
            ) -> CacheKeyT:
        assert not axes_to_squeeze
        return expr

    def _squeeze_axes(
            self,
            expr: Array,
            axes_to_squeeze: tuple[int, ...]) -> Array:
        return (
            expr[
                tuple(
                    slice(None) if idim not in axes_to_squeeze else 0
                    for idim in range(expr.ndim))]
            if axes_to_squeeze else expr)

    def rec(
            self,
            expr: ArrayOrNames,
            axes_to_squeeze: tuple[int, ...]) -> ArrayOrNames:
        inputs = self._make_cache_inputs(expr, axes_to_squeeze)
        try:
            return self._cache_retrieve(inputs)
        except KeyError:
            rec_result: ArrayOrNames = Mapper.rec(self, expr, ())
            result: ArrayOrNames
            if isinstance(expr, Array):
                result = self._squeeze_axes(
                    _verify_is_array(rec_result),
                    axes_to_squeeze)
            else:
                result = rec_result
            return self._cache_add(inputs, result)

    def map_einsum(
            self, expr: Einsum, axes_to_squeeze: tuple[int, ...]) -> Array:
        new_args: list[Array] = []
        new_access_descriptors: list[tuple[EinsumAxisDescriptor, ...]] = []
        descr_to_axis_len = expr._access_descr_to_axis_len()

        for arg, acc_descrs in zip(expr.args, expr.access_descriptors, strict=True):
            axes_to_squeeze_list: list[int] = []
            for idim, acc_descr in enumerate(acc_descrs):
                if not are_shape_components_equal(arg.shape[idim],
                                                  descr_to_axis_len[acc_descr]):
                    assert are_shape_components_equal(arg.shape[idim], 1)
                    axes_to_squeeze_list.append(idim)
            axes_to_squeeze = tuple(axes_to_squeeze_list)

            if axes_to_squeeze:
                new_arg = _verify_is_array(self.rec(arg, axes_to_squeeze))
                new_acc_descrs = tuple(acc_descr
                                   for idim, acc_descr in enumerate(acc_descrs)
                                   if idim not in axes_to_squeeze)
            else:
                new_arg = _verify_is_array(self.rec(arg, ()))
                new_acc_descrs = acc_descrs

            new_args.append(new_arg)
            new_access_descriptors.append(new_acc_descrs)

        assert len(new_args) == len(expr.args)
        assert len(new_access_descriptors) == len(expr.access_descriptors)

        return expr.replace_if_different(
            args=tuple(new_args), access_descriptors=tuple(new_access_descriptors))


def rewrite_einsums_with_no_broadcasts(expr: ArrayOrNamesTc) -> ArrayOrNamesTc:
    """
    Rewrites all instances of :class:`~pytato.array.Einsum` in *expr* such that the
    einsum expressions avoid broadcasting axes of its operands. We do
    so by updating the :attr:`pytato.array.Einsum.access_descriptors` and slicing
    the operands.

    .. testsetup::

        >>> import pytato as pt
        >>> import numpy as np

    .. doctest::

        >>> a = pt.make_placeholder("a", (10, 4, 1), np.float64)
        >>> b = pt.make_placeholder("b", (10, 1, 4), np.float64)
        >>> expr = pt.einsum("ijk,ijk->i", a, b)
        >>> new_expr = pt.rewrite_einsums_with_no_broadcasts(expr)
        >>> pt.analysis.is_einsum_similar_to_subscript(new_expr, "ij,ik->i")
        True

    .. note::

        This transformation preserves the semantics of the expression i.e. does not
        alter its value.
    """
    mapper = EinsumWithNoBroadcastsRewriter()
    return cast("ArrayOrNamesTc", mapper(expr, ()))

# vim:fdm=marker
