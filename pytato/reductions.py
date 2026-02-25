from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Xiaoyu Wei
Copyright (C) 2021 Kaushik Kulkarni
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


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from constantdict import constantdict
from typing_extensions import override

import pymbolic.primitives as prim
from pymbolic import ArithmeticExpression

from pytato.array import Array, ReductionDescriptor, ShapeType, make_index_lambda
from pytato.scalar_expr import INT_CLASSES, Reduce


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# {{{ docs

__doc__ = """
.. currentmodule:: pytato

.. autofunction:: sum
.. autofunction:: amin
.. autofunction:: amax
.. autofunction:: prod
.. autofunction:: all
.. autofunction:: any

.. currentmodule:: pytato.reductions

.. autoclass:: ReductionOperation
.. autoclass:: SumReductionOperation
.. autoclass:: ProductReductionOperation
.. autoclass:: MaxReductionOperation
.. autoclass:: MinReductionOperation
.. autoclass:: AllReductionOperation
.. autoclass:: AnyReductionOperation
"""

# }}}


class _NoValue:
    pass


# {{{ reduction operations

class ReductionOperation(ABC):
    """
    .. automethod:: scalar_op_name
    .. automethod:: neutral_element
    .. automethod:: __hash__
    .. automethod:: __eq__
    """
    @classmethod
    @abstractmethod
    def scalar_op_name(cls) -> str:
        ...

    @abstractmethod
    def neutral_element(self, dtype: np.dtype[Any]) -> Any:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...


class _StatelessReductionOperation(ReductionOperation):
    def update_persistent_hash(self, key_hash: Any, key_builder: Any) -> None:
        key_builder.rec(key_hash, type(self))

    def __hash__(self) -> int:
        return hash(type(self))

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)


class SumReductionOperation(_StatelessReductionOperation):
    @override
    @classmethod
    def scalar_op_name(cls):
        return "+"

    def neutral_element(self, dtype: np.dtype[Any]) -> Any:
        return 0


class ProductReductionOperation(_StatelessReductionOperation):
    @override
    @classmethod
    def scalar_op_name(cls):
        return "*"

    def neutral_element(self, dtype: np.dtype[Any]) -> Any:
        return 1


class MaxReductionOperation(_StatelessReductionOperation):
    @override
    @classmethod
    def scalar_op_name(cls):
        return "max"

    def neutral_element(self, dtype: np.dtype[Any]) -> Any:
        if dtype.kind == "f":
            return dtype.type(float("-inf"))
        elif dtype.kind == "i":
            return np.iinfo(dtype).min
        else:
            raise TypeError(f"unknown neutral element for max and {dtype}")


class MinReductionOperation(_StatelessReductionOperation):
    @override
    @classmethod
    def scalar_op_name(cls):
        return "min"

    def neutral_element(self, dtype: np.dtype[Any]) -> Any:
        if dtype.kind == "f":
            return dtype.type(float("inf"))
        elif dtype.kind == "i":
            return np.iinfo(dtype).max
        else:
            raise TypeError(f"unknown neutral element for min and {dtype}")


class AllReductionOperation(_StatelessReductionOperation):
    @override
    @classmethod
    def scalar_op_name(cls):
        return "or"

    def neutral_element(self, dtype: np.dtype[Any]) -> Any:
        return np.bool_(True)


class AnyReductionOperation(_StatelessReductionOperation):
    @override
    @classmethod
    def scalar_op_name(cls):
        return "and"

    def neutral_element(self, dtype: np.dtype[Any]) -> Any:
        return np.bool_(False)

# }}}


# {{{ reductions

def _normalize_reduction_axes(
        shape: ShapeType,
        reduction_axes: int | tuple[int, ...] | None
        ) -> tuple[ShapeType, tuple[int, ...]]:
    """
    Returns a :class:`tuple` of ``(new_shape, normalized_redn_axes)``, where
    *new_shape* is the shape of the ndarray after the axes corresponding to
    *reduction_axes* are reduced and *normalized_redn_axes* is a :class:`tuple`
    of axes indices over which reduction is to be performed.

    :arg reduction_axes: Axis indices over which reduction is to be performed.
        If *reduction_axes* is None, the reduction is performed over all
        axes.
    :arg shape: Shape of the array over which reduction is to be
        performed
    """
    if reduction_axes is None:
        return (), tuple(range(len(shape)))

    if isinstance(reduction_axes, INT_CLASSES):
        reduction_axes = reduction_axes,

    if not isinstance(reduction_axes, tuple):
        raise TypeError("Reduction axes expected to be of type 'NoneType', 'int'"
                f" or 'tuple'. (Got {type(reduction_axes)})")

    for axis in reduction_axes:
        if not (0 <= axis < len(shape)):
            raise ValueError(f"{axis} is out of bounds for array of dimension"
                    f" {len(shape)}.")

    new_shape = tuple(axis_len
        for i, axis_len in enumerate(shape)
        if i not in reduction_axes)
    return new_shape, reduction_axes


def _get_reduction_indices_bounds(shape: ShapeType,
                                  axes: tuple[int, ...],
                                  ) -> tuple[Sequence[prim.Variable],
                                             Mapping[str, tuple[ArithmeticExpression,
                                                             ArithmeticExpression]]]:
    """
    Given *shape* and reduction axes *axes*, produce a list of inames
    ``indices`` named appropriately for reduction inames.
    Also fill a dictionary with bounds for reduction inames
    ``redn_bounds = {red_iname: (lower_bound, upper_bound)}``,
    where the bounds are given as a Python-style half-open interval.

    :returns: ``indices, redn_bounds, var_to_redn_descr``
    """
    indices: list[prim.Variable] = []
    redn_bounds: dict[str, tuple[ArithmeticExpression, ArithmeticExpression]] = {}

    n_out_dims = 0
    n_redn_dims = 0
    for idim, axis_len in enumerate(shape):
        if idim in axes:
            if not isinstance(axis_len, INT_CLASSES):
                # TODO: add bindings for shape array expressions
                raise NotImplementedError("Parametric shapes for reduction axes"
                                          " not yet supported.")

            idx = f"_r{n_redn_dims}"
            indices.append(prim.Variable(idx))
            redn_bounds[idx] = (0, axis_len)

            n_redn_dims += 1
        else:
            indices.append(prim.Variable(f"_{n_out_dims}"))
            n_out_dims += 1

    return indices, constantdict(redn_bounds)


def _get_var_to_redn_descr(
        shape: ShapeType,
        axes: tuple[int, ...],
        axis_to_reduction_descr: Mapping[int, ReductionDescriptor] | None
    ) -> Mapping[str, ReductionDescriptor]:
    """
    :arg axis_to_reduction_descr: Mapping from a reduction axis to
        its instance of :class:`~pytato.ReductionDescriptor`. This mapping
        is provided by the caller of top-level functions like
        :func:`pytato.sum`, :func:`pytato.prod`.
    """
    var_to_redn_descr = {}

    if axis_to_reduction_descr is None:
        axis_to_reduction_descr = {}

    if not (frozenset(axis_to_reduction_descr) <= frozenset(axes)):
        raise ValueError("Axes "
                         f"'{frozenset(axis_to_reduction_descr) - frozenset(axes)}'"
                         " in 'axis_to_reduction_descr' not a part of axes"
                         " to be reduced over.")

    n_redn_dims = 0
    for idim, axis_len in enumerate(shape):
        if idim in axes:
            if not isinstance(axis_len, INT_CLASSES):
                # TODO: add bindings for shape array expressions
                raise NotImplementedError("Parametric shapes for reduction axes"
                                          " not yet supported.")

            idx = f"_r{n_redn_dims}"
            redn_descr = axis_to_reduction_descr.get(
                idim,
                ReductionDescriptor(frozenset()))
            if not isinstance(redn_descr, ReductionDescriptor):
                raise TypeError(f"'axis_to_reduction_descr[{idim}]': "
                                "expected an instance of ReductionDescriptor, "
                                f"got {type(redn_descr)}.")
            var_to_redn_descr[idx] = redn_descr
            n_redn_dims += 1

    return constantdict(var_to_redn_descr)


def _make_reduction_lambda(
        op: ReductionOperation, a: Array,
        axis: int | tuple[int, ...] | None = None,
        axis_to_reduction_descr: Mapping[int, ReductionDescriptor] | None = None,
        initial: Any = _NoValue) -> Array:
    """
    Return a :class:`IndexLambda` that performs reduction over the *axis* axes
    of *a* with the reduction op *op*.

    :arg op: The reduction operation to perform.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes over which the reduction is to be performed. If axis is
        *None*, perform reduction over all of *a*'s axes.
    """
    new_shape, reduction_axes = _normalize_reduction_axes(a.shape, axis)
    del axis
    indices, redn_bounds = _get_reduction_indices_bounds(a.shape,
                                                         reduction_axes)

    var_to_redn_descr = _get_var_to_redn_descr(a.shape,
                                               reduction_axes,
                                               axis_to_reduction_descr)

    if initial is _NoValue:
        for iax in reduction_axes:
            shape_iax = a.shape[iax]

            from pytato.utils import are_shape_components_equal
            if are_shape_components_equal(shape_iax, 0):
                raise ValueError(
                        "zero-size reduction operation with no supplied "
                        "'initial' value")

            if isinstance(iax, Array):
                raise NotImplementedError(
                        "cannot statically determine emptiness of "
                        f"reduction axis {iax} (0-based)")

    elif initial != op.neutral_element(a.dtype):
        raise NotImplementedError("reduction with 'initial' not equal to the "
                "neutral element")

    return make_index_lambda(
            Reduce(
                prim.Subscript(prim.Variable("in"), tuple(indices)),
                op,
                redn_bounds),
            {"in": a},
            new_shape,
            a.dtype,
            var_to_reduction_descr=var_to_redn_descr)


def sum(a: Array,
        axis: int | tuple[int, ...] | None = None,
        initial: Any = 0,
        axis_to_reduction_descr: Mapping[int, ReductionDescriptor] | None = None
        ) -> Array:
    """
    Sums array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be sum-reduced.
        Defaults to all axes of the input array.
    :arg initial: The value returned for an empty array, if supplied.
        This value also serves as the base value onto which any additional
        array entries are accumulated.

    :arg axis_to_reduction_descr: A mapping from axis in *axis* to the
        corresponding instance of :class:`~pytato.ReductionDescriptor` that the
        :class:`~pytato.array.IndexLambda` is to be instantiated with.
    """
    return _make_reduction_lambda(SumReductionOperation(), a, axis,
                                  axis_to_reduction_descr, initial)


def amax(a: Array, axis: int | tuple[int, ...] | None = None, *,
         initial: Any = _NoValue,
         axis_to_reduction_descr: Mapping[int, ReductionDescriptor] | None = None
         ) -> Array:
    """
    Returns the max of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be max-reduced.
        Defaults to all axes of the input array.
    :arg initial: The value returned for an empty array, if supplied.
        This value also serves as the base value onto which any additional
        array entries are accumulated.
        If not supplied, an :exc:`ValueError` will be raised
        if the reduction is empty.
        In that case, the reduction size must not be symbolic.

    :arg axis_to_reduction_descr: A mapping from axis in *axis* to the
        corresponding instance of :class:`~pytato.ReductionDescriptor` that the
        :class:`~pytato.array.IndexLambda` is to be instantiated with.
    """
    return _make_reduction_lambda(MaxReductionOperation(), a, axis,
                                  axis_to_reduction_descr, initial)


def amin(a: Array,
         axis: int | tuple[int, ...] | None = None,
         initial: Any = _NoValue,
         axis_to_reduction_descr: Mapping[int, ReductionDescriptor] | None = None
         ) -> Array:
    """
    Returns the min of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be min-reduced.
        Defaults to all axes of the input array.
    :arg initial: The value returned for an empty array, if supplied.
        This value also serves as the base value onto which any additional
        array entries are accumulated.
        If not supplied, an :exc:`ValueError` will be raised
        if the reduction is empty.
        In that case, the reduction size must not be symbolic.
    :arg axis_to_reduction_descr: A mapping from axis in *axis* to the
        corresponding instance of :class:`~pytato.ReductionDescriptor` that the
        :class:`~pytato.array.IndexLambda` is to be instantiated with.
    """
    return _make_reduction_lambda(MinReductionOperation(), a, axis,
                                  axis_to_reduction_descr, initial)


def prod(a: Array,
         axis: int | tuple[int, ...] | None = None,
         initial: Any = 1,
         axis_to_reduction_descr: Mapping[int, ReductionDescriptor] | None = None
         ) -> Array:
    """
    Returns the product of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be product-reduced.
        Defaults to all axes of the input array.
    :arg initial: The value returned for an empty array, if supplied.
        This value also serves as the base value onto which any additional
        array entries are accumulated.
    :arg axis_to_reduction_descr: A mapping from axis in *axis* to the
        corresponding instance of :class:`~pytato.ReductionDescriptor` that the
        :class:`~pytato.array.IndexLambda` is to be instantiated with.
    """
    return _make_reduction_lambda(ProductReductionOperation(), a, axis,
                                  axis_to_reduction_descr, initial)


def all(a: Array,
        axis: int | tuple[int, ...] | None = None,
        axis_to_reduction_descr: Mapping[int, ReductionDescriptor] | None = None
        ) -> Array:
    """
    Returns the logical-and array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be product-reduced.
        Defaults to all axes of the input array.

    :arg axis_to_reduction_descr: A mapping from axis in *axis* to the
        corresponding instance of :class:`~pytato.ReductionDescriptor` that the
        :class:`~pytato.array.IndexLambda` is to be instantiated with.
    """
    return _make_reduction_lambda(AllReductionOperation(), a, axis,
                                  axis_to_reduction_descr, initial=True)


def any(a: Array,
        axis: int | tuple[int, ...] | None = None,
        axis_to_reduction_descr: Mapping[int, ReductionDescriptor] | None = None
        ) -> Array:
    """
    Returns the logical-or of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be product-reduced.
        Defaults to all axes of the input array.

    :arg axis_to_reduction_descr: A mapping from axis in *axis* to the
        corresponding instance of :class:`~pytato.ReductionDescriptor` that the
        :class:`~pytato.array.IndexLambda` is to be instantiated with.
    """
    return _make_reduction_lambda(AnyReductionOperation(), a, axis,
                                  axis_to_reduction_descr, initial=False)

# }}}

# vim: foldmethod=marker
