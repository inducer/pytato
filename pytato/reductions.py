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

from typing import Optional, Tuple, Union, Sequence, Dict, List
from pytato.array import ShapeType, Array, make_index_lambda
from pytato.scalar_expr import ScalarExpression, Reduce
import pymbolic.primitives as prim

# {{{ docs

__doc__ = """
.. currentmodule:: pytato

.. autofunction:: sum
.. autofunction:: amin
.. autofunction:: amax
.. autofunction:: prod
"""

# }}}


# {{{ reductions

def _normalize_reduction_axes(
        shape: ShapeType,
        reduction_axes: Optional[Union[int, Tuple[int]]]
        ) -> Tuple[ShapeType, Tuple[int, ...]]:
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

    if isinstance(reduction_axes, int):
        reduction_axes = reduction_axes,

    if not isinstance(reduction_axes, tuple):
        raise TypeError("Reduction axes expected to be of type 'NoneType', 'int'"
                f" or 'tuple'. (Got {type(reduction_axes)})")

    for axis in reduction_axes:
        if not (0 <= axis < len(shape)):
            raise ValueError(f"{axis} is out of bounds for array of dimension"
                    f" {len(shape)}.")

    new_shape = tuple([axis_len
        for i, axis_len in enumerate(shape)
        if i not in reduction_axes])
    return new_shape, reduction_axes


def _get_reduction_indices_bounds(shape: ShapeType,
        axes: Tuple[int, ...]) -> Tuple[
                Sequence[ScalarExpression],
                Dict[str, Tuple[ScalarExpression, ScalarExpression]]]:
    """Given *shape* and reduction axes *axes*, produce a list of inames
    ``indices`` named appropriately for reduction inames.
    Also fill a dictionary with bounds for reduction inames
    ``redn_bounds = {red_iname: (lower_bound, upper_bound)}``,
    where the bounds are given as a Python-style half-open interval.
    :returns: ``indices, redn_bounds``
    """
    indices: List[prim.Variable] = []
    redn_bounds: Dict[str, Tuple[ScalarExpression, ScalarExpression]] = {}

    n_out_dims = 0
    n_redn_dims = 0
    for idim, axis_len in enumerate(shape):
        if idim in axes:
            if not isinstance(axis_len, int):
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

    from pyrsistent import pmap

    # insufficient type annotation in pyrsistent
    return indices, pmap(redn_bounds)  # type: ignore


def _make_reduction_lambda(op: str, a: Array,
                      axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Return a :class:`IndexLambda` that performs reduction over the *axis* axes
    of *a* with the reduction op *op*.

    :arg op: The reduction operation to perform.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes over which the reduction is to be performed. If axis is
        *None*, perform reduction over all of *a*'s axes.
    """
    new_shape, axes = _normalize_reduction_axes(a.shape, axis)
    del axis
    indices, redn_bounds = _get_reduction_indices_bounds(a.shape, axes)

    return make_index_lambda(
            Reduce(
                prim.Subscript(prim.Variable("in"), tuple(indices)),
                op,
                redn_bounds),
            {"in": a},
            new_shape,
            a.dtype)


def sum(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Sums array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be sum-reduced.
        Defaults to all axes of the input array.
    """
    return _make_reduction_lambda("sum", a, axis)


def amax(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Returns the max of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be max-reduced.
        Defaults to all axes of the input array.
    """
    return _make_reduction_lambda("max", a, axis)


def amin(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Returns the min of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be min-reduced.
        Defaults to all axes of the input array.
    """
    return _make_reduction_lambda("min", a, axis)


def prod(a: Array, axis: Optional[Union[int, Tuple[int]]] = None) -> Array:
    """
    Returns the product of array *a*'s elements along the *axis* axes.

    :arg a: The :class:`pytato.Array` on which to perform the reduction.

    :arg axis: The axes along which the elements are to be product-reduced.
        Defaults to all axes of the input array.
    """
    return _make_reduction_lambda("product", a, axis)

# }}}

# vim: foldmethod=marker
