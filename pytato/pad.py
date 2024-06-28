"""
.. autofunction:: pad
"""

__copyright__ = "Copyright (C) 2023 Kaushik Kulkarni"

from pytato.array import Array, IndexLambda
from pytato.scalar_expr import IntegralT, INT_CLASSES, Scalar
from typing import Union, Sequence, Any, Tuple, List, Dict
from pytools import UniqueNameGenerator

import collections.abc as abc
import pymbolic.primitives as prim
import numpy as np


def _get_constant_padded_idx_lambda(
    array: Array,
    pad_widths: Sequence[Tuple[IntegralT, IntegralT]],
    constant_vals: Sequence[Tuple[Scalar, Scalar]]
) -> IndexLambda:
    """
    Internal routine used by :func:`pad` for constant-mode padding.
    """
    from pytato.array import make_index_lambda
    assert array.ndim == len(pad_widths) == len(constant_vals)

    array_name_in_expr = "in_0"
    bindings: Dict[str, Array] = {array_name_in_expr: array}
    vng = UniqueNameGenerator()
    vng.add_name(array_name_in_expr)

    expr = prim.Variable(array_name_in_expr)[
        tuple((prim.Variable(f"_{idim}") - pad_width[0])
              for idim, pad_width in enumerate(pad_widths))]

    for idim, (pad_width, constant_val) in enumerate(zip(pad_widths, constant_vals)):
        idx_var = prim.Variable(f"_{idim}")
        axis_len = array.shape[idim]

        if isinstance(axis_len, Array):
            binding_name = vng("in_0")
            bindings[binding_name] = axis_len + pad_width[0]
            expr = prim.If(
                prim.Comparison(idx_var, ">=", prim.Variable(binding_name)),
                constant_val[1], expr)
        else:
            assert isinstance(axis_len, INT_CLASSES)
            expr = prim.If(
                prim.Comparison(idx_var, ">=", axis_len + pad_width[0]),
                constant_val[1], expr)

        expr = prim.If(prim.Comparison(idx_var, "<", pad_width[0]),
                       constant_val[0],
                       expr)

    return make_index_lambda(
        expr,
        bindings,
        shape=tuple(axis_len + pad_width[0] + pad_width[1]
                    for axis_len, pad_width in zip(array.shape, pad_widths)),
        dtype=array.dtype)


def _normalize_pad_width(
        array: Array,
        pad_width: Union[IntegralT, Sequence[IntegralT]],
        ) -> Sequence[Tuple[IntegralT, IntegralT]]:
    processed_pad_widths: List[Tuple[IntegralT, IntegralT]]

    if isinstance(pad_width, INT_CLASSES):
        processed_pad_widths = [(pad_width, pad_width)
                                for _ in range(array.ndim)]
    elif (isinstance(pad_width, abc.Sequence)
          and len(pad_width) == 1
          and isinstance(pad_width, INT_CLASSES)):
        processed_pad_widths = [(pad_width[0], pad_width[0])
                                for _ in range(array.ndim)]
    elif (isinstance(pad_width, abc.Sequence)
          and len(pad_width) == 2
          and isinstance(pad_width[0], INT_CLASSES)
          and isinstance(pad_width[1], INT_CLASSES)
          ):
        processed_pad_widths = [(pad_width[0], pad_width[1])] * array.ndim
    elif isinstance(pad_width, abc.Sequence):
        if len(pad_width) != array.ndim:
            raise ValueError(f"Number of pad widths != {array.ndim}"
                             " (the array's dimension)")

        processed_pad_widths = []

        for k in pad_width:
            if (isinstance(k, tuple)
                    and len(k) == 2
                    and isinstance(k[0], INT_CLASSES)
                    and isinstance(k[1], INT_CLASSES)):
                processed_pad_widths.append(k)
            else:
                raise ValueError("Elements of pad_width must be of type"
                                 f" `Tuple[int, int]`, got '{k}'.")

        if all(isinstance(k, INT_CLASSES) for k in pad_width):
            processed_pad_widths = [(k, k) for k in pad_width]
    else:
        raise TypeError("'pad_width' can be an int or "
                        " sequence of pad widths along each"
                        " direction.")

    return processed_pad_widths


def pad(array: Array,
        pad_width: Union[IntegralT, Sequence[IntegralT]],
        mode: str = "constant",
        **kwargs: Any) -> Array:
    r"""
    Returns an array with padded elements along each axis.

    :param array: The array to be padded.
    :param pad_width: Number of elements to be padded along each axis. Can be
        one of:

        - An instance of :class:`int` denoting the constant number of elements
          to pad before and after each axis.
        - A tuple of the form ``(before, after)`` denoting that *before* number
          of padded elements must precede each axis and *after* number of
          padded elements must succeed each axis.
        - A sequence with i-th element as the tuple ``(before_i, after_i)``
          denoting that *before_i* number of padded elements must precede the
          i-th axis and *after_i* number of padded elements must succeed the
          i-th axis.

    :param mode: An instance of :class:`str` denoting the values of the padded
        elements in the returned array. It can be one of:

        - ``"constant"`` denoting that the padded elements must be filled with
          constant entries. See *constant_values*.

    :param constant_values: Optional argument when operating under
        ``"constant"`` *mode*. Can be one of:

        - An instance of :class:`int` denoting the value of every padded
          element.
        - A :class:`tuple` of the form ``(before, after)`` denoting that every
          padded element that precedes *array*'s axes must be set to
          *before* and every padded element that succeeds *array*'s axes must
          be set to *after*.
        - A sequence with the i-th element of the form ``(before_i, after_i)``
          denoting that the padded elements preceding *array*'s i-th axis must
          be set to *before_i* and the padded elements succeeding *array*'s
          i-th axis must be set to *after_i*.

        Defaults to *0*.

    .. note::

        As of March, 2023 the values of the padded elements that are preceding
        wrt certain axes and succeeding wrt other axes is undefined as per
        :func:`numpy.pad`\ 's spec.
    """

    processed_pad_widths = _normalize_pad_width(array, pad_width)

    if mode == "constant":

        # {{{ normalize constant_values

        processed_constant_vals: Sequence[Tuple[Scalar, Scalar]]

        try:
            constant_vals = kwargs.pop("constant_values")
        except KeyError:
            processed_constant_vals = [(0, 0) for _ in range(array.ndim)]
        else:
            if np.isscalar(constant_vals):
                # type-ignore-reason: mypy does not understand the guarding
                # predicate
                processed_constant_vals = [
                    (constant_vals, constant_vals)  # type: ignore[misc]
                    for _ in range(array.ndim)]
            elif (isinstance(constant_vals, tuple)
                    and len(constant_vals) == 2
                    and np.isscalar(constant_vals[0])
                    and np.isscalar(constant_vals[1])
                  ):
                processed_constant_vals = [constant_vals for _ in range(array.ndim)]
            elif isinstance(constant_vals, abc.Sequence):
                if len(constant_vals) != array.ndim:
                    raise ValueError("")

                processed_constant_vals = []
                for constant_val in constant_vals:
                    if (isinstance(constant_val, tuple)
                            and len(constant_val) == 2
                            and np.isscalar(constant_val[0])
                            and np.isscalar(constant_val[1])):
                        processed_constant_vals.append(constant_val)
                    else:
                        raise ValueError(
                            "Elements of `constant_vals` must be of type"
                            f"Tuple[int, int], got '{constant_val}'")
            else:
                raise TypeError("`constant_vals` must be of type int"
                                f" or a sequence of ints, got '{constant_vals}'")

        # }}}

        idx_lambda = _get_constant_padded_idx_lambda(
            array, processed_pad_widths, processed_constant_vals)
    else:
        raise NotImplementedError(f"Mode: '{mode}'")

    if kwargs:
        raise ValueError(f"Some options ('{kwargs.keys()}')"
                         " were left unused.")

    return idx_lambda

# vim: fdm=marker
