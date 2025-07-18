from __future__ import annotations


__copyright__ = "Copyright (C) 2021 Kaushik Kulkarni"

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
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

import islpy as isl
import numpy as np
from immutabledict import immutabledict
from typing_extensions import Never

import pymbolic.primitives as prim
from pymbolic import ArithmeticExpression, Bool, Scalar
from pytools import UniqueNameGenerator

from pytato.array import (
    AdvancedIndexInContiguousAxes,
    AdvancedIndexInNoncontiguousAxes,
    Array,
    ArrayOrScalar,
    BasicIndex,
    ConvertibleToIndexExpr,
    Einsum,
    IndexExpr,
    IndexLambda,
    NormalizedSlice,
    Placeholder,
    ShapeComponent,
    ShapeType,
    SizeParam,
    _dtype_any,
)
from pytato.scalar_expr import (
    INT_CLASSES,
    SCALAR_CLASSES,
    ScalarExpression,
    TypeCast,
)
from pytato.transform import CachedMapper


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from pytools.tag import Tag


__doc__ = """
Helper routines
---------------

.. autofunction:: are_shape_components_equal
.. autofunction:: are_shapes_equal
.. autofunction:: get_shape_after_broadcasting
.. autofunction:: dim_to_index_lambda_components
.. autofunction:: get_common_dtype_of_ary_or_scalars
.. autofunction:: get_einsum_subscript_str

References
^^^^^^^^^^

.. class:: UniqueNameGenerator

    See :class:`pytools.UniqueNameGenerator`.
"""


# {{{ partition

Tpart = TypeVar("Tpart")


def partition(pred: Callable[[Tpart], bool],
              iterable: Iterable[Tpart]) -> tuple[list[Tpart],
                                                  list[Tpart]]:
    """
    Use a predicate to partition entries into false entries and true
    entries
    """
    # Inspired from https://docs.python.org/3/library/itertools.html
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    from itertools import filterfalse, tee
    t1, t2 = tee(iterable)
    return list(filterfalse(pred, t1)), list(filter(pred, t2))

# }}}


def get_shape_after_broadcasting(
            exprs: Iterable[Array | Scalar]
        ) -> ShapeType:
    """
    Returns the shape after broadcasting *exprs* in an operation.
    """
    from pytato.diagnostic import CannotBroadcastError
    shapes = [expr.shape if isinstance(expr, Array) else () for expr in exprs]

    result_dim = max((len(s) for s in shapes), default=0)

    # append leading dimensions of all the shapes with 1's to match result_dim.
    augmented_shapes = [((1,)*(result_dim-len(s)) + s) for s in shapes]

    def _get_result_axis_length(axis_lengths: list[ShapeComponent]
                                ) -> ShapeComponent:
        result_axis_len = axis_lengths[0]
        for axis_len in axis_lengths[1:]:
            if (are_shape_components_equal(axis_len, result_axis_len)
                    or are_shape_components_equal(axis_len, 1)):
                pass
            elif are_shape_components_equal(result_axis_len, 1):
                result_axis_len = axis_len
            else:
                raise CannotBroadcastError("operands could not be broadcasted "
                                           "together with shapes "
                                           f"{' '.join(str(s) for s in shapes)}.")
        return result_axis_len

    return tuple(_get_result_axis_length([s[i] for s in augmented_shapes])
                 for i in range(result_dim))


def get_indexing_expression(shape: ShapeType,
                            result_shape: ShapeType) -> tuple[ScalarExpression, ...]:
    """
    Returns the indices while broadcasting an array of shape *shape* into one of
    shape *result_shape*.
    """
    assert len(shape) <= len(result_shape)
    i_start = len(result_shape) - len(shape)
    indices: list[ArithmeticExpression] = []
    for i, (dim1, dim2) in enumerate(zip(shape, result_shape[i_start:], strict=True)):
        if not are_shape_components_equal(dim1, dim2):
            assert are_shape_components_equal(dim1, 1)
            indices.append(0)
        else:
            indices.append(prim.Variable(f"_{i+i_start}"))

    return tuple(indices)


def with_indices_for_broadcasted_shape(val: prim.Variable, shape: ShapeType,
                                       result_shape: ShapeType) -> ArithmeticExpression:
    if len(shape) == 0:
        # scalar expr => do not index
        return val
    else:
        return val[get_indexing_expression(shape, result_shape)]


def update_bindings_and_get_broadcasted_expr(arr: ArrayOrScalar,
                                             bnd_name: str,
                                             bindings: dict[str, Array],
                                             result_shape: ShapeType
                                             ) -> ScalarExpression | Bool:
    """
    Returns an instance of :class:`~pytato.scalar_expr.ScalarExpression` to address
    *arr* in a :class:`pytato.array.IndexLambda` of shape *result_shape*.
    """

    if isinstance(arr, SCALAR_CLASSES):
        assert not isinstance(arr, Array)
        if np.isnan(arr):
            # allowing NaNs to stay in our expression trees could potentially
            # lead to spuriously unequal comparisons between expressions
            from pymbolic.primitives import NaN
            return NaN(np.array(arr).dtype.type)
        else:
            return arr

    assert isinstance(arr, Array)
    bindings[bnd_name] = arr
    return with_indices_for_broadcasted_shape(prim.Variable(bnd_name),
                                              arr.shape,
                                              result_shape)


def broadcast_binary_op(a1: ArrayOrScalar, a2: ArrayOrScalar,
                        op: Callable[[ScalarExpression, ScalarExpression], ScalarExpression],  # noqa:E501
                        get_result_type: Callable[[ArrayOrScalar, ArrayOrScalar], np.dtype[Any]],  # noqa:E501
                        *,
                        tags: frozenset[Tag],
                        non_equality_tags: frozenset[Tag],
                        cast_to_result_dtype: bool,
                        is_pow: bool,
                        ) -> ArrayOrScalar:
    from pytato.array import _get_default_axes

    if np.isscalar(a1) and np.isscalar(a2):
        from pytato.scalar_expr import evaluate
        return evaluate(op(a1, a2))  # type: ignore

    result_shape = get_shape_after_broadcasting([a1, a2])

    # Note: get_result_type calls np.result_type by default, which means
    # that we are passing a pytato array to numpy. Luckily, np.result_type
    # only looks at the dtype of input arrays as of numpy v2.1.
    result_dtype = get_result_type(a1, a2)

    bindings: dict[str, Array] = {}

    expr1 = update_bindings_and_get_broadcasted_expr(a1, "_in0", bindings,
                                                     result_shape)
    expr2 = update_bindings_and_get_broadcasted_expr(a2, "_in1", bindings,
                                                     result_shape)

    def cast_to_result_type(
                array: ArrayOrScalar,
                expr: ScalarExpression | Bool
            ) -> ScalarExpression | Bool:
        if ((isinstance(array, Array | np.generic))
                and array.dtype != result_dtype):
            # Loopy's type casts don't like casting to bool
            assert result_dtype != np.bool_

            # See https://github.com/inducer/pytato/issues/542
            # on why pow() + integers is not typecast to float or complex.
            if not (is_pow
                    and np.issubdtype(array.dtype, np.integer)
                    and not np.issubdtype(result_dtype, np.integer)):
                expr = TypeCast(result_dtype, expr)
        elif isinstance(expr, SCALAR_CLASSES):
            # See https://github.com/inducer/pytato/issues/542
            # on why pow() + integers is not typecast to float or complex.
            if not (is_pow
                    and np.issubdtype(type(expr), np.integer)
                    and not np.issubdtype(result_dtype, np.integer)):
                expr = result_dtype.type(expr)

        return expr

    if cast_to_result_dtype:
        expr1 = cast_to_result_type(a1, expr1)
        expr2 = cast_to_result_type(a2, expr2)

    return IndexLambda(expr=op(expr1, expr2),
                       shape=result_shape,
                       dtype=result_dtype,
                       bindings=immutabledict(bindings),
                       tags=tags,
                       non_equality_tags=non_equality_tags,
                       var_to_reduction_descr=immutabledict(),
                       axes=_get_default_axes(len(result_shape)))


# {{{ dim_to_index_lambda_components

class ShapeExpressionMapper(CachedMapper[ScalarExpression, Never, []]):
    """
    Mapper that takes a shape component and returns it as a scalar expression.
    """
    def __init__(self, var_name_gen: UniqueNameGenerator):
        super().__init__()
        self.var_name_gen = var_name_gen
        self.bindings: dict[str, SizeParam] = {}

    def map_index_lambda(self, expr: IndexLambda) -> ScalarExpression:
        from pytato.scalar_expr import substitute
        res = substitute(expr.expr, {name: self.rec(val)
                                      for name, val in expr.bindings.items()})
        assert prim.is_arithmetic_expression(res)
        return res

    def map_size_param(self, expr: SizeParam) -> ScalarExpression:
        name = self.var_name_gen("_in")
        self.bindings[name] = expr
        return prim.Variable(name)


def dim_to_index_lambda_components(expr: ShapeComponent,
                                   vng: UniqueNameGenerator | None = None,
                                   ) -> tuple[ScalarExpression,
                                              dict[str, SizeParam]]:
    """
    Returns the scalar expressions and bindings to use the shape
    component within an index lambda.

    .. testsetup::

        >>> import pytato as pt
        >>> from pytato.utils import dim_to_index_lambda_components
        >>> from pytools import UniqueNameGenerator

    .. doctest::

        >>> n = pt.make_size_param("n")
        >>> expr, bnds = dim_to_index_lambda_components(3*n+8, UniqueNameGenerator())
        >>> print(expr)
        3*_in + 8
        >>> bnds
        {'_in': SizeParam(name='n')}
    """
    if isinstance(expr, INT_CLASSES):
        return expr, {}

    if vng is None:
        vng = UniqueNameGenerator()

    assert isinstance(vng, UniqueNameGenerator)
    assert isinstance(expr, Array)
    mapper = ShapeExpressionMapper(vng)
    result = mapper(expr)
    return result, mapper.bindings

# }}}


def are_shape_components_equal(
            dim1: ShapeComponent,
            dim2: ShapeComponent,
        ) -> bool:
    """
    Returns *True* iff *dim1* and *dim2* are have equal
    :class:`~pytato.array.SizeParam` coefficients in their expressions.
    """
    if isinstance(dim1, INT_CLASSES) and isinstance(dim2, INT_CLASSES):
        return dim1 == dim2

    from pytato.transform import deduplicate
    dim1_minus_dim2 = dim1 - dim2
    assert isinstance(dim1_minus_dim2, Array)
    dim1_minus_dim2 = deduplicate(dim1_minus_dim2)

    from pytato.transform import InputGatherer
    inputs = InputGatherer()(dim1_minus_dim2)
    named_inputs: list[Placeholder | SizeParam] = [
        expr for expr in inputs
        if isinstance(expr, SizeParam | Placeholder)
    ]

    space = _create_size_param_space({expr.name for expr in named_inputs})

    # pytato requires the shape expressions be affine expressions of the size
    # params, so converting them to ISL expressions.
    aff = ShapeToISLExpressionMapper(space)(dim1_minus_dim2)
    return (aff.is_cst()  # type: ignore[no-any-return]
            and aff.get_constant_val().is_zero())


def are_shapes_equal(shape1: ShapeType, shape2: ShapeType) -> bool:
    """
    Returns *True* iff *shape1* and *shape2* have the same dimensionality and the
    corresponding components are equal as defined by
    :func:`~pytato.utils.are_shape_components_equal`.
    """
    return ((len(shape1) == len(shape2))
            and all(are_shape_components_equal(dim1, dim2)
                    for dim1, dim2 in zip(shape1, shape2, strict=True)))


# {{{ ShapeToISLExpressionMapper

class ShapeToISLExpressionMapper(CachedMapper[isl.Aff, Never, []]):
    """
    Mapper that takes a shape component and returns it as :class:`isl.Aff`.
    """
    def __init__(self, space: isl.Space):
        super().__init__()
        self.space = space

    def map_index_lambda(self, expr: IndexLambda) -> isl.Aff:
        from pytato.scalar_expr import evaluate
        return evaluate(expr.expr, {name: self.rec(val)
                                    for name, val in expr.bindings.items()})

    def map_size_param(self, expr: SizeParam) -> isl.Aff:
        dt, pos = self.space.get_var_dict()[expr.name]
        return isl.Aff.var_on_domain(self.space, dt, pos)

# }}}


def _create_size_param_space(names: Iterable[str]) -> isl.Space:
    return isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
                                       set=[],
                                       params=sorted(names)).params()


def _get_size_params_assumptions_bset(space: isl.Space) -> isl.BasicSet:
    bset = isl.BasicSet.universe(space)
    for name in bset.get_var_dict():
        bset = bset.add_constraint(isl.Constraint.ineq_from_names(space, {name: 1}))

    return bset


def _is_non_negative(expr: ShapeComponent) -> Bool:
    """
    Returns *True* iff it can be proven that ``expr >= 0``.
    """
    if isinstance(expr, INT_CLASSES):
        return expr >= 0

    assert isinstance(expr, Array) and expr.shape == ()
    from pytato.transform import InputGatherer
    # FIXME: This will run into trouble for data-dependent shape components, which
    # may contain inputs other than Placeholders.
    space = _create_size_param_space({cast("Placeholder", expr).name
                                      for expr in InputGatherer()(expr)})
    aff = ShapeToISLExpressionMapper(space)(expr)

    return aff.ge_set(aff * 0) >= _get_size_params_assumptions_bset(space).to_set()


def _is_non_positive(expr: ShapeComponent) -> Bool:
    """
    Returns *True* iff it can be proven that ``expr <= 0``.
    """
    return _is_non_negative(-expr)


# {{{ _index_into

# {{{ normalized slice

def _normalize_slice(slice_: slice,
                     axis_len: ShapeComponent) -> NormalizedSlice:
    start, stop, step = slice_.start, slice_.stop, slice_.step
    if step is None:
        step = 1
    if not isinstance(step, INT_CLASSES):
        raise ValueError(f"slice step must be an int or 'None' (got a {type(step)})")
    if step == 0:
        raise ValueError("slice step cannot be zero")

    if step > 0:
        default_start: ShapeComponent = 0
        default_stop: ShapeComponent = axis_len
    else:
        default_start = axis_len - 1
        default_stop = -1

    if start is None:
        start = default_start
    else:
        if isinstance(axis_len, INT_CLASSES):
            if -axis_len <= start < axis_len:
                start = start % axis_len
            elif start >= axis_len:
                start = axis_len if step > 0 else axis_len - 1
            else:
                start = 0 if step > 0 else -1
        else:
            raise NotImplementedError

    if stop is None:
        stop = default_stop
    else:
        if isinstance(axis_len, INT_CLASSES):
            if -axis_len <= stop < axis_len:
                stop = stop % axis_len
            elif stop >= axis_len:
                stop = axis_len if step > 0 else axis_len - 1
            else:
                stop = 0 if step > 0 else -1
        else:
            raise NotImplementedError

    return NormalizedSlice(start, stop, step)


def _normalized_slice_len(slice_: NormalizedSlice) -> ShapeComponent:
    start, stop, step = cast(
                "tuple[int, int, int]",
                (slice_.start, slice_.stop, slice_.step))

    if step > 0:
        if _is_non_negative(stop - start):
            return (stop - start + step - 1) // step
        elif _is_non_positive(stop - start):
            return 0
        else:
            # ISL could not ascertain the expression's sign
            raise NotImplementedError("could not ascertain the sign of "
                                      f"{stop-start} while computing the axis"
                                      " length.")
    else:
        if _is_non_negative(start - stop):
            return (start - stop - step - 1) // (-step)
        elif _is_non_positive(start - stop):
            return 0
        else:
            # ISL could not ascertain the expression's sign
            raise NotImplementedError("could not ascertain the sign of "
                                      f"{start-stop} while computing the axis"
                                      " length.")


def normalized_slice_does_not_change_axis(slice_: NormalizedSlice,
                                           shape_to_compare_to: ShapeComponent) -> bool:

    return (are_shape_components_equal(
                    slice_.stop, shape_to_compare_to)
        and are_shape_components_equal(slice_.step, 1)
        and are_shape_components_equal(slice_.start, 0))

# }}}


def _index_into(
        ary: Array,
        indices: tuple[ConvertibleToIndexExpr, ...],
        tags: frozenset[Tag],
        non_equality_tags: frozenset[Tag]) -> Array:
    from pytato.array import _get_default_axes
    from pytato.diagnostic import CannotBroadcastError

    # {{{ handle ellipsis

    if indices.count(...) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    if indices.count(...):
        ellipsis_pos = indices.index(...)
        indices = (indices[:ellipsis_pos]
                   + (slice(None, None, None),) * (ary.ndim - len(indices) + 1)
                   + indices[ellipsis_pos+1:])

    indices = cast("tuple[int, slice, Array, None]", indices)

    # }}}

    # {{{ "pad" index with complete slices to match ary's ndim

    if len(indices) < ary.ndim:
        indices = indices + (slice(None, None, None),) * (ary.ndim - len(indices))

    # }}}

    if len(indices) != ary.ndim:
        raise IndexError(f"Too many indices (expected {ary.ndim}"
                         f", got {len(indices)})")

    if any(idx is None for idx in indices):
        raise NotImplementedError("newaxis is not supported")

    # {{{ validate broadcastability of the array indices

    try:
        array_idx_shape = get_shape_after_broadcasting(
                [idx for idx in indices if isinstance(idx, Array)])
    except CannotBroadcastError as e:
        raise IndexError(str(e)) from None

    # }}}

    # {{{ validate index

    for i, idx in enumerate(indices):
        if isinstance(idx, slice):
            pass
        elif isinstance(idx, INT_CLASSES):
            if not (_is_non_negative(idx + ary.shape[i])
                    and _is_non_negative(ary.shape[i] - 1 - idx)):
                raise IndexError(f"{idx} is out of bounds for axis {i}")
        elif isinstance(idx, Array):
            if idx.dtype.kind not in ["i", "u"]:
                raise IndexError("only integer arrays are valid array indices")
            if (_is_non_positive(ary.shape[i])
                    and (not are_shape_components_equal(idx.size, 0))):
                raise IndexError("Indirect indexing into a non-positive"
                                 f" dimension (axis {i}) is illegal.")
        else:
            raise IndexError("only integers, slices, ellipsis and integer arrays"
                             " are valid indices")

    # }}}

    # {{{ normalize slices

    normalized_indices: list[IndexExpr] = [_normalize_slice(idx, axis_len)
                                           if isinstance(idx, slice)
                                           else idx
                                           for idx, axis_len
                                           in zip(indices, ary.shape, strict=True)]

    del indices

    # }}}

    if any(isinstance(idx, Array) for idx in normalized_indices):
        # advanced indexing expression
        i_adv_indices, i_basic_indices = partition(
                                            lambda idx: isinstance(
                                                            normalized_indices[idx],
                                                            NormalizedSlice),
                                            range(len(normalized_indices)))
        if any(i_adv_indices[0] < i_basic_idx < i_adv_indices[-1]
               for i_basic_idx in i_basic_indices):
            # non contiguous advanced indices
            return AdvancedIndexInNoncontiguousAxes(
                ary,
                tuple(normalized_indices),
                tags=tags,
                non_equality_tags=non_equality_tags,
                axes=_get_default_axes(len(array_idx_shape)
                                       + len(i_basic_indices)))
        else:
            return AdvancedIndexInContiguousAxes(
                ary,
                tuple(normalized_indices),
                tags=tags,
                non_equality_tags=non_equality_tags,
                axes=_get_default_axes(len(array_idx_shape)
                                       + len(i_basic_indices)))
    else:
        # basic indexing expression
        return BasicIndex(ary,
                          tuple(normalized_indices),
                          tags=tags,
                          non_equality_tags=non_equality_tags,
                          axes=_get_default_axes(
                              len([idx
                                   for idx in normalized_indices
                                   if isinstance(idx, NormalizedSlice)])))

# }}}


def get_common_dtype_of_ary_or_scalars(ary_or_scalars: Sequence[ArrayOrScalar]
                                       ) -> _dtype_any:
    array_types: list[_dtype_any] = []
    scalars: list[Scalar] = []

    for ary_or_scalar in ary_or_scalars:
        if isinstance(ary_or_scalar, Array):
            array_types.append(ary_or_scalar.dtype)
        else:
            assert prim.is_arithmetic_expression(ary_or_scalar)
            scalars.append(ary_or_scalar)

    return np.result_type(*array_types, *scalars)


def get_einsum_subscript_str(expr: Einsum) -> str:
    """
    Returns the index subscript expression that can be
    used in constructing *expr* using the :func:`pytato.einsum` routine.

    Deprecated: use get_einsum_specification_str instead.

    .. testsetup::

        >>> import pytato as pt
        >>> import numpy as np
        >>> from pytato.utils import get_einsum_subscript_str

    .. doctest::

        >>> A = pt.make_placeholder("A", (10, 6), np.float64)
        >>> B = pt.make_placeholder("B", (6, 5), np.float64)
        >>> C = pt.make_placeholder("C", (5, 4), np.float64)
        >>> ABC = pt.einsum("ij,jk,kl->il", A, B, C)
        >>> get_einsum_subscript_str(ABC)
        'ij,jk,kl->il'
    """
    from warnings import warn

    warn("get_einsum_subscript_str has been deprecated and will be removed in "
         " Oct 2024. Use get_einsum_specification instead.",
         DeprecationWarning, stacklevel=2)

    return get_einsum_specification(expr)


def get_einsum_specification(expr: Einsum) -> str:
    """
    Returns the index subscript expression that can be
    used in constructing *expr* using the :func:`pytato.einsum` routine.

    Note this function may not return the exact same string as the
    string you input as part of a call to :func:`pytato.einsum'.
    Instead you will get a canonical version of the specification
    starting the indices with the letter 'i'.


    .. testsetup::

        >>> import pytato as pt
        >>> import numpy as np
        >>> from pytato.utils import get_einsum_subscript_str

    .. doctest::

        >>> A = pt.make_placeholder("A", (10, 6), np.float64)
        >>> B = pt.make_placeholder("B", (6, 5), np.float64)
        >>> C = pt.make_placeholder("C", (5, 4), np.float64)
        >>> ABC = pt.einsum("ab,bc,cd->ad", A, B, C)
        >>> get_einsum_subscript_str(ABC)
        'ij,jk,kl->il'
    """

    from pytato.array import EinsumAxisDescriptor, EinsumElementwiseAxis

    index_letters = (chr(i) for i in range(ord("i"), ord("z")))
    axis_descr_to_idx: dict[EinsumAxisDescriptor, str] = {}
    input_specs = []
    for access_descr in expr.access_descriptors:
        spec = ""
        for axis_descr in access_descr:
            try:
                spec += axis_descr_to_idx[axis_descr]
            except KeyError:
                axis_descr_to_idx[axis_descr] = next(index_letters)
                spec += axis_descr_to_idx[axis_descr]

        input_specs.append(spec)

    output_spec = "".join(axis_descr_to_idx[EinsumElementwiseAxis(i)]
                          for i in range(expr.ndim))

    return f"{','.join(input_specs)}->{output_spec}"
# vim: fdm=marker
