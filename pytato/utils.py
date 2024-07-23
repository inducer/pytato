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

import numpy as np
import islpy as isl
import pymbolic.primitives as prim

from typing import (Tuple, List, Union, Callable, Any, Sequence, Dict,
                    Optional, Iterable, TypeVar, FrozenSet)
from pytato.array import (Array, ShapeType, IndexLambda, SizeParam, ShapeComponent,
                          DtypeOrPyScalarType, ArrayOrScalar, BasicIndex,
                          AdvancedIndexInContiguousAxes,
                          AdvancedIndexInNoncontiguousAxes,
                          ConvertibleToIndexExpr, IndexExpr, NormalizedSlice,
                          _dtype_any, Einsum)
from pytato.scalar_expr import (
    PYTHON_SCALAR_CLASSES, ScalarExpression, IntegralScalarExpression,
    SCALAR_CLASSES, INT_CLASSES, BoolT, Scalar, TypeCast)
from pytools import UniqueNameGenerator
from pytato.transform import Mapper
from pytools.tag import Tag
from immutabledict import immutabledict


__doc__ = """
Helper routines
---------------

.. autofunction:: are_shape_components_equal
.. autofunction:: are_shapes_equal
.. autofunction:: get_shape_after_broadcasting
.. autofunction:: dim_to_index_lambda_components
.. autofunction:: get_common_dtype_of_ary_or_scalars
.. autofunction:: get_einsum_subscript_str
"""


# {{{ partition

Tpart = TypeVar("Tpart")


def partition(pred: Callable[[Tpart], bool],
              iterable: Iterable[Tpart]) -> Tuple[List[Tpart],
                                                  List[Tpart]]:
    """
    Use a predicate to partition entries into false entries and true
    entries
    """
    # Inspired from https://docs.python.org/3/library/itertools.html
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    from itertools import tee, filterfalse
    t1, t2 = tee(iterable)
    return list(filterfalse(pred, t1)), list(filter(pred, t2))

# }}}


def get_shape_after_broadcasting(
        exprs: Iterable[Union[Array, ScalarExpression]]) -> ShapeType:
    """
    Returns the shape after broadcasting *exprs* in an operation.
    """
    from pytato.diagnostic import CannotBroadcastError
    shapes = [expr.shape if isinstance(expr, Array) else () for expr in exprs]

    result_dim = max((len(s) for s in shapes), default=0)

    # append leading dimensions of all the shapes with 1's to match result_dim.
    augmented_shapes = [((1,)*(result_dim-len(s)) + s) for s in shapes]

    def _get_result_axis_length(axis_lengths: List[IntegralScalarExpression]
                                ) -> IntegralScalarExpression:
        result_axis_len = axis_lengths[0]
        for axis_len in axis_lengths[1:]:
            if are_shape_components_equal(axis_len, result_axis_len):
                pass
            elif are_shape_components_equal(axis_len, 1):
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
                            result_shape: ShapeType) -> Tuple[ScalarExpression, ...]:
    """
    Returns the indices while broadcasting an array of shape *shape* into one of
    shape *result_shape*.
    """
    assert len(shape) <= len(result_shape)
    i_start = len(result_shape) - len(shape)
    indices = []
    for i, (dim1, dim2) in enumerate(zip(shape, result_shape[i_start:])):
        if not are_shape_components_equal(dim1, dim2):
            assert are_shape_components_equal(dim1, 1)
            indices.append(0)
        else:
            indices.append(prim.Variable(f"_{i+i_start}"))

    return tuple(indices)


def with_indices_for_broadcasted_shape(val: prim.Variable, shape: ShapeType,
                                       result_shape: ShapeType) -> prim.Expression:
    if len(shape) == 0:
        # scalar expr => do not index
        return val
    else:
        return val[get_indexing_expression(shape, result_shape)]


def _extract_dtypes(
        exprs: Sequence[ArrayOrScalar]) -> List[DtypeOrPyScalarType]:
    dtypes: List[DtypeOrPyScalarType] = []
    for expr in exprs:
        if isinstance(expr, Array):
            dtypes.append(expr.dtype)
        elif isinstance(expr, np.generic):
            dtypes.append(expr.dtype)
        elif isinstance(expr, PYTHON_SCALAR_CLASSES):
            dtypes.append(type(expr))
        else:
            raise TypeError(f"unexpected expression type: '{type(expr)}'")

    return dtypes


def update_bindings_and_get_broadcasted_expr(arr: ArrayOrScalar,
                                             bnd_name: str,
                                             bindings: Dict[str, Array],
                                             result_shape: ShapeType
                                             ) -> ScalarExpression:
    """
    Returns an instance of :class:`~pytato.scalar_expr.ScalarExpression` to address
    *arr* in a :class:`pytato.array.IndexLambda` of shape *result_shape*.
    """

    if isinstance(arr, SCALAR_CLASSES):
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
                        get_result_type: Callable[[DtypeOrPyScalarType, DtypeOrPyScalarType], np.dtype[Any]],  # noqa:E501
                        *,
                        tags: FrozenSet[Tag],
                        non_equality_tags: FrozenSet[Tag],
                        cast_to_result_dtype: bool,
                        ) -> ArrayOrScalar:
    from pytato.array import _get_default_axes

    if np.isscalar(a1) and np.isscalar(a2):
        from pytato.scalar_expr import evaluate
        return evaluate(op(a1, a2))  # type: ignore

    result_shape = get_shape_after_broadcasting([a1, a2])

    dtypes = _extract_dtypes([a1, a2])
    result_dtype = get_result_type(*dtypes)

    bindings: Dict[str, Array] = {}

    expr1 = update_bindings_and_get_broadcasted_expr(a1, "_in0", bindings,
                                                     result_shape)
    expr2 = update_bindings_and_get_broadcasted_expr(a2, "_in1", bindings,
                                                     result_shape)

    def cast_to_result_type(
                array: ArrayOrScalar,
                expr: ScalarExpression
            ) -> ScalarExpression:
        if ((isinstance(array, Array) or isinstance(array, np.generic))
                and array.dtype != result_dtype):
            # Loopy's type casts don't like casting to bool
            assert result_dtype != np.bool_

            expr = TypeCast(result_dtype, expr)
        elif isinstance(expr, SCALAR_CLASSES):
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

class ShapeExpressionMapper(Mapper):
    """
    Mapper that takes a shape component and returns it as a scalar expression.
    """
    def __init__(self, var_name_gen: UniqueNameGenerator):
        super().__init__()
        self.cache: Dict[Array, ScalarExpression] = {}
        self.var_name_gen = var_name_gen
        self.bindings: Dict[str, SizeParam] = {}

    def rec(self, expr: Array) -> ScalarExpression:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: Array = super().rec(expr)
        self.cache[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda) -> ScalarExpression:
        from pytato.scalar_expr import substitute
        return substitute(expr.expr, {name: self.rec(val)
                                      for name, val in expr.bindings.items()})

    def map_size_param(self, expr: SizeParam) -> ScalarExpression:
        name = self.var_name_gen("_in")
        self.bindings[name] = expr
        return prim.Variable(name)


def dim_to_index_lambda_components(expr: ShapeComponent,
                                   vng: Optional[UniqueNameGenerator] = None,
                                   ) -> Tuple[ScalarExpression,
                                              Dict[str, SizeParam]]:
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


def are_shape_components_equal(dim1: ShapeComponent, dim2: ShapeComponent) -> bool:
    """
    Returns *True* iff *dim1* and *dim2* are have equal
    :class:`~pytato.array.SizeParam` coefficients in their expressions.
    """
    if isinstance(dim1, INT_CLASSES) and isinstance(dim2, INT_CLASSES):
        return dim1 == dim2

    dim1_minus_dim2 = dim1 - dim2
    assert isinstance(dim1_minus_dim2, Array)

    from pytato.transform import InputGatherer

    # type-ignore reason: not all InputArgumentBase have a name attr.
    space = _create_size_param_space({expr.name  # type: ignore[attr-defined]
                                      for expr in InputGatherer()(dim1_minus_dim2)})

    # pytato requires the shape expressions be affine expressions of the size
    # params, so converting them to ISL expressions.
    aff = ShapeToISLExpressionMapper(space)(dim1_minus_dim2)
    return (aff.is_cst()  # type: ignore[no-any-return]
            and aff.get_constant_val().is_zero())


def are_shapes_equal(shape1: ShapeType, shape2: ShapeType) -> bool:
    """
    Returns *True* iff *shape1* and *shape2* have the same dimensionality and the
    correpsonding components are equal as defined by
    :func:`~pytato.utils.are_shape_components_equal`.
    """
    return ((len(shape1) == len(shape2))
            and all(are_shape_components_equal(dim1, dim2)
                    for dim1, dim2 in zip(shape1, shape2)))


# {{{ ShapeToISLExpressionMapper

class ShapeToISLExpressionMapper(Mapper):
    """
    Mapper that takes a shape component and returns it as :class:`isl.Aff`.
    """
    def __init__(self, space: isl.Space):
        super().__init__()
        self.cache: Dict[Array, isl.Aff] = {}
        self.space = space

    # type-ignore reason: incompatible return type with super class
    def rec(self, expr: Array) -> isl.Aff:  # type: ignore[override]
        if expr in self.cache:
            return self.cache[expr]
        result: Array = super().rec(expr)
        self.cache[expr] = result
        return result

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


def _is_non_negative(expr: ShapeComponent) -> BoolT:
    """
    Returns *True* iff it can be proven that ``expr >= 0``.
    """
    if isinstance(expr, INT_CLASSES):
        return expr >= 0

    assert isinstance(expr, Array) and expr.shape == ()
    from pytato.transform import InputGatherer
    # type-ignore reason: passed Set[Optional[str]]; function expects Set[str]
    space = _create_size_param_space({expr.name  # type: ignore
                                      for expr in InputGatherer()(expr)})
    aff = ShapeToISLExpressionMapper(space)(expr)
    # type-ignore reason: mypy doesn't know comparing isl.Sets returns bool
    return (aff.ge_set(aff * 0)  # type: ignore[no-any-return]
            >= _get_size_params_assumptions_bset(space))


def _is_non_positive(expr: ShapeComponent) -> BoolT:
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
                if step > 0:
                    start = axis_len
                else:
                    start = axis_len - 1
            else:
                if step > 0:
                    start = 0
                else:
                    start = -1
        else:
            raise NotImplementedError

    if stop is None:
        stop = default_stop
    else:
        if isinstance(axis_len, INT_CLASSES):
            if -axis_len <= stop < axis_len:
                stop = stop % axis_len
            elif stop >= axis_len:
                if step > 0:
                    stop = axis_len
                else:
                    stop = axis_len - 1
            else:
                if step > 0:
                    stop = 0
                else:
                    stop = -1
        else:
            raise NotImplementedError

    return NormalizedSlice(start, stop, step)


def _normalized_slice_len(slice_: NormalizedSlice) -> ShapeComponent:
    start, stop, step = slice_.start, slice_.stop, slice_.step

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

# }}}


def _index_into(
        ary: Array,
        indices: Tuple[ConvertibleToIndexExpr, ...],
        tags: FrozenSet[Tag],
        non_equality_tags: FrozenSet[Tag]) -> Array:
    from pytato.diagnostic import CannotBroadcastError
    from pytato.array import _get_default_axes

    # {{{ handle ellipsis

    if indices.count(...) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    if indices.count(...):
        ellipsis_pos = indices.index(...)
        indices = (indices[:ellipsis_pos]
                   + (slice(None, None, None),) * (ary.ndim - len(indices) + 1)
                   + indices[ellipsis_pos+1:])

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
                raise IndexError("Indirect indexing into a non-postive"
                                 f" dimension (axis {i}) is illegal.")
        else:
            raise IndexError("only integers, slices, ellipsis and integer arrays"
                             " are valid indices")

    # }}}

    # {{{ normalize slices

    normalized_indices: List[IndexExpr] = [_normalize_slice(idx, axis_len)
                                           if isinstance(idx, slice)
                                           else idx
                                           for idx, axis_len in zip(indices,
                                                                    ary.shape)]

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
    array_types: List[_dtype_any] = []
    scalars: List[Scalar] = []

    for ary_or_scalar in ary_or_scalars:
        if isinstance(ary_or_scalar, Array):
            array_types.append(ary_or_scalar.dtype)
        else:
            assert isinstance(ary_or_scalar, SCALAR_CLASSES)
            scalars.append(ary_or_scalar)

    return np.result_type(*array_types, *scalars)


def get_einsum_subscript_str(expr: Einsum) -> str:
    """
    Returns the index subscript expression that can be
    used in constructing *expr* using the :func:`pytato.einsum` routine.

    Note this is not ensured to be the same string as what you entered
    when you called :func:`pytato.einsum`.


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
    from pytato.array import EinsumElementwiseAxis

    access_descr_to_index = {descr: key for key, descr
                             in expr.index_to_access_descriptor.items()}

    arg_subscripts: List[str] = []

    for input_descriptors in expr.access_descriptors:
        arg_subscripts.append("".join([access_descr_to_index[descr]
                                       for descr in input_descriptors]))

    output_subscripts = "".join([index for acc_desc, index in
                                 access_descr_to_index.items()
                                 if isinstance(acc_desc, EinsumElementwiseAxis)])

    return f"{','.join(arg_subscripts)}->{output_subscripts}"

# vim: fdm=marker
