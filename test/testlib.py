import types
from typing import Any, Dict, Optional, List, Tuple, Union
import operator
import pyopencl as cl
import numpy as np
import pytato as pt
from pytato.transform import Mapper
from pytato.array import (Array, Placeholder, MatrixProduct, Stack, Roll,
                          AxisPermutation, DataWrapper, Reshape,
                          Concatenate)


# {{{ tools for comparison to numpy

class NumpyBasedEvaluator(Mapper):
    """
    Mapper to return the result according to an eager evaluation array package
    *np*.
    """
    def __init__(self, np: Any, placeholders: Dict[Placeholder, Array]) -> None:
        self.np = np
        self.placeholders = placeholders
        super().__init__()

    def map_placeholder(self, expr: Placeholder) -> Any:
        return self.placeholders[expr]

    def map_data_wrapper(self, expr: DataWrapper) -> Any:
        return expr.data

    def map_matrix_product(self, expr: MatrixProduct) -> Any:
        return self.np.dot(self.rec(expr.x1), self.rec(expr.x2))

    def map_stack(self, expr: Stack) -> Any:
        arrays = [self.rec(array) for array in expr.arrays]
        return self.np.stack(arrays, expr.axis)

    def map_roll(self, expr: Roll) -> Any:
        return self.np.roll(self.rec(expr.array), expr.shift, expr.axis)

    def map_axis_permutation(self, expr: AxisPermutation) -> Any:
        return self.np.transpose(self.rec(expr.array), expr.axis_permutation)

    def map_reshape(self, expr: Reshape) -> Any:
        return self.np.reshape(self.rec(expr.array), expr.newshape, expr.order)

    def map_concatenate(self, expr: Concatenate) -> Any:
        arrays = [self.rec(array) for array in expr.arrays]
        return self.np.concatenate(arrays, expr.axis)


def assert_allclose_to_numpy(expr: Array, queue: cl.CommandQueue,
                              parameters: Optional[Dict[Placeholder, Any]] = None,
                              rtol: float = 1e-7) -> None:
    """
    Raises an :class:`AssertionError`, if there is a discrepancy between *expr*
    evaluated lazily via :mod:`pytato` and eagerly via :mod:`numpy`.

    :arg queue: An instance of :class:`pyopencl.CommandQueue` to which the
        generated kernel must be enqueued.
    """
    if parameters is None:
        parameters = {}

    np_result = NumpyBasedEvaluator(np, parameters)(expr)
    prog = pt.generate_loopy(expr, cl_device=queue.device)

    evt, (pt_result,) = prog(queue, **{placeholder.name: data
                                for placeholder, data in parameters.items()})

    assert pt_result.shape == np_result.shape
    assert pt_result.dtype == np_result.dtype

    np.testing.assert_allclose(np_result, pt_result, rtol=rtol)  # type: ignore[no-untyped-call]  # noqa: E501

# }}}


# {{{ random DAG generation

class RandomDAGContext:
    def __init__(self, rng: np.random.Generator, axis_len: int, use_numpy: bool) \
            -> None:
        self.rng = rng
        self.axis_len = axis_len
        self.past_results: List[Array] = []
        self.use_numpy = use_numpy

        if self.use_numpy:
            self.np: types.ModuleType = np
        else:
            self.np = pt


def make_random_constant(rdagc: RandomDAGContext, naxes: int) -> Any:
    shape = (rdagc.axis_len,) * naxes

    result = rdagc.rng.uniform(1e-3, 1, size=shape)
    if rdagc.use_numpy:
        return result
    else:
        return pt.make_data_wrapper(result)


def make_random_reshape(
        rdagc: RandomDAGContext, s: Tuple[int, ...], shape_len: int) \
        -> Tuple[int, ...]:
    rng = rdagc.rng

    s_list = list(s)
    naxes = rng.integers(len(s), len(s)+2)
    while len(s_list) < naxes:
        insert_at = rng.integers(0, len(s)+1)
        s_list.insert(insert_at, 1)

    return tuple(s_list)


_BINOPS = [operator.add, operator.sub, operator.mul, operator.truediv,
        operator.pow, "maximum", "minimum"]


def make_random_dag_inner(rdagc: RandomDAGContext) -> Any:
    rng = rdagc.rng

    while True:
        v = rng.integers(0, 1500)

        if v < 600:
            return make_random_constant(rdagc, naxes=rng.integers(1, 3))

        elif v < 1000:
            op1 = make_random_dag(rdagc)
            op2 = make_random_dag(rdagc)
            m = min(op1.ndim, op2.ndim)
            naxes = rng.integers(m, m+2)

            # Introduce a few new 1-long axes to test broadcasting.
            op1 = op1.reshape(*make_random_reshape(rdagc, op1.shape, naxes))
            op2 = op2.reshape(*make_random_reshape(rdagc, op2.shape, naxes))

            # type ignore because rng.choice doesn't have broad enough type
            # annotation to represent choosing callables.
            which_op = rng.choice(_BINOPS)  # type: ignore[arg-type]

            if which_op is operator.pow:
                op1 = abs(op1)

            # Squeeze because all axes need to be of rdagc.axis_len, and we've
            # just inserted a few new 1-long axes. Those need to go before we
            # return.
            if which_op in ["maximum", "minimum"]:
                # type ignore because we haven't told mypy what's in rdagc.np.
                return rdagc.np.squeeze(  # type: ignore[attr-defined]
                        getattr(rdagc.np, which_op)(op1, op2))
            else:
                # type ignore because we haven't told mypy what's in rdagc.np.
                return rdagc.np.squeeze(  # type: ignore[attr-defined]
                        which_op(op1, op2))

        elif v < 1075:
            op1 = make_random_dag(rdagc)
            op2 = make_random_dag(rdagc)
            if op1.ndim <= 1 and op2.ndim <= 1:
                continue

            return op1 @ op2

        elif v < 1275:
            if not rdagc.past_results:
                continue
            return rdagc.past_results[rng.integers(0, len(rdagc.past_results))]

        elif v < 1500:
            result = make_random_dag(rdagc)
            # type ignore because we haven't told mypy what's in rdagc.np.
            return rdagc.np.transpose(  # type: ignore[attr-defined]  # noqa: E501
                    result,
                    tuple(rng.permuted(list(range(result.ndim)))))

        else:
            raise AssertionError()

    # FIXME: include Stack
    # FIXME: include comparisons/booleans
    # FIXME: include <<, >>
    # FIXME: include integer computations
    # FIXME: include DictOfNamedArrays


def make_random_dag(rdagc: RandomDAGContext) -> Any:
    """Return a :class:`pytato.Array` or a :class:`numpy.ndarray`
    (cf. :attr:`RandomDAGContext.use_numpy`) that is the result of a random
    (cf. :attr:`RandomDAGContext.rng`) array computation. All axes
    of the array are of length :attr:`RandomDAGContext.axis_len` (there is
    at least one axis, but arbitrarily more may be present).
    """
    rng = rdagc.rng
    result = make_random_dag_inner(rdagc)

    if result.ndim > 2:
        # FIXME Enable this to provoke reduction errors
        #v = rng.integers(0, 2)
        v = rng.integers(0, 1)
        if v == 0:
            # index away an axis
            subscript: List[Union[int, slice]] = [slice(None)] * result.ndim
            subscript[rng.integers(0, result.ndim)] = int(
                    rng.integers(0, rdagc.axis_len))

            return result[tuple(subscript)]

        elif v == 1:
            # reduce away an axis

            # FIXME do reductions other than sum?
            # type ignore because we haven't told mypy what's in rdagc.np.
            return rdagc.np.sum(  # type: ignore[attr-defined]
                    result, axis=int(rng.integers(0, result.ndim)))

        else:
            raise AssertionError()

    rdagc.past_results.append(result)

    return result

# }}}

# vim: foldmethod=marker
