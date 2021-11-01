from typing import Any, Dict, Optional
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
    def __init__(self, np: Any, placeholders):
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
        return self.np.transpose(self.rec(expr.array), expr.axes)

    def map_reshape(self, expr: Reshape) -> Any:
        return self.np.reshape(self.rec(expr.array), expr.newshape, expr.order)

    def map_concatenate(self, expr: Concatenate) -> Any:
        arrays = [self.rec(array) for array in expr.arrays]
        return self.np.concatenate(arrays, expr.axis)


def assert_allclose_to_numpy(expr: Array, queue: cl.CommandQueue,
                              parameters: Optional[Dict[Placeholder, Any]] = None,
                              rtol=1e-7):
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

    np.testing.assert_allclose(np_result, pt_result, rtol=rtol)

# }}}


# {{{ random DAG generation

class RandomDAGContext:
    def __init__(self, rng, axis_len, use_numpy):
        self.rng = rng
        self.axis_len = axis_len
        self.past_results = []
        self.use_numpy = use_numpy

        if self.use_numpy:
            self.np = np
        else:
            self.np = pt


def make_random_array(rdagc: RandomDAGContext, naxes: int, axis_len=None):
    shape = (rdagc.axis_len,) * naxes

    result = rdagc.rng.uniform(1e-3, 1, size=shape)
    if rdagc.use_numpy:
        return result
    else:
        return pt.make_data_wrapper(result)


def make_random_reshape(rdagc, s, shape_len):
    rng = rdagc.rng

    s = list(s)
    naxes = rng.integers(len(s), len(s)+2)
    while len(s) < naxes:
        insert_at = rng.integers(0, len(s)+1)
        s.insert(insert_at, 1)

    return tuple(s)


_BINOPS = [operator.add, operator.sub, operator.mul, operator.truediv,
        operator.pow, "maximum", "minimum"]


def make_random_dag_inner(rdagc):
    rng = rdagc.rng

    while True:
        v = rng.integers(0, 1500)

        if v < 600:
            return make_random_array(rdagc, naxes=rng.integers(1, 3))

        elif v < 1000:
            op1 = make_random_dag(rdagc)
            op2 = make_random_dag(rdagc)
            m = min(len(op1.shape), len(op2.shape))
            naxes = rng.integers(m, m+2)
            op1 = op1.reshape(*make_random_reshape(rdagc, op1.shape, naxes))
            op2 = op2.reshape(*make_random_reshape(rdagc, op2.shape, naxes))

            which_op = rng.choice(_BINOPS)

            if which_op is operator.pow:
                op1 = abs(op1)

            if isinstance(which_op, str):
                return rdagc.np.squeeze(getattr(rdagc.np, which_op)(op1, op2))
            else:
                return rdagc.np.squeeze(which_op(op1, op2))

        elif v < 1075:
            op1 = make_random_dag(rdagc)
            op2 = make_random_dag(rdagc)
            if len(op1.shape) <= 1 and len(op2.shape) <= 1:
                continue

            return op1 @ op2

        elif v < 1275:
            if not rdagc.past_results:
                continue
            return rdagc.past_results[rng.integers(0, len(rdagc.past_results))]

        elif v < 1500:
            result = make_random_dag(rdagc)
            return rdagc.np.transpose(result,
                    tuple(rng.permuted(list(range(len(result.shape))))))

        else:
            raise AssertionError()

    # FIXME: include Stack
    # FIXME: include comparisons/booleans
    # FIXME: include <<, >>


def make_random_dag(rdagc=None):
    if not rdagc:
        from numpy.random import default_rng
        rdagc = RandomDAGContext(default_rng(), 2)
    rng = rdagc.rng
    result = make_random_dag_inner(rdagc)

    if len(result.shape) > 2:

        # FIXME Enable this to provoke reduction errors
        #v = rng.integers(0, 2)
        v = rng.integers(0, 1)
        if v == 0:
            # index away an axis
            subscript = [slice(None)] * len(result.shape)
            subscript[rng.integers(0, len(result.shape))] = \
                    int(rng.integers(0, rdagc.axis_len))

            return result[tuple(subscript)]

        elif v == 1:
            # reduce away an axis

            # FIXME do reductions other than sum?
            return rdagc.np.sum(result, axis=int(rng.integers(0, len(result.shape))))

        else:
            raise AssertionError()

    rdagc.past_results.append(result)

    return result

# }}}

# vim: foldmethod=marker
