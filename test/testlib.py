from typing import Any, Dict, Optional
import operator
import pyopencl as cl
import numpy
import pytato as pt
from pytato.transform import Mapper
from pytato.array import (Array, Placeholder, MatrixProduct, Stack, Roll,
                          AxisPermutation, DataWrapper, Reshape,
                          Concatenate)


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

    np_result = NumpyBasedEvaluator(numpy, parameters)(expr)
    prog = pt.generate_loopy(expr, cl_device=queue.device)

    evt, (pt_result,) = prog(queue, **{placeholder.name: data
                                for placeholder, data in parameters.items()})

    assert pt_result.shape == np_result.shape
    assert pt_result.dtype == np_result.dtype

    numpy.testing.assert_allclose(np_result, pt_result, rtol=rtol)


# {{{ random DAG generation

class RandomDAGContext:
    def __init__(self, rng, axis_len):
        self.rng = rng
        self.axis_len = axis_len
        self.past_results = []


def make_random_array(rdagc: RandomDAGContext, naxes: int, axis_len=None):
    shape = (rdagc.axis_len,) * naxes

    return pt.make_data_wrapper(rdagc.rng.normal(size=shape))


def make_random_reshape(rdagc, s, shape_len):
    rng = rdagc.rng

    s = list(s)
    naxes = rng.integers(len(s), len(s)+2)
    while len(s) < naxes:
        insert_at = rng.integers(0, len(s)+1)
        s.insert(insert_at, 1)

    return tuple(s)


_BINOPS = [operator.add, operator.sub, operator.mul, operator.truediv,
        operator.pow, operator.floordiv, "maximum", "minimum"]


def make_random_dag_inner(rdagc):
    rng = rdagc.rng

    v = rng.integers(0, 1500)

    if v < 500:
        return make_random_array(rdagc, naxes=rng.integers(1, 3))

    elif v < 1000:
        op1 = make_random_dag(rdagc)
        op2 = make_random_dag(rdagc)
        # m = min(len(op1.shape), len(op2.shape))
        # naxes = rng.integers(m, m+2)
        # op1 = op1.reshape(*make_random_reshape(rdagc, op1.shape, naxes))
        # op2 = op2.reshape(*make_random_reshape(rdagc, op1.shape, naxes))

        which_op = rng.choice(_BINOPS)

        if isinstance(which_op, str):
            return pt.squeeze(getattr(pt, which_op)(op1, op2))
        else:
            return pt.squeeze(which_op(op1, op2))

    elif v < 1075:
        return make_random_dag(rdagc) @ make_random_dag(rdagc)

    # elif v < 1275:
    #     ul = len(rdagc.past_results)
    #     return rdagc.past_results[rng.integers(0, ul) if ul > 0 else 1]

    elif v < 1500:
        result = make_random_dag(rdagc)
        return pt.transpose(result,
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
        v = rng.integers(0, 2)
        if v == 0:
            # reduce away an axis

            # FIXME do reductions other than sum?
            return pt.sum(result, axis=rng.integers(0, len(result.shape)))

        elif v == 1:
            # index away an axis
            subscript = [slice()] * len(result.shape)
            subscript[rng.integers(0, len(result.shape))] = \
                    rng.integers(0, rdagc.axis_len)

            return result[tuple(subscript)]

        else:
            raise AssertionError()

    return result

# }}}

# vim: foldmethod=marker
