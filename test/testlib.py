from typing import (Any, Dict)
import pyopencl as cl
import numpy
import pytato as pt
from pytato.transform import Mapper
from pytato.array import (Array, Placeholder, MatrixProduct, Stack, Roll,
                          AxisPermutation, Slice, DataWrapper, Concatenate,
                          Namespace)


class EagerEvaluator(Mapper):
    """
    Mapper to return the result according to an eager evaluation array package
    *np*.
    """
    def __init__(self, np: Any, namespace: Namespace, placeholders):
        self.np = np
        self.namespace = namespace
        self.placeholders = placeholders
        super().__init__()

    def map_placeholder(self, expr: Placeholder) -> Any:
        return self.placeholders[expr]

    def map_data_wrapper(self, expr: DataWrapper) -> Any:
        return self.namespace[expr.name].data

    def map_matrix_product(self, expr: MatrixProduct) -> Any:
        return self.np.dot(self.rec(expr.x1), self.rec(expr.x2))

    def map_stack(self, expr: Stack) -> Any:
        arrays = [self.rec(array) for array in expr.arrays]
        return self.np.stack(arrays, expr.axis)

    def map_roll(self, expr: Roll) -> Any:
        return self.np.roll(self.rec(expr.array), expr.shift, expr.axis)

    def map_axis_permutation(self, expr: AxisPermutation) -> Any:
        return self.np.transpose(self.rec(expr.array), expr.axes)

    def map_slice(self, expr: Slice) -> Any:
        array = self.rec(expr.array)
        return array[tuple(slice(start, stop)
                           for start, stop in zip(expr.starts, expr.stops))]

    def map_concatenate(self, expr: Concatenate) -> Any:
        arrays = [self.rec(array) for array in expr.arrays]
        return self.np.concatenate(arrays, expr.axis)


def assert_allclose_to_numpy(expr: Array, queue: cl.CommandQueue,
                              parameters: Dict[Placeholder, Any] = {}):
    """
    Raises an :class:`AssertionError`, if there is a discrepancy between *expr*
    evaluated lazily via :mod:`pytato` and eagerly via :mod:`numpy`.

    :arg queue: An instance of :class:`pyopencl.CommandQueue` to which the
        generated kernel must be enqueued.
    """
    np_result = EagerEvaluator(numpy, expr.namespace, parameters)(expr)
    prog = pt.generate_loopy(expr, target=pt.PyOpenCLTarget(queue))

    evt, (pt_result,) = prog(**{placeholder.name: data
                                for placeholder, data in parameters.items()})

    assert pt_result.shape == np_result.shape
    assert pt_result.dtype == np_result.dtype

    numpy.testing.assert_allclose(np_result, pt_result)
