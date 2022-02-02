from __future__ import annotations

import types
from typing import Any, Dict, Optional, List, Tuple, Union, Sequence, Callable
import operator
import pyopencl as cl
import numpy as np
import pytato as pt
from pytato.transform import Mapper
from pytato.array import (Array, Placeholder, Stack, Roll,
                          AxisPermutation, DataWrapper, Reshape,
                          Concatenate)
from pytools.tag import Tag


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
    prog = pt.generate_loopy(expr)

    evt, (pt_result,) = prog(queue, **{placeholder.name: data
                                for placeholder, data in parameters.items()})

    assert pt_result.shape == np_result.shape
    assert pt_result.dtype == np_result.dtype

    np.testing.assert_allclose(np_result, pt_result, rtol=rtol)  # noqa: E501

# }}}


# {{{ random DAG generation

class RandomDAGContext:
    def __init__(
            self, rng: np.random.Generator, axis_len: int, use_numpy: bool,
            additional_generators: Optional[Sequence[
                Tuple[int, Callable[[RandomDAGContext, int],
                    Tuple[Array, int]]]]] = None
            ) -> None:
        """
        :param additional_generators: A sequence of tuples
            ``(fake_probability, gen_func)``, where *fake_probability* is
            an integer of magnitude ~100 and *gen_func* is a generation function
            that will be called with the :class:`RandomDAGContext` as an argument.
        """
        self.rng = rng
        self.axis_len = axis_len
        self.past_results: List[Array] = []
        self.use_numpy = use_numpy

        if additional_generators is None:
            additional_generators = []

        self.additional_generators = additional_generators

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


def make_random_binary_op(rdagc: RandomDAGContext,
        op1: Any, *, size: int) -> Tuple[Any, int]:
    rng = rdagc.rng

    op2, op2_size = make_random_dag_rec(rdagc, size=size - 1)
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
        result = getattr(rdagc.np, which_op)(op1, op2)
    else:
        result = which_op(op1, op2)

    return rdagc.np.squeeze(result), op2_size + 1


def make_random_dag_choice(rdagc: RandomDAGContext, *, size: int) -> Tuple[Any, int]:
    rng = rdagc.rng

    max_prob_hardcoded = 1500
    additional_prob = sum(
            fake_prob
            for fake_prob, _func in rdagc.additional_generators)

    if size <= 1:
        return make_random_constant(rdagc, naxes=rng.integers(1, 3)), 1

    while True:
        v = rng.integers(0, max_prob_hardcoded + additional_prob)

        if v < 600:
            return make_random_constant(rdagc, naxes=rng.integers(1, 3)), 1

        elif v < 1000:
            op1, op1_size = make_random_dag_rec(rdagc, size=size)

            if op1_size > size:
                continue

            result, op2_and_root_size = make_random_binary_op(
                    rdagc, op1, size=size - op1_size)

            return result, op1_size + op2_and_root_size

        elif v < 1075:
            op1, op1_size = make_random_dag_rec(rdagc, size=size)
            op2, op2_size = make_random_dag_rec(rdagc, size=size-op1_size-1)
            if op1.ndim <= 1 and op2.ndim <= 1:
                continue

            return op1 @ op2, op1_size + 1 + op2_size

        elif v < 1275:
            if not rdagc.past_results:
                continue

            return rdagc.past_results[rng.integers(0, len(rdagc.past_results))], 0

        elif v < max_prob_hardcoded:
            result, res_size = make_random_dag_rec(rdagc, size=size-1)
            return rdagc.np.transpose(
                    result,
                    tuple(rng.permuted(list(range(result.ndim))))), res_size + 1

        else:
            base_prob = max_prob_hardcoded
            for fake_prob, gen_func in rdagc.additional_generators:
                if base_prob <= v < base_prob + fake_prob:
                    return gen_func(rdagc, size)

                base_prob += fake_prob

            # should never get here
            raise AssertionError()

    # FIXME: include Stack
    # FIXME: include comparisons/booleans
    # FIXME: include <<, >>
    # FIXME: include integer computations
    # FIXME: include DictOfNamedArrays


def make_random_dag_rec(rdagc: RandomDAGContext, *, size: int) -> Tuple[Any, int]:
    rng = rdagc.rng
    result, actual_size = make_random_dag_choice(rdagc, size=size)

    if result.ndim > 2:
        v = rng.integers(0, 2)
        if v == 0:
            # index away an axis
            subscript: List[Union[int, slice]] = [slice(None)] * result.ndim
            subscript[rng.integers(0, result.ndim)] = int(
                    rng.integers(0, rdagc.axis_len))

            return result[tuple(subscript)], actual_size+1

        elif v == 1:
            # reduce away an axis

            # FIXME do reductions other than sum?
            return rdagc.np.sum(
                    result, axis=int(rng.integers(0, result.ndim))), actual_size+1

        else:
            raise AssertionError()

    rdagc.past_results.append(result)

    return result, actual_size


def make_random_dag(rdagc: RandomDAGContext, *, size: int) -> Tuple[Any, int]:
    """Return a :class:`pytato.Array` or a :class:`numpy.ndarray`
    (cf. :attr:`RandomDAGContext.use_numpy`) that is the result of a random
    (cf. :attr:`RandomDAGContext.rng`) array computation. All axes
    of the array are of length :attr:`RandomDAGContext.axis_len` (there is
    at least one axis, but arbitrarily more may be present).
    """
    result, actual_size = make_random_dag_rec(rdagc, size=size)

    while actual_size < size:
        result, op2_and_root_size = make_random_binary_op(
                rdagc, result, size=size - actual_size)

        actual_size = actual_size + op2_and_root_size

    return result, actual_size

# }}}


# {{{ tags used only by the regression tests

class FooInameTag(Tag):
    """
    foo
    """


class BarInameTag(Tag):
    """
    bar
    """


class BazInameTag(Tag):
    """
    baz
    """


class FooTag(Tag):
    """
    foo
    """


class BarTag(Tag):
    """
    bar
    """


class BazTag(Tag):
    """
    baz
    """

# }}}

# vim: foldmethod=marker
