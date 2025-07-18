from __future__ import annotations

import operator
import random
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Never

from pytools.tag import Tag

import pytato as pt
from pytato.transform import Mapper


if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Sequence

    import pyopencl as cl

    from pytato.array import (
        Array,
        AxisPermutation,
        Concatenate,
        DataWrapper,
        Placeholder,
        Reshape,
        Roll,
        Stack,
    )


# {{{ tools for comparison to numpy

class NumpyBasedEvaluator(Mapper[Any, Never, []]):
    """
    Mapper to return the result according to an eager evaluation array package
    *np*.
    """
    def __init__(self, np: Any, placeholders: dict[Placeholder, Array]) -> None:
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
        return self.np.reshape(self.rec(expr.array), expr.newshape, order=expr.order)

    def map_concatenate(self, expr: Concatenate) -> Any:
        arrays = [self.rec(array) for array in expr.arrays]
        return self.np.concatenate(arrays, expr.axis)


def assert_allclose_to_numpy(expr: Array, queue: cl.CommandQueue,
                              parameters: dict[Placeholder, Any] | None = None,
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

    _evt, (pt_result,) = prog(queue, **{placeholder.name: data
                                for placeholder, data in parameters.items()})

    assert pt_result.shape == np_result.shape
    assert pt_result.dtype == np_result.dtype

    np.testing.assert_allclose(np_result, pt_result, rtol=rtol)

# }}}


# {{{ random DAG generation

class RandomDAGContext:
    def __init__(
                 self,
                 rng: np.random.Generator,
                 axis_len: int,
                 use_numpy: bool,
                 allow_duplicate_nodes: bool = False,
                 additional_generators: (
                     Sequence[tuple[int, Callable[[RandomDAGContext], Array]]]
                         | None) = None
             ) -> None:
        """
        :param additional_generators: A sequence of tuples
            ``(fake_probability, gen_func)``, where *fake_probability* is
            an integer of magnitude ~100 and *gen_func* is a generation function
            that will be called with the :class:`RandomDAGContext` as an argument.
        """
        self.rng = rng
        self.axis_len = axis_len
        self.past_results: list[Array] = []
        self.use_numpy = use_numpy
        self.allow_duplicate_nodes = allow_duplicate_nodes

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
        rdagc: RandomDAGContext, s: tuple[int, ...], shape_len: int) \
        -> tuple[int, ...]:
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
    if not rdagc.use_numpy and not rdagc.allow_duplicate_nodes:
        def dedup(expr: Array) -> Array:
            return pt.transform._verify_is_array(pt.transform.deduplicate(expr))

    else:
        def dedup(expr: Array) -> Array:
            return expr

    rng = rdagc.rng

    max_prob_hardcoded = 1500
    additional_prob = sum(
            fake_prob
            for fake_prob, _func in rdagc.additional_generators)
    while True:
        v = rng.integers(0, max_prob_hardcoded + additional_prob)

        if v < 600:
            return dedup(make_random_constant(rdagc, naxes=int(rng.integers(1, 3))))

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
                return dedup(rdagc.np.squeeze(getattr(rdagc.np, which_op)(op1, op2)))
            else:
                return dedup(rdagc.np.squeeze(which_op(op1, op2)))

        elif v < 1075:
            op1 = make_random_dag(rdagc)
            op2 = make_random_dag(rdagc)
            if op1.ndim <= 1 and op2.ndim <= 1:
                continue

            return dedup(op1 @ op2)

        elif v < 1275:
            if not rdagc.past_results:
                continue
            return dedup(
                rdagc.past_results[rng.integers(0, len(rdagc.past_results))])

        elif v < max_prob_hardcoded:
            result = make_random_dag(rdagc)
            return dedup(
                rdagc.np.transpose(
                    result,
                    tuple(rng.permuted(list(range(result.ndim))))))

        else:
            base_prob = max_prob_hardcoded
            for fake_prob, gen_func in rdagc.additional_generators:
                if base_prob <= v < base_prob + fake_prob:
                    return dedup(gen_func(rdagc))

                base_prob += fake_prob

            # should never get here
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
    if not rdagc.use_numpy and not rdagc.allow_duplicate_nodes:
        def dedup(expr: Array) -> Array:
            return pt.transform._verify_is_array(pt.transform.deduplicate(expr))

    else:
        def dedup(expr: Array) -> Array:
            return expr

    rng = rdagc.rng
    result = make_random_dag_inner(rdagc)

    if result.ndim > 2:
        v = rng.integers(0, 2)
        if v == 0:
            # index away an axis
            subscript: list[int | slice] = [slice(None)] * result.ndim
            subscript[rng.integers(0, result.ndim)] = int(
                    rng.integers(0, rdagc.axis_len))

            return dedup(result[tuple(subscript)])

        elif v == 1:
            # reduce away an axis

            # FIXME do reductions other than sum?
            return dedup(
                rdagc.np.sum(
                    result, axis=int(rng.integers(0, result.ndim))))

        else:
            raise AssertionError()

    rdagc.past_results.append(result)

    return result

# }}}


# {{{ get_random_dag_w_no_placeholders

def get_random_pt_dag(seed: int,
                      *,
                      additional_generators: (
                          Sequence[tuple[int, Callable[[RandomDAGContext], Array]]]
                              | None) = None,
                      axis_len: int = 4,
                      convert_dws_to_placeholders: bool = False,
                      allow_duplicate_nodes: bool = False
                      ) -> pt.DictOfNamedArrays:
    if additional_generators is None:
        additional_generators = []

    from testlib import RandomDAGContext, make_random_dag

    rdagc_comm = RandomDAGContext(np.random.default_rng(seed=seed),
            axis_len=axis_len, use_numpy=False,
            allow_duplicate_nodes=allow_duplicate_nodes,
            additional_generators=additional_generators)
    dag = pt.make_dict_of_named_arrays({"result": make_random_dag(rdagc_comm)})

    if convert_dws_to_placeholders:
        from pytools import UniqueNameGenerator
        vng = UniqueNameGenerator()

        def make_dws_placeholder(expr: pt.transform.ArrayOrNames
                                 ) -> pt.transform.ArrayOrNames:
            if isinstance(expr, pt.DataWrapper):
                return pt.make_placeholder(vng("_pt_ph"),
                                           expr.shape, expr.dtype)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, make_dws_placeholder)

    return dag


def get_random_pt_dag_with_send_recv_nodes(
        seed: int,
        rank: int,
        size: int,
        *,
        comm_fake_probability: int = 500,
        axis_len: int = 4,
        convert_dws_to_placeholders: bool = False
        ) -> pt.DictOfNamedArrays:
    comm_tag = 17

    def gen_comm(rdagc: RandomDAGContext) -> pt.Array:
        nonlocal comm_tag
        inner = make_random_dag(rdagc)
        comm_tag += 1
        return pt.staple_distributed_send(
                inner, dest_rank=(rank-1) % size, comm_tag=comm_tag,
                stapled_to=pt.make_distributed_recv(
                    src_rank=(rank+1) % size, comm_tag=comm_tag,
                    shape=inner.shape, dtype=inner.dtype))

    return get_random_pt_dag(
        seed=seed,
        axis_len=axis_len,
        convert_dws_to_placeholders=convert_dws_to_placeholders,
        additional_generators=[(comm_fake_probability, gen_comm)])


def make_large_dag(iterations: int, seed: int = 0) -> pt.DictOfNamedArrays:
    """
    Builds a DAG with emphasis on number of operations.
    """

    rng = np.random.default_rng(seed)
    random.seed(seed)

    a = pt.make_placeholder(name="a", shape=(2, 2), dtype=np.float64)
    current = a

    # Will randomly choose from the operators
    operations = [operator.add, operator.sub, operator.mul, operator.truediv]

    for _ in range(iterations):
        operation = random.choice(operations)
        value = rng.uniform(1, 10)
        current = operation(current, value)

    # DAG should have `iterations` number of operations
    return pt.make_dict_of_named_arrays({"result": current})


def make_small_dag_with_duplicates() -> pt.DictOfNamedArrays:
    x = pt.make_placeholder(name="x", shape=(2, 2), dtype=np.float64)

    expr1 = 2 * x
    expr2 = 2 * x

    y = expr1 + expr2

    # Has duplicates of the 2*x operation
    return pt.make_dict_of_named_arrays({"result": y})


def make_large_dag_with_duplicates(iterations: int,
                                   seed: int = 0) -> pt.DictOfNamedArrays:

    random.seed(seed)
    rng = np.random.default_rng(seed)
    a = pt.make_placeholder(name="a", shape=(2, 2), dtype=np.float64)
    current = a

    # Will randomly choose from the operators
    operations = [operator.add, operator.sub, operator.mul, operator.truediv]
    duplicates = []

    for _ in range(iterations):
        operation = random.choice(operations)
        value = rng.uniform(1, 10)
        current = operation(current, value)

        # Introduce duplicates intentionally
        if rng.uniform() > 0.2:
            dup1 = operation(a, value)
            dup2 = operation(a, value)
            duplicates.append(dup1)
            duplicates.append(dup2)
            current = operation(current, dup1)

    all_exprs = [current, *duplicates]
    combined_expr = pt.stack(all_exprs, axis=0)

    result = pt.sum(combined_expr, axis=0)
    return pt.make_dict_of_named_arrays({"result": result})

# }}}


# {{{ tags used only by the regression tests

class TestlibTag(Tag):
    pass


class FooRednTag(TestlibTag):
    """
    foo
    """


class FooInameTag(TestlibTag):
    """
    foo
    """


class BarInameTag(TestlibTag):
    """
    bar
    """


class BazInameTag(TestlibTag):
    """
    baz
    """


class FooTag(TestlibTag):
    """
    foo
    """


class BarTag(TestlibTag):
    """
    bar
    """


class BazTag(TestlibTag):
    """
    baz
    """


class QuuxTag(TestlibTag):
    """
    quux
    """

# }}}

# vim: foldmethod=marker
