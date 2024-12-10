"""
.. autoclass:: SymbolicIndex

.. autoclass:: CodeGenPreprocessor
.. autoclass:: PreprocessResult

.. autofunction:: preprocess
.. autofunction:: normalize_outputs
"""

from __future__ import annotations

from typing import TypeAlias

from typing_extensions import TypeIs


__copyright__ = """Copyright (C) 2020 Matt Wala"""

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

import dataclasses
from typing import TYPE_CHECKING, Any

from immutabledict import immutabledict

import loopy as lp
from pymbolic.mapper.optimize import optimize_mapper
from pytools import UniqueNameGenerator

from pytato.array import (
    Array,
    DataInterface,
    DataWrapper,
    DictOfNamedArrays,
    InputArgumentBase,
    Placeholder,
    SizeParam,
    make_dict_of_named_arrays,
)
from pytato.loopy import LoopyCall
from pytato.scalar_expr import IntegralScalarExpression, is_integral_scalar_expression
from pytato.transform import (
    ArrayOrNames,
    CachedWalkMapper,
    CopyMapper,
    SubsetDependencyMapper,
)
from pytato.transform.lower_to_index_lambda import ToIndexLambdaMixin


if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytato.function import NamedCallResult
    from pytato.target import Target


SymbolicIndex: TypeAlias = tuple[IntegralScalarExpression, ...]


def is_symbolic_index(o: object) -> TypeIs[SymbolicIndex]:
    if isinstance(o, tuple):
        for i in o:
            if not is_integral_scalar_expression(i):
                return False
        return True
    else:
        return False


# {{{ _generate_name_for_temp

def _generate_name_for_temp(
        expr: Array, var_name_gen: UniqueNameGenerator,
        default_prefix: str = "_pt_temp") -> str:
    from pytato.tags import Named, PrefixNamed, _BaseNameTag
    if expr.tags_of_type(_BaseNameTag):
        if expr.tags_of_type(Named):
            name_tag, = expr.tags_of_type(Named)
            assert isinstance(name_tag, Named)
            if var_name_gen.is_name_conflicting(name_tag.name):
                raise ValueError(f"Cannot assign the name {name_tag.name} to the"
                                 f" temporary corresponding to {expr} as it "
                                 "conflicts with an existing name. ")
            var_name_gen.add_name(name_tag.name)
            return name_tag.name
        elif expr.tags_of_type(PrefixNamed):
            prefix_tag, = expr.tags_of_type(PrefixNamed)
            return var_name_gen(prefix_tag.prefix)
        else:
            raise NotImplementedError(type(next(iter(expr.tags_of_type(_BaseNameTag)))))
    else:
        return var_name_gen(default_prefix)

# }}}


# {{{ preprocessing for codegen

# type-ignore-reason: incompatible 'rec' types between ToIndexLambdaMixin, CopyMapper
class CodeGenPreprocessor(ToIndexLambdaMixin, CopyMapper):  # type: ignore[misc]
    """A mapper that preprocesses graphs to simplify code generation.

    The following node simplifications are performed:

    ======================================  =====================================
    Source Node Type                        Target Node Type
    ======================================  =====================================
    :class:`~pytato.array.DataWrapper`      :class:`~pytato.array.Placeholder`
    :class:`~pytato.array.Roll`             :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.AxisPermutation`  :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.IndexBase`        :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Reshape`          :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Concatenate`      :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Einsum`           :class:`~pytato.array.IndexLambda`
    :class:`~pytato.array.Stack`            :class:`~pytato.array.IndexLambda`
    ======================================  =====================================
    """

    def __init__(self, target: Target,
                 kernels_seen: dict[str, lp.LoopKernel] | None = None
                 ) -> None:
        super().__init__()
        self.bound_arguments: dict[str, DataInterface] = {}
        self.var_name_gen: UniqueNameGenerator = UniqueNameGenerator()
        self.target = target
        self.kernels_seen: dict[str, lp.LoopKernel] = kernels_seen or {}

    def map_size_param(self, expr: SizeParam) -> Array:
        name = expr.name
        assert name is not None
        return SizeParam(  # pylint: disable=missing-kwoa
            name=name,
            tags=expr.tags,
            non_equality_tags=expr.non_equality_tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        name = expr.name
        if name is None:
            name = self.var_name_gen("_pt_in")
        return Placeholder(name=name,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_loopy_call(self, expr: LoopyCall) -> LoopyCall:
        from pytato.target.loopy import LoopyTarget
        if not isinstance(self.target, LoopyTarget):
            raise ValueError("Got a LoopyCall for a non-loopy target.")
        translation_unit = expr.translation_unit.copy(
                                        target=self.target.get_loopy_target())
        namegen = UniqueNameGenerator(set(self.kernels_seen))
        entrypoint = expr.entrypoint

        # {{{ eliminate callable name collision

        for name, clbl in translation_unit.callables_table.items():
            if isinstance(clbl, lp.CallableKernel):
                assert isinstance(name, str)
                if name in self.kernels_seen and (
                        translation_unit[name] != self.kernels_seen[name]):
                    # callee name collision => must rename

                    # {{{ see if it's one of the other kernels

                    for other_knl in self.kernels_seen.values():
                        if other_knl.copy(name=name) == translation_unit[name]:
                            new_name = other_knl.name
                            break
                    else:
                        # didn't find any other equivalent kernel, rename to
                        # something unique
                        new_name = namegen(name)

                    # }}}

                    if name == entrypoint:
                        # if the colliding name is the entrypoint, then rename the
                        # entrypoint as well.
                        entrypoint = new_name

                    translation_unit = lp.rename_callable(
                                            translation_unit, name, new_name)
                    name = new_name

                self.kernels_seen[name] = clbl.subkernel

        # }}}

        bindings: Mapping[str, Any] = immutabledict(
                    {name: (self.rec(subexpr) if isinstance(subexpr, Array)
                           else subexpr)
                    for name, subexpr in sorted(expr.bindings.items())})

        return LoopyCall(translation_unit=translation_unit,
                         bindings=bindings,
                         entrypoint=entrypoint,
                         tags=expr.tags
                         )

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        name = _generate_name_for_temp(expr, self.var_name_gen, "_pt_data")

        self.bound_arguments[name] = expr.data
        return Placeholder(name=name,
                shape=tuple(self.rec(s) if isinstance(s, Array) else s
                            for s in expr.shape),
                dtype=expr.dtype,
                axes=expr.axes,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        raise NotImplementedError("CodeGenPreprocessor does not support functions.")

# }}}


def normalize_outputs(
            result: Array | DictOfNamedArrays | dict[str, Array]
        ) -> DictOfNamedArrays:
    """Convert outputs of a computation to the canonical form.

    Performs a conversion to :class:`~pytato.DictOfNamedArrays` if necessary.

    :param result: Outputs of the computation.
    """
    if not isinstance(result, Array | DictOfNamedArrays | dict):
        raise TypeError("outputs of the computation should be "
                "either an Array or a DictOfNamedArrays")

    if isinstance(result, Array):
        outputs = make_dict_of_named_arrays({"_pt_out": result})
    elif isinstance(result, dict):
        outputs = make_dict_of_named_arrays(result)
    else:
        assert isinstance(result, DictOfNamedArrays)
        outputs = result

    return outputs


# {{{ input naming check

@optimize_mapper(drop_args=True, drop_kwargs=True, inline_get_cache_key=True)
class NamesValidityChecker(CachedWalkMapper[[]]):
    def __init__(self) -> None:
        self.name_to_input: dict[str, InputArgumentBase] = {}
        super().__init__()

    def get_cache_key(self, expr: ArrayOrNames) -> int:
        return id(expr)

    def post_visit(self, expr: Any) -> None:
        if isinstance(expr, Placeholder | SizeParam | DataWrapper):
            if expr.name is not None:
                try:
                    ary = self.name_to_input[expr.name]
                except KeyError:
                    self.name_to_input[expr.name] = expr
                else:
                    if ary is not expr:
                        from pytato.diagnostic import NameClashError
                        raise NameClashError(
                                "Received two separate instances of inputs "
                                f"named '{expr.name}'.")


def check_validity_of_outputs(exprs: DictOfNamedArrays) -> None:
    name_validation_mapper = NamesValidityChecker()

    for ary in exprs.values():
        name_validation_mapper(ary)

# }}}


@dataclasses.dataclass(init=True, repr=False, eq=False)
class PreprocessResult:
    outputs: DictOfNamedArrays
    compute_order: tuple[str, ...]
    bound_arguments: dict[str, DataInterface]


def preprocess(outputs: DictOfNamedArrays, target: Target) -> PreprocessResult:
    """Preprocess a computation for code generation."""
    from pytato.transform import copy_dict_of_named_arrays
    from pytato.transform.calls import inline_calls

    check_validity_of_outputs(outputs)

    # {{{ compute the order in which the outputs must be computed

    # semantically order does not matter, but doing a toposort ordering of the
    # outputs leads to a FLOP optimal choice

    from pytools.graph import compute_topological_order

    get_deps = SubsetDependencyMapper(frozenset(out.expr
                                                for out in outputs.values()))

    # only look for dependencies between the outputs
    deps: Mapping[str, Any] = immutabledict({name: get_deps(output.expr)
            for name, output in outputs.items()})

    # represent deps in terms of output names
    output_expr_to_name = {output.expr: name for name, output in outputs.items()}
    dag = {name: (frozenset([output_expr_to_name[output] for output in val])
                  - frozenset([name]))
           for name, val in deps.items()}

    output_order: list[str] = compute_topological_order(dag, key=lambda x: x)[::-1]

    # }}}

    new_outputs = inline_calls(outputs)
    assert isinstance(new_outputs, DictOfNamedArrays)

    mapper = CodeGenPreprocessor(target)
    new_outputs = copy_dict_of_named_arrays(new_outputs, mapper)

    return PreprocessResult(outputs=new_outputs,
                            compute_order=tuple(output_order),
                            bound_arguments=mapper.bound_arguments)

# vim: fdm=marker
