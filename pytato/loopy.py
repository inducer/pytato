from __future__ import annotations


__copyright__ = """
Copyright (C) 2021 Kaushik Kulkarni
"""

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
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
)

import islpy as isl
import numpy as np
from immutabledict import immutabledict
from typing_extensions import override

import loopy as lp
import pymbolic.primitives as prim
from loopy.typing import assert_tuple
from pytools import memoize_method

from pytato.array import (
    AbstractResultWithNamedArrays,
    Array,
    ArrayOrScalar,
    NamedArray,
    ShapeType,
    SizeParam,
    array_dataclass,
)
from pytato.scalar_expr import (
    EvaluationMapper,
    ScalarExpression,
    SubstitutionMapper,
)


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from pymbolic.typing import ArithmeticExpression, Expression, Integer


__doc__ = r"""
.. currentmodule:: pytato.loopy

.. autoclass:: LoopyCall

.. autoclass:: LoopyCallResult

.. autofunction:: call_loopy

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. class:: Tag

    See :class:`pytools.tag.Tag`.

.. class:: AxesT

    See :class:`pytato.array.AxesT`.

.. class:: ArrayOrScalar

    A :class:`~pytato.Array` or a scalar.

.. currentmodule:: lp

.. class:: TranslationUnit

   See :class:`loopy.TranslationUnit`.
"""


@array_dataclass()
class LoopyCall(AbstractResultWithNamedArrays):
    """
    An array expression node representing a call to an entrypoint in a
    :mod:`loopy` translation unit.
    """
    translation_unit: lp.TranslationUnit
    bindings: Mapping[str, ArrayOrScalar]
    entrypoint: str

    copy = dataclasses.replace

    def __post_init__(self) -> None:
        assert isinstance(self.bindings, immutabledict)
        super().__post_init__()

    @property
    def _result_names(self) -> frozenset[str]:
        return frozenset({name
                          for name, lp_arg in self._entry_kernel.arg_dict.items()
                          if lp_arg.is_output})

    @memoize_method
    def _to_pytato(self, expr: ScalarExpression) -> Expression:
        from pytato.scalar_expr import substitute
        return substitute(expr, self.bindings)

    @property
    def _entry_kernel(self) -> lp.LoopKernel:
        return self.translation_unit[self.entrypoint]

    @override
    def __hash__(self) -> int:
        return hash((self.translation_unit, tuple(self.bindings.items()),
                     self.entrypoint, self.tags))

    @override
    def __contains__(self, name: object) -> bool:
        return name in self._result_names

    @memoize_method
    def __getitem__(self, name: str) -> LoopyCallResult:
        from pytato.array import _get_default_axes

        if name not in self._result_names:
            raise KeyError(name)

        # TODO: Attach a filtered set of tags from loopy's arg.
        return LoopyCallResult(_container=self,
                               name=name,
                               axes=_get_default_axes(len(self
                                                          ._entry_kernel
                                                          .arg_dict[name]
                                                          .shape)),
                               tags=frozenset())

    @override
    def __len__(self) -> int:
        return len(self._result_names)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._result_names)

    # type-ignore-reason: AbstractResultWithNamedArrays returns a KeysView here
    @override
    def keys(self) -> frozenset[str]:  # type: ignore[override]
        return self._result_names


@array_dataclass()
class LoopyCallResult(NamedArray):
    """
    Named array for :class:`LoopyCall`'s result.
    Inherits from :class:`~pytato.array.NamedArray`.
    """
    _container: LoopyCall

    @property
    @override
    def expr(self) -> Array:
        raise ValueError("Expressions for results of loopy functions aren't defined")

    @property
    @override
    def shape(self) -> ShapeType:
        # pylint: disable=E1101
        # reason: (pylint doesn't respect the asserts)
        assert isinstance(self._container, LoopyCall)
        loopy_arg = self._container._entry_kernel.arg_dict[self.name]
        shape: ShapeType = self._container._to_pytato(  # type:ignore[assignment]
                loopy_arg.shape)
        return shape

    @property
    @override
    def dtype(self) -> np.dtype[Any]:
        # pylint: disable=E1101
        # reason: (pylint doesn't respect the asserts)
        assert isinstance(self._container, LoopyCall)
        loopy_arg = self._container._entry_kernel.arg_dict[self.name]
        assert loopy_arg.dtype is not None
        return np.dtype(loopy_arg.dtype.numpy_dtype)


def call_loopy(translation_unit: lp.TranslationUnit,
               bindings: dict[str, ArrayOrScalar],
               entrypoint: str | None = None) -> LoopyCall:
    """
    Invokes an entry point of a :class:`loopy.TranslationUnit` on the array inputs as
    specified by *bindings*.

    Restrictions on the structure of ``translation_unit[entrypoint]``:

    * array arguments of ``translation_unit[entrypoint]`` must either be either
      input-only or output-only.
    * all input-only arguments of ``translation_unit[entrypoint]`` must appear in
      *bindings*.
    * all output-only arguments of ``translation_unit[entrypoint]`` must appear
      in *bindings*.
    * if *translation_unit* has been declared with multiple entrypoints,
      *entrypoint* can not be *None*.

    :arg translation_unit: the translation unit to call.
    :arg bindings: mapping from argument names of ``translation_unit[entrypoint]``
      to :class:`pytato.array.Array`.
    :arg entrypoint: the entrypoint of the ``translation_unit`` parameter.
    """
    from pytato.array import _get_default_tags

    if entrypoint is None:
        if len(translation_unit.entrypoints) != 1:
            raise ValueError("cannot infer entrypoint")

        entrypoint, = translation_unit.entrypoints

    translation_unit = translation_unit.with_entrypoints(entrypoint)

    # {{{ sanity checks

    if any(arg.is_input and arg.is_output
            for arg in translation_unit[entrypoint].args):
        # Pytato DAG cannot have stateful nodes.
        raise ValueError("Cannot call a kernel with side-effects.")

    for name in bindings:
        if name not in translation_unit[entrypoint].arg_dict:
            raise ValueError(f"Kernel '{entrypoint}' got an unexpected input: "
                    f"'{name}'.")
        if translation_unit[entrypoint].arg_dict[name].is_output:
            raise ValueError(f"Kernel '{entrypoint}' got an output arg '{name}' "
                    f"as input.")

    # {{{ perform shape inference here

    bindings_new = extend_bindings_with_shape_inference(translation_unit[entrypoint],
                                                    immutabledict(bindings))
    del bindings

    # }}}

    for arg in translation_unit[entrypoint].args:
        if arg.is_input:
            if arg.name not in bindings_new:
                raise ValueError(f"Kernel '{entrypoint}' expects an input"
                        f" '{arg.name}'")

            arg_binding = bindings_new[arg.name]

            if isinstance(arg, lp.ArrayArg | lp.ConstantArg):
                if not isinstance(arg_binding, Array):
                    raise ValueError(f"Argument '{arg.name}' expected to be a "
                            f"pytato.Array, got {type(arg_binding)}.")
            else:
                assert isinstance(arg, lp.ValueArg)
                if not (isinstance(arg_binding, Number)
                        or (isinstance(arg_binding, Array)
                            and arg_binding.shape == ())):
                    raise ValueError(f"Argument '{arg.name}' expected to be a "
                            " number or a scalar expression, got "
                            f"{type(arg_binding)}.")

    # }}}

    # {{{ infer types of the translation_unit

    for name, ary in bindings_new.items():
        if translation_unit[entrypoint].arg_dict[name].dtype is not None:
            continue

        if isinstance(ary, Array):
            translation_unit = lp.add_dtypes(translation_unit, {name: ary.dtype})
        else:
            assert isinstance(ary, Number)
            translation_unit = lp.add_dtypes(translation_unit,
                                             {name: np.dtype(type(ary))})

    translation_unit = lp.infer_unknown_types(translation_unit)

    # }}}

    # {{{ infer shapes of the translation_unit

    translation_unit = lp.infer_arg_descr(translation_unit)

    # }}}

    translation_unit = translation_unit.with_entrypoints(frozenset())

    return LoopyCall(translation_unit, bindings_new, entrypoint,
                     tags=_get_default_tags())


# {{{ shape inference

class ShapeInferenceFailure(RuntimeError):  # noqa: N818
    pass


def _get_val_in_bset(bset: isl.BasicSet, idim: int) -> ScalarExpression:
    """
    Gets the value of *bset*'s *idim*-th set-dim in terms of it's param-dims.

    .. note::

        Assumes all constraints in *bset* are equality constraints.
    """
    from loopy.symbolic import aff_to_expr

    max_val = bset.dim_max(idim)

    assert max_val.is_equal(bset.dim_min(idim))

    if max_val.n_piece() != 1:
        raise NotImplementedError("Shape inference resulted in a piecewise"
                                    " result.")

    (_, aff), = max_val.get_pieces()

    return aff_to_expr(aff)


def solve_constraints(variables: Sequence[str],
                      parameters: Sequence[str],
                      constraints: Sequence[tuple[ArithmeticExpression,
                                                  ArithmeticExpression]],

                      ) -> Mapping[str, ScalarExpression]:
    """
    :arg variables: Names of the variables to solve for
    :arg parameters: Names of the parameters that to express that are allowed
        to be a part of the solution expressions.
    :arg constraints: A :class:`list` of constraints. Each constraint is
        represented as a tuple ``(lhs, rhs)``, that corresponds to the
        constraint ``lhs = rhs``. ``lhs`` and ``rhs`` are quasi-affine
        expressions in *variables* and *constraints*.
    :returns: A mapping from variable name in *variables* to
        :class:`ScalarExpression` obtained after solving for them.
    """
    from loopy.symbolic import aff_from_expr

    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
                                        set=variables,
                                        params=parameters)

    shape_inference_bset = isl.BasicSet.universe(space)

    for lhs, rhs in constraints:
        aff = aff_from_expr(space, lhs-rhs)

        shape_inference_bset = (shape_inference_bset
                                .add_constraint(isl.Constraint
                                                .equality_from_aff(aff)))

    if shape_inference_bset.is_empty():
        raise ShapeInferenceFailure

    solution = {}

    # {{{ get the value of each unknown variable

    for idim in range(shape_inference_bset.dim(isl.dim_type.set)):
        arg_name = shape_inference_bset.get_dim_name(isl.dim_type.set, idim)
        solved_val = _get_val_in_bset(shape_inference_bset, idim)
        solution[arg_name] = solved_val

    # }}}

    return solution


# {{{ shape inference helpers

def _lp_var_to_global_namespace(name: str) -> str:
    return f"_lp_{name}"


def _lp_var_from_global_namespace(name: str) -> str:
    assert name[:4] == "_lp_"
    return name[4:]


def _pt_var_to_global_namespace(name: str | None) -> str:
    assert name is not None  # size params are always named
    return f"_pt_{name}"


def _get_pt_dim_expr(dim: Integer | Array) -> ScalarExpression:
    from pytato.scalar_expr import substitute
    from pytato.utils import dim_to_index_lambda_components
    dim_expr, dim_bnds = dim_to_index_lambda_components(dim)
    assert all(isinstance(dim_bnd, SizeParam)
                for dim_bnd in dim_bnds.values())

    res = substitute(dim_expr,
                        {k: prim.Variable(v.name)
                        for k, v in dim_bnds.items()})
    assert not isinstance(res, tuple)
    return res

# }}}


def extend_bindings_with_shape_inference(knl: lp.LoopKernel,
                                         bindings: Mapping[str, ArrayOrScalar]
                                         ) -> immutabledict[str, ArrayOrScalar]:
    from functools import reduce

    from loopy.kernel.array import ArrayBase
    from loopy.symbolic import get_dependencies as lpy_get_deps
    from pymbolic.mapper.substitutor import (
        make_subst_func,  # pyright: ignore[reportUnknownVariableType]
    )
    from pymbolic.primitives import is_expression

    from pytato.transform import SizeParamGatherer

    get_size_param_deps = SizeParamGatherer()

    lp_size_params: frozenset[str] = reduce(frozenset.union,
                                            (lpy_get_deps(assert_tuple(arg.shape))
                                             for arg in knl.args
                                             if isinstance(arg, ArrayBase)
                                             and is_expression(arg.shape)
                                         ), frozenset())

    pt_size_params: frozenset[SizeParam] = reduce(frozenset.union,
                                                  (get_size_param_deps(bnd)
                                                   for bnd in bindings.values()
                                                   if isinstance(bnd, Array)),
                                                  frozenset())

    # {{{ mappers to map expressions to a global namespace

    pt_subst_map = SubstitutionMapper(
                        make_subst_func({
                            arg.name: prim.Variable(_pt_var_to_global_namespace(arg
                                                                                .name
                                                                                ))
                            for arg in pt_size_params}))

    lp_subst_map = SubstitutionMapper(
                        make_subst_func({
                            arg: prim.Variable(_lp_var_to_global_namespace(arg))
                            for arg in lp_size_params}))

    # }}}

    constraints: list[tuple[ArithmeticExpression, ArithmeticExpression]] = []

    # {{{ collect constraints from passed arguments

    for lp_arg_name, lp_arg in knl.arg_dict.items():
        if lp_arg_name not in bindings:
            # value not passed => don't add any constraints
            continue

        pt_arg = bindings[lp_arg_name]

        if isinstance(lp_arg, ArrayBase):

            # {{{ sanity checks

            if lp_arg.shape is None:
                # no constraints to add here
                continue

            if lp_arg.shape is lp.auto:
                # TODO: Can lp.auto as shape really appear here?
                raise NotImplementedError("'loopy.auto' as shape dim.")

            assert isinstance(lp_arg.shape, tuple)

            if not isinstance(pt_arg, Array):
                raise ValueError(f"'{knl.name}' got scalar value for '{lp_arg_name}'"
                                 ", expected an array.")

            if len(lp_arg.shape) != len(pt_arg.shape):
                raise ValueError(f"ndim mismatch for argument '{lp_arg_name}'"
                                 f"of '{knl.name}'")

            # }}}

            for lp_dim, pt_dim in zip(lp_arg.shape, pt_arg.shape, strict=True):
                pt_dim_expr = pt_subst_map(_get_pt_dim_expr(pt_dim))
                lp_dim_expr = lp_subst_map(lp_dim)
                assert prim.is_arithmetic_expression(pt_dim_expr)
                assert prim.is_arithmetic_expression(lp_dim_expr)
                constraints.append((pt_dim_expr, lp_dim_expr))

        else:
            if lp_arg_name not in lp_size_params:
                continue

            assert isinstance(lp_arg, lp.ValueArg)
            assert isinstance(pt_arg, int | Array)
            pt_arg_expr = pt_subst_map(_get_pt_dim_expr(pt_arg))
            lp_arg_expr = lp_subst_map(prim.Variable(lp_arg.name))
            assert prim.is_arithmetic_expression(pt_arg_expr)
            assert prim.is_arithmetic_expression(lp_arg_expr)
            constraints.append((pt_arg_expr, lp_arg_expr))

    # }}}

    solutions = solve_constraints(variables=list({_lp_var_to_global_namespace(var)
                                             for var in lp_size_params}),
                                  parameters=list({_pt_var_to_global_namespace(var.name)
                                              for var in pt_size_params}),
                                  constraints=constraints)

    as_pt_size_param = EvaluationMapper({_pt_var_to_global_namespace(arg.name): arg
                                         for arg in pt_size_params})

    bindings_dict = dict(bindings)

    for var, val in solutions.items():
        # map the pymbolic expression back into an expression in terms of
        # pt.SizeParams
        var = _lp_var_from_global_namespace(var)
        val_sp = as_pt_size_param(val)

        # {{{ respect callee's scalar dtype preference if there exists one

        # TODO: remove this once
        # https://github.com/inducer/loopy/issues/442 is resolved.
        dtype = knl.arg_dict[var].dtype
        if (isinstance(val, Number) and dtype is not None):
            val = dtype.numpy_dtype.type(val)

        # }}}

        bindings_dict[var] = val_sp

    return immutabledict(bindings_dict)

# }}}


# vim: fdm=marker
