from __future__ import annotations

__copyright__ = """
Copyright (C) 2020 Matt Wala
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

from typing import Any, Callable, Dict, FrozenSet, Optional, Union

from pytato.array import (
        Array, IndexLambda, Namespace, Placeholder, MatrixProduct, Stack,
        Roll, AxisPermutation, Slice, DataWrapper, SizeParam,
        DictOfNamedArrays, Reshape, Concatenate, InputArgumentBase,
        NamedArray)
from pytato.loopy import LoopyFunction

__doc__ = """
.. currentmodule:: pytato.transform

Transforming Computations
-------------------------

.. autoclass:: CopyMapper
.. autoclass:: DependencyMapper
.. autofunction:: copy_namespace
.. autofunction:: copy_dict_of_named_arrays
.. autofunction:: get_dependencies

"""


# {{{ mapper classes

class UnsupportedArrayError(ValueError):
    pass


class Mapper:
    def handle_unsupported_array(self, expr: Array, *args: Any,
            **kwargs: Any) -> Any:
        """Mapper method that is invoked for
        :class:`pytato.Array` subclasses for which a mapper
        method does not exist in this mapper.
        """
        raise UnsupportedArrayError("%s cannot handle expressions of type %s"
                % (type(self).__name__, type(expr)))

    def map_foreign(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("%s encountered invalid foreign object: %s"
                % (type(self).__name__, repr(expr)))

    def rec(self, expr: Array, *args: Any, **kwargs: Any) -> Any:
        method: Callable[..., Array]

        try:
            method = getattr(self, expr._mapper_method)
        except AttributeError:
            if isinstance(expr, Array):
                return self.handle_unsupported_array(expr, *args, **kwargs)
            else:
                return self.map_foreign(expr, *args, **kwargs)

        return method(expr, *args, **kwargs)

    def __call__(self, expr: Array, *args: Any, **kwargs: Any) -> Any:
        return self.rec(expr, *args, **kwargs)


class CopyMapper(Mapper):

    def __init__(self, namespace: Namespace):
        self.namespace = namespace
        self.cache: Dict[Union[Array, DictOfNamedArrays], Array] = {}

    def rec(self, expr: Union[Array, DictOfNamedArrays]) -> Union[Array,
            DictOfNamedArrays]:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: Union[Array, DictOfNamedArrays] = super().rec(expr)
        self.cache[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        bindings = {
                name: self.rec(subexpr)
                for name, subexpr in expr.bindings.items()}
        return IndexLambda(namespace=self.namespace,
                expr=expr.expr,
                shape=expr.shape,
                dtype=expr.dtype,
                bindings=bindings,
                tags=expr.tags)

    def map_placeholder(self, expr: Placeholder) -> Array:
        return Placeholder(namespace=self.namespace,
                name=expr.name,
                shape=expr.shape,
                dtype=expr.dtype,
                tags=expr.tags)

    def map_matrix_product(self, expr: MatrixProduct) -> Array:
        return MatrixProduct(x1=self.rec(expr.x1),
                x2=self.rec(expr.x2),
                tags=expr.tags)

    def map_stack(self, expr: Stack) -> Array:
        arrays = tuple(self.rec(arr) for arr in expr.arrays)
        return Stack(arrays=arrays, axis=expr.axis, tags=expr.tags)

    def map_roll(self, expr: Roll) -> Array:
        return Roll(array=self.rec(expr.array),
                shift=expr.shift,
                axis=expr.axis,
                tags=expr.tags)

    def map_axis_permutation(self, expr: AxisPermutation) -> Array:
        return AxisPermutation(array=self.rec(expr.array),
                axes=expr.axes,
                tags=expr.tags)

    def map_slice(self, expr: Slice) -> Array:
        return Slice(array=self.rec(expr.array),
                starts=expr.starts,
                stops=expr.stops,
                tags=expr.tags)

    def map_data_wrapper(self, expr: DataWrapper) -> Array:
        return DataWrapper(namespace=self.namespace,
                name=expr.name,
                data=expr.data,
                shape=expr.shape,
                tags=expr.tags)

    def map_size_param(self, expr: SizeParam) -> Array:
        return SizeParam(self.namespace, name=expr.name, tags=expr.tags)

    def map_named_array(self, expr: NamedArray) -> Array:
        return type(expr)(self.rec(expr.dict_of_named_arrays), expr.name)

    def map_dict_of_named_arrays(self,
            expr: DictOfNamedArrays) -> DictOfNamedArrays:
        return DictOfNamedArrays({key: self.rec(val.expr)
                                  for key, val in expr.items()})

    def map_loopy_function(self, expr: LoopyFunction) -> LoopyFunction:
        bindings = {
                name: self.rec(subexpr)
                for name, subexpr in expr.bindings.items()}
        return LoopyFunction(namespace=self.namespace,
                program=expr.program,
                bindings=bindings,
                entrypoint=expr.entrypoint)


class DependencyMapper(Mapper):
    """
    Maps a :class:`pytato.array.Array` to a :class:`frozenset` of
    :class:`pytato.array.Array`'s it depends on.
    """
    R = FrozenSet[Union[Array, DictOfNamedArrays]]

    def __init__(self) -> None:
        self.cache: Dict[Array, FrozenSet[Array]] = {}

    def rec(self, expr: Array) -> R:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: FrozenSet[Array] = super().rec(expr)
        self.cache[expr] = result
        return result

    def combine(self, *args: FrozenSet[Array]) -> R:
        from functools import reduce
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_index_lambda(self, expr: IndexLambda) -> R:
        return self.combine(frozenset([expr]), *(self.rec(bnd)
                                                 for bnd in expr.bindings.values()))

    def map_placeholder(self, expr: Placeholder) -> R:
        return frozenset([expr])

    def map_data_wrapper(self, expr: DataWrapper) -> R:
        return frozenset([expr])

    def map_size_param(self, expr: SizeParam) -> R:
        return frozenset([expr])

    def map_matrix_product(self, expr: MatrixProduct) -> R:
        return self.combine(frozenset([expr]), self.rec(expr.x1), self.rec(expr.x2))

    def map_stack(self, expr: Stack) -> R:
        return self.combine(frozenset([expr]), *(self.rec(ary)
                                                 for ary in expr.arrays))

    def map_roll(self, expr: Roll) -> R:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_axis_permutation(self, expr: AxisPermutation) -> R:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_slice(self, expr: Slice) -> R:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_reshape(self, expr: Reshape) -> R:
        return self.combine(frozenset([expr]), self.rec(expr.array))

    def map_concatenate(self, expr: Concatenate) -> R:
        return self.combine(frozenset([expr]), *(self.rec(ary)
                                                 for ary in expr.arrays))

    def map_named_array(self, expr: NamedArray) -> R:
        return self.combine(frozenset([expr]), self.rec(expr.dict_of_named_arrays))

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> R:
        return self.combine(frozenset([expr]), *(self.rec(ary)
                                                 for ary in expr.exprs))

    def map_loopy_function(self, expr: LoopyFunction) -> R:
        return self.combine(frozenset([expr]), *(self.rec(ary)
                                                 for ary in expr.bindings.values()
                                                 if isinstance(ary, Array)))

# }}}


# {{{ mapper frontends

def copy_namespace(source_namespace: Namespace,
        copy_mapper: CopyMapper, only_names: Optional[FrozenSet[str]]) -> Namespace:
    """Copy the elements of *namespace* into a new namespace.

    :param source_namespace: The namespace to copy
    :param copy_mapper: A mapper that performs copies into a new namespace
    :param only_names: Elements of *namespace* that are to be copied into the
        new namespace. All elements are copied if *only_names=None*,
    :returns: A new namespace containing copies of the items in *source_namespace*
    """
    if only_names is None:
        only_names = frozenset(source_namespace.keys())

    for name in only_names:
        mapped_val = copy_mapper(source_namespace[name])
        if name not in copy_mapper.namespace:
            copy_mapper.namespace.assign(name, mapped_val)
    return copy_mapper.namespace


def copy_dict_of_named_arrays(source_dict: DictOfNamedArrays,
        copy_mapper: CopyMapper) -> DictOfNamedArrays:
    """Copy the elements of a :class:`~pytato.DictOfNamedArrays` into a
    :class:`~pytato.DictOfNamedArrays` with a new namespace.

    :param source_dict: The :class:`~pytato.DictOfNamedArrays` to copy
    :param copy_mapper: A mapper that performs copies into a new namespace
    :returns: A new :class:`~pytato.DictOfNamedArrays` containing copies of the
        items in *source_dict*
    """
    from pytato.scalar_expr import get_dependencies as get_scalar_expr_deps

    if not source_dict:
        return DictOfNamedArrays({})

    # {{{ extract dependencies elements of the namespace

    # https://github.com/python/mypy/issues/2013
    deps: FrozenSet[Array] = frozenset().union(
            *list(get_dependencies(source_dict).values()))  # type: ignore
    dep_names = (frozenset([dep.name
                            for dep in deps
                            if isinstance(dep, InputArgumentBase)])
                 | get_scalar_expr_deps([dep.shape
                                         for dep in deps
                                         if isinstance(dep, Array)]))

    # }}}

    data = {}
    copy_namespace(source_dict.namespace, copy_mapper, dep_names)
    for name, val in source_dict.items():
        data[name] = copy_mapper(val.expr)
    return DictOfNamedArrays(data)


def get_dependencies(expr: DictOfNamedArrays) -> Dict[str, FrozenSet[Array]]:
    """Returns the dependencies of each named array in *expr*.
    """
    dep_mapper = DependencyMapper()

    return {name: dep_mapper(val.expr) for name, val in expr.items()}

# }}}

# vim: foldmethod=marker
