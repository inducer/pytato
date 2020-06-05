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

from typing import Any, Callable, Dict, no_type_check

from pytato.array import Array, IndexLambda, Namespace, Output, Placeholder

__doc__ = """
.. currentmodule:: pytato.transform

Transforming Computations
-------------------------

.. autoclass:: CopyMapper
.. autofunction:: copy_namespace

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

    def __call__(self, expr: Array, *args: Any, **kwargs: Any) -> Any:
        method: Callable[..., Array]

        try:
            method = getattr(self, expr.mapper_method)
        except AttributeError:
            if isinstance(expr, Array):
                return self.handle_unsupported_array(expr, *args, **kwargs)
            else:
                return self.map_foreign(expr, *args, **kwargs)

        return method(expr, *args, **kwargs)

    rec = __call__


class CopyMapper(Mapper):
    namespace: Namespace

    def __init__(self, namespace: Namespace):
        self.namespace = namespace
        self.cache: Dict[Array, Array] = {}

    def rec(self, expr: Array) -> Array:  # type: ignore[override]
        if expr in self.cache:
            return self.cache[expr]
        result: Array = super().rec(expr)
        self.cache[expr] = result
        return result

    def __call__(self, expr: Array) -> Array:  # type: ignore[override]
        return self.rec(expr)

    def map_index_lambda(self, expr: IndexLambda) -> Array:
        bindings = {
                name: self.rec(subexpr)
                for name, subexpr in expr.bindings.items()}
        return IndexLambda(self.namespace,
                expr=expr.expr,
                shape=expr.shape,
                dtype=expr.dtype,
                bindings=bindings)

    def map_placeholder(self, expr: Placeholder) -> Array:
        return Placeholder(self.namespace, expr.name, expr.shape, expr.dtype,
                expr.tags)

    def map_output(self, expr: Output) -> Array:
        return Output(self.namespace, expr.name, self.rec(expr.array),
                expr.tags)

# }}}


# {{{ mapper frontends

def copy_namespace(namespace: Namespace, copy_mapper: CopyMapper) -> Namespace:
    """Copy the elements of *namespace* into a new namespace.

    :param namespace: The source namespace
    :param copy_mapper: A mapper that performs copies into a new namespace
    :returns: The new namespace
    """
    for name, val in namespace.items():
        mapped_val = copy_mapper(val)
        if name not in copy_mapper.namespace:
            copy_mapper.namespace.assign(name, mapped_val)
    return copy_mapper.namespace

# }}}

# vim: foldmethod=marker
