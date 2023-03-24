"""
.. currentmodule:: pytato.transform.remove_broadcasts_einsum

.. autofunction:: rewrite_einsums_with_no_broadcasts
"""

__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

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


from typing import List, Tuple
from pytato.array import Array, Einsum, EinsumAxisDescriptor
from pytato.transform import CopyMapper, MappedT
from pytato.utils import are_shape_components_equal


class EinsumWithNoBroadcastsRewriter(CopyMapper):
    def map_einsum(self, expr: Einsum) -> Array:
        new_args: List[Array] = []
        new_access_descriptors: List[Tuple[EinsumAxisDescriptor, ...]] = []
        descr_to_axis_len = expr._access_descr_to_axis_len()

        for acc_descrs, arg in zip(expr.access_descriptors, expr.args):
            arg = self.rec(arg)
            axes_to_squeeze: List[int] = []
            for idim, acc_descr in enumerate(acc_descrs):
                if not are_shape_components_equal(arg.shape[idim],
                                                  descr_to_axis_len[acc_descr]):
                    assert are_shape_components_equal(arg.shape[idim], 1)
                    axes_to_squeeze.append(idim)

            if axes_to_squeeze:
                arg = arg[tuple(slice(None) if idim not in axes_to_squeeze else 0
                                for idim in range(arg.ndim))]
                acc_descrs = tuple(acc_descr
                                   for idim, acc_descr in enumerate(acc_descrs)
                                   if idim not in axes_to_squeeze)

            new_args.append(arg)
            new_access_descriptors.append(acc_descrs)

        assert len(new_args) == len(expr.args)
        assert len(new_access_descriptors) == len(expr.access_descriptors)

        return Einsum(tuple(new_access_descriptors),
                      tuple(new_args),
                      expr.redn_axis_to_redn_descr,
                      expr.index_to_access_descr,
                      tags=expr.tags,
                      axes=expr.axes,)


def rewrite_einsums_with_no_broadcasts(expr: MappedT) -> MappedT:
    """
    Rewrites all instances of :class:`~pytato.array.Einsum` in *expr* such that the
    einsum expressions avoid broadcasting axes of its operands. We do
    so by updating the :attr:`pytato.array.Einsum.access_descriptors` and slicing
    the operands.

    .. testsetup::

        >>> import pytato as pt
        >>> import numpy as np

    .. doctest::

        >>> a = pt.make_placeholder("a", (10, 4, 1), np.float64)
        >>> b = pt.make_placeholder("b", (10, 1, 4), np.float64)
        >>> expr = pt.einsum("ijk,ijk->i", a, b)
        >>> new_expr = pt.rewrite_einsums_with_no_broadcasts(expr)
        >>> pt.analysis.is_einsum_similar_to_subscript(new_expr, "ij,ik->i")
        True

    .. note::

        This transformation preserves the semantics of the expression i.e. does not
        alter its value.
    """
    mapper = EinsumWithNoBroadcastsRewriter()

    # type-ignore-reason: mypy is right i.e. CopyMapper.__call__ is imprecise
    return mapper(expr)  # type: ignore[no-any-return]

# vim:fdm=marker
