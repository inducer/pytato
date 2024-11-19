r"""
Distributed-memory evaluation of expression graphs is accomplished
by :ref:`partitioning <partitioning>` the graph to reveal communication-free
pieces of the computation. Communication (i.e. sending/receiving data) is then
accomplished at the boundaries of the parts of the resulting graph partitioning.

Recall the requirement for partitioning that, "no part may depend on its own
outputs as inputs". That sounds obvious, but in the distributed-memory case,
this is harder to decide than it looks, since we do not have full knowledge of
the computation graph.  Edges go off to other nodes and then come back.

.. automodule:: pytato.distributed.nodes
.. automodule:: pytato.distributed.partition
.. automodule:: pytato.distributed.verify
.. automodule:: pytato.distributed.execute

Internal stuff that is only here because the documentation tool wants it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. class:: Tag

    See :class:`pytools.tag.Tag`.

.. class:: CommTagType

    A type representing a communication tag. Communication tags must be
    hashable.

.. class:: ShapeType

    A type representing a shape.

.. class:: AxesT

    A :class:`tuple` of :class:`Axis` objects.
"""

from __future__ import annotations


__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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

from typing import Any


# These are here to support old versions of grudge.

_depr_names = {
        "DistributedGraphPartition",
        "number_distributed_tags",
        "execute_distributed_partition", "pytato.distributed.execute"
        }


def __getattr__(name: str) -> Any:
    if name in _depr_names:
        from warnings import warn
        warn(f"'pytato.distributed.{name}' is deprecated. "
             f"Import as 'pytato.{name}' instead. "
             "This will stop working in July 2023.",
             DeprecationWarning, stacklevel=2)

        import pytato
        return getattr(pytato, name)

    # let name lookup proceed normally
    raise AttributeError(name)
