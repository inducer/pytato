"""
Nodes
-----
The following nodes represent communication in the DAG:

.. currentmodule:: pytato
.. autoclass:: DistributedSend
.. autoclass:: DistributedSendRefHolder
.. autoclass:: DistributedRecv

These functions aid in creating communication nodes:

.. autofunction:: make_distributed_send
.. autofunction:: make_distributed_send_ref_holder
.. autofunction:: staple_distributed_send
.. autofunction:: make_distributed_recv

Redirections for the documentation tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: np

.. class:: dtype

    See :class:`numpy.dtype`.

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

import dataclasses
from typing import Any, ClassVar, Hashable

import numpy as np

from pytools.tag import Tag, Taggable

from pytato.array import (
    Array,
    AxesT,
    ConvertibleToShape,
    ShapeType,
    _get_created_at_tag,
    _get_default_axes,
    _get_default_tags,
    _SuppliedAxesAndTagsMixin,
    _SuppliedShapeAndDtypeMixin,
    array_dataclass,
    normalize_shape,
)


CommTagType = Hashable


# {{{ send

@array_dataclass()
class DistributedSend(Taggable):
    """Class representing a distributed send operation.
    See :class:`DistributedSendRefHolder` for a way to ensure that nodes
    of this type remain part of a DAG.

    .. attribute:: data

        The :class:`~pytato.Array` to be sent.

    .. attribute:: dest_rank

        An :class:`int`. The rank to which :attr:`data` is to be sent.

    .. attribute:: comm_tag

        A hashable, picklable object to serve as a 'tag' for the communication.
        Only a :class:`DistributedRecv` with the same tag will be able to
        receive the data being sent here.
    """

    data: Array
    dest_rank: int
    comm_tag: CommTagType
    tags: frozenset[Tag] = dataclasses.field(kw_only=True, default=frozenset())  # pylint: disable=invalid-field-call

    def _with_new_tags(self, tags: frozenset[Tag]) -> DistributedSend:
        return dataclasses.replace(self, tags=tags)

    def copy(self, **kwargs: Any) -> DistributedSend:
        return dataclasses.replace(self, **kwargs)

# }}}


# {{{ send ref holder

@array_dataclass()
class DistributedSendRefHolder(Array):
    """A node acting as an identity on :attr:`passthrough_data` while also holding
    a reference to a :class:`DistributedSend` in :attr:`send`. Since
    :mod:`pytato` represents data flow, and since no data flows 'out'
    of a :class:`DistributedSend`, no node in all of :mod:`pytato` has
    a good reason to hold a reference to a send node, since there is
    no useful result of a send (at least of an :class:`~pytato.Array` type).

    This is where this node type comes in. Its value is the same as that of
    :attr:`passthrough_data`, *and* it holds a reference to the send node.

    .. note::

        This all seems a wee bit inelegant, but nobody who has written
        or reviewed this code so far had a better idea. If you do, please speak up!

    .. attribute:: send

        The :class:`DistributedSend` to which a reference is to be held.

    .. attribute:: passthrough_data

        A :class:`~pytato.Array`. The value of this node.

    .. note::

        It is the user's responsibility to ensure matching sends and receives
        are part of the computation graph on all ranks. If this rule is not heeded,
        undefined behavior (in particular deadlock) may result.
        Notably, by the nature of the data flow graph built by :mod:`pytato`,
        unused results do not appear in the graph. It is thus possible for a
        :class:`DistributedSendRefHolder` to be constructed and yet to not
        become part of the graph constructed by the user.
    """
    send: DistributedSend
    passthrough_data: Array

    _mapper_method: ClassVar[str] = "map_distributed_send_ref_holder"

    @property
    def shape(self) -> ShapeType:
        return self.passthrough_data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.passthrough_data.dtype

    @property
    def axes(self) -> AxesT:
        return self.passthrough_data.axes

    @property
    def tags(self) -> frozenset[Tag]:
        return self.passthrough_data.tags

    @property
    def non_equality_tags(self) -> frozenset[Tag]:
        return self.passthrough_data.non_equality_tags

# }}}


# {{{ receive

@array_dataclass()
class DistributedRecv(_SuppliedAxesAndTagsMixin, _SuppliedShapeAndDtypeMixin, Array):
    """Class representing a distributed receive operation.

    .. attribute:: src_rank

        An :class:`int`. The rank from which an array is to be received.

    .. attribute:: comm_tag

        A hashable, picklable object to serve as a 'tag' for the communication.
        Only a :class:`DistributedSend` with the same tag will be able to
        send the data being received here.

    .. attribute:: shape
    .. attribute:: dtype

    .. note::

        It is the user's responsibility to ensure matching sends and receives
        are part of the computation graph on all ranks. If this rule is not heeded,
        undefined behavior (in particular deadlock) may result.
        Notably, by the nature of the data flow graph built by :mod:`pytato`,
        unused results do not appear in the graph. It is thus possible for a
        :class:`DistributedRecv` to be constructed and yet to not become part
        of the graph constructed by the user.
    """
    src_rank: int
    comm_tag: CommTagType

    _mapper_method: ClassVar[str] = "map_distributed_recv"

# }}}


# {{{ constructor functions

def make_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          send_tags: frozenset[Tag] = frozenset()) -> \
         DistributedSend:
    """Make a :class:`DistributedSend` object."""
    return DistributedSend(data=sent_data, dest_rank=dest_rank, comm_tag=comm_tag,
                           tags=(send_tags | _get_default_tags()))


def make_distributed_send_ref_holder(
        send: DistributedSend,
        passthrough_data: Array,
        ) -> DistributedSendRefHolder:
    """Make a :class:`DistributedSendRefHolder` object."""
    return DistributedSendRefHolder(
        send=send,
        passthrough_data=passthrough_data)


def staple_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          stapled_to: Array, *,
                          send_tags: frozenset[Tag] = frozenset(),
                          ) -> DistributedSendRefHolder:
    """Make a :class:`DistributedSend` object wrapped in a
    :class:`DistributedSendRefHolder` object."""
    return make_distributed_send_ref_holder(
        send=make_distributed_send(
            sent_data=sent_data, dest_rank=dest_rank, comm_tag=comm_tag,
            send_tags=send_tags),
        passthrough_data=stapled_to,
        )


def make_distributed_recv(src_rank: int, comm_tag: CommTagType,
                          shape: ConvertibleToShape, dtype: Any,
                          axes: AxesT | None = None,
                          tags: frozenset[Tag] = frozenset()
                          ) -> DistributedRecv:
    """Make a :class:`DistributedRecv` object."""
    shape = normalize_shape(shape)

    if axes is None:
        axes = _get_default_axes(len(shape))

    dtype = np.dtype(dtype)
    return DistributedRecv(
            src_rank=src_rank, comm_tag=comm_tag, shape=shape, dtype=dtype,
            axes=axes, tags=(tags | _get_default_tags()),
            non_equality_tags=_get_created_at_tag())

# }}}

# vim: foldmethod=marker
