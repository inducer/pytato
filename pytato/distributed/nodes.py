"""
Nodes
-----
The following nodes represent communication in the DAG:

.. currentmodule:: pytato
.. autoclass:: DistributedSend
.. autoclass:: DistributedSendRefHolder
.. autoclass:: DistributedRecv

These functions aid in creating communication nodes:

.. autofunction:: staple_distributed_send
.. autofunction:: make_distributed_recv

For completeness, individual (non-held/"stapled") :class:`DistributedSend` nodes
can be made via this function:

.. autofunction:: make_distributed_send
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

from typing import Hashable, FrozenSet, Optional, Any, cast, ClassVar, Tuple

import attrs
import numpy as np

from pytools.tag import Taggable, Tag

from pytato.array import (
        Array, _SuppliedShapeAndDtypeMixin, ShapeType, AxesT,
        _get_default_axes, ConvertibleToShape, normalize_shape)

CommTagType = Hashable


# {{{ send

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

    def __init__(self, data: Array, dest_rank: int, comm_tag: CommTagType,
                 tags: FrozenSet[Tag] = frozenset()) -> None:
        super().__init__(tags=tags)
        self.data = data
        self.dest_rank = dest_rank
        self.comm_tag = comm_tag

    def __hash__(self) -> int:
        return (
                hash(self.__class__)
                ^ hash(self.data)
                ^ hash(self.dest_rank)
                ^ hash(self.comm_tag)
                ^ hash(self.tags)
                )

    def __eq__(self, other: Any) -> bool:
        return (
                self.__class__ is other.__class__
                and self.data == other.data
                and self.dest_rank == other.dest_rank
                and self.comm_tag == other.comm_tag
                and self.tags == other.tags)

    def _with_new_tags(self, tags: FrozenSet[Tag]) -> DistributedSend:
        return DistributedSend(
                data=self.data,
                dest_rank=self.dest_rank,
                comm_tag=self.comm_tag,
                tags=tags)

    def copy(self, **kwargs: Any) -> DistributedSend:
        data: Optional[Array] = kwargs.get("data")
        dest_rank: Optional[int] = kwargs.get("dest_rank")
        comm_tag: Optional[CommTagType] = kwargs.get("comm_tag")
        tags = cast(FrozenSet[Tag], kwargs.get("tags"))
        return type(self)(
                data=data if data is not None else self.data,
                dest_rank=dest_rank if dest_rank is not None else self.dest_rank,
                comm_tag=comm_tag if comm_tag is not None else self.comm_tag,
                tags=tags if tags is not None else self.tags)

    def __repr__(self) -> str:
        # self.data takes a lot of space, shorten it
        return (f"DistributedSend(data={self.data.__class__} "
                f"at {hex(id(self.data))}, "
                f"dest_rank={self.dest_rank}, "
                f"tags={self.tags}, comm_tag={self.comm_tag})")

# }}}


# {{{ send ref holder

@attrs.define(frozen=True, eq=False, repr=False, init=False)
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
    _fields: ClassVar[Tuple[str, ...]] = Array._fields + ("passthrough_data", "send")

    def __init__(self, send: DistributedSend, passthrough_data: Array,
                 tags: FrozenSet[Tag] = frozenset()) -> None:
        super().__init__(axes=passthrough_data.axes, tags=tags)
        object.__setattr__(self, "send", send)
        object.__setattr__(self, "passthrough_data", passthrough_data)

    @property
    def shape(self) -> ShapeType:
        return self.passthrough_data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.passthrough_data.dtype

    def copy(self, **kwargs: Any) -> DistributedSendRefHolder:
        # override 'Array.copy' because
        # 'DistributedSendRefHolder.axes' is a read-only field.
        send = kwargs.pop("send", self.send)
        passthrough_data = kwargs.pop("passthrough_data", self.passthrough_data)
        tags = kwargs.pop("tags", self.tags)

        if kwargs:
            raise ValueError("Cannot assign"
                             f" DistributedSendRefHolder.'{set(kwargs)}'")
        return DistributedSendRefHolder(send,
                                        passthrough_data,
                                        tags)

# }}}


# {{{ receive

@attrs.define(frozen=True, eq=False)
class DistributedRecv(_SuppliedShapeAndDtypeMixin, Array):
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
    _shape: ShapeType
    _dtype: Any  # FIXME: sphinx does not like `_dtype: _dtype_any`

    _fields: ClassVar[Tuple[str, ...]] = Array._fields + ("shape", "dtype",
                                                          "src_rank", "comm_tag")
    _mapper_method: ClassVar[str] = "map_distributed_recv"

# }}}


# {{{ constructor functions

def make_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          send_tags: FrozenSet[Tag] = frozenset()) -> \
         DistributedSend:
    """Make a :class:`DistributedSend` object."""
    return DistributedSend(sent_data, dest_rank, comm_tag, send_tags)


def staple_distributed_send(sent_data: Array, dest_rank: int, comm_tag: CommTagType,
                          stapled_to: Array, *,
                          send_tags: FrozenSet[Tag] = frozenset(),
                          ref_holder_tags: FrozenSet[Tag] = frozenset()) -> \
         DistributedSendRefHolder:
    """Make a :class:`DistributedSend` object wrapped in a
    :class:`DistributedSendRefHolder` object."""
    return DistributedSendRefHolder(
            DistributedSend(sent_data, dest_rank, comm_tag, send_tags),
            stapled_to, tags=ref_holder_tags)


def make_distributed_recv(src_rank: int, comm_tag: CommTagType,
                          shape: ConvertibleToShape, dtype: Any,
                          axes: Optional[AxesT] = None,
                          tags: FrozenSet[Tag] = frozenset()
                          ) -> DistributedRecv:
    """Make a :class:`DistributedRecv` object."""
    shape = normalize_shape(shape)

    if axes is None:
        axes = _get_default_axes(len(shape))

    dtype = np.dtype(dtype)
    return DistributedRecv(src_rank, comm_tag, shape, dtype, tags=tags, axes=axes)

# }}}

# vim: foldmethod=marker
