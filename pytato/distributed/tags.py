from __future__ import annotations


"""
.. currentmodule:: pytato

.. autofunction:: verify_distributed_partition
"""

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


from typing import TYPE_CHECKING, TypeVar, cast

from pytato.distributed.partition import DistributedGraphPartition


if TYPE_CHECKING:
    import mpi4py.MPI

    from pytato.distributed.nodes import CommTagType


T = TypeVar("T")


# {{{ construct tag numbering

def number_distributed_tags(
        mpi_communicator: mpi4py.MPI.Comm,
        partition: DistributedGraphPartition,
        base_tag: int) -> tuple[DistributedGraphPartition, int]:
    """Return a new :class:`~pytato.distributed.DistributedGraphPartition`
    in which symbolic tags are replaced by unique integer tags, created by
    counting upward from *base_tag*.

    :returns: a tuple ``(partition, next_tag)``, where *partition* is the new
        :class:`~pytato.distributed.DistributedGraphPartition` with numerical
        tags, and *next_tag* is the lowest integer tag above *base_tag* that
        was not used.

    .. note::

        This is a potentially heavyweight MPI-collective operation on
        *mpi_communicator*.
    """
    from pytools import flatten

    # A potential optimization here could be to use a 'set' to collect the tags,
    # but this would introduce non-determinism in the tag numbering. Another
    # option would be use something like pytools.unique() to reduce the amount
    # of data communicated, but since all sends and receives should each
    # have unique tags, this would at most buy us a factor of 2.
    tags = tuple([
            recv.comm_tag
            for part in partition.parts.values()
            for recv in part.name_to_recv_node.values()
            ] + [
            send.comm_tag
            for part in partition.parts.values()
            for sends in part.name_to_send_nodes.values()
            for send in sends])

    root_rank = 0

    # We can't let MPI do a set union here, since the result would be
    # non-deterministic.
    all_tags = cast("list[tuple[CommTagType, ...]]",
                    mpi_communicator.gather(tags, root=root_rank))

    if mpi_communicator.rank == root_rank:
        sym_tag_to_int_tag = {}
        next_tag = base_tag
        assert isinstance(all_tags, list)
        assert len(all_tags) == mpi_communicator.size

        for sym_tag in flatten(all_tags):
            if sym_tag not in sym_tag_to_int_tag:
                sym_tag_to_int_tag[sym_tag] = next_tag
                next_tag += 1

        mpi_communicator.bcast((sym_tag_to_int_tag, next_tag), root=root_rank)
    else:
        sym_tag_to_int_tag, next_tag = mpi_communicator.bcast(None, root=root_rank)

    from dataclasses import replace
    return DistributedGraphPartition(
            parts={
                pid: replace(part,
                    name_to_recv_node={
                        name: recv.copy(comm_tag=sym_tag_to_int_tag[recv.comm_tag])
                        for name, recv in part.name_to_recv_node.items()},
                    name_to_send_nodes={
                        name: [
                            send.copy(comm_tag=sym_tag_to_int_tag[send.comm_tag])
                            for send in sends]
                        for name, sends in part.name_to_send_nodes.items()},
                    )
                for pid, part in partition.parts.items()
                },
            name_to_output=partition.name_to_output,
            overall_output_names=partition.overall_output_names,
            ), next_tag

# }}}

# vim: foldmethod=marker
