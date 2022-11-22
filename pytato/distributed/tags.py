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


from typing import TYPE_CHECKING, Tuple, FrozenSet, Any

from pytato.distributed.partition import DistributedGraphPartition


if TYPE_CHECKING:
    import mpi4py.MPI


# {{{ construct tag numbering

def number_distributed_tags(
        mpi_communicator: mpi4py.MPI.Comm,
        partition: DistributedGraphPartition,
        base_tag: int) -> Tuple[DistributedGraphPartition, int]:
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
    tags = frozenset({
            recv.comm_tag
            for part in partition.parts.values()
            for recv in part.input_name_to_recv_node.values()
            } | {
            send.comm_tag
            for part in partition.parts.values()
            for send in part.output_name_to_send_node.values()})

    from mpi4py import MPI

    def set_union(
            set_a: FrozenSet[Any], set_b: FrozenSet[Any],
            mpi_data_type: MPI.Datatype) -> FrozenSet[str]:
        assert mpi_data_type is None
        assert isinstance(set_a, frozenset)
        assert isinstance(set_b, frozenset)

        return set_a | set_b

    root_rank = 0

    set_union_mpi_op = MPI.Op.Create(
            # type ignore reason: mpi4py misdeclares op functions as returning
            # None.
            set_union,  # type: ignore[arg-type]
            commute=True)
    try:
        all_tags = mpi_communicator.reduce(
                tags, set_union_mpi_op, root=root_rank)
    finally:
        set_union_mpi_op.Free()

    if mpi_communicator.rank == root_rank:
        sym_tag_to_int_tag = {}
        next_tag = base_tag
        assert isinstance(all_tags, frozenset)

        for sym_tag in all_tags:
            sym_tag_to_int_tag[sym_tag] = next_tag
            next_tag += 1

        mpi_communicator.bcast((sym_tag_to_int_tag, next_tag), root=root_rank)
    else:
        sym_tag_to_int_tag, next_tag = mpi_communicator.bcast(None, root=root_rank)

    from attrs import evolve as replace
    return DistributedGraphPartition(
            parts={
                pid: replace(part,
                    input_name_to_recv_node={
                        name: recv.copy(comm_tag=sym_tag_to_int_tag[recv.comm_tag])
                        for name, recv in part.input_name_to_recv_node.items()},
                    output_name_to_send_node={
                        name: send.copy(comm_tag=sym_tag_to_int_tag[send.comm_tag])
                        for name, send in part.output_name_to_send_node.items()},
                    )
                for pid, part in partition.parts.items()
                },
            var_name_to_result=partition.var_name_to_result,
            toposorted_part_ids=partition.toposorted_part_ids), next_tag

# }}}

# vim: foldmethod=marker
