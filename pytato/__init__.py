from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Xiaoyu Wei
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

# {{{ debug control

import os

from numpy import dtype

from pytools import strtobool


DEBUG_ENABLED = strtobool(os.environ.get("PYTATO_DEBUG", "no"))


def set_debug_enabled(flag: bool) -> None:
    """Set whether :mod:`pytato` should log additional debug information."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = flag

# }}}


from pytato import analysis, function, tags, transform
from pytato.array import (
    AbstractResultWithNamedArrays,
    AdvancedIndexInContiguousAxes,
    AdvancedIndexInNoncontiguousAxes,
    Array,
    Axis,
    AxisPermutation,
    BasicIndex,
    Concatenate,
    CSRMatmul,
    CSRMatrix,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexLambda,
    IndexRemappingBase,
    InputArgumentBase,
    NamedArray,
    Placeholder,
    ReductionDescriptor,
    Reshape,
    Roll,
    SizeParam,
    SparseMatmul,
    SparseMatrix,
    Stack,
    arange,
    broadcast_to,
    concatenate,
    dot,
    einsum,
    equal,
    expand_dims,
    eye,
    full,
    greater,
    greater_equal,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    make_csr_matrix,
    make_data_wrapper,
    make_dict_of_named_arrays,
    make_placeholder,
    make_size_param,
    matmul,
    maximum,
    minimum,
    not_equal,
    ones,
    reshape,
    roll,
    set_traceback_tag_enabled,
    sparse_matmul,
    squeeze,
    stack,
    transpose,
    vdot,
    where,
    zeros,
)
from pytato.cmath import (
    abs,
    arccos,
    arcsin,
    arctan,
    arctan2,
    conj,
    cos,
    cosh,
    exp,
    imag,
    isnan,
    log,
    log10,
    ones_like,
    real,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    zeros_like,
)
from pytato.distributed.execute import (
    execute_distributed_partition,
    generate_code_for_partition,
)
from pytato.distributed.nodes import (
    DistributedRecv,
    DistributedSend,
    DistributedSendRefHolder,
    make_distributed_recv,
    make_distributed_send,
    make_distributed_send_ref_holder,
    staple_distributed_send,
)
from pytato.distributed.partition import (
    DistributedGraphPart,
    DistributedGraphPartition,
    find_distributed_partition,
)
from pytato.distributed.tags import number_distributed_tags
from pytato.distributed.verify import verify_distributed_partition
from pytato.function import trace_call
from pytato.loopy import LoopyCall
from pytato.pad import pad
from pytato.reductions import all, amax, amin, any, prod, sum
from pytato.target import Target
from pytato.target.loopy import LoopyPyOpenCLTarget
from pytato.target.loopy.codegen import generate_loopy
from pytato.target.python.jax import generate_jax
from pytato.transform.calls import inline_calls, tag_all_calls_to_be_inlined
from pytato.transform.dead_code_elimination import eliminate_dead_code
from pytato.transform.lower_to_index_lambda import to_index_lambda
from pytato.transform.materialize import materialize_with_mpms
from pytato.transform.metadata import unify_axes_tags
from pytato.transform.remove_broadcasts_einsum import rewrite_einsums_with_no_broadcasts
from pytato.visualization import (
    get_dot_graph,
    get_dot_graph_from_partition,
    show_dot_graph,
    show_fancy_placeholder_data_flow,
)


__all__ = (
    "AbstractResultWithNamedArrays",
    "AdvancedIndexInContiguousAxes",
    "AdvancedIndexInNoncontiguousAxes",
    "Array",
    "Axis",
    "AxisPermutation",
    "BasicIndex",
    "CSRMatmul",
    "CSRMatrix",
    "Concatenate",
    "DataWrapper",
    "DictOfNamedArrays",
    "DistributedGraphPart",
    "DistributedGraphPartition",
    "DistributedRecv",
    "DistributedSend",
    "DistributedSendRefHolder",
    "Einsum",
    "IndexBase",
    "IndexLambda",
    "IndexRemappingBase",
    "InputArgumentBase",
    "LoopyCall",
    "LoopyPyOpenCLTarget",
    "NamedArray",
    "Placeholder",
    "ReductionDescriptor",
    "Reshape",
    "Roll",
    "SizeParam",
    "SparseMatmul",
    "SparseMatrix",
    "Stack",
    "Target",
    "abs",
    "all",
    "amax",
    "amin",
    "analysis",
    "any",
    "arange",
    "arccos",
    "arcsin",
    "arctan",
    "arctan2",
    "broadcast_to",
    "concatenate",
    "conj",
    "cos",
    "cosh",
    "dot",
    "dtype",
    "einsum",
    "eliminate_dead_code",
    "equal",
    "execute_distributed_partition",
    "exp",
    "expand_dims",
    "eye",
    "find_distributed_partition",
    "full",
    "function",
    "generate_code_for_partition",
    "generate_jax",
    "generate_loopy",
    "get_dot_graph",
    "get_dot_graph_from_partition",
    "greater",
    "greater_equal",
    "imag",
    "inline_calls",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log10",
    "logical_and",
    "logical_not",
    "logical_or",
    "make_csr_matrix",
    "make_data_wrapper",
    "make_dict_of_named_arrays",
    "make_distributed_recv",
    "make_distributed_send",
    "make_distributed_send_ref_holder",
    "make_placeholder",
    "make_size_param",
    "materialize_with_mpms",
    "matmul",
    "maximum",
    "minimum",
    "not_equal",
    "number_distributed_tags",
    "ones",
    "ones_like",
    "pad",
    "prod",
    "real",
    "reshape",
    "rewrite_einsums_with_no_broadcasts",
    "roll",
    "set_traceback_tag_enabled",
    "show_dot_graph",
    "show_fancy_placeholder_data_flow",
    "sin",
    "sinh",
    "sparse_matmul",
    "sqrt",
    "squeeze",
    "stack",
    "staple_distributed_send",
    "sum",
    "tag_all_calls_to_be_inlined",
    "tags",
    "tan",
    "tanh",
    "to_index_lambda",
    "trace_call",
    "transform",
    "transpose",
    "unify_axes_tags",
    "vdot",
    "verify_distributed_partition",
    "where",
    "zeros",
    "zeros_like",
)
