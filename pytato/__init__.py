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

from numpy import dtype


# {{{ debug control

import os
try:
    v = os.environ.get("PYTATO_DEBUG")
    if v is None:
        v = ""

    DEBUG_ENABLED = bool(eval(v))
except Exception:
    DEBUG_ENABLED = False


def set_debug_enabled(flag: bool) -> None:
    """Set whether :mod:`pytato` should log additional debug information."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = flag

# }}}


from pytato.array import (
        Array, AbstractResultWithNamedArrays, DictOfNamedArrays, Placeholder,
        IndexLambda, NamedArray, DataWrapper, InputArgumentBase, Reshape,
        Einsum, Stack, Concatenate, AxisPermutation,
        IndexBase, Roll, IndexRemappingBase, BasicIndex,
        AdvancedIndexInContiguousAxes, AdvancedIndexInNoncontiguousAxes,
        SizeParam, Axis, ReductionDescriptor,

        make_dict_of_named_arrays,
        make_placeholder, make_size_param, make_data_wrapper,
        einsum,

        matmul, roll, transpose, stack, reshape, concatenate,
        expand_dims,

        maximum, minimum, where,

        full, zeros, ones, eye, arange,

        equal, not_equal, less, less_equal, greater, greater_equal,

        logical_or, logical_and, logical_not,

        dot, vdot, squeeze,

        broadcast_to,

        )
from pytato.reductions import sum, amax, amin, prod, any, all
from pytato.cmath import (abs, sin, cos, tan, arcsin, arccos, arctan, sinh,
                          cosh, tanh, exp, log, log10, isnan, sqrt, conj,
                          arctan2, real, imag)
from pytato.pad import pad


from pytato.loopy import LoopyCall
from pytato.target.loopy.codegen import generate_loopy
from pytato.target import Target
from pytato.target.loopy import LoopyPyOpenCLTarget
from pytato.target.python.jax import generate_jax
from pytato.visualization import (get_dot_graph, show_dot_graph,
                                  get_dot_graph_from_partition,
                                  show_fancy_placeholder_data_flow,
                                  )
from pytato.transform.calls import tag_all_calls_to_be_inlined, inline_calls
import pytato.analysis as analysis
import pytato.tags as tags
import pytato.function as function
import pytato.transform as transform
from pytato.distributed.nodes import (make_distributed_send, make_distributed_recv,
                                DistributedRecv, DistributedSend,
                                DistributedSendRefHolder,
                                make_distributed_send_ref_holder,
                                staple_distributed_send)
from pytato.distributed.partition import (
        find_distributed_partition, DistributedGraphPart, DistributedGraphPartition)
from pytato.distributed.tags import number_distributed_tags
from pytato.distributed.execute import (
        generate_code_for_partition, execute_distributed_partition)
from pytato.distributed.verify import verify_distributed_partition

from pytato.transform.lower_to_index_lambda import to_index_lambda
from pytato.transform.remove_broadcasts_einsum import (
    rewrite_einsums_with_no_broadcasts)
from pytato.transform.metadata import unify_axes_tags
from pytato.function import trace_call
from pytato.array import enable_traceback_tag

__all__ = (
        "dtype",

        "Array", "AbstractResultWithNamedArrays", "DictOfNamedArrays",
        "Placeholder", "IndexLambda", "NamedArray", "LoopyCall",
        "DataWrapper", "InputArgumentBase", "Reshape", "Einsum",
        "Stack", "Concatenate", "AxisPermutation",
        "IndexBase", "Roll", "IndexRemappingBase",
        "AdvancedIndexInContiguousAxes", "AdvancedIndexInNoncontiguousAxes",
        "BasicIndex", "SizeParam", "Axis", "ReductionDescriptor",

        "make_dict_of_named_arrays", "make_placeholder", "make_size_param",
        "make_data_wrapper", "einsum",

        "matmul", "roll", "transpose", "stack", "reshape", "expand_dims",
        "concatenate",

        "generate_loopy", "generate_jax",

        "Target", "LoopyPyOpenCLTarget",

        "get_dot_graph", "show_dot_graph",
        "get_dot_graph_from_partition",
        "show_fancy_placeholder_data_flow",

        "abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh",
        "tanh", "exp", "log", "log10", "isnan", "sqrt", "conj", "arctan2",

        "maximum", "minimum", "where",

        "full", "zeros", "ones", "eye", "arange",

        "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",

        "logical_or", "logical_and", "logical_not",

        "sum", "amax", "amin", "prod", "all", "any",

        "real", "imag",

        "dot", "vdot", "squeeze",

        "broadcast_to", "pad",

        "trace_call",

        "make_distributed_recv", "make_distributed_send", "DistributedRecv",
        "DistributedSend", "make_distributed_send_ref_holder",
        "staple_distributed_send", "DistributedSendRefHolder",

        "DistributedGraphPart",
        "DistributedGraphPartition",
        "tag_all_calls_to_be_inlined", "inline_calls",

        "find_distributed_partition",

        "number_distributed_tags",
        "generate_code_for_partition",
        "execute_distributed_partition",
        "verify_distributed_partition",

        "to_index_lambda",

        "rewrite_einsums_with_no_broadcasts",

        "unify_axes_tags",

        # sub-modules
        "analysis", "tags", "transform", "function",

        "enable_traceback_tag",
)
