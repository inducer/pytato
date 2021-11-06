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

from pytato.array import (
        Array, AbstractResultWithNamedArrays, DictOfNamedArrays, Placeholder,
        IndexLambda, NamedArray, DataWrapper, InputArgumentBase, Reshape,
        Einsum, MatrixProduct, Stack, Concatenate, AxisPermutation,
        IndexBase, Roll, IndexRemappingBase, BasicIndex,
        AdvancedIndexInContiguousAxes, AdvancedIndexInNoncontiguousAxes,
        SizeParam,


        make_dict_of_named_arrays,
        make_placeholder, make_size_param, make_data_wrapper,
        einsum,

        matmul, roll, transpose, stack, reshape, concatenate,

        maximum, minimum, where,

        full, zeros, ones,

        eye,

        equal, not_equal, less, less_equal, greater, greater_equal,

        logical_or, logical_and, logical_not,

        dot, vdot, squeeze,

        broadcast_to,

        )
from pytato.reductions import sum, amax, amin, prod, any, all
from pytato.cmath import (abs, sin, cos, tan, arcsin, arccos, arctan, sinh,
                          cosh, tanh, exp, log, log10, isnan, sqrt, conj,
                          arctan2, real, imag)


from pytato.loopy import LoopyCall
from pytato.target.loopy.codegen import generate_loopy
from pytato.target import Target
from pytato.target.loopy import LoopyPyOpenCLTarget
from pytato.visualization import (get_dot_graph, show_dot_graph,
                                  get_ascii_graph, show_ascii_graph,
                                  get_dot_graph_from_partitions)
import pytato.analysis as analysis
import pytato.tags as tags
import pytato.transform as transform

__all__ = (
        "Array", "AbstractResultWithNamedArrays", "DictOfNamedArrays",
        "Placeholder", "IndexLambda", "NamedArray", "LoopyCall",
        "DataWrapper", "InputArgumentBase", "Reshape", "Einsum",
        "MatrixProduct", "Stack", "Concatenate", "AxisPermutation",
        "IndexBase", "Roll", "IndexRemappingBase",
        "AdvancedIndexInContiguousAxes", "AdvancedIndexInNoncontiguousAxes",
        "BasicIndex", "SizeParam",

        "make_dict_of_named_arrays", "make_placeholder", "make_size_param",
        "make_data_wrapper", "einsum",

        "matmul", "roll", "transpose", "stack", "reshape", "concatenate",

        "generate_loopy",

        "Target", "LoopyPyOpenCLTarget",

        "get_dot_graph", "show_dot_graph", "get_ascii_graph",
        "show_ascii_graph", "get_dot_graph_from_partitions",

        "abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh",
        "tanh", "exp", "log", "log10", "isnan", "sqrt", "conj", "arctan2",

        "maximum", "minimum", "where",

        "full", "zeros", "ones", "eye",

        "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",

        "logical_or", "logical_and", "logical_not",

        "sum", "amax", "amin", "prod", "all", "any",

        "real", "imag",

        "dot", "vdot", "squeeze",

        "broadcast_to",

        # sub-modules
        "analysis", "tags", "transform",

)
