"""
.. currentmodule:: pytato

.. autofunction:: get_ascii_graph
.. autofunction:: show_ascii_graph
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

from typing import Union, List, Dict
from pytato.transform import ArrayOrNames
from pytato.array import Array, DictOfNamedArrays, InputArgumentBase
from pytato.visualization.dot import ArrayToDotNodeInfoMapper
from pytato.codegen import normalize_outputs
from pytools import UniqueNameGenerator


# {{{ Show ASCII representation of DAG

def get_ascii_graph(result: Union[Array, DictOfNamedArrays],
                    use_color: bool = True) -> str:
    """Return a string representing the computation of *result*
    using the `asciidag <https://pypi.org/project/asciidag/>`_ package.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`).
    :arg use_color: Colorized output
    """
    outputs: DictOfNamedArrays = normalize_outputs(result)
    del result

    mapper = ArrayToDotNodeInfoMapper()
    for elem in outputs._data.values():
        mapper(elem)

    nodes = mapper.nodes

    input_arrays: List[Array] = []
    internal_arrays: List[ArrayOrNames] = []
    array_to_id: Dict[ArrayOrNames, str] = {}

    id_gen = UniqueNameGenerator()
    for array in nodes:
        array_to_id[array] = id_gen("array")
        if isinstance(array, InputArgumentBase):
            input_arrays.append(array)
        else:
            internal_arrays.append(array)

    # Since 'asciidag' prints the DAG from top to bottom (ie, with the inputs
    # at the bottom), we need to invert our representation of it, that is, the
    # 'parents' constructor argument to Node() actually means 'children'.
    from asciidag.node import Node  # type: ignore[import]
    asciidag_nodes: Dict[ArrayOrNames, Node] = {}

    from collections import defaultdict
    asciidag_edges: Dict[ArrayOrNames, List[ArrayOrNames]] = defaultdict(list)

    # Reverse edge directions
    for array in internal_arrays:
        for _, v in nodes[array].edges.items():
            asciidag_edges[v].append(array)

    # Add the internal arrays in reversed order
    for array in internal_arrays[::-1]:
        ary_edges = [asciidag_nodes[v] for v in asciidag_edges[array]]

        if array == internal_arrays[-1]:
            ary_edges.append(Node("Outputs"))

        asciidag_nodes[array] = Node(f"{nodes[array].title}",
                              parents=ary_edges)

    # Add the input arrays last since they have no predecessors
    for array in input_arrays:
        ary_edges = [asciidag_nodes[v] for v in asciidag_edges[array]]
        asciidag_nodes[array] = Node(f"{nodes[array].title}", parents=ary_edges)

    input_node = Node("Inputs", parents=[asciidag_nodes[v] for v in input_arrays])

    from asciidag.graph import Graph  # type: ignore[import]
    from io import StringIO

    f = StringIO()
    graph = Graph(fh=f, use_color=use_color)

    graph.show_nodes([input_node])

    # Get the graph and remove trailing whitespace
    res = "\n".join([s.rstrip() for s in f.getvalue().split("\n")])

    return res


def show_ascii_graph(result: Union[Array, DictOfNamedArrays]) -> None:
    """Print a graph representing the computation of *result* to stdout using the
    `asciidag <https://pypi.org/project/asciidag/>`_ package.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`) or the output of :func:`get_dot_graph`.
    """

    print(get_ascii_graph(result, use_color=True))
