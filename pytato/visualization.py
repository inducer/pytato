from __future__ import annotations

__copyright__ = """
Copyright (C) 2020 Matt Wala
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


import contextlib
import dataclasses
import html
from typing import Callable, Dict, Union, Iterator, List, Mapping, Hashable

from pytools import UniqueNameGenerator
from pytools.codegen import CodeGenerator as CodeGeneratorBase
from pytools.tag import TagsType

from pytato.array import (
        Array, DataWrapper, DictOfNamedArrays, IndexLambda, InputArgumentBase,
        Stack, ShapeType, Einsum, Placeholder)
from pytato.codegen import normalize_outputs
from pytato.transform import CachedMapper

from pytato.partition import GraphPartition


__doc__ = """
.. currentmodule:: pytato

.. autofunction:: get_dot_graph
.. autofunction:: get_dot_graph_from_partition
.. autofunction:: show_dot_graph
.. autofunction:: get_ascii_graph
.. autofunction:: show_ascii_graph
"""


# {{{ array -> dot node converter

@dataclasses.dataclass
class DotNodeInfo:
    title: str
    fields: Dict[str, str]
    edges: Dict[str, Array]


def stringify_tags(tags: TagsType) -> str:
    components = sorted(str(elem) for elem in tags)
    return "{" + ", ".join(components) + "}"


def stringify_shape(shape: ShapeType) -> str:
    components = [str(elem) for elem in shape]
    if not components:
        components = [","]
    elif len(components) == 1:
        components[0] += ","
    return "(" + ", ".join(components) + ")"


class ArrayToDotNodeInfoMapper(CachedMapper[Array]):
    def __init__(self) -> None:
        super().__init__()
        self.nodes: Dict[Array, DotNodeInfo] = {}

    def get_common_dot_info(self, expr: Array) -> DotNodeInfo:
        title = type(expr).__name__
        fields = dict(addr=hex(id(expr)),
                shape=stringify_shape(expr.shape),
                dtype=str(expr.dtype),
                tags=stringify_tags(expr.tags))
        edges: Dict[str, Array] = {}
        return DotNodeInfo(title, fields, edges)

    # type-ignore-reason: incompatible with supertype
    def handle_unsupported_array(self,  # type: ignore[override]
            expr: Array) -> None:
        # Default handler, does its best to guess how to handle fields.
        info = self.get_common_dot_info(expr)

        for field in expr._fields:
            if field in info.fields:
                continue
            attr = getattr(expr, field)

            if isinstance(attr, Array):
                self.rec(attr)
                info.edges[field] = attr
            elif isinstance(attr, tuple):
                info.fields[field] = stringify_shape(attr)
            else:
                info.fields[field] = str(attr)

        self.nodes[expr] = info

    def map_data_wrapper(self, expr: DataWrapper) -> None:
        info = self.get_common_dot_info(expr)
        if expr.name is not None:
            info.fields["name"] = expr.name

        # Only show summarized data
        import numpy as np
        with np.printoptions(threshold=4, precision=2):
            info.fields["data"] = str(expr.data)

        self.nodes[expr] = info

    def map_index_lambda(self, expr: IndexLambda) -> None:
        info = self.get_common_dot_info(expr)
        info.fields["expr"] = str(expr.expr)

        for name, val in expr.bindings.items():
            self.rec(val)
            info.edges[name] = val

        self.nodes[expr] = info

    def map_stack(self, expr: Stack) -> None:
        info = self.get_common_dot_info(expr)
        info.fields["axis"] = str(expr.axis)

        for i, array in enumerate(expr.arrays):
            self.rec(array)
            info.edges[str(i)] = array

        self.nodes[expr] = info

    def map_einsum(self, expr: Einsum) -> None:
        info = self.get_common_dot_info(expr)

        for iarg, (access_descr, val) in enumerate(zip(expr.access_descriptors,
                                                       expr.args)):
            self.rec(val)
            info.edges[f"{iarg}: {access_descr}"] = val

        self.nodes[expr] = info


def dot_escape(s: str) -> str:
    # "\" and HTML are significant in graphviz.
    return html.escape(s.replace("\\", "\\\\"))


class DotEmitter(CodeGeneratorBase):
    @contextlib.contextmanager
    def block(self, name: str) -> Iterator[None]:
        self(name + " {")
        self.indent()
        yield
        self.dedent()
        self("}")


def _emit_array(emit: DotEmitter, info: DotNodeInfo, id: str,
                color: str = "white") -> None:
    td_attrib = 'border="0"'
    table_attrib = 'border="0" cellborder="1" cellspacing="0"'

    rows = ['<tr><td colspan="2" %s>%s</td></tr>'
            % (td_attrib, dot_escape(info.title))]

    for name, field in info.fields.items():
        field_content = dot_escape(field).replace("\n", "<br/>")
        rows.append(
                f"<tr><td {td_attrib}>{dot_escape(name)}:</td><td {td_attrib}>"
                f"<FONT FACE='monospace'>{field_content}</FONT></td></tr>"
        )
    table = "<table %s>\n%s</table>" % (table_attrib, "".join(rows))
    emit("%s [label=<%s> style=filled fillcolor=%s]" % (id, table, color))


def _emit_name_cluster(emit: DotEmitter, names: Mapping[str, Array],
        array_to_id: Mapping[Array, str], id_gen: Callable[[str], str],
        label: str) -> None:
    edges = []

    with emit.block("subgraph cluster_%s" % label):
        emit("node [shape=ellipse]")
        emit('label="%s"' % label)

        for name, array in names.items():
            name_id = id_gen(label)
            emit('%s [label="%s"]' % (name_id, dot_escape(name)))
            array_id = array_to_id[array]
            # Edges must be outside the cluster.
            edges.append((name_id, array_id))

    for name_id, array_id in edges:
        emit("%s -> %s" % (name_id, array_id))

# }}}


def get_dot_graph(result: Union[Array, DictOfNamedArrays]) -> str:
    r"""Return a string in the `dot <https://graphviz.org>`_ language depicting the
    graph of the computation of *result*.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`).
    """
    outputs: DictOfNamedArrays = normalize_outputs(result)
    del result

    mapper = ArrayToDotNodeInfoMapper()
    for elem in outputs._data.values():
        mapper(elem)

    nodes = mapper.nodes

    input_arrays: List[Array] = []
    internal_arrays: List[Array] = []
    array_to_id: Dict[Array, str] = {}

    id_gen = UniqueNameGenerator()
    for array in nodes:
        array_to_id[array] = id_gen("array")
        if isinstance(array, InputArgumentBase):
            input_arrays.append(array)
        else:
            internal_arrays.append(array)

    emit = DotEmitter()

    with emit.block("digraph computation"):
        emit("node [shape=rectangle]")

        # Emit inputs.
        with emit.block("subgraph cluster_Inputs"):
            emit('label="Inputs"')
            for array in input_arrays:
                _emit_array(emit, nodes[array], array_to_id[array])

        # Emit non-inputs.
        for array in internal_arrays:
            _emit_array(emit, nodes[array], array_to_id[array])

        # Emit edges.
        for array, node in nodes.items():
            for label, tail_array in node.edges.items():
                tail = array_to_id[tail_array]
                head = array_to_id[array]
                emit('%s -> %s [label="%s"]' % (tail, head, dot_escape(label)))

        # Emit output/namespace name mappings.
        _emit_name_cluster(emit, outputs._data, array_to_id, id_gen, label="Outputs")

    return emit.get()


def get_dot_graph_from_partition(partition: GraphPartition) -> str:
    r"""Return a string in the `dot <https://graphviz.org>`_ language depicting the
    graph of the partitioned computation of *partition*.

    :arg partition: Outputs of :func:`~pytato.partition.find_partition`.
    """
    # Maps each partition to a dict of its arrays with the node info
    part_id_to_node_info: Dict[Hashable, Dict[Array, DotNodeInfo]] = {}

    for part in partition.parts.values():
        mapper = ArrayToDotNodeInfoMapper()
        for out_name in part.output_names:
            mapper(partition.var_name_to_result[out_name])

        part_id_to_node_info[part.pid] = mapper.nodes

    id_gen = UniqueNameGenerator()

    emit = DotEmitter()

    emitted_placeholders = set()

    with emit.block("digraph computation"):
        emit("node [shape=rectangle]")
        array_to_id: Dict[Array, str] = {}

        # First pass: generate names for all nodes
        for part in partition.parts.values():
            for array, _ in part_id_to_node_info[part.pid].items():
                array_to_id[array] = id_gen("array")

        # Second pass: emit the graph.
        for part in partition.parts.values():
            part_node_to_info = part_id_to_node_info[part.pid]
            input_arrays: List[Array] = []
            internal_arrays: List[Array] = []

            for array, _ in part_node_to_info.items():
                if isinstance(array, InputArgumentBase):
                    input_arrays.append(array)
                else:
                    internal_arrays.append(array)

            # {{{ emit inputs

            # Placeholders are unique, i.e. the same Placeholder object may be
            # shared among partitions. Therefore, they should not live inside
            # the (dot) subgraph, otherwise they would be forced into multiple
            # subgraphs.

            for array in input_arrays:
                # Non-Placeholders are emitted *inside* their subgraphs below.
                if isinstance(array, Placeholder):
                    if array not in emitted_placeholders:
                        _emit_array(emit, part_node_to_info[array],
                                    array_to_id[array], "deepskyblue")
                        emitted_placeholders.add(array)

                        # Emit cross-partition edges
                        tgt = array_to_id[partition.var_name_to_result[array.name]]
                        emit(f"{tgt} -> {array_to_id[array]} [style=dashed]")

            # }}}

            with emit.block(f'subgraph "cluster_part_{part.pid}"'):
                emit("style=dashed")
                emit(f'label="{part.pid}"')

                for array in input_arrays:
                    if not isinstance(array, Placeholder):
                        _emit_array(emit, part_node_to_info[array],
                                    array_to_id[array], "deepskyblue")

                # Emit internal nodes
                for array in internal_arrays:
                    _emit_array(emit, part_node_to_info[array], array_to_id[array])

            # Emit intra-partition edges
            for array, node in part_node_to_info.items():
                for label, tail_array in node.edges.items():
                    tail = array_to_id[tail_array]
                    head = array_to_id[array]
                    emit('%s -> %s [label="%s"]' %
                        (tail, head, dot_escape(label)))

    return emit.get()


def show_dot_graph(result: Union[str, Array, DictOfNamedArrays, GraphPartition]) \
        -> None:
    """Show a graph representing the computation of *result* in a browser.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`) or the output of :func:`get_dot_graph`,
        or the output of :func:`~pytato.partition.find_partition`.
    """
    dot_code: str

    if isinstance(result, str):
        dot_code = result
    elif isinstance(result, GraphPartition):
        dot_code = get_dot_graph_from_partition(result)
    else:
        dot_code = get_dot_graph(result)

    from pymbolic.imperative.utils import show_dot
    show_dot(dot_code)


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
    internal_arrays: List[Array] = []
    array_to_id: Dict[Array, str] = {}

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
    asciidag_nodes: Dict[Array, Node] = {}

    from collections import defaultdict
    asciidag_edges: Dict[Array, List[Array]] = defaultdict(list)

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
    """Show a graph representing the computation of *result* in a browser.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`) or the output of :func:`get_dot_graph`.
    """

    print(get_ascii_graph(result, use_color=True))
# }}}
