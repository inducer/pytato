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

from typing import (TYPE_CHECKING, Callable, Dict, Union, Iterator, List,
        Mapping, Hashable, Any, FrozenSet)

from pytools import UniqueNameGenerator
from pytools.tag import Tag
from pytools.codegen import CodeGenerator as CodeGeneratorBase
from pytato.loopy import LoopyCall

from pytato.array import (
        Array, DataWrapper, DictOfNamedArrays, IndexLambda, InputArgumentBase,
        Stack, ShapeType, Einsum, Placeholder, AbstractResultWithNamedArrays,
        IndexBase)

from pytato.codegen import normalize_outputs
from pytato.transform import CachedMapper, ArrayOrNames

from pytato.partition import GraphPartition
from pytato.distributed import DistributedGraphPart

if TYPE_CHECKING:
    from pytato.distributed import DistributedSendRefHolder


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
    edges: Dict[str, ArrayOrNames]


def stringify_created_at(tags: TagsType) -> str:
    from pytato.tags import CreatedAt
    for tag in tags:
        if isinstance(tag, CreatedAt):
            return tag.traceback.short_str()

    return "<unknown>"


def stringify_tags(tags: TagsType) -> str:
    # The CreatedAt tag is handled in stringify_created_at()
    from pytato.tags import CreatedAt
    tags = frozenset(tag for tag in tags if not isinstance(tag, CreatedAt))

    components = sorted(str(elem) for elem in tags)
    return "{" + ", ".join(components) + "}"


def stringify_shape(shape: ShapeType) -> str:
    from pytato.tags import CreatedAt
    from pytato import SizeParam

    new_elems = set()
    for elem in shape:
        if not isinstance(elem, SizeParam):
            new_elems.add(elem)
        else:
            # Remove CreatedAt tags from SizeParam
            new_elem = elem.copy(
                    tags=frozenset(tag for tag in elem.tags
                                    if not isinstance(tag, CreatedAt)))
            new_elems.add(new_elem)

    components = [str(elem) for elem in new_elems]
    if not components:
        components = [","]
    elif len(components) == 1:
        components[0] += ","
    return "(" + ", ".join(components) + ")"


class ArrayToDotNodeInfoMapper(CachedMapper[ArrayOrNames]):
    def __init__(self) -> None:
        super().__init__()
        self.nodes: Dict[ArrayOrNames, DotNodeInfo] = {}

    def get_common_dot_info(self, expr: Array) -> DotNodeInfo:
        title = type(expr).__name__
        fields = dict(addr=hex(id(expr)),
                created_at=stringify_created_at(expr.tags),
                shape=stringify_shape(expr.shape),
                dtype=str(expr.dtype),
                tags=stringify_tags(expr.tags))

        edges: Dict[str, ArrayOrNames] = {}
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

            elif isinstance(attr, AbstractResultWithNamedArrays):
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

    map_concatenate = map_stack

    def map_basic_index(self, expr: IndexBase) -> None:
        info = self.get_common_dot_info(expr)

        from pytato.scalar_expr import INT_CLASSES

        indices_parts = []
        for i, index in enumerate(expr.indices):
            if isinstance(index, INT_CLASSES):
                indices_parts.append(str(index))

            elif isinstance(index, Array):
                label = f"i{i}"
                self.rec(index)
                indices_parts.append(label)
                info.edges[label] = index

            elif index is None:
                indices_parts.append("newaxis")

            else:
                indices_parts.append(str(index))

        info.fields["indices"] = ", ".join(indices_parts)

        self.rec(expr.array)
        info.edges["array"] = expr.array

        self.nodes[expr] = info

    map_contiguous_advanced_index = map_basic_index
    map_non_contiguous_advanced_index = map_basic_index

    def map_einsum(self, expr: Einsum) -> None:
        info = self.get_common_dot_info(expr)

        for iarg, (access_descr, val) in enumerate(zip(expr.access_descriptors,
                                                       expr.args)):
            self.rec(val)
            info.edges[f"{iarg}: {access_descr}"] = val

        self.nodes[expr] = info

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        edges: Dict[str, ArrayOrNames] = {}
        for name, val in expr._data.items():
            edges[name] = val
            self.rec(val)

        self.nodes[expr] = DotNodeInfo(
                title=type(expr).__name__,
                fields={},
                edges=edges)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        edges: Dict[str, ArrayOrNames] = {}
        for name, arg in expr.bindings.items():
            if isinstance(arg, Array):
                edges[name] = arg
                self.rec(arg)

        self.nodes[expr] = DotNodeInfo(
                title=type(expr).__name__,
                fields=dict(addr=hex(id(expr)),
                            entrypoint=expr.entrypoint),
                edges=edges)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> None:

        info = self.get_common_dot_info(expr)

        self.rec(expr.passthrough_data)
        info.edges["passthrough"] = expr.passthrough_data

        self.rec(expr.send.data)
        info.edges["sent"] = expr.send.data

        info.fields["dest_rank"] = str(expr.send.dest_rank)

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


def _emit_array(emit: DotEmitter, title: str, fields: Dict[str, str],
        dot_node_id: str, color: str = "white") -> None:
    td_attrib = 'border="0"'
    table_attrib = 'border="0" cellborder="1" cellspacing="0"'

    rows = [f"<tr><td colspan='2' {td_attrib}>{dot_escape(title)}</td></tr>"]

    created_at = fields.pop("created_at", "")
    tooltip = dot_escape(created_at)

    for name, field in fields.items():
        field_content = dot_escape(field).replace("\n", "<br/>")
        rows.append(
                f"<tr><td {td_attrib}>{dot_escape(name)}:</td><td {td_attrib}>"
                f"<FONT FACE='monospace'>{field_content}</FONT></td></tr>"
        )
    table = f"<table {table_attrib}>\n{''.join(rows)}</table>"
    emit(f"{dot_node_id} [label=<{table}> style=filled fillcolor={color} "
         f'tooltip="{tooltip}"]')


def _emit_name_cluster(emit: DotEmitter, names: Mapping[str, ArrayOrNames],
        array_to_id: Mapping[ArrayOrNames, str], id_gen: Callable[[str], str],
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
    internal_arrays: List[ArrayOrNames] = []
    array_to_id: Dict[ArrayOrNames, str] = {}

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
                _emit_array(emit,
                        nodes[array].title, nodes[array].fields, array_to_id[array])

        # Emit non-inputs.
        for array in internal_arrays:
            _emit_array(emit,
                    nodes[array].title, nodes[array].fields, array_to_id[array])

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
    part_id_to_node_info: Dict[Hashable, Dict[ArrayOrNames, DotNodeInfo]] = {}

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
        array_to_id: Dict[ArrayOrNames, str] = {}

        # First pass: generate names for all nodes
        for part in partition.parts.values():
            for array, _ in part_id_to_node_info[part.pid].items():
                array_to_id[array] = id_gen("array")

        # Second pass: emit the graph.
        for part in partition.parts.values():
            # {{{ emit receives nodes if distributed

            if isinstance(part, DistributedGraphPart):
                part_dist_recv_var_name_to_node_id = {}
                for name, recv in (
                        part.input_name_to_recv_node.items()):
                    node_id = id_gen("recv")
                    _emit_array(emit, "Recv", {
                        "shape": stringify_shape(recv.shape),
                        "dtype": str(recv.dtype),
                        "src_rank": str(recv.src_rank),
                        "comm_tag": str(recv.comm_tag),
                        }, node_id)

                    part_dist_recv_var_name_to_node_id[name] = node_id
            else:
                part_dist_recv_var_name_to_node_id = {}

            # }}}

            part_node_to_info = part_id_to_node_info[part.pid]
            input_arrays: List[Array] = []
            internal_arrays: List[ArrayOrNames] = []

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
                        _emit_array(emit,
                                    part_node_to_info[array].title,
                                    part_node_to_info[array].fields,
                                    array_to_id[array], "deepskyblue")

                        # Emit cross-partition edges
                        if array.name in part_dist_recv_var_name_to_node_id:
                            tgt = part_dist_recv_var_name_to_node_id[array.name]
                            emit(f"{tgt} -> {array_to_id[array]} [style=dotted]")
                            emitted_placeholders.add(array)
                        elif array.name in part.user_input_names:
                            # These are placeholders for external input. They
                            # are cleanly associated with a single partition
                            # and thus emitted below.
                            pass
                        else:
                            # placeholder for a value from a different partition
                            tgt = array_to_id[
                                    partition.var_name_to_result[array.name]]
                            emit(f"{tgt} -> {array_to_id[array]} [style=dashed]")
                            emitted_placeholders.add(array)

            # }}}

            with emit.block(f'subgraph "cluster_part_{part.pid}"'):
                emit("style=dashed")
                emit(f'label="{part.pid}"')

                for array in input_arrays:
                    if (not isinstance(array, Placeholder)
                            or array.name in part.user_input_names):
                        _emit_array(emit,
                                    part_node_to_info[array].title,
                                    part_node_to_info[array].fields,
                                    array_to_id[array], "deepskyblue")

                # Emit internal nodes
                for array in internal_arrays:
                    _emit_array(emit,
                                part_node_to_info[array].title,
                                part_node_to_info[array].fields,
                                array_to_id[array])

                # {{{ emit send nodes if distributed

                deferred_send_edges = []
                if isinstance(part, DistributedGraphPart):
                    for name, send in (
                            part.output_name_to_send_node.items()):
                        node_id = id_gen("send")
                        _emit_array(emit, "Send", {
                            "dest_rank": str(send.dest_rank),
                            "comm_tag": str(send.comm_tag),
                            }, node_id)

                        deferred_send_edges.append(
                                f"{array_to_id[send.data]} -> {node_id}"
                                f'[style=dotted, label="{dot_escape(name)}"]')

                # }}}

            # If an edge is emitted in a subgraph, it drags its nodes into the
            # subgraph, too. Not what we want.
            for edge in deferred_send_edges:
                emit(edge)

            # Emit intra-partition edges
            for array, node in part_node_to_info.items():
                for label, tail_array in node.edges.items():
                    tail = array_to_id[tail_array]
                    head = array_to_id[array]
                    emit('%s -> %s [label="%s"]' %
                        (tail, head, dot_escape(label)))

    return emit.get()


def show_dot_graph(result: Union[str, Array, DictOfNamedArrays, GraphPartition],
        **kwargs: Any) -> None:
    """Show a graph representing the computation of *result* in a browser.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`) or the output of :func:`get_dot_graph`,
        or the output of :func:`~pytato.partition.find_partition`.
    :arg kwargs: Passed on to :func:`pymbolic.imperative.utils.show_dot` unmodified.
    """
    dot_code: str

    if isinstance(result, str):
        dot_code = result
    elif isinstance(result, GraphPartition):
        dot_code = get_dot_graph_from_partition(result)
    else:
        dot_code = get_dot_graph(result)

    from pymbolic.imperative.utils import show_dot
    show_dot(dot_code, **kwargs)


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
# }}}
