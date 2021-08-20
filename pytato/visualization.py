from __future__ import annotations

__copyright__ = """
Copyright (C) 2020 Matt Wala
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
from typing import Callable, Dict, Union, Iterator, List, Mapping

from pytools import UniqueNameGenerator
from pytools.codegen import CodeGenerator as CodeGeneratorBase
from pytools.tag import TagsType

from pytato.array import (
        Array, DictOfNamedArrays, IndexLambda, InputArgumentBase,
        Stack, ShapeType, Einsum)
from pytato.codegen import normalize_outputs
import pytato.transform


__doc__ = """
.. currentmodule:: pytato

.. autofunction:: get_dot_graph
.. autofunction:: show_dot_graph
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


class ArrayToDotNodeInfoMapper(pytato.transform.Mapper):
    def get_common_dot_info(self, expr: Array) -> DotNodeInfo:
        title = type(expr).__name__
        fields = dict(shape=stringify_shape(expr.shape),
                dtype=str(expr.dtype),
                tags=stringify_tags(expr.tags))
        edges: Dict[str, Array] = {}
        return DotNodeInfo(title, fields, edges)

    def handle_unsupported_array(self, expr: Array,  # type: ignore
            nodes: Dict[Array, DotNodeInfo]) -> None:
        # Default handler, does its best to guess how to handle fields.
        if expr in nodes:
            return
        info = self.get_common_dot_info(expr)

        for field in expr._fields:
            if field in info.fields:
                continue
            attr = getattr(expr, field)

            if isinstance(attr, Array):
                self.rec(attr, nodes)
                info.edges[field] = attr
            elif isinstance(attr, tuple):
                info.fields[field] = stringify_shape(attr)
            else:
                info.fields[field] = str(attr)

        nodes[expr] = info

    def map_index_lambda(self, expr: IndexLambda,
            nodes: Dict[Array, DotNodeInfo]) -> None:
        if expr in nodes:
            return

        info = self.get_common_dot_info(expr)
        info.fields["expr"] = str(expr.expr)

        for name, val in expr.bindings.items():
            self.rec(val, nodes)
            info.edges[name] = val

        nodes[expr] = info

    def map_stack(self, expr: Stack, nodes: Dict[Array, DotNodeInfo]) -> None:
        if expr in nodes:
            return

        info = self.get_common_dot_info(expr)
        info.fields["axis"] = str(expr.axis)

        for i, array in enumerate(expr.arrays):
            self.rec(array, nodes)
            info.edges[str(i)] = array

        nodes[expr] = info

    def map_einsum(self, expr: Einsum,
                   nodes: Dict[Array, DotNodeInfo]) -> None:
        if expr in nodes:
            return

        info = self.get_common_dot_info(expr)

        for access_descr, val in zip(expr.access_descriptors, expr.args):
            self.rec(val, nodes)
            info.edges[str(access_descr)] = val

        nodes[expr] = info


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


def _emit_array(emit: DotEmitter, info: DotNodeInfo, id: str) -> None:
    td_attrib = 'border="0"'
    table_attrib = 'border="0" cellborder="1" cellspacing="0"'

    rows = ['<tr><td colspan="2" %s>%s</td></tr>'
            % (td_attrib, dot_escape(info.title))]

    for name, field in info.fields.items():
        rows.append(
                "<tr><td %s>%s:</td><td %s>%s</td></tr>"
                % (td_attrib, dot_escape(name), td_attrib, dot_escape(field)))

    table = "<table %s>\n%s</table>" % (table_attrib, "".join(rows))
    emit("%s [label=<%s>]" % (id, table))


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

    nodes: Dict[Array, DotNodeInfo] = {}
    mapper = ArrayToDotNodeInfoMapper()
    for elem in outputs._data.values():
        mapper(elem, nodes)

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


def show_dot_graph(result: Union[Array, DictOfNamedArrays]) -> None:
    """Show a graph representing the computation of *result* in a browser.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`).
    """
    dot_code: str

    if isinstance(result, str):
        dot_code = result
    else:
        dot_code = get_dot_graph(result)

    from pymbolic.imperative.utils import show_dot
    show_dot(dot_code)
