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


from functools import partial
import html
import attrs
import gc

from typing import (TYPE_CHECKING, Callable, Dict, Tuple, Union, List,
        Mapping, Any, FrozenSet, Set, Optional)

from pytools import UniqueNameGenerator
from pytools.tag import Tag
from pytato.loopy import LoopyCall
from pytato.function import Call, FunctionDefinition, NamedCallResult
from pytato.tags import FunctionIdentifier
from pytools.codegen import remove_common_indentation

from pytato.array import (
        Array, DataWrapper, DictOfNamedArrays, IndexLambda, InputArgumentBase,
        Stack, ShapeType, Einsum, Placeholder, AbstractResultWithNamedArrays,
        IndexBase)

from pytato.codegen import normalize_outputs
from pytato.transform import CachedMapper, ArrayOrNames, InputGatherer

from pytato.distributed.partition import (
        DistributedGraphPartition, DistributedGraphPart, PartId)

from pytato.distributed.nodes import DistributedSendRefHolder


__doc__ = """
.. currentmodule:: pytato

.. autofunction:: get_dot_graph
.. autofunction:: get_dot_graph_from_partition
.. autofunction:: show_dot_graph
"""


# {{{ _DotEmitter

@attrs.define
class _SubgraphTree:
    contents: Optional[List[str]]
    subgraphs: Dict[str, _SubgraphTree]


class DotEmitter:
    def __init__(self) -> None:
        self.subgraph_to_lines: Dict[Tuple[str, ...], List[str]] = {}

    def __call__(self, subgraph_path: Tuple[str, ...], s: str) -> None:
        line_list = self.subgraph_to_lines.setdefault(subgraph_path, [])

        if not s.strip():
            line_list.append("")
        else:
            if "\n" in s:
                s = remove_common_indentation(s)

            for line in s.split("\n"):
                line_list.append(line)

    def _get_subgraph_tree(self) -> _SubgraphTree:
        subgraph_tree = _SubgraphTree(contents=None, subgraphs={})

        def insert_into_subgraph_tree(
                root: _SubgraphTree, path: Tuple[str, ...], contents: List[str]
                ) -> None:
            if not path:
                assert root.contents is None
                root.contents = contents

            else:
                subgraph = root.subgraphs.setdefault(
                        path[0],
                        _SubgraphTree(contents=None, subgraphs={}))

                insert_into_subgraph_tree(subgraph, path[1:], contents)

        for sgp, lines in self.subgraph_to_lines.items():
            insert_into_subgraph_tree(subgraph_tree, sgp, lines)

        return subgraph_tree

    def generate(self) -> str:
        result = ["digraph computation {"]

        indent_level = 1

        def emit_subgraph(sg: _SubgraphTree) -> None:
            nonlocal indent_level

            indent = (indent_level*4)*" "
            if sg.contents:
                for ln in sg.contents:
                    result.append(indent + ln)

            indent_level += 1
            for sg_name, sub_sg in sg.subgraphs.items():
                result.append(f"{indent}subgraph {sg_name} {{")
                emit_subgraph(sub_sg)
                result.append(f"{indent}" "}")
            indent_level -= 1

        emit_subgraph(self._get_subgraph_tree())

        result.append("}")

        return "\n".join(result)

# }}}


# {{{ array -> dot node converter

@attrs.define
class _DotNodeInfo:
    title: str
    fields: Dict[str, Any]
    edges: Dict[str, Union[ArrayOrNames, FunctionDefinition]]


def stringify_tags(tags: FrozenSet[Optional[Tag]]) -> str:
    components = sorted(str(elem) for elem in tags)
    return "{" + ", ".join(components) + "}"


def stringify_shape(shape: ShapeType) -> str:
    components = [str(elem) for elem in shape]
    if not components:
        components = [","]
    elif len(components) == 1:
        components[0] += ","
    return "(" + ", ".join(components) + ")"


def get_object_by_id(object_id):
    """Find an object by its ID."""
    for obj in gc.get_objects():
        if id(obj) == object_id:
            return obj
    return None


class ArrayToDotNodeInfoMapper:
    def __init__(self, count_duplicates: bool = False):
        self.count_duplicates = count_duplicates
        self.node_to_dot = {}
        self.functions = set()

    def get_cache_key(self, expr):
        return id(expr) if self.count_duplicates else expr
    
    def get_common_dot_info(self, expr: Array) -> _DotNodeInfo:
        title = type(expr).__name__
        fields = {"addr": hex(id(expr)),
                  "shape": stringify_shape(expr.shape),
                  "dtype": str(expr.dtype),
                  "tags": stringify_tags(expr.tags),
                  "non_equality_tags": expr.non_equality_tags,
                  }

        edges: Dict[str, Union[ArrayOrNames, FunctionDefinition]] = {}
        return _DotNodeInfo(title, fields, edges)

    def process_node(self, expr: Array) -> None:
        if isinstance(expr, DataWrapper):
            self.map_data_wrapper(expr)
        elif isinstance(expr, IndexLambda):
            self.map_index_lambda(expr)
        elif isinstance(expr, Stack):
            self.map_stack(expr)
        elif isinstance(expr, (IndexBase, IndexLambda)):
            self.map_basic_index(expr)
        elif isinstance(expr, Einsum):
            self.map_einsum(expr)
        elif isinstance(expr, DictOfNamedArrays):
            self.map_dict_of_named_arrays(expr)
        elif isinstance(expr, LoopyCall):
            self.map_loopy_call(expr)
        elif isinstance(expr, DistributedSendRefHolder):
            self.map_distributed_send_ref_holder(expr)
        elif isinstance(expr, Call):
            self.map_call(expr)
        elif isinstance(expr, NamedCallResult):
            self.map_named_call_result(expr)
        else:
            self.handle_unsupported_array(expr)

    def handle_unsupported_array(self, expr: Array) -> None:
        info = self.get_common_dot_info(expr)
        expr_key = self.get_cache_key(expr)
        for field in attrs.fields(type(expr)):
            if field.name in info.fields:
                continue
            attr = getattr(expr, field.name)
            if isinstance(attr, Array):
                self.process_node(attr)
                key = self.get_cache_key(attr)
                info.edges[field.name] = (key, attr)
            elif isinstance(attr, AbstractResultWithNamedArrays):
                self.process_node(attr)
                key = self.get_cache_key(attr)
                info.edges[field.name] = (key, attr)
            elif isinstance(attr, tuple):
                info.fields[field.name] = stringify_shape(attr)
            else:
                info.fields[field.name] = str(attr)
        self.node_to_dot[expr_key] = info

    def map_data_wrapper(self, expr: DataWrapper) -> None:
        info = self.get_common_dot_info(expr)
        if expr.name is not None:
            info.fields["name"] = expr.name

        import numpy as np
        with np.printoptions(threshold=4, precision=2):
            info.fields["data"] = str(expr.data)

        self.node_to_dot[self.get_cache_key(expr)] = info

    def map_index_lambda(self, expr: IndexLambda) -> None:
        info = self.get_common_dot_info(expr)
        info.fields["expr"] = str(expr.expr)

        for name, val in expr.bindings.items():
            self.process_node(val)
            key = self.get_cache_key(val)
            info.edges[name] = (key, val)

        self.node_to_dot[self.get_cache_key(expr)] = info

    def map_stack(self, expr: Stack) -> None:
        info = self.get_common_dot_info(expr)
        info.fields["axis"] = str(expr.axis)

        for i, array in enumerate(expr.arrays):
            self.process_node(array)
            key = self.get_cache_key(array)
            info.edges[str(i)] = (key, array)

        self.node_to_dot[self.get_cache_key(expr)] = info

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
                self.process_node(index)
                key = self.get_cache_key(index)
                indices_parts.append(label)
                info.edges[label] = (key, index)

            elif index is None:
                indices_parts.append("newaxis")

            else:
                indices_parts.append(str(index))

        info.fields["indices"] = ", ".join(indices_parts)

        self.process_node(expr.array)
        key = self.get_cache_key(expr.array)
        info.edges["array"] = (key, expr.array)

        self.node_to_dot[self.get_cache_key(expr)] = info

    map_contiguous_advanced_index = map_basic_index
    map_non_contiguous_advanced_index = map_basic_index

    def map_einsum(self, expr: Einsum) -> None:
        info = self.get_common_dot_info(expr)

        for iarg, (access_descr, val) in enumerate(zip(expr.access_descriptors,
                                                       expr.args)):
            self.process_node(val)
            key = self.get_cache_key(val)
            info.edges[f"{iarg}: {access_descr}"] = (key, val)

        self.node_to_dot[self.get_cache_key(expr)] = info

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> None:
        edges: Dict[str, Union[ArrayOrNames, FunctionDefinition]] = {}
        for name, val in expr._data.items():
            self.process_node(val)
            key = self.get_cache_key(val)
            edges[name] = (key, val)

        self.node_to_dot[self.get_cache_key(expr)] = _DotNodeInfo(
                title=type(expr).__name__,
                fields={},
                edges=edges)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        edges: Dict[str, Union[ArrayOrNames, FunctionDefinition]] = {}
        for name, arg in expr.bindings.items():
            if isinstance(arg, Array):
                self.process_node(arg)
                key = self.get_cache_key(arg)
                edges[name] = (key, arg)

        self.node_to_dot[self.get_cache_key(expr)] = _DotNodeInfo(
                title=type(expr).__name__,
                fields={"addr": hex(id(expr)), "entrypoint": expr.entrypoint},
                edges=edges)

    def map_distributed_send_ref_holder(
            self, expr: DistributedSendRefHolder) -> None:

        info = self.get_common_dot_info(expr)

        self.process_node(expr.passthrough_data)
        key = self.get_cache_key(expr.passthrough_data)
        info.edges["passthrough"] = (key, expr.passthrough_data)

        self.process_node(expr.send.data)
        key = self.get_cache_key(expr.send.data)
        info.edges["sent"] = (key, expr.send.data)

        info.fields["dest_rank"] = str(expr.send.dest_rank)
        info.fields["comm_tag"] = str(expr.send.comm_tag)

        self.node_to_dot[self.get_cache_key(expr)] = info

    def map_call(self, expr: Call) -> None:
        self.functions.add(expr.function)

        for bnd in expr.bindings.values():
            self.process_node(bnd)

        self.node_to_dot[self.get_cache_key(expr)] = _DotNodeInfo(
            title=expr.__class__.__name__,
            edges={
                "": expr.function,
                **{name: (self.get_cache_key(bnd), bnd) for name, bnd in expr.bindings.items()}},
            fields={
                "addr": hex(id(expr)),
                "tags": stringify_tags(expr.tags),
            }
        )

    def map_named_call_result(self, expr: NamedCallResult) -> None:
        self.process_node(expr._container)
        key = self.get_cache_key(expr._container)
        self.node_to_dot[self.get_cache_key(expr)] = _DotNodeInfo(
                title=expr.__class__.__name__,
                edges={"": (key, expr._container)},
                fields={"addr": hex(id(expr)),
                        "name": expr.name},
        )


# }}}


def dot_escape(s: str) -> str:
    # "\" and HTML are significant in graphviz.
    return html.escape(s.replace("\\", "\\\\").replace(" ", "_"))


def dot_escape_leave_space(s: str) -> str:
    # "\" and HTML are significant in graphviz.
    return html.escape(s.replace("\\", "\\\\"))


def get_array_key(array, count_duplicates):
    """Return a consistent key for the array."""
    return id(array) if count_duplicates and not isinstance(array, int) else array


# {{{ emit helpers

def _stringify_created_at(non_equality_tags: FrozenSet[Tag]) -> str:
    from pytato.tags import CreatedAt
    for tag in non_equality_tags:
        if isinstance(tag, CreatedAt):
            return tag.traceback.short_str()

    return "<unknown>"


def _emit_array(emit: Callable[[str], None], title: str, fields: Dict[str, Any],
                dot_node_id: str, color: str = "white") -> None:
    td_attrib = 'border="0"'
    table_attrib = 'border="0" cellborder="1" cellspacing="0"'

    rows = [f"<tr><td colspan='2' {td_attrib}>{dot_escape(title)}</td></tr>"]

    non_equality_tags: FrozenSet[Any] = fields.pop("non_equality_tags", frozenset())

    tooltip = dot_escape_leave_space(_stringify_created_at(non_equality_tags))

    for name, field in fields.items():
        field_content = dot_escape(field).replace("\n", "<br/>")
        rows.append(
                f"<tr><td {td_attrib}>{dot_escape(name)}:</td><td {td_attrib}>"
                f"<FONT FACE='monospace'>{field_content}</FONT></td></tr>"
        )

    table = f"<table {table_attrib}>\n{''.join(rows)}</table>"
    emit(f"{dot_node_id} [label=<{table}> style=filled fillcolor={color} "
         f'tooltip="{tooltip}"]')


def preprocess_all_nodes(partition, array_to_id, id_gen, count_duplicates):
    mapper = ArrayToDotNodeInfoMapper(count_duplicates)
    for part in partition.parts.values():
        for out_name in part.output_names:
            node = partition.name_to_output[out_name]
            mapper.process_node(node)


def _emit_name_cluster(
        emit: DotEmitter, subgraph_path: Tuple[str, ...],
        names: Mapping[str, ArrayOrNames],
        array_to_id: Mapping[ArrayOrNames, str], id_gen: Callable[[str], str],
        label: str,
        count_duplicates: bool = False) -> None:
    edges = []

    cluster_subgraph_path = subgraph_path + (f"cluster_{dot_escape(label)}",)
    emit_cluster = partial(emit, cluster_subgraph_path)
    emit_cluster("node [shape=ellipse]")
    emit_cluster(f'label="{label}"')

    for name, array in names.items():
        name_id = id_gen(dot_escape(name))
        emit_cluster('%s [label="%s"]' % (name_id, dot_escape(name)))
        array_key = get_array_key(array, count_duplicates)
        array_id = array_to_id[array_key]
        # Edges must be outside the cluster.
        edges.append((name_id, array_id))

    for name_id, array_id in edges:
        emit(subgraph_path, "%s -> %s" % (array_id, name_id))


def _emit_function(
        emitter: DotEmitter, subgraph_path: Tuple[str, ...],
        id_gen: UniqueNameGenerator,
        node_to_dot: Mapping[ArrayOrNames, _DotNodeInfo],
        func_to_id: Mapping[FunctionDefinition, str],
        outputs: Mapping[str, Array],
        count_duplicates: bool = False) -> None:
    input_arrays: List[Array] = []
    internal_arrays: List[ArrayOrNames] = []
    array_to_id: Dict[ArrayOrNames, str] = {}

    emit = partial(emitter, subgraph_path)
    for array in node_to_dot:
        key = get_array_key(array, count_duplicates)
        array_to_id[key] = id_gen("array")
        if isinstance(array, InputArgumentBase):
            input_arrays.append(array)
        else:
            internal_arrays.append(array)

    # Emit inputs.
    input_subgraph_path = subgraph_path + ("cluster_inputs",)
    emit_input = partial(emitter, input_subgraph_path)
    emit_input('label="Arguments"')

    for array in input_arrays:
        key = get_array_key(array, count_duplicates)
        _emit_array(
                emit_input,
                node_to_dot[array].title,
                node_to_dot[array].fields,
                array_to_id[key])

    # Emit non-inputs.
    for array in internal_arrays:
        key = get_array_key(array, count_duplicates)
        _emit_array(emit,
                    node_to_dot[array].title,
                    node_to_dot[array].fields,
                    array_to_id[key])

    # Emit edges.
    for array, node in node_to_dot.items():
        key = get_array_key(array, count_duplicates)
        for label, edge_info in node.edges.items():
            if isinstance(edge_info, tuple):
                tail_key, tail_item = edge_info
            else:
                tail_item = edge_info
                tail_key = get_array_key(tail_item, count_duplicates)

            head = array_to_id[key]
            if isinstance(tail_item, (Array, AbstractResultWithNamedArrays)):
                tail = array_to_id[tail_key]
            elif isinstance(tail_item, FunctionDefinition):
                tail = func_to_id[tail_item]
            else:
                raise ValueError(f"unexpected type of tail on edge: {type(tail_item)}")

            emit('%s -> %s [label="%s"]' % (tail, head, dot_escape(label)))

    # Emit output/namespace name mappings.
    _emit_name_cluster(
            emitter, subgraph_path,
            outputs, array_to_id, id_gen,
            label="Returns", count_duplicates=count_duplicates)

# }}}


# {{{ information gathering

def _get_function_name(f: FunctionDefinition) -> Optional[str]:
    func_id_tags = f.tags_of_type(FunctionIdentifier)
    if func_id_tags:
        func_id_tag, = func_id_tags
        return str(func_id_tag.identifier)
    else:
        return None


def _gather_partition_node_information(
        id_gen: UniqueNameGenerator,
        partition: DistributedGraphPartition,
        count_duplicates: bool = False
        ) -> Tuple[
                Mapping[PartId, Mapping[FunctionDefinition, str]],
                Mapping[Tuple[PartId, Optional[FunctionDefinition]],
                        Mapping[ArrayOrNames, _DotNodeInfo]]
                ]:
    part_id_to_func_to_id: Dict[PartId, Dict[FunctionDefinition, str]] = {}
    part_id_func_to_node_info: Dict[Tuple[PartId, Optional[FunctionDefinition]],
                     Dict[ArrayOrNames, _DotNodeInfo]] = {}

    for part in partition.parts.values():
        mapper = ArrayToDotNodeInfoMapper(count_duplicates)
        for out_name in part.output_names:
            mapper.process_node(partition.name_to_output[out_name])

        part_id_func_to_node_info[part.pid, None] = mapper.node_to_dot
        part_id_to_func_to_id[part.pid] = {}

        # It is important that seen functions are emitted callee-first.
        # (Otherwise function 'entry' nodes will get declared in the wrong
        # cluster.) So use a data type that preserves order.
        seen_functions: List[FunctionDefinition] = []

        def gather_function_info(f: FunctionDefinition) -> None:
            key = (part.pid, f)  # noqa: B023
            if key in part_id_func_to_node_info:
                return

            mapper = ArrayToDotNodeInfoMapper(count_duplicates)
            for elem in f.returns.values():
                mapper.process_node(elem)

            part_id_func_to_node_info[key] = mapper.node_to_dot

            for subfunc in mapper.functions:
                gather_function_info(subfunc)

            if f not in seen_functions:  # noqa: B023
                seen_functions.append(f)  # noqa: B023

        for f in mapper.functions:
            gather_function_info(f)

        # Again, important to preserve function order. Here we're relying
        # on dicts to preserve order.
        for f in seen_functions:
            func_name = _get_function_name(f)
            if func_name is not None:
                fid = id_gen(dot_escape(func_name))
            else:
                fid = id_gen("func")

            part_id_to_func_to_id.setdefault(part.pid, {})[f] = fid

    return part_id_to_func_to_id, part_id_func_to_node_info


# }}}


def get_dot_graph(result: Union[Array, DictOfNamedArrays],
                  count_duplicates: bool = False) -> str:
    r"""Return a string in the `dot <https://graphviz.org>`_ language depicting the
    graph of the computation of *result*.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`).
    """

    outputs: DictOfNamedArrays = normalize_outputs(result)

    partition = DistributedGraphPartition(
        parts={
            None: DistributedGraphPart(
                pid=None,
                needed_pids=frozenset(),
                user_input_names=frozenset(
                            expr.name
                            for expr in InputGatherer()(outputs)
                            if isinstance(expr, Placeholder)
                            ),
                partition_input_names=frozenset(),
                output_names=frozenset(outputs.keys()),
                name_to_recv_node={},
                name_to_send_nodes={},
            )
        },
        name_to_output=outputs._data,
        overall_output_names=tuple(outputs),
    )

    return get_dot_graph_from_partition(partition, count_duplicates)


def get_dot_graph_from_partition(partition: DistributedGraphPartition, 
                                 count_duplicates: bool = False) -> str:
    """Return a string in the `dot <https://graphviz.org>`_ language depicting the
    graph of the partitioned computation of *partition*.

    :arg partition: Outputs of :func:`~pytato.find_distributed_partition`.
    """
    id_gen = UniqueNameGenerator()

    # {{{ gather up node info, per partition and per function

    # The "None" function is the body of the partition.

    part_id_to_func_to_id, part_id_func_to_node_info = \
        _gather_partition_node_information(id_gen, partition, count_duplicates)

    emitter = DotEmitter()
    emit_root = partial(emitter, ())

    emitted_placeholders = set()

    emit_root("node [shape=rectangle]")

    placeholder_to_id: Dict[Union[int, ArrayOrNames], str] = {}
    part_id_to_array_to_id: Dict[PartId, Dict[Union[int, ArrayOrNames], str]] = {}

    part_id_to_id = {pid: dot_escape(str(pid)) for pid in partition.parts}
    assert len(set(part_id_to_id.values())) == len(partition.parts)

    # {{{ generate names for all nodes in the root/None function

    for part in partition.parts.values():
        array_to_id = {}
        for array in part_id_func_to_node_info[part.pid, None].keys():
            if isinstance(array, int):  # if the key is an ID
                array = get_object_by_id(array)
            key = get_array_key(array, count_duplicates)
            if isinstance(array, Placeholder):
                if key in placeholder_to_id:
                    node_id = placeholder_to_id[key]
                else:
                    node_id = id_gen("array")
                    placeholder_to_id[key] = node_id
            else:
                node_id = id_gen("array")
            array_to_id[key] = node_id
        part_id_to_array_to_id[part.pid] = array_to_id

    # }}}

    # {{{ emit the graph

    for part in partition.parts.values():
        array_to_id = part_id_to_array_to_id[part.pid]

        is_trivial_partition = part.pid is None and len(partition.parts) == 1
        if is_trivial_partition:
            part_subgraph_path: Tuple[str, ...] = ()
        else:
            part_subgraph_path = (f"cluster_{part_id_to_id[part.pid]}",)

        emit_part = partial(emitter, part_subgraph_path)

        if not is_trivial_partition:
            emit_part("style=dashed")
            emit_part(f'label="{part.pid}"')

        # {{{ emit functions

        # It is important that seen functions are emitted callee-first.
        # Here we're relying on the part_id_to_func_to_id dict to preserve order.

        for func, fid in part_id_to_func_to_id[part.pid].items():
            func_subgraph_path = part_subgraph_path + (f"cluster_{fid}",)
            label = _get_function_name(func) or fid

            emitter(func_subgraph_path, f'label="{label}"')
            emitter(func_subgraph_path, f'{fid} [label="{label}",shape="ellipse"]')

            _emit_function(emitter, func_subgraph_path,
                           id_gen, part_id_func_to_node_info[part.pid, func],
                           part_id_to_func_to_id[part.pid],
                           func.returns,
                           count_duplicates=count_duplicates)

        # }}}

        # {{{ emit receives nodes

        part_dist_recv_var_name_to_node_id = {}
        for name, recv in part.name_to_recv_node.items():
            node_id = id_gen("recv")
            _emit_array(emit_part, "DistributedRecv", {
                "shape": stringify_shape(recv.shape),
                "dtype": str(recv.dtype),
                "src_rank": str(recv.src_rank),
                "comm_tag": str(recv.comm_tag),
            }, node_id)

            part_dist_recv_var_name_to_node_id[name] = node_id

        # }}}
        
        part_node_to_info = part_id_func_to_node_info[part.pid, None]
        input_arrays: List[Array] = []
        internal_arrays: List[ArrayOrNames] = []

        for array in part_node_to_info.keys():
            if isinstance(array, int):  # if the key is an ID
                array = get_object_by_id(array)
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
            key = array = get_array_key(array, count_duplicates)
            if not isinstance(array, Placeholder):
                _emit_array(emit_part,
                            part_node_to_info[array].title,
                            part_node_to_info[array].fields,
                            array_to_id[key], "deepskyblue")
            else:
                # Is a Placeholder
                if key in emitted_placeholders:
                    continue
                _emit_array(emit_root,
                            part_node_to_info[array].title,
                            part_node_to_info[array].fields,
                            array_to_id[key], "deepskyblue")

                # Emit cross-partition edges
                if array.name in part_dist_recv_var_name_to_node_id:
                    tgt = part_dist_recv_var_name_to_node_id[array.name]
                    emit_root(f"{tgt} -> {array_to_id[key]} [style=dotted]")
                    emitted_placeholders.add(key)
                elif array.name in part.user_input_names:
                    # no arrows for these
                    pass
                else:
                    # placeholder for a value from a different partition
                    computing_pid = None
                    for other_part in partition.parts.values():
                        if array.name in other_part.output_names:
                            computing_pid = other_part.pid
                            break
                    assert computing_pid is not None
                    tgt = part_id_to_array_to_id[computing_pid][
                        id(partition.name_to_output[array.name])
                        if count_duplicates
                        else partition.name_to_output[array.name]]
                    emit_root(f"{tgt} -> {array_to_id[key]} [style=dashed]")
                    emitted_placeholders.add(key)

        # }}}

        # Emit internal nodes
        
        for array in internal_arrays:
            key = array = get_array_key(array, count_duplicates)
            _emit_array(emit_part,
                        part_node_to_info[array].title,
                        part_node_to_info[array].fields,
                        array_to_id[key])

        # {{{ emit send nodes if distributed

        if isinstance(part, DistributedGraphPart):
            for name, sends in part.name_to_send_nodes.items():
                for send in sends:
                    node_id = id_gen("send")
                    _emit_array(emit_part, "DistributedSend", {
                        "dest_rank": str(send.dest_rank),
                        "comm_tag": str(send.comm_tag),
                    }, node_id)

                    # If an edge is emitted in a subgraph, it drags its
                    # nodes into the subgraph, too. Not what we want.
                    emit_root(
                        f"{array_to_id[id(send.data) if count_duplicates else send.data]} -> {node_id}"
                        f'[style=dotted, label="{dot_escape(name)}"]')

        # }}}

        # Emit intra-partition edges
        for array, node in part_node_to_info.items():
            key = get_array_key(array, count_duplicates)
            for label, edge_info in node.edges.items():
                if isinstance(edge_info, tuple):
                    tail_key, tail_item = edge_info
                else:
                    tail_item = edge_info
                    tail_key = get_array_key(tail_item, count_duplicates)

                head = array_to_id[key]

                if isinstance(tail_item, (Array, AbstractResultWithNamedArrays)):
                    tail = array_to_id[tail_key]
                elif isinstance(tail_item, FunctionDefinition):
                    tail = part_id_to_func_to_id[part.pid][tail_item]
                else:
                    raise ValueError(
                        f"unexpected type of tail on edge: {type(tail_item)}")

                emit_root('%s -> %s [label="%s"]' %
                          (tail, head, dot_escape(label)))

        _emit_name_cluster(
            emitter, part_subgraph_path,
            {name: partition.name_to_output[name]
             for name in part.output_names},
            array_to_id, id_gen, "Part outputs",
            count_duplicates)

    # }}}

    # Arrays may occur in multiple partitions, they get drawn separately anyhow
    # (unless they're Placeholders). Don't be tempted to use
    # combined_array_to_id everywhere.

    # {{{ draw overall outputs

    combined_array_to_id: Dict[Union[int, ArrayOrNames], str] = {}
    for part_id in partition.parts.keys():
        combined_array_to_id.update(part_id_to_array_to_id[part_id])

    _emit_name_cluster(
        emitter, (),
        {name: partition.name_to_output[name]
         for name in partition.overall_output_names},
        combined_array_to_id, id_gen, "Overall outputs",
        count_duplicates)

    # }}}

    return emitter.generate()


def show_dot_graph(result: Union[str, Array, DictOfNamedArrays,
                                 DistributedGraphPartition],
                   count_duplicates: bool = False,
                   **kwargs: Any) -> None:
    """Show a graph representing the computation of *result* in a browser.

    :arg result: Outputs of the computation (cf.
        :func:`pytato.generate_loopy`) or the output of :func:`get_dot_graph`,
        or the output of :func:`~pytato.find_distributed_partition`.
    :arg kwargs: Passed on to :func:`pytools.graphviz.show_dot` unmodified.
    """
    dot_code: str

    if isinstance(result, str):
        dot_code = result
    elif isinstance(result, DistributedGraphPartition):
        dot_code = get_dot_graph_from_partition(result, count_duplicates)
    else:
        dot_code = get_dot_graph(result, count_duplicates)

    from pytools.graphviz import show_dot
    show_dot(dot_code, **kwargs)

# vim:fdm=marker
