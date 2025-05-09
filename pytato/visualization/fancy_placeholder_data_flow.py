"""
.. currentmodule:: pytato

.. autofunction:: show_fancy_placeholder_data_flow
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typing_extensions import Never

from pytools import UniqueNameGenerator

from pytato.array import (
    AdvancedIndexInContiguousAxes,
    AdvancedIndexInNoncontiguousAxes,
    Array,
    Concatenate,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexLambda,
    IndexRemappingBase,
    Placeholder,
    Stack,
)
from pytato.transform import CachedMapper


if TYPE_CHECKING:
    from collections.abc import Collection


# {{{ Graph node colors

PLACEHOLDER_COLOR = "lightgrey"
ELEMWISE_COLOR = "coral1"
OUTPUT_COLOR = "springgreen"
EINSUM_COLOR = "crimson"
STACK_CONCAT_COLOR = "deepskyblue"
INDIRECTION_COLOR = "darkblue"

# }}}

# {{{ Graph node shapes

PLACEHOLDER_SHAPE = "ellipse"
OUTPUT_SHAPE = "ellipse"
ELEMWISE_SHAPE = "diamond"
EINSUM_SHAPE = "box3d"
STACK_CONCAT_SHAPE = "folder"
INDIRECTION_SHAPE = "hexagon"

# }}}


@dataclass(frozen=True)
class _FancyDotWriterNode:
    """
    Return type for :class:`FancyDotWriter`.
    """
    pass


@dataclass(frozen=True)
class PlainOldDotNode(_FancyDotWriterNode):
    """
    Node that would appear in the graphviz graph.

    .. attribute:: node_id

        ID of the node in the graphviz-based graph.
    """
    node_id: str


@dataclass(frozen=True)
class NoShowNode(_FancyDotWriterNode):
    """
    Node that will not appear in the graphviz graph being built by a
    :class:`FancyDotWriter`.
    """


def _get_dot_node_from_predecessors(node_id: str,
                                    predecessors: Collection[_FancyDotWriterNode],
                                    ) -> tuple[_FancyDotWriterNode,
                                               frozenset[tuple[str, str]]]:

    new_edges: set[tuple[str, str]] = set()

    for pred in predecessors:
        if isinstance(pred, PlainOldDotNode):
            new_edges.add((pred.node_id, node_id))
        else:
            assert isinstance(pred, NoShowNode)

    if new_edges:
        return PlainOldDotNode(node_id), frozenset(new_edges)
    else:
        return NoShowNode(), frozenset()


class FancyDotWriter(CachedMapper[_FancyDotWriterNode, Never, []]):
    def __init__(self) -> None:
        super().__init__()
        self.vng = UniqueNameGenerator()

        self.node_decls: list[str] = []
        self.edges: set[tuple[str, str]] = set()

    def map_placeholder(self, expr: Placeholder) -> _FancyDotWriterNode:
        node_decl = (f"{expr.name} [color={PLACEHOLDER_COLOR}, "
                     f"shape={PLACEHOLDER_SHAPE}]")
        self.node_decls.append(node_decl)
        return PlainOldDotNode(expr.name)

    def map_data_wrapper(self, expr: DataWrapper) -> _FancyDotWriterNode:
        return NoShowNode()

    def map_index_lambda(self, expr: IndexLambda) -> _FancyDotWriterNode:
        from pytato.raising import (
            BinaryOp,
            BroadcastOp,
            C99CallOp,
            FullOp,
            LogicalNotOp,
            WhereOp,
            index_lambda_to_high_level_op,
        )

        hlo = index_lambda_to_high_level_op(expr)

        if isinstance(hlo, FullOp):
            return NoShowNode()
        elif isinstance(hlo,
                        BinaryOp | C99CallOp | WhereOp | BroadcastOp | LogicalNotOp):
            node_id = self.vng("_pt_elem")

            node_decl = (f"{node_id}"
                         f' [label="",color={ELEMWISE_COLOR},'
                         f" shape={ELEMWISE_SHAPE}]")
        else:
            raise NotImplementedError(type(hlo))

        ret_node, new_edges = _get_dot_node_from_predecessors(
            node_id,
            [self.rec(bnd) for bnd in expr.bindings.values()]
        )

        if new_edges:
            self.node_decls.append(node_decl)
            self.edges.update(new_edges)

        return ret_node

    def map_einsum(self, expr: Einsum) -> _FancyDotWriterNode:
        from pytato.utils import get_einsum_specification

        ensm_spec = get_einsum_specification(expr)
        node_id = self.vng("_pt_ensm")
        spec = ensm_spec.replace("->", "→")
        node_decl = (f'{node_id} [label="{spec}",'
                     f" color={EINSUM_COLOR},"
                     f" shape={EINSUM_SHAPE},"
                     " style=unfilled]")

        ret_node, new_edges = _get_dot_node_from_predecessors(
            node_id,
            [self.rec(arg) for arg in expr.args]
        )

        if new_edges:
            self.node_decls.append(node_decl)
            self.edges.update(new_edges)

        return ret_node

    def _map_stack_concat(self,
                          expr: Stack | Concatenate) -> _FancyDotWriterNode:
        node_id = self.vng("_pt_stack_concat")
        node_decl = (f'{node_id} [label="",'
                     f" color={STACK_CONCAT_COLOR},"
                     f" shape={STACK_CONCAT_SHAPE}]")

        ret_node, new_edges = _get_dot_node_from_predecessors(
            node_id,
            [self.rec(ary) for ary in expr.arrays]
        )

        if new_edges:
            self.node_decls.append(node_decl)
            self.edges.update(new_edges)

        return ret_node

    map_stack = _map_stack_concat
    map_concatenate = _map_stack_concat

    def _map_index_remapping(self,
                             expr: IndexRemappingBase) -> _FancyDotWriterNode:
        node_id = self.vng("_pt_idx_remap")

        node_decl = (f"{node_id}"
                     f' [label="",color={ELEMWISE_COLOR},'
                     f" shape={ELEMWISE_SHAPE}]")
        ret_node, new_edges = _get_dot_node_from_predecessors(
            node_id,
            [self.rec(expr.array)]
        )

        if new_edges:
            self.node_decls.append(node_decl)
            self.edges.update(new_edges)

        return ret_node

    map_reshape = _map_index_remapping
    map_roll = _map_index_remapping
    map_axis_permutation = _map_index_remapping
    map_basic_index = _map_index_remapping

    def _map_advanced_index(
                self,
                expr: AdvancedIndexInContiguousAxes | AdvancedIndexInNoncontiguousAxes
            ) -> _FancyDotWriterNode:
        node_id = self.vng("_pt_adv")
        node_decl = (f"{node_id}"
                     f' [label="",color={INDIRECTION_COLOR},'
                     f" shape={INDIRECTION_SHAPE}]")

        ret_node, new_edges = _get_dot_node_from_predecessors(
            node_id,
            [self.rec(expr.array),
             *[self.rec(idx) for idx in expr.indices if isinstance(idx, Array)]]
        )

        if new_edges:
            self.node_decls.append(node_decl)
            self.edges.update(new_edges)

        return ret_node

    map_contiguous_advanced_index = _map_advanced_index
    map_non_contiguous_advanced_index = _map_advanced_index

    def map_dict_of_named_arrays(self,
                                 expr: DictOfNamedArrays) -> _FancyDotWriterNode:

        for name, subexpr in expr._data.items():
            node_id = self.vng("_pt_out")
            node_decl = (f"{node_id}"
                         f' [label="{name}",color={OUTPUT_COLOR},'
                         f" shape={OUTPUT_SHAPE}]")
            rec_subexpr = self.rec(subexpr)
            if isinstance(rec_subexpr, PlainOldDotNode):
                self.node_decls.append(node_decl)
                self.edges.add((rec_subexpr.node_id, node_id))
            else:
                assert isinstance(rec_subexpr, NoShowNode)

        return NoShowNode()


def show_fancy_placeholder_data_flow(dag: Array | DictOfNamedArrays,
                                     **kwargs: Any) -> None:
    """
    Visualizes the data-flow from the placeholders into outputs.

    :arg dag: The expression to be plotted.
    :arg kwargs: Graphviz visualization options to be passed to
        :func:`pytools.graphviz.show_dot`.

    .. note::

        This is a heavily opinionated visualization of data-flow graph in
        *dag*. Displaying all the information about the node is not the
        priority. See :func:`pytato.show_dot_graph` that aims to be more
        verbose.
    """
    try:
        from mako.template import Template
    except ImportError as err:
        raise RuntimeError("'show_fancy_placeholder_data_flow' requires"
                           " mako. Install as `pip install mako`.") from err

    if isinstance(dag, Array):
        from pytato.array import make_dict_of_named_arrays
        dag = make_dict_of_named_arrays({"_pt_out": dag})

    assert isinstance(dag, DictOfNamedArrays)

    dot_writer = FancyDotWriter()
    dot_writer(dag)

    dot_src = """
digraph {
    // set default properties
    node[style=filled,fontsize=20]
    edge[arrowhead=vee]

    // NODES
    // ------------------------
    % for node_decl in node_decls:
    ${node_decl}
    % endfor

    // EDGES
    // ------------------------
    % for edge in edges:
    ${edge[0]} -> ${edge[1]}
    % endfor
}
    """

    dot_code = (Template(dot_src, strict_undefined=True)
                .render(node_decls=dot_writer.node_decls,
                        edges=dot_writer.edges))

    from pytools.graphviz import show_dot
    show_dot(dot_code, **kwargs)

# vim:fdm=marker
