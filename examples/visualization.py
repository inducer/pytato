#!/usr/bin/env python
"""Demonstrates graph visualization with Graphviz."""

import logging
import numpy as np
import subprocess

import pytato as pt


logger = logging.getLogger(__name__)


GRAPH_DOT = "graph.dot"
GRAPH_SVG = "graph.svg"


def main():
    ns = pt.Namespace()

    pt.make_size_param(ns, "n")
    array = pt.make_placeholder(ns, "array", shape="n", dtype=np.float)
    stack = pt.stack([array, 2*array, array + 6])
    ns.assign("stack", stack)
    result = stack @ stack.T

    from pytato.visualization import get_dot_graph
    dot_code = get_dot_graph(result)

    with open(GRAPH_DOT, "w") as outf:
        outf.write(dot_code)
    logger.info("wrote '%s'", GRAPH_DOT)

    subprocess.run(["dot", "-Tsvg", GRAPH_DOT, "-o", GRAPH_SVG], check=True)
    logger.info("wrote '%s'", GRAPH_SVG)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
