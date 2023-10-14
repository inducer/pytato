#!/usr/bin/env python
"""Demonstrates graph visualization with Graphviz."""

import logging
import numpy as np
import shutil
import subprocess

import pytato as pt


logger = logging.getLogger(__name__)


GRAPH_DOT = "graph.dot"
GRAPH_SVG = "graph.svg"


def main():
    n = pt.make_size_param("n")
    array = pt.make_placeholder(name="array", shape=n, dtype=np.float64)
    stack = pt.stack([array, 2*array, array + 6])
    result = stack @ stack.T

    dot_code = pt.get_dot_graph(result)

    with open(GRAPH_DOT, "w") as outf:
        outf.write(dot_code)
    logger.info("wrote '%s'", GRAPH_DOT)

    dot_path = shutil.which("dot")
    if dot_path is not None:
        subprocess.run([dot_path, "-Tsvg", GRAPH_DOT, "-o", GRAPH_SVG], check=True)
        logger.info("wrote '%s'", GRAPH_SVG)
    else:
        logger.info("'dot' executable not found; cannot convert to SVG")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
