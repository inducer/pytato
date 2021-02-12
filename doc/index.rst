Welcome to Pytato's documentation!
==================================

Imagine TensorFlow, but aimed at HPC. Produces a data flow graph, where the
edges carry arrays and the nodes are (give or take) static-control programs
that compute array outputs from inputs, possibly (but not necessarily)
expressed in `Loopy <https://github.com/inducer/loopy>`__. A core assumption is
that the graph represents a computation that's being repeated often enough that
it is worthwhile to do expensive processing on it (code generation, fusion,
OpenCL compilation, etc).

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    reference
    design
    misc
    🚀 Github <https://github.com/inducer/pytato>
    💾 Download Releases <https://pypi.org/project/pytato>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
