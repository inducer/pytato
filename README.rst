Pytato: Get Descriptions of Array Computations via Lazy Evaluation
==================================================================

.. image:: https://gitlab.tiker.net/inducer/pytato/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/pytato/commits/main
.. image:: https://github.com/inducer/pytato/workflows/CI/badge.svg?branch=main&event=push
    :alt: Github Build Status
    :target: https://github.com/inducer/pytato/actions?query=branch%3Amain+workflow%3ACI+event%3Apush
.. image:: https://badge.fury.io/py/pytato.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/pytato/

Imagine TensorFlow, but aimed at HPC. Produces a data flow graph, where the
edges carry arrays and the nodes are (give or take) static-control programs
that compute array outputs from inputs, possibly (but not necessarily)
expressed in `Loopy <https://github.com/inducer/loopy>`__. A core assumption is
that the graph represents a computation that's being repeated often enough that
it is worthwhile to do expensive processing on it (code generation, fusion,
OpenCL compilation, etc).

* `Documentation <https://documen.tician.de/pytato>`__ (read how things work)

Example::

    import pytato as pt
    import numpy as np

    ns = pt.Namespace()
    pt.SizeParameter(ns, "n")  # -> prescribes shape=(), dtype=np.intp
    a = pt.Placeholder(ns, "a", "n,n", dtype=np.float32)

    # Also: pt.roll
    # If we can: np.roll
    a2a = a@(2*a)

    aat = a@a.T

    # FIXME: those names are only local...?
    # maybe change name of DictOfNamedArrays
    result = pt.DictOfNamedArrays({"a2a": a2a, "aat": aat})

    prg = pt.generate_loopy(result)

Pytato is licensed to you under the MIT/X Consortium license. See
the `documentation <https://documen.tician.de/pytato/misc.html>`__
for further details.

Numpy compatibility
-------------------

Pytato is written to pose no particular restrictions on the version of numpy
used for execution. To use mypy-based type checking on Pytato itself or
packages using Pytato, numpy 1.20 or newer is required, due to the
typing-based changes to numpy in that release.
