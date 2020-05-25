Pytato: Get Descriptions of Array Computations via Lazy Evaluation
==================================================================

.. image:: https://gitlab.tiker.net/inducer/pytato/badges/master/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/pytato/commits/master
.. image:: https://github.com/inducer/pytato/workflows/CI/badge.svg?branch=master&event=push
    :alt: Github Build Status
    :target: https://github.com/inducer/pytato/actions?query=branch%3Amaster+workflow%3ACI+event%3Apush
.. image:: https://badge.fury.io/py/pytato.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/pytato/

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
for further details
