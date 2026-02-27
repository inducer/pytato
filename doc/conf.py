from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2020, Pytato Contributors"
author = "Pytato Contributors"

ver_dic = {}
with open("../pytato/version.py") as vfile:
    exec(compile(vfile.read(), "../pytato/version.py", "exec"), ver_dic)

version = ".".join(str(x) for x in ver_dic["VERSION"])
release = ver_dic["VERSION_TEXT"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "boxtree": ("https://documen.tician.de/boxtree/", None),
    "meshmode": ("https://documen.tician.de/meshmode/", None),
    "modepy": ("https://documen.tician.de/modepy/", None),
    "pyopencl": ("https://documen.tician.de/pyopencl/", None),
    "pytools": ("https://documen.tician.de/pytools/", None),
    "pymbolic": ("https://documen.tician.de/pymbolic/", None),
    "loopy": ("https://documen.tician.de/loopy/", None),
    "sumpy": ("https://documen.tician.de/sumpy/", None),
    "islpy": ("https://documen.tician.de/islpy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/latest", None),
    "constantdict": ("https://matthiasdiener.github.io/constantdict/", None),
    "orderedsets": ("https://matthiasdiener.github.io/orderedsets", None),
    "bidict": ("https://bidict.readthedocs.io/en/main/", None)
}

# Some modules need to import things just so that sphinx can resolve symbols in
# type annotations. Often, we do not want these imports (e.g. of PyOpenCL) when
# in normal use (because they would introduce unintended side effects or hard
# dependencies). This flag exists so that these imports only occur during doc
# build. Since sphinx appears to resolve type hints lexically (as it should),
# this needs to be cross-module (since, e.g. an inherited arraycontext
# docstring can be read by sphinx when building meshmode, a dependent package),
# this needs a setting of the same name across all packages involved, that's
# why this name is as global-sounding as it is.
import sys


sys._BUILDING_SPHINX_DOCS = True


nitpick_ignore_regex = [
    ["py:class", r"numpy.(u?)int[\d]+"],
    ["py:class", r"numpy.bool_"],
    ["py:class", r"typing_extensions(.+)"],
    ["py:class", r"P\.args"],
    ["py:class", r"P\.kwargs"],
    ["py:class", r"lp\.LoopKernel"],
    ["py:class", r"_dtype_any"],

    # It's :data:, not :class:, but we can't tell autodoc that.
    ["py:class", r"types\.EllipsisType"],
    # pytools
    # Got documented in Feb 2026, try removing?
    ["py:class", "ToTagSetConvertible"],
]


sphinxconfig_missing_reference_aliases = {
    # pymbolic
    "ArithmeticExpression": "obj:pymbolic.ArithmeticExpression",
    # pytools
    "lp.TemporaryVariable": "class:loopy.TemporaryVariable",
}


def setup(app):
    app.connect("missing-reference", process_autodoc_missing_reference)  # noqa: F821
