from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2020, Pytato Contributors"
author = "Pytato Contributors"

ver_dic = {}
exec(compile(open("../pytato/version.py").read(), "../pytato/version.py",
    "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
release = ver_dic["VERSION_TEXT"]

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/boxtree/": None,
    "https://documen.tician.de/meshmode/": None,
    "https://documen.tician.de/modepy/": None,
    "https://documen.tician.de/pyopencl/": None,
    "https://documen.tician.de/pytools/": None,
    "https://documen.tician.de/pymbolic/": None,
    "https://documen.tician.de/loopy/": None,
    "https://documen.tician.de/sumpy/": None,
    "https://documen.tician.de/islpy/": None,
    "https://pyrsistent.readthedocs.io/en/latest/": None,
}

import sys
sys.PYTATO_BUILDING_SPHINX_DOCS = True

nitpick_ignore_regex = [
    ["py:class", r"numpy.(u?)int[\d]+"],
    ["py:class", r"pyrsistent.typing.(.+)"],

]
