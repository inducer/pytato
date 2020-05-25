__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Xiaoyu Wei
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

__doc__ = """Interface classes and type specifications.
Each type is paired with a check_* function that, when used together, achieves
contracts-like functionality.
"""

import re
from pymbolic.primitives import Expression
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Tuple

# {{{ abstract classes


class NamespaceInterface():
    __metaclass__ = ABC

    @abstractmethod
    def assign(self, name, value):
        pass


class TagInterface():
    __metaclass__ = ABC


class ArrayInterface():
    """Abstract class for types implementing the Array interface.
    """
    __metaclass__ = ABC

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, val: NamespaceInterface):
        self._namespace = val

    @property
    @abstractmethod
    def ndim(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def copy(self, **kwargs):
        pass

    @abstractmethod
    def with_tag(self, tag_key, tag_val):
        pass

    @abstractmethod
    def without_tag(self, tag_key):
        pass

    @abstractmethod
    def with_name(self, name):
        pass

# }}} End abstract classes

# {{{ name type


NameType = str
C_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def check_name(name: NameType) -> bool:
    assert re.match(C_IDENTIFIER, name) is not None, \
            f"{name} is not a C identifier"
    return True

# }}} End name type

# {{{ tags type


TagsType = Dict[TagInterface, TagInterface]


def check_tags(tags: TagsType) -> bool:
    # assuming TagInterface implementation gurantees correctness
    return True

# }}} End tags type

# {{{ shape type


ShapeComponentType = Union[int, Expression, str]
ShapeType = Tuple[ShapeComponentType, ...]


def check_shape(shape: ShapeType,
                ns: Optional[NamespaceInterface] = None) -> bool:
    for s in shape:
        if isinstance(s, int):
            assert s > 0, f"size parameter must be positive (got {s})"
        elif isinstance(s, str):
            assert check_name(s)
        elif isinstance(s, Expression) and ns is not None:
            # TODO: check expression in namespace
            pass
    return True

# }}} End shape type
