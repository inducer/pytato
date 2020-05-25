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
from typing import Optional, Union, Dict, Tuple, Any, List

# {{{ abstract classes


class NamespaceInterface(ABC):

    @property
    def symbol_table(self) -> Dict['NameType', 'ArrayInterface']:
        return self._namespace

    @symbol_table.setter
    def symbol_table(self, val: Dict['NameType', 'ArrayInterface']) -> None:
        self._namespace = val

    @abstractmethod
    def assign(self, name: 'NameType',
               value: 'ArrayInterface') -> None:
        pass


class ArrayInterface(ABC):
    """Abstract class for types implementing the Array interface.
    """

    @property
    def namespace(self) -> NamespaceInterface:
        return self._namespace

    @namespace.setter
    def namespace(self, val: NamespaceInterface) -> None:
        self._namespace = val

    @property
    @abstractmethod
    def ndim(self) -> Any:
        pass

    @property
    @abstractmethod
    def shape(self) -> Any:
        pass

    @abstractmethod
    def copy(self, **kwargs: Any) -> 'ArrayInterface':
        pass

    @abstractmethod
    def with_tag(self, tag_key: Any,
                 tag_val: Any) -> 'ArrayInterface':
        pass

    @abstractmethod
    def without_tag(self, tag_key: Any) -> 'ArrayInterface':
        pass

    @abstractmethod
    def with_name(self, name: 'NameType') -> 'ArrayInterface':
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

# {{{ tags type


class DottedName():
    """
    .. attribute:: name_parts

        A tuple of strings, each of which is a valid
        C identifier (non-Unicode Python identifier).

    The name (at least morally) exists in the
    name space defined by the Python module system.
    It need not necessarily identify an importable
    object.
    """

    def __init__(self, name_parts: List[str]):
        assert len(name_parts) > 0
        assert all(check_name(p) for p in name_parts)
        self.name_parts = name_parts


TagsType = Dict[DottedName, DottedName]

# }}} End tags type
