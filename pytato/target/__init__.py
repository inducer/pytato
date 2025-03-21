from __future__ import annotations


__copyright__ = """Copyright (C) 2020 Matt Wala"""

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

__doc__ = """
.. currentmodule:: pytato.target

Code Generation Targets
-----------------------

.. autoclass:: Target
.. autoclass:: BoundProgram
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping


class Target:
    """
    An abstract code generation target.
    """


@dataclass(init=True, repr=False, eq=False)
class BoundProgram:
    """A container for the result of code generation along with data
    bindings for already-bound arguments.

    .. attribute:: target

       The code generation target.

    .. attribute:: program

        Description of the program as per :attr:`BoundProgram.target`.

    .. attribute:: bound_arguments

        A map from names to pre-bound kernel arguments.

    .. method:: __call__

        It is expected that every concrete subclass of this class
        have a ``__call__`` method to run the generated code,
        however interfaces may vary based on the specific
        subclass.
    """
    program: Any
    bound_arguments: Mapping[str, Any]
    target: Target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

# vim: foldmethod=marker
