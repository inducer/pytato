from __future__ import annotations


"""
An eager evaluation wrapper around pytato that conforms to the
`Python Array API standard <https://data-apis.org/array-api/latest/>`_.

Each operation builds a pytato computation graph and then immediately
evaluates it via :mod:`numpy`. This exercises pytato's code-generation paths
while still producing concrete numpy results that the conformance-test suite
can inspect.

Usage::

    import pytato.array_api as xp
    x = xp.asarray([1, 2, 3], dtype=xp.float64)
    y = xp.sum(x)

.. note::

    This module is intended primarily for compliance testing. Not all Array
    API features are fully supported; consult ``array-api-tests.xfails.txt``
    for the current list of known failures.
"""

__copyright__ = """
Copyright (C) 2025 Andreas Kloeckner
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

# {{{ imports

from functools import reduce
from typing import Any, Literal, NamedTuple

import numpy as np

import pytato as pt
from pytato.target.python import BoundPythonProgram, NumpyPythonTarget
from pytato.target.python.numpy_like import generate_numpy_like


# }}}

# {{{ Array API version

__array_api_version__ = "2023.12"

# }}}

# {{{ evaluation helpers

_NUMPY_TARGET = NumpyPythonTarget()


def _eval_pt(expr: pt.Array) -> np.ndarray:
    """Evaluate a pytato expression to a numpy array using the numpy target."""
    bound: BoundPythonProgram = generate_numpy_like(  # type: ignore[assignment]
        expr, _NUMPY_TARGET,
        function_name="_pt_kernel",
        show_code=False,
        entrypoint_decorators=(),
        extra_preambles=())
    result = bound()
    return np.asarray(result)

# }}}


# {{{ dtype constants

bool = np.dtype("bool")

int8 = np.dtype("int8")
int16 = np.dtype("int16")
int32 = np.dtype("int32")
int64 = np.dtype("int64")

uint8 = np.dtype("uint8")
uint16 = np.dtype("uint16")
uint32 = np.dtype("uint32")
uint64 = np.dtype("uint64")

float32 = np.dtype("float32")
float64 = np.dtype("float64")

complex64 = np.dtype("complex64")
complex128 = np.dtype("complex128")

# }}}


# {{{ scalar constants

e = np.e
inf = np.inf
nan = np.nan
newaxis = np.newaxis
pi = np.pi

# }}}


# {{{ finfo / iinfo result types

class finfo_object(NamedTuple):  # noqa: N801
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: np.dtype[Any]


class iinfo_object(NamedTuple):  # noqa: N801
    bits: int
    max: int
    min: int
    dtype: np.dtype[Any]

# }}}


# {{{ Array class

class Array:
    """
    An Array API-compliant array backed by eager pytato evaluation.

    Wraps a numpy :class:`numpy.ndarray`; operations build pytato graphs
    that are evaluated immediately via :mod:`numpy`.
    """

    def __init__(self, data: np.ndarray) -> None:
        if isinstance(data, Array):
            self._data = data._data
        elif isinstance(data, np.ndarray):
            self._data: np.ndarray = data
        else:
            # Accept numpy scalars and Python scalars
            self._data = np.asarray(data)

    # {{{ standard attributes

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def T(self) -> Array:
        return permute_dims(self, tuple(range(self.ndim - 1, -1, -1)))

    @property
    def mT(self) -> Array:
        if self.ndim < 2:
            raise ValueError("mT requires at least 2 dimensions")
        perm = list(range(self.ndim))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return permute_dims(self, tuple(perm))

    @property
    def device(self) -> str:
        return "cpu"

    def __dlpack__(self, *,
                   stream: Any = None,
                   max_version: tuple[int, int] | None = None,
                   dl_device: tuple[int, int] | None = None,
                   copy: builtins_bool | None = None) -> Any:
        return self._data.__dlpack__()

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._data.__dlpack_device__()

    # }}}

    # {{{ array namespace

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        import sys
        name = __spec__.parent + ".array_api" if __spec__ is not None \
            else "pytato.array_api"
        return sys.modules[name]

    # }}}

    # {{{ conversion

    def __bool__(self) -> builtins_bool:
        return builtins_bool(self._data)

    def __complex__(self) -> complex:
        return complex(self._data)

    def __float__(self) -> float:
        return float(self._data)

    def __index__(self) -> int:
        return int(self._data)

    def __int__(self) -> int:
        return int(self._data)

    def to_device(self, device: str, /, *, stream: Any = None) -> Array:
        if device != "cpu":
            raise ValueError(f"Only 'cpu' device is supported, got {device!r}")
        return self

    # }}}

    # {{{ repr

    def __repr__(self) -> str:
        return f"Array({self._data!r}, dtype={self.dtype})"

    # }}}

    # {{{ arithmetic operators

    def __abs__(self) -> Array:
        return abs(self)

    def __add__(self, other: Array | complex) -> Array:
        return add(self, other)

    def __radd__(self, other: Array | complex) -> Array:
        return add(other, self)

    def __sub__(self, other: Array | complex) -> Array:
        return subtract(self, other)

    def __rsub__(self, other: Array | complex) -> Array:
        return subtract(other, self)

    def __mul__(self, other: Array | complex) -> Array:
        return multiply(self, other)

    def __rmul__(self, other: Array | complex) -> Array:
        return multiply(other, self)

    def __truediv__(self, other: Array | complex) -> Array:
        return divide(self, other)

    def __rtruediv__(self, other: Array | complex) -> Array:
        return divide(other, self)

    def __floordiv__(self, other: Array | complex) -> Array:
        return floor_divide(self, other)

    def __rfloordiv__(self, other: Array | complex) -> Array:
        return floor_divide(other, self)

    def __mod__(self, other: Array | complex) -> Array:
        return remainder(self, other)

    def __rmod__(self, other: Array | complex) -> Array:
        return remainder(other, self)

    def __pow__(self, other: Array | complex) -> Array:
        return pow(self, other)

    def __rpow__(self, other: Array | complex) -> Array:
        return pow(other, self)

    def __neg__(self) -> Array:
        return negative(self)

    def __pos__(self) -> Array:
        return positive(self)

    def __matmul__(self, other: Array) -> Array:
        return matmul(self, other)

    def __rmatmul__(self, other: Array) -> Array:
        return matmul(other, self)

    # in-place arithmetic operators
    def __iadd__(self, other: Array | complex) -> Array:
        result = add(self, other)
        self._data = result._data
        return self

    def __isub__(self, other: Array | complex) -> Array:
        result = subtract(self, other)
        self._data = result._data
        return self

    def __imul__(self, other: Array | complex) -> Array:
        result = multiply(self, other)
        self._data = result._data
        return self

    def __itruediv__(self, other: Array | complex) -> Array:
        result = divide(self, other)
        self._data = result._data
        return self

    def __ifloordiv__(self, other: Array | complex) -> Array:
        result = floor_divide(self, other)
        self._data = result._data
        return self

    def __imod__(self, other: Array | complex) -> Array:
        result = remainder(self, other)
        self._data = result._data
        return self

    def __ipow__(self, other: Array | complex) -> Array:
        result = pow(self, other)
        self._data = result._data
        return self

    def __imatmul__(self, other: Array) -> Array:
        result = matmul(self, other)
        self._data = result._data
        return self

    def __iand__(self, other: Array | int) -> Array:
        result = bitwise_and(self, other)
        self._data = result._data
        return self

    def __ior__(self, other: Array | int) -> Array:
        result = bitwise_or(self, other)
        self._data = result._data
        return self

    def __ixor__(self, other: Array | int) -> Array:
        result = bitwise_xor(self, other)
        self._data = result._data
        return self

    def __ilshift__(self, other: Array | int) -> Array:
        result = bitwise_left_shift(self, other)
        self._data = result._data
        return self

    def __irshift__(self, other: Array | int) -> Array:
        result = bitwise_right_shift(self, other)
        self._data = result._data
        return self

    # }}}

    # {{{ bitwise operators

    def __invert__(self) -> Array:
        return bitwise_invert(self)

    def __and__(self, other: Array | int) -> Array:
        return bitwise_and(self, other)

    def __rand__(self, other: Array | int) -> Array:
        return bitwise_and(other, self)

    def __or__(self, other: Array | int) -> Array:
        return bitwise_or(self, other)

    def __ror__(self, other: Array | int) -> Array:
        return bitwise_or(other, self)

    def __xor__(self, other: Array | int) -> Array:
        return bitwise_xor(self, other)

    def __rxor__(self, other: Array | int) -> Array:
        return bitwise_xor(other, self)

    def __lshift__(self, other: Array | int) -> Array:
        return bitwise_left_shift(self, other)

    def __rlshift__(self, other: Array | int) -> Array:
        return bitwise_left_shift(other, self)

    def __rshift__(self, other: Array | int) -> Array:
        return bitwise_right_shift(self, other)

    def __rrshift__(self, other: Array | int) -> Array:
        return bitwise_right_shift(other, self)

    # }}}

    # {{{ comparison operators

    def __eq__(self, other: object) -> Array:  # type: ignore[override]
        return equal(self, other)  # type: ignore[arg-type]

    def __ne__(self, other: object) -> Array:  # type: ignore[override]
        return not_equal(self, other)  # type: ignore[arg-type]

    def __lt__(self, other: Array | float) -> Array:
        return less(self, other)

    def __le__(self, other: Array | float) -> Array:
        return less_equal(self, other)

    def __gt__(self, other: Array | float) -> Array:
        return greater(self, other)

    def __ge__(self, other: Array | float) -> Array:
        return greater_equal(self, other)

    # }}}

    # {{{ indexing

    def __getitem__(self, key: Any) -> Array:
        return Array(self._data[_unwrap_index(key)])

    def __setitem__(self, key: Any, value: Array | complex) -> None:
        if isinstance(value, Array):
            self._data[_unwrap_index(key)] = value._data
        else:
            self._data[_unwrap_index(key)] = value

    # }}}

    # {{{ hash

    def __hash__(self) -> int:
        return id(self)

    # }}}

# }}}


# {{{ helpers

import builtins as _builtins


builtins_bool = _builtins.bool
builtins_abs = _builtins.abs


def _unwrap_index(key: Any) -> Any:
    """Convert Array indices to numpy indices."""
    if isinstance(key, Array):
        return key._data
    if isinstance(key, tuple):
        return tuple(_unwrap_index(k) for k in key)
    return key


def _as_array(x: Any) -> Array:
    """Ensure *x* is an :class:`Array`."""
    if isinstance(x, Array):
        return x
    return Array(np.asarray(x))


def _pt_wrap(np_data: np.ndarray) -> pt.Array:
    """Wrap a numpy array in a pytato DataWrapper."""
    return pt.make_data_wrapper(np_data)


def _normalize_axes(
        ndim: int,
        axis: int | tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    """Normalize *axis* to non-negative values, returning None for full reduction."""
    if axis is None:
        return None
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(a % ndim for a in axis)


def _eval_binop(a: np.ndarray, b: np.ndarray,
                pt_op: Any) -> np.ndarray:
    """Build a pytato expression for a binary op and evaluate it."""
    pt_a = _pt_wrap(a)
    pt_b = _pt_wrap(b)
    return _eval_pt(pt_op(pt_a, pt_b))


def _eval_unop(a: np.ndarray, pt_op: Any) -> np.ndarray:
    """Build a pytato expression for a unary op and evaluate it."""
    pt_a = _pt_wrap(a)
    return _eval_pt(pt_op(pt_a))

# }}}


# {{{ array_namespace info

class __array_namespace_info__:  # noqa: N801
    """Return information about the array namespace (Array API)."""

    def capabilities(self) -> dict[str, Any]:
        return {
            "boolean indexing": True,
            "data-dependent shapes": False,
        }

    def default_device(self) -> str:
        return "cpu"

    def default_dtypes(
            self,
            *,
            device: str | None = None,
    ) -> dict[str, np.dtype[Any]]:
        return {
            "real floating": float64,
            "complex floating": complex128,
            "integral": int64,
            "indexing": int64,
        }

    def devices(self) -> list[str]:
        return ["cpu"]

    def dtypes(
            self,
            *,
            device: str | None = None,
            kind: str | tuple[str, ...] | None = None,
    ) -> dict[str, np.dtype[Any]]:
        all_dtypes: dict[str, np.dtype[Any]] = {
            "bool": bool,
            "int8": int8, "int16": int16, "int32": int32, "int64": int64,
            "uint8": uint8, "uint16": uint16, "uint32": uint32, "uint64": uint64,
            "float32": float32, "float64": float64,
            "complex64": complex64, "complex128": complex128,
        }
        if kind is None:
            return all_dtypes
        if isinstance(kind, str):
            kind = (kind,)
        result = {}
        for k in kind:
            if k == "bool":
                result["bool"] = bool
            elif k == "signed integer":
                for n in ("int8", "int16", "int32", "int64"):
                    result[n] = all_dtypes[n]
            elif k == "unsigned integer":
                for n in ("uint8", "uint16", "uint32", "uint64"):
                    result[n] = all_dtypes[n]
            elif k == "integer":
                for n in ("int8", "int16", "int32", "int64",
                          "uint8", "uint16", "uint32", "uint64"):
                    result[n] = all_dtypes[n]
            elif k == "real floating":
                result["float32"] = float32
                result["float64"] = float64
            elif k == "complex floating":
                result["complex64"] = complex64
                result["complex128"] = complex128
            elif k == "numeric":
                for n in ("int8", "int16", "int32", "int64",
                          "uint8", "uint16", "uint32", "uint64",
                          "float32", "float64", "complex64", "complex128"):
                    result[n] = all_dtypes[n]
        return result

# }}}


# {{{ dtype utilities

def astype(x: Array, dtype: np.dtype[Any], /, *,
           copy: builtins_bool = True,
           device: str | None = None) -> Array:
    """Cast *x* to dtype *dtype*."""
    result = x._data.astype(dtype, copy=copy)
    return Array(result)


def can_cast(from_: np.dtype[Any] | Array,
             to: np.dtype[Any], /) -> builtins_bool:
    """Return whether *from_* can be cast to *to*."""
    if isinstance(from_, Array):
        from_ = from_.dtype
    return builtins_bool(np.can_cast(from_, to))


def finfo(type: np.dtype[Any] | Array, /) -> finfo_object:
    """Machine limits for floating-point types."""
    if isinstance(type, Array):
        type = type.dtype
    dtype = np.dtype(type)
    # For complex dtypes, finfo should be for the real component
    if np.issubdtype(dtype, np.complexfloating):
        real_dtype = np.result_type(dtype, np.float32)  # complex64->float32, etc.
        if dtype == np.dtype("complex64"):
            real_dtype = np.dtype("float32")
        elif dtype == np.dtype("complex128"):
            real_dtype = np.dtype("float64")
        else:
            real_dtype = np.dtype(np.zeros(1, dtype=dtype).real.dtype)
    else:
        real_dtype = dtype
    info = np.finfo(real_dtype)
    return finfo_object(
        bits=info.bits,
        eps=float(info.eps),
        max=float(info.max),
        min=float(info.min),
        smallest_normal=float(info.tiny),
        dtype=real_dtype,
    )


def iinfo(type: np.dtype[Any] | Array, /) -> iinfo_object:
    """Machine limits for integer types."""
    if isinstance(type, Array):
        type = type.dtype
    info = np.iinfo(type)
    return iinfo_object(
        bits=info.bits,
        max=int(info.max),
        min=int(info.min),
        dtype=np.dtype(type),
    )


def isdtype(dtype: np.dtype[Any],
            kind: str | np.dtype[Any] | tuple[str | np.dtype[Any], ...],
            ) -> builtins_bool:
    """Return whether *dtype* is of the given *kind*."""
    if isinstance(kind, tuple):
        return builtins_bool(_builtins.any(isdtype(dtype, k) for k in kind))
    if isinstance(kind, np.dtype):
        return builtins_bool(dtype == kind)
    # string kind
    if kind == "bool":
        return builtins_bool(np.issubdtype(dtype, np.bool_))
    if kind == "signed integer":
        return builtins_bool(np.issubdtype(dtype, np.signedinteger))
    if kind == "unsigned integer":
        return builtins_bool(np.issubdtype(dtype, np.unsignedinteger))
    if kind == "integer":
        return builtins_bool(np.issubdtype(dtype, np.integer))
    if kind == "real floating":
        return builtins_bool(np.issubdtype(dtype, np.floating))
    if kind == "complex floating":
        return builtins_bool(np.issubdtype(dtype, np.complexfloating))
    if kind == "numeric":
        return builtins_bool(np.issubdtype(dtype, np.number))
    raise ValueError(f"Unknown dtype kind: {kind!r}")


def result_type(*arrays_and_dtypes: Array | np.dtype[Any]) -> np.dtype[Any]:
    """Compute the result type for the given inputs."""
    dtypes = [x.dtype if isinstance(x, Array) else x
              for x in arrays_and_dtypes]
    return reduce(np.result_type, dtypes)

# }}}


# {{{ array creation functions

def arange(
        start: float,
        /,
        stop: float | None = None,
        step: float = 1,
        *,
        dtype: np.dtype[Any] | None = None,
        device: str | None = None,
) -> Array:
    """Return evenly-spaced values within a given interval."""
    if stop is None:
        stop = start
        start = 0
    return Array(np.arange(start, stop, step, dtype=dtype))


def asarray(
        obj: Any,
        /,
        *,
        dtype: np.dtype[Any] | None = None,
        device: str | None = None,
        copy: builtins_bool | None = None,
) -> Array:
    """Convert *obj* to an :class:`Array`."""
    if isinstance(obj, Array):
        data = obj._data
        if dtype is not None and data.dtype != np.dtype(dtype):
            data = data.astype(dtype)
        elif copy:
            data = data.copy()
        return Array(data)
    return Array(np.asarray(obj, dtype=dtype))


def empty(
        shape: int | tuple[int, ...],
        *,
        dtype: np.dtype[Any] = float64,
        device: str | None = None,
) -> Array:
    """Return an uninitialized array of given shape and dtype."""
    return Array(np.empty(shape, dtype=dtype))


def empty_like(
        x: Array,
        /,
        *,
        dtype: np.dtype[Any] | None = None,
        device: str | None = None,
) -> Array:
    """Return an uninitialized array with the same shape/dtype as *x*."""
    return Array(np.empty_like(x._data, dtype=dtype))


def eye(
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
        dtype: np.dtype[Any] = float64,
        device: str | None = None,
) -> Array:
    """Return a 2D identity(-like) array."""
    return Array(np.eye(n_rows, n_cols, k=k, dtype=dtype))


def from_dlpack(
        x: Any,
        /,
        *,
        device: str | None = None,
        copy: builtins_bool | None = None,
) -> Array:
    """Construct an array from a DLPack-compatible object."""
    return Array(np.from_dlpack(x))


def full(
        shape: int | tuple[int, ...],
        fill_value: Array | complex,
        *,
        dtype: np.dtype[Any] | None = None,
        device: str | None = None,
) -> Array:
    """Return an array filled with *fill_value*."""
    if isinstance(fill_value, Array):
        fill_value = fill_value._data  # type: ignore[assignment]
    return Array(np.full(shape, fill_value, dtype=dtype))


def full_like(
        x: Array,
        /,
        fill_value: Array | complex,
        *,
        dtype: np.dtype[Any] | None = None,
        device: str | None = None,
) -> Array:
    """Return an array filled with *fill_value*, same shape/dtype as *x*."""
    if isinstance(fill_value, Array):
        fill_value = fill_value._data  # type: ignore[assignment]
    return Array(np.full_like(x._data, fill_value, dtype=dtype))


def linspace(
        start: complex,
        stop: complex,
        /,
        num: int,
        *,
        dtype: np.dtype[Any] | None = None,
        device: str | None = None,
        endpoint: builtins_bool = True,
) -> Array:
    """Return evenly-spaced numbers over a specified interval."""
    return Array(np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint))


def meshgrid(
        *arrays: Array,
        indexing: Literal["xy", "ij"] = "xy",
) -> list[Array]:
    """Return coordinate matrices from coordinate vectors."""
    np_arrays = [a._data for a in arrays]
    return [Array(a) for a in np.meshgrid(*np_arrays, indexing=indexing)]


def ones(
        shape: int | tuple[int, ...],
        *,
        dtype: np.dtype[Any] = float64,
        device: str | None = None,
) -> Array:
    """Return an array filled with ones."""
    return Array(np.ones(shape, dtype=dtype))


def ones_like(
        x: Array,
        /,
        *,
        dtype: np.dtype[Any] | None = None,
        device: str | None = None,
) -> Array:
    """Return an array of ones with the same shape/dtype as *x*."""
    return Array(np.ones_like(x._data, dtype=dtype))


def tril(x: Array, /, *, k: int = 0) -> Array:
    """Return lower triangle of *x*."""
    return Array(np.tril(x._data, k=k))


def triu(x: Array, /, *, k: int = 0) -> Array:
    """Return upper triangle of *x*."""
    return Array(np.triu(x._data, k=k))


def zeros(
        shape: int | tuple[int, ...],
        *,
        dtype: np.dtype[Any] = float64,
        device: str | None = None,
) -> Array:
    """Return an array filled with zeros."""
    return Array(np.zeros(shape, dtype=dtype))


def zeros_like(
        x: Array,
        /,
        *,
        dtype: np.dtype[Any] | None = None,
        device: str | None = None,
) -> Array:
    """Return an array of zeros with the same shape/dtype as *x*."""
    return Array(np.zeros_like(x._data, dtype=dtype))

# }}}


# {{{ element-wise functions - routed through pytato where supported

def _pt_unary(x: Any, pt_op: Any) -> Array:
    x = _as_array(x)
    return Array(_eval_unop(x._data, pt_op))


def _pt_binary(x1: Any, x2: Any, pt_op: Any) -> Array:
    x1 = _as_array(x1)
    x2 = _as_array(x2)
    return Array(_eval_binop(x1._data, x2._data, pt_op))


def _np_unary(x: Any, np_op: Any) -> Array:
    x = _as_array(x)
    return Array(np_op(x._data))


def _np_binary(x1: Any, x2: Any, np_op: Any) -> Array:
    x1 = _as_array(x1)
    x2 = _as_array(x2)
    return Array(np_op(x1._data, x2._data))


# arithmetic (pytato)
def add(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a + b)


def subtract(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a - b)


def multiply(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a * b)


def divide(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a / b)


def floor_divide(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a // b)


def remainder(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a % b)


def pow(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a ** b)


def negative(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: -a)


def positive(x: Any, /) -> Array:
    x = _as_array(x)
    return Array(+x._data)


# comparisons (pytato)
def equal(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.equal(a, b))


def not_equal(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.not_equal(a, b))


def less(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.less(a, b))


def less_equal(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.less_equal(a, b))


def greater(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.greater(a, b))


def greater_equal(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.greater_equal(a, b))


# logical (pytato)
def logical_and(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.logical_and(a, b))


def logical_or(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.logical_or(a, b))


def logical_not(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.logical_not(a))


def logical_xor(x1: Any, x2: Any, /) -> Array:
    return _np_binary(x1, x2, np.logical_xor)


# bitwise (numpy - pytato supports these via IndexLambda but it is simpler
# to use numpy directly here, keeping the interface light)
def bitwise_and(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a & b)


def bitwise_or(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a | b)


def bitwise_xor(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a ^ b)


def bitwise_invert(x: Any, /) -> Array:
    # pytato does not support ~ on arrays, fall back to numpy
    return _np_unary(x, np.bitwise_not)


def bitwise_left_shift(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a << b)


def bitwise_right_shift(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: a >> b)


# mathematical (pytato)
def abs(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.abs(a))


def sqrt(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.sqrt(a))


def exp(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.exp(a))


def log(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.log(a))


def log2(x: Any, /) -> Array:
    return _np_unary(x, np.log2)


def log10(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.log10(a))


def log1p(x: Any, /) -> Array:
    return _np_unary(x, np.log1p)


def expm1(x: Any, /) -> Array:
    return _np_unary(x, np.expm1)


def sin(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.sin(a))


def cos(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.cos(a))


def tan(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.tan(a))


def asin(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.arcsin(a))


def acos(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.arccos(a))


def atan(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.arctan(a))


def atan2(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.arctan2(a, b))


def sinh(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.sinh(a))


def cosh(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.cosh(a))


def tanh(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.tanh(a))


def asinh(x: Any, /) -> Array:
    return _np_unary(x, np.arcsinh)


def acosh(x: Any, /) -> Array:
    return _np_unary(x, np.arccosh)


def atanh(x: Any, /) -> Array:
    return _np_unary(x, np.arctanh)


def conj(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.conj(a))


def imag(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.imag(a))


def real(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.real(a))


def isnan(x: Any, /) -> Array:
    return _pt_unary(x, lambda a: pt.isnan(a))


def isfinite(x: Any, /) -> Array:
    return _np_unary(x, np.isfinite)


def isinf(x: Any, /) -> Array:
    return _np_unary(x, np.isinf)


def maximum(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.maximum(a, b))


def minimum(x1: Any, x2: Any, /) -> Array:
    return _pt_binary(x1, x2, lambda a, b: pt.minimum(a, b))


def where(condition: Any, x1: Any, x2: Any, /) -> Array:
    cond = _as_array(condition)
    x1 = _as_array(x1)
    x2 = _as_array(x2)
    pt_cond = _pt_wrap(cond._data)
    pt_x1 = _pt_wrap(x1._data)
    pt_x2 = _pt_wrap(x2._data)
    return Array(_eval_pt(pt.where(pt_cond, pt_x1, pt_x2)))


# mathematical (numpy fallback for unimplemented ops)
def ceil(x: Any, /) -> Array:
    return _np_unary(x, np.ceil)


def floor(x: Any, /) -> Array:
    return _np_unary(x, np.floor)


def trunc(x: Any, /) -> Array:
    return _np_unary(x, np.trunc)


def round(x: Any, /) -> Array:
    return _np_unary(x, np.round)


def sign(x: Any, /) -> Array:
    return _np_unary(x, np.sign)


def signbit(x: Any, /) -> Array:
    return _np_unary(x, np.signbit)


def square(x: Any, /) -> Array:
    return _np_unary(x, np.square)


def copysign(x1: Any, x2: Any, /) -> Array:
    return _np_binary(x1, x2, np.copysign)


def hypot(x1: Any, x2: Any, /) -> Array:
    return _np_binary(x1, x2, np.hypot)


def logaddexp(x1: Any, x2: Any, /) -> Array:
    return _np_binary(x1, x2, np.logaddexp)


def nextafter(x1: Any, x2: Any, /) -> Array:
    return _np_binary(x1, x2, np.nextafter)


def clip(
        x: Any,
        /,
        min: Array | float | None = None,
        max: Array | float | None = None,
) -> Array:
    """Clip (limit) the values in an array."""
    x = _as_array(x)
    mn = min._data if isinstance(min, Array) else min
    mx = max._data if isinstance(max, Array) else max
    return Array(np.clip(x._data, mn, mx))

# }}}


# {{{ reduction functions

def all(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Test whether all elements evaluate to True."""
    x = _as_array(x)
    if keepdims:
        return Array(np.all(x._data, axis=axis, keepdims=True))
    pt_x = _pt_wrap(x._data)
    axes = _normalize_axes(x.ndim, axis)
    return Array(_eval_pt(pt.all(pt_x, axis=axes)))


def any(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Test whether any element evaluates to True."""
    x = _as_array(x)
    if keepdims:
        return Array(np.any(x._data, axis=axis, keepdims=True))
    pt_x = _pt_wrap(x._data)
    axes = _normalize_axes(x.ndim, axis)
    return Array(_eval_pt(pt.any(pt_x, axis=axes)))


def sum(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype[Any] | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Return the sum of array elements."""
    x = _as_array(x)
    if keepdims:
        result = np.sum(x._data, axis=axis, dtype=dtype, keepdims=True)
        return Array(result)
    pt_x = _pt_wrap(x._data)
    axes = _normalize_axes(x.ndim, axis)
    result = _eval_pt(pt.sum(pt_x, axis=axes))
    if dtype is not None:
        result = result.astype(dtype)
    return Array(result)


def prod(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype[Any] | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Return the product of array elements."""
    x = _as_array(x)
    if keepdims:
        result = np.prod(x._data, axis=axis, dtype=dtype, keepdims=True)
        return Array(result)
    pt_x = _pt_wrap(x._data)
    axes = _normalize_axes(x.ndim, axis)
    result = _eval_pt(pt.prod(pt_x, axis=axes))
    if dtype is not None:
        result = result.astype(dtype)
    return Array(result)


def max(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Return the maximum of array elements."""
    x = _as_array(x)
    if keepdims:
        return Array(np.max(x._data, axis=axis, keepdims=True))
    pt_x = _pt_wrap(x._data)
    axes = _normalize_axes(x.ndim, axis)
    return Array(_eval_pt(pt.amax(pt_x, axis=axes)))


def min(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Return the minimum of array elements."""
    x = _as_array(x)
    if keepdims:
        return Array(np.min(x._data, axis=axis, keepdims=True))
    pt_x = _pt_wrap(x._data)
    axes = _normalize_axes(x.ndim, axis)
    return Array(_eval_pt(pt.amin(pt_x, axis=axes)))


def mean(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Return the arithmetic mean of array elements."""
    x = _as_array(x)
    return Array(np.mean(x._data, axis=axis, keepdims=keepdims))


def std(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: float = 0,
        keepdims: builtins_bool = False,
) -> Array:
    """Return the standard deviation of array elements."""
    x = _as_array(x)
    return Array(np.std(x._data, axis=axis, ddof=correction, keepdims=keepdims))


def var(
        x: Any,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: float = 0,
        keepdims: builtins_bool = False,
) -> Array:
    """Return the variance of array elements."""
    x = _as_array(x)
    return Array(np.var(x._data, axis=axis, ddof=correction, keepdims=keepdims))

# }}}


# {{{ manipulation functions

def broadcast_arrays(*arrays: Array) -> list[Array]:
    """Broadcast any number of arrays against one another."""
    np_arrays = [a._data for a in arrays]
    return [Array(a) for a in np.broadcast_arrays(*np_arrays)]


def broadcast_to(x: Any, /, shape: tuple[int, ...]) -> Array:
    """Broadcast *x* to *shape*."""
    x = _as_array(x)
    # Fall back to numpy: pytato's broadcast_to can fail for 0-d scalars
    return Array(np.broadcast_to(x._data, shape).copy())


def concat(
        arrays: tuple[Array, ...] | list[Array],
        /,
        *,
        axis: int | None = 0,
) -> Array:
    """Join arrays along an existing axis."""
    pt_arrays = [_pt_wrap(a._data) for a in arrays]
    if axis is None:
        # flatten then concatenate
        flat = [_pt_wrap(a._data.ravel()) for a in arrays]
        return Array(_eval_pt(pt.concatenate(flat, axis=0)))
    return Array(_eval_pt(pt.concatenate(pt_arrays, axis=axis)))


def expand_dims(x: Any, /, axis: int) -> Array:
    """Expand the shape of *x* by inserting a new axis."""
    x = _as_array(x)
    pt_x = _pt_wrap(x._data)
    return Array(_eval_pt(pt.expand_dims(pt_x, axis)))


def flip(x: Any, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
    """Reverse the order of elements in *x*."""
    x = _as_array(x)
    return Array(np.flip(x._data, axis=axis))


def moveaxis(
        x: Any,
        source: int | tuple[int, ...],
        destination: int | tuple[int, ...],
        /,
) -> Array:
    """Move axes to new positions."""
    x = _as_array(x)
    return Array(np.moveaxis(x._data, source, destination))


def permute_dims(x: Any, /, axes: tuple[int, ...]) -> Array:
    """Permute axes of *x*."""
    x = _as_array(x)
    pt_x = _pt_wrap(x._data)
    return Array(_eval_pt(pt.transpose(pt_x, axes=axes)))


def reshape(
        x: Any,
        /,
        shape: tuple[int, ...],
        *,
        copy: builtins_bool | None = None,
) -> Array:
    """Reshape *x* to *shape*."""
    x = _as_array(x)
    pt_x = _pt_wrap(x._data)
    return Array(_eval_pt(pt.reshape(pt_x, shape)))


def roll(
        x: Any,
        /,
        shift: int | tuple[int, ...],
        *,
        axis: int | tuple[int, ...] | None = None,
) -> Array:
    """Roll array elements along given axes."""
    x = _as_array(x)
    if isinstance(shift, int) and isinstance(axis, int):
        pt_x = _pt_wrap(x._data)
        return Array(_eval_pt(pt.roll(pt_x, shift, axis=axis)))
    return Array(np.roll(x._data, shift, axis=axis))


def squeeze(x: Any, /, axis: int | tuple[int, ...]) -> Array:
    """Remove length-one axes from *x*."""
    x = _as_array(x)
    # normalize negative axes
    if isinstance(axis, int):
        axes: tuple[int, ...] = (axis % x.ndim,)
    else:
        axes = tuple(a % x.ndim for a in axis)
    pt_x = _pt_wrap(x._data)
    return Array(_eval_pt(pt.squeeze(pt_x, axis=axes)))


def stack(
        arrays: tuple[Array, ...] | list[Array],
        /,
        *,
        axis: int = 0,
) -> Array:
    """Join arrays along a new axis."""
    pt_arrays = [_pt_wrap(a._data) for a in arrays]
    return Array(_eval_pt(pt.stack(pt_arrays, axis=axis)))


def unstack(
        x: Any,
        /,
        *,
        axis: int = 0,
) -> tuple[Array, ...]:
    """Split *x* into sub-arrays along *axis*."""
    x = _as_array(x)
    return tuple(Array(a) for a in np.moveaxis(x._data, axis, 0))

# }}}


# {{{ linear algebra

def matmul(x1: Any, x2: Any, /) -> Array:
    """Compute the matrix product."""
    x1 = _as_array(x1)
    x2 = _as_array(x2)
    pt_x1 = _pt_wrap(x1._data)
    pt_x2 = _pt_wrap(x2._data)
    return Array(_eval_pt(pt.matmul(pt_x1, pt_x2)))


def matrix_transpose(x: Any, /) -> Array:
    """Transpose the last two dimensions of *x*."""
    x = _as_array(x)
    return x.mT


def tensordot(
        x1: Any,
        x2: Any,
        /,
        *,
        axes: int | tuple[tuple[int, ...], tuple[int, ...]] = 2,
) -> Array:
    """Compute the tensor dot product."""
    x1 = _as_array(x1)
    x2 = _as_array(x2)
    return Array(np.tensordot(x1._data, x2._data, axes=axes))


def vecdot(x1: Any, x2: Any, /, *, axis: int = -1) -> Array:
    """Compute the (batched) vector dot product."""
    x1 = _as_array(x1)
    x2 = _as_array(x2)
    return Array(np.tensordot(
        x1._data, x2._data,
        axes=([axis % x1.ndim], [axis % x2.ndim])))

# }}}


# {{{ searching functions

def argmax(
        x: Any,
        /,
        *,
        axis: int | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Return indices of maximum values along an axis."""
    x = _as_array(x)
    return Array(np.argmax(x._data, axis=axis, keepdims=keepdims))


def argmin(
        x: Any,
        /,
        *,
        axis: int | None = None,
        keepdims: builtins_bool = False,
) -> Array:
    """Return indices of minimum values along an axis."""
    x = _as_array(x)
    return Array(np.argmin(x._data, axis=axis, keepdims=keepdims))


def nonzero(x: Any, /) -> tuple[Array, ...]:
    """Return indices of non-zero elements."""
    x = _as_array(x)
    return tuple(Array(a) for a in np.nonzero(x._data))


def searchsorted(
        x1: Any,
        x2: Any,
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: Array | None = None,
) -> Array:
    """Find indices where *x2* should be inserted into sorted *x1*."""
    x1 = _as_array(x1)
    x2 = _as_array(x2)
    s = sorter._data if sorter is not None else None
    return Array(np.searchsorted(x1._data, x2._data, side=side, sorter=s))

# }}}


# {{{ sorting functions

def argsort(
        x: Any,
        /,
        *,
        axis: int = -1,
        descending: builtins_bool = False,
        stable: builtins_bool = True,
) -> Array:
    """Return indices that sort *x*."""
    x = _as_array(x)
    kind = "stable" if stable else "quicksort"
    idx = np.argsort(x._data, axis=axis, kind=kind)
    if descending:
        idx = np.flip(idx, axis=axis)
    return Array(idx)


def sort(
        x: Any,
        /,
        *,
        axis: int = -1,
        descending: builtins_bool = False,
        stable: builtins_bool = True,
) -> Array:
    """Return a sorted copy of *x*."""
    x = _as_array(x)
    kind = "stable" if stable else "quicksort"
    result = np.sort(x._data, axis=axis, kind=kind)
    if descending:
        result = np.flip(result, axis=axis)
    return Array(result)

# }}}


# {{{ set functions

def unique_all(x: Any, /) -> Any:
    """Return unique elements, indices, inverse indices, and counts."""
    x = _as_array(x)
    values, indices, inverse_indices, counts = np.unique(
        x._data, return_index=True, return_inverse=True, return_counts=True)

    class UniqueAllResult(NamedTuple):
        values: Array
        indices: Array
        inverse_indices: Array
        counts: Array

    return UniqueAllResult(
        values=Array(values),
        indices=Array(indices),
        inverse_indices=Array(inverse_indices),
        counts=Array(counts),
    )


def unique_counts(x: Any, /) -> Any:
    """Return unique elements and their counts."""
    x = _as_array(x)
    values, counts = np.unique(x._data, return_counts=True)

    class UniqueCountsResult(NamedTuple):
        values: Array
        counts: Array

    return UniqueCountsResult(values=Array(values), counts=Array(counts))


def unique_inverse(x: Any, /) -> Any:
    """Return unique elements and inverse indices."""
    x = _as_array(x)
    values, inverse_indices = np.unique(x._data, return_inverse=True)

    class UniqueInverseResult(NamedTuple):
        values: Array
        inverse_indices: Array

    return UniqueInverseResult(
        values=Array(values), inverse_indices=Array(inverse_indices))


def unique_values(x: Any, /) -> Array:
    """Return unique elements of *x*."""
    x = _as_array(x)
    return Array(np.unique(x._data))

# }}}


# {{{ additional functions (cumulative, indexing, manipulation)

def cumulative_sum(
        x: Any,
        /,
        *,
        axis: int | None = None,
        dtype: np.dtype[Any] | None = None,
        include_initial: builtins_bool = False,
) -> Array:
    """Return the cumulative sum of elements."""
    x = _as_array(x)
    result = np.cumsum(x._data, axis=axis, dtype=dtype)
    if include_initial:
        if axis is None:
            zeros_shape = (1,)
        else:
            zeros_shape = list(result.shape)
            zeros_shape[axis] = 1
            zeros_shape = tuple(zeros_shape)
        initial = np.zeros(zeros_shape, dtype=result.dtype)
        result = np.concatenate([initial, result], axis=axis or 0)
    return Array(result)


def take(
        x: Any,
        indices: Any,
        /,
        *,
        axis: int | None = None,
) -> Array:
    """Take elements from *x* along *axis*."""
    x = _as_array(x)
    indices = _as_array(indices)
    return Array(np.take(x._data, indices._data, axis=axis))


def repeat(
        x: Any,
        repeats: Any,
        /,
        *,
        axis: int | None = None,
) -> Array:
    """Repeat elements of *x*."""
    x = _as_array(x)
    if isinstance(repeats, Array):
        repeats = repeats._data
    return Array(np.repeat(x._data, repeats, axis=axis))


def tile(x: Any, repetitions: tuple[int, ...], /) -> Array:
    """Tile *x* by *repetitions*."""
    x = _as_array(x)
    return Array(np.tile(x._data, repetitions))

# }}}


# {{{ __all__

__all__ = [
    "Array",
    "__array_api_version__",
    "__array_namespace_info__",
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "any",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asarray",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "bool",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "ceil",
    "clip",
    "complex64",
    "complex128",
    "concat",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "cumulative_sum",
    "divide",
    "e",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "expand_dims",
    "expm1",
    "eye",
    "finfo",
    "flip",
    "float32",
    "float64",
    "floor",
    "floor_divide",
    "from_dlpack",
    "full",
    "full_like",
    "greater",
    "greater_equal",
    "hypot",
    "iinfo",
    "imag",
    "inf",
    "int8",
    "int16",
    "int32",
    "int64",
    "isdtype",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "matrix_transpose",
    "max",
    "maximum",
    "mean",
    "meshgrid",
    "min",
    "minimum",
    "moveaxis",
    "multiply",
    "nan",
    "negative",
    "newaxis",
    "nextafter",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "permute_dims",
    "pi",
    "positive",
    "pow",
    "prod",
    "real",
    "remainder",
    "repeat",
    "reshape",
    "result_type",
    "roll",
    "round",
    "searchsorted",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sort",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "take",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "tril",
    "triu",
    "trunc",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unstack",
    "var",
    "vecdot",
    "where",
    "zeros",
    "zeros_like",
]

# }}}
