""" Normalizes the array expression into Numpy-expression like nodes. """
from __future__ import annotations


import abc

from typing import Union, Tuple
from dataclasses import dataclass

from .scalar_expr import ScalarType


NumpyOpOrScalarT = Union["NumpyOp", ScalarType]
IndexExpr = Union[int, slice, "NumpyOp"]
