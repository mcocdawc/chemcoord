"""Define some types that do not fit into one particular module

In particular it enables barebone typechecking for the shape of numpy arrays

Inspired by
https://stackoverflow.com/questions/75495212/type-hinting-numpy-arrays-and-batches

Note that most numpy functions return :python:`ndarray[Any, Any]`
i.e. the type is mostly useful to document intent to the developer.
"""

import os
from collections.abc import Sequence
from typing import Any, Dict, NewType, Tuple, TypeVar, Union

import numpy as np

# Reexpose some pandas types
from pandas._typing import Axes  # noqa: F401
from typing_extensions import TypeAlias

# We want the dtype to behave covariant, i.e. if a
#  Vector[float] is allowed, then the more specific
#  Vector[float64] should also be allowed.
# Also see here:
# https://stackoverflow.com/questions/61568462/what-does-typevara-b-covariant-true-mean
#: Type annotation of a generic covariant type.
T_dtype_co = TypeVar("T_dtype_co", bound=np.generic, covariant=True)

# Currently we can define :code:`Matrix` and higher order tensors
# only with shape :code`Tuple[int, ...]` because of
# https://github.com/numpy/numpy/issues/27957
# make the typechecks more strict over time, when shape checking finally comes to numpy.

#: Type annotation of a vector.
Vector = np.ndarray[Tuple[int], np.dtype[T_dtype_co]]
#: Type annotation of a matrix.
Matrix = np.ndarray[Tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor3D = np.ndarray[Tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor4D = np.ndarray[Tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor5D = np.ndarray[Tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor6D = np.ndarray[Tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor7D = np.ndarray[Tuple[int, ...], np.dtype[T_dtype_co]]
#: Type annotation of a tensor.
Tensor = np.ndarray[Tuple[int, ...], np.dtype[T_dtype_co]]

#: Type annotation for pathlike objects.
PathLike: TypeAlias = Union[str, os.PathLike]
#: Type annotation for dictionaries holding keyword arguments.
KwargDict: TypeAlias = Dict[str, Any]

ArithmeticOther = Union[
    float, np.floating, Sequence, Sequence[Sequence], Vector, Matrix
]


AtomIdx = NewType("AtomIdx", int)
# AtomIdx: TypeAlias = int

Real: TypeAlias = float | np.floating
Integral: TypeAlias = int | np.integer
