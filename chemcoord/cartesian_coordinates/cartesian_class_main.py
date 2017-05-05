from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
try:
    # import itertools.imap as map
    import itertools.izip as zip
except ImportError:
    pass
import numpy as np
import pandas as pd
import collections
from threading import Thread
import subprocess
import os
import tempfile
import warnings
from chemcoord._exceptions import PhysicalMeaningError
from chemcoord.algebra_utilities import utilities
from chemcoord.cartesian_coordinates.cartesian_class_io import Cartesian_io
from chemcoord.cartesian_coordinates.cartesian_class_to_zmat \
    import Cartesian_to_zmat
from chemcoord import export
from chemcoord.configuration.configuration import settings
import io
from io import open
import re


def pick(my_set):
    """Return one element from a set.

    **Do not** make any assumptions about the element to be returned.
    ``pick`` just returns a random element,
    could be the same, could be different.
    """
    assert type(my_set) is set, 'Pick can be applied only on sets.'
    x = my_set.pop()
    my_set.add(x)
    return x


class Cartesian(Cartesian_io, Cartesian_to_zmat):
    """The main class for dealing with cartesian Coordinates.

    **Mathematical Operations**:

    It supports binary operators in the logic of the scipy stack, but you need
    python3.x for using the matrix multiplication operator ``@``.

    The general rule is that mathematical operations using the binary operators
    ``+ - * / @`` and the unary operatos ``+ - abs``
    are only applied to the ``['x', 'y', 'z']`` columns.

    **Addition/Subtraction/Multiplication/Division**:
    If you add a scalar to a Cartesian it is added elementwise onto the
    ``['x', 'y', 'z']`` columns.
    If you add a 3-dimensional vector, list, tuple... the first element of this
    vector is added elementwise to the ``'x'`` column of the
    Cartesian instance and so on.
    The last possibility is to add a matrix with
    ``shape=(Cartesian.n_atoms, 3)`` which is again added elementwise.
    The same rules are true for subtraction, division and multiplication.

    **Matrixmultiplication**:
    Only leftsided multiplication with a matrix of ``shape=(n, 3)``,
    where ``n`` is a natural number, are supported.
    The usual usecase is for example
    ``np.diag([1, 1, -1]) @ cartesian_instance``
    to mirror on the x-y plane.

    **Slicing**:

    Slicing is supported and behaves like the ``.loc`` method of pandas.
    The returned type depends on the remaining columns after the slice.
    If the information of the remaining columns
    is sufficient to describe the geometry
    of a molecule, a Cartesian instance is returned.
    Otherwise a ``pandas.DataFrame`` or in the case of one remaining column
    a ``pandas.Series`` is returned.

    ``molecule[:, ['atom', 'x', 'y', 'z']]`` returns a ``Cartesian``.

    ``molecule[:, ['atom', 'x']]`` returns a ``pandas.DataFrame``.

    ``molecule[:, 'atom']`` returns a ``pandas.Series``.

    **Comparison**:

    Comparison for equality with ``==`` is supported.
    It behaves exactly like the equality comparison of DataFrames in pandas.
    Amongst other things this means that the index has to be the same and
    the comparison of floating point numbers is exact and not numerical.
    For this reason you rarely want to use ``==``.
    Usually the question is "are two given molecules chemically the same".
    For this comparison you have to use the function :func:`isclose`, which
    moves to the barycenter, aligns along the principal axes of inertia and
    compares numerically.
    """
