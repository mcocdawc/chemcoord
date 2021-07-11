# -*- coding: utf-8 -*-
from chemcoord.cartesian_coordinates._cartesian_class_get_zmat import \
    CartesianGetZmat
from chemcoord.cartesian_coordinates._cartesian_class_io import CartesianIO
from chemcoord.cartesian_coordinates._cartesian_class_symmetry import \
    CartesianSymmetry


class Cartesian(CartesianIO, CartesianGetZmat, CartesianSymmetry):
    """The main class for dealing with cartesian Coordinates.

    **Mathematical Operations**:

    It supports binary operators in the logic of the scipy stack, but you need
    python3.x for using the matrix multiplication operator ``@``.

    The general rule is that mathematical operations using the binary operators
    ``+ - * / @`` and the unary operators ``+ - abs``
    are only applied to the ``['x', 'y', 'z']`` columns.

    **Addition/Subtraction/Multiplication/Division**:
    If you add a scalar to a Cartesian it is added elementwise onto the
    ``['x', 'y', 'z']`` columns.
    If you add a 3-dimensional vector, list, tuple... the first element of this
    vector is added elementwise to the ``'x'`` column of the
    Cartesian instance and so on.
    The last possibility is to add a matrix with
    ``shape=(len(Cartesian), 3)`` which is again added elementwise.
    The same rules are true for subtraction, division and multiplication.

    **Matrixmultiplication**:
    Only leftsided multiplication with a matrix of ``shape=(n, 3)``,
    where ``n`` is a natural number, is supported.
    The usual usecase is for example
    ``np.diag([1, 1, -1]) @ cartesian_instance``
    to mirror on the x-y plane.

    **Indexing**:

    The indexing behaves like Indexing and Selecting data in
    `Pandas <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_.
    You can slice with :meth:`~chemcoord.Cartesian.loc`,
    :meth:`~chemcoord.Cartesian.iloc`
    and ``Cartesian[...]``.
    The only question is about the return type.
    If the information in the columns is enough to draw a molecule,
    an instance of the own class (e.g. :class:`~chemcoord.Cartesian`)
    is returned.
    If the information in the columns is not enough to draw a molecule,
    there are two cases to consider:

        * A :class:`~pandas.Series` instance is returned for one dimensional
          slices.
        * A :class:`~pandas.DataFrame` instance is returned in all other cases.

    This means that:

        ``molecule.loc[:, ['atom', 'x', 'y', 'z']]`` returns a
        :class:`~chemcoord.Cartesian`.

        ``molecule.loc[:, ['atom', 'x']]`` returns a
        :class:`pandas.DataFrame`.

        ``molecule.loc[:, 'atom']`` returns a
        :class:`pandas.Series`.

    **Comparison**:

    Comparison for equality with ``==`` is supported.
    It behaves exactly like the equality comparison of DataFrames in pandas.
    Amongst other things this means that the index has to be the same and
    the comparison of floating point numbers is exact and not numerical.
    For this reason you rarely want to use ``==``.
    Usually the question is "are two given molecules chemically the same".
    For this comparison you have to use the function
    :func:`~xyz_functions.allclose`, which
    moves to the barycenter, aligns along the principal axes of inertia and
    compares numerically.
    """
