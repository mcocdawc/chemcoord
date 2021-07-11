# -*- coding: utf-8 -*-
from chemcoord.internal_coordinates._zmat_class_io import ZmatIO
# from chemcoord.internal_coordinates._zmat_class_gradient import ZmatGradient


# class Zmat(ZmatIO, ZmatGradient):
class Zmat(ZmatIO):
    """The main class for dealing with internal Coordinates.

    **Rotational direction:**

    Chemcoord uses the
    `IUPAC definition <https://goldbook.iupac.org/html/T/T06406.html>`_.
    Note that this does not include the automatic choosing of the
    canonical equivalence class representation.
    An angle of -30° could be represented by 270°.
    Use :meth:`~Zmat.iupacify` to choose also the
    IUPAC conform angle representation.

    **Mathematical Operations**:

    The general rule is that mathematical operations using the binary operators
    ``+ - * /`` and the unary operators ``+ - abs``
    are only applied to the ``['bond', 'angle', 'dihedral']`` columns.

    **Addition/Subtraction/Multiplication/Division**:
    The most common case is to add another Zmat instance.
    In this case it is tested, if the used references are the same.
    Afterwards the addition in the ``['bond', 'angle', 'dihedral']`` columns
    is performed.
    If you add a scalar to a Zmat it is added elementwise onto the
    ``['bond', 'angle', 'dihedral']`` columns.
    If you add a 3-dimensional vector, list, tuple... the first element of this
    vector is added elementwise to the ``'bond'`` column of the
    Zmat instance and so on.
    The third possibility is to add a matrix with
    ``shape=(len(Zmat), 3)`` which is again added elementwise.
    The same rules are true for subtraction, division and multiplication.

    **Indexing**:

    The indexing behaves like Indexing and Selecting data in
    `Pandas <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_.
    You can slice with :meth:`~chemcoord.Zmat.loc`,
    :meth:`~chemcoord.Zmat.iloc`, and ``Zmat[...]``.
    The only question is about the return type.
    If the information in the columns is enough to draw a molecule,
    an instance of the own class (e.g. :class:`~chemcoord.Zmat`)
    is returned.
    If the information in the columns is enough to draw a molecule,
    an instance of the own class (e.g. :class:`~chemcoord.Zmat`)
    is returned.
    If the information in the columns is not enough to draw a molecule,
    there are two cases to consider:

        * A :class:`~pandas.Series` instance is returned for one dimensional
          slices.
        * A :class:`~pandas.DataFrame` instance is returned in all other cases.

    This means that:

        ``molecule.loc[:, ['atom', 'b', 'bond', 'a', 'angle', 'd', 'dihedral']]``
        returns a :class:`~chemcoord.Zmat`.

        ``molecule.loc[:, ['atom', 'bond']]`` returns a
        :class:`pandas.DataFrame`.

        ``molecule.loc[:, 'atom']`` returns a
        :class:`pandas.Series`.

    **Comparison**:

    Comparison for equality with ``==`` is supported.
    It behaves exactly like the equality comparison of DataFrames in pandas.
    Amongst other things this means that the index has to be the same and
    the comparison of floating point numbers is exact and not numerical.
    """
