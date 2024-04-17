# -*- coding: utf-8 -*-
import copy
import warnings
from functools import partial

import chemcoord.constants as constants
import chemcoord.internal_coordinates._indexers as indexers
import chemcoord.internal_coordinates._zmat_transformation as transformation
import numpy as np
import pandas as pd
from chemcoord._generic_classes.generic_core import GenericCore
from chemcoord.exceptions import (ERR_CODE_OK, ERR_CODE_InvalidReference,
                                  InvalidReference, PhysicalMeaning)
from chemcoord.internal_coordinates._zmat_class_pandas_wrapper import \
    PandasWrapper
from chemcoord.utilities import _decorators
from chemcoord.utilities._temporary_deprecation_workarounds import replace_without_warn

append_indexer_docstring = _decorators.Appender(
    """In the case of obtaining elements, the indexing behaves like
Indexing and Selecting data in
`Pandas <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_.

For assigning elements it is necessary to make a explicit decision
between safe and unsafe assignments.
The differences are explained in the stub page of
:meth:`~Zmat.safe_loc`.""", join='\n\n')


class ZmatCore(PandasWrapper, GenericCore):
    """The main class for dealing with internal coordinates.
    """
    _required_cols = frozenset({'atom', 'b', 'bond', 'a', 'angle',
                                'd', 'dihedral'})
    dummy_manipulation_allowed = True
    test_operators = True
    pure_internal_mov = False

    def __init__(self, frame, metadata=None, _metadata=None):
        """How to initialize a Zmat instance.

        Args:
            init (pd.DataFrame): A Dataframe with at least the columns
                ``['atom', 'b', 'bond', 'a', 'angle',
                'd', 'dihedral']``.
                Where ``'atom'`` is a string for the elementsymbol.
            order_of_definition (list like): Specify in which order
                the Zmatrix is defined. If ``None`` it just uses
                ``self.index``.

        Returns:
            Zmat: A new zmat instance.
        """
        if not self._required_cols <= set(frame.columns):
            raise PhysicalMeaning('There are columns missing for a '
                                  'meaningful description of a molecule')
        self._frame = frame.copy()
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata.copy()

        if _metadata is None:
            self._metadata = {}
        else:
            self._metadata = copy.deepcopy(_metadata)

        def fill_missing_keys_with_defaults(_metadata):
            if 'last_valid_cartesian' not in _metadata:
                _metadata['last_valid_cartesian'] = self.get_cartesian()
            if 'has_dummies' not in _metadata:
                _metadata['has_dummies'] = {}

        fill_missing_keys_with_defaults(self._metadata)

    def copy(self):
        molecule = self.__class__(
            self._frame, metadata=self.metadata, _metadata=self._metadata)
        return molecule

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self._frame[key[0], key[1]]
        else:
            selected = self._frame[key]
        return selected

    def __getattr__(self, name: str):
        """
        After regular attribute access, try looking up the name
        This allows simpler access to columns for interactive use.
        """
        # Note: obj.x will always call obj.__getattribute__('x') prior to
        # calling obj.__getattr__('x').

        if name.startswith('__'):
            # See here, why we do this
            # https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
            raise AttributeError()
        if (name in self._frame.columns):
            return self[name]
        return object.__getattribute__(self, name)

    @property
    @append_indexer_docstring
    def loc(self):
        """Label based indexing for obtaining elements.
        """
        return indexers._Loc(self)

    @property
    @append_indexer_docstring
    def unsafe_loc(self):
        """Label based indexing for obtaining elements
and assigning values unsafely.
        """
        return indexers._Unsafe_Loc(self)

    @property
    def safe_loc(self):
        """Label based indexing for obtaining elements and assigning
        values safely.

        In the case of obtaining elements, the indexing behaves like
        Indexing and Selecting data in
        `Pandas <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_.
        """
        # TODO(Extend docstring)
        return indexers._Safe_Loc(self)

    @property
    @append_indexer_docstring
    def iloc(self):
        """Integer position based indexing for obtaining elements.
        """
        return indexers._ILoc(self)

    @property
    @append_indexer_docstring
    def unsafe_iloc(self):
        """Integer position based indexing for obtaining elements
and assigning values unsafely.
        """
        return indexers._Unsafe_ILoc(self)

    @property
    @append_indexer_docstring
    def safe_iloc(self):
        """Integer position based indexing for obtaining elements
and assigning values safely.
        """
        return indexers._Safe_ILoc(self)

    def _test_if_can_be_added(self, other):
        cols = ['atom', 'b', 'a', 'd']
        if not ((self.loc[:, cols] == other.loc[:, cols]).all(axis=None)
                and (self.index == other.index).all()):
            message = ("You can add only those zmatrices that have the same "
                       "index, use the same construction table, have the same "
                       "ordering... The only allowed difference is in the "
                       "columns ['bond', 'angle', 'dihedral']")
            raise PhysicalMeaning(message)

    def __add__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        if isinstance(other, ZmatCore):
            self._test_if_can_be_added(other)
            result = self.loc[:, coords] + other.loc[:, coords]
        else:
            result = self.loc[:, coords] + other
        new = self.copy()
        if self.test_operators:
            new.safe_loc[:, coords] = result
        else:
            new.unsafe_loc[:, coords] = result
        return new

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        if isinstance(other, ZmatCore):
            self._test_if_can_be_added(other)
            result = self.loc[:, coords] - other.loc[:, coords]
        else:
            result = self.loc[:, coords] - other
        new = self.copy()
        if self.test_operators:
            new.safe_loc[:, coords] = result
        else:
            new.unsafe_loc[:, coords] = result
        return new

    def __rsub__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        if isinstance(other, ZmatCore):
            self._test_if_can_be_added(other)
            result = other.loc[:, coords] - self.loc[:, coords]
        else:
            result = other - self.loc[:, coords]
        new = self.copy()
        if self.test_operators:
            new.safe_loc[:, coords] = result
        else:
            new.unsafe_loc[:, coords] = result
        return new

    def __mul__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        if isinstance(other, ZmatCore):
            self._test_if_can_be_added(other)
            result = self.loc[:, coords] * other.loc[:, coords]
        else:
            result = self.loc[:, coords] * other
        new = self.copy()
        if self.test_operators:
            new.safe_loc[:, coords] = result
        else:
            new.unsafe_loc[:, coords] = result
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        if isinstance(other, ZmatCore):
            self._test_if_can_be_added(other)
            result = self.loc[:, coords] / other.loc[:, coords]
        else:
            result = self.loc[:, coords] / other
        new = self.copy()
        if self.test_operators:
            new.safe_loc[:, coords] = result
        else:
            new.unsafe_loc[:, coords] = result
        return new

    def __rtruediv__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        if isinstance(other, ZmatCore):
            self._test_if_can_be_added(other)
            result = other.loc[:, coords] / self.loc[:, coords]
        else:
            result = other / self.loc[:, coords]
        new = self.copy()
        if self.test_operators:
            new.safe_loc[:, coords] = result
        else:
            new.unsafe_loc[:, coords] = result
        return new

    def __pow__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        if self.test_operators:
            new.safe_loc[:, coords] = self.loc[:, coords]**other
        else:
            new.unsafe_loc[:, coords] = self.loc[:, coords]**other
        return new

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        if self.test_operators:
            new.safe_loc[:, coords] = abs(self.loc[:, coords])
        else:
            new.unsafe_loc[:, coords] = abs(self.loc[:, coords])
        return new

    def __eq__(self, other):
        self._test_if_can_be_added(other)
        return self._frame == other._frame

    def __ne__(self, other):
        self._test_if_can_be_added(other)
        return self._frame != other._frame

    @staticmethod
    def _cast_correct_types(frame):
        new = frame.copy()
        zmat_values = ['bond', 'angle', 'dihedral']
        new.loc[:, zmat_values] = frame.loc[:, zmat_values].astype('f8')
        zmat_cols = ['b', 'a', 'd']
        new.loc[:, zmat_cols] = frame.loc[:, zmat_cols].astype('i8')
        return new

    def iupacify(self):
        r"""Give the IUPAC conform representation.

        Mathematically speaking the angles in a zmatrix are
        representations of an equivalence class.
        We will denote an equivalence relation with :math:`\sim`
        and use :math:`\alpha` for an angle and :math:`\delta` for a dihedral
        angle. Then the following equations hold true.

        .. math::

           (\alpha, \delta) &\sim (-\alpha, \delta + \pi) \\
           \alpha &\sim \alpha \mod 2\pi \\
           \delta &\sim \delta \mod 2\pi

        `IUPAC <https://goldbook.iupac.org/html/T/T06406.html>`_ defines
        a designated representation of these equivalence classes, by asserting:

        .. math::

           0 \leq &\alpha \leq \pi \\
           -\pi \leq &\delta \leq \pi

        Args:
            None

        Returns:
            Zmat: Zmatrix with accordingly changed angles and dihedrals.
        """
        def convert_d(d):
            r = d % 360
            return r - (r // 180) * 360

        new = self.copy()

        new.unsafe_loc[:, 'angle'] = new['angle'] % 360
        select = new['angle'] > 180
        new.unsafe_loc[select, 'angle'] = new.loc[select, 'angle'] - 180
        new.unsafe_loc[select, 'dihedral'] = new.loc[select, 'dihedral'] + 180

        new.unsafe_loc[:, 'dihedral'] = convert_d(new.loc[:, 'dihedral'])
        return new

    def minimize_dihedrals(self):
        r"""Give a representation of the dihedral with minimized absolute value.

        Mathematically speaking the angles in a zmatrix are
        representations of an equivalence class.
        We will denote an equivalence relation with :math:`\sim`
        and use :math:`\alpha` for an angle and :math:`\delta` for a dihedral
        angle. Then the following equations hold true.

        .. math::

           (\alpha, \delta) &\sim (-\alpha, \delta + \pi) \\
           \alpha &\sim \alpha \mod 2\pi \\
           \delta &\sim \delta \mod 2\pi

        This function asserts:

        .. math::

           -\pi \leq \delta \leq \pi

        The main application of this function is the construction of
        a transforming movement from ``zmat1`` to ``zmat2``.
        This is under the assumption that ``zmat1`` and ``zmat2`` are the same
        molecules (regarding their topology) and have the same
        construction table (:meth:`~Cartesian.get_construction_table`)::

          with cc.TestOperators(False):
              D = zm2 - zm1
              zmats1 = [zm1 + D * i / n for i in range(n)]
              zmats2 = [zm1 + D.minimize_dihedrals() * i / n for i in range(n)]

        The movement described by ``zmats1`` might be too large,
        because going from :math:`5^\circ` to :math:`355^\circ` is
        :math:`350^\circ` in this case and not :math:`-10^\circ` as
        in ``zmats2`` which is the desired :math:`\Delta` in most cases.

        Args:
            None

        Returns:
            Zmat: Zmatrix with accordingly changed angles and dihedrals.
        """
        new = self.copy()

        def convert_d(d):
            r = d % 360
            return r - (r // 180) * 360
        new.unsafe_loc[:, 'dihedral'] = convert_d(new.loc[:, 'dihedral'])
        return new

    # python 3.x is so much butter than 2.7
    # https://www.python.org/dev/peps/pep-3102/
    # def subs(self, *args, perform_checks=True):
    def subs(self, *args, **kwargs):
        """Substitute a symbolic expression in ``['bond', 'angle', 'dihedral']``

        This is a wrapper around the substitution mechanism of
        `sympy <http://docs.sympy.org/latest/tutorial/basic_operations.html>`_.
        Any symbolic expression in the columns
        ``['bond', 'angle', 'dihedral']`` of ``self`` will be substituted
        with value.

        .. note:: This function is not side-effect free.
            If all symbolic expressions are evaluated and are concrete numbers
            and ``perform_checks`` is True, a check for the transformation
            to cartesian coordinates is performed.
            If no :class:`~chemcoord.exceptions.InvalidReference`
            exceptions are raised, the resulting cartesian is written to
            ``self._metadata['last_valid_cartesian']``.

        Args:
            symb_expr (sympy expression):
            value :
            perform_checks (bool): If ``perform_checks is True``,
                it is asserted, that the resulting Zmatrix can be converted
                to cartesian coordinates.
                Dummy atoms will be inserted automatically if necessary.

        Returns:
            Zmat: Zmatrix with substituted symbolic expressions.
            If all resulting sympy expressions in a column are numbers,
            the column is recasted to 64bit float.
        """
        perform_checks = kwargs.pop('perform_checks', True)
        cols = ['bond', 'angle', 'dihedral']
        out = self.copy()

        def get_subs_f(*args):
            def subs_function(x):
                if hasattr(x, 'subs'):
                    x = x.subs(*args)
                    try:
                        x = float(x)
                    except TypeError:
                        pass
                return x
            return subs_function

        for col in cols:
            if out.loc[:, col].dtype is np.dtype('O'):
                out.unsafe_loc[:, col] = out.loc[:, col].map(get_subs_f(*args))
                try:
                    out._frame = out._frame.astype({col: 'f8'})
                except (SystemError, TypeError):
                    pass
        if perform_checks:
            try:
                new_cartesian = out.get_cartesian()
            except (AttributeError, TypeError):
                # Unevaluated symbolic expressions are remaining.
                pass
            except InvalidReference as e:
                if out.dummy_manipulation_allowed:
                    out._manipulate_dummies(e, inplace=True)
                else:
                    raise e
            else:
                out._metadata['last_valid_cartesian'] = new_cartesian
                self._metadata['last_valid_cartesian'] = new_cartesian
        return out

    def change_numbering(self, new_index=None):
        """Change numbering to a new index.

        Changes the numbering of index and all dependent numbering
            (bond_with...) to a new_index.
        The user has to make sure that the new_index consists of distinct
            elements.

        Args:
            new_index (list): If None the new_index is taken from 1 to the
                number of atoms.

        Returns:
            Zmat: Reindexed version of the zmatrix.
        """
        if (new_index is None):
            new_index = range(len(self))
        elif len(new_index) != len(self):
            raise ValueError('len(new_index) has to be the same as len(self)')

        c_table = self.loc[:, ['b', 'a', 'd']]
        # Strange bug in pandas where .replace is transitive for object columns
        # and non-transitive for all other types.
        # (Remember that string columns are just object columns)
        # Example:
        # A = {1: 2, 2: 3}
        # Transtitive [1].replace(A) gives [3]
        # Non-Transtitive [1].replace(A) gives [2]
        # https://github.com/pandas-dev/pandas/issues/5338
        # https://github.com/pandas-dev/pandas/issues/16051
        # https://github.com/pandas-dev/pandas/issues/5541
        # For this reason convert to int and replace then.

        c_table = replace_without_warn(c_table, constants.int_label)
        try:
            c_table = c_table.astype('i8')
        except ValueError:
            raise ValueError('Due to a bug in pandas it is necessary to have '
                             'integer columns')
        c_table = c_table.replace(self.index, new_index)
        c_table = c_table.replace(
            {v: k for k, v in constants.int_label.items()})

        out = self.copy()
        out.unsafe_loc[:, ['b', 'a', 'd']] = c_table
        out._frame.index = new_index
        return out

    def _insert_dummy_cart(self, exception, last_valid_cartesian=None):
        """Insert dummy atom into the already built cartesian of exception
        """
        def get_normal_vec(cartesian, reference_labels):
            b_pos, a_pos, d_pos = cartesian._get_positions(reference_labels)
            BA = a_pos - b_pos
            AD = d_pos - a_pos
            N1 = np.cross(BA, AD)
            n1 = N1 / np.linalg.norm(N1)
            return n1

        def insert_dummy(cartesian, reference_labels, n1):
            cartesian = cartesian.copy()
            b_pos, a_pos, d_pos = cartesian._get_positions(reference_labels)
            BA = a_pos - b_pos
            N2 = np.cross(n1, BA)
            n2 = N2 / np.linalg.norm(N2)
            i_dummy = max(self.index) + 1
            cartesian.loc[i_dummy, 'atom'] = 'X'
            cartesian.loc[i_dummy, ['x', 'y', 'z']] = a_pos + n2
            return cartesian, i_dummy

        if last_valid_cartesian is None:
            last_valid_cartesian = self._metadata['last_valid_cartesian']
        ref_labels = self.loc[exception.index, ['b', 'a', 'd']]
        n1 = get_normal_vec(last_valid_cartesian, ref_labels)
        return insert_dummy(exception.already_built_cartesian, ref_labels, n1)

    def _insert_dummy_zmat(self, exception, inplace=False):
        """Works INPLACE"""
        def insert_row(df, pos, key):
            if pos < len(df):
                middle = df.iloc[pos:(pos + 1)]
                middle.index = [key]
                start, end = df.iloc[:pos], df.iloc[pos:]
                return pd.concat([start, middle, end])
            elif pos == len(df):
                start = df.copy()
                start.loc[key] = start.iloc[-1]
                return start

        def raise_warning(i, dummy_d):
            give_message = ('For the dihedral reference of atom {i} the '
                            'dummy atom {dummy_d} was inserted').format
            warnings.warn(give_message(i=i, dummy_d=dummy_d), UserWarning)

        def insert_dummy(zmat, i, dummy_cart, dummy_d):
            """Works INPLACE on self._frame"""
            cols = ['b', 'a', 'd']
            actual_d = zmat.loc[i, 'd']
            zframe = insert_row(zmat, zmat.index.get_loc(i), dummy_d)
            zframe.loc[i, 'd'] = dummy_d
            zframe.loc[dummy_d, 'atom'] = 'X'
            zframe.loc[dummy_d, cols] = zmat.loc[actual_d, cols]
            zmat_values = dummy_cart._calculate_zmat_values(
                [dummy_d] + list(zmat.loc[actual_d, cols]))[0]
            zframe.loc[dummy_d, ['bond', 'angle', 'dihedral']] = zmat_values

            zmat._frame = zframe
            zmat._metadata['has_dummies'][i] = {'dummy_d': dummy_d,
                                                'actual_d': actual_d}
            raise_warning(i, dummy_d)

        zmat = self if inplace else self.copy()

        if exception.index in zmat._metadata['has_dummies']:
            zmat._remove_dummies(to_remove=[exception.index], inplace=True)
        else:
            insert_dummy(zmat, exception.index,
                         *zmat._insert_dummy_cart(exception))

        try:
            zmat._metadata['last_valid_cartesian'] = zmat.get_cartesian()
        except InvalidReference as e:
            zmat._insert_dummy_zmat(e, inplace=True)

        if not inplace:
            return zmat

    def _has_removable_dummies(self):
        has_dummies = self._metadata['has_dummies']
        to_be_tested = has_dummies.keys()
        c_table = self.loc[to_be_tested, ['b', 'a', 'd']]
        c_table['d'] = [has_dummies[i]['actual_d'] for i in to_be_tested]
        xyz = self.get_cartesian().loc[:, ['x', 'y', 'z']]
        BA = (xyz.loc[c_table['a']].values - xyz.loc[c_table['b']].values)
        AD = (xyz.loc[c_table['d']].values - xyz.loc[c_table['a']].values)

        remove = ~np.isclose(np.cross(BA, AD), np.zeros_like(BA)).all(axis=1)
        return [k for r, k in enumerate(to_be_tested) if remove[r]]

    def _remove_dummies(self, to_remove=None, inplace=False):
        """Works INPLACE"""
        zmat = self if inplace else self.copy()
        if to_remove is None:
            to_remove = zmat._has_removable_dummies()
        if not to_remove:
            if inplace:
                return None
            else:
                return zmat
        has_dummies = zmat._metadata['has_dummies']

        c_table = zmat.loc[to_remove, ['b', 'a', 'd']]
        c_table['d'] = [has_dummies[k]['actual_d'] for k in to_remove]
        zmat.unsafe_loc[to_remove, 'd'] = c_table['d'].astype('i8')

        zmat_values = zmat.get_cartesian()._calculate_zmat_values(c_table)
        zmat.unsafe_loc[to_remove, ['bond', 'angle', 'dihedral']] = zmat_values
        zmat._frame.drop([has_dummies[k]['dummy_d'] for k in to_remove],
                         inplace=True)
        warnings.warn('The dummy atoms {} were removed'.format(to_remove),
                      UserWarning)
        for k in to_remove:
            zmat._metadata['has_dummies'].pop(k)
        if not inplace:
            return zmat

    def _manipulate_dummies(self, exception, inplace=False):
        if inplace:
            self._insert_dummy_zmat(exception, inplace=True)
            self._remove_dummies(inplace=True)
        else:
            zmat = self.copy()
            zmat = zmat._insert_dummy_zmat(exception, inplace=False)
            return zmat._remove_dummies(inplace=False)

    def get_cartesian(self):
        """Return the molecule in cartesian coordinates.

        Raises an :class:`~exceptions.InvalidReference` exception,
        if the reference of the i-th atom is undefined.

        Args:
            None

        Returns:
            Cartesian: Reindexed version of the zmatrix.
        """
        def create_cartesian(positions, row):
            xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'],
                                     index=self.index[:row], dtype='f8')
            xyz_frame['atom'] = self.loc[xyz_frame.index, 'atom']
            xyz_frame.loc[:, ['x', 'y', 'z']] = positions[:row]
            from chemcoord.cartesian_coordinates.cartesian_class_main \
                import Cartesian
            cartesian = Cartesian(xyz_frame, metadata=self.metadata)
            return cartesian

        c_table = self.loc[:, ['b', 'a', 'd']]
        c_table = (replace_without_warn(c_table, constants.int_label)
                    .astype('i8')
                    .replace({k: v for v, k in enumerate(c_table.index)})
                    .values
                    .T)

        C = self.loc[:, ['bond', 'angle', 'dihedral']].values.T
        C[[1, 2], :] = np.radians(C[[1, 2], :])

        err, row, positions = transformation.get_X(C, c_table)
        positions = positions.T

        if err == ERR_CODE_InvalidReference:
            rename = dict(enumerate(self.index))
            i = rename[row]
            b, a, d = self.loc[i, ['b', 'a', 'd']]
            cartesian = create_cartesian(positions, row)
            raise InvalidReference(i=i, b=b, a=a, d=d,
                                   already_built_cartesian=cartesian)
        elif err == ERR_CODE_OK:
            return create_cartesian(positions, row + 1)

    def get_grad_cartesian(self, as_function=True, chain=True,
                           drop_auto_dummies=True, pure_internal=None):
        r"""Return the gradient for the transformation to a Cartesian.

        If ``as_function`` is True, a function is returned that can be directly
        applied onto instances of :class:`~Zmat`, which contain the
        applied distortions in Zmatrix space.
        In this case the user does not have to worry about indexing and
        correct application of the tensor product.
        Basically this is the function
        :func:`zmat_functions.apply_grad_cartesian_tensor`
        with partially replaced arguments.

        If ``as_function`` is False, a ``(3, n, n, 3)`` tensor is returned,
        which contains the values of the derivatives.

        Since a ``n * 3`` matrix is deriven after a ``n * 3``
        matrix, it is important to specify the used rules for indexing the
        resulting tensor.

        The rule is very simple: The indices of the numerator are used first
        then the indices of the denominator get swapped and appended:

        .. math::
            \left(
                \frac{\partial \mathbf{Y}}{\partial \mathbf{X}}
            \right)_{i, j, k, l}
            =
            \frac{\partial \mathbf{Y}_{i, j}}{\partial \mathbf{X}_{l, k}}

        Applying this rule to an example function:

        .. math::
            f \colon \mathbb{R}^3 \rightarrow \mathbb{R}

        Gives as derivative the known row-vector gradient:

        .. math::
                (\nabla f)_{1, i}
            =
                \frac{\partial f}{\partial x_i} \qquad i \in \{1, 2, 3\}

        .. note::
            The row wise alignment of the zmat files makes sense for these
            CSV like files.
            But it is mathematically advantageous and
            sometimes (depending on the memory layout) numerically better
            to use a column wise alignment of the coordinates.
            In this function the resulting tensor assumes a ``3 * n`` array
            for the coordinates.

        If

        .. math::

            \mathbf{C}_{i, j} &\qquad 1 \leq i \leq 3, \quad 1 \leq j \leq n \\
            \mathbf{X}_{i, j} &\qquad 1 \leq i \leq 3, \quad 1 \leq j \leq n

        denote the positions in Zmatrix and cartesian space,

        The complete tensor may be written as:

        .. math::

            \left(
                \frac{\partial \mathbf{X}}{\partial \mathbf{C}}
            \right)_{i, j, k, l}
            =
            \frac{\partial \mathbf{X}_{i, j}}{\partial \mathbf{C}_{l, k}}

        Args:
            construction_table (pandas.DataFrame):
            as_function (bool): Return a tensor or
                :func:`xyz_functions.apply_grad_zmat_tensor`
                with partially replaced arguments.
            chain (bool):
            drop_auto_dummies (bool): Drop automatically created
                dummies from the gradient.
                This means, that only changes in regularly placed atoms are
                considered for the gradient.
            pure_internal (bool): Clean the gradient using
                Eckart conditions to have only pure internal
                movements. (Compare 10.1063/1.2902290)
                Uses by default the information from
                `:class:zmat_functions.PureInternalMovement`.

        Returns:
            (func, :class:`numpy.ndarray`): Depending on ``as_function``
            return a tensor or
            :func:`~chemcoord.zmat_functions.apply_grad_cartesian_tensor`
            with partially replaced arguments.
        """
        zmat = self.change_numbering()

        c_table = (replace_without_warn(zmat.loc[:, ['b', 'a', 'd']],
                                        constants.int_label)
                    .astype('i8')
                    .values
                    .T)

        C = zmat.loc[:, ['bond', 'angle', 'dihedral']].values.T
        if C.dtype == np.dtype('i8'):
            C = C.astype('f8')
        C[[1, 2], :] = np.radians(C[[1, 2], :])


        grad_X = transformation.get_grad_X(C, c_table, chain=chain)

        if pure_internal or (pure_internal is None and self.pure_internal_mov):
            masses = zmat.add_data('mass').loc[:, 'mass'].values
            X = zmat.get_cartesian().loc[:, ['x', 'y', 'z']].values.T
            theta = zmat.get_cartesian().get_inertia()['inertia_tensor']
            grad_X = transformation.pure_internal_grad(X, grad_X, masses, theta)

        if drop_auto_dummies:
            def drop_dummies(grad_X, zmolecule):
                rename = dict(zip(zmolecule.index, range(len(zmolecule))))
                dummies = [rename[v['dummy_d']] for v in
                           self._metadata['has_dummies'].values()]
                excluded = np.full(grad_X.shape[1], True)
                excluded[dummies] = False
                coord_rows = np.full(3, True)
                selection = np.ix_(coord_rows, excluded, excluded, coord_rows)
                return grad_X[selection]

            grad_X = drop_dummies(grad_X, self)

        if as_function:
            from chemcoord.internal_coordinates.zmat_functions import (
                apply_grad_cartesian_tensor)
            return partial(apply_grad_cartesian_tensor, grad_X)
        else:
            return grad_X

    def to_xyz(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.get_cartesian`
        """
        message = 'Will be removed in the future. Please use get_cartesian.'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.get_cartesian(*args, **kwargs)
