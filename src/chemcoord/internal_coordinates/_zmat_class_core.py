# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord._exceptions import ERR_CODE_OK, \
    PhysicalMeaning, \
    InvalidReference, ERR_CODE_InvalidReference
import chemcoord.constants as constants
import chemcoord.internal_coordinates._indexers as indexers
from chemcoord.internal_coordinates._zmat_class_pandas_wrapper import \
    PandasWrapper
from chemcoord.utilities.algebra_utilities import \
    _jit_normalize, \
    _jit_rotation_matrix, \
    _jit_isclose, \
    _jit_cross
from numba import jit
import numpy as np
import pandas as pd
import sympy
import warnings


@jit(nopython=True)
def _jit_calculate_single_position(references, zmat_values, row):
    bond, angle, dihedral = zmat_values[row]
    vb, va, vd = references[0], references[1], references[2]
    zeros = np.zeros(3)
    err = ERR_CODE_OK

    BA = va - vb
    ba = _jit_normalize(BA)
    if _jit_isclose(angle, np.pi):
        d = bond * -ba
    elif _jit_isclose(angle, 0.):
        d = bond * ba
    else:
        AD = vd - va
        # N1 = _jit_cross(_jit_normalize(BA), _jit_normalize(AD))
        N1 = _jit_cross(BA, AD)
        if _jit_isclose(N1, zeros).all():
            err = ERR_CODE_InvalidReference
            d = zeros
        else:
            n1 = _jit_normalize(N1)
            d = bond * ba
            d = np.dot(_jit_rotation_matrix(n1, angle), d)
            d = np.dot(_jit_rotation_matrix(ba, dihedral), d)

    return (err, vb + d)


@jit(nopython=True)
def _jit_give_reference_absolute_position(j):
    # Because dicts are not supported in numba :(
    # Assumes 64bit system
    maxsize = 2**63 - 1
    if j == -maxsize - 1:
        return np.array([0., 0., 0.])
    elif j == -maxsize:
        return np.array([1., 0., 0.])
    elif j == -maxsize + 1:
        return np.array([0., 1., 0.])
    elif j == -maxsize + 2:
        return np.array([0., 0., 1.])
    else:
        raise ValueError


@jit(nopython=True)
def _jit_calculate_everything(positions, c_table, zmat_values, start_row=0):
    for row in range(start_row, c_table.shape[0]):
        ref_pos = np.empty((3, 3))
        # Assumes 64bit system
        keys_below_are_abs_refs = -2**63 + 100
        for k in range(3):
            j = c_table[row, k]
            if j < keys_below_are_abs_refs:
                ref_pos[k] = _jit_give_reference_absolute_position(j)
            else:
                ref_pos[k] = positions[j]
        err, pos = _jit_calculate_single_position(ref_pos, zmat_values, row)
        if err == ERR_CODE_OK:
            positions[row] = pos
        elif err == ERR_CODE_InvalidReference:
            return (ERR_CODE_InvalidReference, row)
    return (ERR_CODE_OK, row)


class ZmatCore(PandasWrapper):
    """The main class for dealing with internal coordinates.
    """
    _required_cols = frozenset({'atom', 'b', 'bond', 'a', 'angle',
                                'd', 'dihedral'})
    _metadata_keys = frozenset({'abs_refs', 'last_valid_cartesian'})

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
            self._metadata = _metadata.copy()
            if 'abs_refs' not in _metadata:
                self._metadata['abs_refs'] = constants.absolute_refs
            if 'last_valid_cartesian' not in _metadata:
                self._metadata['last_valid_cartesian'] = self.give_cartesian()
            if 'inserted_dummies' not in _metadata:
                self._metadata['inserted_dummies'] = {}

    def copy(self):
        molecule = self.__class__(self._frame)
        molecule.metadata = self.metadata.copy()
        molecule._metadata = self._metadata.copy()
        return molecule

    def _repr_html_(self):
        out = self.copy()
        representation = {key: out._metadata['abs_refs'][key][1]
                          for key in out._metadata['abs_refs']}

        def absolute_ref_formatter(x, representation=representation):
            try:
                return representation[x]
            except KeyError:
                return x

        def sympy_formatter(x):
            if (isinstance(x, sympy.Basic)):
                return '${}$'.format(sympy.latex(x))
            else:
                return x

        for col in ['b', 'a', 'd']:
            out.unsafe_loc[:, col] = out[col].apply(absolute_ref_formatter)
        for col in ['bond', 'angle', 'dihedral']:
            out.unsafe_loc[:, col] = out[col].apply(sympy_formatter)

        def insert_before_substring(insert_txt, substr, txt):
            """Under the assumption that substr only appears once.
            """
            return (insert_txt + substr).join(txt.split(substr))
        html_txt = out._frame._repr_html_()
        insert_txt = '<caption>{}</caption>\n'.format(self.__class__.__name__)
        return insert_before_substring(insert_txt, '<thead>', html_txt)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self._frame[key[0], key[1]]
        else:
            selected = self._frame[key]
        return selected

    @property
    def loc(self):
        return indexers._Loc(self)

    @property
    def unsafe_loc(self):
        return indexers._Unsafe_Loc(self)

    @property
    def safe_loc(self):
        return indexers._Safe_Loc(self)

    @property
    def iloc(self):
        return indexers._ILoc(self)

    @property
    def unsafe_iloc(self):
        return indexers._Unsafe_ILoc(self)

    @property
    def safe_iloc(self):
        return indexers._Safe_ILoc(self)

    def _test_if_can_be_added(self, other):
        cols = ['atom', 'b', 'a', 'd']
        if not isinstance(other, ZmatCore):
            raise PhysicalMeaning('You can only add zmatrices with each other')
        if not (np.alltrue(self.loc[:, cols] == other.loc[:, cols])
                and np.alltrue(self.index == other.index)):
            message = ("You can add only those zmatrices that have the same "
                       "index, use the same construction table, have the same "
                       "ordering... The only allowed difference is in the "
                       "columns ['bond', 'angle', 'dihedral']")
            raise PhysicalMeaning(message)

    def __add__(self, other):
        self._test_if_can_be_added(other)
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        new.safe_loc[:, coords] = self.loc[:, coords] + other.loc[:, coords]
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        self._test_if_can_be_added(other)
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        new.safe_loc[:, coords] = self.loc[:, coords] - other.loc[:, coords]
        return new

    def __rsub__(self, other):
        self._test_if_can_be_added(other)
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        new.safe_loc[:, coords] = other.loc[:, coords] - self.loc[:, coords]
        return new

    def __mul__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        new.safe_loc[:, coords] = self.loc[:, coords] * other
        return new

    def __rmul__(self, other):
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        new.safe_loc[:, coords] = self.loc[:, coords] * other
        return new

    def __abs__(self):
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        new.safe_loc[:, coords] = abs(new.loc[:, coords])
        return new

    def __neg__(self):
        return -1 * self.copy()

    def subs(self, variable, value):
        cols = ['bond', 'angle', 'dihedral']
        out = self.copy()

        def give_subs_function(variable, value):
            def subs_function(x):
                try:
                    new = x.subs(variable, value)
                except AttributeError:
                    new = x

                sympy_numbers = (sympy.numbers.Float, sympy.numbers.Integer)
                if isinstance(new, sympy_numbers):
                    return float(new)
                else:
                    return new
            return subs_function

        for col in cols:
            if out.loc[:, col].dtype is np.dtype('O'):
                series = out.loc[:, col]
                out.unsafe_loc[:, col] = series.map(
                    give_subs_function(variable, value))
                try:
                    out.unsafe_loc[:, col] = out.loc[:, col].astype('float')
                except TypeError:
                    pass
        return out

    def _to_Zmat(self):
        return self.copy()

    def change_numbering(self, new_index=None, inplace=False):
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
        out = self if inplace else self.copy()

        if (new_index is None):
            new_index = range(len(self))
        elif len(new_index) != len(self):
            raise ValueError('len(new_index) has to be the same as len(self)')

        cols = ['b', 'a', 'd']
        out.unsafe_loc[:, cols] = out.loc[:, cols].replace(
            out.index, new_index)
        out._frame.index = new_index
        if not inplace:
            return out

    def _insert_dummy_cart(self, exception):
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

        ref_labels = self.loc[exception.index, ['b', 'a', 'd']]
        n1 = get_normal_vec(self._metadata['last_valid_cartesian'], ref_labels)
        return insert_dummy(exception.already_built_cartesian,
                            ref_labels, n1)

    def _insert_dummy_zmat(self, exception):
        cols = ['b', 'a', 'd']
        dummy_cart, i_dummy = self._insert_dummy_cart(exception)
        d = self.loc[exception.index, 'd']
        ref_labels = np.array([i_dummy] + list(self.loc[d, cols]))

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

        zframe = insert_row(exception.zmat_after_assignment._frame,
                            self.index.get_loc(exception.index), i_dummy)
        zframe.loc[exception.index, 'd'] = i_dummy
        zframe.loc[i_dummy, 'atom'] = 'X'
        zframe.loc[i_dummy, cols] = self.loc[d, cols]
        zmat_values = dummy_cart._calculate_zmat_values(ref_labels[None, :])[0]
        zframe.loc[i_dummy, ['bond', 'angle', 'dihedral']] = zmat_values

        zmat = self.__class__(zframe, metadata=self.metadata,
                              _metadata=self._metadata)

        zmat._metadata['inserted_dummies'][exception.index] = {
            'dummy': i_dummy, 'd': d}
        try:
            zmat._metadata['last_valid_cartesian'] = zmat.give_cartesian()
        except InvalidReference as e:
            e.zmat_after_assignment = zmat
            # Recursion
            zmat = zmat._insert_dummy_zmat(e)
        return zmat

    def _has_removable_dummies(self):
        # coords = ['x', 'y', 'z']
        xyz = self.give_cartesian().loc[:, ['x', 'y', 'z']]
        inserted_dummies = self._metadata['inserted_dummies']
        indices_to_test = inserted_dummies.keys()
        c_table = self.loc[indices_to_test, ['b', 'a', 'd']]
        c_table['d'] = [inserted_dummies[key]['d'] for key in indices_to_test]
        BA = (xyz.loc[c_table['a']].values - xyz.loc[c_table['b']].values)
        AD = (xyz.loc[c_table['d']].values - xyz.loc[c_table['a']].values)

        may_be_removed = ~np.isclose(
            np.cross(BA, AD), np.zeros_like(BA)).all(axis=1)
        return [key for row, key in enumerate(indices_to_test)
                if may_be_removed[row]]

    def _remove_dummies(self):
        zmat = self.copy()
        xyz = zmat.give_cartesian()
        has_remove_dum = self._has_removable_dummies()
        inserted_dummies = self._metadata['inserted_dummies']
        zmat.unsafe_loc[has_remove_dum, 'd'] = [inserted_dummies[k]['d']
                                                for k in has_remove_dum]
        c_table = zmat.loc[has_remove_dum, ['b', 'a', 'd']]
        value_cols = ['bond', 'angle', 'dihedral']
        zmat_values = xyz._calculate_zmat_values(c_table)
        zmat.unsafe_loc[has_remove_dum, value_cols] = zmat_values

        dummies = [inserted_dummies[k]['dummy'] for k in has_remove_dum]

        return self.__class__(zmat[~zmat.index.isin(dummies)],
                              metadata=self.metadata, _metadata=self._metadata)

    def give_cartesian(self):
        old_index = self.index
        rename = dict(enumerate(old_index))
        self.change_numbering(inplace=True)
        c_table = self.loc[:, ['b', 'a', 'd']].values
        zmat_values = self.loc[:, ['bond', 'angle', 'dihedral']].values
        zmat_values[:, [1, 2]] = np.radians(zmat_values[:, [1, 2]])
        positions = np.empty((len(self), 3), dtype='float64')

        err, row = _jit_calculate_everything(positions, c_table, zmat_values)
        if err == ERR_CODE_InvalidReference:
            i = rename[row]
            self.change_numbering(old_index, inplace=True)
            b, a, d = self.loc[i, ['b', 'a', 'd']]

            self.change_numbering(old_index, inplace=True)
            xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'],
                                     index=self.index[:row], dtype=float)
            xyz_frame['atom'] = self.loc[xyz_frame.index, 'atom']
            xyz_frame.loc[:, ['x', 'y', 'z']] = positions[:row]

            from chemcoord.cartesian_coordinates.cartesian_class_main \
                import Cartesian
            cartesian = Cartesian(xyz_frame)
            raise InvalidReference(i=i, b=b, a=a, d=d,
                                   already_built_cartesian=cartesian)

        self.change_numbering(old_index, inplace=True)
        xyz_frame = pd.DataFrame(
            index=self.index, columns=['atom', 'x', 'y', 'z'], dtype=float)
        xyz_frame['atom'] = self['atom']
        xyz_frame.loc[:, ['x', 'y', 'z']] = positions

        from chemcoord.cartesian_coordinates.cartesian_class_main \
            import Cartesian
        return Cartesian(xyz_frame)

    def to_xyz(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.give_cartesian`
        """
        message = 'Will be removed in the future. Please use give_cartesian.'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.give_cartesian(*args, **kwargs)

    def add_data(self, new_cols=None):
        """Adds a column with the requested data.

        If you want to see for example the mass, the colormap used in
        jmol and the block of the element, just use::

            ['mass', 'jmol_color', 'block']

        The underlying ``pd.DataFrame`` can be accessed with
        ``constants.elements``.
        To see all available keys use ``constants.elements.info()``.

        The data comes from the module `mendeleev
        <http://mendeleev.readthedocs.org/en/latest/>`_ written
        by Lukasz Mentel.

        Please note that I added three columns to the mendeleev data::

            ['atomic_radius_cc', 'atomic_radius_gv', 'gv_color',
                'valency']

        The ``atomic_radius_cc`` is used by default by this module
        for determining bond lengths.
        The three others are taken from the MOLCAS grid viewer written
        by Valera Veryazov.

        Args:
            new_cols (str): You can pass also just one value.
                E.g. ``'mass'`` is equivalent to ``['mass']``. If
                ``new_cols`` is ``None`` all available data
                is returned.
            inplace (bool):

        Returns:
            Cartesian:
        """
        atoms = self['atom']
        data = constants.elements
        if pd.api.types.is_list_like(new_cols):
            new_cols = set(new_cols)
        elif new_cols is None:
            new_cols = set(data.columns)
        else:
            new_cols = [new_cols]
        new_frame = data.loc[atoms, set(new_cols) - set(self.columns)]
        new_frame.index = self.index
        return self.__class__(pd.concat([self._frame, new_frame], axis=1))

    def total_mass(self):
        """Returns the total mass in g/mol.

        Args:
            None

        Returns:
            float:
        """
        try:
            return self['mass'].sum()
        except KeyError:
            return self.add_data('mass')['mass'].sum()
