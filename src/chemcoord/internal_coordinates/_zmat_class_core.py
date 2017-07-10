# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord.exceptions import ERR_CODE_OK, \
    PhysicalMeaning, \
    InvalidReference, ERR_CODE_InvalidReference
import chemcoord.constants as constants
import chemcoord.internal_coordinates._indexers as indexers
from chemcoord.internal_coordinates._zmat_class_pandas_wrapper import \
    PandasWrapper
from chemcoord._generic_classes.generic_core import GenericCore
from chemcoord.utilities.algebra_utilities import \
    _jit_normalize, \
    _jit_rotation_matrix, \
    _jit_isclose, \
    _jit_cross
from collections import namedtuple
from numba import jit
import numba as nb
import numpy as np
import pandas as pd
import sympy
import warnings
import copy


@jit(nopython=True)
def _jit_calculate_single_position(references, zmat_values, row):
    bond, angle, dihedral = zmat_values[row]
    vb, va, vd = references[0], references[1], references[2]
    zeros = np.zeros(3, dtype=nb.types.f8)

    BA = va - vb
    if _jit_isclose(BA, zeros).all():
        return (ERR_CODE_InvalidReference, zeros)
    ba = _jit_normalize(BA)
    if _jit_isclose(angle, np.pi):
        d = bond * -ba
    elif _jit_isclose(angle, 0.):
        d = bond * ba
    else:
        AD = vd - va
        N1 = _jit_cross(BA, AD)
        if _jit_isclose(N1, zeros).all():
            return (ERR_CODE_InvalidReference, zeros)
        else:
            n1 = _jit_normalize(N1)
            d = bond * ba
            d = np.dot(_jit_rotation_matrix(n1, angle), d)
            d = np.dot(_jit_rotation_matrix(ba, dihedral), d)

    return (ERR_CODE_OK, vb + d)


@jit(nopython=True)
def _jit_calculate_everything(positions, c_table, zmat_values, start_row=0):
    for row in range(start_row, c_table.shape[0]):
        ref_pos = np.empty((3, 3))
        for k in range(3):
            j = c_table[row, k]
            if j < constants.keys_below_are_abs_refs:
                ref_pos[k] = constants._jit_absolute_refs(j)
            else:
                ref_pos[k] = positions[j]
        err, pos = _jit_calculate_single_position(ref_pos, zmat_values, row)
        if err == ERR_CODE_OK:
            positions[row] = pos
        else:
            break
    return (err, row)


class ZmatCore(PandasWrapper, GenericCore):
    """The main class for dealing with internal coordinates.
    """
    _required_cols = frozenset({'atom', 'b', 'bond', 'a', 'angle',
                                'd', 'dihedral'})
    _metadata_keys = frozenset({'last_valid_cartesian', 'has_dummies',
                                'dummy_manipulation_allowed'})

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
                _metadata['last_valid_cartesian'] = self.give_cartesian()
            if 'has_dummies' not in _metadata:
                _metadata['has_dummies'] = {}
            if 'dummy_manipulation_allowed' not in _metadata:
                _metadata['dummy_manipulation_allowed'] = True

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

    @staticmethod
    def _cast_correct_types(frame):
        new = frame.copy()
        zmat_values = ['bond', 'angle', 'dihedral']
        new.loc[:, zmat_values] = frame.loc[:, zmat_values].astype('f8')
        zmat_cols = ['b', 'a', 'd']
        new.loc[:, zmat_cols] = frame.loc[:, zmat_cols].astype('i8')
        return new

    def subs(self, variable, value, perform_checks=True):
        cols = ['bond', 'angle', 'dihedral']
        out = self.copy()

        def give_subs_function(variable, value):
            def subs_function(x):
                try:
                    x = x.subs(variable, value)
                    try:
                        x = float(x)
                    except TypeError:
                        pass
                except AttributeError:
                    pass
                return x
            return subs_function

        for col in cols:
            if out.loc[:, col].dtype is np.dtype('O'):
                out.unsafe_loc[:, col] = out.loc[:, col].map(
                    give_subs_function(variable, value))
                try:
                    out.unsafe_loc[:, col] = out.loc[:, col].astype('float')
                except TypeError:
                    pass
        if perform_checks:
            try:
                out._metadata['last_valid_cartesian'] = out.give_cartesian()
            except AttributeError:
                # Unevaluated symbolic expressions are remaining.
                pass
            except InvalidReference as e:
                if out._metadata['dummy_manipulation_allowed']:
                    out._manipulate_dummies(e, inplace=True)
                else:
                    raise e
        return out

    def _to_Zmat(self):
        return self.copy()

    def change_numbering(self, new_index=None, inplace=False,
                         exclude_upper_triangle=True):
        """Change numbering to a new index.

        Changes the numbering of index and all dependent numbering
            (bond_with...) to a new_index.
        The user has to make sure that the new_index consists of distinct
            elements.

        Args:
            new_index (list): If None the new_index is taken from 1 to the
                number of atoms.
            exclude_upper_triangle (bool): Exclude the upper triangle from
                being replaced with the new index

        Returns:
            Zmat: Reindexed version of the zmatrix.
        """
        cols = ['b', 'a', 'd']
        out = self if inplace else self.copy()

        if (new_index is None):
            new_index = range(len(self))
        elif len(new_index) != len(self):
            raise ValueError('len(new_index) has to be the same as len(self)')

        if exclude_upper_triangle:
            previous = [out.iloc[i, [1, 3, 5][i:]]
                        if i < 2 else out.iloc[i, 5]
                        for i in range(min(len(self), 3))]
            out.unsafe_loc[:, cols] = out.loc[:, cols].replace(
                out.index, new_index)
            for i in range(min(len(self), 3)):
                out.unsafe_iloc[i, [1, 3, 5][i:]] = previous[i]
        else:
            out.unsafe_loc[:, cols] = out.loc[:, cols].replace(
                out.index, new_index)
        out._frame.index = new_index
        if not inplace:
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

        def insert_dummy(zmat, i, dummy_cart, dummy_d, inplace=inplace):
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
            zmat._metadata['last_valid_cartesian'] = zmat.give_cartesian()
        except InvalidReference as e:
            zmat._insert_dummy_zmat(e, inplace=True)

        if not inplace:
            return zmat

    def _has_removable_dummies(self):
        has_dummies = self._metadata['has_dummies']
        to_be_tested = has_dummies.keys()
        c_table = self.loc[to_be_tested, ['b', 'a', 'd']]
        c_table['d'] = [has_dummies[i]['actual_d'] for i in to_be_tested]
        xyz = self.give_cartesian().loc[:, ['x', 'y', 'z']]
        BA = (xyz.loc[c_table['a']].values - xyz.loc[c_table['b']].values)
        AD = (xyz.loc[c_table['d']].values - xyz.loc[c_table['a']].values)

        remove = ~np.isclose(np.cross(BA, AD), np.zeros_like(BA)).all(axis=1)
        return [k for r, k in enumerate(to_be_tested) if remove[r]]

    def _remove_dummies(self, to_remove=None, inplace=False):
        """Works INPLACE"""
        cols = ['b', 'a', 'd']
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

        zmat_values = zmat.give_cartesian()._calculate_zmat_values(c_table)
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
