# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from numba import jit
import pandas as pd
import math as m
import warnings
from chemcoord.internal_coordinates._zmat_class_core import Zmat_core
from chemcoord.utilities import algebra_utilities
from chemcoord.utilities.algebra_utilities import _jit_normalize, \
    _jit_rotation_matrix, _jit_isclose, _jit_cross
from chemcoord.configuration import settings


@jit(nopython=True)
def _jit_calculate_position(references, zmat_values, row):
    bond, angle, dihedral = zmat_values[row]
    vb, va, vd = references

    BA = va - vb
    ba = _jit_normalize(BA)
    if _jit_isclose(angle, np.pi):
        d = bond * -ba
    elif _jit_isclose(angle, 0.):
        d = bond * ba
    else:
        AD = vd - va
        n1 = _jit_normalize(_jit_cross(BA, AD))
        d = bond * ba
        d = np.dot(_jit_rotation_matrix(n1, angle), d)
        d = np.dot(_jit_rotation_matrix(ba, dihedral), d)
    return vb + d


@jit(nopython=True)
def _jit_calculate_rest(positions, c_table, zmat_values):
    for row in range(3, len(c_table)):
        b, a, d = c_table[row, :]
        references = positions[b], positions[a], positions[d]
        positions[row] = _jit_calculate_position(references, zmat_values, row)


class Zmat_give_cartesian(Zmat_core):
    """The main class for dealing with internal coordinates.
    """
    def give_cartesian(self):
        abs_refs = self._metadata['abs_refs']
        old_index = self.index
        self.change_numbering(inplace=True)
        c_table = self.loc[:, ['b', 'a', 'd']].values
        zmat_values = self.loc[:, ['bond', 'angle', 'dihedral']].values
        zmat_values[:, [1, 2]] = np.radians(zmat_values[:, [1, 2]])
        positions = np.empty((len(self), 3), dtype='float64', order='C')

        def get_ref_first_three_atoms(c_table, positions, row):
            b, a, d = c_table[row, :]
            if row == 0:
                vb = abs_refs[b][0]
                va = abs_refs[a][0]
            elif row == 1:
                vb = positions[b]
                va = abs_refs[a][0]
            elif row == 2:
                vb = positions[b]
                va = positions[a]
            vd = abs_refs[d][0]
            return vb, va, vd

        for row in range(min(3, len(c_table))):
            refs = get_ref_first_three_atoms(c_table, positions, row)
            positions[row] = _jit_calculate_position(refs, zmat_values, row)

        _jit_calculate_rest(positions, c_table, zmat_values)

        xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'])
        xyz_frame['atom'] = self['atom']
        xyz_frame.loc[:, ['x', 'y', 'z']] = positions

        self.change_numbering(old_index, inplace=True)
        xyz_frame.index = self.index
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
