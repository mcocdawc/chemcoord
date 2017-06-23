# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord.internal_coordinates._zmat_class_core import Zmat_core
from chemcoord.utilities.algebra_utilities import \
    _jit_normalize, \
    _jit_rotation_matrix, \
    _jit_isclose, \
    _jit_cross
from numba import jit
from chemcoord._exceptions import ERR_CODE_OK, \
    InvalidReference, ERR_CODE_InvalidReference
import numpy as np
import pandas as pd
import warnings


@jit(nopython=True)
def _jit_calculate_position(references, zmat_values, row):
    bond, angle, dihedral = zmat_values[row]
    vb, va, vd = references
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
def _jit_calculate_rest(positions, c_table, zmat_values, start_row=3):
    for row in range(start_row, len(c_table)):
        b, a, d = c_table[row, :]
        refs = positions[b], positions[a], positions[d]
        err, pos = _jit_calculate_position(refs, zmat_values, row)
        if err == ERR_CODE_OK:
            positions[row] = pos
        elif err == ERR_CODE_InvalidReference:
            return row


class Zmat_give_cartesian(Zmat_core):
    """The main class for dealing with internal coordinates.
    """
    def give_cartesian(self):
        abs_refs = self._metadata['abs_refs']
        old_index = self.index
        rename = dict(enumerate(old_index))
        self.change_numbering(inplace=True)
        c_table = self.loc[:, ['b', 'a', 'd']].values
        zmat_values = self.loc[:, ['bond', 'angle', 'dihedral']].values
        zmat_values[:, [1, 2]] = np.radians(zmat_values[:, [1, 2]])
        positions = np.empty((len(self), 3), dtype='float64')

        for row in range(min(3, len(c_table))):
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
            refs = vb, va, vd

            err, pos = _jit_calculate_position(refs, zmat_values, row)
            if err == ERR_CODE_OK:
                positions[row] = pos
            elif err == ERR_CODE_InvalidReference:
                print('Error Handling required', rename[row])

        row = _jit_calculate_rest(positions, c_table, zmat_values)
        if row < len(self):
            print('Error handling required', rename[row])

        xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'], dtype=float)
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
