from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import os

import chemcoord as cc
import sympy


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_structure_path(script_path):
    test_path = os.path.join(script_path)
    while True:
        structure_path = os.path.join(test_path, 'structures')
        if os.path.exists(structure_path):
            return structure_path
        else:
            test_path = os.path.join(test_path, '..')


STRUCTURE_PATH = get_structure_path(get_script_path())


def test_multiple_subs():
    # Issue 46
    path = os.path.join(STRUCTURE_PATH, 'water.xyz')
    water = cc.Cartesian.read_xyz(path, start_index=1)

    zwater = water.get_zmat()
    symb_zwater = zwater.copy()
    a, b, c = sympy.symbols('a, b, c')
    symb_zwater.safe_loc[4, 'dihedral'] = a
    symb_zwater.safe_loc[5, 'dihedral'] = b
    symb_zwater.subs(a, 180)
    symb_zwater.safe_loc[6, 'dihedral'] = c
    symb_zwater.subs(a, 180.)
