from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import chemcoord as cc
from chemcoord.xyz_functions import allclose
import pytest
from chemcoord.exceptions import UndefinedCoordinateSystem, InvalidReference
import os
import sys
from sympy import Symbol


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


def test_addition_with_sympy():
    theta, x = Symbol('theta', real=True), Symbol('x', real=True)

    molecule = cc.Cartesian.read_xyz(
        os.path.join(STRUCTURE_PATH, 'MIL53_small.xyz'), start_index=1)

    zmolecule = molecule.get_zmat()

    zmolecule = molecule.get_zmat()
    zmolecule2 = zmolecule.copy()

    zmolecule2.unsafe_loc[:, ['bond', 'angle', 'dihedral']] = 0.
    zmolecule2.unsafe_loc[24, 'bond'] = x
    zmolecule2.unsafe_loc[32, 'bond'] = -x

    zmolecule = zmolecule + zmolecule2
    zmolecule.subs(x, 3)
