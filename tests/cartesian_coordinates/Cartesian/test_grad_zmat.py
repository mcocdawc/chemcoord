from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import itertools
import os
import sys

import chemcoord as cc
import numpy as np
import pandas as pd
import pytest
import sympy
from chemcoord.exceptions import UndefinedCoordinateSystem
from chemcoord.xyz_functions import allclose


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


def test_grad_zmat():
    path = os.path.join(STRUCTURE_PATH, 'MIL53_beta.xyz')
    molecule = cc.Cartesian.read_xyz(path, start_index=1)
    fragment = molecule.get_fragment([(12, 17), (55, 60)])
    connection = np.array([[3, 99, 1, 12], [17, 3, 99, 12], [60, 3, 17, 12]])
    connection = pd.DataFrame(connection[:, 1:], index=connection[:, 0],
                              columns=['b', 'a', 'd'])
    c_table = molecule.get_construction_table([(fragment, connection)])
    molecule = molecule.loc[c_table.index]

    x = sympy.symbols('x', real=True)

    dist_mol = molecule.copy()
    dist_mol.loc[:, ['x', 'y', 'z']] = 0.
    dist_mol.loc[13, 'x'] = x

    zmat_dist = molecule.get_grad_zmat(c_table)(dist_mol)

    moved_atoms = zmat_dist.index[
        (zmat_dist.loc[:, ['bond', 'angle', 'dihedral']] != 0.).any(axis=1)]

    assert moved_atoms[0] == 13
    assert np.alltrue(
        moved_atoms[1:] == c_table.index[(c_table == 13).any(axis=1)])
