from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import itertools
import os
import sys

import chemcoord as cc
import numpy as np
import pandas as pd
import pytest
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


def test_grad_cartesian():
    path = os.path.join(STRUCTURE_PATH, 'MIL53_beta.xyz')
    molecule = cc.Cartesian.read_xyz(path, start_index=1)
    fragment = molecule.get_fragment([(12, 17), (55, 60)])
    connection = np.array([[3, 99, 1, 12], [17, 3, 99, 12], [60, 3, 17, 12]])
    connection = pd.DataFrame(connection[:, 1:], index=connection[:, 0],
                              columns=['b', 'a', 'd'])
    c_table = molecule.get_construction_table([(fragment, connection)])
    zmolecule = molecule.get_zmat(c_table)

    r = 0.3
    zmolecule2 = zmolecule.copy()
    zmolecule2.safe_loc[3, 'bond'] += r

    dist_zmol = zmolecule.copy()
    dist_zmol.unsafe_loc[:, ['bond', 'angle', 'dihedral']] = 0
    dist_zmol.unsafe_loc[3, 'bond'] = r

    new = zmolecule.get_grad_cartesian(chain=False)(
            dist_zmol).loc[:, ['x', 'y', 'z']]
    index = new.index[~np.isclose(new, 0.).all(axis=1)]
    assert (index == [3]).all()

    new = zmolecule.get_grad_cartesian()(dist_zmol).loc[:, ['x', 'y', 'z']]
    index = new.index[~np.isclose(new, 0.).all(axis=1)]
    assert (index
            == [3, 17, 60, 6, 19, 62, 38, 37, 81, 80, 7, 39, 82, 10]).all()
