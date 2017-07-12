from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import chemcoord as cc
from chemcoord.xyz_functions import allclose
import pytest
from chemcoord.exceptions import UndefinedCoordinateSystem, PhysicalMeaning
import itertools
import os
import sys
import numpy as np


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


def get_complete_path(structure):
    return os.path.join(STRUCTURE_PATH, structure)


molecule = cc.Cartesian.read_xyz(get_complete_path('MIL53_small.xyz'),
                                 start_index=1)
bond_dict = {1: {2, 51},
             2: {1, 9, 27},
             3: {6, 55, 56},
             4: {5, 52},
             5: {4, 15, 31},
             6: {3, 7, 8, 9, 15, 16},
             7: {6, 11, 53},
             8: {6, 10},
             9: {2, 6},
             10: {8, 12, 24},
             11: {7, 12, 13, 18, 19, 20},
             12: {10, 11},
             13: {11, 14},
             14: {13, 22, 33},
             15: {5, 6},
             16: {6, 17},
             17: {16, 20, 32},
             18: {11, 48, 54},
             19: {11, 21},
             20: {11, 17},
             21: {19, 23, 34},
             22: {14, 49},
             23: {21, 50},
             24: {10, 25, 26, 36},
             25: {24},
             26: {24},
             27: {2, 28, 29, 30},
             28: {27},
             29: {27},
             30: {27},
             31: {5, 41, 44, 47},
             32: {17, 35, 37, 38},
             33: {14, 40, 43, 46},
             34: {21, 39, 42, 45},
             35: {32},
             36: {24},
             37: {32},
             38: {32},
             39: {34},
             40: {33},
             41: {31},
             42: {34},
             43: {33},
             44: {31},
             45: {34},
             46: {33},
             47: {31},
             48: {18},
             49: {22},
             50: {23},
             51: {1},
             52: {4},
             53: {7},
             54: {18},
             55: {3},
             56: {3}}


def test_init():
    with pytest.raises(ValueError):
        cc.Cartesian(5)
    with pytest.raises(PhysicalMeaning):
        cc.Cartesian(molecule.loc[:, ['atom', 'x']])


def test_overloaded_operators():
    assert allclose(molecule + 1, molecule + [1, 1, 1])
    assert allclose(1 + molecule, molecule + [1, 1, 1])
    assert allclose(molecule + molecule, 2 * molecule)
    index = molecule.index
    assert allclose(molecule + molecule.loc[reversed(index)], 2 * molecule)
    assert allclose(molecule + molecule.loc[:, ['x', 'y', 'z']].values,
                    2 * molecule)
    assert allclose(1 * molecule, molecule)
    assert allclose(molecule * 1, molecule)
    assert allclose(1 * molecule, +molecule)
    assert allclose(-1 * molecule, -molecule)
    molecule2 = molecule[~(np.isclose(molecule['x'], 0)
                           | np.isclose(molecule['y'], 0)
                           | np.isclose(molecule['z'], 0))]
    assert np.allclose(np.full(molecule2.loc[:, ['x', 'y', 'z']].shape, 1),
                       (molecule2 / molecule2).loc[:, ['x', 'y', 'z']])
    assert np.allclose(np.full(molecule2.loc[:, ['x', 'y', 'z']].shape, 0),
                       (molecule2 - molecule2).loc[:, ['x', 'y', 'z']])


def test_get_bonds():
    assert bond_dict == molecule.get_bonds()
    molecule._metadata['bond_dict'][56].add(4)
    assert not (bond_dict == molecule.get_bonds(use_lookup=True))
    assert bond_dict == molecule.get_bonds()
    modified_expected = {1: {2, 51}, 2: {1, 9, 27}, 3: set(), 4: {5, 52},
                         5: {4, 31}, 6: set(), 7: {53}, 8: {10}, 9: {2},
                         10: {8, 12, 24}, 11: set(), 12: {10}, 13: {14},
                         14: {13}, 15: set(), 16: {17}, 17: {16, 20},
                         18: set(), 19: {21}, 20: {17}, 21: {19, 23},
                         22: set(), 23: {21, 50}, 24: {10, 26, 36},
                         25: set(), 26: {24}, 27: {2, 29, 30}, 28: set(),
                         29: {27}, 30: {27}, 31: {5, 41, 44, 47}, 32: set(),
                         33: set(), 34: set(), 35: set(), 36: {24}, 37: set(),
                         38: set(), 39: set(), 40: set(), 41: {31}, 42: set(),
                         43: set(), 44: {31}, 45: set(), 46: set(), 47: {31},
                         48: set(), 49: set(), 50: {23}, 51: {1}, 52: {4},
                         53: {7}, 54: set(), 55: set(), 56: set()}
    assert molecule.get_bonds(
        modified_properties={
            k: 0. for k in
            molecule[molecule['atom'] == 'C'].index}) == modified_expected


def test_coordination_sphere():
    expctd = {}
    expctd[1] = {6, 11, 53}
    expctd[2] = {3, 8, 9, 12, 13, 15, 16, 18, 19, 20}
    expctd[3] = {2, 5, 10, 14, 17, 21, 48, 54, 55, 56}
    expctd[4] = {1, 4, 22, 23, 24, 27, 31, 32, 33, 34}
    expctd[5] = {25, 26, 28, 29, 30, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                 44, 45, 46, 47, 49, 50, 51, 52}
    expctd[6] = set()

    for i in expctd:
        assert expctd[i] == set(molecule.give_coordination_sphere(7, i).index)


def test_cutsphere():
    expected = {6, 7, 8, 9, 11, 12, 13, 15, 16, 19, 20, 53}
    assert expected == set(molecule.cutsphere(radius=3, origin=7).index)
    assert np.alltrue(
        molecule == molecule.cutsphere(radius=3, origin=7,
                                       preserve_bonds=True))
    assert (set(molecule.index) - expected
            == set(molecule.cutsphere(radius=3, origin=7,
                                      outside_sliced=False).index))


def test_cutcuboid():
    expected = {3, 4, 5, 6, 7, 15, 16, 17, 32, 35, 37, 38, 47, 52, 53, 55, 56}
    assert expected == set(molecule.cutcuboid(a=2, origin=7).index)
    assert np.alltrue(
        molecule == molecule.cutcuboid(a=2, origin=7,
                                       preserve_bonds=True))
    assert (set(molecule.index) - expected
            == set(molecule.cutcuboid(a=2, origin=7,
                                      outside_sliced=False).index))


def test_inertia():
    A = molecule.inertia()
    eig, t_mol = A['eigenvectors'], A['transformed_Cartesian']
    cc.xyz_functions.allclose(eig @ (molecule - molecule.barycenter()), t_mol)
