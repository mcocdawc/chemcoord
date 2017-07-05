import chemcoord as cc
from chemcoord.xyz_functions import allclose
import pytest
from chemcoord._exceptions import UndefinedCoordinateSystem
import itertools
import numpy as np
import os
import sys


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


STRUCTURES = get_structure_path(get_script_path())


def test_concat():
    path = os.path.join(STRUCTURES, 'MIL53_small.xyz')
    molecule = cc.Cartesian.read_xyz(path)
    cartesians = [molecule + i * 10 for i in range(5)]

    with pytest.raises(ValueError):
        cc.xyz_functions.concat(cartesians)

    joined = cc.xyz_functions.concat(cartesians, ignore_index=True)
    for i, molecule in enumerate(cartesians):
        molecule2 = joined.iloc[(i * len(molecule)):(i + 1) * len(molecule)]
        molecule2.index = molecule.index
        assert allclose(molecule, molecule2)

    key_joined = cc.xyz_functions.concat(cartesians, keys=['a', 'b'])
    assert allclose(key_joined.loc['a'], (key_joined.loc['b'] - 10))
