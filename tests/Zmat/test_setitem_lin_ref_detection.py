import chemcoord as cc
from chemcoord.xyz_functions import allclose
import pytest
from chemcoord._exceptions import UndefinedCoordinateSystem
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


STRUCTURE_PATH = get_structure_path(get_script_path())


def test_assignment_leading_to_linear_reference():
    molecule = cc.Cartesian.read_xyz(os.path.join(STRUCTURE_PATH, 'water.xyz'))
    zmolecule = molecule.give_zmat()

    with pytest.raises(UndefinedCoordinateSystem):
        zmolecule.loc[4, 'angle'] = 180
