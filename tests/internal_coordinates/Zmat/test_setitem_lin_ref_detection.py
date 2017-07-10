import chemcoord as cc
from chemcoord.xyz_functions import allclose
import pytest
from chemcoord.exceptions import UndefinedCoordinateSystem, InvalidReference
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
    zmolecule1 = molecule.give_zmat()

    angle_before_assignment = zmolecule1.loc[4, 'angle']

    with cc.dummy_manipulation(zmolecule1, False):
        with pytest.raises(InvalidReference):
            zmolecule1.safe_loc[3, 'angle'] = 180

    with pytest.warns(UserWarning):
        zmolecule1.safe_loc[4, 'angle'] = 180
        zmolecule1.safe_loc[5, 'dihedral'] = 90

    with pytest.warns(UserWarning):
        zmolecule1.safe_loc[4, 'angle'] = angle_before_assignment

    zmolecule2 = molecule.give_zmat()
    zmolecule2.unsafe_loc[5, 'dihedral'] = 90
    assert cc.xyz_functions.allclose(
        zmolecule2.give_cartesian(), zmolecule1.give_cartesian())
