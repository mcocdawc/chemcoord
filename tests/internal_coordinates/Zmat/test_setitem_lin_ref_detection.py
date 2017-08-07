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
    molecule = cc.Cartesian.read_xyz(
        os.path.join(STRUCTURE_PATH, 'water.xyz'), start_index=1)
    zmolecule = molecule.get_zmat()
    zmolecule1 = zmolecule.copy()

    angle_before_assignment = zmolecule1.loc[4, 'angle']

    with cc.DummyManipulation(False):
        with pytest.raises(InvalidReference):
            zmolecule1.safe_loc[4, 'angle'] = 180

    with pytest.warns(UserWarning):
        zmolecule1.safe_loc[4, 'angle'] = 180
        zmolecule1.safe_loc[5, 'dihedral'] = 90

    with pytest.warns(UserWarning):
        zmolecule1.safe_loc[4, 'angle'] = angle_before_assignment

    zmolecule2 = molecule.get_zmat()
    zmolecule2.safe_loc[5, 'dihedral'] = 90
    assert cc.xyz_functions.allclose(
        zmolecule2.get_cartesian(), zmolecule1.get_cartesian())

    zmolecule3 = zmolecule.copy()
    with cc.DummyManipulation(False):
        try:
            zmolecule3.safe_loc[4, 'angle'] = 180
        except InvalidReference as e:
            with pytest.warns(UserWarning):
                test = e.zmat_after_assignment._insert_dummy_zmat(e)
    assert len(test) == len(zmolecule3) + 1
