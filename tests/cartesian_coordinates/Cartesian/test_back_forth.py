from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import chemcoord as cc
from chemcoord.xyz_functions import allclose
import pytest
from chemcoord.exceptions import UndefinedCoordinateSystem
import itertools
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


def back_and_forth(filepath):
    molecule1 = cc.Cartesian.read_xyz(filepath)
    zmolecule = molecule1.get_zmat()
    molecule2 = zmolecule.get_cartesian()
    assert allclose(molecule1, molecule2, align=False, atol=1e-7)


def test_back_and_forth1():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'MIL53_small.xyz'))


def test_back_and_forth1():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'MIL53_middle.xyz'))


def test_back_and_forth2():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'ruthenium.xyz'))


def test_back_and_forth3():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'Cd_lattice.xyz'))


def test_back_and_forth4():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'nasty_cube.xyz'))


def test_specified_c_table_assert_first_three_nonlinear():
    path = os.path.join(STRUCTURE_PATH, 'MIL53_beta.xyz')
    molecule = cc.Cartesian.read_xyz(path, start_index=1)
    fragment = molecule.get_fragment([(12, 2), (55, 2), (99, 2)])
    connection = fragment.get_construction_table()
    connection.loc[2] = [99, 12, 18]
    connection.loc[8] = [2, 99, 12]
    connection = connection.loc[[2, 8]]
    c_table = molecule.get_construction_table(
        fragment_list=[(fragment, connection)], perform_checks=False)
    with pytest.raises(UndefinedCoordinateSystem):
        c_table = molecule.correct_dihedral(c_table)

    new = c_table.iloc[:4].copy()

    new.loc[99] = c_table.loc[17]
    new.loc[17] = c_table.loc[99]
    new = new.loc[[17, 12, 55, 99]]
    new.loc[12, 'b'] = 17
    new.loc[55, 'b'] = 17
    new.loc[99, 'a'] = 17

    c_table = molecule.get_construction_table(
        fragment_list=[(molecule.get_without(fragment)[0], new),
                       (fragment, connection)])
    c_table = molecule.correct_dihedral(c_table)
    zmolecule = molecule.get_zmat(c_table)
    assert allclose(molecule, zmolecule.get_cartesian(), align=False,
                    atol=1e-6)


def test_issue_18():
    path = os.path.join(STRUCTURE_PATH, 'temp_lig.xyz')
    a = cc.Cartesian.read_xyz(path)
    con_table = a.get_construction_table()
    amat = a.get_zmat(con_table)
    a_new = amat.get_cartesian()
    # without construction table
    b = cc.Cartesian.read_xyz(path)
    bmat = a.get_zmat()
    b_new = bmat.get_cartesian()

    structures = [a, a_new, b, b_new]

    for i, j in itertools.product(structures, structures):
        assert cc.xyz_functions.allclose(i, j, align=False)
