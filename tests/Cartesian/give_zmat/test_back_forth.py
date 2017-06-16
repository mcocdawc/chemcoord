import chemcoord as cc
import pytest
import os
import sys


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_structure_path(script_path):
    found, n = False, 0
    while not found:
        parents = ['..' for _ in range(n)]
        structure_path = os.path.join(script_path, *parents, 'structures')
        if os.path.exists(structure_path):
            return structure_path
        else:
            n += 1


STRUCTURE_PATH = get_structure_path(get_script_path())


def back_and_forth(filepath):
    molecule1 = cc.Cartesian.read_xyz(filepath)
    zmolecule = molecule1.give_zmat()
    molecule2 = zmolecule.give_cartesian()
    assert cc.xyz_functions.isclose(molecule1, molecule2, align=False)


def test_back_and_forth1():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'MIL53_small.xyz'))


def test_back_and_forth2():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'ruthenium.xyz'))


def test_back_and_forth3():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'Cd_lattice.xyz'))


def test_back_and_forth4():
    back_and_forth(os.path.join(STRUCTURE_PATH, 'nasty_cube.xyz'))
