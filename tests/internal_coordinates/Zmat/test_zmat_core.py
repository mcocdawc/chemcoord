import os
from os.path import join
from io import StringIO
from sympy import Symbol
import pytest

import chemcoord as cc
from chemcoord.xyz_functions import allclose


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_structure_path(script_path):
    test_path = join(script_path)
    while True:
        structure_path = join(test_path, 'structures')
        if os.path.exists(structure_path):
            return structure_path

        test_path = join(test_path, '..')


STRUCTURE_PATH = get_structure_path(get_script_path())


def test_addition_with_sympy():
    theta, x = Symbol('theta', real=True), Symbol('x', real=True)

    molecule = cc.Cartesian.read_xyz(
        join(STRUCTURE_PATH, 'MIL53_small.xyz'), start_index=1)

    zmolecule = molecule.get_zmat()

    zmolecule = molecule.get_zmat()
    zmolecule2 = zmolecule.copy()

    zmolecule2.unsafe_loc[:, ['bond', 'angle', 'dihedral']] = 0.
    zmolecule2.unsafe_loc[24, 'bond'] = x
    zmolecule2.unsafe_loc[32, 'bond'] = -x

    zmolecule = zmolecule + zmolecule2
    zmolecule.subs(x, 3)


def test_write_and_read():
    # Created because of https://github.com/mcocdawc/chemcoord/issues/58
    water_1 = cc.Cartesian.read_xyz(
        join(STRUCTURE_PATH, 'water.xyz'), start_index=1)
    z_water_str = water_1.get_zmat().to_zmat(upper_triangle=True)
    water_2 = cc.Zmat.read_zmat(StringIO(z_water_str)).get_cartesian()
    assert allclose(water_1, water_2, atol=1e-6)

    z_water_str = water_1.get_zmat().to_zmat(upper_triangle=False)
    cc.Zmat.read_zmat(StringIO(z_water_str)).get_cartesian()
