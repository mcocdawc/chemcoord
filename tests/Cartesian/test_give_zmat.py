import chemcoord as cc
import pytest


def test_back_and_forth():
    molecule = cc.Cartesian.read_xyz('MIL53_small.xyz')
    molecule2 = molecule.give_zmat().give_cartesian()
    assert cc.xyz_functions.isclose(molecule, molecule2, atol=1e-3, rtol=1e-4)
