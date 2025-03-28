import os
from io import StringIO
from itertools import accumulate
from os.path import join
from typing import Final

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

import chemcoord as cc
from chemcoord.xyz_functions import allclose

pd.set_option("future.no_silent_downcasting", True)


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_structure_path(script_path):
    test_path = join(script_path)
    while True:
        structure_path = join(test_path, "structures")
        if os.path.exists(structure_path):
            return structure_path

        test_path = join(test_path, "..")


STRUCTURE_PATH = get_structure_path(get_script_path())


def test_assignment():
    x = Symbol("x", real=True)

    molecule = cc.Cartesian.read_xyz(
        join(STRUCTURE_PATH, "MIL53_small.xyz"), start_index=1
    )
    zmolecule = molecule.get_zmat()

    zmolecule.safe_loc[:, "bond"] = x
    assert zmolecule.bond.dtype == np.dtype("O")

    assert zmolecule.subs(x, 10).bond.dtype == np.dtype("f8")


def test_addition_with_sympy():
    x = Symbol("x", real=True)

    molecule = cc.Cartesian.read_xyz(
        join(STRUCTURE_PATH, "MIL53_small.xyz"), start_index=1
    )

    zmolecule = molecule.get_zmat()

    zmolecule = molecule.get_zmat()
    zmolecule2 = zmolecule.copy()

    zmolecule2.unsafe_loc[:, ["bond", "angle", "dihedral"]] = 0.0
    zmolecule2.unsafe_loc[24, "bond"] = x
    zmolecule2.unsafe_loc[32, "bond"] = -x

    zmolecule = zmolecule + zmolecule2
    zmolecule.subs(x, 3)


def test_indexing():
    molecule = cc.Cartesian.read_xyz(
        join(STRUCTURE_PATH, "MIL53_small.xyz"), start_index=1
    ).get_zmat()
    assert (molecule.b == molecule.loc[:, "b"]).all()
    assert (molecule.a == molecule.loc[:, "a"]).all()
    assert (molecule.atom == molecule.loc[:, "atom"]).all()
    assert ((molecule.bond - molecule.loc[:, "bond"]) == 0).all()


def test_write_and_read():
    # Created because of https://github.com/mcocdawc/chemcoord/issues/58
    water_1 = cc.Cartesian.read_xyz(join(STRUCTURE_PATH, "water.xyz"), start_index=1)
    z_water_str = water_1.get_zmat().to_zmat(upper_triangle=True)
    water_2 = cc.Zmat.read_zmat(StringIO(z_water_str)).get_cartesian()
    assert allclose(water_1, water_2, atol=1e-6)

    z_water_str = water_1.get_zmat().to_zmat(upper_triangle=False)
    cc.Zmat.read_zmat(StringIO(z_water_str)).get_cartesian()


def test_pure_internal_move():
    ref = cc.Cartesian.read_xyz(join(STRUCTURE_PATH, "water.xyz"))
    zm = ref.get_zmat()

    def set_angle(zm, a):
        zm = zm.copy()
        zm.safe_loc[2, "angle"] = a
        return zm.get_cartesian()

    with cc.zmat_functions.PureInternalMovement(True):
        structures = [set_angle(zm, 106 + a) for a in range(-30, 40, 10)]

    for m in structures:
        assert cc.xyz_functions.allclose(m, ref.align(m, mass_weight=True)[1])


def test_MeOH_Furan_interpolation():
    start = cc.Cartesian.read_xyz(
        join(STRUCTURE_PATH, "MeOH_Furan_start.xyz"), start_index=1
    )
    end = cc.Cartesian.read_xyz(
        join(STRUCTURE_PATH, "MeOH_Furan_end.xyz"), start_index=1
    )
    N: Final = 21

    interpolated = cc.xyz_functions.read_molden(
        join(STRUCTURE_PATH, "MeOH_Furan_interpolated.molden"),
        start_index=1,
    )

    z_start = start.get_zmat()
    z_end = end.get_zmat(z_start.loc[:, ["b", "a", "d"]])

    with cc.zmat_functions.TestOperators(False):
        diff: Final = (z_end - z_start).minimize_dihedrals() / (N - 1)

    with pytest.warns(UserWarning), cc.zmat_functions.CleanDihedralOrientation(False):
        with cc.zmat_functions.TestOperators(False):
            z_steps = [i * diff for i in range(N)]

        assert allclose(start, (z_start + z_steps[0]).get_cartesian())
        assert allclose(end, (z_start + z_steps[-1]).get_cartesian())

        assert all(
            allclose((z_start + D).get_cartesian(), ref, atol=5e-6)
            for D, ref in zip(z_steps, interpolated)
        )

    with cc.zmat_functions.CleanDihedralOrientation(True):
        with cc.zmat_functions.TestOperators(False):
            z_steps = [i * diff for i in range(N)]

        assert allclose(start, (z_start + z_steps[0]).get_cartesian())
        assert not allclose(end, (z_start + z_steps[-1]).get_cartesian())

        assert not all(
            allclose((z_start + D).get_cartesian(), ref, atol=5e-6)
            for D, ref in zip(z_steps, interpolated)
        )

    for clean_orientation in (True, False):
        with cc.zmat_functions.CleanDihedralOrientation(clean_orientation):
            cartesians = [
                zm.get_cartesian()
                for zm in accumulate((diff for _ in range(N - 1)), initial=z_start)
            ]

            assert allclose(start, cartesians[0])
            assert allclose(end, cartesians[-1])
            assert all(
                allclose(m, ref, atol=5e-6) for m, ref in zip(cartesians, interpolated)
            )

    cartesians = cc.interpolate(start, end, 21, coord="zmat")
    assert allclose(start, cartesians[0])
    assert allclose(end, cartesians[-1])
    assert all(allclose(m, ref, atol=5e-6) for m, ref in zip(cartesians, interpolated))


def test_water_dimer_interpolation():
    molecule = cc.Cartesian.read_xyz(join(STRUCTURE_PATH, "water.xyz"), start_index=1)
    reference_cartesians = cc.xyz_functions.read_molden(
        join(STRUCTURE_PATH, "water_dimer_interpolated.molden"), start_index=1
    )
    zmolecule = molecule.get_zmat()

    # Assert that dummy atoms were inserted; the UserWarning is expected
    with pytest.warns(UserWarning), cc.zmat_functions.CleanDihedralOrientation(True):
        cartesians = [
            x.get_cartesian().loc[lambda m: m.atom != "X"]
            for x in (
                zmolecule.assign(4, "angle", x) for x in np.linspace(150, 210, 21)
            )
        ]
    assert all(
        allclose(m, ref, atol=5e-5) for m, ref in zip(cartesians, reference_cartesians)
    )
