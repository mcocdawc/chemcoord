import os

import pandas as pd
import pytest

import chemcoord as cc
from chemcoord.xyz_functions import allclose

pd.set_option("future.no_silent_downcasting", True)


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_structure_path(script_path):
    test_path = os.path.join(script_path)
    while True:
        structure_path = os.path.join(test_path, "structures")
        if os.path.exists(structure_path):
            return structure_path
        else:
            test_path = os.path.join(test_path, "..")


STRUCTURES = get_structure_path(get_script_path())


def test_concat():
    path = os.path.join(STRUCTURES, "MIL53_small.xyz")
    molecule = cc.Cartesian.read_xyz(path)
    cartesians = [molecule + i * 10 for i in range(5)]

    with pytest.raises(ValueError):
        cc.xyz_functions.concat(cartesians)

    joined = cc.xyz_functions.concat(cartesians, ignore_index=True)
    for i, molecule in enumerate(cartesians):
        molecule2 = joined.iloc[(i * len(molecule)) : (i + 1) * len(molecule)]
        molecule2.index = molecule.index
        assert allclose(molecule, molecule2)

    key_joined = cc.xyz_functions.concat(cartesians, keys=["a", "b", "c", "d", "e"])
    assert allclose(key_joined.loc["a"], (key_joined.loc["b"] - 10))


def test_concat_with_zmats():
    path = os.path.join(STRUCTURES, "MIL53_small.xyz")
    m = cc.Cartesian.read_xyz(path, start_index=1)
    zm = m.get_zmat()
    zm1 = zm.change_numbering(new_index=range(1, len(zm) + 1))
    zm2 = zm.change_numbering(new_index=zm1.index + len(zm1))

    new = cc.xyz_functions.concat(
        [zm1.get_cartesian(), zm2.get_cartesian() + [0, 0, 10]]
    )

    c_table = zm2.loc[:, ["b", "a", "d"]]
    c_table.loc[57, ["b", "a", "d"]] = [4, 1, 2]
    c_table.loc[58, ["a", "d"]] = [1, 2]
    c_table.loc[59, "d"] = 1

    large_c_table = new.get_construction_table(
        fragment_list=[(zm2.get_cartesian(), c_table)]
    )
    znew = new.get_zmat(large_c_table)

    cc.xyz_functions.allclose(new, znew.get_cartesian())

    znew.safe_loc[57, "bond"] = 20.0 - 0.89
    assert allclose(
        cc.xyz_functions.concat(
            [zm1.get_cartesian(), zm2.get_cartesian() + [0, 0, 20]]
        ),
        znew.get_cartesian(),
    )


def test_multiple_xyz():
    output_path = os.path.join(STRUCTURES, "xyz_test.xyz")

    path1 = os.path.join(STRUCTURES, "MIL53_small.xyz")
    path2 = os.path.join(STRUCTURES, "nasty_cube.xyz")
    path3 = os.path.join(STRUCTURES, "MeOH_Furan_end.xyz")
    molecule1 = cc.Cartesian.read_xyz(path1)
    molecule2 = cc.Cartesian.read_xyz(path2)
    molecule3 = cc.Cartesian.read_xyz(path3)

    cartesian_list = [molecule1, molecule2, molecule3]

    cc._cartesian_coordinates.xyz_functions.multiple_to_xyz(cartesian_list, output_path)
    test_cartesian_list = cc._cartesian_coordinates.xyz_functions.read_multiple_xyz(
        output_path
    )

    for i in range(len(cartesian_list)):
        assert allclose(
            cartesian_list[i],
            test_cartesian_list[i],
        )
