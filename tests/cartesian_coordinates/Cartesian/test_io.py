import os

import pandas as pd
import pytest

import chemcoord as cc
from chemcoord.xyz_functions import allclose

try:
    pd.set_option("future.no_silent_downcasting", True)
except:
    # Yes I want a bare except
    pass


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


def get_complete_path(structure):
    STRUCTURES = get_structure_path(get_script_path())
    return os.path.join(STRUCTURES, structure)


molecule = cc.Cartesian.read_xyz(get_complete_path("water.xyz"), start_index=1)


def test_to_string():
    expected = (
        "  atom         x    y         z\n"
        "1    O  0.000000  0.0  0.000000\n"
        "2    H  0.758602  0.0  0.504284\n"
        "3    H  0.260455  0.0 -0.872893\n"
        "4    O  3.000000  0.5  0.000000\n"
        "5    H  3.758602  0.5  0.504284\n"
        "6    H  3.260455  0.5 -0.872893"
    )
    assert molecule.to_string() == expected


def test_to_xyz():
    base = """6
Created by chemcoord http://chemcoord.readthedocs.io/
{}O 0.000000 0.000000  0.000000
{}H 0.758602 0.000000  0.504284
{}H 0.260455 0.000000 -0.872893
{}O 3.000000 0.500000  0.000000
{}H 3.758602 0.500000  0.504284
{}H 3.260455 0.500000 -0.872893"""
    expected = [
        base.format(*6 * [i]) for i in ["", " "]
    ]  # accepts both initial whitespace or not
    assert molecule.to_xyz() in expected

    with pytest.warns(DeprecationWarning):
        assert molecule.write_xyz() in expected


def test_xyz_cast():
    molecule_float = cc.Cartesian.read_xyz(get_complete_path("hydrogen_float.xyz"))

    molecule_int = cc.Cartesian.read_xyz(get_complete_path("hydrogen_int.xyz"))

    assert molecule_int.x.dtype == float

    assert allclose(molecule_float, molecule_int)
