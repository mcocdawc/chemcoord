import os

import numpy as np

from chemcoord import Cartesian
from chemcoord._redundant_internal_coordinates.main import (
    RIC_interpolate,
    get_primitives_idx,
)
from chemcoord.typing import AtomIdx
from chemcoord.xyz_functions import allclose, read_multiple_xyz


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


molecule1 = Cartesian.read_xyz(get_complete_path("cyclohexane_chair.xyz"))
molecule2 = Cartesian.read_xyz(get_complete_path("cyclohexane_twist_boat.xyz"))
molecule3 = Cartesian.read_xyz(get_complete_path("peroxide.xyz"))
molecule4 = Cartesian.read_xyz(get_complete_path("MIL53_beta.xyz"))
molecule5 = Cartesian.read_xyz(
    get_complete_path("cyclohexane_chair.xyz"), start_index=72
)
molecule6 = Cartesian.read_xyz(
    get_complete_path("cyclohexane_twist_boat.xyz"), start_index=72
)
molecule7 = Cartesian.read_xyz(get_complete_path("default_args_start.xyz"))
molecule8 = Cartesian.read_xyz(get_complete_path("default_args_end.xyz"))


def test_path():
    reference_path = read_multiple_xyz(get_complete_path("correct_path.xyz"))

    path1 = RIC_interpolate(
        molecule1,
        molecule2,
        20,
        schedule="independent",
        atol=1e-8,
        seeds=reference_path[:20],
        opt_alg="LM",
    )
    path2 = RIC_interpolate(
        molecule1,
        molecule2,
        20,
        schedule="from_both",
        atol=1e-8,
        seeds=reference_path[20:40],
        opt_alg="LM",
    )
    path3 = RIC_interpolate(
        molecule1,
        molecule2,
        20,
        schedule="from_start",
        atol=1e-8,
        seeds=reference_path[40:60],
        opt_alg="LM",
    )
    path4 = RIC_interpolate(
        molecule1,
        molecule2,
        20,
        schedule="from_end",
        atol=1e-8,
        seeds=reference_path[60:],
        opt_alg="LM",
    )

    path = path1 + path2 + path3 + path4

    for ref, just_read in zip(path, reference_path):
        assert allclose(ref, just_read, atol=1e-4, align=True)


def test_back_forth_with_bending():
    idx = get_primitives_idx(molecule4, molecule4)
    q = molecule4.get_ric(internal_coords_idx=idx)
    test_cartesian = q.get_cartesian()
    assert allclose(test_cartesian, molecule4, align=True)


def test_set_coord():
    idx = get_primitives_idx(molecule4, molecule4)
    q = molecule4.get_ric(internal_coords_idx=idx)

    q[[(2, 5), (2, 5, 61), (2, 5, 61, 80)]] = [0.1, 0.3, 0.5]

    assert (
        q[(AtomIdx(2), AtomIdx(5))] == 0.1
        and q[(AtomIdx(2), AtomIdx(5), AtomIdx(61))] == 0.3
        and q[(AtomIdx(2), AtomIdx(5), AtomIdx(61), AtomIdx(80))] == 0.5
    )

    assert np.allclose(
        q[[(5, 2), (61, 5, 2), (80, 61, 5, 2)]], q[[(2, 5), (2, 5, 61), (2, 5, 61, 80)]]
    )

    q[[(5, 2), (61, 5, 2), (80, 61, 5, 2)]] = [20, 10, 5]

    assert (
        q[(AtomIdx(2), AtomIdx(5))] == 20
        and q[(AtomIdx(2), AtomIdx(5), AtomIdx(61))] == 10
        and q[(AtomIdx(2), AtomIdx(5), AtomIdx(61), AtomIdx(80))] == 5
    )


def test_nonzero_start():
    RIC_interpolate(molecule1, molecule2, 20)


def test_default_args():
    correct_path = get_complete_path("default_args_path.xyz")

    reference_path = read_multiple_xyz(correct_path)

    path = RIC_interpolate(
        molecule7, molecule8, 10, atol=1e-8, seeds=reference_path, opt_alg="LM"
    )

    for ref, just_read in zip(path, reference_path):
        assert allclose(ref, just_read, atol=1e-4, align=True)
