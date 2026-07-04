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

# The reference path holds 80 images: 20 for each of the four schedules below.
reference_path = read_multiple_xyz(get_complete_path("correct_path.xyz"))


def _assert_ric_path(schedule, expected):
    # ``expected`` doubles as the seed (the interpolation should reproduce it).
    path = RIC_interpolate(
        molecule1,
        molecule2,
        20,
        schedule=schedule,
        atol=1e-8,
        seeds=expected,
        opt_alg="LM",
    )
    for ref, just_read in zip(path, expected):
        assert allclose(ref, just_read, atol=1e-4, align=True)


# ``test_path`` was split per-schedule so pytest emits output between the
# (numba-compilation-heavy) interpolations, keeping CI under its no-output
# timeout. See https://github.com/mcocdawc/chemcoord for context.
def test_path_independent():
    _assert_ric_path("independent", reference_path[:20])


def test_path_from_both():
    _assert_ric_path("from_both", reference_path[20:40])


def test_path_from_start():
    _assert_ric_path("from_start", reference_path[40:60])


def test_path_from_end():
    _assert_ric_path("from_end", reference_path[60:])


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
