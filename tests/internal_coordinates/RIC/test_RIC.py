import os

import numpy as np

from chemcoord import Cartesian
from chemcoord._redundant_internal_coordinates.main import (
    DefaultWeights,
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


def weight_vector(idx):
    """``diag(W)`` for the default weights, in the order of ``idx``."""
    default_weights = DefaultWeights()
    return np.array([default_weights.get_weight(coord) for coord in idx])


def weighted_residual(target, structure, idx, weights):
    """``‖W Δq‖`` -- the quantity the back-transformation actually minimises."""
    Δq = (target - structure.get_ric(internal_coords_idx=idx)).minimize_dihedral()
    return np.linalg.norm(weights * Δq.delta_q)


# The numbers below are a benchmark baseline, not only a regression guard. The
# coordinate assertions say the path did not move; ``‖W Δq‖`` says how well the
# back-transformation actually solved the problem it was given. Recording them here
# means a future change to the optimizer shows up as a number that got better or
# worse, rather than as a bare pass/fail. Lowering one is an improvement and should be
# committed together with the change that caused it; raising one needs a reason. Do
# not update them reflexively to make the suite green.
#
# The interpolation targets are only reconstructible for the ``independent`` schedule
# (image ``i`` targets ``q1 + i·Δq/(N-1)``). ``from_start``/``from_both``/``from_end``
# build each target from the *previously computed* image, so reproducing them here
# would mean duplicating the schedule; those are tracked as the weighted deviation
# from the reference image instead -- same metric, reference-relative rather than
# target-relative.

#: ‖W Δq‖ of each image of the ``independent`` schedule against its own target.
INDEPENDENT_RESIDUALS = (
    9.491e-16,
    5.656e-03,
    1.066e-02,
    1.502e-02,
    1.873e-02,
    2.181e-02,
    2.425e-02,
    2.607e-02,
    2.728e-02,
    2.787e-02,
    2.787e-02,
    2.725e-02,
    2.602e-02,
    2.419e-02,
    2.173e-02,
    1.865e-02,
    1.493e-02,
    1.059e-02,
    5.610e-03,
    1.657e-15,
)

#: ‖W Δq‖ of each image of the ``test_default_args`` path against its own target.
DEFAULT_ARGS_RESIDUALS = (
    9.000e-11,
    3.765e-02,
    7.488e-02,
    1.116e-01,
    1.477e-01,
    1.829e-01,
    2.241e-01,
    1.280e-01,
    6.873e-02,
    3.921e-15,
)

#: Bound on the weighted deviation from the reference image for the three schedules
#: whose targets are path-dependent. Observed maxima are ~2.2e-6 for all three; the
#: bound is an order of magnitude above that so BLAS differences do not flake it.
PATH_DEPENDENT_DEVIATION = 1e-5

#: ‖W Δq‖ of the ``x -> q(x) -> x`` round trips. Observed 5.4e-11 (MIL53_beta, with
#: bending coordinates) and 2.2e-13 (1A8I, ~7500 atoms); bounded well above both.
ROUND_TRIP_RESIDUAL = 1e-8


def _assert_ric_path(schedule, expected, residuals=None):
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

    idx = get_primitives_idx(molecule1, molecule2)
    weights = weight_vector(idx)
    if residuals is None:
        # Path-dependent schedule: track the weighted deviation from the reference.
        for ref, got in zip(expected, path):
            Δq = (ref.get_ric(idx) - got.get_ric(idx)).minimize_dihedral()
            assert np.linalg.norm(weights * Δq.delta_q) <= PATH_DEPENDENT_DEVIATION
    else:
        q1 = molecule1.get_ric(idx)
        Δ = (molecule2.get_ric(idx) - q1).minimize_dihedral()
        for i, (got, recorded) in enumerate(zip(path, residuals)):
            target = q1 + i * Δ / (len(path) - 1)
            assert (
                weighted_residual(target, got, idx, weights) <= recorded * 1.05 + 1e-12
            )


# ``test_path`` was split per-schedule so pytest emits output between the
# (numba-compilation-heavy) interpolations, keeping CI under its no-output
# timeout. See https://github.com/mcocdawc/chemcoord for context.
def test_path_independent():
    _assert_ric_path("independent", reference_path[:20], INDEPENDENT_RESIDUALS)


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
    assert (
        weighted_residual(q, test_cartesian, idx, weight_vector(idx))
        <= ROUND_TRIP_RESIDUAL
    )


def test_back_forth_large_molecule():
    # 1A8I is a ~7500-atom protein. This exercises the RIC back-transformation at that
    # scale and confirms it is seed-stable there (x(q(x)) == x): the default
    # ``lm_step="auto"`` uses the seed-stable full LM step, which reproduces the
    # structure it started from. (The molecule is read inside the test to keep it out
    # of module-import time.)
    molecule = Cartesian.read_xyz(get_complete_path("1A8I.xyz"))
    idx = get_primitives_idx(molecule, molecule)
    q = molecule.get_ric(internal_coords_idx=idx)
    test_cartesian = q.get_cartesian()
    assert allclose(test_cartesian, molecule, align=True)
    assert (
        weighted_residual(q, test_cartesian, idx, weight_vector(idx))
        <= ROUND_TRIP_RESIDUAL
    )


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

    idx = get_primitives_idx(molecule7, molecule8)
    weights = weight_vector(idx)
    q1 = molecule7.get_ric(idx)
    Δ = (molecule8.get_ric(idx) - q1).minimize_dihedral()
    for i, (got, recorded) in enumerate(zip(path, DEFAULT_ARGS_RESIDUALS)):
        target = q1 + i * Δ / (len(path) - 1)
        assert weighted_residual(target, got, idx, weights) <= recorded * 1.05 + 1e-12


def test_documented_default_weights_mapping():
    """The mapping spelled out in the ``default_weights`` docstrings must be usable.

    It is forwarded verbatim to ``DefaultWeights(**mapping)``, so a wrong key there
    is a ``TypeError`` for anyone copying it out of the docs.
    """
    documented = {"bond": 1.0, "angle": 0.1, "dihedral": 0.05, "bending": 0.01}

    assert DefaultWeights(**documented) == DefaultWeights()

    path = RIC_interpolate(molecule1, molecule2, 5, default_weights=documented)
    reference = RIC_interpolate(molecule1, molecule2, 5)

    for with_weights, without in zip(path, reference):
        assert allclose(with_weights, without, atol=1e-6, align=True)
