import os

import numpy as np
import pytest

from chemcoord import Cartesian
from chemcoord._redundant_internal_coordinates.main import (
    DefaultWeights,
    RIC_interpolate,
    get_primitives_idx,
)
from chemcoord.typing import AtomIdx
from chemcoord.xyz_functions import allclose, interpolate, read_multiple_xyz


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
    # scale and confirms it is seed-stable there (x(q(x)) == x): seeded with a converged
    # structure it reproduces it. (The molecule is read inside the test to keep it out
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


def test_back_forth_shuffled_start_guess():
    """A ``start_guess`` whose index is not sorted must give the same answer.

    Nothing else passes one. The loop sorts it once, caches the reindexed ``coord_arr``
    arrays against that order, and relies on every later operation preserving it.
    """
    idx = get_primitives_idx(molecule4, molecule4)
    q = molecule4.get_ric(internal_coords_idx=idx)
    weights = weight_vector(idx)

    shuffled = molecule4.loc[np.random.RandomState(42).permutation(molecule4.index)]
    assert not (shuffled.index == sorted(shuffled.index)).all()

    from_shuffled = q.get_cartesian(start_guess=shuffled)
    from_sorted = q.get_cartesian(start_guess=molecule4.sort_index())

    assert allclose(from_shuffled, molecule4, align=True)
    assert weighted_residual(q, from_shuffled, idx, weights) <= ROUND_TRIP_RESIDUAL
    assert allclose(from_shuffled, from_sorted, align=True)


def test_back_forth_singular_normal_equations():
    """Peroxide is the case that makes the direct sparse factorisation fail.

    ``_sparse_lstsq`` solves through the normal equations ``AᵀA x = Aᵀb``, and for a
    molecule this small ``AᵀA`` is singular enough that SuperLU hits an *exactly* zero
    pivot and raises. H-O-O-H has 12 cartesian degrees of freedom against 6 primitives
    (3 bonds, 2 angles, 1 dihedral), so the rigid-body null space is half the solve
    space. That fraction is ``6 / 3N``, i.e. it shrinks as ``2 / N``, which is why only
    the smallest molecules reach it. Whether the pivot comes out *exactly* zero is
    geometry dependent even at this size -- ammonia has the same 50% and factorises
    fine -- so the ``lsmr`` fallback cannot be replaced by a size check.

    Without that fallback, this raises ``RuntimeError: Factor is exactly singular``.
    """
    idx = get_primitives_idx(molecule3, molecule3)
    q = molecule3.get_ric(internal_coords_idx=idx)
    perturbed = molecule3 + np.random.default_rng(1).normal(
        0, 0.05, (len(molecule3), 3)
    )

    out = q.get_cartesian(start_guess=perturbed, opt_alg="LM")

    assert allclose(out, molecule3, align=True)
    assert weighted_residual(q, out, idx, weight_vector(idx)) <= ROUND_TRIP_RESIDUAL


@pytest.mark.parametrize("molecule", [molecule1, molecule4])
def test_back_forth_gauss(molecule):
    """``opt_alg="gauss"`` is public API and was otherwise never exercised.

    Unlike the Levenberg-Marquardt path it adds no damping, so its ``AᵀA`` is
    ``Bᵀ W² B`` -- singular by the six rigid-body motions at *every* iteration. That is
    harmless, because the right-hand side ``Bᵀ W² Δq`` lies in the row space of ``B``
    and is therefore orthogonal to that null space: the system stays consistent, and
    two solutions differ only by a rigid-body motion, which changes no internal
    coordinate and is removed by the per-iteration superposition anyway.
    """
    idx = get_primitives_idx(molecule, molecule)
    q = molecule.get_ric(internal_coords_idx=idx)
    perturbed = molecule + np.random.default_rng(2).normal(0, 0.05, (len(molecule), 3))

    out = q.get_cartesian(start_guess=perturbed, opt_alg="gauss")

    assert allclose(out, molecule, align=True)
    assert weighted_residual(q, out, idx, weight_vector(idx)) <= ROUND_TRIP_RESIDUAL


@pytest.mark.xfail(
    strict=True,
    reason="undamped Gauss-Newton stalls on the 180 deg "
    "dihedral branch; see the docstring",
)
def test_back_forth_gauss_on_a_flat_dihedral():
    """Known limitation: ``opt_alg="gauss"`` does not converge on planar peroxide.

    The committed geometry is planar, so its dihedral is exactly 180 deg -- sitting on
    the 2*pi branch that ``minimize_dihedral`` wraps at, where the coordinate is at its
    worst conditioned. Levenberg-Marquardt converges for every perturbation below; the
    undamped Gauss-Newton step has nothing to regularise it and stalls, sometimes
    *worse* for a smaller perturbation, which is the signature of the outer loop
    stopping because the structure stopped moving rather than because it found the
    minimum.

    Several perturbations, because which of them stalls is decided at the 1e-15 level:
    reordering the primitives is enough to flip any single one. Asserting that *all*
    converge keeps the xfail stable while still reporting a genuine fix.

    Strict, so that fixing it is noticed rather than silently absorbed. The cause is
    the step control, not the sparse solver.
    """
    idx = get_primitives_idx(molecule3, molecule3)
    q = molecule3.get_ric(internal_coords_idx=idx)
    weights = weight_vector(idx)

    for sigma in (0.01, 0.05, 0.1):
        perturbed = molecule3 + np.random.default_rng(1).normal(
            0, sigma, (len(molecule3), 3)
        )
        out = q.get_cartesian(start_guess=perturbed, opt_alg="gauss")
        assert weighted_residual(q, out, idx, weights) <= ROUND_TRIP_RESIDUAL


def test_get_ric_is_independent_of_row_order():
    """``q[i]`` must belong to ``primitives_idx[i]`` whatever order the rows are in.

    The row order of ``q`` comes from ``_reindex_to_0``, while ``primitives_idx`` keeps
    the order the caller passed. Ordering the former by the reindexed labels makes the
    two agree only for a frame whose rows are already sorted, and silently attaches
    every value to the wrong coordinate for any other -- including the frames that
    ``interpolate(..., coord="zmat")`` returns, which are the default RIC seeds.
    """
    idx = get_primitives_idx(molecule3, molecule3)
    reference = molecule3.get_ric(internal_coords_idx=idx)

    for seed in (0, 1, 2):
        shuffled = molecule3.loc[
            np.random.RandomState(seed).permutation(molecule3.index)
        ]
        assert list(shuffled.index) != list(molecule3.index)

        q = shuffled.get_ric(internal_coords_idx=idx)

        assert list(q.primitives_idx) == list(reference.primitives_idx)
        assert np.allclose(q.q, reference.q, atol=1e-10)


def test_get_ric_on_a_zmat_interpolated_frame():
    """The zmat interpolation orders its rows by the construction table, not by label.

    Those frames are what ``_get_start_guess`` hands the back-transformation, so a
    row-order dependence in ``get_ric`` is reachable through the public interpolation.
    """
    seed = interpolate(molecule3, molecule3, 3, coord="zmat")[0]
    assert list(seed.index) != sorted(seed.index)
    assert allclose(seed, molecule3, align=True, atol=1e-8)

    idx = get_primitives_idx(molecule3, molecule3)

    assert np.allclose(
        seed.get_ric(internal_coords_idx=idx).q,
        molecule3.get_ric(internal_coords_idx=idx).q,
        atol=1e-8,
    )


def test_interpolate_between_near_mirror_images():
    """The path between two near mirror images must not jump.

    MeOH/Furan differs between these two structures almost only in the dihedrals, which
    flip sign. ``minimize_dihedral`` resolves each of them to the shortest arc
    independently, and for one coordinate that is the wrong way round: the interpolated
    targets are then unrealisable, the images stick near whichever endpoint they came
    from, and the path steps 2.5 A in the middle while every other step is 0.15 A.
    Refining does not help -- the jump is the same at N = 11, 21 and 41 -- because the
    target path itself is discontinuous.
    """
    start = Cartesian.read_xyz(get_complete_path("MeOH_Furan_start.xyz"))
    end = Cartesian.read_xyz(get_complete_path("MeOH_Furan_end.xyz"))
    N = 11

    path = RIC_interpolate(start, end, N, schedule="independent")

    steps = [
        np.linalg.norm(
            (b - a).loc[:, ["x", "y", "z"]].values,
            axis=1,
        ).max()
        for a, b in (x.align(y) for x, y in zip(path[:-1], path[1:]))
    ]
    assert max(steps) <= 2 * np.median(steps)
    assert allclose(path[-1], end, align=True, atol=1e-3)
