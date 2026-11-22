import os

import numpy as np
import pandas as pd
import pytest
from sortedcontainers import SortedSet

from chemcoord import Cartesian
from chemcoord._cartesian_coordinates._cartesian_class_bmat import (
    _jit_get_Wilson_B_coo,
    _jit_x_to_ric,
    _SetOfPrimitives,
)
from chemcoord.exceptions import (
    ERR_CODE_OK,
    CoincidingAtoms,
    ERR_CODE_CoincidingAtoms,
    ERR_CODE_UndefinedDihedral,
    UndefinedDihedral,
)


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


def test_primitive_coordinates():
    coords = molecule1.get_primitives_idx()
    expected = SortedSet(
        [
            (0, 1),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 2),
            (1, 8),
            (1, 9),
            (2, 3),
            (2, 10),
            (2, 11),
            (3, 4),
            (3, 12),
            (3, 13),
            (4, 5),
            (4, 14),
            (4, 15),
            (5, 16),
            (5, 17),
            (0, 1, 2),
            (0, 1, 8),
            (0, 1, 9),
            (0, 5, 4),
            (0, 5, 16),
            (0, 5, 17),
            (1, 0, 5),
            (1, 0, 6),
            (1, 0, 7),
            (1, 2, 3),
            (1, 2, 10),
            (1, 2, 11),
            (2, 1, 8),
            (2, 1, 9),
            (2, 3, 4),
            (2, 3, 12),
            (2, 3, 13),
            (3, 2, 10),
            (3, 2, 11),
            (3, 4, 5),
            (3, 4, 14),
            (3, 4, 15),
            (4, 3, 12),
            (4, 3, 13),
            (4, 5, 16),
            (4, 5, 17),
            (5, 0, 6),
            (5, 0, 7),
            (5, 4, 14),
            (5, 4, 15),
            (6, 0, 7),
            (8, 1, 9),
            (10, 2, 11),
            (12, 3, 13),
            (14, 4, 15),
            (16, 5, 17),
            (0, 1, 2, 3),
            (0, 1, 2, 10),
            (0, 1, 2, 11),
            (0, 5, 4, 3),
            (0, 5, 4, 14),
            (0, 5, 4, 15),
            (1, 0, 5, 4),
            (1, 0, 5, 16),
            (1, 0, 5, 17),
            (1, 2, 3, 4),
            (1, 2, 3, 12),
            (1, 2, 3, 13),
            (2, 1, 0, 5),
            (2, 1, 0, 6),
            (2, 1, 0, 7),
            (2, 3, 4, 5),
            (2, 3, 4, 14),
            (2, 3, 4, 15),
            (3, 2, 1, 8),
            (3, 2, 1, 9),
            (3, 4, 5, 16),
            (3, 4, 5, 17),
            (4, 3, 2, 10),
            (4, 3, 2, 11),
            (4, 5, 0, 6),
            (4, 5, 0, 7),
            (5, 0, 1, 8),
            (5, 0, 1, 9),
            (5, 4, 3, 12),
            (5, 4, 3, 13),
            (6, 0, 1, 8),
            (6, 0, 1, 9),
            (6, 0, 5, 16),
            (6, 0, 5, 17),
            (7, 0, 1, 8),
            (7, 0, 1, 9),
            (7, 0, 5, 16),
            (7, 0, 5, 17),
            (8, 1, 2, 10),
            (8, 1, 2, 11),
            (9, 1, 2, 10),
            (9, 1, 2, 11),
            (10, 2, 3, 12),
            (10, 2, 3, 13),
            (11, 2, 3, 12),
            (11, 2, 3, 13),
            (12, 3, 4, 14),
            (12, 3, 4, 15),
            (13, 3, 4, 14),
            (13, 3, 4, 15),
            (14, 4, 5, 16),
            (14, 4, 5, 17),
            (15, 4, 5, 16),
            (15, 4, 5, 17),
        ]
    )
    assert len(expected.symmetric_difference(coords)) == 0


def test_Wilson_B():
    B_mat = molecule3.get_Wilson_B()
    expected = np.array(
        [
            [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                -7.07106781e-01,
                7.07106781e-01,
                0.0,
                7.07106781e-01,
                -7.07106781e-01,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -5.0e-01, -1.5, 0.0, 5.0e-01, 5.0e-01, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -5.0e-01, -5.0e-01, 0.0, 5.0e-01, 1.5, 0.0, 0.0, -1.0, 0.0],
            # dihedral row, antisymmetric about the central bond; matches finite
            # differences to 2e-10
            [
                0.0,
                0.0,
                1.41421356,
                0.0,
                0.0,
                -1.41421356,
                0.0,
                0.0,
                -1.41421356,
                0.0,
                0.0,
                1.41421356,
            ],
        ]
    )
    assert np.allclose(B_mat, expected)


def test_transformation():
    qs = molecule1.get_ric()
    expected = np.array(
        [
            1.5356843,
            1.53568994,
            1.11107867,
            1.11249443,
            1.5356865,
            1.11108307,
            1.11248366,
            1.53568177,
            1.11248232,
            1.11108774,
            1.53568433,
            1.1110901,
            1.11248459,
            1.5356865,
            1.11248368,
            1.11109124,
            1.11108774,
            1.11248201,
            1.92568934,
            1.91466761,
            1.90982223,
            1.92568508,
            1.91466708,
            1.90981874,
            1.92568046,
            1.91468135,
            1.90981809,
            1.92568132,
            1.90982257,
            1.91467242,
            1.91467235,
            1.90982424,
            1.92568807,
            1.91467136,
            1.90982652,
            1.90983294,
            1.91466896,
            1.92568361,
            1.90983146,
            1.91467643,
            1.9146688,
            1.90982691,
            1.91467242,
            1.90982597,
            1.91467406,
            1.90982145,
            1.90981968,
            1.91466224,
            1.88875525,
            1.88875473,
            1.88875245,
            1.8887488,
            1.88875727,
            1.88876143,
            1.00920743,
            -1.09301968,
            3.12054727,
            -1.00920407,
            1.09302079,
            -3.12054809,
            1.00920213,
            3.12054523,
            -1.09301626,
            -1.00920676,
            -3.12055004,
            1.09301962,
            -1.00920465,
            -3.12055606,
            1.09300484,
            1.00920811,
            -1.09300969,
            3.12054366,
            3.12055069,
            -1.09301251,
            -3.12054398,
            1.09300999,
            1.09301412,
            -3.12054867,
            3.1205579,
            -1.09300534,
            -3.12055074,
            1.09301651,
            3.12055292,
            -1.09301803,
            1.05128316,
            -1.01833491,
            -1.05128432,
            1.0183395,
            -1.01834125,
            -3.08795931,
            1.01833775,
            3.08796157,
            1.01832359,
            -1.05129477,
            3.08794569,
            1.01832733,
            -1.01832916,
            -3.08794481,
            1.05129336,
            -1.01832229,
            1.01833512,
            -1.05129683,
            3.08794948,
            1.01831752,
            -1.01831912,
            -3.08795046,
            1.05129731,
            -1.01833403,
        ]
    )
    assert np.allclose(qs.q, expected)


def _four_atoms(angle_012: float) -> Cartesian:
    """Four atoms with a bond angle ``angle_012`` (degrees) for atoms 10-11-12.

    The non-contiguous index checks that the offending coordinate is reported in the
    index of the molecule.
    """
    th = np.radians(180 - angle_012)
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [1.4 + 1.4 * np.cos(th), 1.4 * np.sin(th), 0.0],
            [3.3, 1.0, 0.5],
        ]
    )
    return Cartesian(
        pd.DataFrame(
            {"atom": list("CCCH"), "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]},
            index=[10, 11, 12, 13],
        )
    )


_FOUR_ATOM_PRIMITIVES = _SetOfPrimitives(
    [(10, 11), (11, 12), (12, 13), (10, 11, 12), (11, 12, 13), (10, 11, 12, 13)]
)


@pytest.mark.parametrize("angle", [180.0, 179.9, 0.0, 0.1])
def test_undefined_dihedral_raised_consistently(angle):
    """Both B matrix and ``get_ric`` raise ``UndefinedDihedral`` for (anti)collinear
    atoms, and they agree on the threshold."""
    molecule = _four_atoms(angle)
    for method in (molecule.get_sparse_Wilson_B, molecule.get_ric):
        with pytest.raises(UndefinedDihedral) as excinfo:
            method(_FOUR_ATOM_PRIMITIVES)
        assert excinfo.value.bad_idxs == [(10, 11, 12, 13)]


@pytest.mark.parametrize("angle", [179.5, 0.5])
def test_nearly_collinear_dihedral_is_defined(angle):
    molecule = _four_atoms(angle)
    assert np.isfinite(molecule.get_Wilson_B(_FOUR_ATOM_PRIMITIVES)).all()
    assert np.isfinite(molecule.get_ric(_FOUR_ATOM_PRIMITIVES).q).all()


def test_kernels_return_err_codes_instead_of_raising():
    """The parallel kernels must not raise: numba does not reliably propagate
    exceptions out of ``prange`` loops (https://github.com/numba/numba/issues/9673)."""
    molecule = _four_atoms(180.0)
    positions = molecule.loc[:, ["x", "y", "z"]].values
    expected = [ERR_CODE_OK] * 5 + [ERR_CODE_UndefinedDihedral]

    data, row, _, err_codes = _jit_get_Wilson_B_coo(
        positions, molecule._to_array_nobending(_FOUR_ATOM_PRIMITIVES)
    )
    assert err_codes.tolist() == expected
    assert np.isnan(data[5]).all()
    assert np.isfinite(data[:5][row[:5] >= 0]).all()

    q, err_codes = _jit_x_to_ric(
        positions, molecule._to_array_full(_FOUR_ATOM_PRIMITIVES)
    )
    assert err_codes.tolist() == expected
    assert np.isnan(q[5]) and np.isfinite(q[:5]).all()


def _coinciding_atoms() -> Cartesian:
    """Like :func:`_four_atoms`, but atom 13 sits on top of atom 12."""
    molecule = _four_atoms(110.0)
    molecule.loc[13, ["x", "y", "z"]] = molecule.loc[12, ["x", "y", "z"]].values
    return molecule


def test_coinciding_atoms_raised_consistently():
    """Every coordinate that contains the coinciding bonded atoms 12 and 13 is
    reported, by both B matrix and ``get_ric``, and coinciding atoms take precedence
    over undefined dihedrals."""
    molecule = _coinciding_atoms()
    expected = [(12, 13), (11, 12, 13), (10, 11, 12, 13)]
    for method in (molecule.get_sparse_Wilson_B, molecule.get_ric):
        with pytest.raises(CoincidingAtoms) as excinfo:
            method(_FOUR_ATOM_PRIMITIVES)
        assert excinfo.value.bad_idxs == expected


def test_kernels_flag_coinciding_atoms():
    molecule = _coinciding_atoms()
    positions = molecule.loc[:, ["x", "y", "z"]].values
    expected = [ERR_CODE_OK] * 2 + [ERR_CODE_CoincidingAtoms]
    expected += [ERR_CODE_OK] + [ERR_CODE_CoincidingAtoms] * 2

    data, row, _, err_codes = _jit_get_Wilson_B_coo(
        positions, molecule._to_array_nobending(_FOUR_ATOM_PRIMITIVES)
    )
    assert err_codes.tolist() == expected
    is_bad = np.array(expected) != ERR_CODE_OK
    assert np.isnan(data[is_bad][row[is_bad] >= 0]).all()
    assert np.isfinite(data[~is_bad][row[~is_bad] >= 0]).all()

    q, err_codes = _jit_x_to_ric(
        positions, molecule._to_array_full(_FOUR_ATOM_PRIMITIVES)
    )
    assert err_codes.tolist() == expected
    assert np.isnan(q[is_bad]).all() and np.isfinite(q[~is_bad]).all()
