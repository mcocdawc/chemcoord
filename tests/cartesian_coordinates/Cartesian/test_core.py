import os

import numpy as np
import pandas as pd
import pytest
import sympy

import chemcoord as cc
from chemcoord._cartesian_coordinates.xyz_functions import get_rotation_matrix
from chemcoord.exceptions import PhysicalMeaning
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


STRUCTURE_PATH = get_structure_path(get_script_path())


def get_complete_path(structure):
    return os.path.join(STRUCTURE_PATH, structure)


molecule = cc.Cartesian.read_xyz(get_complete_path("MIL53_small.xyz"), start_index=1)
bond_dict = {
    1: {2, 51},
    2: {1, 9, 27},
    3: {6, 55, 56},
    4: {5, 52},
    5: {4, 15, 31},
    6: {3, 7, 8, 9, 15, 16},
    7: {6, 11, 53},
    8: {6, 10},
    9: {2, 6},
    10: {8, 12, 24},
    11: {7, 12, 13, 18, 19, 20},
    12: {10, 11},
    13: {11, 14},
    14: {13, 22, 33},
    15: {5, 6},
    16: {6, 17},
    17: {16, 20, 32},
    18: {11, 48, 54},
    19: {11, 21},
    20: {11, 17},
    21: {19, 23, 34},
    22: {14, 49},
    23: {21, 50},
    24: {10, 25, 26, 36},
    25: {24},
    26: {24},
    27: {2, 28, 29, 30},
    28: {27},
    29: {27},
    30: {27},
    31: {5, 41, 44, 47},
    32: {17, 35, 37, 38},
    33: {14, 40, 43, 46},
    34: {21, 39, 42, 45},
    35: {32},
    36: {24},
    37: {32},
    38: {32},
    39: {34},
    40: {33},
    41: {31},
    42: {34},
    43: {33},
    44: {31},
    45: {34},
    46: {33},
    47: {31},
    48: {18},
    49: {22},
    50: {23},
    51: {1},
    52: {4},
    53: {7},
    54: {18},
    55: {3},
    56: {3},
}


def test_init():
    with pytest.raises(TypeError):
        cc.Cartesian(5)
    with pytest.raises(PhysicalMeaning):
        cc.Cartesian(molecule.loc[:, ["atom", "x"]])


def test_overloaded_operators():
    assert allclose(molecule + 1, molecule + [1, 1, 1])
    assert allclose(1 + molecule, molecule + [1, 1, 1])
    assert allclose(molecule + molecule, 2 * molecule)
    index = molecule.index
    assert allclose(molecule + molecule.loc[reversed(index)], 2 * molecule)
    assert allclose(molecule + molecule.loc[:, ["x", "y", "z"]].values, 2 * molecule)
    assert allclose(1 * molecule, molecule)
    assert allclose(molecule * 1, molecule)
    assert allclose(1 * molecule, +molecule)
    assert allclose(-1 * molecule, -molecule)
    assert allclose(-molecule, 0 - molecule)
    assert allclose(molecule, molecule - 0)
    molecule2 = molecule[
        ~(
            np.isclose(molecule["x"], 0)
            | np.isclose(molecule["y"], 0)
            | np.isclose(molecule["z"], 0)
        )
    ]
    assert np.allclose(
        np.full(molecule2.loc[:, ["x", "y", "z"]].shape, 1),
        (molecule2 / molecule2).loc[:, ["x", "y", "z"]],
    )
    assert np.allclose(
        np.full(molecule2.loc[:, ["x", "y", "z"]].shape, 0),
        (molecule2 - molecule2).loc[:, ["x", "y", "z"]],
    )


def test_indexing():
    assert (molecule.x == molecule.loc[:, "x"]).all()
    assert (molecule.y == molecule.loc[:, "y"]).all()
    assert (molecule.z == molecule.loc[:, "z"]).all()
    assert ((molecule.x - molecule.loc[:, "x"]) == 0).all()


def test_assignment():
    molecule = cc.Cartesian.read_xyz(
        get_complete_path("MIL53_small.xyz"), start_index=1
    )
    x = sympy.symbols("x", real=True)

    molecule.loc[:, "x"] = 3
    assert (molecule.x == 3).all()

    molecule.loc[1, "x"] = x
    assert molecule.x.dtypes == np.dtype("O")
    assert molecule.y.dtypes == np.dtype("f8")
    molecule = molecule.subs(x, 1)
    assert molecule.x.dtypes == np.dtype("f8")

    molecule.loc[1, ["x", "y"]] = x
    assert molecule.x.dtypes == np.dtype("O")
    assert molecule.y.dtypes == np.dtype("O")
    molecule = molecule.subs(x, 1)
    assert molecule.x.dtypes == np.dtype("f8")
    assert molecule.y.dtypes == np.dtype("f8")

    molecule = cc.Cartesian.read_xyz(
        get_complete_path("MIL53_small.xyz"), start_index=1
    )
    x = sympy.symbols("x", real=True)

    molecule.iloc[:, 1] = 3
    assert (molecule.x == 3).all()

    molecule.iloc[1, 1] = x
    assert molecule.x.dtypes == np.dtype("O")
    assert molecule.y.dtypes == np.dtype("f8")
    molecule = molecule.subs(x, 1)
    assert molecule.x.dtypes == np.dtype("f8")

    molecule.iloc[1, [1, 2]] = x
    assert molecule.x.dtypes == np.dtype("O")
    assert molecule.y.dtypes == np.dtype("O")
    molecule = molecule.subs(x, 1)
    assert molecule.x.dtypes == np.dtype("f8")
    assert molecule.y.dtypes == np.dtype("f8")


def test_get_bonds():
    assert bond_dict == molecule.get_bonds()
    modified_expected = {
        1: {51},
        3: {6, 55, 56},
        4: {52},
        6: {3, 7, 8, 9, 15, 16},
        7: {6, 11, 53},
        8: {6},
        9: {6},
        11: {7, 12, 13, 18, 19, 20},
        12: {11},
        13: {11},
        15: {6},
        16: {6},
        18: {11, 48, 54},
        19: {11},
        20: {11},
        22: {49},
        23: {50},
        48: {18},
        49: {22},
        50: {23},
        51: {1},
        52: {4},
        53: {7},
        54: {18},
        55: {3},
        56: {3},
        2: set(),
        5: set(),
        10: set(),
        14: set(),
        17: set(),
        21: set(),
        24: set(),
        25: set(),
        26: set(),
        27: set(),
        28: set(),
        29: set(),
        30: set(),
        31: set(),
        32: set(),
        33: set(),
        34: set(),
        35: set(),
        36: set(),
        37: set(),
        38: set(),
        39: set(),
        40: set(),
        41: set(),
        42: set(),
        43: set(),
        44: set(),
        45: set(),
        46: set(),
        47: set(),
    }
    assert (
        molecule.get_bonds(
            modify_atom_data={k: 0.0 for k in molecule[molecule.atom == "C"].index}
        )
        == molecule.get_bonds(modify_element_data={"C": 0.0})
        == modified_expected
    )

    bonds = molecule.get_bonds(
        modify_element_data=lambda r: r * 2000, self_bonding_allowed=True
    )
    for k, v in bonds.items():
        assert v == set(molecule.index)


def test_coordination_sphere():
    expctd = {}
    expctd[1] = {6, 11, 53}
    expctd[2] = {3, 8, 9, 12, 13, 15, 16, 18, 19, 20}
    expctd[3] = {2, 5, 10, 14, 17, 21, 48, 54, 55, 56}
    expctd[4] = {1, 4, 22, 23, 24, 27, 31, 32, 33, 34}
    expctd[5] = {
        25,
        26,
        28,
        29,
        30,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        49,
        50,
        51,
        52,
    }
    expctd[6] = set()

    for n_sphere, connected_atoms in expctd.items():
        assert connected_atoms == set(
            molecule.get_coordination_sphere(7, n_sphere=n_sphere).index
        )


def test_cut_sphere():
    expected = {6, 7, 8, 9, 11, 12, 13, 15, 16, 19, 20, 53}
    assert expected == set(molecule.cut_sphere(radius=3, origin=7).index)
    assert (
        molecule == molecule.cut_sphere(radius=3, origin=7, preserve_bonds=True)
    ).all(axis=None)
    assert set(molecule.index) - expected == set(
        molecule.cut_sphere(radius=3, origin=7, outside_sliced=False).index
    )


def test_cut_cuboid():
    expected = {3, 4, 5, 6, 7, 15, 16, 17, 32, 35, 37, 38, 47, 52, 53, 55, 56}
    assert expected == set(molecule.cut_cuboid(a=2, origin=7).index)
    assert (molecule == molecule.cut_cuboid(a=2, origin=7, preserve_bonds=True)).all(
        axis=None
    )
    assert set(molecule.index) - expected == set(
        molecule.cut_cuboid(a=2, origin=7, outside_sliced=False).index
    )


def test_get_inertia():
    A = molecule.get_inertia()
    eig, t_mol = A["eigenvectors"], A["transformed_Cartesian"]
    assert cc.xyz_functions.allclose(
        eig @ (molecule - molecule.get_barycenter()), t_mol
    )

    molecule2 = get_rotation_matrix([1, 1, 1], 72) @ molecule
    B = molecule2.get_inertia()
    assert cc.xyz_functions.allclose(B["transformed_Cartesian"], t_mol)


def test_partition_chem_env():
    xpctd = {
        ("C", frozenset({("C", 4), ("Cr", 2), ("H", 7), ("O", 7)})): {2, 5, 14, 21},
        ("C", frozenset({("C", 6), ("Cr", 2), ("H", 8), ("O", 11)})): {10, 17},
        ("C", frozenset({("C", 1), ("Cr", 2), ("H", 3), ("O", 11)})): {24, 32},
        ("C", frozenset({("C", 1), ("Cr", 1), ("H", 4), ("O", 7)})): {27, 31, 33, 34},
        ("Cr", frozenset({("C", 10), ("Cr", 1), ("H", 19), ("O", 13)})): {6, 11},
        ("H", frozenset({("C", 2), ("Cr", 2), ("H", 2), ("O", 2)})): {
            25,
            26,
            35,
            36,
            37,
            38,
        },
        ("H", frozenset({("C", 2), ("Cr", 1), ("H", 3), ("O", 2)})): {
            28,
            29,
            30,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            49,
            50,
            51,
            52,
        },
        ("H", frozenset({("C", 4), ("Cr", 2), ("H", 2), ("O", 6)})): {48, 54, 55, 56},
        ("H", frozenset({("C", 6), ("Cr", 2), ("H", 4), ("O", 11)})): {53},
        ("O", frozenset({("C", 2), ("Cr", 1), ("H", 4), ("O", 6)})): {1, 4, 22, 23},
        ("O", frozenset({("C", 8), ("Cr", 2), ("H", 3), ("O", 12)})): {3, 18},
        ("O", frozenset({("C", 12), ("Cr", 2), ("H", 5), ("O", 14)})): {7},
        ("O", frozenset({("C", 8), ("Cr", 2), ("H", 6), ("O", 12)})): {8, 12, 16, 20},
        ("O", frozenset({("C", 8), ("Cr", 2), ("H", 7), ("O", 12)})): {9, 13, 15, 19},
    }
    assert xpctd == molecule.partition_chem_env()


def test_change_numbering():
    molecule2 = molecule.copy()
    molecule2.index = reversed(molecule.index)
    dct = dict(zip(molecule.index, reversed(molecule.index)))
    assert (molecule2.index == molecule.change_numbering(dct).index).all()


def test_align():
    cartesians = cc.xyz_functions.read_molden(
        get_complete_path("total_movement.molden"), start_index=1
    )
    m1, m2 = cartesians[0], cartesians[-1]
    m2 = get_rotation_matrix([1, 1, 1], 0.334) @ m2 + 5
    m1, m2_aligned = m1.align(m2)
    dev = abs((m2_aligned - m1).loc[:, ["x", "y", "z"]]).sum() / len(m1)
    assert np.allclose(dev, [0.73398451, 1.61863496, 0.13181807])

    assert cc.xyz_functions.allclose(m1.align(m2_aligned)[1], m2_aligned)


def test_mass_align():
    cartesians = cc.xyz_functions.read_molden(
        get_complete_path("total_movement.molden"), start_index=1
    )
    m1, m2 = cartesians[0], cartesians[-1]
    m2 = get_rotation_matrix([1, 1, 1], 0.334) @ m2 + 5
    m1, m2_aligned = m1.align(m2, mass_weight=True)
    assert cc.xyz_functions.allclose(
        m1.align(m2_aligned, mass_weight=True)[1], m2_aligned
    )


def test_align_and_reindex_similar():
    cartesians = cc.xyz_functions.read_molden(
        get_complete_path("total_movement.molden"), start_index=1
    )
    m2 = cartesians[-1]

    m2_shuffled = get_rotation_matrix([1, 1, 1], 0.2) @ m2 + 8
    rng = np.random.RandomState(77)
    m2_shuffled.index = rng.permutation(m2.index)

    m2 = (m2 - m2.get_centroid()).sort_index()
    m2_shuffled = (m2_shuffled - m2_shuffled.get_centroid()).sort_index()

    R = (
        m2.loc[[42, 41, 153, 152], :]
        .reset_index()
        .get_align_transf(m2_shuffled.loc[[87, 115, 24, 208], :].reset_index())
    )

    m2_shuffled = R @ m2_shuffled

    m2_backindexed = m2.reindex_similar(m2_shuffled)
    assert cc.xyz_functions.allclose(m2, m2_backindexed)
