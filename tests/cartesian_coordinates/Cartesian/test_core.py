import os
from itertools import combinations

import numpy as np
import pytest
import sympy

import chemcoord as cc
from chemcoord._cartesian_coordinates.xyz_functions import get_rotation_matrix
from chemcoord.exceptions import PhysicalMeaning
from chemcoord.xyz_functions import allclose


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
    # ``eigenvectors`` contains the principal axes as columns, i.e. it is the basis of
    # the principal-axis frame expressed in the old basis. Transforming into that frame
    # is therefore done with the transpose (compare :meth:`Cartesian.basistransform`).
    assert np.allclose(eig.T @ eig, np.identity(3))
    assert np.allclose(
        eig.T @ A["inertia_tensor"] @ eig, np.diag(A["diag_inertia_tensor"])
    )
    assert cc.xyz_functions.allclose(
        eig.T @ (molecule - molecule.get_barycenter()), t_mol
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


def _mst_length_bruteforce(molecule, bond_dict=None):
    """Total length of the Euclidean minimum spanning tree over the fragments.

    Reference implementation for :meth:`Cartesian._fragment_connecting_bonds`:
    the complete graph over all fragment pairs, weighted by the shortest
    inter-fragment atom distance, run through Kruskal. Quadratic in the number of
    fragments, which is exactly what the tested method avoids.
    """
    fragments = molecule.fragmentate(give_only_index=True, bond_dict=bond_dict)
    pos = molecule.loc[:, ["x", "y", "z"]].values
    row_of = {label: row for row, label in enumerate(molecule.index)}
    rows = [[row_of[label] for label in fragment] for fragment in fragments]

    edges = sorted(
        (
            np.linalg.norm(
                pos[rows[a]][:, None, :] - pos[rows[b]][None, :, :], axis=-1
            ).min(),
            a,
            b,
        )
        for a, b in combinations(range(len(rows)), 2)
    )

    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0.0
    for distance, a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b
            total += distance
    return total


@pytest.mark.parametrize(
    "structure",
    [
        # two fragments, the trivial case
        "MeOH_Furan_start.xyz",
        # 53 fragments over 56 atoms, i.e. many more fragments than the F - 1
        # bonds that may be kept
        "Cd_lattice.xyz",
    ],
)
def test_fragment_connecting_bonds(structure):
    m = cc.Cartesian.read_xyz(get_complete_path(structure), start_index=1)
    fragments = m.fragmentate(give_only_index=True)
    assert len(fragments) > 1

    bonds = m._fragment_connecting_bonds()

    # A spanning tree over the fragments, addressed by the molecule's own labels
    # (1-based here, not row numbers).
    assert len(bonds) == len(fragments) - 1
    assert set(m.index).issuperset(i for bond in bonds for i in bond)

    # Adding them makes the molecule a single connected component.
    bond_dict = {i: set(connected) for i, connected in m.get_bonds().items()}
    for i, j in bonds:
        bond_dict[i].add(j)
        bond_dict[j].add(i)
    assert len(m.fragmentate(give_only_index=True, bond_dict=bond_dict)) == 1

    # ...and a *minimum* one. Compared by total length rather than by edge set,
    # which is not unique when distances tie.
    pos = m.loc[:, ["x", "y", "z"]]
    length = sum(np.linalg.norm(pos.loc[i] - pos.loc[j]) for i, j in bonds)
    assert np.isclose(length, _mst_length_bruteforce(m))


def test_fragment_connecting_bonds_single_fragment():
    assert len(molecule.fragmentate(give_only_index=True)) == 1
    assert molecule._fragment_connecting_bonds() == []


def test_fragment_connecting_bonds_uses_given_bond_dict():
    # An empty connectivity makes every atom its own fragment, so the result is the
    # Euclidean MST over all 56 atoms. If the argument were ignored and ``get_bonds``
    # recomputed instead, ``molecule`` would be a single fragment and the result empty.
    no_bonds = {i: set() for i in molecule.index}

    bonds = molecule._fragment_connecting_bonds(no_bonds)

    assert len(bonds) == len(molecule) - 1
    pos = molecule.loc[:, ["x", "y", "z"]]
    length = sum(np.linalg.norm(pos.loc[i] - pos.loc[j]) for i, j in bonds)
    assert np.isclose(length, _mst_length_bruteforce(molecule, bond_dict=no_bonds))


def test_fragment_connecting_bonds_grows_search_radius():
    # Two copies of the same molecule, far enough apart that no atom has an atom of
    # the *other* copy among its nearest neighbours. The initial candidate
    # neighbourhood (``k = 5``) then yields no inter-fragment edge at all and has to
    # be grown until the two fragments are spanned.
    base = cc.Cartesian.read_xyz(get_complete_path("cis_platin.xyz"), start_index=1)
    far = base + np.array([50.0, 0.0, 0.0])
    far.index = far.index + len(base)
    pair = cc.xyz_functions.concat([base, far])
    assert len(pair.fragmentate(give_only_index=True)) == 2

    bonds = pair._fragment_connecting_bonds()

    # the single bond joins the closest pair of atoms across the two copies
    assert len(bonds) == 1
    ((i, j),) = bonds
    pos = pair.loc[:, ["x", "y", "z"]]
    assert np.isclose(
        np.linalg.norm(pos.loc[i] - pos.loc[j]), _mst_length_bruteforce(pair)
    )
