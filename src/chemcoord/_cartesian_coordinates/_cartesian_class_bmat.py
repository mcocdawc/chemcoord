from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange
from numpy import cross, float64, int64
from numpy.linalg import norm

from chemcoord._cartesian_coordinates._cart_transformation import (
    _jit_normalize,
)
from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import COORDS
from chemcoord._redundant_internal_coordinates.main import (
    Primitives,
    SetOfPrimitives,
)
from chemcoord.typing import BondDict, Matrix, Vector

if TYPE_CHECKING:
    from chemcoord import Cartesian
    from chemcoord._redundant_internal_coordinates.main import (
        RedundantInternalCoordinates,
    )


class CartesianBmat(CartesianCore):
    def get_primitives_idx(
        self, bonds: BondDict | None = None, connect_fragments: bool = True
    ) -> Primitives:
        """Generate set of redundant internal coordinates for the system.
        Stored in a sortedcontainers.SortedSet to maintain order while
        being able to use Python's union operator. Sorted by length of
        coordinate, then by standard order based on the atoms' indices.

        Args:
            coordinates (SortedSet[tuple]): default None, SortedSet of primitive
                coordinates to use in the calculation. If None, calculates using the
                get_primitive_coords method
            use_lookup (bool): default False, if True, uses a lookup table for bond
                determination when generating primitive internal coordinates
        Returns:
            SortedSet[tuple]: SortedSet of redundant internal coordinates
        """
        if bonds is None:
            bond_dict = self.get_bonds()
        else:
            # Change to mutable set, i.e. dict[AtomIdx, set[AtomIdx]] instead of
            # non-mutable ``collections.abc.Set`` since we need the ``add`` method.
            bond_dict = {i_atom: set(connected) for i_atom, connected in bonds.items()}

        if connect_fragments:
            fragments = self.fragmentate()
            if len(fragments) != 1:
                for fragment_pair in combinations(fragments, 2):
                    index1, index2, _ = fragment_pair[0].get_shortest_distance(
                        fragment_pair[1]
                    )
                    bond_dict[index1].add(index2)
                    bond_dict[index2].add(index1)
        return self._get_primitives_single_molecule(bond_dict)

    def _get_primitives_single_molecule(
        self, bonds: BondDict | None = None
    ) -> Primitives:
        """This function calculated the primitive internal coordinates
        purely based on chemical connectivity and does not connect fragments."""
        idx_primitive_coords = SetOfPrimitives()

        if bonds is None:
            bonds = self.get_bonds()

        def canonicalize(*args: int) -> tuple[int, ...]:
            if args[0] < args[-1]:
                return tuple(args)
            else:
                return tuple(reversed(args))

        # TODO early returns (purely for  performance)
        for atom1 in self.index:
            for atom2 in bonds[atom1]:
                idx_primitive_coords.add(canonicalize(atom1, atom2))
                for atom3 in bonds[atom2] - {atom1}:
                    idx_primitive_coords.add(canonicalize(atom1, atom2, atom3))
                    for atom4 in bonds[atom3] - {atom1, atom2}:
                        idx_primitive_coords.add(
                            canonicalize(atom1, atom2, atom3, atom4)
                        )

        return idx_primitive_coords

    def get_Wilson_B(
        self,
        idx_internal_coords: Primitives | None = None,
        bonds: BondDict | None = None,
    ) -> Matrix:
        """Generate Wilson's B matrix for the current structure.

        Args:
            internal_coordinates (SortedSet[tuple]): default None, SortedSet of
                primitive internal coordinates to use in the calculation. If None,
                calculates using the get_primitive_coords method
            use_lookup (bool): default False, if True, uses a lookup table for bond
                determination when generating primitive internal coordinates

        Returns:
            NDArray[float64]: Wilson's B matrix
        """
        if idx_internal_coords is None:
            idx_internal_coords = self.get_primitives_idx(bonds)

        return _jit_get_Wilson_B(
            self.loc[:, COORDS].values, self._to_array(idx_internal_coords)
        )

    def get_ric(
        self,
        internal_coords_idx: Primitives | None = None,
        bonds: BondDict | None = None,
    ) -> RedundantInternalCoordinates:
        """Conversion to redundant internal coordinates

        Args:
            internal_coordinates (SortedSet[tuple]): default None, SortedSet of
                primitive coordinates to convert to. If None, calculates them using the
                get_primitive_coords method
            use_lookup (bool): default False, if True, uses a lookup table for bond
                determination when generating primitive internal coordinates

        """
        from chemcoord._redundant_internal_coordinates.main import (
            RedundantInternalCoordinates,
        )

        # get primitive coordinates
        if internal_coords_idx is None:
            internal_coords_idx = self.get_primitives_idx(bonds=bonds)

        return RedundantInternalCoordinates(
            _jit_x_to_ric(
                self.loc[:, COORDS].values, self._to_array(internal_coords_idx)
            ),
            internal_coords_idx,
            self.copy(),  # type: ignore[arg-type]
        )

    def _reindex_to_0(self, internal_coords_idx: Primitives) -> Primitives:
        """Return a reindexed version of `primitives` as if `self` was indexed
        contiguously from 0 to n - 1."""
        index_to_rownum = {index: row for row, index in enumerate(self.index)}

        return SetOfPrimitives(
            {
                tuple(index_to_rownum[index] for index in coordinate_idx)
                for coordinate_idx in internal_coords_idx
            }
        )

    def _to_array(self, internal_coords_idx: Primitives) -> Matrix[int64]:
        """Converts the index of the primitive internal coordinates to an array
        and changes to 0-based indexing.

        The array is a rectangualar (n, 5) array, where n is the number of
        internal coordinates. The last column denotes the number of defined columns and
        the type of coordinate, i.e. (n=2) bond, (n=3) angle, (n=4) dihedral.

        In addition, this function switches from the arbitrary flexible index of a
        molecule to 0-based indexing.
        """
        internal_coord_idx_arr = np.empty((len(internal_coords_idx), 5), dtype=int64)
        for i, coordinate in enumerate(self._reindex_to_0(internal_coords_idx)):
            internal_coord_idx_arr[i, : len(coordinate)] = coordinate
            internal_coord_idx_arr[i, 4] = len(coordinate)
        return internal_coord_idx_arr

    def _fix_undef_dihedrals(
        self, coord_idx: Primitives
    ) -> tuple[Primitives, Primitives]:
        # getting rid of poorly-defined dihedrals because of linearity at the start
        new_coord_idx = coord_idx.copy()
        bad_indices = SetOfPrimitives()
        for i, index in enumerate(coord_idx.copy()):
            if len(index) == 4:
                first_three = self.loc[index[:-1], COORDS].values
                last_three = self.loc[index[1:], COORDS].values

                # vectors making up the angle
                normedu1 = _jit_normalize(first_three[0] - first_three[1])
                normedv1 = _jit_normalize(first_three[2] - first_three[1])

                normedu2 = _jit_normalize(last_three[0] - last_three[1])
                normedv2 = _jit_normalize(last_three[2] - last_three[1])

                angle1 = np.arccos(normedu1 @ normedv1)
                angle2 = np.arccos(normedu2 @ normedv2)

                if not (5 < angle1 * 180 / np.pi < 175):
                    bad_idx = index
                    bad_indices.add(index)
                    new_coord_idx.add(
                        _correct_dihedral_idx(self, bad_idx, coord_idx, 1)  # type: ignore[arg-type]
                    )
                if not (5 < angle2 * 180 / np.pi < 175):
                    bad_idx = index
                    bad_indices.add(index)
                    new_coord_idx.add(
                        _correct_dihedral_idx(self, bad_idx, coord_idx, 2)  # type: ignore[arg-type]
                    )

        return (new_coord_idx, bad_indices)


@njit(cache=True, nogil=True)
def _jit_dihedral_deriv(positions: Matrix) -> Matrix:
    # vectors making up dihedral
    u = positions[0] - positions[1]
    w = positions[2] - positions[1]
    v = positions[3] - positions[2]

    normedu = _jit_normalize(u)
    normedw = _jit_normalize(w)
    normedv = _jit_normalize(v)

    cosu = normedu @ normedw
    # note this could be 0 if undefined dihedral
    sinu = np.sqrt(1 - (normedu @ normedw) ** 2)

    cosv = -(normedv @ normedw)
    # note this could be 0 if undefined dihedral
    sinv = np.sqrt(1 - (normedv @ normedw) ** 2)

    # catching cases where certain dihedrals are undefined
    if np.isclose(sinu, 0.0) or np.isclose(sinv, 0.0):
        raise ValueError("sinu or sinv is 0")
    else:
        return np.stack(
            (
                cross(normedu, normedw) / (norm(u) * (sinu**2)),
                -cross(normedu, normedw) / (norm(u) * (sinu**2))
                + (
                    ((cross(normedu, normedw) * cosu) / (norm(w) * (sinu**2)))
                    - ((cross(normedv, normedw) * cosv) / (norm(w) * (sinv**2)))
                ),
                cross(normedv, normedw) / (norm(v) * (sinv**2))
                - (
                    ((cross(normedu, normedw) * cosu) / (norm(w) * (sinu**2)))
                    - ((cross(normedv, normedw) * cosv) / (norm(w) * (sinv**2)))
                ),
                -cross(normedv, normedw) / (norm(v) * (sinv**2)),
            )
        )


@njit(cache=True, nogil=True)
def _jit_angle_deriv(positions: Matrix) -> Matrix:
    # vectors making up the angle

    u = positions[0] - positions[1]
    v = positions[2] - positions[1]

    normedu = _jit_normalize(u)
    normedv = _jit_normalize(v)

    w = cross(u, v)

    # if they were parallel
    if np.allclose(w, 0.0):
        w = cross(normedu, np.array([1, -1, 1]))
    # if u and [1, -1, 1] were parallel
    if np.allclose(w, 0.0):
        w = cross(normedu, np.array([-1, 1, 1]))

    normedw = _jit_normalize(w)
    A = cross(normedu, normedw) / norm(u)
    B = cross(normedw, normedv) / norm(v)
    return np.stack((A, -(A + B), B))


@njit(cache=True, nogil=True)
def _jit_second_dihedral_deriv(positions: Matrix) -> Matrix:
    # vectors making up dihedral
    u = positions[0] - positions[1]
    w = positions[2] - positions[1]
    v = positions[3] - positions[2]

    normedu = _jit_normalize(u)
    normedw = _jit_normalize(w)
    normedv = _jit_normalize(v)

    cosu = normedu @ normedw
    # note this could be 0 if undefined dihedral
    sinu = np.sqrt(1 - (normedu @ normedw) ** 2)

    cosv = -(normedv @ normedw)
    # note this could be 0 if undefined dihedral
    sinv = np.sqrt(1 - (normedv @ normedw) ** 2)

    # catching cases where certain dihedrals are undefined
    if np.isclose(sinu, 0.0) or np.isclose(sinv, 0.0):
        raise ValueError("sinu or sinv is 0")
    else:
        return np.stack(
            (
                cross(normedu, normedw) / (norm(u) * (sinu**2)),
                -cross(normedu, normedw) / (norm(u) * (sinu**2))
                + (
                    ((cross(normedu, normedw) * cosu) / (norm(w) * (sinu**2)))
                    - ((cross(normedv, normedw) * cosv) / (norm(w) * (sinv**2)))
                ),
                cross(normedv, normedw) / (norm(v) * (sinv**2))
                - (
                    ((cross(normedu, normedw) * cosu) / (norm(w) * (sinu**2)))
                    - ((cross(normedv, normedw) * cosv) / (norm(w) * (sinv**2)))
                ),
                -cross(normedv, normedw) / (norm(v) * (sinv**2)),
            )
        )


@njit(cache=True, nogil=True)
def _jit_second_angle_deriv(positions: Matrix) -> Matrix:
    # vectors making up the angle

    u = positions[0] - positions[1]
    v = positions[2] - positions[1]

    normedu = _jit_normalize(u)
    normedv = _jit_normalize(v)

    w = cross(u, v)

    # if they were parallel
    if np.allclose(w, 0.0):
        w = cross(normedu, np.array([1, -1, 1]))
    # if u and [1, -1, 1] were parallel
    if np.allclose(w, 0.0):
        w = cross(normedu, np.array([-1, 1, 1]))

    normedw = _jit_normalize(w)
    A = cross(normedu, normedw) / norm(u)
    B = cross(normedw, normedv) / norm(v)
    return np.stack((A, -(A + B), B))


@njit(parallel=True, cache=True, nogil=True)
def _jit_get_Wilson_B(
    position_arr: Matrix[float64],
    internal_coord_arr: Matrix[int64],
) -> Matrix[float64]:
    """Jit-compiled Wilson's B matrix generator.

    Args:
        position_arr (Matrix): array of cartesian coordinate locations of the
            atoms in the Cartesian
        internal_coord_arr (Matrix): array of internal coordinates, followed by the
            length of the coordinate. If the coordinate is not a dihedral, there are
            unused numbers in the 4th, or 3rd and 4th, places to ensure a
            rectangular array

    Returns:
        Matrix[float64]: Wilson's B matrix
    """

    # initialize B matrix
    B_matrix = np.zeros((len(internal_coord_arr), position_arr.size))

    for i in prange(len(internal_coord_arr)):  # type: ignore[attr-defined]
        # separate cases for distances, angles, and dihedrals
        # procedure from J. Chem. Phys. 117, 9160 (2002); https://doi.org/10.1063/1.1515483

        # get ith internal coordinate
        coord = internal_coord_arr[i, :]

        # distances
        if coord[-1] == 2:
            # get positions of participating atoms
            positions = position_arr[coord[:2]]

            # derivatives are just components of unit vector along distance
            normedu = _jit_normalize(positions[0] - positions[1])
            # for each cartesian coordinate
            for j in prange(3):  # type: ignore[attr-defined]
                B_matrix[i, j + 3 * coord[0]] = normedu[j]
                B_matrix[i, j + 3 * coord[1]] = -normedu[j]

        # angles
        elif coord[-1] == 3:
            # get positions of participating atoms
            positions = position_arr[coord[:3]]

            angle_derivs = _jit_angle_deriv(positions)

            for j in prange(3):  # type: ignore[attr-defined]
                B_matrix[i, j + 3 * coord[:3]] = angle_derivs[:3, j]

        # dihedrals
        else:
            # get positions of participating atoms
            positions = position_arr[coord[:4]]

            dihedral_derivs = _jit_dihedral_deriv(positions)

            for j in prange(3):  # type: ignore[attr-defined]
                B_matrix[i, j + 3 * coord[:4]] = dihedral_derivs[:4, j]

    return B_matrix


@njit(parallel=True, cache=True, nogil=True)
def _jit_x_to_ric(
    cart_positions: Matrix[float64], internal_coords_idx: Matrix[int64]
) -> Vector[float64]:
    """Jit-compiled conversion between cartesian coordinates and internal coordinates

    .. note:: This function implicitly assumes that `internal_coords_idx`
        is for a 0-indexed molecule.

    Args:
        cart_positions (Matrix): array of cartesian coordinate locations of the
            atoms in the Cartesian
        internal_coords_idx (Matrix): array of internal coordinates, followed by the
            length of the coordinate. If the coordinate is not a dihedral, there are
            unused numbers in the 4th, or 3rd and 4th, places to ensure a
            rectangular array

    Returns:
        Vector[float64]: array of internal coordinate values
    """
    internal_coordinates = np.empty(len(internal_coords_idx))

    for i in range(len(internal_coords_idx)):  # type: ignore[attr-defined]
        # get ith internal coordinate
        coord = internal_coords_idx[i]

        # separate cases for distances, angles, and dihedrals

        # distances
        if coord[-1] == 2:
            # get positions of participating atoms
            positions = cart_positions[coord[:2]]

            u = positions[1] - positions[0]
            internal_coordinates[i] = norm(u)

        # angles
        elif coord[-1] == 3:
            # get positions of participating atoms
            positions = cart_positions[coord[:3]]

            # vectors making up the angle
            normedu = _jit_normalize(positions[0] - positions[1])
            normedv = _jit_normalize(positions[2] - positions[1])

            internal_coordinates[i] = np.arccos(normedu @ normedv)

        # dihedrals
        else:
            # get positions of participating atoms
            positions = cart_positions[coord[:4]]

            # vectors making up dihedral
            normedu = _jit_normalize(positions[1] - positions[0])
            normedw = _jit_normalize(positions[2] - positions[1])
            normedv = _jit_normalize(positions[3] - positions[2])

            uw = normedu @ normedw
            wv = normedw @ normedv

            x = cross(cross(normedu, normedw), cross(normedw, normedv)) @ normedw
            y = cross(normedu, normedw) @ cross(normedw, normedv)

            if not np.isclose(uw, 1.0) and not np.isclose(wv, 1.0):
                internal_coordinates[i] = np.arctan2(x, y)
            else:
                raise ValueError("sinu or sinv is 0")

    return internal_coordinates


def _clean_dihedral(delta_c: Vector[float64], coord_idx: Primitives) -> Vector[float64]:
    return np.array(
        [
            coord_val if len(idx) != 4 else np.mod(coord_val + np.pi, 2 * np.pi) - np.pi
            for idx, coord_val in zip(coord_idx, delta_c)
        ]
    )  # type: ignore[return-value]


def _correct_dihedral_idx(
    struct: Cartesian,
    bad_idx: tuple[int, int, int, int],
    old_idx: Primitives,
    which_half: int,
) -> tuple[int, int, int, int]:
    # to check that new index is not already used

    if which_half == 1:
        origin = struct._get_origin(bad_idx[0])
        distances = [
            (i, struct.get_distance_to(origin).loc[i, "distance"]) for i in struct.index
        ]
        distances.sort(key=lambda x: x[1])
        for index, _ in distances:
            if index not in bad_idx:
                new_positions = struct.loc[[index] + list(bad_idx[1:-1]), COORDS].values

                # vectors making up the angle
                normedu = _jit_normalize(new_positions[0] - new_positions[1])
                normedv = _jit_normalize(new_positions[2] - new_positions[1])

                angle = np.arccos(normedu @ normedv)

                if (
                    not (angle < 5 * np.pi / 180) | (angle > 175 * np.pi / 180)
                    and tuple([index] + list(bad_idx[1:])) not in old_idx
                ):
                    return tuple([index] + list(bad_idx[1:]))
        raise RuntimeError(
            f"No suitable redifinition of poorly-defined dihedral {bad_idx}"
        )

    else:
        origin = struct._get_origin(bad_idx[3])
        distances = [
            (i, struct.get_distance_to(origin).loc[i, "distance"]) for i in struct.index
        ]
        distances.sort(key=lambda x: x[1])
        for index, _ in distances:
            if index not in bad_idx:
                new_positions = struct.loc[list(bad_idx[1:-1]) + [index], COORDS].values

                # vectors making up the angle
                normedu = _jit_normalize(new_positions[0] - new_positions[1])
                normedv = _jit_normalize(new_positions[2] - new_positions[1])

                angle = np.arccos(normedu @ normedv)

                if (
                    (5 < angle * 180 / np.pi < 175)  # fmt: skip
                    and tuple(list(bad_idx[:-1]) + [index]) not in old_idx
                ):
                    return tuple(list(bad_idx[:-1]) + [index])
        raise RuntimeError(
            f"No suitable redifinition of poorly-defined dihedral {bad_idx}"
        )
