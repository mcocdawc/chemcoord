from functools import partial
from itertools import combinations
from typing import TypeAlias

import numpy as np
from numba import njit, prange
from numpy import cross, float64, int64
from numpy.linalg import lstsq, norm
from sortedcontainers import SortedSet
from typing_extensions import Self

from chemcoord._cartesian_coordinates._cart_transformation import (
    _jit_normalize,
)
from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord.typing import BondDict, Matrix, Vector

#: Unfortunately SortedSet is not a generic type, if it was, the primitives
#: would be declared as
#: ``SortedSet[tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]``
primitives: TypeAlias = SortedSet


# the key prioritizes length, then sorts lexicographically
MySortedSet = partial(SortedSet, key=lambda x: (len(x), x))


class CartesianBmat(CartesianCore):
    def get_primitives_idx(
        self, bonds: BondDict | None = None, connect_fragments: bool = True
    ) -> primitives:
        """
        Generate set of redundant internal coordinates for the system.
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
    ) -> primitives:
        """This function calculated the primitive internal coordinates
        purely based on chemical connectivity and does not connect fragments."""
        idx_primitive_coords = MySortedSet()

        if bonds is None:
            bonds = self.get_bonds()

        def canonicalize(*args: int) -> tuple[int, ...]:
            if args[0] < args[-1]:
                return tuple(args)
            else:
                return tuple(reversed(args))

        # TODO early returns (purely for  performance))
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
        idx_internal_coords: primitives | None = None,
        bonds: BondDict | None = None,
    ) -> Matrix:
        """
        Generate Wilson's B matrix for the current structure.

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

        return self._jit_get_Wilson_B(
            self.loc[:, ["x", "y", "z"]].values,
            self._to_array(idx_internal_coords),
        )

    @staticmethod
    @njit(parallel=True, cache=True)
    def _jit_get_Wilson_B(
        position_arr: Matrix[float64],
        internal_coord_arr: Matrix[int64],
    ) -> Matrix[float64]:
        """
        Jit-compiled Wilson's B matrix generator.

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

    def x_to_c(
        self,
        internal_coords_idx: primitives | None = None,
        bonds: BondDict | None = None,
    ) -> Vector:
        """
        Conversion between cartesian coordinates and internal coordinates

        Args:
            internal_coordinates (SortedSet[tuple]): default None, SortedSet of
                primitive coordinates to convert to. If None, calculates them using the
                get_primitive_coords method
            use_lookup (bool): default False, if True, uses a lookup table for bond
                determination when generating primitive internal coordinates

        Returns:
            Vector[float64]: array of internal coordinate values
        """
        # get primitive coordinates
        if internal_coords_idx is None:
            internal_coords_idx = self.get_primitives_idx(bonds=bonds)

        return self.jit_x_to_c(
            self.loc[:, ["x", "y", "z"]].values, self._to_array(internal_coords_idx)
        )

    def _reindex_to_0(self, internal_coords_idx: primitives) -> primitives:
        """Return a reindexed version of `primitives` as if `self` was indexed
        contiguously from 0 to n - 1."""
        index_to_rownum = {index: row for row, index in enumerate(self.index)}

        return MySortedSet(
            {
                tuple(index_to_rownum[index] for index in coordinate_idx)
                for coordinate_idx in internal_coords_idx
            }
        )

    def _to_array(self, internal_coords_idx: primitives) -> Matrix[int64]:
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

    @staticmethod
    @njit(parallel=True, cache=True)
    def jit_x_to_c(
        cart_positions: Matrix[float64], internal_coords_idx: Matrix[int64]
    ) -> Vector[float64]:
        """
        Jit-compiled conversion between cartesian coordinates and internal coordinates

        .. note:: This function implicitly assumes that `internal_coords_idx`
            is for a 0-indexed molecule.

        Args:
            position_arr (Matrix): array of cartesian coordinate locations of the
                atoms in the Cartesian
            internal_coord_arr (Matrix): array of internal coordinates, followed by the
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

    def B_traj_step(
        self,
        end: Self,
        N: int,
        coord_idx: primitives,
        rcond: float | None = None,
    ) -> Self:
        # TODO: make sure additional_coords are getting passed along correctly.

        # this could be made faster by not recalculating this at every step, just
        # storing it in the main loop and passing it to this function

        c_current = self.x_to_c(internal_coords_idx=coord_idx)
        c2 = end.x_to_c(internal_coords_idx=coord_idx)

        # interpolated difference between current point
        # and final point in internal coordinates
        delta_c = self._clean_dihedral(c2 - c_current, coord_idx) / (N - 1)

        B = self.get_Wilson_B(idx_internal_coords=coord_idx)
        delta_x = lstsq(B, delta_c, rcond=rcond)[0]

        return self + delta_x.reshape(len(delta_x) // 3, 3)

    @staticmethod
    def _clean_dihedral(
        delta_c: Vector[float64], coord_idx: primitives
    ) -> Vector[float64]:
        return np.array(
            [
                coord_val
                if len(idx) != 4
                else (coord_val % (2 * np.pi))
                - ((coord_val % (2 * np.pi)) // np.pi) * (2 * np.pi)
                for idx, coord_val in zip(coord_idx, delta_c)
            ]
        )

    def get_B_traj(
        self,
        end: Self,
        N: int,
        *,
        primitives_idx: primitives | None = None,
        rcond: float | None = None,
    ) -> list[Self]:
        """
        Create a trajectory between two structures.

        This should be called in the following manner:
        StartCartesian.get_B_traj(EndCartesian, N)

        The trajectory should end close to the end Cartesian, but this is
        not currently guaranteed. I plan to update this soon.

        Args:
            end (Cartesian): end structure
            N (int): number of subdivisions
            rcond (float): ...

        Returns:
            list[Cartesian]: pathway between self and end
        """
        if primitives_idx is None:
            primitives_idx = self.get_primitives_idx() | end.get_primitives_idx()

        path = [self]

        # TODO: interpolate rotation

        # for each subdivision,
        for i in range(1, N - 2):
            new_struct = path[-1].B_traj_step(end, N - i, primitives_idx, rcond=rcond)
            path.append(new_struct)

        # interpolate from end, but align with last interpolated structure
        path.append(
            path[-1].align(end.B_traj_step(path[-1], 3, primitives_idx, rcond=rcond))[1]
            + path[-1].get_centroid()
        )
        # append aligned end
        path.append(path[-1].align(end)[1] + path[-1].get_centroid())

        return path


@njit(cache=True)
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


@njit(cache=True)
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
