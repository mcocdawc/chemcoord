from itertools import combinations
from typing import Callable, TypeAlias

import numpy as np
from numba import njit, prange
from numpy import cross
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


class CartesianBmat(CartesianCore):
    def get_primitive_coords(self, bonds: BondDict | None = None) -> primitives:
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
        # the key prioritizes length, then sorts lexicographically
        primitive_coordinates = SortedSet(key=lambda x: (len(x), x))

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
                primitive_coordinates.add(canonicalize(atom1, atom2))
                for atom3 in bonds[atom2] - {atom1}:
                    primitive_coordinates.add(canonicalize(atom1, atom2, atom3))
                    for atom4 in bonds[atom3] - {atom1, atom2}:
                        primitive_coordinates.add(
                            canonicalize(atom1, atom2, atom3, atom4)
                        )

        return primitive_coordinates

    def get_Wilson_B(
        self,
        internal_coordinates: primitives | None = None,
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

        # get primitive coordinates
        if internal_coordinates is None:
            internal_coordinates = self.get_primitive_coords(bonds)

        position_arr = np.array(self.loc[:, ["x", "y", "z"]])

        internal_coord_arr = np.empty((len(internal_coordinates), 5), dtype=int)
        for i, coordinate in enumerate(internal_coordinates):
            for j, index in enumerate(coordinate):
                internal_coord_arr[i, j] = index
            # label to differentiate bond distances, angles, and dihedrals
            internal_coord_arr[i, 4] = len(coordinate)

        return self.jit_get_Wilson_B(
            position_arr,
            internal_coord_arr,
            len(self),
            self.jit_angle_deriv,
            self.jit_dihedral_deriv,
        )

    @staticmethod
    @njit(parallel=True, cache=True)
    def jit_get_Wilson_B(
        position_arr: Matrix,
        internal_coord_arr: Matrix,
        n_atoms: int,
        angle_deriv: Callable,
        dihedral_deriv: Callable,
    ) -> Matrix:
        """
        Jit-compiled Wilson's B matrix generator.

        Args:
            position_arr (Matrix): array of cartesian coordinate locations of the
                atoms in the Cartesian
            internal_coord_arr (Matrix): array of internal coordinates, followed by the
                length of the coordinate. If the coordinate is not a dihedral, there are
                unused numbers in the 4th, or 3rd and 4th, places to ensure a
                rectangular array
            n_atoms (int): the number of atoms in the system, used to get matrix size

        Returns:
            Matrix[float64]: Wilson's B matrix
        """

        # initialize B matrix
        B_matrix = np.zeros((len(internal_coord_arr), 3 * n_atoms))

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

                angle_derivs = angle_deriv(positions)

                for j in prange(3):  # type: ignore[attr-defined]
                    B_matrix[i, j + 3 * coord[:3]] = angle_derivs[:3, j]

            # dihedrals
            else:
                # get positions of participating atoms
                positions = position_arr[coord[:4]]

                dihedral_derivs = dihedral_deriv(positions)

                for j in prange(3):  # type: ignore[attr-defined]
                    B_matrix[i, j + 3 * coord[:4]] = dihedral_derivs[:4, j]

        return B_matrix

    @staticmethod
    @njit(cache=True)
    def jit_angle_deriv(positions: Matrix) -> Matrix:
        # vectors making up the angle

        u = positions[0] - positions[1]
        normedu = _jit_normalize(u)
        v = positions[2] - positions[1]
        normedv = _jit_normalize(v)

        w = cross(u, v)

        # if they were parallel
        if np.allclose(w, 0.0):
            w = cross(normedu, np.array([1, -1, 1]))
        # if u and [1, -1, 1] were parallel
        if np.allclose(w, 0.0):
            w = cross(normedu, np.array([-1, 1, 1]))

        normedw = _jit_normalize(w)
        return np.stack(
            (
                cross(normedu, normedw) / norm(u),
                -(cross(normedu, normedw) / norm(u))
                - (cross(normedw, normedv) / norm(v)),
                cross(normedw, normedv) / norm(v),
            )
        )

    @staticmethod
    @njit(cache=True)
    def jit_dihedral_deriv(positions: Matrix) -> Matrix:
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
        if any(np.isclose([sinu, sinv], 0)):
            raise ValueError("sinu or sinv is 0")
        else:
            return np.stack(
                (
                    cross(normedu, normedw) / (norm(u) * (sinu**2)),
                    -cross(normedu, normedw) / (norm(u) * (sinu**2))
                        + (
                            (
                                (cross(normedu, normedw) * cosu)
                                / (norm(w) * (sinu**2))
                            )
                            - (
                                (cross(normedv, normedw) * cosv)
                                / (norm(w) * (sinv**2))
                            )
                        ),
                    cross(normedv, normedw) / (norm(v) * (sinv**2))
                        - (
                            (
                                (cross(normedu, normedw) * cosu)
                                / (norm(w) * (sinu**2))
                            )
                            - (
                                (cross(normedv, normedw) * cosv)
                                / (norm(w) * (sinv**2))
                            )
                        ),
                    -cross(normedv, normedw) / (norm(v) * (sinv**2)),
                )
            )

    def x_to_c(
        self,
        internal_coordinates: primitives | None = None,
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
        if internal_coordinates is None:
            internal_coordinates = self.get_primitive_coords(bonds=bonds)

        position_arr = np.array(self.loc[:, ["x", "y", "z"]])

        internal_coord_arr = np.empty((len(internal_coordinates), 5), dtype=int)
        for i, coordinate in enumerate(internal_coordinates):
            for j, index in enumerate(coordinate):
                internal_coord_arr[i, j] = index
            internal_coord_arr[i, 4] = len(coordinate)

        return self.jit_x_to_c(position_arr, internal_coord_arr)

    @staticmethod
    @njit(parallel=True, cache=True)
    def jit_x_to_c(position_arr: Matrix, internal_coord_arr: Matrix) -> Vector:
        """
        Jit-compiled conversion between cartesian coordinates and internal coordinates

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
        internal_coordinates = np.empty(len(internal_coord_arr))

        for i in prange(len(internal_coord_arr)): # type: ignore[attr-defined]
            # get ith internal coordinate
            coord = internal_coord_arr[i]

            # separate cases for distances, angles, and dihedrals

            # distances
            if coord[-1] == 2:
                # get positions of participating atoms
                positions = position_arr[coord[:2]]

                u = positions[1] - positions[0]
                internal_coordinates[i] = norm(u)

            # angles
            elif coord[-1] == 3:
                # get positions of participating atoms
                positions = position_arr[coord[:3]]

                # vectors making up the angle
                normedu = _jit_normalize(positions[0] - positions[1])
                normedv = _jit_normalize(positions[2] - positions[1])

                internal_coordinates[i] = np.arccos(normedu @ normedv)

            # dihedrals
            else:
                # get positions of participating atoms
                positions = position_arr[coord[:4]]

                # vectors making up dihedral
                normedu = _jit_normalize(positions[1] - positions[0])
                normedw = _jit_normalize(positions[2] - positions[1])
                normedv = _jit_normalize(positions[3] - positions[2])

                uw = normedu @ normedw
                wv = normedw @ normedv

                x = cross(cross(normedu, normedw), cross(normedw, normedv)) @ normedw
                y = cross(normedu, normedw) @ cross(normedw, normedv)

                if not any(np.isclose([uw, wv], 1)):
                    internal_coordinates[i] = np.arctan2(x, y)
                else:
                    raise ValueError("sinu or sinv is 0")

        return internal_coordinates

    def B_traj_step(
        self,
        end: Self,
        N: int,
        coords: primitives,
        rcond: float | None = None,
    ) -> Self:
        # TODO: make sure additional_coords are getting passed along correctly.
        current_struct = self.copy()

        x_current = np.array(self.loc[:, ["x", "y", "z"]]).flatten()

        c_current = self.x_to_c(internal_coordinates=coords)
        # this could be made faster by not recalculating this at every step, just
        # storing it in the main loop and passing it to this function
        c2 = end.x_to_c(internal_coordinates=coords)

        # difference between current point and final point in internal coordinates
        delta_c = c2 - c_current

        # TODO: change this to use the modulus function already written in here
        # check to make sure it takes the shorter dihedral and angle
        delta_c = np.array(
            [
                delta_c[i]
                if len(coords[i]) != 4
                else (delta_c[i] % (2 * np.pi))
                - ((delta_c[i] % (2 * np.pi)) // np.pi) * (2 * np.pi)
                for i in range(len(delta_c))
            ]
        )

        B = current_struct.get_Wilson_B(internal_coordinates=coords)
        delta_x = lstsq(B, delta_c, rcond=rcond)[0]

        x_current = x_current + (delta_x / N)

        assert len(x_current) % 3 == 0
        current_struct.loc[:, ["x", "y", "z"]] = np.reshape(
            x_current, (len(x_current) // 3, 3)
        )

        return current_struct

    def get_B_traj(
        self,
        end: Self,
        N: int,
        additional_coords: primitives | None = None,
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
            additional_coords (SortedSet[tuple]): SortedSet of additional primitive
                coordinates to use in the calculation of the trajectory.
            rcond (float): ...

        Returns:
            list[Cartesian]: pathway between self and end
        """

        if additional_coords is None:
            additional_coords = SortedSet([], key=lambda x: (len(x), x))

        bonds = self.get_bonds()
        fragments = self.fragmentate()
        if len(fragments) != 1:
            for fragment_pair in combinations(fragments, 2):
                index1, index2, _ = fragment_pair[0].get_shortest_distance(
                    fragment_pair[1]
                )
                bonds[index1].add(index2)
                bonds[index2].add(index1)

        coords = (
            self.get_primitive_coords(bonds=bonds)
            | end.get_primitive_coords()
            | SortedSet(additional_coords, key=lambda x: (len(x), x))
        )

        path = [self]

        # TODO: interpolate rotation

        # for each subdivision,
        for i in range(N):
            new_struct = path[i].B_traj_step(end, N - i, coords, rcond=rcond)
            path.append(new_struct)

        # temporary rotational and translational alignment
        for i, image in enumerate(path[1:]):
            path[i + 1] = path[i].align(path[i + 1])[1]
        path.append(path[-1].align(end)[1])

        return path