from itertools import combinations

import numpy as np
from numba import njit, prange
from numpy import cross
from numpy.linalg import norm
from sortedcontainers import SortedSet
from typing_extensions import TypeAlias, Union

from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord._utilities.typing import Matrix, Tensor3D, Vector

primitives: TypeAlias = SortedSet[
    Union[tuple[int, int], tuple[int, int, int], tuple[int, int, int, int]]
]


class CartesianBmat(CartesianCore):
    def get_primitive_coords(self) -> primitives:
        """
        Generate set of redundant internal coordinates for the system.

        Stored in a sortedcontainers.SortedSet to maintain order while
        being able to use Python's union operator. Sorted by length of
        coordinate, then by standard order based on the atoms' indices.

        Args:
            coordinates (SortedSet[tuple]): default None, SortedSet of primitive
                coordinates to use in the calculation. If None, calculates using the
                get_primitive_coords method

        Returns:
            SortedSet[tuple]: SortedSet of redundant internal coordinates
        """

        # the key prioritizes length, then sorts lexicographically
        prims = SortedSet(key=lambda x: (len(x), x))

        bonds = self.get_bonds()

        for atom1, atom2 in combinations(range(len(bonds)), 2):
            if atom2 in bonds[atom1]:
                prims.add((atom1, atom2))

                for atom0 in bonds[atom1] - {atom2}:
                    prims.add((atom0, atom1, atom2))

                    for atom00 in bonds[atom0] - {atom1, atom2}:
                        prims.add((atom00, atom0, atom1, atom2))

                for atom3 in bonds[atom2] - {atom1}:
                    prims.add((atom1, atom2, atom3))

                    for atom4 in bonds[atom3] - {atom1, atom2}:
                        prims.add((atom1, atom2, atom3, atom4))

        # get rid of reversed duplicates
        for item in prims:
            rev = tuple(reversed(item))
            if rev in prims:
                prims.remove(rev)

        return prims

    def get_Wilson_B(self, coordinates: Union[primitives, None] = None) -> Matrix:
        """
        Generate Wilson's B matrix for the current structure.

        Args:
            coordinates (SortedSet[tuple]): default None, SortedSet of primitive
                coordinates to use in the calculation. If None, calculates using the
                get_primitive_coords method

        Returns:
            NDArray[float64]: Wilson's B matrix
        """

        # get primitive coordinates
        if coordinates is None:
            coordinates = self.get_primitive_coords()

        position_arr = np.array(self.loc[:, ["x", "y", "z"]])

        internal_coord_arr = np.array(
            [
                np.append(np.resize(coord, 4), np.array(len(coord)))
                for coord in coordinates
            ]
        )

        return self.jit_get_Wilson_B(position_arr, internal_coord_arr, len(self))

    @staticmethod
    @njit(parallel=True, cache=True)
    def jit_get_Wilson_B(
        position_arr: Tensor3D, internal_coord_arr: Matrix, n_atoms: int
    ) -> Matrix:
        """
        Jit-compiled Wilson's B matrix generator.

        Args:
            position_arr (Tensor3D): array of cartesian coordinate locations of the
                atoms in the Cartesian object.
            internal_coord_arr (Matrix): array of internal coordinates, followed by the
                length of the coordinate. If the coordinate is not a dihedral, there are
                unused numbers in the 4th, or 3rd and 4th, places to ensure a
                rectangular array
            n_atoms (int): the number of atoms in the system, used to get matrix size

        Returns:
            Matrix[float64]: Wilson's B matrix
        """

        # initialize B matrix
        B_mat = np.zeros((len(internal_coord_arr), 3 * n_atoms))

        for i in prange(len(internal_coord_arr)):
            # separate cases for distances, angles, and dihedrals
            # procedure from J. Chem. Phys. 117, 9160 (2002); https://doi.org/10.1063/1.1515483

            # get ith internal coordinate
            coord = internal_coord_arr[i]

            # distances
            if coord[-1] == 2:
                # get positions of participating atoms
                pos = position_arr[coord[:2]]

                # derivatives are just components of unit vector along distance

                u = pos[0] - pos[1]

                normedu = u / norm(u)
                # for each cartesian coordinate
                for j in range(3):
                    B_mat[i, j + 3 * coord[0]] = normedu[j]
                    B_mat[i, j + 3 * coord[1]] = -normedu[j]

            # angles
            elif coord[-1] == 3:
                # get positions of participating atoms
                pos = position_arr[coord[:3]]

                # vectors making up the angle
                u = pos[0] - pos[1]
                v = pos[2] - pos[1]

                normedu = u / norm(u)
                normedv = v / norm(v)

                # for testing if we are using the right angle definitions
                # print(f"{coord}: {np.arccos(np.dot(normedu,normedv))}")

                w = cross(u, v)

                # if they were parallel
                if np.isclose(w, np.array([0, 0, 0])).all():
                    w = cross(u, np.array([1, -1, 1]))

                # if u and [1, -1, 1] were parallel
                if np.isclose(w, np.array([0, 0, 0])).all():
                    w = cross(u, np.array([-1, 1, 1]))

                normedw = w / norm(w)

                for j in range(3):
                    B_mat[i, j + 3 * coord[0]] = cross(normedu, normedw)[j] / norm(u)
                    B_mat[i, j + 3 * coord[1]] = -(
                        cross(normedu, normedw)[j] / norm(u)
                    ) - (cross(normedw, normedv)[j] / norm(v))
                    B_mat[i, j + 3 * coord[2]] = cross(normedw, normedv)[j] / norm(v)

            # dihedrals
            else:
                # get positions of participating atoms
                pos = position_arr[coord[:4]]

                # vectors making up dihedral
                u = pos[0] - pos[1]
                w = pos[2] - pos[1]
                v = pos[3] - pos[2]

                normedu = u / norm(u)
                normedw = w / norm(w)
                normedv = v / norm(v)

                cosu = np.dot(normedu, normedw)
                # note this could be 0 if undefined dihedral
                sinu = np.sqrt(1 - (np.dot(normedu, normedw)) ** 2)

                cosv = -np.dot(normedv, normedw)
                # note this could be 0 if undefined dihedral
                sinv = np.sqrt(1 - (np.dot(normedv, normedw)) ** 2)

                # catching cases where certain dihedrals are undefined
                if (sinu, sinv) == (0, 0):
                    for j in range(3):
                        B_mat[i, j + 3 * coord[0]] = float("nan")
                        B_mat[i, j + 3 * coord[1]] = float("nan")
                        B_mat[i, j + 3 * coord[2]] = float("nan")
                        B_mat[i, j + 3 * coord[3]] = float("nan")
                elif sinv == 0:
                    for j in range(3):
                        B_mat[i, j + 3 * coord[0]] = cross(normedu, normedw)[j] / (
                            norm(u) * (sinu**2)
                        )
                        B_mat[i, j + 3 * coord[1]] = float("nan")
                        B_mat[i, j + 3 * coord[2]] = float("nan")
                        B_mat[i, j + 3 * coord[3]] = float("nan")
                elif sinu == 0:
                    for j in range(3):
                        B_mat[i, j + 3 * coord[0]] = float("nan")
                        B_mat[i, j + 3 * coord[1]] = float("nan")
                        B_mat[i, j + 3 * coord[2]] = float("nan")
                        B_mat[i, j + 3 * coord[3]] = -cross(normedv, normedw)[j] / (
                            norm(v) * (sinv**2)
                        )
                else:
                    for j in range(3):
                        B_mat[i, j + 3 * coord[0]] = cross(normedu, normedw)[j] / (
                            norm(u) * (sinu**2)
                        )
                        B_mat[i, j + 3 * coord[1]] = -cross(normedu, normedw)[j] / (
                            norm(u) * (sinu**2)
                        ) + (
                            (
                                (cross(normedu, normedw)[j] * cosu)
                                / (norm(w) * (sinu**2))
                            )
                            - (
                                (cross(normedv, normedw)[j] * cosv)
                                / (norm(w) * (sinv**2))
                            )
                        )
                        B_mat[i, j + 3 * coord[2]] = cross(normedv, normedw)[j] / (
                            norm(v) * (sinv**2)
                        ) - (
                            (
                                (cross(normedu, normedw)[j] * cosu)
                                / (norm(w) * (sinu**2))
                            )
                            - (
                                (cross(normedv, normedw)[j] * cosv)
                                / (norm(w) * (sinv**2))
                            )
                        )
                        B_mat[i, j + 3 * coord[3]] = -cross(normedv, normedw)[j] / (
                            norm(v) * (sinv**2)
                        )

        return B_mat

    def x_to_c(self, coordinates: Union[primitives, None] = None) -> Vector:
        """
        Conversion between cartesian coordinates and internal coordinates

        Args:
            coordinates (SortedSet[tuple]): default None, SortedSet of primitive
                coordinates to convert to. If None, calculates them using the
                get_primitive_coords method

        Returns:
            Vector[float64]: array of internal coordinate values
        """
        # get primitive coordinates
        if coordinates is None:
            coordinates = self.get_primitive_coords()

        internal_coord_arr = np.array(
            [
                np.append(np.resize(coord, 4), np.array(len(coord)))
                for coord in coordinates
            ]
        )

        pos_arr = np.array(self.loc[:, ["x", "y", "z"]])

        return self.jit_x_to_c(pos_arr, internal_coord_arr)

    @staticmethod
    @njit(parallel=True, cache=True)
    def jit_x_to_c(pos_arr: Tensor3D, coord_arr: Matrix) -> Vector:
        """
        Jit-compiled conversion between cartesian coordinates and internal coordinates

        Args:
            pos_arr (Tensor3D): array of cartesian coordinate locations of the atoms
                associated with each internal coordinate. If the coordinate is not a
                dihedral, there are unused coordinates in the 4th, or 3rd and 4th,
                places
            coord_arr (Matrix): array of internal coordinates, followed by the length
                of the coordinate. If the coordinate is not a dihedral, there are unused
                numbers in the 4th, or 3rd and 4th, places to ensure a rectangular array

        Returns:
            Vector[float64]: array of internal coordinate values
        """
        cs = np.empty(len(coord_arr))

        for i in prange(len(coord_arr)):
            # get ith internal coordinate
            coord = coord_arr[i]

            # separate cases for distances, angles, and dihedrals

            # distances
            if coord[-1] == 2:
                # get positions of participating atoms
                pos = pos_arr[coord[:2]]

                # derivatives are just components of unit vector along distance
                u = pos[1] - pos[0]
                cs[i] = norm(u)

            # angles
            elif coord[-1] == 3:
                # get positions of participating atoms
                pos = pos_arr[coord[:3]]

                # vectors making up the angle
                u = pos[0] - pos[1]
                v = pos[2] - pos[1]

                normedu = u / norm(u)
                normedv = v / norm(v)

                cs[i] = np.arccos(np.dot(normedu, normedv))

            # dihedrals
            else:
                # get positions of participating atoms
                pos = pos_arr[coord[:4]]

                # vectors making up dihedral
                u = pos[1] - pos[0]
                w = pos[2] - pos[1]
                v = pos[3] - pos[2]

                normedu = u / norm(u)
                normedw = w / norm(w)
                normedv = v / norm(v)

                uw = np.dot(normedu, normedw)
                wv = np.dot(normedw, normedv)

                x = -np.dot(cross(normedu, normedw), cross(normedv, normedw))
                y = np.dot(
                    cross(cross(normedu, normedw), normedw), cross(normedv, normedw)
                )

                if uw != 1 and wv != 1:
                    cs[i] = np.arctan2(y, x)
                else:
                    cs[i] = float("nan")

        return cs
