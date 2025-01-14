from itertools import chain, combinations

import numpy as np
from numba import float64, int32, njit, prange
from numpy import cross
from numpy.linalg import norm, pinv
from numpy.typing import NDArray
from sortedcontainers import SortedSet
from typing_extensions import Self, TypeAlias, Union

from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord.xyz_functions import to_molden

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

        for item in prims:
            rev = tuple(reversed(item))
            if rev in prims:
                prims.remove(rev)

        return prims

    def get_Wilson_B(self, coordinates: Union[primitives, None] = None) -> NDArray:
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

        coord_arr = np.array(
            [
                np.append(np.resize(coord, 4), np.array(len(coord)))
                for coord in coordinates
            ]
        )
        pos_arr = np.array(
            [np.resize(self._get_positions(coord[:4]), (4, 3)) for coord in coord_arr]
        )

        return self.jit_get_Wilson_B(pos_arr, coord_arr, len(self))

    @staticmethod
    @njit((float64[:, :, :], int32[:, :], int32), parallel=True)
    def jit_get_Wilson_B(pos_arr: NDArray, coord_arr: NDArray, n_atoms: int) -> NDArray:
        """
        Jit-compiled Wilson's B matrix generator.

        Args:
            pos_arr (NDArray): array of cartesian coordinate locations of the atoms
                associated with each internal coordinate. If the coordinate is not a
                dihedral, there are unused coordinates in the 4th, or 3rd and 4th,
                places
            coord_arr (NDArray): array of internal coordinates, followed by the length
                of the coordinate. If the coordinate is not a dihedral, there are unused
                numbers in the 4th, or 3rd and 4th, places to ensure a rectangular array
            n_atoms (int): the number of atoms in the system, used to get matrix size

        Returns:
            NDArray[float64]: Wilson's B matrix
        """

        # initialize B matrix
        B_mat = np.zeros((len(coord_arr), 3 * n_atoms))

        for i in prange(len(coord_arr)):
            # separate cases for distances, angles, and dihedrals
            # procedure from J. Chem. Phys. 117, 9160 (2002); https://doi.org/10.1063/1.1515483
            coord = coord_arr[i]
            pos = pos_arr[i]
            # distances
            if coord[-1] == 2:
                # derivatives are just components of unit vector along distance

                u = pos[0] - pos[1]

                normedu = u / norm(u)
                # for each cartesian coordinate
                for j in range(3):
                    B_mat[i][j + 3 * coord[0]] = normedu[j]
                    B_mat[i][j + 3 * coord[1]] = -normedu[j]

            # angles
            elif coord[-1] == 3:
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
                    B_mat[i][j + 3 * coord[0]] = cross(normedu, normedw)[j] / norm(u)
                    B_mat[i][j + 3 * coord[1]] = -(
                        cross(normedu, normedw)[j] / norm(u)
                    ) - (cross(normedw, normedv)[j] / norm(v))
                    B_mat[i][j + 3 * coord[2]] = cross(normedw, normedv)[j] / norm(v)

            # dihedrals
            else:
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
                        B_mat[i][j + 3 * coord[0]] = float("nan")
                        B_mat[i][j + 3 * coord[1]] = float("nan")
                        B_mat[i][j + 3 * coord[2]] = float("nan")
                        B_mat[i][j + 3 * coord[3]] = float("nan")
                elif sinv == 0:
                    for j in range(3):
                        B_mat[i][j + 3 * coord[0]] = cross(normedu, normedw)[j] / (
                            norm(u) * (sinu**2)
                        )
                        B_mat[i][j + 3 * coord[1]] = float("nan")
                        B_mat[i][j + 3 * coord[2]] = float("nan")
                        B_mat[i][j + 3 * coord[3]] = float("nan")
                elif sinu == 0:
                    for j in range(3):
                        B_mat[i][j + 3 * coord[0]] = float("nan")
                        B_mat[i][j + 3 * coord[1]] = float("nan")
                        B_mat[i][j + 3 * coord[2]] = float("nan")
                        B_mat[i][j + 3 * coord[3]] = -cross(normedv, normedw)[j] / (
                            norm(v) * (sinv**2)
                        )
                else:
                    for j in range(3):
                        B_mat[i][j + 3 * coord[0]] = cross(normedu, normedw)[j] / (
                            norm(u) * (sinu**2)
                        )
                        B_mat[i][j + 3 * coord[1]] = -cross(normedu, normedw)[j] / (
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
                        B_mat[i][j + 3 * coord[2]] = cross(normedv, normedw)[j] / (
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
                        B_mat[i][j + 3 * coord[3]] = -cross(normedv, normedw)[j] / (
                            norm(v) * (sinv**2)
                        )

        return B_mat

    def x_to_c(self, coordinates: Union[primitives, None] = None) -> NDArray:
        """
        Conversion between cartesian coordinates and internal coordinates

        Args:
            coordinates (SortedSet[tuple]): default None, SortedSet of primitive
                coordinates to convert to. If None, calculates them using the
                get_primitive_coords method

        Returns:
            NDArray[float64]: array of internal coordinate values
        """
        # get primitive coordinates
        if coordinates is None:
            coordinates = self.get_primitive_coords()

        coord_arr = np.array(
            [
                np.append(np.resize(coord, 4), np.array(len(coord)))
                for coord in coordinates
            ]
        )
        pos_arr = np.array(
            [np.resize(self._get_positions(coord[:4]), (4, 3)) for coord in coord_arr]
        )
        return self.jit_x_to_c(pos_arr, coord_arr)

    @staticmethod
    @njit((float64[:, :, :], int32[:, :]), parallel=True)
    def jit_x_to_c(pos_arr: NDArray, coord_arr: NDArray) -> NDArray:
        """
        Jit-compiled conversion between cartesian coordinates and internal coordinates

        Args:
            pos_arr (NDArray): array of cartesian coordinate locations of the atoms
                associated with each internal coordinate. If the coordinate is not a
                dihedral, there are unused coordinates in the 4th, or 3rd and 4th,
                places
            coord_arr (NDArray): array of internal coordinates, followed by the length
                of the coordinate. If the coordinate is not a dihedral, there are unused
                numbers in the 4th, or 3rd and 4th, places to ensure a rectangular array

        Returns:
            NDArray[float64]: array of internal coordinate values
        """
        cs = np.empty(len(coord_arr))

        for i in prange(len(coord_arr)):
            # list of positions of atoms participating in coordinate
            pos = pos_arr[i]
            coord = coord_arr[i]

            # separate cases for distances, angles, and dihedrals

            # distances
            if coord[-1] == 2:
                # derivatives are just components of unit vector along distance
                u = pos[1] - pos[0]
                cs[i] = norm(u)

            # angles
            elif coord[-1] == 3:
                # vectors making up the angle
                u = pos[0] - pos[1]
                v = pos[2] - pos[1]

                normedu = u / norm(u)
                normedv = v / norm(v)

                cs[i] = np.arccos(np.dot(normedu, normedv))

            # dihedrals
            else:
                # vectors making up dihedral
                u = pos[1] - pos[0]
                w = pos[2] - pos[1]
                v = pos[3] - pos[2]

                normedu = u / norm(u)
                normedw = w / norm(w)
                normedv = v / norm(v)

                if np.dot(normedu, normedw) != 1 and np.dot(normedw, normedv) != 1:
                    cs[i] = np.arccos(
                        np.dot(cross(normedu, normedw), cross(normedv, normedw))
                        / (
                            np.sqrt(
                                (1 - np.dot(normedu, normedw) ** 2)
                                * (1 - np.dot(normedw, normedv) ** 2)
                            )
                        )
                    )
                else:
                    cs[i] = float("nan")

        return cs

    def get_B_traj(
        self,
        end: Self,
        N: int,
        M: int,
        generate_file: bool = True,
        filename: str = "",
        additional_coords: primitives = {},
        verbose: bool = False,
    ) -> NDArray:
        """
        Create a trajectory between two structures.

        This should be called in the following manner:
        StartCartesian.get_B_traj(EndCartesian, N, M)

        The trajectory should end close to the end Cartesian, but this is
        not currently guaranteed. I plan to update this soon.

        If generate_file is True, creates a molden file with name filename.

        Args:
            end (Cartesian): end structure
            N (int): number of subdivisions
            M (int): number of subdivisions before recalculating B-matrix
            generate_file (bool): default True, if True, generates molden out file
            filename (str): default "", filename for molden out file. If "", defaults to
                trajectory.out
            additional_coords (SortedSet[tuple]): SortedSet of additional primitive
                coordinates to use in the calculation of the trajectory.
            verbose (bool): default False, if True, prints extra information

        Returns:
            list[Cartesian]: pathway between self and end
        """

        path = np.concatenate((np.array([self]), np.empty(N)))

        # initial cartesian coordinates
        x_current = list(chain.from_iterable(self.loc[:, ["x", "y", "z"]].values))

        coords = (
            self.get_primitive_coords()
            | end.get_primitive_coords()
            | SortedSet(additional_coords, key=lambda x: (len(x), x))
        )

        B = self.get_Wilson_B()
        invB = np.array(pinv(B))

        c_current = self.x_to_c(coordinates=coords)
        c2 = end.x_to_c(coordinates=coords)

        # copy self to avoid side-effects
        current_struct = self.copy()
        # for each subdivision,
        for i in range(N):
            # difference between current point and final point in internal coordinates
            delta_c = c2 - c_current

            if i % M == 0:
                if verbose:
                    print(f"recalculating B at iteration {i}")
                B = current_struct.get_Wilson_B(coordinates=coords)
                invB = pinv(B)

            x_current = x_current + (invB @ delta_c / (N - i))

            assert len(x_current) % 3 == 0
            current_struct.loc[:, ["x", "y", "z"]] = np.reshape(
                x_current, (len(x_current) // 3, 3)
            )

            # converting to internal coordinates
            c_current = current_struct.x_to_c(coordinates=coords)

            path[i + 1] = current_struct.copy()

        if verbose:
            print(f"end internal coordinates: {c_current}")
            print(f"target end internal coordinates: {c2}")
            print(f"difference: {np.array(c2) - np.array(c_current)}")

        if generate_file:
            if not filename:
                to_molden(path, buf="trajectory.out")
            else:
                to_molden(path, buf=filename)

        return path
