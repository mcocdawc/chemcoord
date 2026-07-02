from __future__ import annotations

from enum import IntEnum
from functools import partial
from itertools import combinations
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numba import njit, prange
from numpy import cross, float64, int64
from numpy.linalg import norm

# had to put these here to avoid circular import
from sortedcontainers import SortedSet

from chemcoord._cartesian_coordinates._cart_transformation import (
    _jit_normalize,
)
from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import COORDS
from chemcoord.exceptions import UndefinedDihedral
from chemcoord.typing import AtomIdx, BondDict, Matrix, Vector

if TYPE_CHECKING:
    from chemcoord import Cartesian
    from chemcoord._redundant_internal_coordinates.main import (
        RedundantInternalCoordinates,
    )


#: Unfortunately SortedSet is not a generic type, if it was, the primitives
#: would be declared as
#: ``SortedSet[tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]``
Primitives: TypeAlias = SortedSet


# the key prioritizes length, then sorts lexicographically
SetOfPrimitives = partial(SortedSet, key=lambda x: (len(x), x))


class BendType(IntEnum):
    """enum to differentiate the in and out-of-plane bending coordinates

    uw and vw planes are defined as in :cite:`in-out-of-plane-paper`"""

    UW = 0
    VW = 1


class WhichHalf(IntEnum):
    """enum to differentiate linearities happening in either half of a dihedral"""

    first = 1
    second = 2


Coordinate: TypeAlias = (
    tuple[AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx, BendType]
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
            coordinates: default :class:`None`, SortedSet of primitive
                coordinates to use in the calculation. If None, calculates using the
                get_primitive_coords method
            bonds: default :class:`None`, mapping containing bonding information.
                Generates a new mapping via :meth:`~Cartesian.get_bonds` if
                :class:`None` is provided
        Returns:
            SortedSet of redundant internal coordinates
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
            idx_internal_coords: default None, SortedSet of
                primitive internal coordinates to use in the calculation. If None,
                calculates using the get_primitive_coords method
            bonds: default :class:`None`, mapping containing bonding information.

        Returns:
            Wilson's B matrix
        """
        if idx_internal_coords is None:
            idx_internal_coords = self.get_primitives_idx(bonds)

        return _jit_get_Wilson_B(
            self.loc[:, COORDS].values,
            self._to_array_nobending(idx_internal_coords),
        )

    def get_ric(
        self,
        internal_coords_idx: Primitives | None = None,
        bonds: BondDict | None = None,
    ) -> RedundantInternalCoordinates:
        """Conversion to redundant internal coordinates

        Args:
            internal_coord_idx: default None, SortedSet of
                primitive coordinates to convert to. If None, calculates them using the
                get_primitive_coords method
            bonds: default :class:`None`, mapping containing bonding information.

        Returns:
            Redundant internal coordinate representation of self

        """
        from chemcoord._redundant_internal_coordinates.main import (  # noqa: PLC0415
            RedundantInternalCoordinates,
        )

        # get primitive coordinates
        if internal_coords_idx is None:
            internal_coords_idx = self.get_primitives_idx(bonds=bonds)

        ric_values, exceptions = _jit_x_to_ric(
            self.loc[:, COORDS].values,
            self._to_array_full(internal_coords_idx),
        )

        if np.any(exceptions[:, -1]):
            exception_list = []
            for coordinate in exceptions:
                if coordinate[-1] != 0:
                    exception_list.append(tuple(coordinate[:-1]))
            raise UndefinedDihedral(exception_list)

        return RedundantInternalCoordinates(
            ric_values,
            internal_coords_idx,
            self.copy(),  # type: ignore[arg-type]
        )

    def _reindex_to_0(self, internal_coords_idx: Primitives) -> Primitives:
        """Return a reindexed version of `primitives` as if `self` was indexed
        contiguously from 0 to n - 1."""
        index_to_rownum = {index: row for row, index in enumerate(self.index)}

        return SetOfPrimitives(
            {
                _reindex_to_0_inner(coordinate_idx, index_to_rownum)
                for coordinate_idx in internal_coords_idx
            }
        )

    def _to_array_nobending(
        self,
        internal_coords_idx: Primitives,
    ) -> Matrix[int64]:
        """Converts the index of the primitive internal coordinates aside from the
        bending coordinates to an array and changes to 0-based indexing.

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

    def _to_array_full(
        self,
        internal_coords_idx: Primitives,
    ) -> Matrix[int64]:
        """Converts the index of the primitive internal coordinates to an array
        and changes to 0-based indexing.

        The array is a rectangualar (n, 5) array, where n is the number of
        internal coordinates. The last column denotes the number of defined columns and
        the type of coordinate, i.e. (n=2) bond, (n=3) angle, (n=4) dihedral.

        The bending coordinates are handeled differently. If it is a y-bending
        coordinate, then it will be in the first bending coordinate set and the last
        column in the array will be 5, otherwise the last column becomes 6.

        In addition, this function switches from the arbitrary flexible index of a
        molecule to 0-based indexing.
        """

        internal_coord_idx_arr = np.empty(
            (
                len(internal_coords_idx),
                5,
            ),
            dtype=int64,
        )

        for i, coordinate in enumerate(self._reindex_to_0(internal_coords_idx)):
            if _is_uw_bending_tuple(coordinate):
                internal_coord_idx_arr[i, :4] = coordinate[:4]
                internal_coord_idx_arr[i, 4] = 5
            elif _is_vw_bending_tuple(coordinate):
                internal_coord_idx_arr[i, :4] = coordinate[:4]
                internal_coord_idx_arr[i, 4] = 6
            else:
                internal_coord_idx_arr[i, : len(coordinate)] = coordinate
                internal_coord_idx_arr[i, 4] = len(coordinate)

        return internal_coord_idx_arr

    def _fix_undef_dihedrals(
        self,
        coord_idx: Primitives,
        tol: float = 5,
    ) -> tuple[Primitives, Primitives]:
        """Finds linear dihedral coordinates (which are thus close to undefined-ness)
        and replaces them with dihedrals constructed from the first non-collinear atom
        replacing the bad end.

        Args:
            coord_idx: SortedSet of primitive coordinates to clean
            tol: default 5, tolerance for collinear atom detection, in degrees.
        Returns:
            Tuple of SortedSet of cleaned redundant
                internal coordinates and SortedSet of removed indices
        """
        # getting rid of poorly-defined dihedrals because of linearity at the start
        new_coord_idx = coord_idx.copy()
        bad_indices = SetOfPrimitives()
        for i, index in enumerate(coord_idx.copy()):
            if _is_dihedral_tuple(index):
                first_three = self.loc[index[:-1], COORDS].values
                last_three = self.loc[index[1:], COORDS].values

                # vectors making up the angle
                normedu1 = _jit_normalize(first_three[0] - first_three[1])
                normedv1 = _jit_normalize(first_three[2] - first_three[1])

                normedu2 = _jit_normalize(last_three[0] - last_three[1])
                normedv2 = _jit_normalize(last_three[2] - last_three[1])

                angle1 = np.arccos(normedu1 @ normedv1)
                angle2 = np.arccos(normedu2 @ normedv2)

                if not (tol < angle1 * 180 / np.pi < (180 - tol)):
                    bad_idx = index
                    bad_indices.add(index)
                    new_coord_idx.add(
                        _correct_dihedral_idx(self, bad_idx, coord_idx, WhichHalf.first)  # type: ignore[arg-type]
                    )
                if not (tol < angle2 * 180 / np.pi < (180 - tol)):
                    bad_idx = index
                    bad_indices.add(index)
                    new_coord_idx.add(
                        _correct_dihedral_idx(
                            self,  # type: ignore[arg-type]
                            bad_idx,
                            coord_idx,
                            WhichHalf.second,
                        )
                    )

        return (new_coord_idx, bad_indices)

    def linearities(self, coord_idx: Primitives, tol: float = 5) -> list[tuple]:
        """Finds linear dihedral coordinates (which are thus close to undefined-ness).

        Args:
            coord_idx: SortedSet of primitive coordinates to search
            tol: default 5, tolerance for collinear atom detection, in degrees.
        Returns:
            List of tuples of linear dihedrals and 1 if the
                linearity is in the first 3 atoms and 2 otherwise
        """
        linear_idx = []
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

                if not (tol < angle1 * 180 / np.pi < (180 - tol)):
                    linear_idx.append((index, 1))
                if not (tol < angle2 * 180 / np.pi < (180 - tol)):
                    linear_idx.append((index, 2))
        return linear_idx


@njit(cache=True, nogil=True)
def _jit_dihedral_deriv(positions: Matrix) -> Matrix:
    """Calculates Cartesian derivatives of a dihedral angle given 4 atom positions"""

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
    """Calculates Cartesian derivatives of an angle given 3 atom positions"""

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
def _jit_uw_deriv(positions: Matrix, axes: Matrix) -> Matrix:
    i = positions[1]
    j = positions[2]
    k = positions[3]

    u = axes[0]
    v = axes[1]
    w = axes[2]

    ddu_uw = np.array(
        [
            -1 / np.linalg.norm(j - i),
            (1 / np.linalg.norm(j - i))
            + ((k) @ w) / np.linalg.norm((k - j) - ((k - j) @ v)),
            -((k) @ w) / np.linalg.norm((k - j) - ((k - j) @ v)),
        ]
    )
    ddw_uw = np.array(
        [
            0,
            -(k) @ u / np.linalg.norm((k - j) - ((k - j) @ v)),
            (k) @ u / np.linalg.norm((k - j) - ((k - j) @ v)),
        ]
    )

    return axes.T @ np.vstack((ddu_uw, np.array([0.0, 0.0, 0.0]), ddw_uw))


@njit(cache=True, nogil=True)
def _jit_vw_deriv(positions: Matrix, axes: Matrix) -> Matrix:
    i = positions[1]
    j = positions[2]
    k = positions[3]

    u = axes[0]
    v = axes[1]
    w = axes[2]

    ddv_vw = np.array(
        [
            -1 / np.linalg.norm(j - i),
            (1 / np.linalg.norm(j - i))
            + ((k) @ w) / np.linalg.norm((k - j) - ((k - j) @ u)),
            -((k) @ w) / np.linalg.norm((k - j) - ((k - j) @ u)),
        ]
    )
    ddw_vw = np.array(
        [
            0,
            -(k) @ v / np.linalg.norm((k - j) - ((k - j) @ u)),
            (k) @ v / np.linalg.norm((k - j) - ((k - j) @ u)),
        ]
    )

    return axes.T @ np.vstack((np.array([0.0, 0.0, 0.0]), ddv_vw, ddw_vw))


@njit(parallel=True, cache=True, nogil=True)
def _jit_get_Wilson_B(
    position_arr: Matrix[float64],
    internal_coord_arr: Matrix[int64],
) -> Matrix[float64]:
    """Jit-compiled Wilson's B matrix generator, sans bending coordinates.

    procedure from: :cite:`B_matrix`

    Args:
        position_arr: array of cartesian coordinate locations of the
            atoms in the Cartesian
        internal_coord_arr: array of internal coordinates, followed by the
            length of the coordinate. If the coordinate is not a dihedral, there are
            unused numbers in the 4th, or 3rd and 4th, places to ensure a
            rectangular array

    Returns:
        Wilson's B matrix
    """

    # initialize B matrix
    B_matrix = np.zeros((len(internal_coord_arr), position_arr.size))

    for i in prange(len(internal_coord_arr)):  # type: ignore[attr-defined]
        # separate cases for distances, angles, and dihedrals
        # procedure from J. Chem. Phys. 117, 9160 (2002); https://doi.org/10.1063/1.1515483

        # get ith internal coordinate
        coord = internal_coord_arr[i, :]

        # distances
        if _is_bond_array(coord):
            # get positions of participating atoms
            positions = position_arr[coord[:2]]

            # derivatives are just components of unit vector along distance
            normedu = _jit_normalize(positions[0] - positions[1])
            # for each cartesian coordinate
            for j in prange(3):  # type: ignore[attr-defined]
                B_matrix[i, j + 3 * coord[0]] = normedu[j]
                B_matrix[i, j + 3 * coord[1]] = -normedu[j]

        # angles
        elif _is_angle_array(coord):
            # get positions of participating atoms
            positions = position_arr[coord[:3]]

            angle_derivs = _jit_angle_deriv(positions)

            for j in prange(3):  # type: ignore[attr-defined]
                B_matrix[i, j + 3 * coord[:3]] = angle_derivs[:3, j]

        # dihedrals
        elif _is_dihedral_array(coord):
            # get positions of participating atoms
            positions = position_arr[coord[:4]]

            dihedral_derivs = _jit_dihedral_deriv(positions)

            for j in prange(3):  # type: ignore[attr-defined]
                B_matrix[i, j + 3 * coord[:4]] = dihedral_derivs[:4, j]

        elif _is_uw_bending_array(coord):
            positions = position_arr[coord[:4]]

            axes = _jit_get_axes(position_arr, coord[:4])

            uw_derivs = _jit_uw_deriv(positions, axes)

            for j in prange(3):  # type: ignore[attr-defined]
                B_matrix[i, j + 3 * coord[1:4]] = uw_derivs[:, j]

        elif _is_vw_bending_array(coord):
            positions = position_arr[coord[:4]]

            axes = _jit_get_axes(position_arr, coord[:4])

            vw_derivs = _jit_vw_deriv(positions, axes)

            for j in prange(3):  # type: ignore[attr-defined]
                B_matrix[i, j + 3 * coord[1:4]] = vw_derivs[:, j]

    return B_matrix


# NOTE: no ``parallel=True`` here. This function operates on a single bending
# coordinate (fixed-size arrays) and has no ``prange`` loop, so parallelisation
# brings no benefit. More importantly, it is called from within the ``prange``
# loops of ``_jit_get_Wilson_B`` and ``_jit_x_to_ric``; marking it
# ``parallel=True`` would create a nested parallel region, which crashes the
# (non-threadsafe) ``workqueue`` threading layer numba falls back to when
# neither TBB nor OpenMP is available.
@njit(cache=True, nogil=True)
def _jit_get_axes(
    cart_positions: Matrix[float64], idx: Vector[int64]
) -> Matrix[float64]:
    """Jit-compiled bending coordinate reference frame calculator.

    Args:
        cart_positions: array of cartesian coordinate locations of the
            atoms in the Cartesian
        idx: array of the bending coordinate

    Returns:
        Bending coordinate reference frame
    """
    h = cart_positions[idx[0]]
    i = cart_positions[idx[1]]
    j = cart_positions[idx[2]]

    u = _jit_normalize(np.cross((h - i), (j - i)))
    w = _jit_normalize(j - i)
    v = np.cross(w, u)

    return np.stack((u, v, w))


# NOTE: no ``parallel=True`` here (see ``_jit_get_axes``): it has no ``prange``
# loop and is called from within the ``prange`` loop of ``_jit_x_to_ric``, so
# making it parallel would nest parallel regions and crash the ``workqueue``
# threading layer.
@njit(cache=True, nogil=True)
def _jit_x_to_plane_coords_nonlinear(
    cart_positions: Matrix[float64], idx: Vector[int64]
) -> Vector[float64]:
    """Jit-compiled method to get an in/out-of-plane bending coordinate from cartesian
    coordinates

    .. note:: This function implicitly assumes that `internal_coords_idx`
        is for a 0-indexed molecule.

    Args:
        cart_positions: array of cartesian coordinate locations of the
            atoms in the Cartesian
        idx: Vector containing the relevant atomic indices for the bending
            coordinates

    Returns:
        Array of bending coordinate values
    """

    # technically inefficient to do this twice but makes it easier the other time we use
    # _jit_get_axes
    j = cart_positions[idx[2]]
    k = cart_positions[idx[3]]

    axes = _jit_get_axes(cart_positions, idx)
    u = axes[0]
    v = axes[1]
    w = axes[2]

    uw_proj = _jit_normalize((k - j) - ((k - j) @ v) * v)
    vw_proj = _jit_normalize((k - j) - ((k - j) @ u) * u)

    alpha_uw = np.pi - np.arctan2(uw_proj @ u, uw_proj @ w)
    alpha_vw = np.pi - np.arctan2(vw_proj @ v, vw_proj @ w)

    return np.array([alpha_uw, alpha_vw])


@njit(parallel=True, cache=True, nogil=True)
def _jit_x_to_ric(
    cart_positions: Matrix[float64], internal_coords_idx: Matrix[int64]
) -> tuple[Vector[float64], Matrix[int64]]:
    """Jit-compiled conversion between cartesian coordinates and internal coordinates

    .. note:: This function implicitly assumes that `internal_coords_idx`
        is for a 0-indexed molecule.

    Args:
        cart_positions: array of cartesian coordinate locations of the
            atoms in the Cartesian
        internal_coords_idx: array of internal coordinates, followed by the
            length of the coordinate. If the coordinate is not a dihedral, there are
            unused numbers in the 4th, or 3rd and 4th, places to ensure a
            rectangular array

    Returns:
        Array of internal coordinate values
    """
    internal_coordinates = np.empty(len(internal_coords_idx))
    bad_coordinates = np.zeros((len(internal_coords_idx), 5), dtype=int64)

    for i in prange(len(internal_coords_idx)):  # type: ignore[attr-defined]
        # get ith internal coordinate
        coord = internal_coords_idx[i]

        # separate cases for distances, angles, and dihedrals

        # distances
        if _is_bond_array(coord):
            # get positions of participating atoms
            positions = cart_positions[coord[:2]]

            u = positions[1] - positions[0]
            internal_coordinates[i] = norm(u)

        # angles
        elif _is_angle_array(coord):
            # get positions of participating atoms
            positions = cart_positions[coord[:3]]

            # vectors making up the angle
            normedu = _jit_normalize(positions[0] - positions[1])
            normedv = _jit_normalize(positions[2] - positions[1])

            internal_coordinates[i] = np.arccos(normedu @ normedv)

        # dihedrals
        elif _is_dihedral_array(coord):
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
                bad_coordinates[i, :-1] = coord[:-1]
                bad_coordinates[i, -1] = True

        elif _is_uw_bending_array(coord):
            internal_coordinates[i] = _jit_x_to_plane_coords_nonlinear(
                cart_positions, coord[:4]
            )[0]
        elif _is_vw_bending_array(coord):
            internal_coordinates[i] = _jit_x_to_plane_coords_nonlinear(
                cart_positions, coord[:4]
            )[1]

    return internal_coordinates, bad_coordinates


def _correct_dihedral_idx(
    struct: Cartesian,
    bad_idx: tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx],
    old_idx: Primitives,
    which_half: WhichHalf,
    tol: float = 5,
) -> tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx]:
    """Finds a suitable redefinition of a given linear dihedral coordinate

    Args:
        struct: Cartesian containing the given dihedral
        bad_idx: tuple containing the dihedral to be
            replaced
        old_idx: SortedSet of primitive coordinates pre-cleaning
        which_half: represents which half of the dihedral is linear
        tol: default 5, tolerance for collinear atom detection, in degrees.
    Returns:
        Tuple of the new dihedral
    """

    # to check that new index is not already used

    if which_half == WhichHalf.first:
        distances = [
            (i, struct.get_distance_to(bad_idx[0]).loc[i, "distance"])
            for i in struct.index
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
                    (tol < angle * 180 / np.pi < (180 - tol))  # fmt: skip
                    and tuple([index] + list(bad_idx[1:])) not in old_idx
                ):
                    return tuple([index] + list(bad_idx[1:]))
        raise RuntimeError(
            f"No suitable redifinition of poorly-defined dihedral {bad_idx}"
        )

    else:
        distances = [
            (i, struct.get_distance_to(bad_idx[3]).loc[i, "distance"])
            for i in struct.index
        ]
        distances.sort(key=lambda x: x[1])
        for index, _ in distances:
            if index not in bad_idx:
                new_positions = struct.loc[list(bad_idx[1:-1]) + [index], COORDS].values

                # vectors making up the angle
                normedu = _jit_normalize(new_positions[0] - new_positions[1])
                normedv = _jit_normalize(new_positions[2] - new_positions[1])

                angle = np.arccos(normedu @ normedv)

                # NOTE: is the check that it is not in the old_idx neccessary? If it is,
                # then the new one is fully redundant.
                if (
                    (tol < angle * 180 / np.pi < (180 - tol))  # fmt: skip
                    and tuple(list(bad_idx[:-1]) + [index]) not in old_idx
                ):
                    return tuple(list(bad_idx[:-1]) + [index])
        raise RuntimeError(
            f"No suitable redifinition of poorly-defined dihedral {bad_idx}"
        )


def _reindex_to_0_inner(
    coordinate_idx: Coordinate, index_to_rownum: dict
) -> Coordinate:
    if _is_bending_tuple(coordinate_idx):
        return tuple(
            [index_to_rownum[index] for index in coordinate_idx[:-1]]
            + [coordinate_idx[-1]]
        )
    else:
        return tuple(index_to_rownum[index] for index in coordinate_idx)


@njit(cache=True)
def _is_bond_array(coord: Vector) -> bool:
    return coord[-1] == 2


@njit(cache=True)
def _is_angle_array(coord: Vector) -> bool:
    return coord[-1] == 3


@njit(cache=True)
def _is_dihedral_array(coord: Vector) -> bool:
    return coord[-1] == 4


@njit(cache=True)
def _is_uw_bending_array(coord: Vector) -> bool:
    return coord[-1] == 5


@njit(cache=True)
def _is_vw_bending_array(coord: Vector) -> bool:
    return coord[-1] == 6


@njit(cache=True)
def _is_uw_bending_tuple(coord: Coordinate) -> bool:
    return len(coord) == 5 and coord[-1] == BendType.UW


@njit(cache=True)
def _is_vw_bending_tuple(coord: Coordinate) -> bool:
    return len(coord) == 5 and coord[-1] == BendType.VW


@njit(cache=True)
def _is_bending_tuple(coord: Coordinate) -> bool:
    return _is_uw_bending_tuple(coord) or _is_vw_bending_tuple(coord)


@njit(cache=True)
def _is_dihedral_tuple(coord: Coordinate) -> bool:
    return len(coord) == 4
