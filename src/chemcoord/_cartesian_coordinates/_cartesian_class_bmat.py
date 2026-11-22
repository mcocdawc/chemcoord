from __future__ import annotations

from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING, Final, NewType, TypeAlias

import numpy as np
from numba import njit, prange
from numpy import cross, float64, int64
from numpy.linalg import norm
from scipy.sparse import csr_array

# had to put these here to avoid circular import
from sortedcontainers import SortedSet

from chemcoord._cartesian_coordinates._cart_transformation import (
    _jit_normalize,
)
from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import COORDS
from chemcoord.exceptions import (
    ERR_CODE_OK,
    CoincidingAtoms,
    ERR_CODE_CoincidingAtoms,
    ERR_CODE_UndefinedDihedral,
    UndefinedDihedral,
)
from chemcoord.typing import AtomIdx, BondDict, Matrix, Vector

if TYPE_CHECKING:
    from chemcoord._redundant_internal_coordinates.main import (
        RedundantInternalCoordinates,
    )


#: Nominal type for a set of primitive internal coordinates.
#: Unfortunately SortedSet is not a generic type, if it was, the primitives
#: would be declared as
#: ``SortedSet[tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]``
Primitives = NewType("Primitives", SortedSet)

#: Runtime constructor for :data:`Primitives`, whose ``NewType`` call is the identity.
#: The key groups the coordinates by type: bonds, angles, dihedrals and bendings.
_SetOfPrimitives = partial(SortedSet, key=lambda x: (len(x), x))


class BendType(IntEnum):
    """enum to differentiate the in and out-of-plane bending coordinates

    uw and vw planes are defined as in :cite:`in-out-of-plane-paper`"""

    UW = 0
    VW = 1


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
        coordinate, then lexicographically by atom index, i.e. grouped into
        bonds, then angles, then dihedrals and bendings.

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
            for index1, index2 in self._fragment_connecting_bonds(bond_dict):
                bond_dict[index1].add(index2)
                bond_dict[index2].add(index1)
        return self._get_primitives_single_molecule(bond_dict)

    def _get_primitives_single_molecule(
        self, bonds: BondDict | None = None
    ) -> Primitives:
        """This function calculated the primitive internal coordinates
        purely based on chemical connectivity and does not connect fragments."""
        idx_primitive_coords = []

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
                idx_primitive_coords.append(canonicalize(atom1, atom2))
                for atom3 in bonds[atom2] - {atom1}:
                    idx_primitive_coords.append(canonicalize(atom1, atom2, atom3))
                    for atom4 in bonds[atom3] - {atom1, atom2}:
                        idx_primitive_coords.append(
                            canonicalize(atom1, atom2, atom3, atom4)
                        )

        return Primitives(_SetOfPrimitives(idx_primitive_coords))

    def get_Wilson_B(
        self,
        idx_internal_coords: Primitives | None = None,
        bonds: BondDict | None = None,
        coord_arr: Matrix[int64] | None = None,
    ) -> Matrix:
        """Generate Wilson's B matrix for the current structure.

        Args:
            idx_internal_coords: default None, SortedSet of
                primitive internal coordinates to use in the calculation. If None,
                calculates using the get_primitive_coords method
            bonds: default :class:`None`, mapping containing bonding information.
            coord_arr: default :class:`None`, cached ``_to_array_nobending`` result.

        Returns:
            Wilson's B matrix
        """
        return self.get_sparse_Wilson_B(idx_internal_coords, bonds, coord_arr).toarray()

    def get_sparse_Wilson_B(
        self,
        idx_internal_coords: Primitives | None = None,
        bonds: BondDict | None = None,
        coord_arr: Matrix[int64] | None = None,
    ) -> csr_array:
        """Generate Wilson's B matrix as a sparse matrix for the current structure.

        Each row couples at most four atoms, so it has at most twelve nonzero entries.

        Args:
            idx_internal_coords: default None, SortedSet of
                primitive internal coordinates to use in the calculation. If None,
                calculates using the get_primitive_coords method
            bonds: default :class:`None`, mapping containing bonding information.
            coord_arr: default :class:`None`, cached ``_to_array_nobending`` result.

        Returns:
            Wilson's B matrix as a :class:`scipy.sparse.csr_array`
        """
        if idx_internal_coords is None:
            idx_internal_coords = self.get_primitives_idx(bonds)

        position_arr = self.loc[:, COORDS].values
        if coord_arr is None:
            coord_arr = self._to_array_nobending(idx_internal_coords)
        data, row, col, err_codes = _jit_get_Wilson_B_coo(position_arr, coord_arr)
        _raise_for_err_codes(idx_internal_coords, err_codes)
        keep = row.ravel() >= 0
        return csr_array(
            (data.ravel()[keep], (row.ravel()[keep], col.ravel()[keep])),
            shape=(len(coord_arr), position_arr.size),
        )

    def get_ric(
        self,
        internal_coords_idx: Primitives | None = None,
        bonds: BondDict | None = None,
        coord_arr: Matrix[int64] | None = None,
    ) -> RedundantInternalCoordinates:
        """Conversion to redundant internal coordinates

        Args:
            internal_coord_idx: default None, SortedSet of
                primitive coordinates to convert to. If None, calculates them using the
                get_primitive_coords method
            bonds: default :class:`None`, mapping containing bonding information.
            coord_arr: default :class:`None`, cached ``_to_array_full`` result.

        Returns:
            Redundant internal coordinate representation of self

        """
        from chemcoord._redundant_internal_coordinates.main import (  # noqa: PLC0415
            RedundantInternalCoordinates,
        )

        # get primitive coordinates
        if internal_coords_idx is None:
            internal_coords_idx = self.get_primitives_idx(bonds=bonds)

        if coord_arr is None:
            coord_arr = self._to_array_full(internal_coords_idx)
        ric_values, err_codes = _jit_x_to_ric(
            self.loc[:, COORDS].values,
            coord_arr,
        )
        _raise_for_err_codes(internal_coords_idx, err_codes)

        return RedundantInternalCoordinates(
            ric_values,
            internal_coords_idx,
            self.copy(),  # type: ignore[arg-type]
        )

    def _reindex_to_0(self, internal_coords_idx: Primitives) -> list[Coordinate]:
        """Return a reindexed version of `primitives` as if `self` was indexed
        contiguously from 0 to n - 1.

        A list in the order of ``internal_coords_idx``: sorting by the reindexed labels
        would misattribute the rows of ``q`` and B for an unsorted frame.
        """
        index_to_rownum = {index: row for row, index in enumerate(self.index)}

        return [
            _reindex_to_0_inner(coordinate_idx, index_to_rownum)
            for coordinate_idx in internal_coords_idx
        ]

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
        for index in coord_idx:
            if len(index) == 4:
                positions = self.loc[list(index), COORDS].values
                if _jit_is_collinear(positions[:3], tol):
                    linear_idx.append((index, 1))
                if _jit_is_collinear(positions[1:], tol):
                    linear_idx.append((index, 2))
        return linear_idx


#: A dihedral is undefined if one of its two bond angles is within this many
#: degrees of 0 or 180 degrees. Not to be confused with the larger ``tol`` of
#: :meth:`~CartesianBmat.linearities`, which decides where bending coordinates
#: replace dihedrals while they are still well defined.
_UNDEFINED_DIHEDRAL_TOL: Final = 0.25

#: Two consecutive atoms of an internal coordinate closer than this (in Angstrom)
#: coincide. This guards the normalisation of the bond vectors and is no chemical
#: criterion.
_COINCIDENCE_TOL: Final = 1e-4


@njit(cache=True, nogil=True)
def _jit_is_collinear(positions: Matrix, tol: float) -> bool:
    """Whether the angle of the three ``positions``, with the vertex in the middle,
    is within ``tol`` degrees of 0 or 180 degrees."""
    u = _jit_normalize(positions[0] - positions[1])
    v = _jit_normalize(positions[2] - positions[1])
    angle = np.degrees(np.arccos(u @ v))
    # For collinear atoms, rounding can push ``u @ v`` outside the domain of
    # ``arccos``, giving NaN. Then the comparison is False, i.e. collinear.
    return not (tol < angle < 180 - tol)


@njit(cache=True, nogil=True)
def _jit_coordinate_err_code(positions: Matrix, is_dihedral: bool) -> int:
    """Error code for the internal coordinate of the atom chain ``positions``.

    This is the single criterion used by both :func:`_jit_x_to_ric` and
    :func:`_jit_get_Wilson_B_coo`, so that a structure accepted by
    :meth:`~CartesianBmat.get_ric` also has a well-defined B matrix.
    """
    for k in range(len(positions) - 1):
        if norm(positions[k + 1] - positions[k]) <= _COINCIDENCE_TOL:
            return ERR_CODE_CoincidingAtoms
    if is_dihedral and (
        _jit_is_collinear(positions[:3], _UNDEFINED_DIHEDRAL_TOL)
        or _jit_is_collinear(positions[1:], _UNDEFINED_DIHEDRAL_TOL)
    ):
        return ERR_CODE_UndefinedDihedral
    return ERR_CODE_OK


@njit(cache=True, nogil=True)
def _jit_dihedral_deriv(positions: Matrix) -> Matrix:
    """Calculates Cartesian derivatives of a dihedral angle given 4 atom positions

    The dihedral has to be defined, see :func:`_jit_coordinate_err_code`.
    This function does not raise, because it is called from within the
    ``prange`` loop of :func:`_jit_get_Wilson_B_coo`, where exceptions are not
    propagated reliably by numba.
    https://github.com/numba/numba/issues/9673
    """

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

    s_first = cross(normedu, normedw) / (norm(u) * (sinu**2))
    s_last = -cross(normedv, normedw) / (norm(v) * (sinv**2))
    shared = (
        ((cross(normedu, normedw) * cosu) / (norm(w) * (sinu**2)))
        + ((cross(normedv, normedw) * cosv) / (norm(w) * (sinv**2)))
    )  # fmt: skip
    return np.stack((s_first, -s_first + shared, -s_last - shared, s_last))


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
def _jit_get_Wilson_B_coo(
    position_arr: Matrix[float64],
    internal_coord_arr: Matrix[int64],
) -> tuple[Matrix[float64], Matrix[int64], Matrix[int64], Vector[int64]]:
    """Wilson's B matrix as ``(data, row, col)`` arrays of shape ``(n_coords, 12)``,
    and an array of error codes of shape ``(n_coords,)``.

    procedure from: :cite:`B_matrix`. A coordinate couples at most four atoms, so a row
    has at most twelve nonzeros; unused slots keep ``row == -1``.

    ``err_codes[i]`` is the :func:`_jit_coordinate_err_code` of the i-th coordinate.
    If it is not :data:`~chemcoord.exceptions.ERR_CODE_OK`, the row is NaN and the
    caller has to raise, see :func:`_raise_for_err_codes`.
    Nothing is raised in here: numba does not reliably propagate exceptions out of
    ``prange`` loops; depending on the threading layer and the kernel they become a
    ``SystemError`` or are silently swallowed
    (https://github.com/numba/numba/issues/9673).
    """
    n_coords = len(internal_coord_arr)
    data = np.zeros((n_coords, 12), dtype=float64)
    row = np.full((n_coords, 12), -1, dtype=int64)
    col = np.zeros((n_coords, 12), dtype=int64)
    err_codes = np.full(n_coords, ERR_CODE_OK, dtype=int64)

    for i in prange(n_coords):  # type: ignore[attr-defined]
        coord = internal_coord_arr[i, :]
        if _is_bond_array(coord):
            atoms = coord[:2]
            err = _jit_coordinate_err_code(position_arr[atoms], False)
            if err == ERR_CODE_OK:
                normedu = _jit_normalize(
                    position_arr[atoms[0]] - position_arr[atoms[1]]
                )
                derivs = np.stack((normedu, -normedu))
            else:
                derivs = np.full((2, 3), np.nan)
        elif _is_angle_array(coord):
            atoms = coord[:3]
            err = _jit_coordinate_err_code(position_arr[atoms], False)
            if err == ERR_CODE_OK:
                derivs = _jit_angle_deriv(position_arr[atoms])
            else:
                derivs = np.full((3, 3), np.nan)
        elif _is_dihedral_array(coord):
            atoms = coord[:4]
            err = _jit_coordinate_err_code(position_arr[atoms], True)
            if err == ERR_CODE_OK:
                derivs = _jit_dihedral_deriv(position_arr[atoms])
            else:
                derivs = np.full((4, 3), np.nan)
        else:
            # The derivatives refer to the last three of the four atoms.
            atoms = coord[1:4]
            err = _jit_coordinate_err_code(position_arr[coord[:4]], False)
            if err == ERR_CODE_OK:
                # coord[:4] is a Matrix-typed slice; _jit_get_axes keeps its Vector
                # signature for its other (genuinely 1-D) caller. numba boundary.
                axes = _jit_get_axes(position_arr, coord[:4])  # type: ignore[arg-type]
                if _is_uw_bending_array(coord):
                    derivs = _jit_uw_deriv(position_arr[coord[:4]], axes)
                else:
                    derivs = _jit_vw_deriv(position_arr[coord[:4]], axes)
            else:
                derivs = np.full((3, 3), np.nan)
        err_codes[i] = err
        for a in range(len(atoms)):
            for j in range(3):
                row[i, 3 * a + j] = i
                col[i, 3 * a + j] = j + 3 * atoms[a]
                data[i, 3 * a + j] = derivs[a, j]

    return data, row, col, err_codes


# NOTE: no ``parallel=True`` here. This function operates on a single bending
# coordinate (fixed-size arrays) and has no ``prange`` loop, so parallelisation
# brings no benefit. More importantly, it is called from within the ``prange``
# loops of ``_jit_get_Wilson_B_coo`` and ``_jit_x_to_ric``; marking it
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
) -> tuple[Vector[float64], Vector[int64]]:
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
        Array of internal coordinate values, and the array of their error codes, see
        :func:`_jit_coordinate_err_code`. Values with an error code other than
        :data:`~chemcoord.exceptions.ERR_CODE_OK` are NaN. As in
        :func:`_jit_get_Wilson_B_coo`, nothing is raised inside the ``prange`` loop;
        the caller has to check the error codes, see :func:`_raise_for_err_codes`.
    """
    internal_coordinates = np.empty(len(internal_coords_idx))
    err_codes = np.full(len(internal_coords_idx), ERR_CODE_OK, dtype=int64)

    for i in prange(len(internal_coords_idx)):  # type: ignore[attr-defined]
        # get ith internal coordinate
        coord = internal_coords_idx[i]

        if _is_bond_array(coord):
            n_atoms = 2
        elif _is_angle_array(coord):
            n_atoms = 3
        else:
            # dihedrals, and bendings, whose chain h-i-j-k spans four atoms
            n_atoms = 4
        err = _jit_coordinate_err_code(
            cart_positions[coord[:n_atoms]], _is_dihedral_array(coord)
        )
        err_codes[i] = err

        if err != ERR_CODE_OK:
            internal_coordinates[i] = np.nan

        # distances
        elif _is_bond_array(coord):
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

            # Rounding can push exactly linear angles outside the ``arccos`` domain.
            cos_angle = float(normedu @ normedv)
            internal_coordinates[i] = np.arccos(min(max(cos_angle, -1.0), 1.0))

        # dihedrals
        elif _is_dihedral_array(coord):
            # get positions of participating atoms
            positions = cart_positions[coord[:4]]

            # vectors making up dihedral
            normedu = _jit_normalize(positions[1] - positions[0])
            normedw = _jit_normalize(positions[2] - positions[1])
            normedv = _jit_normalize(positions[3] - positions[2])

            x = cross(cross(normedu, normedw), cross(normedw, normedv)) @ normedw
            y = cross(normedu, normedw) @ cross(normedw, normedv)
            internal_coordinates[i] = np.arctan2(x, y)

        elif _is_uw_bending_array(coord):
            internal_coordinates[i] = _jit_x_to_plane_coords_nonlinear(
                cart_positions, coord[:4]
            )[0]
        elif _is_vw_bending_array(coord):
            internal_coordinates[i] = _jit_x_to_plane_coords_nonlinear(
                cart_positions, coord[:4]
            )[1]

    return internal_coordinates, err_codes


def _raise_for_err_codes(
    internal_coords_idx: Primitives, err_codes: Vector[int64]
) -> None:
    """Raise the exception belonging to the error codes of the jitted kernels.

    The coordinates are reported with the index of the molecule, i.e. as they
    appear in ``internal_coords_idx``. Coinciding atoms take precedence, because
    they make every coordinate they are part of meaningless.

    Raises:
        ~chemcoord.exceptions.CoincidingAtoms: for
            :data:`~chemcoord.exceptions.ERR_CODE_CoincidingAtoms`.
        ~chemcoord.exceptions.UndefinedDihedral: for
            :data:`~chemcoord.exceptions.ERR_CODE_UndefinedDihedral`.
        RuntimeError: for any other error code, which would be a bug.
    """
    if (err_codes == ERR_CODE_OK).all():
        return

    def affected(code: int) -> list[Coordinate]:
        return [internal_coords_idx[i] for i in np.flatnonzero(err_codes == code)]

    if coinciding := affected(ERR_CODE_CoincidingAtoms):
        raise CoincidingAtoms(coinciding)
    if undefined := affected(ERR_CODE_UndefinedDihedral):
        raise UndefinedDihedral(undefined)
    # Not an ``assert``: an error code without exception is a bug in chemcoord.
    raise RuntimeError(f"Unknown error codes {set(err_codes.tolist())}")


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


# NOTE: ``coord`` is a row of a 2-D ``internal_coord_arr`` (``Matrix``). numpy
# does not narrow ``arr[i, :]`` to a 1-D ``Vector``, and ``np.ndarray`` is
# invariant in its shape parameter, so these predicates annotate ``Matrix`` to
# match what callers actually pass.
@njit(cache=True)
def _is_bond_array(coord: Matrix) -> bool:
    return coord[-1] == 2


@njit(cache=True)
def _is_angle_array(coord: Matrix) -> bool:
    return coord[-1] == 3


@njit(cache=True)
def _is_dihedral_array(coord: Matrix) -> bool:
    return coord[-1] == 4


@njit(cache=True)
def _is_uw_bending_array(coord: Matrix) -> bool:
    return coord[-1] == 5


@njit(cache=True)
def _is_vw_bending_array(coord: Matrix) -> bool:
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
