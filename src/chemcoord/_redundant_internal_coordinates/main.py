from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Final, Literal, Mapping, TypeAlias, cast, overload
from warnings import warn

import numpy as np
from attrs import define, field
from joblib import Parallel, delayed
from numpy import float64
from scipy.sparse import diags_array
from typing_extensions import Self, assert_never

from chemcoord._cartesian_coordinates._cartesian_class_bmat import (
    BendType,
    Primitives,
)
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord._redundant_internal_coordinates._backtransformation import (
    backtransform,
)
from chemcoord.configuration import settings
from chemcoord.exceptions import (
    ConvergenceError,
    PhysicalMeaning,
    UndefinedDihedral,
)
from chemcoord.typing import (
    ArithmeticOther,
    AtomIdx,
    BondDict,
    Matrix,
    Real,
    Vector,
)

Coordinate: TypeAlias = (
    tuple[AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx, BendType]
)


@define(frozen=True)
class DefaultWeights:
    """Default weights for the cost function in the weighted least-squares."""

    #: The bond length weighting
    bond: float = 1.0
    angle: float = 0.1
    dihedral: float = 0.05
    bending: float = 0.01

    def get_weight(self, coord: Coordinate) -> float:
        if _is_bond(coord):
            return self.bond
        elif _is_angle(coord):
            return self.angle
        elif _is_dihedral(coord):
            return self.dihedral
        elif _is_bending(coord):
            return self.bending
        else:
            raise ValueError("Invalid coordinate.")


@define(frozen=True)
class RedundantInternalCoordinates:
    q: Vector[float64]
    primitives_idx: Final[Primitives]

    #: The reference is an example cartesian for which the redundant
    #: internal coordinates could be defined.
    #: This is relevant as
    #: 1. starting guess and
    #: 2. to keep track of the index of the molecule.
    reference: Cartesian

    coord_to_idx: Final[Mapping[Coordinate, int]] = field(init=False)

    #: The coordinates set inside a :meth:`prioritize_manual_changes` block and their
    #: weight, or :class:`None` outside of such a block.
    _priority: tuple[set[Coordinate], float] | None = field(
        init=False, default=None, eq=False, repr=False
    )

    @coord_to_idx.default
    def _get_coord_to_idx(self) -> Mapping[Coordinate, int]:
        return dict(zip(self.primitives_idx, range(len(self.primitives_idx))))

    def copy(self) -> Self:
        return self.__class__(
            self.q.copy(),
            self.primitives_idx.copy(),
            self.reference.copy(),
        )

    @contextmanager
    def prioritize_manual_changes(self, weight: float = 100.0) -> Iterator[Self]:
        """Weight the coordinates set within the block more strongly.

        Every coordinate assigned via ``q[...] = ...`` inside the block is passed as
        ``prioritized`` with ``weight`` to each :meth:`get_cartesian` call inside the
        block, so the change is enforced and the rest of the molecule adjusts.

        .. code-block:: python

            q = molecule.get_ric()
            with q.prioritize_manual_changes():
                q[(0, 1, 2, 3)] = 0.0
                new = q.get_cartesian()

        Only assignments inside the block count. Copies of ``self`` made inside the
        block, e.g. by :meth:`copy` or arithmetic, are not tracked.

        Args:
            weight: default 100, weight of the set coordinates.

        Raises:
            RuntimeError: If blocks are nested.
        """
        if self._priority is not None:
            raise RuntimeError("prioritize_manual_changes() blocks cannot be nested.")
        object.__setattr__(self, "_priority", (set(), weight))
        try:
            yield self
        finally:
            object.__setattr__(self, "_priority", None)

    def __sub__(self, other: Self) -> DeltaRedundantInternalCoordinates:
        if self.primitives_idx != other.primitives_idx:
            raise ValueError("Can only add q with the same primitive indices")
        return DeltaRedundantInternalCoordinates(
            self.q - other.q,  # type: ignore[arg-type]
            self.primitives_idx,
            self.reference,
        )

    def __add__(
        self, other: DeltaRedundantInternalCoordinates
    ) -> RedundantInternalCoordinates:
        if self.primitives_idx != other.primitives_idx:
            raise ValueError("Can only add q with the same primitive indices")
        return RedundantInternalCoordinates(
            self.q + other.delta_q,  # type: ignore[arg-type]
            self.primitives_idx,
            self.reference,
        )

    @overload
    def __getitem__(self, key: Coordinate) -> float64: ...

    @overload
    def __getitem__(self, key: Sequence[Coordinate]) -> Vector[float64]: ...

    def __getitem__(
        self, key: Coordinate | Sequence[Coordinate]
    ) -> float64 | Vector[float64]:
        if isinstance(key[0], int):
            return self.q[self.coord_to_idx[_correct_order(key)]]  # type: ignore[index,arg-type]
        else:
            return self.q[[self.coord_to_idx[_correct_order(coord)] for coord in key]]  # type: ignore[index,return-value,arg-type]

    @overload
    def __setitem__(self, key: Coordinate, value: Real) -> None: ...

    @overload
    def __setitem__(
        self, key: Sequence[Coordinate], value: Vector[np.floating] | Sequence[Real]
    ) -> None: ...

    def __setitem__(
        self,
        key: Coordinate | Sequence[Coordinate],
        value: Real | Vector[np.floating] | Sequence[Real],
    ) -> None:
        # checking if key is one coord, or multiple
        if isinstance(key[0], int):
            self.q[self.coord_to_idx[_correct_order(key)]] = value  # type: ignore[index,arg-type]
            keys: Sequence[Coordinate] = [key]  # type: ignore[list-item]
        else:
            assert not isinstance(value, int)
            self.q[[self.coord_to_idx[_correct_order(coord)] for coord in key]] = value  # type: ignore[arg-type]
            keys = key  # type: ignore[assignment]
        if self._priority is not None:
            self._priority[0].update(_correct_order(coord) for coord in keys)

    def get_cartesian(
        self,
        *,
        start_guess: Cartesian | None = None,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        max_iter: int = 100,
        opt_alg: Literal["LM", "gauss"] = "LM",
        weights: Vector[np.floating] | Sequence[float] | None = None,
        default_weights: DefaultWeights | Mapping[str, float] | None = None,
        prioritized: Iterable[Coordinate] = (),
        priority_weight: float | None = None,
    ) -> Cartesian:
        """Finds the closest physical structure to self. Uses an iterative algorithm
        with Wilson's B matrix to converge to said structure.

        Args:
            start_guess: default :class:`None`, starting guess for the
                physical structure. If :class:`None` is given, uses self.reference
            rtol: default 1e-5, relative tolerance for convergence
            atol: default 1e-8, absolute tolerance for convergence
            max_iter: default 100, maximum allowed iterations for convergence
            opt_alg: default 'LM', either Levenberg-Marquardt or Gauss-Newton, the
                optimization algorithm used to generate :class:`~chemcoord.Cartesian`
                representations via :meth:`~.RedundantInternalCoordinates.get_cartesian`
            weights: default :class:`None`, weights used for each internal coordinate in
                the weighted least-squares step. A higher value means that that
                coordinate will be more likely to change linearly. Using values far
                above 1 can cause instability
            default_weights: default
                {"bond": 1.0, "angle": 0.1, "dihedral": 0.05, "bending": 0.01},
                the weights which each type of coordinate default to
            prioritized: default (), coordinates whose weight is set to
                ``priority_weight``, on top of ``weights`` or ``default_weights``.
                Use it to enforce values set for these coordinates while the rest of
                the molecule adjusts. The deviation from the set values decreases
                inversely with the weight. Inside a
                :meth:`prioritize_manual_changes` block, the coordinates set in the
                block are added.
            priority_weight: default :class:`None`, the weight of ``prioritized``.
                If :class:`None`, the weight of the enclosing
                :meth:`prioritize_manual_changes` block, or else 100, is used.
                Weights of about 1e4 and above slow down or prevent convergence.
        Returns:
            Closest physical structure to self, aligned to start_guess

        Raises:
            KeyError: If a coordinate in ``prioritized`` is not in
                ``self.primitives_idx``.

        Raises:
            ~chemcoord.exceptions.ConvergenceError: If the back-transformation does not
                converge within ``max_iter`` iterations.
        """

        if start_guess is None:
            start_guess = self.reference
        elif set(start_guess.index) != set(self.reference.index):
            raise ValueError(
                "The start guess has to be indexed in the same way as self.reference"
            )
        start_guess = start_guess.loc[self.reference.index, :]

        if weights is not None and default_weights is not None:
            raise ValueError("weights and default_weights cannot both be defined.")
        elif weights is None:
            if default_weights is None:
                default_weights = DefaultWeights()
            elif isinstance(default_weights, Mapping):
                default_weights = DefaultWeights(**default_weights)
            weights = cast(
                Vector[np.float64],
                np.array(
                    [default_weights.get_weight(coord) for coord in self.primitives_idx]
                ),
            )
        else:
            assert weights is not None

        weights = self._apply_priority(weights, prioritized, priority_weight)
        W = diags_array(weights)

        new = backtransform(
            self,
            start_guess,
            W,
            max_iter=max_iter,
            rtol=rtol,
            atol=atol,
            opt_alg=opt_alg,
        )
        return start_guess.align(new)[1] + start_guess.get_centroid()

    def _apply_priority(
        self,
        weights: Vector[np.floating] | Sequence[float],
        prioritized: Iterable[Coordinate],
        priority_weight: float | None,
    ) -> Vector[np.float64]:
        """``weights`` with ``prioritized``, and the coordinates set in an enclosing
        :meth:`prioritize_manual_changes` block, set to the priority weight."""
        coords = {_correct_order(coord) for coord in prioritized}
        block_weight = None
        if self._priority is not None:
            coords |= self._priority[0]
            block_weight = self._priority[1]
        result = np.array(weights, dtype=np.float64)
        if not coords:
            return result
        if priority_weight is None:
            priority_weight = 100.0 if block_weight is None else block_weight
        for coord in coords:
            try:
                result[self.coord_to_idx[coord]] = priority_weight
            except KeyError:
                raise KeyError(
                    f"{coord} is not one of the primitive internal coordinates."
                ) from None
        return result

    def minimize_dihedral(self) -> Self:
        """Reduces dihedral coordinates to the shorter angle, i.e., an angle of 3 pi / 2
        becomes an angle of -pi / 2

        Args:

        Returns:
            Copy of self with reduced dihedral coordinate values
        """
        cleaned_vals = np.array(
            [
                coord_val
                if len(idx) != 4
                else np.mod(coord_val + np.pi, 2 * np.pi) - np.pi
                for idx, coord_val in zip(self.primitives_idx, self.q)
            ]
        )
        full_cleaned = self.q.copy()

        full_cleaned[: len(cleaned_vals)] = cleaned_vals

        return self.__class__(full_cleaned, self.primitives_idx, self.reference)  # type: ignore[arg-type]


@define
class DeltaRedundantInternalCoordinates:
    delta_q: Vector[float64]
    primitives_idx: Final[Primitives]

    #: The reference is an example cartesian for which the redundant
    #: internal coordinates could be defined.
    #: This is relevant as
    #: 1. starting guess and
    #: 2. to keep track of the index of the molecule.
    reference: Final[Cartesian]

    coord_to_idx: Final[Mapping[Coordinate, int]] = field(init=False)

    @coord_to_idx.default
    def _get_coord_to_idx(self) -> dict[Coordinate, int]:
        return dict(zip(self.primitives_idx, range(len(self.primitives_idx))))

    def copy(self) -> Self:
        return self.__class__(
            self.delta_q.copy(),
            self.primitives_idx.copy(),
            self.reference.copy(),
        )

    def __mul__(self, other: ArithmeticOther) -> Self:
        new = self.copy()
        new.delta_q = new.delta_q * other
        return new

    def __rmul__(self, other: ArithmeticOther) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: ArithmeticOther) -> Self:
        new = self.copy()
        new.delta_q = new.delta_q / other
        return new

    @overload
    def __getitem__(self, key: Coordinate) -> float64: ...

    @overload
    def __getitem__(self, key: Sequence[Coordinate]) -> Vector[float64]: ...

    def __getitem__(
        self, key: Coordinate | Sequence[Coordinate]
    ) -> float64 | Vector[float64]:
        if isinstance(key[0], int):
            return self.delta_q[self.coord_to_idx[_correct_order(key)]]  # type: ignore[index,arg-type]
        else:
            return self.delta_q[
                [self.coord_to_idx[_correct_order(coord)] for coord in key]  # type: ignore[arg-type,return-value]
            ]

    @overload
    def __setitem__(self, key: Coordinate, value: Real) -> None: ...

    @overload
    def __setitem__(
        self, key: Sequence[Coordinate], value: Vector[np.floating] | Sequence[Real]
    ) -> None: ...

    def __setitem__(
        self,
        key: Coordinate | Sequence[Coordinate],
        value: Real | Vector[np.floating] | Sequence[Real],
    ) -> None:
        # checking if key is one coord, or multiple
        if isinstance(key[0], int):
            self.delta_q[self.coord_to_idx[_correct_order(key)]] = value  # type: ignore[index,arg-type]
        else:
            assert not isinstance(value, int)
            self.delta_q[
                [self.coord_to_idx[_correct_order(coord)] for coord in key]  # type: ignore[arg-type]
            ] = value

    def minimize_dihedral(self) -> Self:
        """Reduces deltas of dihedral coordinates and bending coordinates to the shorter
        rotation, i.e., a rotation of 3 pi / 2 becomes a rotation of -pi / 2

        Args:

        Returns:
            Copy of self with reduced dihedral and bending coordinate deltas
        """
        cleaned_vals = np.array(
            [
                coord_val
                if len(idx) != 4
                else np.mod(coord_val + np.pi, 2 * np.pi) - np.pi
                for idx, coord_val in zip(self.primitives_idx, self.delta_q)
            ]
        )

        return self.__class__(cleaned_vals, self.primitives_idx, self.reference)  # type: ignore[arg-type]


def get_primitives_idx(
    start: Cartesian,
    end: Cartesian,
    bonds: BondDict | None = None,
    linearity_thrshld: float = 5,
) -> Primitives:
    """Returns the set of primitive internal coordinates for a pair of start and end
    structures. Takes a union of the required sets for both, as start and end need to
    use the same coordinates. Takes care of linearities by adding linear
    bending coordinates.

    Args:
        start: starting structure
        end: ending structure
        bonds: default :class:`None`, optional specification of connectivity. If not
            specified, generated automatically
        linearity_thrshld: default 5, tolerance for linearity, in degrees
    Returns:
        tuple of regular redundant primitive internal coordinates and linear bending
        coordinates.
    """
    if bonds is None:
        bonds = _find_joint_bond_dict(start, end)

    start_and_end = start.get_primitives_idx(bonds=bonds) | end.get_primitives_idx(
        bonds=bonds
    )
    for linearity in start.linearities(
        start_and_end, tol=linearity_thrshld
    ) + end.linearities(start_and_end, tol=linearity_thrshld):
        # TODO no magic numbers for the 2
        ordered_lin = linearity[0] if linearity[1] == 2 else linearity[0][::-1]
        start_and_end.add(ordered_lin + (BendType.UW,))
        start_and_end.add(ordered_lin + (BendType.VW,))
        start_and_end.discard(linearity[0])
    return Primitives(start_and_end)


def _get_start_guess(
    start: Cartesian,
    end: Cartesian,
    N: int,
    seeds: Cartesian | Sequence[Cartesian] | None,
) -> list[Cartesian]:
    from chemcoord._cartesian_coordinates.xyz_functions import (  # noqa: PLC0415
        interpolate,
    )

    if seeds is None:
        try:
            return interpolate(start, end, N, coord="zmat")
        except (PhysicalMeaning, ValueError):
            # The Z-matrix interpolation can fail when the construction table has
            # invalid/linear references or the transformation is singular
            # (raised as ``PhysicalMeaning`` subclasses or ``ValueError``). In
            # that case fall back to plain cartesian interpolation.
            return interpolate(start, end, N, coord="cart")
    elif isinstance(seeds, Cartesian):
        return [seeds for _ in range(N)]
    else:
        return list(seeds)


def _find_joint_bond_dict(
    start: Cartesian, end: Cartesian
) -> dict[AtomIdx, set[AtomIdx]]:
    bonds_1, bonds_2 = start.get_bonds(), end.get_bonds()
    bonds = {
        k: bonds_1.get(k, set()) | bonds_2.get(k, set())
        for k in (bonds_1.keys() | bonds_2.keys())
    }
    for molecule in (start, end):
        for index1, index2 in molecule._fragment_connecting_bonds():
            bonds[index1].add(index2)
            bonds[index2].add(index1)
    return bonds


def RIC_interpolate(
    start: Cartesian,
    end: Cartesian,
    N: int,
    *,
    opt_alg: Literal["gauss", "LM"] = "LM",
    coord_idx: Primitives | None = None,
    max_iter: int = 500,
    seeds: Cartesian | Sequence[Cartesian] | None = None,
    bond_dict: BondDict | None = None,
    linearity_thrshld: float = 5,
    schedule: Literal[
        "automatic", "independent", "from_both", "from_start", "from_end"
    ] = "automatic",
    rtol: float = 1e-4,
    atol: float = 1e-8,
    weights: Vector[np.floating] | Sequence[float] | None = None,
    default_weights: DefaultWeights | Mapping[str, float] | None = None,
) -> list[Cartesian]:
    """Generates an N-image interpolation between start and end.

    Args:
        start: starting structure
        end: ending structure
        N: number of images, including start and end (so minimum 2)
        opt_alg: default 'LM', either Levenberg-Marquardt or Gauss-Newton, the
            optimization algorithm used to generate :class:`~chemcoord.Cartesian`
            representations of :class:`~.RedundantInternalCoordinates` via
            :meth:`~.RedundantInternalCoordinates.get_cartesian`
        coord_idx: default :class:`None`, optional specification of internal coordinate
            set to use
        max_iter: default 500, maximum number of steps for the
            :meth:`~.RedundantInternalCoordinates.get_cartesian` optimization cycle
        seeds: default :class:`None`, specifies the seed value for the
            :meth:`~.RedundantInternalCoordinates.get_cartesian` optimization cycle.
            Can be set to one :class:`~chemcoord.Cartesian`, which is used for each
            image, or to a sequence of :class:`~chemcoord.Cartesian` of length N.
            If it is :class:`None`, it uses appropiate method-dependent seeds, e.g. for
            ``"from_start"`` it uses the previous, converged solution.
        bond_dict: default :class:`None`, optional specification of connectivity. If not
            specified, generated automatically. NOTE: this connects disconnected
            fragments in both start and end with a bond between the closest two
            atoms in each fragment
        linearity_thrshld: default 5, tolerance for linearity, in degrees
        schedule: default "automatic", the scheduling to be used when generating the
            path. Can be "from_both" which builds it from the endpoints in,
            "from_start", or "from_end". "automatic" attempts each in that order,
            returning the first one to succeed
        rtol: default 1e-4, relative tolerance for convergence
        atol: default 1e-8, absolute tolerance for convergence
        weights: default :class:`None`, weights used for each internal coordinate in the
            weighted least-squares step. A higher value means that that coordinate
            will be more likely to change linearly. Using values far above 1 can cause
            instability
        default_weights: default
            {"bond": 1.0, "angle": 0.1, "dihedral": 0.05, "bending": 0.01},
            the weights which each type of coordinate default to

    Returns:
        The generated path as list of :class:`~chemcoord.Cartesian`.

    References:
        The algorithm is described in :cite:`whelpley_efficient_2026`.
        If you use this function, please cite it.
    """

    def to_cart(
        q: RedundantInternalCoordinates,
        seed: Cartesian,
    ) -> Cartesian:
        return q.get_cartesian(
            max_iter=max_iter,
            start_guess=seed,
            weights=weights,
            default_weights=default_weights,
            rtol=rtol,
            atol=atol,
            opt_alg=opt_alg,
        )

    if schedule == "independent":
        # A single repeated structure says nothing about which way a dihedral travels.
        seeds_are_a_path = not isinstance(seeds, Cartesian)
        seeds = _get_start_guess(start, end, N, seeds)

        if coord_idx is None:
            coord_idx = get_primitives_idx(
                start, end, bonds=bond_dict, linearity_thrshld=linearity_thrshld
            )

        return _RIC_interpolate_indpdt(
            start, end, N, coord_idx, to_cart, seeds, seeds_are_a_path
        )

    elif schedule == "from_both":
        return _RIC_interpolate_from_both(
            start,
            end,
            N,
            to_cart,
            linearity_thrshld,
            bond_dict,
            seeds,
        )

    elif schedule == "from_start":
        if coord_idx is None:
            coord_idx = get_primitives_idx(
                start, end, bonds=bond_dict, linearity_thrshld=linearity_thrshld
            )
        return _RIC_interpolate_from_start(start, end, N, coord_idx, to_cart, seeds)

    elif schedule == "from_end":
        # from_end is simply from_start but end and start are swapped.
        if coord_idx is None:
            coord_idx = get_primitives_idx(
                start, end, bonds=bond_dict, linearity_thrshld=linearity_thrshld
            )
        # The path is built end->start, so per-image seeds are reversed as well.
        inner_seeds = list(reversed(seeds)) if isinstance(seeds, Sequence) else seeds
        return list(
            reversed(
                _RIC_interpolate_from_start(
                    end, start, N, coord_idx, to_cart, inner_seeds
                )
            )
        )

    elif schedule == "automatic":
        AutoSchedules: TypeAlias = Literal[
            "independent", "from_both", "from_start", "from_end"
        ]

        def run_interpolate(
            auto_schedule: AutoSchedules,
        ) -> list[Cartesian]:
            return RIC_interpolate(
                start,
                end,
                N,
                opt_alg=opt_alg,
                coord_idx=coord_idx,
                weights=weights,
                max_iter=max_iter,
                seeds=seeds,
                bond_dict=bond_dict,
                linearity_thrshld=linearity_thrshld,
                schedule=auto_schedule,
                rtol=rtol,
                atol=atol,
            )

        strategies: Final[Sequence[AutoSchedules]] = [
            "independent",
            "from_both",
            "from_start",
            "from_end",
        ]
        for mode in strategies:
            try:
                return run_interpolate(mode)
            except (ConvergenceError, UndefinedDihedral):
                if mode != "from_end":
                    warn(f"{mode} scheduling failed; attempting next strategy")
        else:  # noqa: PLW0120
            raise RuntimeError("All scheduling strategies failed")

    else:
        assert_never(schedule)


RIC_ToCartesian: TypeAlias = Callable[
    [RedundantInternalCoordinates, Cartesian], Cartesian
]


def _wrap(angle: Vector[float64]) -> Vector[float64]:
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def _dihedral_angles(
    p0: Matrix[float64], p1: Matrix[float64], p2: Matrix[float64], p3: Matrix[float64]
) -> Vector[float64]:
    """Dihedral angles of the rows of ``p0``, ..., ``p3`` in the convention of
    :meth:`~chemcoord.Cartesian.get_ric`."""
    axis = p2 - p1
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    v = p0 - p1
    v -= np.sum(v * axis, axis=1)[:, None] * axis
    w = p3 - p2
    w -= np.sum(w * axis, axis=1)[:, None] * axis
    return np.arctan2(np.sum(np.cross(axis, v) * w, axis=1), np.sum(v * w, axis=1))


def _positions(molecule: Cartesian, atoms: Sequence[AtomIdx]) -> Matrix[float64]:
    xyz = np.asarray(molecule.loc[:, ["x", "y", "z"]], dtype=float)
    row = {atom: i for i, atom in enumerate(molecule.index)}
    return xyz[[row[atom] for atom in atoms]]


def _bond_angles(
    molecule: Cartesian,
    first: Sequence[AtomIdx],
    vertex: Sequence[AtomIdx],
    last: Sequence[AtomIdx],
) -> Vector[float64]:
    v, w = (
        _positions(molecule, atoms) - _positions(molecule, vertex)
        for atoms in (first, last)
    )
    cos = np.sum(v * w, axis=1) / (
        np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1)
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _dihedral_offsets(
    molecule: Cartesian,
    members: Sequence[tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx]],
    references: Sequence[tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx]],
) -> tuple[Vector[float64], Vector[float64]]:
    """The two contributions to ``τ(member) - τ(reference)`` of dihedrals about the
    same central bond ``b -> c``.

    They are the angles, viewed along the central bond, between the terminal atoms on
    ``b`` and between the terminal atoms on ``c``. Each is fixed up to its sign by the
    bond angles at that atom, so a rotation about the central bond leaves it unchanged.
    """
    a, b, c, d = (_positions(molecule, [m[k] for m in members]) for k in range(4))
    a_ref = _positions(molecule, [r[0] for r in references])
    d_ref = _positions(molecule, [r[3] for r in references])
    # A point attached to the other atom of the bond, pointing in the same direction
    # as the reference's terminal atom, has a dihedral of zero with it.
    on_b = _dihedral_angles(a, b, c, c + (a_ref - b))
    on_c = _dihedral_angles(b + (d_ref - c), b, c, d)
    return on_b, on_c


def _consistent_dihedral_branch(
    Δq: DeltaRedundantInternalCoordinates,
    start: Cartesian,
    end: Cartesian,
    preferred: Mapping[Coordinate, float] | None = None,
    off_axis: float = np.radians(20),
) -> DeltaRedundantInternalCoordinates:
    """Put the dihedrals of ``Δq`` on 2π branches that a motion of the atoms realises.

    ``minimize_dihedral`` takes the shortest arc of every dihedral independently. For
    dihedrals about the same central bond this can ask for rotations in opposite
    directions, e.g. between near mirror images, which no structure realises.
    Their differences are angles between bonds at the central atoms (see
    :func:`_dihedral_offsets`), whose changes are unambiguous. So within a group only a
    common number of turns is free. It is chosen closest to ``preferred``, e.g. the
    change along a seed path, or else as the smallest total rotation of the group.

    The angle between two bonds viewed along the central bond is only well defined if
    neither bond is close to collinear with it. Dihedrals with a terminal atom within
    ``off_axis`` of the axis, in start or end, are therefore not coupled to the others
    and keep their own branch. Rings, which couple dihedrals about different bonds, are
    not treated.
    """
    oriented: dict[int, tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx]] = {}
    for i, coord in enumerate(Δq.primitives_idx):
        if _is_dihedral(coord):
            a, b, c, d = coord  # type: ignore[misc]
            # τ(a, b, c, d) = τ(d, c, b, a), so orient every member along min -> max.
            oriented[i] = (a, b, c, d) if b < c else (d, c, b, a)
    if not oriented:
        return Δq

    well_defined = np.ones(len(oriented), dtype=bool)
    for molecule in (start, end):
        for first, second in ((0, 1), (3, 2)):
            angles = _bond_angles(
                molecule,
                [dihedral[first] for dihedral in oriented.values()],
                [dihedral[second] for dihedral in oriented.values()],
                [dihedral[3 - second] for dihedral in oriented.values()],
            )
            well_defined &= (off_axis < angles) & (angles < np.pi - off_axis)

    groups: dict[tuple[AtomIdx, ...], list[int]] = {}
    for (i, dihedral), coupled in zip(oriented.items(), well_defined):
        groups.setdefault(dihedral[1:3] if coupled else dihedral, []).append(i)

    members = [i for group in groups.values() for i in group]
    reference = {i: group[0] for group in groups.values() for i in group}
    changes = [
        _wrap(after - before)
        for before, after in zip(
            _dihedral_offsets(
                start,
                [oriented[i] for i in members],
                [oriented[reference[i]] for i in members],
            ),
            _dihedral_offsets(
                end,
                [oriented[i] for i in members],
                [oriented[reference[i]] for i in members],
            ),
        )
    ]
    relative = dict(zip(members, changes[0] + changes[1]))

    shortest = Δq.minimize_dihedral().delta_q
    new = Δq.copy()
    for group in groups.values():
        r = group[0]
        wanted = (
            None
            if preferred is None
            else np.array(
                [preferred.get(Δq.primitives_idx[i], np.nan) for i in group]  # type: ignore[arg-type]
            )
        )
        centre = (
            0
            if wanted is None or np.isnan(wanted[0])
            else round((wanted[0] - shortest[r]) / (2 * np.pi))
        )
        best, best_cost = None, np.inf
        for turns in (centre, centre - 1, centre + 1):
            Δτ_r = shortest[r] + 2 * np.pi * turns
            Δτ = np.array(
                [
                    shortest[i]
                    + 2
                    * np.pi
                    * round((Δτ_r + relative[i] - shortest[i]) / (2 * np.pi))
                    for i in group
                ]
            )
            cost = (
                np.sum(np.abs(Δτ)) if wanted is None else np.nansum(np.abs(Δτ - wanted))
            )
            if cost < best_cost - 1e-12:
                best, best_cost = Δτ, cost
        new.delta_q[group] = best
    return new


def _seed_path_change(
    coord_idx: Primitives, seeds: Sequence[Cartesian]
) -> dict[Coordinate, float] | None:
    """The change of every dihedral accumulated along the continuous seed path."""
    dihedrals = [coord for coord in coord_idx if _is_dihedral(coord)]
    try:
        trajectory = np.array(
            [seed.get_ric(coord_idx)[dihedrals] for seed in seeds]  # type: ignore[arg-type]
        )
    except UndefinedDihedral:
        # e.g. the cartesian fallback of the seeds passes through a linear geometry
        return None
    travelled = np.unwrap(trajectory, axis=0)[-1] - trajectory[0]
    return dict(zip(dihedrals, travelled))


def _as_preference(
    Δq: DeltaRedundantInternalCoordinates,
) -> dict[Coordinate, float]:
    return {
        coord: value
        for coord, value in zip(Δq.primitives_idx, Δq.delta_q)
        if _is_dihedral(coord)
    }


def _RIC_interpolate_indpdt(
    start: Cartesian,
    end: Cartesian,
    N: int,
    coord_idx: Primitives,
    to_cart: RIC_ToCartesian,
    seeds: Sequence[Cartesian],
    seeds_are_a_path: bool,
) -> list[Cartesian]:
    q1, q2 = start.get_ric(coord_idx), end.get_ric(coord_idx)
    Δq = _consistent_dihedral_branch(
        (q2 - q1).minimize_dihedral(),
        start,
        end,
        _seed_path_change(coord_idx, seeds) if seeds_are_a_path else None,
    )
    Qs = [q1 + i * Δq / (N - 1) for i in range(N)]

    return Parallel(n_jobs=settings.defaults.n_worker)(
        delayed(to_cart)(q, seed) for q, seed in zip(Qs, seeds)
    )


def _RIC_interpolate_from_start(
    start: Cartesian,
    end: Cartesian,
    N: int,
    coord_idx: Primitives,
    to_cart: RIC_ToCartesian,
    seeds: Cartesian | Sequence[Cartesian] | None,
) -> list[Cartesian]:
    q1 = start.get_ric(coord_idx)
    q2: Final = end.get_ric(coord_idx)

    path = [start]
    # The rest of the previous step's change, so the branch is not re-decided per image.
    remaining: dict[Coordinate, float] | None = None
    for i in range(1, N - 1):
        q1 = path[i - 1].get_ric(coord_idx)
        Δq = _consistent_dihedral_branch(
            (q2 - q1).minimize_dihedral(), path[i - 1], end, remaining
        )
        remaining = _as_preference(Δq * ((N - i - 1) / (N - i)))

        if seeds is None:
            seed = path[-1]
        elif isinstance(seeds, Sequence):
            seed = seeds[i]
        else:
            seed = seeds

        path.append(to_cart(q1 + Δq / (N - i), seed))

    path.append(end)

    return path


def _RIC_interpolate_from_both(
    start: Cartesian,
    end: Cartesian,
    N: int,
    to_cart: RIC_ToCartesian,
    linearity_thrshld: float,
    bond_dict: BondDict | None,
    seeds: Cartesian | Sequence[Cartesian] | None,
) -> list[Cartesian]:
    from_start, from_end = [start], [end]
    coord_idx = get_primitives_idx(
        start, end, bonds=bond_dict, linearity_thrshld=linearity_thrshld
    )

    # If there is an odd number of N we skip a final computation
    is_even: Final = (N + 1) % 2
    last_iter: Final = (N - 3) // 2

    remaining: dict[Coordinate, float] | None = None
    for i in range((N - 1) // 2):
        n_to_add = N - 2 * (i + 1)
        x1, x2 = from_start[-1], from_end[-1]

        coord_idx = get_primitives_idx(
            x1, x2, bonds=bond_dict, linearity_thrshld=linearity_thrshld
        )

        q1, q2 = x1.get_ric(coord_idx), x2.get_ric(coord_idx)
        Δq = _consistent_dihedral_branch(
            (q2 - q1).minimize_dihedral(), x1, x2, remaining
        )
        remaining = _as_preference(Δq * ((n_to_add - 1) / (n_to_add + 1)))

        if seeds is None:
            start_seed = from_start[-1]
            end_seed = from_end[-1]
        elif isinstance(seeds, Sequence):
            start_seed = seeds[i]
            end_seed = seeds[-(i + 1)]
        else:
            start_seed = seeds
            end_seed = seeds

        from_start.append(to_cart(q1 + Δq / (n_to_add + 1), start_seed))
        if is_even or i < last_iter:  # skip final from_end on odd N
            from_end.append(to_cart(q1 + Δq * n_to_add / (n_to_add + 1), end_seed))

    return from_start + list(reversed(from_end))


def _correct_order(coord: Coordinate) -> Coordinate:
    """Return coordinate tuples in the canonical order.

    .. python::
        (0, 1) -> (0, 1)
        (1, 0) -> (1, 0)
        (4, 3, 2, 1) -> (1, 2, 3, 4)
    """
    assert coord[0] != coord[-1]
    if coord[0] < coord[-1]:
        return coord
    else:
        return cast(Coordinate, tuple(reversed(coord)))


def _is_bond(idx: Coordinate) -> bool:
    return len(idx) == 2


def _is_angle(idx: Coordinate) -> bool:
    return len(idx) == 3


def _is_dihedral(idx: Coordinate) -> bool:
    return len(idx) == 4


def _is_bending(idx: Coordinate) -> bool:
    return len(idx) == 5
