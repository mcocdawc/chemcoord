from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from itertools import combinations
from typing import Final, Literal, Mapping, TypeAlias, cast, overload
from warnings import warn

import numpy as np
from attrs import define, field
from joblib import Parallel, delayed
from numpy import float64
from numpy.linalg import lstsq, norm
from sortedcontainers import SortedSet
from typing_extensions import Self, assert_never

from chemcoord._cartesian_coordinates._cartesian_class_bmat import BendType
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord.configuration import settings
from chemcoord.exceptions import UndefinedDihedral
from chemcoord.typing import ArithmeticOther, AtomIdx, BondDict, Matrix, Real, Vector

Coordinate: TypeAlias = (
    tuple[AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx, BendType]
)

#: Unfortunately SortedSet is not a generic type, if it was, the primitives
#: would be declared as
#: ``SortedSet[tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]``
Primitives: TypeAlias = SortedSet

# the key prioritizes length, then sorts lexicographically
SetOfPrimitives = partial(SortedSet, key=lambda x: (len(x), x))


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

    @coord_to_idx.default
    def _get_coord_to_idx(self) -> Mapping[Coordinate, int]:
        return dict(zip(self.primitives_idx, range(len(self.primitives_idx))))

    def copy(self) -> Self:
        return self.__class__(
            self.q.copy(),
            self.primitives_idx.copy(),
            self.reference.copy(),
        )

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
        else:
            assert not isinstance(value, int)
            self.q[[self.coord_to_idx[_correct_order(coord)] for coord in key]] = value  # type: ignore[arg-type]

    def _lambda_cycle(
        self,
        previous: Cartesian,
        B: Matrix,
        W: Matrix,
        start_lam: float,
        nu: float,
        reduction_factor: float,
        Δq: DeltaRedundantInternalCoordinates,
    ) -> tuple[Cartesian, float]:
        """This gets the best choice of lambda for a Levenberg-Marquardt optimization
        step.

        see: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm"""

        good_lam = False

        D = np.diag(B.T @ W @ W @ B) * np.eye(len(B[0]))

        lm_mat = np.vstack((W @ B, np.sqrt(start_lam) * D))
        lm_vec = np.hstack((W @ Δq.delta_q, np.zeros(len(B[0]))))

        Δx = lstsq(lm_mat, lm_vec, rcond=-1)[0][: 3 * len(self.reference)]
        Δx = Δx.reshape(len(previous), 3)
        new = previous + Δx
        new_Δq = (
            self - new.get_ric(internal_coords_idx=self.primitives_idx)
        ).minimize_dihedral()
        if norm(new_Δq.delta_q) <= norm(Δq.delta_q):
            lam = start_lam
            good_lam = True
        else:
            lam = start_lam / reduction_factor

        if not good_lam:
            lm_mat = np.vstack((W @ B, np.sqrt(lam) * D))
            lm_vec = np.hstack((W @ Δq.delta_q, np.zeros(len(B[0]))))

            Δx = lstsq(lm_mat, lm_vec, rcond=-1)[0][: 3 * len(self.reference)]
            Δx = Δx.reshape(len(previous), 3)
            new = previous + Δx
            new_Δq = (
                self - new.get_ric(internal_coords_idx=self.primitives_idx)
            ).minimize_dihedral()
            if norm(new_Δq.delta_q) <= norm(Δq.delta_q):
                good_lam = True
            else:
                lam *= nu**2
                while not good_lam:
                    lm_mat = np.vstack((W @ B, np.sqrt(lam) * D))
                    lm_vec = np.hstack((W @ Δq.delta_q, np.zeros(len(B[0]))))

                    Δx = lstsq(lm_mat, lm_vec, rcond=-1)[0][: 3 * len(self.reference)]
                    Δx = Δx.reshape(len(previous), 3)
                    new = previous + Δx
                    new_Δq = (
                        self - new.get_ric(internal_coords_idx=self.primitives_idx)
                    ).minimize_dihedral()
                    if norm(new_Δq.delta_q) <= norm(Δq.delta_q):
                        good_lam = True
                    else:
                        lam *= nu

        return new, lam

    def _gauss_newton_opt(
        self, start_guess: Cartesian, max_iter: int, W: Matrix, rtol: float, atol: float
    ) -> Cartesian:
        from chemcoord._cartesian_coordinates.xyz_functions import (  # noqa: PLC0415
            allclose,
        )

        previous = start_guess

        converged = False
        i = 0
        while not converged:
            if (i := i + 1) > max_iter:
                raise ValueError(f"Not converged after {max_iter} iterations.")

            B = previous.get_Wilson_B(idx_internal_coords=self.primitives_idx)

            q_current = previous.get_ric(internal_coords_idx=self.primitives_idx)

            Δq = (self - q_current).minimize_dihedral()

            Δx = lstsq(W @ B, W @ Δq.delta_q, rcond=-1)[0]
            Δx = Δx.reshape(len(previous), 3)

            new = _linesearch(B, Δq.delta_q, Δx, self, previous)

            converged = allclose(
                new,
                previous,
                rtol=rtol,
                atol=atol,
                align=True,
            )
            previous = previous.align(new)[1]

        if i > 100:
            warn(f"The transformation to cartesian coordinates took {i} iterations.")

        return new

    def _levenberg_marquardt_opt(
        self,
        start_guess: Cartesian,
        max_iter: int,
        W: Matrix,
        rtol: float,
        atol: float,
        start_lam: float = 1e-5,
        nu: float = 1.5,
        reduction_factor: float = 10,
    ) -> Cartesian:
        from chemcoord._cartesian_coordinates.xyz_functions import (  # noqa: PLC0415
            allclose,
        )

        previous = start_guess

        converged = False
        i = 0

        lam = start_lam
        while not converged:
            assert previous is not None
            if (i := i + 1) > max_iter:
                raise ValueError(f"Not converged after {max_iter} iterations.")

            B = previous.get_Wilson_B(idx_internal_coords=self.primitives_idx)

            q_current = previous.get_ric(internal_coords_idx=self.primitives_idx)

            Δq = (self - q_current).minimize_dihedral()

            new, lam = self._lambda_cycle(previous, B, W, lam, nu, reduction_factor, Δq)

            converged = allclose(
                new,
                previous,
                rtol=rtol,
                atol=atol,
                align=True,
            )
            previous = previous.align(new)[1]

        if i > 100:
            warn(f"The transformation to cartesian coordinates took {i} iterations.")

        return new

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
                {"length" : 1.0, "angle" : 0.1, "dihedral" : 0.05, "bending" : 0.01},
                the weights which each type of coordinate default to
        Returns:
            Closest physical structure to self, aligned to start_guess
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

        W = np.diag(weights)  # type: ignore[arg-type]

        if opt_alg == "LM":
            new = self._levenberg_marquardt_opt(start_guess, max_iter, W, rtol, atol)
        elif opt_alg == "gauss":
            new = self._gauss_newton_opt(start_guess, max_iter, W, rtol, atol)
        else:
            assert_never(opt_alg)

        return start_guess.align(new)[1] + start_guess.get_centroid()

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
    return start_and_end


def _linesearch(
    B: Matrix,
    Δq: Vector,
    Δx: Vector,
    current: RedundantInternalCoordinates,
    previous: Cartesian,
    alpha: float = 1.0,
    c: float = 1e-4,
    tau: float = 0.5,
    max_iter: int = 100,
) -> Cartesian:
    # NOTE: alpha is a backtracking-line-search scalar
    # see: https://en.wikipedia.org/wiki/Backtracking_line_search
    too_far = True
    t = c * 2 * norm(B.T @ Δq)

    backstep = 0
    while too_far:
        new = previous + alpha * Δx
        q_new = new.get_ric(internal_coords_idx=current.primitives_idx)
        if norm(Δq) < alpha * t + norm((current - q_new).minimize_dihedral().delta_q):
            alpha *= tau
        else:
            too_far = False
        if backstep > max_iter:
            raise ValueError(f"Line search not terminated after {max_iter} iterations")

    return new


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
        except Exception:  # noqa: E722
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
    start_fragments = start.fragmentate()
    end_fragments = end.fragmentate()
    if len(start_fragments) != 1:
        for fragment_pair in combinations(start_fragments, 2):
            index1, index2, _ = fragment_pair[0].get_shortest_distance(fragment_pair[1])
            bonds[index1].add(index2)
            bonds[index2].add(index1)
    if len(end_fragments) != 1:
        for fragment_pair in combinations(end_fragments, 2):
            index1, index2, _ = fragment_pair[0].get_shortest_distance(fragment_pair[1])
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
            {"length" : 1.0, "angle" : 0.1, "dihedral" : 0.05, "bending" : 0.01},
            the weights which each type of coordinate default to

    Returns:
        The generated path as list of :class:`~chemcoord.Cartesian`.
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
        seeds = _get_start_guess(start, end, N, seeds)

        if coord_idx is None:
            coord_idx = get_primitives_idx(
                start, end, bonds=bond_dict, linearity_thrshld=linearity_thrshld
            )

        return _RIC_interpolate_indpdt(start, end, N, coord_idx, to_cart, seeds)

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
        return list(
            reversed(
                _RIC_interpolate_from_start(end, start, N, coord_idx, to_cart, seeds)
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
            except (ValueError, UndefinedDihedral):
                if mode != "from_end":
                    warn(f"{mode} scheduling failed; attempting next strategy")
        else:  # noqa: PLW0120
            raise RuntimeError("All scheduling strategies failed")

    else:
        assert_never(schedule)


RIC_ToCartesian: TypeAlias = Callable[
    [RedundantInternalCoordinates, Cartesian], Cartesian
]


def _RIC_interpolate_indpdt(
    start: Cartesian,
    end: Cartesian,
    N: int,
    coord_idx: Primitives,
    to_cart: RIC_ToCartesian,
    seeds: Sequence[Cartesian],
) -> list[Cartesian]:
    q1, q2 = start.get_ric(coord_idx), end.get_ric(coord_idx)
    Δq = (q2 - q1).minimize_dihedral()
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
    for i in range(1, N - 1):
        q1 = path[i - 1].get_ric(coord_idx)
        Δq = (q2 - q1).minimize_dihedral()

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

    for i in range((N - 1) // 2):
        n_to_add = N - 2 * (i + 1)
        x1, x2 = from_start[-1], from_end[-1]

        coord_idx = get_primitives_idx(
            x1, x2, bonds=bond_dict, linearity_thrshld=linearity_thrshld
        )

        q1, q2 = x1.get_ric(coord_idx), x2.get_ric(coord_idx)
        Δq = (q2 - q1).minimize_dihedral()

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
