from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, Literal, Mapping, TypeAlias, cast, overload
from warnings import warn

import numpy as np
from attrs import define, field
from joblib import Parallel, delayed
from numpy import float64
from numpy.linalg import norm
from scipy.sparse import csr_array, diags_array
from scipy.sparse import vstack as sparse_vstack
from scipy.sparse.linalg import lsmr
from typing_extensions import Self, assert_never

from chemcoord._cartesian_coordinates._cartesian_class_bmat import (
    BendType,
    Primitives,
)
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord.configuration import settings
from chemcoord.exceptions import PhysicalMeaning, UndefinedDihedral
from chemcoord.typing import ArithmeticOther, AtomIdx, BondDict, Matrix, Real, Vector

Coordinate: TypeAlias = (
    tuple[AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx]
    | tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx, BendType]
)

#: Which Levenberg-Marquardt step control the back-transformation uses.
#: ``"full_step"`` is the seed-stable full damped step (``x(q(x)) == x``) that can stall
#: on large, stiff systems; ``"line_search"`` is the robust backtracking variant that
#: scales but is not seed-stable on flat/degenerate minima; ``"auto"`` (default) runs
#: ``"full_step"`` first and switches to ``"line_search"`` if it has not converged after
#: :data:`_LM_AUTO_SWITCH_ITER` iterations.
LMStep: TypeAlias = Literal["auto", "full_step", "line_search"]

#: Type of a single LM step cycle (``_lambda_cycle`` / ``_full_step_cycle``): it maps
#: the current state to ``(new_cartesian, lam)``.
_LMCycle: TypeAlias = Callable[..., tuple["Cartesian", float]]


# Upper bound on the number of LSMR iterations per linear solve. The augmented
# Levenberg-Marquardt system is ill-conditioned, so without a cap LSMR chases the
# stopping tolerance for thousands of iterations. Since every solve is only one
# inexact step of the outer Gauss-Newton/LM loop, a bounded, approximate solution is
# enough and keeps each iteration cheap.
_LSTSQ_MAX_ITER: Final = 200

# Bounds for the Levenberg-Marquardt damping ``lam``. It is shrunk after every
# accepted step (drifting back toward fast Gauss-Newton) and only grown when the
# damped direction admits no descent, so these are loose safety rails rather than
# tuned values. ``_LM_MAX_DAMPING_STEPS`` caps how often ``lam`` may be grown within
# a single outer iteration before giving up.
_LM_MIN_LAMBDA: Final = 1e-14
_LM_MAX_LAMBDA: Final = 1e6
_LM_MAX_DAMPING_STEPS: Final = 30

# In the default ``lm_step="auto"`` back-transformation the seed-stable full-step LM is
# tried first and, if it has not converged after this many outer iterations, the solve
# switches to the robust (line-search) variant. The full step stalls on large, stiff
# systems (the residual only creeps down), so this bounds the wasted effort before the
# fallback takes over.
_LM_AUTO_SWITCH_ITER: Final = 50


def _sparse_lstsq(
    A: Matrix, b: Vector, atol: float = 1e-8, btol: float = 1e-8
) -> Vector[np.float64]:
    """Solve the least-squares problem ``min_x ||A x - b||`` exploiting the sparsity
    of the (banded) Wilson B matrix.

    Every internal coordinate involves at most four atoms, so each row of the Wilson
    B matrix (and of the Levenberg-Marquardt augmented system) has at most twelve
    nonzero entries irrespective of the system size. Converting to a compressed
    sparse row representation and using :func:`scipy.sparse.linalg.lsmr` is therefore
    considerably cheaper than a dense SVD-based solve for larger systems, while
    converging to the same minimum-norm least-squares solution. Started from the
    default ``x0 = 0``, LSMR yields the minimum-norm solution for the rank-deficient
    (rigid-body null space) Gauss-Newton system, matching :func:`numpy.linalg.lstsq`.

    LSMR is preferred over LSQR here because it is more robust on the ill-conditioned
    augmented LM system and typically reaches an equivalent solution in fewer
    iterations. The tolerances are ``1e-8`` (rather than machine precision): the outer
    loop only converges to ``rtol=1e-5``/``atol=1e-8``, so solving each linear
    subproblem to twelve digits is wasted work.
    """
    return lsmr(csr_array(A), b, atol=atol, btol=btol, maxiter=_LSTSQ_MAX_ITER)[0]


def _dense_lstsq(A: Matrix, b: Vector) -> Vector[np.float64]:
    """Dense counterpart of :func:`_sparse_lstsq`, kept so that the dense and sparse
    back-transformations can be run side by side (see ``coord="RIC_dense"`` in
    :func:`~chemcoord.xyz_functions.interpolate`).

    Uses :func:`numpy.linalg.lstsq` with ``rcond=-1`` on the dense
    :meth:`~chemcoord.Cartesian.get_Wilson_B` matrix. Started from the implicit zero
    guess it returns the minimum-norm least-squares solution for the rank-deficient
    (rigid-body null space) Gauss-Newton system, matching :func:`_sparse_lstsq`.
    """
    return np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=-1)[0]


def _cached_coord_arrays(
    structure: Cartesian, primitives_idx: Primitives
) -> tuple[Matrix, Matrix]:
    """Pre-compute the 0-based reindexed coordinate arrays for ``primitives_idx``.

    :meth:`~chemcoord.Cartesian.get_sparse_Wilson_B` /
    :meth:`~chemcoord.Cartesian.get_Wilson_B` (via ``_to_array_nobending``) and
    :meth:`~chemcoord.Cartesian.get_ric` (via ``_to_array_full``) otherwise rebuild
    these integer index arrays on every call -- the dominant cost of the
    back-transformation for large systems. Within an optimization loop the atom index
    order and the primitive set are constant, so the arrays can be computed once here
    and fed back in through the methods' ``coord_arr`` argument.

    Returns the ``(nobending, full)`` arrays, for the Wilson B and RIC calls
    respectively.
    """
    return (
        structure._to_array_nobending(primitives_idx),
        structure._to_array_full(primitives_idx),
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
        B: csr_array | Matrix,
        W: Matrix,
        start_lam: float,
        nu: float,
        reduction_factor: float,
        Δq: DeltaRedundantInternalCoordinates,
        sparse: bool = True,
        ric_coord_arr: Matrix | None = None,
    ) -> tuple[Cartesian, float]:
        """Take a single damped Gauss-Newton (Levenberg-Marquardt) step.

        The damped search direction for the current ``lam`` is refined with a
        backtracking line search (:func:`_linesearch`): an over-shooting full step is
        *shortened* rather than the direction being rotated toward gradient descent by
        an ever-growing ``lam``. Plain lambda-only acceptance stalls badly near the
        solution -- once ``lam`` ratchets up it never recovers and the residual only
        crawls, so on large systems the outer loop never converges within ``max_iter``
        -- whereas shortening the good Gauss-Newton direction keeps the fast
        convergence rate.

        ``lam`` is decreased after every accepted step (drifting back toward the fast
        Gauss-Newton regime) and only increased when even the damped direction admits
        no descent, which restores the regularisation that makes LM more robust than
        plain Gauss-Newton. The returned ``lam`` seeds the next outer iteration.

        see: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm"""

        # Invariants of the damped least-squares system, independent of lambda. In the
        # sparse path the Wilson B matrix is banded, so ``W @ B`` and the augmented
        # system stay sparse.
        WB = W @ B
        W_Δq = W @ Δq.delta_q
        zeros = np.zeros(B.shape[1])
        # diagonal of the (Gauss-Newton) approximate Hessian, used as LM damping
        damping = (B.T @ W @ W @ B).diagonal()

        lstsq = _sparse_lstsq if sparse else _dense_lstsq

        lam = start_lam
        new = previous
        for _ in range(_LM_MAX_DAMPING_STEPS):
            if sparse:
                lm_mat = sparse_vstack((WB, diags_array(np.sqrt(lam) * damping)))
            else:
                lm_mat = np.vstack((WB, np.diag(np.sqrt(lam) * damping)))
            lm_vec = np.hstack((W_Δq, zeros))
            Δx = lstsq(lm_mat, lm_vec)[: 3 * len(self.reference)]
            Δx = Δx.reshape(len(previous), 3)
            try:
                new = _linesearch(
                    B, Δq.delta_q, Δx, self, previous, ric_coord_arr=ric_coord_arr
                )
            except ValueError:
                # No descent along this direction: damp harder and retry.
                lam = min(lam * nu, _LM_MAX_LAMBDA)
                continue
            return new, max(lam / reduction_factor, _LM_MIN_LAMBDA)

        return new, lam

    def _full_step_cycle(
        self,
        previous: Cartesian,
        B: csr_array | Matrix,
        W: Matrix,
        start_lam: float,
        nu: float,
        reduction_factor: float,
        Δq: DeltaRedundantInternalCoordinates,
        sparse: bool = True,
        ric_coord_arr: Matrix | None = None,
    ) -> tuple[Cartesian, float]:
        """Take a single *full* damped Levenberg-Marquardt step (no line search).

        This is the classic LM lambda-adaptation: take the full damped step; if it
        decreases the residual, accept it; otherwise grow ``lam`` (more damping -> a
        shorter, more gradient-like step) and re-solve until it does. Because the step
        is never shortened by a separate scalar, the outer loop's ``new == previous``
        test only trips at a genuine stationary point, so this variant is *seed-stable*:
        ``x(q(x)) == x``. It is, however, prone to stalling on large, stiff systems
        where the residual only creeps down -- hence the ``line_search`` alternative.

        Same signature/return as :meth:`_lambda_cycle` so the two are interchangeable as
        the ``cycle`` of :meth:`_levenberg_marquardt_opt`.

        see: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm"""
        WB = W @ B
        W_Δq = W @ Δq.delta_q
        zeros = np.zeros(B.shape[1])
        damping = (B.T @ W @ W @ B).diagonal()
        lstsq = _sparse_lstsq if sparse else _dense_lstsq
        base = norm(Δq.delta_q)

        def step(lam: float) -> Cartesian:
            if sparse:
                lm_mat = sparse_vstack((WB, diags_array(np.sqrt(lam) * damping)))
            else:
                lm_mat = np.vstack((WB, np.diag(np.sqrt(lam) * damping)))
            Δx = lstsq(lm_mat, np.hstack((W_Δq, zeros)))[: 3 * len(self.reference)]
            return previous + Δx.reshape(len(previous), 3)

        def decreases(cand: Cartesian) -> bool:
            new_Δq = (
                self
                - cand.get_ric(
                    internal_coords_idx=self.primitives_idx, coord_arr=ric_coord_arr
                )
            ).minimize_dihedral()
            return bool(norm(new_Δq.delta_q) <= base)

        new = step(start_lam)
        if decreases(new):
            return new, start_lam
        lam = start_lam / reduction_factor
        new = step(lam)
        if decreases(new):
            return new, lam
        lam *= nu**2
        while True:
            new = step(lam)
            if decreases(new):
                return new, lam
            if lam >= _LM_MAX_LAMBDA:
                # At maximal damping the step is vanishingly small (``new ~ previous``),
                # so we are effectively at a stationary point of this cycle.
                # The outer loop's ``new == previous`` check then trips. This never
                # raises (matching the classic unbounded LM lambda growth); the outer
                # loop's ``max_iter`` is what signals non-convergence and, under
                # ``lm_step="auto"``, triggers the switch to the line search.
                return new, lam
            lam = min(lam * nu, _LM_MAX_LAMBDA)

    def _gauss_newton_opt(
        self,
        start_guess: Cartesian,
        max_iter: int,
        W: Matrix,
        rtol: float,
        atol: float,
        sparse: bool = True,
    ) -> Cartesian:
        from chemcoord._cartesian_coordinates.xyz_functions import (  # noqa: PLC0415
            allclose,
        )

        lstsq = _sparse_lstsq if sparse else _dense_lstsq

        # The 0-based reindexed coordinate arrays depend only on the atom index order
        # and the primitive set, not on the coordinate values, so they are invariant
        # across the optimization. ``sort_index`` fixes the order that ``align`` (called
        # every iteration) also produces, keeping the cached arrays valid throughout.
        previous = start_guess.sort_index()
        nobending_arr, full_arr = _cached_coord_arrays(previous, self.primitives_idx)

        converged = False
        i = 0
        while not converged:
            if (i := i + 1) > max_iter:
                raise ValueError(f"Not converged after {max_iter} iterations.")

            get_wilson_B = (
                previous.get_sparse_Wilson_B if sparse else previous.get_Wilson_B
            )
            B: csr_array | Matrix = get_wilson_B(
                self.primitives_idx, coord_arr=nobending_arr
            )

            q_current = previous.get_ric(self.primitives_idx, coord_arr=full_arr)

            Δq = (self - q_current).minimize_dihedral()

            Δx_flat = lstsq(W @ B, W @ Δq.delta_q)
            Δx = Δx_flat.reshape(len(previous), 3)

            new = _linesearch(B, Δq.delta_q, Δx, self, previous, ric_coord_arr=full_arr)

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
        cycle: _LMCycle,
        start_lam: float = 1e-5,
        nu: float = 1.5,
        reduction_factor: float = 10,
        sparse: bool = True,
    ) -> Cartesian:
        """Outer Levenberg-Marquardt loop. ``cycle`` supplies a single step and is
        either :meth:`_full_step_cycle` (seed-stable) or :meth:`_lambda_cycle` (robust
        line search); both share the same signature and ``(new, lam)`` return."""
        from chemcoord._cartesian_coordinates.xyz_functions import (  # noqa: PLC0415
            allclose,
        )

        # The 0-based reindexed coordinate arrays depend only on the atom index order
        # and the primitive set, not on the coordinate values, so they are invariant
        # across the optimization. ``sort_index`` fixes the order that ``align`` (called
        # every iteration) also produces, keeping the cached arrays valid throughout.
        previous = start_guess.sort_index()
        nobending_arr, full_arr = _cached_coord_arrays(previous, self.primitives_idx)

        converged = False
        i = 0

        lam = start_lam
        while not converged:
            assert previous is not None
            if (i := i + 1) > max_iter:
                raise ValueError(f"Not converged after {max_iter} iterations.")

            get_wilson_B = (
                previous.get_sparse_Wilson_B if sparse else previous.get_Wilson_B
            )
            B: csr_array | Matrix = get_wilson_B(
                self.primitives_idx, coord_arr=nobending_arr
            )

            q_current = previous.get_ric(self.primitives_idx, coord_arr=full_arr)

            Δq = (self - q_current).minimize_dihedral()

            new, lam = cycle(
                previous,
                B,
                W,
                lam,
                nu,
                reduction_factor,
                Δq,
                sparse=sparse,
                ric_coord_arr=full_arr,
            )

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
        sparse: bool = True,
        lm_step: LMStep = "auto",
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
            sparse: default :class:`True`, whether to use the sparse linear-algebra
                back-transformation (sparse Wilson B matrix and ``lsmr``) or the dense
                one (dense Wilson B matrix and :func:`numpy.linalg.lstsq`). Both paths
                are numerically equivalent; the sparse one scales better with system
                size. Exposed mainly to compare the two side by side.
            lm_step: default ``"auto"``, the Levenberg-Marquardt step control (only used
                when ``opt_alg="LM"``). ``"full_step"`` takes the full damped step and
                adapts the damping -- it is *seed-stable* (``x(q(x)) == x``) but can
                stall on large, stiff systems. ``"line_search"`` shortens an overshoot
                by backtracking -- it scales to large systems but is not seed-stable on
                flat/degenerate minima. ``"auto"`` runs ``"full_step"`` first and, if it
                has not converged within a bounded number of iterations, restarts the
                solve with ``"line_search"`` -- giving seed-stability whenever the
                full step converges and robustness otherwise.
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

        # W is diagonal. In the sparse path keeping it sparse lets ``W @ B`` stay sparse
        # throughout the weighted least-squares solve (the Wilson B matrix is banded);
        # in the dense path W must be a dense diagonal so ``W @ B`` stays a dense array.
        W = diags_array(np.asarray(weights)) if sparse else np.diag(np.asarray(weights))

        if opt_alg == "LM":

            def run_lm(step_cycle: _LMCycle, n_iter: int) -> Cartesian:
                return self._levenberg_marquardt_opt(
                    start_guess, n_iter, W, rtol, atol, step_cycle, sparse=sparse
                )

            if lm_step == "full_step":
                new = run_lm(self._full_step_cycle, max_iter)
            elif lm_step == "line_search":
                new = run_lm(self._lambda_cycle, max_iter)
            elif lm_step == "auto":
                # Prefer the seed-stable full step; fall back to the robust line search
                # only if it has not converged within the bounded budget.
                try:
                    new = run_lm(
                        self._full_step_cycle, min(max_iter, _LM_AUTO_SWITCH_ITER)
                    )
                except ValueError:
                    new = run_lm(self._lambda_cycle, max_iter)
            else:
                assert_never(lm_step)
        elif opt_alg == "gauss":
            new = self._gauss_newton_opt(
                start_guess, max_iter, W, rtol, atol, sparse=sparse
            )
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
    return Primitives(start_and_end)


def _linesearch(
    B: csr_array | Matrix,
    Δq: Vector,
    Δx: Matrix,
    current: RedundantInternalCoordinates,
    previous: Cartesian,
    alpha: float = 1.0,
    c: float = 1e-4,
    tau: float = 0.5,
    max_iter: int = 100,
    ric_coord_arr: Matrix | None = None,
) -> Cartesian:
    # NOTE: alpha is a backtracking-line-search scalar
    # see: https://en.wikipedia.org/wiki/Backtracking_line_search
    too_far = True
    t = c * 2 * norm(B.T @ Δq)

    backstep = 0
    while too_far:
        backstep += 1
        new = previous + alpha * Δx
        q_new = new.get_ric(
            internal_coords_idx=current.primitives_idx, coord_arr=ric_coord_arr
        )
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
    sparse: bool = True,
    lm_step: LMStep = "auto",
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
        sparse: default :class:`True`, whether the back-transformation of each image via
            :meth:`~.RedundantInternalCoordinates.get_cartesian` uses the sparse
            (sparse Wilson B + ``lsmr``) or dense (dense Wilson B +
            :func:`numpy.linalg.lstsq`) linear algebra. Both are numerically equivalent;
            the sparse path scales better. Mainly useful for comparing the two.
        lm_step: default ``"auto"``, the Levenberg-Marquardt step control passed to
            :meth:`~.RedundantInternalCoordinates.get_cartesian` for each image. See
            there; ``"auto"`` prefers the seed-stable full step and falls back to the
            robust line search when it does not converge.

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
            sparse=sparse,
            lm_step=lm_step,
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
        # The path is built end->start and then reversed, so a per-image ``seeds``
        # sequence (indexed in the final start->end order) has to be reversed too --
        # otherwise every image is seeded with its mirror image's guess.
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
                sparse=sparse,
                lm_step=lm_step,
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
