"""The back-transformation from redundant internal to cartesian coordinates.

Given a target set of redundant internal coordinates ``q`` there is in general no
closed-form cartesian structure, because the primitive coordinates are redundant and
mutually constrained. The structure is instead found iteratively: linearise the
coordinate map around the current guess with Wilson's B matrix, solve the resulting
weighted least-squares problem for a cartesian displacement, step, and repeat until
the structure stops moving.

This module holds that solve and nothing else:

* :func:`backtransform` -- the entry point, dispatching on ``opt_alg``/``lm_step``.
  :meth:`~chemcoord.RedundantInternalCoordinates.get_cartesian` validates its
  arguments, builds the weight matrix ``W`` and then calls this.
* :func:`_gauss_newton_opt` and :func:`_levenberg_marquardt_opt` -- the two outer
  loops, both iterating until :func:`~chemcoord.xyz_functions.allclose` reports that
  the structure no longer changes.
* :func:`_λ_cycle` and :func:`_full_step_cycle` -- the two interchangeable
  Levenberg-Marquardt inner steps (see :data:`LMStep`).
* :func:`_linesearch` -- Armijo backtracking, shared by the Gauss-Newton loop and
  :func:`_λ_cycle`.
* :func:`_sparse_lstsq` / :func:`_dense_lstsq` and the array caching helpers -- the
  linear algebra underneath.

The functions take the :class:`~chemcoord.RedundantInternalCoordinates` whose ``q`` is
being realised as their leading argument. They only read ``q.primitives_idx`` /
``q.reference`` and use its arithmetic, never constructing one, so the class is
imported for typing only and this module has no runtime dependency on its own package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast
from warnings import warn

import numpy as np
from numpy import float64
from numpy.linalg import norm
from scipy.sparse import csr_array, diags_array
from scipy.sparse import vstack as sparse_vstack
from scipy.sparse.linalg import lsmr
from typing_extensions import assert_never

from chemcoord._cartesian_coordinates._cartesian_class_bmat import Primitives
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord.typing import Matrix, Vector

if TYPE_CHECKING:
    from chemcoord._redundant_internal_coordinates.main import (
        DeltaRedundantInternalCoordinates,
        RedundantInternalCoordinates,
    )


#: Which Levenberg-Marquardt step control the back-transformation uses.
#: ``"full_step"`` is the seed-stable full damped step (``x(q(x)) == x``) that can stall
#: on large, stiff systems; ``"line_search"`` is the robust backtracking variant that
#: scales but is not seed-stable on flat/degenerate minima; ``"auto"`` (default) runs
#: ``"full_step"`` first and switches to ``"line_search"`` if it has not converged after
#: :data:`_LM_AUTO_SWITCH_ITER` iterations.
LMStep: TypeAlias = Literal["auto", "full_step", "line_search"]

#: Type of a single LM step cycle (``_λ_cycle`` / ``_full_step_cycle``): it maps
#: the current state to ``(new_cartesian, λ)``.
_LMCycle: TypeAlias = Callable[..., tuple["Cartesian", float]]


# Upper bound on the number of LSMR iterations per linear solve. The augmented
# Levenberg-Marquardt system is ill-conditioned, so without a cap LSMR chases the
# stopping tolerance for thousands of iterations. Since every solve is only one
# inexact step of the outer Gauss-Newton/LM loop, a bounded, approximate solution is
# enough and keeps each iteration cheap.
_LSTSQ_MAX_ITER: Final = 200

# Bounds for the Levenberg-Marquardt damping ``λ``. It is shrunk after every
# accepted step (drifting back toward fast Gauss-Newton) and only grown when the
# damped direction admits no descent, so these are loose safety rails rather than
# tuned values. ``_LM_MAX_DAMPING_STEPS`` caps how often ``λ`` may be grown within
# a single outer iteration before giving up.
_LM_MIN_λ: Final = 1e-14
_LM_MAX_λ: Final = 1e6
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


def _as_vector(v: Matrix | Vector) -> Vector[float64]:
    """Reassert the 1-D shape of an expression that is one-dimensional by construction.

    Numpy's stubs type the result of a matrix-vector product and of
    :func:`numpy.hstack` with the unspecified shape ``tuple[int, ...]``, which does not
    typecheck against the strictly 1-D :data:`~chemcoord.typing.Vector`. (Depending on
    the numpy version; the stubs became more precise in numpy 2.3.) Returns its argument
    unchanged, it is purely a typing helper.
    """
    return cast(Vector[float64], v)


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


def backtransform(
    q: RedundantInternalCoordinates,
    start_guess: Cartesian,
    W: Matrix,
    *,
    max_iter: int,
    rtol: float,
    atol: float,
    opt_alg: Literal["LM", "gauss"],
    sparse: bool,
    lm_step: LMStep,
) -> Cartesian:
    """Iterate to the cartesian structure whose internal coordinates are ``q.q``.

    ``W`` is the (already assembled) diagonal weight matrix over ``q.primitives_idx``;
    it is sparse when ``sparse`` is set, so that ``W @ B`` stays sparse. The returned
    structure is *not* yet aligned onto ``start_guess`` -- that is done by the caller,
    :meth:`~chemcoord.RedundantInternalCoordinates.get_cartesian`.
    """
    if opt_alg == "LM":

        def run_lm(step_cycle: _LMCycle, n_iter: int) -> Cartesian:
            return _levenberg_marquardt_opt(
                q, start_guess, n_iter, W, rtol, atol, step_cycle, sparse=sparse
            )

        if lm_step == "full_step":
            return run_lm(_full_step_cycle, max_iter)
        elif lm_step == "line_search":
            return run_lm(_λ_cycle, max_iter)
        elif lm_step == "auto":
            # Prefer the seed-stable full step; fall back to the robust line search
            # only if it has not converged within the bounded budget.
            try:
                return run_lm(_full_step_cycle, min(max_iter, _LM_AUTO_SWITCH_ITER))
            except ValueError:
                return run_lm(_λ_cycle, max_iter)
        else:
            assert_never(lm_step)
    elif opt_alg == "gauss":
        return _gauss_newton_opt(q, start_guess, max_iter, W, rtol, atol, sparse=sparse)
    else:
        assert_never(opt_alg)


def _gauss_newton_opt(
    q: RedundantInternalCoordinates,
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
    nobending_arr, full_arr = _cached_coord_arrays(previous, q.primitives_idx)

    converged = False
    i = 0
    while not converged:
        if (i := i + 1) > max_iter:
            raise ValueError(f"Not converged after {max_iter} iterations.")

        get_wilson_B = previous.get_sparse_Wilson_B if sparse else previous.get_Wilson_B
        B: csr_array | Matrix = get_wilson_B(q.primitives_idx, coord_arr=nobending_arr)

        q_current = previous.get_ric(q.primitives_idx, coord_arr=full_arr)

        Δq = (q - q_current).minimize_dihedral()

        Δx_flat = lstsq(W @ B, _as_vector(W @ Δq.delta_q))
        Δx = Δx_flat.reshape(len(previous), 3)

        new = _linesearch(B, Δq.delta_q, Δx, q, previous, ric_coord_arr=full_arr)

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
    q: RedundantInternalCoordinates,
    start_guess: Cartesian,
    max_iter: int,
    W: Matrix,
    rtol: float,
    atol: float,
    cycle: _LMCycle,
    start_λ: float = 1e-5,
    damping_growth: float = 1.5,
    reduction_factor: float = 10,
    sparse: bool = True,
) -> Cartesian:
    """Outer Levenberg-Marquardt loop. ``cycle`` supplies a single step and is
    either :func:`_full_step_cycle` (seed-stable) or :func:`_λ_cycle` (robust
    line search); both share the same signature and ``(new, λ)`` return."""
    from chemcoord._cartesian_coordinates.xyz_functions import (  # noqa: PLC0415
        allclose,
    )

    # The 0-based reindexed coordinate arrays depend only on the atom index order
    # and the primitive set, not on the coordinate values, so they are invariant
    # across the optimization. ``sort_index`` fixes the order that ``align`` (called
    # every iteration) also produces, keeping the cached arrays valid throughout.
    previous = start_guess.sort_index()
    nobending_arr, full_arr = _cached_coord_arrays(previous, q.primitives_idx)

    converged = False
    i = 0

    λ = start_λ
    while not converged:
        assert previous is not None
        if (i := i + 1) > max_iter:
            raise ValueError(f"Not converged after {max_iter} iterations.")

        get_wilson_B = previous.get_sparse_Wilson_B if sparse else previous.get_Wilson_B
        B: csr_array | Matrix = get_wilson_B(q.primitives_idx, coord_arr=nobending_arr)

        q_current = previous.get_ric(q.primitives_idx, coord_arr=full_arr)

        Δq = (q - q_current).minimize_dihedral()

        new, λ = cycle(
            q,
            previous,
            B,
            W,
            λ,
            damping_growth,
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


def _λ_cycle(
    q: RedundantInternalCoordinates,
    previous: Cartesian,
    B: csr_array | Matrix,
    W: Matrix,
    start_λ: float,
    damping_growth: float,
    reduction_factor: float,
    Δq: DeltaRedundantInternalCoordinates,
    sparse: bool = True,
    ric_coord_arr: Matrix | None = None,
) -> tuple[Cartesian, float]:
    """Take a single damped Gauss-Newton (Levenberg-Marquardt) step.

    The damped search direction for the current ``λ`` is refined with a
    backtracking line search (:func:`_linesearch`): an over-shooting full step is
    *shortened* rather than the direction being rotated toward gradient descent by
    an ever-growing ``λ``. Plain λ-only acceptance stalls badly near the
    solution -- once ``λ`` ratchets up it never recovers and the residual only
    crawls, so on large systems the outer loop never converges within ``max_iter``
    -- whereas shortening the good Gauss-Newton direction keeps the fast
    convergence rate.

    ``λ`` is decreased after every accepted step (drifting back toward the fast
    Gauss-Newton regime) and only increased when even the damped direction admits
    no descent, which restores the regularisation that makes LM more robust than
    plain Gauss-Newton. The returned ``λ`` seeds the next outer iteration.

    see: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm"""

    # Invariants of the damped least-squares system, independent of λ. In the
    # sparse path the Wilson B matrix is banded, so ``W @ B`` and the augmented
    # system stay sparse.
    WB = W @ B
    W_Δq = W @ Δq.delta_q
    zeros = np.zeros(B.shape[1])
    # diagonal of the (Gauss-Newton) approximate Hessian, used as LM damping
    damping_diag = (B.T @ W @ W @ B).diagonal()
    # right-hand side of the augmented system
    lm_vec = _as_vector(np.hstack((W_Δq, zeros)))

    lstsq = _sparse_lstsq if sparse else _dense_lstsq

    λ = start_λ
    new = previous
    for _ in range(_LM_MAX_DAMPING_STEPS):
        if sparse:
            lm_mat = sparse_vstack((WB, diags_array(np.sqrt(λ) * damping_diag)))
        else:
            lm_mat = np.vstack((WB, np.diag(np.sqrt(λ) * damping_diag)))
        Δx = lstsq(lm_mat, lm_vec)[: 3 * len(q.reference)]
        Δx = Δx.reshape(len(previous), 3)
        try:
            new = _linesearch(
                B, Δq.delta_q, Δx, q, previous, ric_coord_arr=ric_coord_arr
            )
        except ValueError:
            # No descent along this direction: damp harder and retry.
            λ = min(λ * damping_growth, _LM_MAX_λ)
            continue
        return new, max(λ / reduction_factor, _LM_MIN_λ)

    return new, λ


def _full_step_cycle(
    q: RedundantInternalCoordinates,
    previous: Cartesian,
    B: csr_array | Matrix,
    W: Matrix,
    start_λ: float,
    damping_growth: float,
    reduction_factor: float,
    Δq: DeltaRedundantInternalCoordinates,
    sparse: bool = True,
    ric_coord_arr: Matrix | None = None,
) -> tuple[Cartesian, float]:
    """Take a single *full* damped Levenberg-Marquardt step (no line search).

    This is the classic LM λ-adaptation: take the full damped step; if it
    decreases the residual, accept it; otherwise grow ``λ`` (more damping -> a
    shorter, more gradient-like step) and re-solve until it does. Because the step
    is never shortened by a separate scalar, the outer loop's ``new == previous``
    test only trips at a genuine stationary point, so this variant is *seed-stable*:
    ``x(q(x)) == x``. It is, however, prone to stalling on large, stiff systems
    where the residual only creeps down -- hence the ``line_search`` alternative.

    Same signature/return as :func:`_λ_cycle` so the two are interchangeable as
    the ``cycle`` of :func:`_levenberg_marquardt_opt`.

    see: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm"""
    WB = W @ B
    W_Δq = W @ Δq.delta_q
    zeros = np.zeros(B.shape[1])
    damping_diag = (B.T @ W @ W @ B).diagonal()
    lstsq = _sparse_lstsq if sparse else _dense_lstsq
    base = norm(Δq.delta_q)
    # right-hand side of the augmented system; independent of λ
    lm_vec = _as_vector(np.hstack((W_Δq, zeros)))

    def step(λ: float) -> Cartesian:
        if sparse:
            lm_mat = sparse_vstack((WB, diags_array(np.sqrt(λ) * damping_diag)))
        else:
            lm_mat = np.vstack((WB, np.diag(np.sqrt(λ) * damping_diag)))
        Δx = lstsq(lm_mat, lm_vec)[: 3 * len(q.reference)]
        return previous + Δx.reshape(len(previous), 3)

    def decreases(cand: Cartesian) -> bool:
        new_Δq = (
            q
            - cand.get_ric(
                internal_coords_idx=q.primitives_idx, coord_arr=ric_coord_arr
            )
        ).minimize_dihedral()
        return bool(norm(new_Δq.delta_q) <= base)

    new = step(start_λ)
    if decreases(new):
        return new, start_λ
    λ = start_λ / reduction_factor
    new = step(λ)
    if decreases(new):
        return new, λ
    λ *= damping_growth**2
    while True:
        new = step(λ)
        if decreases(new):
            return new, λ
        if λ >= _LM_MAX_λ:
            # At maximal damping the step is vanishingly small (``new ~ previous``),
            # so we are effectively at a stationary point of this cycle.
            # The outer loop's ``new == previous`` check then trips. This never
            # raises (matching the classic unbounded LM λ growth); the outer
            # loop's ``max_iter`` is what signals non-convergence and, under
            # ``lm_step="auto"``, triggers the switch to the line search.
            return new, λ
        λ = min(λ * damping_growth, _LM_MAX_λ)


def _linesearch(
    B: csr_array | Matrix,
    Δq: Vector,
    Δx: Matrix,
    current: RedundantInternalCoordinates,
    previous: Cartesian,
    α: float = 1.0,
    c: float = 1e-4,
    τ: float = 0.5,
    max_iter: int = 100,
    ric_coord_arr: Matrix | None = None,
) -> Cartesian:
    # NOTE: α is a backtracking-line-search scalar
    # see: https://en.wikipedia.org/wiki/Backtracking_line_search
    too_far = True
    t = c * 2 * norm(B.T @ Δq)

    backstep = 0
    while too_far:
        backstep += 1
        new = previous + α * Δx
        q_new = new.get_ric(
            internal_coords_idx=current.primitives_idx, coord_arr=ric_coord_arr
        )
        if norm(Δq) < α * t + norm((current - q_new).minimize_dihedral().delta_q):
            α *= τ
        else:
            too_far = False
        if backstep > max_iter:
            raise ValueError(f"Line search not terminated after {max_iter} iterations")

    return new
