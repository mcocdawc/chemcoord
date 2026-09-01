"""The back-transformation from redundant internal to cartesian coordinates.

Given a target set of redundant internal coordinates ``q`` there is in general no
closed-form cartesian structure, because the primitive coordinates are redundant and
mutually constrained. The structure is instead found iteratively: linearise the
coordinate map around the current guess with Wilson's B matrix, solve the resulting
weighted least-squares problem for a cartesian displacement, step, and repeat until
the structure stops moving.

This module holds that solve and nothing else:

* :func:`backtransform` -- the entry point, dispatching on ``opt_alg``.
  :meth:`~chemcoord.RedundantInternalCoordinates.get_cartesian` validates its
  arguments, builds the weight matrix ``W`` and then calls this.
* :func:`_gauss_newton_opt` and :func:`_levenberg_marquardt_opt` -- the two outer
  loops, both iterating until :func:`~chemcoord.xyz_functions.allclose` reports that
  the structure no longer changes.
* :func:`_line_search_cycle` -- one damped Levenberg-Marquardt step.
* :func:`_linesearch` -- Armijo backtracking, shared by the Gauss-Newton loop and
  :func:`_line_search_cycle`.
* :func:`_sparse_lstsq` and the array caching helpers -- the linear algebra
  underneath.

The functions take the :class:`~chemcoord.RedundantInternalCoordinates` whose ``q`` is
being realised as their leading argument. They only read ``q.primitives_idx`` /
``q.reference`` and use its arithmetic, never constructing one, so the class is
imported for typing only and this module has no runtime dependency on its own package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, cast
from warnings import warn

import numpy as np
from numpy import float64
from numpy.linalg import norm
from scipy.sparse import csr_array, diags_array
from scipy.sparse import vstack as sparse_vstack
from scipy.sparse.linalg import lsmr, splu
from typing_extensions import assert_never

from chemcoord._cartesian_coordinates._cartesian_class_bmat import Primitives
from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import COORDS
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord.exceptions import ConvergenceError, LineSearchFailed
from chemcoord.typing import Matrix, Vector

if TYPE_CHECKING:
    from chemcoord._redundant_internal_coordinates.main import (
        DeltaRedundantInternalCoordinates,
        RedundantInternalCoordinates,
    )


# Iteration budget for the ``lsmr`` fallback in :func:`_sparse_lstsq`, used only when
# the direct factorisation fails on a singular system. It is not reached in the test
# suite. LSMR converges slowly on these systems -- a representative 101M solve needs
# ~3000 iterations -- so this is a bound on wasted effort, not a tolerance.
_LSTSQ_MAX_ITER: Final = 2000

# Bounds for the Levenberg-Marquardt damping ``λ``. It is shrunk after every
# accepted step (drifting back toward fast Gauss-Newton) and only grown when the
# damped direction admits no descent, so these are loose safety rails rather than
# tuned values. ``_LM_MAX_DAMPING_STEPS`` caps how often ``λ`` may be grown within
# a single outer iteration before giving up.
_LM_MIN_λ: Final = 1e-14
_LM_MAX_λ: Final = 1e6
_LM_MAX_DAMPING_STEPS: Final = 30


def _sparse_lstsq(
    A: Matrix, b: Vector, atol: float = 1e-8, btol: float = 1e-8
) -> Vector[np.float64]:
    """Solve the least-squares problem ``min_x ||A x - b||`` exploiting the sparsity
    of the (banded) Wilson B matrix.

    Every internal coordinate involves at most four atoms, so each row of the Wilson B
    matrix has at most twelve nonzero entries irrespective of the system size, and the
    Levenberg-Marquardt system ``A = [W B; sqrt(lambda) D]`` inherits that.

    Solved through the normal equations ``(AᵀA) x = Aᵀb`` with a direct sparse LU
    factorisation. ``AᵀA`` is the much smaller ``(3 n_atoms, 3 n_atoms)`` matrix, and
    for a molecule it is the (sparse) connectivity graph squared, so the factorisation
    barely fills in -- 2.2x on a 1413-atom protein. Forming and factorising it costs
    less than a single ``lsmr`` sweep:

    ==================  ========  =============================
    solver              time      relative error vs the exact x
    ==================  ========  =============================
    ``lsmr``, 2000 it   0.219 s   7.9e-03
    this                0.007 s   8.1e-08
    ==================  ========  =============================

    (101M, a 10002 x 4239 augmented system; see ``BENCHMARKS.md``.) The iterative
    alternative is a poor fit here: the system is ill-conditioned and rank-deficient by
    the rigid-body null space, so ``lsmr`` needs ~3000 iterations to converge on that
    example and every solve used to terminate on its iteration cap rather than on its
    tolerance -- which truncated every step and set an accuracy floor for the whole
    back-transformation.

    Normal equations square the condition number, which is the usual reason to avoid
    them. That is tolerable here because the Levenberg-Marquardt damping regularises the
    system, and it is measured rather than assumed: the residual above is five orders of
    magnitude below the iterative solve it replaces. Should the factorisation fail
    anyway -- a singular ``AᵀA`` at vanishing damping -- this falls back to ``lsmr``.

    ``atol``/``btol`` are kept for the fallback and for signature compatibility.
    """
    sparse_A = csr_array(A)
    N = (sparse_A.T @ sparse_A).tocsc()
    rhs = _as_vector(sparse_A.T @ b)
    try:
        return cast(Vector[np.float64], splu(N).solve(rhs))
    except RuntimeError:
        # Singular normal equations: no damping left to regularise the rigid-body
        # null space. Fall back to the iterative solve, which handles rank deficiency
        # by returning the minimum-norm solution.
        return lsmr(sparse_A, b, atol=atol, btol=btol, maxiter=_LSTSQ_MAX_ITER)[0]


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


def _align_and_check(
    previous: Cartesian, new: Cartesian, rtol: float, atol: float
) -> tuple[Cartesian, bool]:
    """Superimpose ``new`` onto ``previous`` and report whether it moved.

    Returns the aligned ``new``, which becomes the next iterate, and whether the two
    are numerically identical.

    Both outer loops need the same Kabsch fit for two purposes -- deciding convergence
    and producing the next iterate -- and the obvious spelling does it twice::

        converged = allclose(new, previous, align=True)   # aligns internally
        previous = previous.align(new)[1]                 # aligns again

    Aligning is the single most expensive operation in the loop, so it is done once
    here. Comparing the coordinate arrays directly also skips the boolean
    :class:`~pandas.DataFrame` that :func:`~chemcoord.xyz_functions.isclose` builds to
    answer a yes/no question, and its atom-label check, which is vacuous here because
    both structures come from the same molecule.
    """
    previous_centered, new_aligned = previous.align(new)
    converged = bool(
        np.isclose(
            previous_centered.loc[:, COORDS].values,
            new_aligned.loc[:, COORDS].values,
            rtol=rtol,
            atol=atol,
        ).all()
    )
    return new_aligned, converged


def backtransform(
    q: RedundantInternalCoordinates,
    start_guess: Cartesian,
    W: Matrix,
    *,
    max_iter: int,
    rtol: float,
    atol: float,
    opt_alg: Literal["LM", "gauss"],
) -> Cartesian:
    """Iterate to the cartesian structure whose internal coordinates are ``q.q``.

    ``W`` is the (already assembled) sparse diagonal weight matrix over
    ``q.primitives_idx``, so that ``W @ B`` stays sparse. The returned structure is
    *not* yet aligned onto ``start_guess`` -- that is done by the caller,
    :meth:`~chemcoord.RedundantInternalCoordinates.get_cartesian`.
    """
    if opt_alg == "LM":
        return _levenberg_marquardt_opt(q, start_guess, max_iter, W, rtol, atol)
    elif opt_alg == "gauss":
        return _gauss_newton_opt(q, start_guess, max_iter, W, rtol, atol)
    else:
        assert_never(opt_alg)


def _gauss_newton_opt(
    q: RedundantInternalCoordinates,
    start_guess: Cartesian,
    max_iter: int,
    W: Matrix,
    rtol: float,
    atol: float,
) -> Cartesian:
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
            raise ConvergenceError(
                f"Not converged after {max_iter} iterations.", last=previous
            )

        B = previous.get_sparse_Wilson_B(q.primitives_idx, coord_arr=nobending_arr)

        q_current = previous.get_ric(q.primitives_idx, coord_arr=full_arr)

        Δq = (q - q_current).minimize_dihedral()

        Δx_flat = _sparse_lstsq(W @ B, _as_vector(W @ Δq.delta_q))
        Δx = Δx_flat.reshape(len(previous), 3)

        new = _linesearch(B, Δq.delta_q, Δx, q, previous, W, ric_coord_arr=full_arr)

        previous, converged = _align_and_check(previous, new, rtol, atol)

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
    start_λ: float = 1e-5,
    damping_growth: float = 1.5,
    reduction_factor: float = 10,
) -> Cartesian:
    """Outer Levenberg-Marquardt loop, stepping with :func:`_line_search_cycle`."""
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
            raise ConvergenceError(
                f"Not converged after {max_iter} iterations.", last=previous
            )

        B = previous.get_sparse_Wilson_B(q.primitives_idx, coord_arr=nobending_arr)

        q_current = previous.get_ric(q.primitives_idx, coord_arr=full_arr)

        Δq = (q - q_current).minimize_dihedral()

        new, λ = _line_search_cycle(
            q,
            previous,
            B,
            W,
            λ,
            damping_growth,
            reduction_factor,
            Δq,
            ric_coord_arr=full_arr,
        )

        previous, converged = _align_and_check(previous, new, rtol, atol)

    if i > 100:
        warn(f"The transformation to cartesian coordinates took {i} iterations.")

    return new


def _line_search_cycle(
    q: RedundantInternalCoordinates,
    previous: Cartesian,
    B: csr_array,
    W: Matrix,
    start_λ: float,
    damping_growth: float,
    reduction_factor: float,
    Δq: DeltaRedundantInternalCoordinates,
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

    # Invariants of the damped least-squares system, independent of λ. The Wilson B
    # matrix is banded, so ``W @ B`` and the augmented system stay sparse.
    WB = W @ B
    W_Δq = W @ Δq.delta_q
    zeros = np.zeros(B.shape[1])
    # diagonal of the (Gauss-Newton) approximate Hessian, used as LM damping
    damping_diag = (B.T @ W @ W @ B).diagonal()
    # right-hand side of the augmented system
    lm_vec = _as_vector(np.hstack((W_Δq, zeros)))

    λ = start_λ
    for _ in range(_LM_MAX_DAMPING_STEPS):
        lm_mat = sparse_vstack((WB, diags_array(np.sqrt(λ) * damping_diag)))
        Δx = _sparse_lstsq(lm_mat, lm_vec)[: 3 * len(q.reference)]
        Δx = Δx.reshape(len(previous), 3)
        try:
            new = _linesearch(
                B, Δq.delta_q, Δx, q, previous, W, ric_coord_arr=ric_coord_arr
            )
        except LineSearchFailed:
            # No descent along this direction: damp harder and retry.
            λ = min(λ * damping_growth, _LM_MAX_λ)
            continue
        return new, max(λ / reduction_factor, _LM_MIN_λ)

    # Every damped direction admitted a descent step that the line search could not
    # find. Returning ``previous`` here would look like a vanishing step to the outer
    # loop, i.e. it would be reported as convergence at a non-minimum.
    raise ConvergenceError(
        f"No descent direction after {_LM_MAX_DAMPING_STEPS} damping steps.",
        last=previous,
    )


def _linesearch(
    B: csr_array,
    Δq: Vector,
    Δx: Matrix,
    current: RedundantInternalCoordinates,
    previous: Cartesian,
    W: Matrix,
    alpha: float = 1.0,
    c: float = 1e-4,
    τ: float = 0.5,
    max_iter: int = 100,
    ric_coord_arr: Matrix | None = None,
) -> Cartesian:
    """Armijo backtracking along ``Δx``, shortening ``alpha`` until the step gives a
    sufficient decrease of the merit function ``f(x) = ‖W Δq(x)‖``.

    ``f`` is the objective the least-squares step actually minimises, hence the
    weighting: a step can reduce ``‖W Δq‖`` while *increasing* the unweighted
    ``‖Δq‖``, since the weights span two orders of magnitude (1.0 for bonds down to
    0.01 for bendings).

    ``Δq`` is measured after ``minimize_dihedral``. The wrap is not optional: a dihedral
    crossing its 2π branch moves the unwrapped residual by ~0.7 where a step moves it by
    ~2e-5, so an unwrapped test would reject nearly every step. It does make ``f``
    discontinuous at the branch.

    The sufficient-decrease threshold uses the directional derivative along the step,
    ``-∇f·Δx = (Bᵀ W² Δq)·Δx / f``, not the gradient norm ``‖Bᵀ Δq‖``. This is what
    makes backtracking work: both the achieved and the required decrease are first
    order in ``alpha``, so a threshold that does not scale with ``Δx`` cancels
    ``alpha`` out of the comparison entirely and the test becomes scale invariant --
    failing at every ``alpha`` for a direction that is merely badly aligned, until
    ``alpha`` underflows the comparison and a step of ~1e-13 is "accepted". With the
    directional derivative the same factor appears on both sides and, since ``c < 1``,
    any genuine descent direction is accepted at a small enough ``alpha``.

    Raises:
        ~chemcoord.exceptions.LineSearchFailed: If no ``alpha`` gives sufficient
            decrease, i.e. ``Δx`` is not a descent direction for ``f``. The caller
            (:func:`_line_search_cycle`) reacts by damping harder.

    see: https://en.wikipedia.org/wiki/Backtracking_line_search
    """
    W_Δq = _as_vector(W @ Δq)
    f = norm(W_Δq)
    if f == 0:
        # Already exactly on target; every alpha is as good as any other.
        return previous + alpha * Δx
    # -∇f·Δx, positive iff Δx points downhill for the weighted residual.
    descent = float((B.T @ _as_vector(W @ W_Δq)) @ np.asarray(Δx).ravel()) / f

    for _ in range(max_iter):
        new = previous + alpha * Δx
        q_new = new.get_ric(
            internal_coords_idx=current.primitives_idx, coord_arr=ric_coord_arr
        )
        Δq_new = (current - q_new).minimize_dihedral().delta_q
        if norm(_as_vector(W @ Δq_new)) <= f - c * alpha * descent:
            return new
        alpha *= τ

    raise LineSearchFailed(
        f"Line search not terminated after {max_iter} iterations",
        last=previous,
    )
