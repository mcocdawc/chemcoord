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
* :func:`_gauss_newton_opt` and :func:`_levenberg_marquardt_opt` -- the two
  algorithms, sharing the outer loop :func:`_optimise`.
* :func:`_line_search_cycle` -- one damped Levenberg-Marquardt step.
* :func:`_linesearch` -- Armijo backtracking, shared by the Gauss-Newton loop and
  :func:`_line_search_cycle`.
* :func:`_sparse_lstsq`, :func:`_remove_rigid_modes` and the array caching helpers --
  the linear algebra underneath.

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
from scipy.sparse import csc_array, csr_array, diags_array
from scipy.sparse import vstack as sparse_vstack
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import lsmr, splu
from typing_extensions import assert_never

from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import COORDS
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord._cartesian_coordinates.xyz_functions import get_kabsch_rotation
from chemcoord.exceptions import ConvergenceError, LineSearchFailed
from chemcoord.typing import Matrix, Vector

if TYPE_CHECKING:
    from chemcoord._redundant_internal_coordinates.main import (
        DeltaRedundantInternalCoordinates,
        RedundantInternalCoordinates,
    )


# Iteration budget for the ``lsmr`` fallback in :func:`_sparse_lstsq`: a bound on wasted
# effort, not a tolerance, since a representative 101M solve needs ~3000 iterations.
_LSTSQ_MAX_ITER: Final = 2000

# Loose safety rails for the Levenberg-Marquardt damping ``λ``, not tuned values.
_LM_MIN_λ: Final = 1e-14
_LM_MAX_λ: Final = 1e6
_LM_MAX_DAMPING_STEPS: Final = 30


def _sparse_lstsq(A: Matrix, b: Vector, perm: Vector[np.int32]) -> Vector[np.float64]:
    """Solve ``min_x ||A x - b||`` via the sparse normal equations ``AᵀA x = Aᵀb``.

    A direct LU factorisation of ``AᵀA``, permuted by ``perm``, is ~30x faster and five
    orders of magnitude more accurate than ``lsmr`` on 101M; see ``BENCHMARKS.md``.

    ``AᵀA`` is singular by the six rigid-body motions. The system stays consistent, but
    a factorisation that survives returns an arbitrary rigid-body component and ``lsmr``
    none; which branch runs varies with the scipy build, so callers strip it with
    :func:`_remove_rigid_modes`.
    """
    sparse_A = csr_array(A)
    N = (sparse_A.T @ sparse_A).tocsr()
    rhs = _as_vector(sparse_A.T @ b)
    try:
        permuted = splu(csc_array(N[perm][:, perm]), permc_spec="NATURAL")
    except RuntimeError:
        # Exactly zero pivot; ``lsmr`` returns the minimum-norm solution.
        return lsmr(sparse_A, b, atol=1e-8, btol=1e-8, maxiter=_LSTSQ_MAX_ITER)[0]
    x = np.empty(N.shape[0], dtype=float64)
    x[perm] = permuted.solve(rhs[perm])
    return cast(Vector[np.float64], x)


def _as_vector(v: Matrix | Vector) -> Vector[float64]:
    """Typing helper: numpy's stubs lose the 1-D shape of ``@`` and ``hstack``."""
    return cast(Vector[float64], v)


def _fill_reducing_permutation(A: Matrix) -> Vector[np.int32]:
    """Reverse Cuthill-McKee ordering of ``AᵀA``, keeping its LU factorisation sparse.

    Atom numbering is unrelated to spatial proximity; RCM brings the bandwidth from 4121
    to 89 on a 1413-atom protein. Any permutation is correct, so it is computed once.
    """
    sparse_A = csr_array(A)
    return cast(
        Vector[np.int32],
        reverse_cuthill_mckee((sparse_A.T @ sparse_A).tocsr(), symmetric_mode=True),
    )


def _rigid_body_basis(structure: Cartesian) -> Matrix:
    """Orthonormal columns spanning the rigid-body motions, i.e. the null space of B.

    Fewer than six for a linear molecule, where one rotation vanishes.
    """
    centered = structure.loc[:, COORDS].values
    centered = centered - centered.mean(axis=0)
    modes = np.zeros((centered.size, 6))
    for k in range(3):
        modes[k::3, k] = 1.0
        modes[:, 3 + k] = np.cross(np.eye(3)[k], centered).ravel()
    U, s, _ = np.linalg.svd(modes, full_matrices=False)
    return cast(Matrix, U[:, s > 1e-10 * s[0]])


def _remove_rigid_modes(Δx: Vector, basis: Matrix) -> Matrix:
    """Project the rigid-body component out of a flat step, returned as ``(n, 3)``.

    ``B Δx`` is unchanged, but the arbitrary component from :func:`_sparse_lstsq` can
    make the step 1600x longer than its useful part (peroxide), and
    :func:`_linesearch` would throttle the whole step to compensate.
    """
    return cast(Matrix, (Δx - basis @ (basis.T @ Δx)).reshape(-1, 3))


def _align_and_check(
    previous: Cartesian, new: Cartesian, rtol: float, atol: float
) -> tuple[Cartesian, bool]:
    """Superimpose ``new`` onto ``previous``; return it and whether it moved.

    One Kabsch fit on the arrays serves both convergence test and next iterate; the atom
    order is fixed by the caller, so :meth:`~chemcoord.Cartesian.align` is not needed.
    """
    pos_previous = previous.loc[:, COORDS].values
    pos_new = new.loc[:, COORDS].values
    pos_previous = pos_previous - pos_previous.mean(axis=0)
    pos_new = pos_new - pos_new.mean(axis=0)
    pos_new = pos_new @ get_kabsch_rotation(pos_previous, pos_new).T
    converged = bool(np.isclose(pos_previous, pos_new, rtol=rtol, atol=atol).all())
    aligned = new.copy()
    aligned.loc[:, COORDS] = pos_new
    return aligned, converged


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

    ``W`` is the sparse diagonal weight matrix. The result is not aligned yet.
    """
    if opt_alg == "LM":
        return _levenberg_marquardt_opt(q, start_guess, max_iter, W, rtol, atol)
    elif opt_alg == "gauss":
        return _gauss_newton_opt(q, start_guess, max_iter, W, rtol, atol)
    else:
        assert_never(opt_alg)


#: ``(previous, B, Δq, perm, ric_coord_arr) -> new``
_Step: TypeAlias = Callable[
    [Cartesian, csr_array, "DeltaRedundantInternalCoordinates", Vector, Matrix],
    Cartesian,
]


def _optimise(
    q: RedundantInternalCoordinates,
    start_guess: Cartesian,
    max_iter: int,
    W: Matrix,
    rtol: float,
    atol: float,
    step: _Step,
) -> Cartesian:
    """Outer loop: linearise, ``step``, and repeat until the structure stops moving."""
    # The cached index arrays depend on the atom order, which the loop preserves.
    previous = start_guess.sort_index()
    nobending_arr = previous._to_array_nobending(q.primitives_idx)
    full_arr = previous._to_array_full(q.primitives_idx)
    perm = None

    converged = False
    i = 0
    while not converged:
        if (i := i + 1) > max_iter:
            raise ConvergenceError(
                f"Not converged after {max_iter} iterations.", last=previous
            )

        B = previous.get_sparse_Wilson_B(q.primitives_idx, coord_arr=nobending_arr)
        if perm is None:
            perm = _fill_reducing_permutation(W @ B)
        q_current = previous.get_ric(q.primitives_idx, coord_arr=full_arr)
        Δq = (q - q_current).minimize_dihedral()

        new = step(previous, B, Δq, perm, full_arr)

        previous, converged = _align_and_check(previous, new, rtol, atol)

    if i > 100:
        warn(f"The transformation to cartesian coordinates took {i} iterations.")

    return new


def _gauss_newton_opt(
    q: RedundantInternalCoordinates,
    start_guess: Cartesian,
    max_iter: int,
    W: Matrix,
    rtol: float,
    atol: float,
) -> Cartesian:
    def step(
        previous: Cartesian,
        B: csr_array,
        Δq: DeltaRedundantInternalCoordinates,
        perm: Vector[np.int32],
        ric_coord_arr: Matrix,
    ) -> Cartesian:
        Δx = _remove_rigid_modes(
            _sparse_lstsq(W @ B, _as_vector(W @ Δq.delta_q), perm),
            _rigid_body_basis(previous),
        )
        return _linesearch(B, Δq.delta_q, Δx, q, previous, W, ric_coord_arr)

    return _optimise(q, start_guess, max_iter, W, rtol, atol, step)


def _levenberg_marquardt_opt(
    q: RedundantInternalCoordinates,
    start_guess: Cartesian,
    max_iter: int,
    W: Matrix,
    rtol: float,
    atol: float,
) -> Cartesian:
    λ = 1e-5

    def step(
        previous: Cartesian,
        B: csr_array,
        Δq: DeltaRedundantInternalCoordinates,
        perm: Vector[np.int32],
        ric_coord_arr: Matrix,
    ) -> Cartesian:
        nonlocal λ
        new, λ = _line_search_cycle(q, previous, B, W, λ, Δq, ric_coord_arr, perm)
        return new

    return _optimise(q, start_guess, max_iter, W, rtol, atol, step)


def _line_search_cycle(
    q: RedundantInternalCoordinates,
    previous: Cartesian,
    B: csr_array,
    W: Matrix,
    start_λ: float,
    Δq: DeltaRedundantInternalCoordinates,
    ric_coord_arr: Matrix,
    perm: Vector[np.int32],
    damping_growth: float = 1.5,
    reduction_factor: float = 10,
) -> tuple[Cartesian, float]:
    """Take a single Levenberg-Marquardt step, returning it and the next ``λ``.

    An overshooting step is shortened by :func:`_linesearch` rather than damped harder:
    once ``λ`` ratchets up it never recovers. ``λ`` grows only when the damped direction
    admits no descent.

    see: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm"""
    WB = W @ B
    damping_diag = (B.T @ W @ W @ B).diagonal()
    lm_vec = _as_vector(np.hstack((W @ Δq.delta_q, np.zeros(B.shape[1]))))
    rigid = _rigid_body_basis(previous)

    λ = start_λ
    for _ in range(_LM_MAX_DAMPING_STEPS):
        lm_mat = sparse_vstack((WB, diags_array(np.sqrt(λ) * damping_diag)))
        Δx = _as_vector(_sparse_lstsq(lm_mat, lm_vec, perm)[: 3 * len(q.reference)])
        try:
            new = _linesearch(
                B,
                Δq.delta_q,
                _remove_rigid_modes(Δx, rigid),
                q,
                previous,
                W,
                ric_coord_arr,
            )
        except LineSearchFailed:
            λ = min(λ * damping_growth, _LM_MAX_λ)
            continue
        return new, max(λ / reduction_factor, _LM_MIN_λ)

    # Returning ``previous`` would be mistaken for convergence by the outer loop.
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
    ric_coord_arr: Matrix,
    alpha: float = 1.0,
    c: float = 1e-4,
    τ: float = 0.5,
    max_iter: int = 100,
) -> Cartesian:
    """Armijo backtracking along ``Δx`` on the merit function ``f(x) = ‖W Δq(x)‖``.

    ``f`` is weighted because that is what the step minimises. The threshold uses the
    directional derivative ``(Bᵀ W² Δq)·Δx / f``, so both sides of the test scale with
    ``alpha`` and any descent direction is accepted at a small enough ``alpha``.

    Raises:
        ~chemcoord.exceptions.LineSearchFailed: If ``Δx`` is not a descent direction.

    see: https://en.wikipedia.org/wiki/Backtracking_line_search
    """
    W_Δq = _as_vector(W @ Δq)
    f = norm(W_Δq)
    if f == 0:
        return previous + alpha * Δx
    descent = (B.T @ _as_vector(W @ W_Δq)) @ np.asarray(Δx).ravel() / f

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
