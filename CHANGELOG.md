# Changelog for v2.2.0 -> unreleased


## Bugfixes

- The `default_weights` argument of `RedundantInternalCoordinates.get_cartesian` and
    `ric_functions.RIC_interpolate` was documented with a `"length"` key, but the
    `DefaultWeights` field is called `bond`. Passing the documented mapping raised
    `TypeError: __init__() got an unexpected keyword argument 'length'`. The
    docstrings now spell the key correctly.

- `Cartesian.get_primitives_idx` now passes a custom `bonds` mapping through to the
    fragment detection. Previously `connect_fragments=True` re-derived the connectivity
    with `fragmentate()`, silently ignoring the user-supplied `bonds`.

- Made the RIC back-transformation seed-stable, i.e. `x(q(x)) == x`. The
    Levenberg-Marquardt line search backtracks the step length toward zero near an
    overshoot, which the outer loop misread as convergence and accepted a non-minimum.
    On the stepwise interpolation schedules this error compounded.

- Fixed the dihedral rows of `Cartesian.get_Wilson_B` (and of the new
    `get_sparse_Wilson_B`, which shares the same kernel). The derivatives for the two
    *central* atoms of a dihedral were wrong: the two contributions that make up their
    shared term were subtracted where they must be added. The two terminal atoms were
    always correct, and the error was equal and opposite between the central pair, so
    the rows still summed to zero and the defect survived the obvious
    translational-invariance check.

    On H-O-O-H the dihedral row was `[0, 0, +1.414, 0, 0, -2.828, 0, 0, 0, 0, 0, +1.414]`
    where finite differences give the symmetric
    `[0, 0, +1.414, 0, 0, -1.414, 0, 0, -1.414, 0, 0, +1.414]`. Against finite
    differences with `h = 1e-7`, the relative error of the dihedral block drops from
    0.92 to 2.0e-07 on `default_args_start` and from 0.31 to 1.8e-08 on
    `cyclohexane_chair` -- the accuracy the bond and angle blocks always had.

    The RIC back-transformation was solving a linearised model that was wrong for
    dihedrals, so its predicted decrease often did not materialise. With the fix, the
    Levenberg-Marquardt damping is no longer driven to its cap anywhere in the test
    suite (previously 6 times on the `default_args` interpolation alone). Every image
    of that
    interpolation ends up with a lower weighted residual (by up to 5%, 2.4% in total),
    and a perturbed MIL53 structure that previously failed to converge now recovers to
    3e-07 A. The committed reference paths and the recorded residual baselines were
    regenerated accordingly.

- `Cartesian.get_ric` no longer returns `NaN` for an angle that is exactly linear. The
    dot product of two unit vectors is in `[-1, 1]` mathematically, but rounding can put
    a collinear pair a few ulp outside it, and `arccos` then yields `NaN`. `MIL53_beta`
    has an angle at exactly 180 degrees, which the back-transformation only reproduces
    closely enough to trigger this now that the Wilson B fix above lets it converge.
    The same clamp was added to the angles used for linearity detection.

- `Cartesian.get_inertia` now diagonalises the inertia tensor with `numpy.linalg.eigh`
    instead of `numpy.linalg.eig`. The inertia tensor is real symmetric, but the general
    LAPACK driver behind `eig` may -- depending on the BLAS/LAPACK implementation --
    return complex arrays whose imaginary parts are round-off noise; these propagated
    into the eigenvectors and the returned `Cartesian`. Note that the sign convention of
    the returned eigenvectors may differ from before; both are valid principal axes.

- The symmetry detection no longer fails with a `numpy.exceptions.ComplexWarning` when
    warnings are turned into errors. `pymatgen`'s `PointGroupAnalyzer` diagonalises the
    inertia tensor with `numpy.linalg.eig` as well and warns while discarding the
    round-off imaginary parts; the warning is now silenced at the `pymatgen` boundary.


## New features

- Added a sparse back-transformation from redundant internal coordinates. Wilson's
    B matrix is banded -- every internal coordinate couples at most four atoms -- so it
    is now available in compressed sparse row form via the new
    `Cartesian.get_sparse_Wilson_B`, and the least-squares solve uses
    `scipy.sparse.linalg.lsmr` instead of a dense SVD. This is numerically equivalent
    to the dense path but scales considerably better with system size.

    It is the only path: the dense linear algebra was kept for a while to compare the
    two side by side, and was removed once the comparison was recorded in
    `BENCHMARKS.md`. `Cartesian.get_Wilson_B` remains as public API.

- A back-transformation that does not converge now raises
    `chemcoord.exceptions.ConvergenceError` instead of a bare `ValueError`, and a line
    search that does not terminate raises the `ConvergenceError` subclass
    `chemcoord.exceptions.LineSearchFailed`. Both still subclass `ValueError`, so
    existing handlers keep working.

    The Levenberg-Marquardt step now raises once it has exhausted its damping steps,
    instead of returning a structure it knows is not a solution. The outer loop only
    checks whether the structure stopped moving, so that return was previously
    reported as convergence at a non-minimum.

    Conversely, `ric_functions.RIC_interpolate(schedule="auto")` now only falls through
    to the next scheduling strategy on a `ConvergenceError`. Previously it caught every
    `ValueError`, so e.g. an invalid argument combination was retried three more times
    and then reported as `RuntimeError: All scheduling strategies failed`.

- The acceptance tests of the back-transformation now measure the *weighted* residual
    `‖W Δq‖` -- the objective the damped least-squares step actually minimises --
    instead of the unweighted `‖Δq‖`. With weights spanning 1.0 (bonds) to 0.01
    (bendings) a step could improve the weighted objective while the unweighted norm
    grew, and be rejected for it.

    The Armijo threshold in the line search additionally uses the directional
    derivative along the step, `c·α·(BᵀW²Δq)·Δx / ‖WΔq‖`, rather than the gradient norm
    `2c‖BᵀΔq‖`. The old threshold did not scale with the step, so both sides of the
    sufficient-decrease test were first order in `α` and `α` cancelled: the test was
    scale invariant and a badly aligned direction failed it at *every* `α`, until `α`
    underflowed the comparison and a step of ~1e-13 was accepted by rounding. The outer
    loop then read the vanishing step as convergence. On a perturbed MIL53 structure
    this returned a structure 4.2 A from the answer with `‖Δq‖ = 5.2`; it now converges
    to `‖Δq‖ = 4.6e-06`.

    The committed interpolation reference paths were regenerated accordingly. On
    `default_args_path`, no image ends up with a worse weighted residual and five are
    better by 0.2-0.7%; the `independent` schedule is unchanged. The path tests now
    assert on the weighted residuals as well as on the coordinates, so the residuals are
    tracked as a benchmark baseline for future work on the optimizer.


## New features

- The back-transformation has a single Levenberg-Marquardt step control. An earlier
    `lm_step` argument (`"auto"`, `"full_step"`, `"line_search"`) let callers choose,
    because the two variants behaved differently -- but only because the truncated
    iterative solve handed both a poor search direction. With the direct solve they are
    the same algorithm in practice: identical residuals and wall clock on MIL53 and
    101M at sigma 0.01/0.1/0.3 and on every interpolation schedule. The backtracking
    variant is kept, being the more robust of the two where they do differ (it
    converges on a 101M seed perturbed by sigma = 0.5, where the full step gives up),
    and the `lm_step` argument is gone. None of it was ever released.


## Performance

- The linear solve inside the RIC back-transformation now uses a direct sparse LU
    factorisation of the damped normal equations instead of the iterative
    `scipy.sparse.linalg.lsmr`. `AᵀA` is the much smaller `(3 n_atoms, 3 n_atoms)`
    matrix and, being a molecular connectivity graph squared, barely fills in -- 2.2x on
    a 1413-atom protein. It is both faster and far more accurate: on a single 101M solve,
    0.007 s and a relative error of 8e-08 against 0.219 s and 7.9e-03 for `lsmr` capped
    at 2000 iterations.

    The iterative solve was a poor fit. The system is ill-conditioned and rank-deficient
    by the rigid-body null space, so `lsmr` needed ~3000 iterations to converge and every
    solve terminated on its iteration cap rather than its tolerance. That truncated every
    step, which set an accuracy floor for the whole back-transformation -- the outer loop
    stopped because short steps stopped moving the structure, not because it had reached
    the minimum.

    End to end on 101M (1413 atoms), back-transforming a perturbed structure: previously
    2.9e-05 in 55 s, now **1.8e-13 in 0.2 s**. `lsmr` is kept as a fallback for a singular
    factorisation; it is not reached in the test suite. See `BENCHMARKS.md`.


- Made `Cartesian.get_primitives_idx` considerably faster.

- The 0-based reindexed coordinate arrays are now computed once and cached across the
    RIC optimization loop instead of being rebuilt on every iteration, which was the
    dominant cost of the back-transformation for large systems.


## Infrastructure

- Migrated the sparse linear algebra from the legacy `scipy.sparse` matrix API
    (`csr_matrix`, `diags`) to the array API (`csr_array`, `diags_array`).
    `Cartesian.get_sparse_Wilson_B` therefore returns a `scipy.sparse.csr_array`.
    This raises the minimum required `scipy` version to 1.11.

- The set of primitive internal coordinates is now the nominal type
    `Primitives = NewType("Primitives", SortedSet)` instead of a plain type alias.


# Changelog for v2.1.2 -> v2.2.0



## Bugfixes

- Restored compatibility with pandas 3. Several assumptions that silently held
    under pandas 2 now raise: `.values` returns read-only arrays (copy-on-write),
    columns mixing integer atom indices with the absolute-reference string labels
    (`'origin'`, `'e_x'`, ...) are no longer coerced to a strict `str` dtype,
    `.replace` on such columns hit an internal error, and assigning symbolic
    (sympy) values into a float column raises `TypeError` instead of warning.
    All of these are handled now, so chemcoord runs on both pandas 2 and 3.

- Ensured that xyz files are always read as floats, even if xyz coordinates are formatted as integers.

- `to_molden` and similar functions accept now an `Iterable[Cartesian]`,
    previously `to_molden(zm.get_cartesian() for zm in zmatrices)` unexpectedly failed.

- fixed bug where get_bonds with modified atom data only worked with 0-indexed molecules (the default).

- Corrected the deprecation warning of `Cartesian.to_zmat`, which pointed to a
    non-existent `give_zmat` method instead of `get_zmat`.

- Fixed a crash (`Numba workqueue threading layer is terminating: Concurrent access
    has been detected`) when computing redundant internal coordinates (RICs) or
    Wilson B-matrices for molecules with bending coordinates. The inner
    `numba` helpers (`_jit_get_axes`, `_jit_x_to_plane_coords_nonlinear`) were
    marked ``parallel=True`` despite containing no ``prange`` loop, which created a
    nested parallel region when called from the outer parallel loops. This aborted
    the process whenever `numba` falls back to the non-threadsafe ``workqueue``
    threading layer (i.e. when neither TBB nor OpenMP is available).


## New features

- Added type hinting.

- Enabled conversion to/from pyscf molecules.

- Added contextmanager to temporarily change element data.

- Better handling of situations where the normal vector of a dihedral reference plane switches sign.

- `xyz_functions.view` function supports now different file types.

- reworked settings as dataclass.

- Exposed an explicit interpolate function.

- Made it easier to change element data in `get_bonds`.

- Added redundant internal coordinates (RICs) class and support for conversion between them and Cartesians

- Added interpolation in RICs via Wilson's B-Matrix, with various schedules and optimizers


## Infrastructure

- Enforced `ruff` and `mypy` checks as part of test suite.

- Added an optional `test` dependency group (`pip install .[test]`) that installs
    the testing infrastructure: `pytest`, `mypy`, the type stubs and the other
    static analysis tools. This replaces the `tests/testsuite_requirements.txt`
    and `tests/static_analysis_requirements.txt` files, which have been removed.

- Fixed `mypy` errors surfaced by newer `pandas-stubs`/`numpy` versions so that
    `mypy src/ tests/` passes cleanly again.

- Moved the `mypy` configuration from `mypy.ini` into `pyproject.toml`
    (`[tool.mypy]`); the `mypy.ini` file has been removed.

- sphinx picks up type hints for the docstring.

- Fixed several broken links in the documentation and enforce that links work.

- enable bibtex in the documentation.

- Removed a leftover debug `print` from `Cartesian.correct_dihedral`.

- Removed the unused `six` runtime dependency and the unused `setuptools-scm`
    build requirement.

- Ship the PEP 561 `py.typed` markers explicitly (via `package-data` and
    `MANIFEST.in`) so downstream type checkers reliably pick up chemcoord's types.

- Raise a proper `ImportError` (instead of an `assert`, which is stripped under
    `python -O`) when `to_pyscf_mole` is used without `pyscf` installed.

- Removed dead code (`_fix_undef_dihedrals` and its now-unreachable helpers
    `_correct_dihedral_idx`, `WhichHalf`, `_is_dihedral_tuple`) and several
    commented-out code blocks.
