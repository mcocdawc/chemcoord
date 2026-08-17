# Changelog for v2.2.0 -> unreleased


## Bugfixes

- `Cartesian.get_primitives_idx` now passes a custom `bonds` mapping through to the
    fragment detection. Previously `connect_fragments=True` re-derived the connectivity
    with `fragmentate()`, silently ignoring the user-supplied `bonds`.

- Made the RIC back-transformation seed-stable, i.e. `x(q(x)) == x`. The
    Levenberg-Marquardt line search backtracks the step length toward zero near an
    overshoot, which the outer loop misread as convergence and accepted a non-minimum.
    On the stepwise interpolation schedules this error compounded.


## New features

- Added a sparse back-transformation from redundant internal coordinates. Wilson's
    B matrix is banded -- every internal coordinate couples at most four atoms -- so it
    is now available in compressed sparse row form via the new
    `Cartesian.get_sparse_Wilson_B`, and the least-squares solve uses
    `scipy.sparse.linalg.lsmr` instead of a dense SVD. This is numerically equivalent
    to the dense path but scales considerably better with system size.

    It is selected with the new `sparse` argument (default `True`) of
    `RedundantInternalCoordinates.get_cartesian` and `ric_functions.RIC_interpolate`,
    and via `coord="RIC_sparse"` / `coord="RIC_dense"` in `xyz_functions.interpolate`,
    where `coord="RIC"` is an alias for the sparse variant.

- Added the `lm_step` argument (`"auto"`, `"full_step"`, `"line_search"`) to
    `RedundantInternalCoordinates.get_cartesian`, `ric_functions.RIC_interpolate`, and
    `xyz_functions.interpolate`. It selects the Levenberg-Marquardt step control of the
    back-transformation: `"full_step"` is seed-stable but can stall on large, stiff
    systems, `"line_search"` is robust but not seed-stable, and the default `"auto"`
    runs the former and falls back to the latter if it has not converged in time.


## Performance

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
