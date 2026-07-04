# Changelog for v2.1.2 -> v2.2.0



## Bugfixes

- Ensured that xyz files are always read as floats, even if xyz coordinates are formatted as integers.

- `to_molden` and similar functions accept now an `Iterable[Cartesian]`,
    previously `to_molden(zm.get_cartesian() for zm in zmatrices)` unexpectedly failed.

- fixed bug where get_bonds with modified atom data only worked with 0-indexed molecules (the default).

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
