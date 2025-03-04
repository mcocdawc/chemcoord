# Changelog for v2.1.2 -> v2.2.1



## Bugfixes

- Ensured that xyz files are always read as floats, even if xyz coordinates are formatted as integers.

- `to_molden` and similar functions accept now an `Iterable[Cartesian]`,
    previously `to_molden(zm.get_cartesian() for zm in zmatrices)` unexpectedly failed.


## New features

- Added type hinting

- Enable conversion to/from pyscf molecules.

- Added contextmanager to temporarily change element data.

- Better handling of situations where the normal vector of a dihedral reference plane switches sign

- `xyz_functions.view` function supports now different file types

- reworked settings as dataclass

- Exposed an explicit interpolate function


## Infrastructure

- Enforce `ruff` and `mypy` checks as part of test suite

- sphinx picks up type hints for the docstring