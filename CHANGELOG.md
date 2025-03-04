# Changelog for v2.1.2 -> v2.2.1



## Bugfixes

- Ensured that xyz files are always read as floats, even if xyz coordinates are formatted as integers.

- fixed bug where get_bonds with modified atom data only worked with 0-indexed molecules (the default)


## New features

- Added type hinting

- Enable conversion to/from pyscf molecules.

- Added contextmanager to temporarily change element data.

- Made it easier to change element data in `get_bonds`


## Infrastructure

- Enforce `ruff` and `mypy` checks as part of test suite