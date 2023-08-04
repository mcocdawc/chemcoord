# Changelog for 2.1.0

## Accounting for changed pandas API

pandas removed indexing with sets, which chemcoord relies on.
We now cast sets explicitly to lists before indexing, so it is still possible
to use sets for indexing of chemcoord objects.

The `append` method of `Cartesian` was removed, because it was removed
from pandas DataFrames.


## Accounting for changed numba API

The warning about `generated_jit` was switched off for the moment.
See https://github.com/mcocdawc/chemcoord/issues/76

## Accounting for changed sympy + pandas interplay

The conversion to the correct column types did not always work correctly
when using `.subs` for subsituting sympy variables.
This is fixed now.

## Allow indexing via attribute access

Instead of `molecule.loc[:, 'atom']` one can also use `molecule.atom`
to access a column.


## Fixed bug in gradient calculation

Fixed bug in gradient calculation.
