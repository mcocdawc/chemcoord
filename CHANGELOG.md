
# Changelog

## Performance

* The jitted numba functions are now cached instead of recompiled.
* subs method improved for both Cartesian and Zmat

## Bugfixes

* to_cjson had undefined variables
* get_grad_cartesian drops automatically inserted dummies.
  (Old behaviour may be retained using a keyword.)
* get_cartesian correctly passes on metadata from Zmat
* read_xyz no longer disturbed by numbers at the element symbols


## Enhancement

* to_molden writes energies, read_molden reads energies
