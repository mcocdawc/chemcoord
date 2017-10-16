
# Changelog

## Performance

* The jitted numba functions are now cached instead of recompiled.
* subs method improved for both Cartesian and Zmat

## Bugfixes

* to_cjson had undefined variables
* get_grad_cartesian supports now drop_auto_dummies as keyword argument
* get_cartesian correctly passes on metadata from Zmat


## Enhancement

* to_molden writes energies, read_molden reads energies
