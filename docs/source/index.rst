Welcome to ChemCoord's documentation!
=====================================


Features
++++++++

* It reliably converts from Cartesian space (xyz-files) to
  non reduntant internal coordinates (zmat-files).
  Dummy atoms are inserted and removed automatically on the fly if necessary.
* The created Zmatrix is not only a structure expressed in internal coordinates,
  it is a "chemical" Zmatrix.
  "Chemical" Zmatrix means, that e.g. distances are along bonds
  or dihedrals are defined as they are drawn in chemical textbooks
  (Newman projections).
* Analytical gradients for the transformations between the different
  coordinate systems are implemented.
* Performance intensive functions/methods are implemented
  with Fortran/C like speed using [numba](http://numba.pydata.org/).
* Geometries may be defined with symbolic expressions using
  [sympy](http://www.sympy.org/en/index.html).
* Built on top of [pandas](http://pandas.pydata.org/) with very similar syntax.
  This allows for example distinct label or row based indexing.
* It derived from my own work and I heavily use it during the day.
  So all functions are tested and tailored around the work flow in
  theoretical chemistry.
* The classes are safe to inherit from and can easily be customized.
* [It as a python module ;)](https://xkcd.com/353/)

Contents
+++++++++

.. toctree::
    :maxdepth: 2

    installation.rst
    tutorial.rst
    documentation.rst
    references.rst
    bugs.rst
    license.rst
