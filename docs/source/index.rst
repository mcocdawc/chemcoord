Welcome to ChemCoord's documentation!
=====================================


Features
++++++++

* Molecules are reliably transformed between cartesian space and
  non redundant internal coordinates (Zmatrices).
  Dummy atoms are inserted and removed automatically on the fly if necessary.
* The created Zmatrix is not only a structure expressed in internal coordinates,
  it is a "chemical" Zmatrix.
  "Chemical" Zmatrix means, that e.g. distances are along bonds
  or dihedrals are defined as they are drawn in chemical textbooks
  (Newman projections).
* Analytical gradients for the transformations between the different
  coordinate systems are implemented.
* Performance intensive functions/methods are implemented
  with Fortran/C like speed using
  `numba <http://numba.pydata.org/>`_.
* Geometries may be defined with symbolic expressions using
  `sympy <http://www.sympy.org/en/index.html>`_
* Built on top of
  `pandas <http://pandas.pydata.org/>`_
  with very similar syntax.
  This allows for example distinct label or row based indexing.
* It derived from my own work and I heavily use it during the day.
  So all functions are tested and tailored around the work flow in
  theoretical chemistry.
* The classes are safe to inherit from and can easily be customized.
* `It is a python module ;) <https://xkcd.com/353/>`_

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


The changelog can be found
`here <https://github.com/mcocdawc/chemcoord/blob/master/CHANGELOG.md>`_.

Citation and mathematical background
++++++++++++++++++++++++++++++++++++

If chemcoord is used in a project that leads to a scientific publication,
please acknowledge this fact by citing Oskar Weser (2017) using this ready-made BibTeX entry::

  @mastersthesis{OWeser2017,
  author = {Oskar Weser},
  title = {An efficient and general library for the definition and use of internal coordinates in large molecular systems},
  school = {Georg August Universität Göttingen},
  year = {2017},
  month = {November},
  }

The master thesis including the derivation of implemented equations
and the mathematical background can be found
`here <https://github.com/mcocdawc/chemcoord/blob/master/docs/source/files/master_thesis_oskar_weser_chemcoord.pdf>`_.
