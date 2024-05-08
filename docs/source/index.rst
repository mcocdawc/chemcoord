Welcome to ChemCoord's documentation!
=====================================

The code can be found `here <https://github.com/mcocdawc/chemcoord>`_.


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
    bugs_and_contributors.rst
    license.rst


The changelog can be found
`here <https://github.com/mcocdawc/chemcoord/blob/master/CHANGELOG.md>`_.

Citation and mathematical background
++++++++++++++++++++++++++++++++++++


The theory behind chemcoord is described in `this paper <https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.27029>`__.
If this package is used in a project that leads to a scientific
publication, please acknowledge it by citing.

::

    @article{https://doi.org/10.1002/jcc.27029,
        author = {Weser, Oskar and Hein-Janke, Björn and Mata, Ricardo A.},
        title = {Automated handling of complex chemical structures in Z-matrix coordinates—The chemcoord library},
        journal = {Journal of Computational Chemistry},
        volume = {44},
        number = {5},
        pages = {710-726},
        keywords = {analytical gradients, geometry optimization, non-linear constraints, transition state search, Z-matrix},
        doi = {10.1002/jcc.27029},
        year = {2023}
    }


My (Oskar Weser) master thesis including a more detailed derivation of implemented equations and
the mathematical background can be found
`here <https://github.com/mcocdawc/chemcoord/blob/master/docs/source/_static/master_thesis_oskar_weser_chemcoord.pdf>`__.
It also has a ready-made BibTeX entry:

::

  @mastersthesis{OWeser2017,
      author = {Oskar Weser},
      title = {An efficient and general library for the definition and use of internal coordinates in large molecular systems},
      school = {Georg August Universität Göttingen},
      year = {2017},
      month = {November},
  }

