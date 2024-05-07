chemcoord: A python module for coordinates of molecules
=======================================================


.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - .. image:: https://github.com/mcocdawc/chemcoord/blob/v2.1.1/docs/source/_static/logo/chemcoord_logo.png
              :align: center
              :width: 140
              :alt: Chemcoord logo
     -
   * - Latest Release
     - .. image:: https://img.shields.io/pypi/v/chemcoord.svg
            :target: https://pypi.python.org/pypi/chemcoord
   * - Package Status
     - .. image:: https://img.shields.io/pypi/status/chemcoord.svg
            :target: https://pypi.python.org/pypi/chemcoord
   * - Documentation
     - .. image:: https://readthedocs.org/projects/chemcoord/badge/?&style=plastic
            :target: https://chemcoord.readthedocs.io/
            :alt: Documentation Status
   * - License
     - .. image:: https://img.shields.io/pypi/l/chemcoord.svg
            :target: https://www.gnu.org/licenses/lgpl-3.0.en.html
   * - Build Status
     - .. image:: https://circleci.com/gh/mcocdawc/chemcoord/tree/master.svg?style=shield
            :target: https://app.circleci.com/pipelines/github/mcocdawc/chemcoord
   * - Coverage
     - .. image:: https://codecov.io/gh/mcocdawc/chemcoord/branch/master/graph/badge.svg
            :target: https://codecov.io/gh/mcocdawc/chemcoord


Website
-------

The project's website with documentation is:
http://chemcoord.readthedocs.org/

Features
--------

-  Molecules are reliably transformed between cartesian space and non
   redundant internal coordinates (Zmatrices). Dummy atoms are inserted
   and removed automatically on the fly if necessary.
-  The created Zmatrix is not only a structure expressed in internal
   coordinates, it is a "chemical" Zmatrix. "Chemical" Zmatrix means,
   that e.g. distances are along bonds or dihedrals are defined as they
   are drawn in chemical textbooks (Newman projections).
-  Analytical gradients for the transformations between the different
   coordinate systems are implemented.
-  Performance intensive functions/methods are implemented with
   Fortran/C like speed using `numba <http://numba.pydata.org/>`__.
-  Geometries may be defined with symbolic expressions using
   `sympy <http://www.sympy.org/en/index.html>`__.
-  Built on top of `pandas <http://pandas.pydata.org/>`__ with very
   similar syntax. This allows for example distinct label or row based
   indexing.
-  It derived from my own work and I heavily use it during the day. So
   all functions are tested and tailored around the work flow in
   theoretical chemistry.
-  `It as a python module ;) <https://xkcd.com/353/>`__

Installation guide
------------------

A working python 3 installation is required (3.7 <= version <= 3.11 are tested).

It is highly recommended to use this module in combination with
`Ipython <http://ipython.org/>`__ and `jupyter <http://jupyter.org/>`__.


Unix
~~~~

There are packaged versions on PyPi and conda-forge.

For the packaged version from `PyPi <https://pypi.org/project/chemcoord/>`__, the following commands have to be executed:

::

   pip install chemcoord


For the packaged version from `conda-forge <https://anaconda.org/conda-forge/chemcoord>`__, the following commands have to be executed:

::

   conda install -c conda-forge chemcoord

For the up to date development version (experimental):

::

   git clone https://github.com/mcocdawc/chemcoord.git
   cd chemcoord
   pip install .


Windows
~~~~~~~

Neither installation nor running the module are tested on windows. To
the best of my knowledge it should work there as well. Just use the same
steps as for UNIX.


Citation and mathematical background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



Acknowledgement
~~~~~~~~~~~~~~~


.. image:: https://github.com/zulip/zulip/blob/main/static/images/logo/zulip-icon-circle.svg
   :width: 80
   :align: left
   :target: https://zulip.com/

Zulip is an open-source modern team chat app designed to keep both live and asynchronous conversations organized,
that supports the development of chemcoord with a free plan.
