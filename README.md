# chemcoord: A python module for coordinates of molecules

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.python.org/pypi/chemcoord">
    <img src="https://img.shields.io/pypi/v/chemcoord.svg" alt="latest release" />
    </a>
  </td>
<tr>
  <td>Package Status</td>
  <td>
    <a href="https://pypi.python.org/pypi/chemcoord">
    <img src="https://img.shields.io/pypi/status/chemcoord.svg"
      alt="status" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://www.gnu.org/licenses/lgpl-3.0.en.html">
    <img src="https://img.shields.io/pypi/l/chemcoord.svg" alt="license" />
    </a>
  </td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://travis-ci.org/mcocdawc/chemcoord">
    <img src="https://travis-ci.org/mcocdawc/chemcoord.svg?branch=master"
      alt="travis build status" />
    </a>
  </td>
</tr>
<tr>
  <td>Code Quality</td>
  <td>
    <a href="https://landscape.io/github/mcocdawc/chemcoord/master">
    <img  src="https://landscape.io/github/mcocdawc/chemcoord/master/landscape.svg?style=flat"
      alt="Code Health" />
    </a>
  </td>
</tr>
<tr>
  <td>Coverage</td>
  <td>
    <a href="https://codecov.io/gh/mcocdawc/chemcoord">
    <img src="https://codecov.io/gh/mcocdawc/chemcoord/branch/master/graph/badge.svg" alt="Codecov" />
    </a>
  </td>
</tr>
</table>

## Website

The project's website with documentation is: http://chemcoord.readthedocs.org/


## Features

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


## Installation guide
A working python installation is required (2.7 and >=3.5 are possible).

It is highly recommended to use this module in combination with
  [Ipython](http://ipython.org/) and [jupyter](http://jupyter.org/).
They come shipped by default with the
  [anaconda3 installer](https://www.continuum.io/downloads/)

### Unix


For the packaged versions, the following commands have to be executed in the
  terminal:
```
conda install -c mcocdawc chemcoord
```
or
```
pip install chemcoord
```
For the up to date development version (experimental):
```
git clone https://github.com/mcocdawc/chemcoord.git
cd chemcoord
pip install .
```

### Windows

Neither installation nor running the module are tested on windows.
To the best of my knowledge it should work there as well.
Just use the same steps as for UNIX.

### Changelog

The changelog can be found
[here](https://github.com/mcocdawc/chemcoord/blob/master/CHANGELOG.md).



### Citation and mathematical background

If chemcoord is used in a project that leads to a scientific publication,
please acknowledge this fact by citing Oskar Weser (2017) using this ready-made BibTeX entry:

```
@mastersthesis{OWeser2017,
author = {Oskar Weser},
title = {An efficient and general library for the definition and use of internal coordinates in large molecular systems},
school = {Georg August Universität Göttingen},
year = {2017},
month = {November},
}
```

The master thesis including the derivation of implemented equations
and the mathematical background can be found
[here](https://github.com/mcocdawc/chemcoord/blob/master/docs/source/files/master_thesis_oskar_weser_chemcoord.pdf).
