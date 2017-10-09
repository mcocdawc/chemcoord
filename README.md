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

The project's website is: http://chemcoord.readthedocs.org/


## Features

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


## Installation guide
A working python (2.7 and 3.x are possible)
You can use for example the [anaconda3 installer](https://www.continuum.io/downloads/)

The advantage of the anaconda3 installer is that you get a lot of additional
modules and programs,
that make it really easy to work with python.
For example [Ipython](http://ipython.org/) and the [jupyter notebooks](http://jupyter.org/)
I highly recommend to use those.

### Unix


For the packaged versions just type in your terminal:
```
conda install chemcoord
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
