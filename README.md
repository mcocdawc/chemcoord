# chemcoord: A python module for coordinates of molecules

<table>
<tr>
  <td>Latest Release</td>
  <td><img src="https://img.shields.io/pypi/v/chemcoord.svg" alt="latest release" /></td>
<!-- </tr>
  <td></td>
  <td><img src="https://anaconda.org/conda-forge/pandas/badges/version.svg" alt="latest release" /></td>
</tr> -->
<tr>
  <td>Package Status</td>
  <td><img src="https://img.shields.io/pypi/status/chemcoord.svg" alt="status" /></td>
</tr>
<tr>
  <td>License</td>
  <td><img src="https://img.shields.io/pypi/l/chemcoord.svg" alt="license" /></td>
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
<!-- <tr>
  <td>Coverage</td>
  <td><img src="https://codecov.io/github/pandas-dev/pandas/coverage.svg?branch=master" alt="coverage" /></td>
</tr> -->
</table>

## Website

The project's website is: http://chemcoord.readthedocs.org/


## Features

* [You can use it as a python module](https://xkcd.com/353/)
* It reliably converts from Cartesian space (xyz-files) to internal coordinates (zmat-files)
  **without** introducing dummy atoms. Even in the case of linearity.
* The created zmatrix is not only a transformation to internal coordinates, it is a "chemical" zmatrix.
  By chemical I mean, that e.g. distances are along bonds or dihedrals are defined as you draw them in chemical textbooks.
* It derived from my own work and I heavily use it during the day.
  So all functions are tested and tailored around the workflow in theoretical chemistry.
* The classes are safe to inherit from and you can easily costumize it for the needs of your project.


## Installation guide
You need a working python (both 2 and 3 are possible) installation together with some standard modules.
You can use for example the [anaconda3 installer](https://www.continuum.io/downloads/)

The advantage of the anaconda3 installer is that you get a lot of additional modules and programs,
that make it really easy to work with python.
For example [Ipython](http://ipython.org/) and the [jupyter notebooks](http://jupyter.org/)
I highly recommend to use those.

### Unix


Just type in your terminal:
```
pip install chemcoord
```
This should also resolve all dependencies automatically.

### Windows

I tested neither installation nor running the module on windows.
As far as I know it should work as well if you use the ``pip`` manager.
If you get it installed and running, please report it on the Github page.
