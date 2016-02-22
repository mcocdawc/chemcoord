# chemcoord: A python module for coordinates of molecules


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

You need a working python 3.x installation together with the modules:

- numpy
- pandas
- math
- copy
- collections
- os
- sys
- distutils.core

All of them come shipped by default with nearly every python installation.
You can use for example the [anaconda3 installer](https://www.continuum.io/downloads/)

The advantage of the anaconda3 installer is that you get a lot of additional modules and programs
that make it really easy to work with python. 
[Ipython](http://ipython.org/) and the [jupyter notebooks](http://jupyter.org/) are an example.

### Unix

#### Installation
Create a directory where you want to put the source code and execute there::
```bash
git clone https://github.com/mcocdawc/chemcoord.git
```
If you want to install it for the user then execute::
```bash
python setup.py install --user
```

For a systemwide installation execute (you need probably `sudo` rights):
```bash
python setup.py install 
```

#### Documentation
To read the documentation go to docs and execute for a browser based documentation::
```bash
make html
```

Afterwards go to `build/html` and open the `index.html` file in a browser.

If you want to have a PDF version of the documentation execute:
```bash
make latex
make latexpdf
make latexpdf
```

Afterwards go to `build/latex` and open the `ChemCoord.pdf` file with a PDF reader.
 

### Windows

I tested neither installation nor running the program on windows.
Since I use mainly the IO functionality provided by pandas for Dataframes and 
do all operations on numpy arrays and pandas Dataframes kept in memory, 
it could work on Windows.

If you get it installed and running, please report it on the Github page.




