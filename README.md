# chemcoord: A python module for coordinates of molecules

## Website

You find the website of the project is: www.http://chemcoord.readthedocs.org/


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
You need a working python 3.x installation together with some standard modules.
You can use for example the [anaconda3 installer](https://www.continuum.io/downloads/)

The advantage of the anaconda3 installer is that you get a lot of additional modules and programs,
that make it really easy to work with python. 
For example [Ipython](http://ipython.org/) and the [jupyter notebooks](http://jupyter.org/)
I highly recommend to use those.

Unix
++++

Just type in your terminal:
'''
pip install chemcoord
'''
This should also resolve all dependencies automatically.

Windows
+++++++

I tested neither installation nor running the module on windows.
As far as I know it should work as well if you use the ``pip`` manager.
If you get it installed and running, please report it on the Github page.

