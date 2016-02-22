.. ChemCoord documentation master file, created by
   sphinx-quickstart on Tue Jan 12 23:12:55 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Welcome to ChemCoord's documentation!
=====================================


Features
++++++++

* `You can use it as a python module. <https://xkcd.com/353/>`_
* It reliably converts from Cartesian space (xyz-files) to internal coordinates (zmat-files)
  **without** introducing dummy atoms. Even in the case of linearity.
* The created zmatrix is not only a transformation to internal coordinates, it is a "chemical" zmatrix. 
  By chemical I mean, that e.g. distances are along bonds or dihedrals are defined as you draw them in chemical textbooks.
* It derived from my own work and I heavily use it during the day.
  So all functions are tested and tailored around the workflow in theoretical chemistry.
* The classes are safe to inherit from and you can easily costumize it for the needs of your project.

Contents
+++++++++

.. toctree::
    :maxdepth: 2

    installation.rst
    remarks.rst 
    documentation.rst
    references.rst


