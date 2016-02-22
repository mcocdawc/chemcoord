# chemcoord: A python module for coordinates of molecules

## Introduction

With this module you can read cartesian (xyz) and internal (zmat) coordinates into pandas DataFrames.
Besides it supplies you with a lot of functions to manipulate them and make your workflow more efficient.

## Features

* `You can use it as a python module. <https://xkcd.com/353/>`_
* It reliably converts from Cartesian space (xyz-files) to internal coordinates (zmat-files)
  **without** introducing dummy atoms. Even in the case of linearity.
* The created zmatrix is not only a transformation to internal coordinates, it is a "chemical" zmatrix. 
  By chemical I mean, that e.g. distances are along bonds or dihedrals are defined as you draw them in chemical textbooks.
* It derived from my own work and I heavily use it during the day.
  So all functions are tested and tailored around the workflow in theoretical chemistry.
* The classes are safe to inherit from and you can easily costumize it for the needs of your project.

## Installation



## Documentation



