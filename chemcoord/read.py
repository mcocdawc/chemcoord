
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities

def zmat(inputfile, implicit_index=True):
    """Reads a zmat file.
    
    Lines beginning with ``#`` are ignored.

    Args:
        inputfile (str): 
        implicit_index (bool): If this option is true the first column has to be the element symbols for the atoms.
            The row number is used to determine the index.

    Returns:
        pd.DataFrame: 
    """
    if implicit_index:
        zmat_frame = pd.read_table(
            inputfile,
            comment='#',
            delim_whitespace=True,
            names=['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'],
            )
        n_atoms = zmat_frame.shape[0]
        zmat_frame.index = range(1, n_atoms+1)
    else:
        zmat_frame = pd.read_table(
            inputfile,
            comment='#',
            delim_whitespace=True,
            names=['temp_index', 'atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'],
            )
        zmat_frame.set_index('temp_index', drop=True)
    return zmat_frame


def xyz(inputfile, pythonic_index=False):
    """Reads a xyz file.

    Args:
        inputfile (str): 
        pythonic_index (bool):

    Returns:
        pd.DataFrame: 
    """
    xyz_frame = pd.read_table(
        inputfile,
        skiprows=2,
        comment='#',
        delim_whitespace=True,
        names=['atom', 'x', 'y', 'z']
        )
    if not pythonic_index:
        n_atoms = xyz_frame.shape[0]
        xyz_frame.index = range(1, n_atoms+1)
    return xyz_frame
