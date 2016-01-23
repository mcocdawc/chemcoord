
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities

def zmat(inputfile, pythonic_index=False):
    """
    The input is a filename.
    The output is a zmat_DataFrame.
    """

    zmat_frame = pd.read_table(
        inputfile,
        comment='#',
        delim_whitespace=True,
        names=['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'],
        )
    n_atoms = zmat_frame.shape[0]
    zmat_frame.index = range(1, n_atoms+1)
    # Changing to pythonic indexing
    if python_index:
        zmat_frame.index = range(n_atoms)
        zmat_frame['bond_with'] = zmat_frame['bond_with'] - 1
        zmat_frame['angle_with'] = zmat_frame['angle_with'] - 1
        zmat_frame['dihedral_with'] = zmat_frame['dihedral_with'] - 1
    return zmat_frame


def xyz(inputfile, pythonic_index=False):
    """
    The input is a filename.
    The output is a xyz_DataFrame.
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
