
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities

def zmat(inputfile):
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
    # Changing to pythonic indexing
    zmat_frame['bond_with'] = zmat_frame['bond_with'] - 1
    zmat_frame['angle_with'] = zmat_frame['angle_with'] - 1
    zmat_frame['dihedral_with'] = zmat_frame['dihedral_with'] - 1
    return zmat_frame


def xyz(inputfile):
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
    return xyz_frame
