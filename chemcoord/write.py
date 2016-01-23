
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities

def zmat(zmat, outputfile, reset_numbering=True):
    """
    Writes the zmatrix into a file.
    If reset_numbering is set, the index of the zmat_frame is changed to ascend from zero to the number of atoms.
    The depending values like bond_with are changed accordingly.
    Since it permamently writes a file, this function is strictly speaking **not sideeffect free**.
    The zmat_frame is of course not changed.
    """
    # The following functions are necessary to deal with the fact, that pandas does not support "NaN" for integers.
    EPSILON = 1e-9

    def _lost_precision(s):
        """
        The total amount of precision lost over Series `s`
        during conversion to int64 dtype
        """
        try:
            return (s - s.fillna(0).astype(np.int64)).sum()
        except ValueError:
            return np.nan

    def _nansafe_integer_convert(s):
        """
        Convert Series `s` to an object type with `np.nan`
        represented as an empty string ""
        """
        if _lost_precision(s) < EPSILON:
            # Here's where the magic happens
            as_object = s.fillna(0).astype(np.int64).astype(np.object)
            as_object[s.isnull()] = ""
            return as_object
        else:
            return s


    def nansafe_to_csv(df, *args, **kwargs):
        """
        Write `df` to a csv file, allowing for missing values
        in integer columns

        Uses `_lost_precision` to test whether a column can be
        converted to an integer data type without losing precision.
        Missing values in integer columns are represented as empty
        fields in the resulting csv.
        """
        df.apply(_nansafe_integer_convert).to_csv(*args, **kwargs)


    if reset_numbering:
        zmat_frame = zmat_functions.change_numbering(zmat)
        nansafe_to_csv(zmat_frame.loc[:, 'atom':], outputfile,
            sep=' ',
            index=False,
            header=False,
            mode='w'
            )
    else:
        nansafe_to_csv(zmat, outputfile,
            sep=' ',
            index=False,
            header=False,
            mode='w'
            )


def xyz(xyz_frame, outputfile, sort_index = True):
    """
    Writes the xyz_DataFrame into a file.
    If (sort = True) the DataFrame is sorted by the index and the index is not written
    since it corresponds to the line.
    Since it permamently writes a file, this function is strictly speaking **not sideeffect free**.
    The xyz_frame is of course not changed.
    """
    frame = xyz_frame[['atom', 'x', 'y','z']].copy()
    if sort_index:
        frame = frame.sort_index()
        n_atoms = frame.shape[0]
        with open(outputfile, mode='w') as f:
            f.write(str(n_atoms) + 2 * '\n')
        frame.to_csv(
            outputfile,
            sep=' ',
            index=False,
            header=False,
            mode='a'
            )
    else:
        frame = frame.sort_values(by='atom')
        n_atoms = frame.shape[0]
        with open(outputfile, mode='w') as f:
            f.write(str(n_atoms) + 2 * '\n')
        frame.to_csv(
            outputfile,
            sep=' ',
            index=False,
            header=False,
            mode='a'
            )


def molden(framelist, outputfile):
    n_frames = len(framelist)
    n_atoms = framelist[0].shape[0]
    string ="""[MOLDEN FORMAT]
[N_GEO]
    """
    values = n_frames *'1\n'
    string = string + str(n_frames) + '\n[GEOCONV]\nenergy\n' + values + 'max-force\n' + values + 'rms-force\n' + values + '[GEOMETRIES] (XYZ)\n'

    with open(outputfile, mode='w') as f:
        f.write(string)

    for frame in framelist:
        frame = frame.sort_index()
        n_atoms = frame.shape[0]
        with open(outputfile, mode='a') as f:
            f.write(str(n_atoms) + 2 * '\n')
        frame.to_csv(
            outputfile,
            sep=' ',
            index=False,
            header=False,
            mode='a'
            )
