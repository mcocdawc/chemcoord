
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities
from . import zmat_functions


# TODO look for indexing
def zmat(zmat, outputfile, implicit_numbering=True):
    """Writes the zmatrix into a file.

    .. note:: Since it permamently writes a file, this function is strictly speaking **not sideeffect free**.
        The frame to be written is of course not changed.

    Args:
        zmat (pd.dataframe): 
        outputfile (str): 
        implicit_numbering (bool): If implicit_numbering is set, the zmat indexing is changed to range(1, number_atoms+1)
            Besides the index is omitted while writing which means, that the index is given implicitly by the row number.

    Returns:
        None: None
    """

    # The following functions are necessary to deal with the fact, that pandas does not support "NaN" for integers.
    # It was written by the user LondonRob at StackExchange:
    # http://stackoverflow.com/questions/25789354/exporting-ints-with-missing-values-to-csv-in-pandas/31208873#31208873
    # Begin of the copied code snippet
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
    # End of the copied code snippet


    if implicit_numbering:
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
            index=True,
            header=False,
            mode='w'
            )


def xyz(xyz_frame, outputfile, sort_index=True):
    """Writes the xyz_frame into a file.

    If sort_index is true, the frame is sorted by the index before writing. 

    .. note:: Since it permamently writes a file, this function is strictly speaking **not sideeffect free**.
        The frame to be written is of course not changed.

    Args:
        xyz_frame (pd.dataframe): 
        outputfile (str): 
        sort_index (bool):

    Returns:
        None: None
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
    """Writes a list of xyz_frames into a molden file.

    .. note:: Since it permamently writes a file, this function is strictly speaking **not sideeffect free**.
        The frame to be written is of course not changed.

    Args:
        xyz_frame (pd.dataframe): 
        outputfile (str): 

    Returns:
        None: None
    """
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
