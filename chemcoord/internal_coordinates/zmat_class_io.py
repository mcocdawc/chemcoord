from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
from chemcoord.internal_coordinates.zmat_class_core import Zmat_core
from chemcoord._exceptions import PhysicalMeaningError
from chemcoord.configuration import settings


class Zmat_io(Zmat_core):
    @classmethod
    def from_zmat(cls, inputfile, implicit_index=True):
        """Reads a zmat file.

        Lines beginning with ``#`` are ignored.

        Args:
            inputfile (str):
            implicit_index (bool): If this option is true the first column
            has to be the element symbols for the atoms.
                The row number is used to determine the index.

        Returns:
            Zmat:
        """
        cols = ['atom', 'bond_with', 'bond', 'angle_with', 'angle',
                'dihedral_with', 'dihedral']
        if implicit_index:
            zmat_frame = pd.read_table(inputfile, comment='#',
                                       delim_whitespace=True,
                                       names=cols)
            Zmat = cls(zmat_frame)
            Zmat.index = range(1, Zmat.n_atoms + 1)
        else:
            zmat_frame = pd.read_table(inputfile, comment='#',
                                       delim_whitespace=True,
                                       names=['temp_index'] + cols)
            Zmat = cls(zmat_frame)
            Zmat.set_index('temp_index', drop=True, inplace=True)
            Zmat.index.name = None
        return Zmat

    @staticmethod
    def _lost_precision(s):
        """
        The total amount of precision lost over Series `s`
        during conversion to int64 dtype
        """
        COULD_BE_ANY_INTEGER = 0
        try:
            diff = (s - s.fillna(COULD_BE_ANY_INTEGER).astype(np.int64))
            return diff.sum()
        except ValueError:
            return np.nan

    def _prepare_nansafe_output(self):
        COULD_BE_ANY_INTEGER = 0

        def _lost_precision(s):
            """
            The total amount of precision lost over Series `s`
            during conversion to int64 dtype
            """
            try:
                diff = (s - s.fillna(COULD_BE_ANY_INTEGER).astype(np.int64))
                return diff.sum()
            except ValueError:
                return np.nan

        def _nansafe_integer_convert(s, epsilon=1e-9):
            """
            Convert Series `s` to an object type with `np.nan`
            represented as an empty string ""
            """
            if _lost_precision(s) < epsilon:
                # Here's where the magic happens
                as_object = s.fillna(COULD_BE_ANY_INTEGER)
                as_object = as_object.astype(np.int64).astype(np.object)
                as_object[s.isnull()] = ""
                return as_object
            else:
                return s

        return self.frame.apply(_nansafe_integer_convert)


    # def to_zmat(self, buf=None, implicit_index=True)

    def write(self, outputfile, implicit_index=True):
        """Writes the zmatrix into a file.

        .. note:: Since it permamently writes a file, this function is
            strictly speaking **not sideeffect free**.
            The frame to be written is of course not changed.

        Args:
            outputfile (str):
            implicit_index (bool): If implicit_index is set, the zmat indexing
                is changed to range(1, number_atoms+1). Besides the index is
                omitted while writing which means, that the index is given
                implicitly by the row number.

        Returns:
            None: None
        """
        # The following functions are necessary to deal with the fact,
        # that pandas does not support "NaN" for integers.
        # It was written by the user LondonRob at StackExchange:
        # http://stackoverflow.com/questions/25789354/
        # exporting-ints-with-missing-values-to-csv-in-pandas/31208873#31208873
        # Begin of the copied code snippet


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

        if implicit_index:
            zmat_frame = self.change_numbering().zmat_frame
            nansafe_to_csv(
                zmat_frame.loc[:, 'atom':],
                outputfile,
                sep=str(' '),
                index=False,
                header=False,
                mode='w'
            )
        else:
            nansafe_to_csv(
                self.zmat_frame,
                outputfile,
                sep=str(' '),
                index=True,
                header=False,
                mode='w'
            )
