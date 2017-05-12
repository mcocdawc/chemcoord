from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import warnings
from chemcoord.internal_coordinates._zmat_class_core import Zmat_core
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

    @classmethod
    def read_zmat(cls, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.from_zmat`
        """
        message = 'Will be removed in the future. Please use from_zmat().'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return cls.from_zmat(*args, **kwargs)

    def to_zmat(self, buf=None, implicit_index=True,
                float_format='{:.6f}'.format, overwrite=True,
                convert_nan_int=True, header=False):
        """Write zmat-file

        Args:
            buf (str): StringIO-like, optional buffer to write to
            implicit_index (bool): If implicit_index is set, the zmat indexing
                is changed to ``range(1, self.n_atoms + 1)``.
                Using :meth:`~chemcoord.Zmat.change_numbering`
                Besides the index is omitted while writing which means,
                that the index is given
                implicitly by the row number.
            float_format (one-parameter function): Formatter function
                to apply to column’s elements if they are floats.
                The result of this function must be a unicode string.
            overwrite (bool): May overwrite existing files.

        Returns:
            formatted : string (or unicode, depending on data and options)
        """
        molecule = self.change_numbering() if implicit_index else self
        molecule = molecule._convert_nan_int() if convert_nan_int else molecule

        content = molecule.to_string(index=not implicit_index,
                                     float_format=float_format, header=header)

        # TODO the following might be removed in the future
        # introduced because of formatting bug in pandas
        # See https://github.com/pandas-dev/pandas/issues/13032
        if not header:
            space = ' ' * (molecule.loc[:, 'atom'].str.len().max()
                           - len(molecule.iloc[0, 0]))
            output = space + content
        else:
            output = content

        if buf is not None:
            if overwrite:
                with open(buf, mode='w') as f:
                    f.write(output)
            else:
                with open(buf, mode='x') as f:
                    f.write(output)
        else:
            return output

    def write(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.to_zmat`
        """
        message = 'Will be removed in the future. Please use to_zmat().'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.to_zmat(*args, **kwargs)
