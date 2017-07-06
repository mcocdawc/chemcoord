# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord.internal_coordinates._zmat_class_core import ZmatCore
from chemcoord.constants import int_label
import numpy as np
import pandas as pd
import warnings


class ZmatIO(ZmatCore):
    def to_string(self, buf=None, upper_triangle='string', columns=None,
                  col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None,
                  float_format=None, sparsify=None, index_names=True,
                  justify=None, line_width=None, max_rows=None,
                  max_cols=None, show_dimensions=False):
        """Render a DataFrame to a console-friendly tabular output.

        Wrapper around the :meth:`pandas.DataFrame.to_string` method.
        """
        def upper_triangle_with_strings():
            new = self._frame.replace(to_replace=int_label.values(),
                                      value=int_label.keys())
            return new
        def upper_triangle_with_latex():
            new = self._frame.replace(to_replace=int_label.values(),
                                      value=int_label.keys())
            return new

        content = frame.to_string(
            buf=buf, columns=columns, col_space=col_space, header=header,
            index=index, na_rep=na_rep, formatters=formatters,
            float_format=float_format, sparsify=sparsify,
            index_names=index_names, justify=justify, line_width=line_width,
            max_rows=max_rows, max_cols=max_cols,
            show_dimensions=show_dimensions)
        return content

    def to_latex(self, buf=None, columns=None, col_space=None, header=True,
                 index=True, na_rep='NaN', formatters=None, float_format=None,
                 sparsify=None, index_names=True, bold_rows=True,
                 column_format=None, longtable=None, escape=None,
                 encoding=None, decimal='.', multicolumn=None,
                 multicolumn_format=None, multirow=None):
        """Render a DataFrame to a tabular environment table.

        You can splice this into a LaTeX document.
        Requires ``\\usepackage{booktabs}``.
        Wrapper around the :meth:`pandas.DataFrame.to_latex` method.
        """
        return self._frame.to_latex(
            buf=buf, columns=columns, col_space=col_space, header=header,
            index=index, na_rep=na_rep, formatters=formatters,
            float_format=float_format, sparsify=sparsify,
            index_names=index_names, bold_rows=bold_rows,
            column_format=column_format, longtable=longtable, escape=escape,
            encoding=encoding, decimal=decimal, multicolumn=multicolumn,
            multicolumn_format=multicolumn_format, multirow=multirow)

    @classmethod
    def read_zmat(cls, inputfile, implicit_index=True):
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
        cols = ['atom', 'b', 'bond', 'a', 'angle',
                'd', 'dihedral']
        if implicit_index:
            zmat_frame = pd.read_table(inputfile, comment='#',
                                       delim_whitespace=True,
                                       names=cols)
            Zmat = cls(zmat_frame)
            Zmat.index = range(1, len(Zmat) + 1)
        else:
            zmat_frame = pd.read_table(inputfile, comment='#',
                                       delim_whitespace=True,
                                       names=['temp_index'] + cols)
            Zmat = cls(zmat_frame)
            Zmat.set_index('temp_index', drop=True, inplace=True)
            Zmat.index.name = None
        return Zmat

    def to_zmat(self, buf=None, implicit_index=True,
                float_format='{:.6f}'.format, overwrite=True,
                header=False):
        """Write zmat-file

        Args:
            buf (str): StringIO-like, optional buffer to write to
            implicit_index (bool): If implicit_index is set, the zmat indexing
                is changed to ``range(1, len(self) + 1)``.
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
        if implicit_index:
            molecule = self.change_numbering(new_index=range(1, len(self) + 1))

        content = molecule.to_string(index=(not implicit_index),
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
