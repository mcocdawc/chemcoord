# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import os
import subprocess
import tempfile
import warnings
from io import open  # pylint:disable=redefined-builtin
from threading import Thread

import pandas as pd

from chemcoord._generic_classes.generic_IO import GenericIO
from chemcoord.cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord.configuration import settings


class CartesianIO(CartesianCore, GenericIO):
    """This class provides IO-methods.

    Contains ``write_filetype`` and ``read_filetype`` methods
    like ``write_xyz()`` and ``read_xyz()``.

    The generic functions ``read`` and ``write``
    figure out themselves what the filetype is and use the
    appropiate IO-method.

    The ``view`` method uses external viewers to display a temporarily
    written xyz-file.
    """
    def __repr__(self):
        return self._frame.__repr__()

    def _repr_html_(self):
        new = self._sympy_formatter()

        def insert_before_substring(insert_txt, substr, txt):
            "Under the assumption that substr only appears once."
            return (insert_txt + substr).join(txt.split(substr))
        html_txt = new._frame._repr_html_()
        insert_txt = '<caption>{}</caption>\n'.format(self.__class__.__name__)
        return insert_before_substring(insert_txt, '<thead>', html_txt)

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None,
                  float_format=None, sparsify=None, index_names=True,
                  justify=None, line_width=None, max_rows=None,
                  max_cols=None, show_dimensions=False):
        """Render a DataFrame to a console-friendly tabular output.

        Wrapper around the :meth:`pandas.DataFrame.to_string` method.
        """
        return self._frame.to_string(
            buf=buf, columns=columns, col_space=col_space, header=header,
            index=index, na_rep=na_rep, formatters=formatters,
            float_format=float_format, sparsify=sparsify,
            index_names=index_names, justify=justify, line_width=line_width,
            max_rows=max_rows, max_cols=max_cols,
            show_dimensions=show_dimensions)

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

    def to_xyz(self, buf=None, sort_index=True,
               index=False, header=False, float_format='{:.6f}'.format,
               overwrite=True):
        """Write xyz-file

        Args:
            buf (str): StringIO-like, optional buffer to write to
            sort_index (bool): If sort_index is true, the
                :class:`~chemcoord.Cartesian`
                is sorted by the index before writing.
            float_format (one-parameter function): Formatter function
                to apply to column’s elements if they are floats.
                The result of this function must be a unicode string.
            overwrite (bool): May overwrite existing files.

        Returns:
            formatted : string (or unicode, depending on data and options)
        """
        if sort_index:
            molecule_string = self.sort_index().to_string(
                header=header, index=index, float_format=float_format)
        else:
            molecule_string = self.to_string(header=header, index=index,
                                             float_format=float_format)

        # NOTE the following might be removed in the future
        # introduced because of formatting bug in pandas
        # See https://github.com/pandas-dev/pandas/issues/13032
        space = ' ' * (self.loc[:, 'atom'].str.len().max()
                       - len(self.iloc[0, 0]))

        output = '{n}\n{message}\n{alignment}{frame_string}'.format(
            n=len(self), alignment=space, frame_string=molecule_string,
            message='Created by chemcoord http://chemcoord.readthedocs.io/')

        if buf is not None:
            if overwrite:
                with open(buf, mode='w') as f:
                    f.write(output)
            else:
                with open(buf, mode='x') as f:
                    f.write(output)
        else:
            return output

    def write_xyz(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Cartesian.to_xyz`
        """
        message = 'Will be removed in the future. Please use to_xyz().'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.to_xyz(*args, **kwargs)

    @classmethod
    def read_xyz(cls, inputfile, start_index=0, get_bonds=True,
                 nrows=None, engine=None):
        """Read a file of coordinate information.

        Reads xyz-files.

        Args:
            inputfile (str):
            start_index (int):
            get_bonds (bool):
            nrows (int): Number of rows of file to read.
                Note that the first two rows are implicitly excluded.
            engine (str): Wrapper for argument of :func:`pandas.read_csv`.

        Returns:
            Cartesian:
        """
        frame = pd.read_table(inputfile, skiprows=2, comment='#',
                              nrows=nrows,
                              delim_whitespace=True,
                              names=['atom', 'x', 'y', 'z'], engine=engine)

        molecule = cls(frame)
        molecule.index = range(start_index, start_index + len(molecule))

        if get_bonds:
            molecule.get_bonds(use_lookup=False, set_lookup=True)
        return molecule

    def view(self, viewer=settings['defaults']['viewer'], use_curr_dir=False):
        """View your molecule.

        .. note:: This function writes a temporary file and opens it with
            an external viewer.
            If you modify your molecule afterwards you have to recall view
            in order to see the changes.

        Args:
            viewer (str): The external viewer to use. The default is
                specified in cc.settings.settings['viewer']
            use_curr_dir (bool): If True, the temporary file is written to
                the current diretory. Otherwise it gets written to the
                OS dependendent temporary directory.

        Returns:
            None:
        """
        if use_curr_dir:
            TEMP_DIR = os.path.curdir
        else:
            TEMP_DIR = tempfile.gettempdir()

        def give_filename(i):
            filename = 'ChemCoord_' + str(i) + '.xyz'
            return os.path.join(TEMP_DIR, filename)

        i = 1
        while os.path.exists(give_filename(i)):
            i = i + 1
        self.to_xyz(give_filename(i))

        def open_file(i):
            """Open file and close after being finished."""
            try:
                subprocess.check_call([viewer, give_filename(i)])
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise
            finally:
                if use_curr_dir:
                    pass
                else:
                    os.remove(give_filename(i))

        Thread(target=open_file, args=(i,)).start()

    def get_pymatgen_molecule(self):
        """Create a Molecule instance of the pymatgen library

        .. warning:: The `pymatgen library <http://pymatgen.org>`_ is imported
            locally in this function and will raise
            an ``ImportError`` exception, if it is not installed.

        Args:
            None

        Returns:
            :class:`pymatgen.core.structure.Molecule`:
        """
        from pymatgen import Molecule
        return Molecule(self['atom'].values,
                        self.loc[:, ['x', 'y', 'z']].values)

    @classmethod
    def from_pymatgen_molecule(cls, molecule):
        """Create an instance of the own class from a pymatgen molecule

        Args:
            molecule (:class:`pymatgen.core.structure.Molecule`):

        Returns:
            Cartesian:
        """
        new = cls(atoms=[el.value for el in molecule.species],
                  coords=molecule.cart_coords)
        return new._to_numeric()

    def get_ase_atoms(self):
        """Create an Atoms instance of the ase library

        .. warning:: The `ase library <https://wiki.fysik.dtu.dk/ase/>`_
            is imported locally in this function and will raise
            an ``ImportError`` exception, if it is not installed.

        Args:
            None

        Returns:
            :class:`ase.atoms.Atoms`:
        """
        from ase import Atoms
        return Atoms(''.join(self['atom']), self.loc[:, ['x', 'y', 'z']])

    @classmethod
    def from_ase_atoms(cls, atoms):
        """Create an instance of the own class from an ase molecule

        Args:
            molecule (:class:`ase.atoms.Atoms`):

        Returns:
            Cartesian:
        """
        return cls(atoms=atoms.get_chemical_symbols(), coords=atoms.positions)
