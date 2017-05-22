# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import pandas as pd
from threading import Thread
import subprocess
import os
import tempfile
import warnings
from chemcoord.cartesian_coordinates._cartesian_class_core \
    import Cartesian_core
from chemcoord.configuration import settings
import io
from io import open
import re


class Cartesian_io(Cartesian_core):
    """This class provides IO-methods.

    Contains ``write_filetype`` and ``read_filetype`` methods
    like ``write_xyz()`` and ``read_xyz()``.

    The generic functions ``read`` and ``write``
    figure out themselves what the filetype is and use the
    appropiate IO-method.

    The ``view`` method uses external viewers to display a temporarily
    written xyz-file.
    """
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
        create_string = '{n}\n{message}\n{alignment}{frame_string}'.format

        # TODO automatically insert last stable version
        message = 'Created by chemcoord \
http://chemcoord.readthedocs.io/en/latest/'

        if sort_index:
            molecule_string = self.sort_index().to_string(
                header=header, index=index, float_format=float_format)
        else:
            molecule_string = self.to_string(header=header, index=index,
                                             float_format=float_format)

        # TODO the following might be removed in the future
        # introduced because of formatting bug in pandas
        # See https://github.com/pandas-dev/pandas/issues/13032
        space = ' ' * (self.loc[:, 'atom'].str.len().max()
                       - len(self.iloc[0, 0]))

        output = create_string(n=self.n_atoms, message=message,
                               alignment=space,
                               frame_string=molecule_string)

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
    def read_xyz(cls, inputfile, start_index=0, get_bonds=True):
        """Read a file of coordinate information.

        Reads xyz-files.

        Args:
            inputfile (str):
            start_index (int):
            get_bonds (bool):

        Returns:
            Cartesian:
        """
        frame = pd.read_table(inputfile, skiprows=2, comment='#',
                              delim_whitespace=True,
                              names=['atom', 'x', 'y', 'z'])

        molecule = cls(frame)
        molecule.index = range(start_index, start_index + molecule.n_atoms)

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
