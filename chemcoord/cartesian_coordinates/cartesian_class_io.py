from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
try:
    # import itertools.imap as map
    import itertools.izip as zip
except ImportError:
    pass
import numpy as np
import pandas as pd
import collections
from threading import Thread
import subprocess
import os
import tempfile
import warnings
from chemcoord._generic_classes._common_class import _common_class
from chemcoord._exceptions import PhysicalMeaningError
from chemcoord.algebra_utilities import utilities
from chemcoord.cartesian_coordinates.cartesian_class_core import Cartesian_core
from chemcoord.configuration.configuration import settings
import io
from io import open
import re


def pick(my_set):
    """Return one element from a set.

    **Do not** make any assumptions about the element to be returned.
    ``pick`` just returns a random element,
    could be the same, could be different.
    """
    assert type(my_set) is set, 'Pick can be applied only on sets.'
    x = my_set.pop()
    my_set.add(x)
    return x


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
    @staticmethod
    def determine_filetype(filepath):
        filetype = re.split('\.', filepath)[-1]
        return filetype

    def write(self, outputfile, sort_index=True, filetype='xyz'):
        """Write the Cartesian into a file.

        .. note:: Since it permamently writes a file, this function
            is strictly speaking **not sideeffect free**.
            The frame to be written is of course not changed.

        Args:
            outputfile (str):
            sort_index (bool): If sort_index is true, the Cartesian
                is sorted by the index before writing.
            filetype (str): The filetype to be used.
                The default is xyz.
                Supported filetypes are: 'xyz'

        Returns:
            None: None
        """
        if filetype == 'auto':
            filetype = self.determine_filetype(outputfile)

        if filetype == 'xyz':
            self.write_xyz(outputfile, sort_index)
        else:
            error_string = 'The desired filetype is not implemented'
            raise NotImplementedError(error_string)

    def write_xyz(self, outputfile, sort_index):
        frame = self.frame[['atom', 'x', 'y', 'z']].copy()
        if sort_index:
            frame = frame.sort_index()
            n_atoms = frame.shape[0]
            with open(outputfile, mode='w') as f:
                f.write(str(n_atoms) + 2 * '\n')
            frame.to_csv(
                outputfile,
                sep=str(' '),
                index=False,
                header=False,
                mode='a')
        else:
            frame = frame.sort_values(by='atom')
            n_atoms = frame.shape[0]
            with open(outputfile, mode='w') as f:
                f.write(str(n_atoms) + 2 * '\n')
            frame.to_csv(
                outputfile,
                sep=str(' '),
                # https://github.com/pydata/pandas/issues/6035
                index=False,
                header=False,
                mode='a')

    @classmethod
    def read(cls, inputfile, filetype='auto', **kwargs):
        """Read a file of coordinate information.

        +------------+------------+-----------+
        | Header 1   | Header 2   | Header 3  |
        +============+============+===========+
        | body row 1 | column 2   | column 3  |
        +------------+------------+-----------+
        | body row 2 | Cells may span columns.|
        +------------+------------+-----------+
        | body row 3 | Cells may  | - Cells   |
        +------------+ span rows. | - contain |
        | body row 4 |            | - blocks. |
        +------------+------------+-----------+

        Args:
            inputfile (str):
            pythonic_index (bool):
            filetype (str): The filetype to be read from.
                The default is ``'auto'``.
                Supported filetypes are: xyz and molden.
                With the option 'auto'  ``determine_filetype()`` is used.
                the charakters after the last point as filetype.

        Returns:
            Cartesian: Depending on the type of file returns a Cartesian,
            or a list of Cartesians.
        """
        if filetype == 'auto':
            filetype = cls.determine_filetype(inputfile)

        if filetype == 'xyz':
            molecule = cls.read_xyz(inputfile, **kwargs)
        elif filetype == 'molden':
            molecule = cls.read_molden(inputfile, **kwargs)
        else:
            error_string = 'The desired filetype is not implemented'
            raise NotImplementedError(error_string)
        return molecule

    @classmethod
    def read_xyz(cls, inputfile, pythonic_index=False, get_bonds=True):
        frame = pd.read_table(
            inputfile,
            skiprows=2,
            comment='#',
            delim_whitespace=True,
            names=['atom', 'x', 'y', 'z'])

        if not pythonic_index:
            n_atoms = frame.shape[0]
            frame.index = range(1, n_atoms+1)

        molecule = cls(frame)
        if get_bonds:
            previous_warnings_bool = settings['show_warnings']['valency']
            settings['show_warnings']['valency'] = False
            molecule.get_bonds(
                use_lookup=False, set_lookup=True, use_valency=False)
            settings['show_warnings']['valency'] = previous_warnings_bool
        return molecule

    @classmethod
    def read_molden(cls, inputfile, pythonic_index=False, get_bonds=True):
        """Read a molden file.

        Args:
            inputfile (str):
            pythonic_index (bool):

        Returns:
            list: A list containing Cartesian is returned.
        """
        f = open(inputfile, 'r')

        found = False
        while not found:
            line = f.readline()
            if line.strip() == '[N_GEO]':
                found = True
                number_of_molecules = int(f.readline().strip())

        found = False
        while not found:
            line = f.readline()
            if line.strip() == '[GEOMETRIES] (XYZ)':
                found = True
                current_line = f.tell()
                number_of_atoms = int(f.readline().strip())
                f.seek(current_line)

        for i in range(number_of_molecules):
            molecule_in = [f.readline()
                           for j in range(number_of_atoms + 2)]
            molecule_in = ''.join(molecule_in)
            molecule_in = io.StringIO(molecule_in)
            molecule = cls.read(molecule_in, pythonic_index=pythonic_index,
                                get_bonds=get_bonds, filetype='xyz')
            try:
                list_of_cartesians.append(molecule)
            except NameError:
                list_of_cartesians = [molecule]

        f.close()
        return list_of_cartesians

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
        self.write(give_filename(i))

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
