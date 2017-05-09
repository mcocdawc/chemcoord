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
    _possible_filetypes = frozenset(['xyz'])
    _default_filetype = 'xyz'

    @staticmethod
    def determine_filetype(filepath):
        """Determine filetype

        The charakters after the last point are interpreted as the filetype.

        Args:
            filepath (str):

        Returns:
            str:
        """
        filetype = re.split('\.', filepath)[-1]
        return filetype

    def write(self, outputfile=None, filetype='xyz', **kwargs):
        """Write Cartesian into a file.

        This is the generic function for file writing.
        Depending on ``filetype`` the possible keyword arguments
        may differ because different writing methods are used.
        The following filetypes are implemented:

        ``'auto'``
            The method :meth:`~Cartesian.determine_filetype()` is used
            to guess filetype.
        ``'xyz'``
            Uses :meth:`~Cartesian.write_xyz` to write file.

        .. note:: Since it permamently writes a file, this function
            is strictly speaking **not sideeffect free**.
            The :class:`~chemcoord.Cartesian`
            to be written is of course not changed.

        Args:
            outputfile (str): If ``'outputfile'`` is ``None``,
                the file is not written, but the text/bytestream is returned.
            filetype (str):

        Returns:
            Depending :
        """
        if filetype == 'auto':
            filetype = self.determine_filetype(outputfile)

        if filetype == 'xyz':
            self.write_xyz(outputfile, **kwargs)
        else:
            error_string = 'The desired filetype is not implemented'
            raise NotImplementedError(error_string)

    # TODO outputfile is None
    def write_xyz(self, outputfile=None, sort_index=True,
                  index=False, header=False, float_format=None):
        """Write xyz-file

        Args:
            outputfile (str): If ``'outputfile'`` is ``None``,
                the file is not written, but the formatted string is returned.
            sort_index (bool): If sort_index is true, the
                :class:`~chemcoord.Cartesian`
                is sorted by the index before writing.
            float_format (one-parameter function): Formatter function
                to apply to columns’ elements if they are floats,
                default None.
                The result of this function must be a unicode string.

        Returns:
            string : If ``outputfile`` is given ``None`` is returned.
        """
        create_string = '{n}\n{message}\n{alignment}{frame_string}'.format

        # TODO automatically insert last stable version
        message = 'Created by chemcoord \
http://chemcoord.readthedocs.io/en/latest/'

        if sort_index:
            molecule_string = self.sort_index().write_string(
                header=header, index=index, float_format=float_format)
        else:
            molecule_string = self.write_string(header=header, index=index,
                                             float_format=float_format)

        def give_alignment_space(self):
            space = ' ' * (self[:, 'atom'].str.len().max()
                           - len(self.frame.iloc[0, 0]))
            return space

        output = create_string(n=self.n_atoms, message=message,
                               alignment=give_alignment_space(self),
                               frame_string=molecule_string)

        if outputfile is not None:
            with open(outputfile, mode='w') as f:
                f.write(output)
        else:
            return output

    @classmethod
    def read(cls, inputfile, filetype='auto', **kwargs):
        """Read a file of coordinate information.

        This is the generic function for file reading.
        Depending on ``filetype`` the possible keyword arguments
        and return types may differ because different
        parsing methods are used.
        The following filetypes are implemented:

        ``'auto'``
            The method :meth:`~Cartesian.determine_filetype()` is used
            to guess filetype.
        ``'xyz'``
            Uses :meth:`~Cartesian.read_xyz` to read file.
        ``'molden'``
            Uses :meth:`~Cartesian.read_molden` to read file.

        Args:
            inputfile (str):
            filetype (str):

        Returns:
            depending : Depending on type of file returns different objects.
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
        """Read a file of coordinate information.

        Reads xyz-files.

        Args:
            inputfile (str):
            pythonic_index (bool):
            get_bonds (bool):

        Returns:
            Cartesian:
        """
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
