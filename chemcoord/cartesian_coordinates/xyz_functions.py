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
from chemcoord._exceptions import PhysicalMeaningError
from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord.zmat import zmat_functions
from chemcoord.zmat.Zmat import Zmat
from chemcoord import export
from chemcoord.configuration.configuration import settings
import io
from io import open
import re


@export
def view(molecule, viewer=settings['defaults']['viewer'], use_curr_dir=False):
    """View your molecule or list of molecules.

    .. note:: This function writes a temporary file and opens it with
        an external viewer.
        If you modify your molecule afterwards you have to recall view
        in order to see the changes.

    Args:
        molecule: Can be a cartesian, or a list of cartesians.
        viewer (str): The external viewer to use. The default is
            specified in settings.viewer
        use_curr_dir (bool): If True, the temporary file is written to
            the current diretory. Otherwise it gets written to the
            OS dependendent temporary directory.

    Returns:
        None:
    """
    try:
        molecule.view(viewer=viewer, use_curr_dir=use_curr_dir)
    except AttributeError:
        if use_curr_dir:
            TEMP_DIR = os.path.curdir
        else:
            TEMP_DIR = tempfile.gettempdir()

        def give_filename(i):
            filename = 'ChemCoord_list_' + str(i) + '.molden'
            return os.path.join(TEMP_DIR, filename)

        i = 1
        while os.path.exists(give_filename(i)):
            i = i + 1

        write(molecule, give_filename(i), filetype='molden')

        def open(i):
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
        Thread(target=open, args=(i,)).start()


@export
def write(to_be_written, outputfile, sort_index=True, filetype='auto'):
    """Write the coordinates into a file.

    .. note:: Since it permamently writes a file, this function is
        strictly speaking **not sideeffect free**.
        The frame to be written is of course not changed.

    Args:
        to_be_written (Cartesian): This can be a :class:`Cartesian` for
            xyz files
            and a :class:`list` of :class:`Cartesian` for molden files.
        outputfile (str):
        sort_index (bool): If sort_index is true, the Cartesian
            is sorted by the index before writing.
        filetype (str): The filetype to be used.
            The default is auto.
            Supported filetypes are: 'xyz' and 'molden'.
            'auto' uses the charakters after the last point as filetype.

    Returns:
        None:
    """
    def write_xyz(to_be_written, outputfile, sort_index):
        frame = to_be_written.frame[['atom', 'x', 'y', 'z']].copy()
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
                sep=str(' '),  # https://github.com/pydata/pandas/issues/6035
                index=False,
                header=False,
                mode='a')

    def write_molden(to_be_written, outputfile, sort_index):
        """Write a list of Cartesians into a molden file.

        .. note:: Since it permamently writes a file, this function
            is strictly speaking **not sideeffect free**.
            The frame to be written is of course not changed.

        Args:
            cartesian_list (list):
            outputfile (str):
            sort_index (bool): If sort_index is true, the Cartesian
                is sorted by the index before writing.

        Returns:
            None:
        """
        if sort_index:
            framelist = [molecule.sort_index().frame
                         for molecule in to_be_written]
        else:
            framelist = [molecule.frame for molecule in to_be_written]
        n_frames = len(framelist)
        n_atoms = to_be_written[0].n_atoms
        values = n_frames * '1\n'
        string = ("[MOLDEN FORMAT]\n"
                  + "[N_GEO]\n"
                  + str(n_frames) + "\n"
                  + '[GEOCONV]\n'
                  + 'energy\n' + values
                  + 'max-force\n' + values
                  + 'rms-force\n' + values
                  + '[GEOMETRIES] (XYZ)\n')

        with open(outputfile, mode='w') as f:
            f.write(string)

        for frame in framelist:
            frame = frame.sort_index()
            n_atoms = frame.shape[0]
            with open(outputfile, mode='a') as f:
                f.write(str(n_atoms) + 2 * '\n')
            frame.to_csv(
                outputfile,
                sep=str(' '),
                index=False,
                header=False,
                mode='a')

    if filetype == 'auto':
        filetype = re.split('\.', outputfile)[-1]

    if filetype == 'xyz':
        write_xyz(to_be_written, outputfile, sort_index)
    elif filetype == 'molden':
        write_molden(to_be_written, outputfile, sort_index)
    else:
        raise NotImplementedError('The desired filetype is not implemented')


@export
def read(inputfile, pythonic_index=False, get_bonds=True, filetype='auto'):
    """Read a file of coordinate information.

    .. note:: This function calls in the background :func:`Cartesian.read`.
        If you inherited from :class:`Cartesian` to tailor it for your project,
        you have to use this method as a constructor.
        Otherwise you can choose.

    Args:
        inputfile (str):
        pythonic_index (bool):
        filetype (str): The filetype to be read from.
            The default is xyz.
            Supported filetypes are: xyz and molden.
            'auto' uses the charakters after the last point as filetype.

    Returns:
        Cartesian: Depending on the type of file returns a Cartesian,
        or a list of Cartesians.
    """
    molecule = Cartesian.read(inputfile, pythonic_index=pythonic_index,
                              get_bonds=get_bonds, filetype=filetype)
    return molecule


@export
def isclose(a, b, align=True, rtol=1.e-5, atol=1.e-8):
    """Compare two molecules for numerical equality.

    Args:
        a (Cartesian):
        b (Cartesian):
        align (bool): a and b are
            prealigned along their principal axes of inertia and moved to their
            barycenters before comparing.
        rtol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for furter explanation.
        atol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for furter explanation.

    Returns:
        bool:
    """
    pretest = (set(a.index) == set(b.index)
               and np.alltrue(a[:, 'atom'] == b[a.index, 'atom']))

    if align and pretest:
        A = a.inertia()['transformed_Cartesian'].location()
        B = b.inertia()['transformed_Cartesian'][a.index, :].location()
        return np.allclose(A, B, rtol=rtol, atol=atol)
    elif pretest:
        A = a.location()
        B = b[a.index, :].location()
        return np.allclose(A, B, rtol=rtol, atol=atol)
    else:
        return False


@export
def is_Cartesian(possible_Cartesian):
    """Tests, if given instance is a Cartesian.

    Args:
        is_Cartesian (any type):

    Returns:
        bool:
    """
    columns = possible_Cartesian.columns.copy()
    try:
        assert type(columns) is not str
        columns = set(columns)
    except (TypeError, AssertionError):
        columns = set([columns])
    return {'atom', 'x', 'y', 'z'} <= columns
