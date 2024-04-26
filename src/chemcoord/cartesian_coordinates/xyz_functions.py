# -*- coding: utf-8 -*-
import math as m
import os
import subprocess
import tempfile
import warnings
from threading import Thread

import numba as nb
import numpy as np
import pandas as pd
import sympy
from chemcoord.configuration import settings
from numba import jit


def view(molecule, viewer=None, use_curr_dir=False):
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
    viewer = settings['defaults']['viewer']
    if hasattr(molecule, 'view'):
        molecule.view(viewer=viewer, use_curr_dir=use_curr_dir)
    else:
        if pd.api.types.is_list_like(molecule):
            cartesian_list = molecule
        else:
            raise ValueError('Argument is neither list nor Cartesian.')
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

        to_molden(cartesian_list, buf=give_filename(i))

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


def to_molden(cartesian_list, buf=None, sort_index=True,
              overwrite=True, float_format='{:.6f}'.format):
    """Write a list of Cartesians into a molden file.

    .. note:: Since it permamently writes a file, this function
        is strictly speaking **not sideeffect free**.
        The list to be written is of course not changed.

    Args:
        cartesian_list (list):
        buf (str): StringIO-like, optional buffer to write to
        sort_index (bool): If sort_index is true, the Cartesian
            is sorted by the index before writing.
        overwrite (bool): May overwrite existing files.
        float_format (one-parameter function): Formatter function
            to apply to column’s elements if they are floats.
            The result of this function must be a unicode string.

    Returns:
        formatted : string (or unicode, depending on data and options)
    """
    if sort_index:
        cartesian_list = [molecule.sort_index() for molecule in cartesian_list]

    give_header = ("[MOLDEN FORMAT]\n"
                   + "[N_GEO]\n"
                   + str(len(cartesian_list)) + "\n"
                   + '[GEOCONV]\n'
                   + 'energy\n{energy}'
                   + 'max-force\n{max_force}'
                   + 'rms-force\n{rms_force}'
                   + '[GEOMETRIES] (XYZ)\n').format

    values = len(cartesian_list) * '1\n'
    energy = [str(m.metadata.get('energy', 1)) for m in cartesian_list]
    energy = '\n'.join(energy) + '\n'

    header = give_header(energy=energy, max_force=values, rms_force=values)

    coordinates = [x.to_xyz(sort_index=sort_index, float_format=float_format)
                   for x in cartesian_list]
    output = header + '\n'.join(coordinates)

    if buf is not None:
        if overwrite:
            with open(buf, mode='w') as f:
                f.write(output)
        else:
            with open(buf, mode='x') as f:
                f.write(output)
    else:
        return output


def write_molden(*args, **kwargs):
    """Deprecated, use :func:`~chemcoord.xyz_functions.to_molden`
    """
    message = 'Will be removed in the future. Please use to_molden().'
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, DeprecationWarning)
    return to_molden(*args, **kwargs)


def read_molden(inputfile, start_index=0, get_bonds=True):
    """Read a molden file.

    Args:
        inputfile (str):
        start_index (int):

    Returns:
        list: A list containing :class:`~chemcoord.Cartesian` is returned.
    """
    from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian
    with open(inputfile, 'r') as f:
        found = False
        while not found:
            line = f.readline()
            if '[N_GEO]' in line:
                found = True
                number_of_molecules = int(f.readline().strip())

        energies = []
        found = False
        while not found:
            line = f.readline()
            if 'energy' in line:
                found = True
                for _ in range(number_of_molecules):
                    energies.append(float(f.readline().strip()))

        found = False
        while not found:
            line = f.readline()
            if '[GEOMETRIES] (XYZ)' in line:
                found = True
                current_line = f.tell()
                number_of_atoms = int(f.readline().strip())
                f.seek(current_line)

        cartesians = []
        for energy in energies:
            cartesian = Cartesian.read_xyz(
                f, start_index=start_index, get_bonds=get_bonds,
                nrows=number_of_atoms, engine='python')
            cartesian.metadata['energy'] = energy
            cartesians.append(cartesian)
    return cartesians


def isclose(a, b, align=False, rtol=1.e-5, atol=1.e-8):
    """Compare two molecules for numerical equality.

    Args:
        a (Cartesian):
        b (Cartesian):
        align (bool): a and b are
            prealigned along their principal axes of inertia and moved to their
            barycenters before comparing.
        rtol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for further explanation.
        atol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for further explanation.

    Returns:
        :class:`numpy.ndarray`: Boolean array.
    """
    coords = ['x', 'y', 'z']
    if not (set(a.index) == set(b.index)
            and (a.loc[:, 'atom'] == b.loc[a.index, 'atom']).all(axis=None)):
        message = 'Can only compare molecules with the same atoms and labels'
        raise ValueError(message)

    if align:
        a = a.get_inertia()['transformed_Cartesian']
        b = b.get_inertia()['transformed_Cartesian']
    A, B = a.loc[:, coords], b.loc[a.index, coords]

    out = pd.DataFrame(index=a.index, columns=['atom'] + coords, dtype=bool)
    out.loc[:, 'atom'] = True
    out.loc[:, coords] = np.isclose(A, B, rtol=rtol, atol=atol)
    return out


def allclose(a, b, align=False, rtol=1.e-5, atol=1.e-8):
    """Compare two molecules for numerical equality.

    Args:
        a (Cartesian):
        b (Cartesian):
        align (bool): a and b are
            prealigned along their principal axes of inertia and moved to their
            barycenters before comparing.
        rtol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.allclose` for further explanation.
        atol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.allclose` for further explanation.

    Returns:
        bool:
    """
    return isclose(a, b, align=align, rtol=rtol, atol=atol).all(axis=None)


def concat(cartesians, ignore_index=False, keys=None):
    """Join list of cartesians into one molecule.

    Wrapper around the :func:`pandas.concat` function.
    Default values are the same as in the pandas function except for
    ``verify_integrity`` which is set to true in case of this library.

    Args:
        cartesians (sequence): A sequence of :class:`~chemcoord.Cartesian`
            to be concatenated.
        ignore_index (sequence, bool, int): If it is a boolean, it
            behaves like in the description of
            :meth:`pandas.DataFrame.append`.
            If it is a sequence, it becomes the new index.
            If it is an integer,
            ``range(ignore_index, ignore_index + len(new))``
            becomes the new index.
        keys (sequence): If multiple levels passed, should contain tuples.
            Construct hierarchical index using the passed keys as
            the outermost level

    Returns:
        Cartesian:
    """
    frames = [molecule._frame for molecule in cartesians]
    new = pd.concat(frames, ignore_index=ignore_index, keys=keys,
                    verify_integrity=True)

    if type(ignore_index) is bool:
        new = pd.concat(frames, ignore_index=ignore_index, keys=keys,
                        verify_integrity=True)
    else:
        new = pd.concat(frames, ignore_index=True, keys=keys,
                        verify_integrity=True)
        if type(ignore_index) is int:
            new.index = range(ignore_index,
                              ignore_index + len(new))
        else:
            new.index = ignore_index
    return cartesians[0].__class__(new)


def dot(A, B):
    """Matrix multiplication between A and B

    This function is equivalent to ``A @ B``, which is unfortunately
    not possible under python 2.x.

    Args:
        A (sequence):
        B (sequence):

    Returns:
        sequence:
    """
    try:
        result = A.__matmul__(B)
        if result is NotImplemented:
            result = B.__rmatmul__(A)
    except AttributeError:
        result = B.__rmatmul__(A)
    return result


@jit(nopython=True, cache=True)
def _jit_isclose(a, b, atol=1e-5, rtol=1e-8):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


@jit(nopython=True, cache=True)
def _jit_allclose(a, b, atol=1e-5, rtol=1e-8):
    n, m = a.shape
    for i in range(n):
        for j in range(m):
            if np.abs(a[i, j] - b[i, j]) > (atol + rtol * np.abs(b[i, j])):
                return False
    return True


@jit(nb.f8[:](nb.f8[:], nb.f8[:]), nopython=True)
def _jit_cross(A, B):
    C = np.empty_like(A)
    C[0] = A[1] * B[2] - A[2] * B[1]
    C[1] = A[2] * B[0] - A[0] * B[2]
    C[2] = A[0] * B[1] - A[1] * B[0]
    return C


def normalize(vector):
    """Normalizes a vector
    """
    normed_vector = vector / np.linalg.norm(vector)
    return normed_vector


@jit(nopython=True, cache=True)
def _jit_normalize(vector):
    """Normalizes a vector
    """
    normed_vector = vector / np.linalg.norm(vector)
    return normed_vector


def get_rotation_matrix(axis, angle):
    """Returns the rotation matrix.

    This function returns a matrix for the counterclockwise rotation
    around the given axis.
    The Input angle is in radians.

    Args:
        axis (vector):
        angle (float):

    Returns:
        Rotation matrix (np.array):
    """
    axis = normalize(np.array(axis))
    if not (np.array([1, 1, 1]).shape) == (3, ):
        raise ValueError('axis.shape has to be 3')
    angle = float(angle)
    return _jit_get_rotation_matrix(axis, angle)


@jit(nopython=True, cache=True)
def _jit_get_rotation_matrix(axis, angle):
    """Returns the rotation matrix.

    This function returns a matrix for the counterclockwise rotation
    around the given axis.
    The Input angle is in radians.

    Args:
        axis (vector):
        angle (float):

    Returns:
        Rotation matrix (np.array):
    """
    axis = _jit_normalize(axis)
    a = m.cos(angle / 2)
    b, c, d = axis * m.sin(angle / 2)
    rot_matrix = np.empty((3, 3))
    rot_matrix[0, 0] = a**2 + b**2 - c**2 - d**2
    rot_matrix[0, 1] = 2. * (b * c - a * d)
    rot_matrix[0, 2] = 2. * (b * d + a * c)
    rot_matrix[1, 0] = 2. * (b * c + a * d)
    rot_matrix[1, 1] = a**2 + c**2 - b**2 - d**2
    rot_matrix[1, 2] = 2. * (c * d - a * b)
    rot_matrix[2, 0] = 2. * (b * d - a * c)
    rot_matrix[2, 1] = 2. * (c * d + a * b)
    rot_matrix[2, 2] = a**2 + d**2 - b**2 - c**2
    return rot_matrix


def orthonormalize_righthanded(basis):
    """Orthonormalizes righthandedly a given 3D basis.

    This functions returns a right handed orthonormalize_righthandedd basis.
    Since only the first two vectors in the basis are used, it does not matter
    if you give two or three vectors.

    Right handed means, that:

    .. math::

        \\vec{e_1} \\times \\vec{e_2} &= \\vec{e_3} \\\\
        \\vec{e_2} \\times \\vec{e_3} &= \\vec{e_1} \\\\
        \\vec{e_3} \\times \\vec{e_1} &= \\vec{e_2} \\\\

    Args:
        basis (np.array): An array of shape = (3,2) or (3,3)

    Returns:
        new_basis (np.array): A right handed orthonormalized basis.
    """
    v1, v2 = basis[:, 0], basis[:, 1]
    e1 = normalize(v1)
    e3 = normalize(np.cross(e1, v2))
    e2 = normalize(np.cross(e3, e1))
    return np.array([e1, e2, e3]).T


def get_kabsch_rotation(Q, P, weights=None):
    """Calculate the optimal rotation from ``P`` unto ``Q``.

    Using the Kabsch algorithm the optimal rotation matrix
    for the rotation of ``other`` unto ``self`` is calculated.
    The algorithm is described very well in
    `wikipedia <http://en.wikipedia.org/wiki/Kabsch_algorithm>`_.

    Args:
        other (Cartesian):

    Returns:
        :class:`~numpy.array`: Rotation matrix
    """
    # The general problem with weights is decribed in
    # https://en.wikipedia.org/wiki/Wahba%27s_problem
    # The problem with equal weights is described
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    # Naming of variables follows wikipedia article about the Kabsch algorithm
    if weights is None:
        A = P.T @ Q
    else:
        A = P.T @ np.diag(weights) @ Q
    # One can't initialize an array over its transposed
    V, S, W = np.linalg.svd(A)  # pylint:disable=unused-variable
    W = W.T
    d = np.linalg.det(W @ V.T)
    return W @ np.diag([1., 1., d]) @ V.T


def apply_grad_zmat_tensor(grad_C, construction_table, cart_dist):
    """Apply the gradient for transformation to Zmatrix space onto cart_dist.

    Args:
        grad_C (:class:`numpy.ndarray`): A ``(3, n, n, 3)`` array.
            The mathematical details of the index layout is explained in
            :meth:`~chemcoord.Cartesian.get_grad_zmat()`.
        construction_table (pandas.DataFrame): Explained in
            :meth:`~chemcoord.Cartesian.get_construction_table()`.
        cart_dist (:class:`~chemcoord.Cartesian`):
            Distortions in cartesian space.

    Returns:
        :class:`Zmat`: Distortions in Zmatrix space.
    """
    if (construction_table.index != cart_dist.index).any():
        message = "construction_table and cart_dist must use the same index"
        raise ValueError(message)
    from chemcoord.internal_coordinates.zmat_class_main import Zmat
    dtypes = [('atom', str),
                    ('b', str), ('bond', float),
                    ('a', str), ('angle', float),
                    ('d', str), ('dihedral', float)]

    new = pd.DataFrame(np.empty(len(construction_table), dtype=dtypes),
                              index=cart_dist.index)

    X_dist = cart_dist.loc[:, ['x', 'y', 'z']].values.T
    C_dist = np.tensordot(grad_C, X_dist, axes=([3, 2], [0, 1])).T
    if C_dist.dtype == np.dtype('i8'):
        C_dist = C_dist.astype('f8')
    try:
        C_dist[:, [1, 2]] = np.rad2deg(C_dist[:, [1, 2]])
    # Unevaluated symbolic expressions are remaining.
    # catches AttributeError as well, because this was
    # the raised exception before https://github.com/numpy/numpy/issues/13666
    except (AttributeError, TypeError):
        C_dist[:, [1, 2]] = sympy.deg(C_dist[:, [1, 2]])
        new = new.astype({k: 'O' for k in ['bond', 'angle', 'dihedral']})

    new.loc[:, ['b', 'a', 'd']] = construction_table
    new.loc[:, 'atom'] = cart_dist.loc[:, 'atom']
    new.loc[:, ['bond', 'angle', 'dihedral']] = C_dist
    return Zmat(new, _metadata={'last_valid_cartesian': cart_dist})
