import math as m
import os
import subprocess
import tempfile
import warnings
from collections.abc import Callable, Iterable, Sequence
from io import StringIO
from pathlib import Path
from threading import Thread
from typing import Literal, overload

import numpy as np
import pandas as pd
import sympy
from pandas.core.frame import DataFrame
from typing_extensions import assert_never

from chemcoord._cartesian_coordinates._cart_transformation import (
    _jit_normalize,
    normalize,
)
from chemcoord._cartesian_coordinates._cartesian_class_core import COORDS
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord._internal_coordinates.zmat_class_main import Zmat
from chemcoord._internal_coordinates.zmat_functions import _zmat_interpolate
from chemcoord._utilities._decorators import njit
from chemcoord.configuration import settings
from chemcoord.typing import Matrix, PathLike, Real, Tensor4D, Vector


def view(
    molecule: Cartesian | Iterable[Cartesian],
    viewer: PathLike | None = None,
    list_viewer_file: Literal["molden", "xyz"] | None = None,
    use_curr_dir: bool = False,
) -> None:
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
    if viewer is None:
        viewer = settings.defaults.viewer
    if list_viewer_file is None:
        list_viewer_file = settings.defaults.list_viewer_file
    if isinstance(molecule, Cartesian):
        molecule.view(viewer=viewer, use_curr_dir=use_curr_dir)
    elif isinstance(molecule, Iterable):
        cartesians = molecule
        if use_curr_dir:
            TEMP_DIR = Path(os.path.curdir)
        else:
            TEMP_DIR = Path(tempfile.gettempdir())

        def give_filename(i: int, suffix: Literal["molden", "xyz"]) -> Path:
            return TEMP_DIR / f"ChemCoord_list_{i}.{suffix}"

        i = 1
        while give_filename(i, list_viewer_file).exists():
            i = i + 1

        if list_viewer_file == "molden":
            to_molden(cartesians, buf=give_filename(i, list_viewer_file))
        elif list_viewer_file == "xyz":
            to_multiple_xyz(cartesians, buf=give_filename(i, list_viewer_file))
        else:
            assert_never("Invalid list_viewer_file.")

        def open_file(i: int) -> None:
            """Open file and close after being finished."""
            try:
                subprocess.check_call([viewer, give_filename(i, list_viewer_file)])
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise
            finally:
                if use_curr_dir:
                    pass
                else:
                    give_filename(i, list_viewer_file).unlink()

        Thread(target=open_file, args=(i,)).start()
    else:
        raise ValueError(
            "Argument is neither iterable of Cartesians nor Cartesian "
            f"but instead: {type(molecule)}"
        )


@overload
def to_multiple_xyz(
    cartesian_list: Iterable[Cartesian],
    buf: None = None,
    sort_index: bool = ...,
    overwrite: bool = ...,
    float_format: Callable[[float], str] = ...,
) -> str: ...


@overload
def to_multiple_xyz(
    cartesian_list: Iterable[Cartesian],
    buf: PathLike,
    sort_index: bool = ...,
    overwrite: bool = ...,
    float_format: Callable[[float], str] = ...,
) -> None: ...


def to_multiple_xyz(
    cartesian_list: Iterable[Cartesian],
    buf: PathLike | None = None,
    sort_index: bool = True,
    overwrite: bool = True,
    float_format: Callable[[float], str] = "{:.6f}".format,
) -> str | None:
    """Write a list of Cartesians into an xyz file.

    .. note:: Since it permamently writes a file, this function
        is strictly speaking **not sideeffect free**.
        The list to be written is of course not changed.

    Args:
        cartesian_list :
        buf : StringIO-like, optional buffer to write to
        sort_index : If sort_index is true, the Cartesian
            is sorted by the index before writing.
        overwrite : May overwrite existing files.
        float_format (one-parameter function): Formatter function
            to apply to column’s elements if they are floats.
            The result of this function must be a unicode string.
    """
    if sort_index:
        cartesian_list = [molecule.sort_index() for molecule in cartesian_list]

    output = ""
    for struct in cartesian_list:
        output += struct.to_xyz(float_format=float_format) + "\n"

    if buf is not None:
        with open(buf, mode="w" if overwrite else "x") as f:
            f.write(output)
        return None
    else:
        return output


def read_multiple_xyz(inputfile: PathLike, start_index: int = 0) -> list[Cartesian]:
    """Read a multiple-xyz file.

    Args:
        inputfile :
        start_index :

    Returns:
        list: A list containing :class:`~chemcoord.Cartesian` is returned.
    """
    with open(inputfile, "r") as f:
        strings = f.readlines()
        cartesians = []
        finished = False
        current_line = 0
        while not finished:
            molecule_len = int(strings[current_line])
            cartesians.append(
                Cartesian.read_xyz(
                    StringIO(
                        "".join(strings[current_line : current_line + molecule_len + 2])
                    ),
                    start_index=start_index,
                    nrows=molecule_len,
                    engine="python",
                )
            )
            current_line += 2 + molecule_len

            finished = current_line == len(strings)

    return cartesians


@overload
def to_molden(
    cartesians: Iterable[Cartesian],
    buf: None = None,
    sort_index: bool = ...,
    overwrite: bool = ...,
    float_format: Callable[[float], str] = ...,
) -> str: ...


@overload
def to_molden(
    cartesians: Iterable[Cartesian],
    buf: PathLike,
    sort_index: bool = ...,
    overwrite: bool = ...,
    float_format: Callable[[float], str] = ...,
) -> None: ...


def to_molden(
    cartesians: Iterable[Cartesian],
    buf: PathLike | None = None,
    sort_index: bool = True,
    overwrite: bool = True,
    float_format: Callable[[float], str] = "{:.6f}".format,
) -> str | None:
    """Write a list of Cartesians into a molden file.

    .. note:: Since it permamently writes a file, this function
        is strictly speaking **not sideeffect free**.
        The list to be written is of course not changed.

    Args:
        cartesian_list :
        buf : StringIO-like, optional buffer to write to
        sort_index : If sort_index is true, the Cartesian
            is sorted by the index before writing.
        overwrite : May overwrite existing files.
        float_format : Formatter function
            to apply to column’s elements if they are floats.
            The result of this function must be a unicode string.
    """
    if sort_index:
        cartesian_list = [molecule.sort_index() for molecule in cartesians]
    else:
        cartesian_list = list(cartesian_list)
    if not all(isinstance(molecule, Cartesian) for molecule in cartesian_list):
        raise TypeError("All elements in cartesians must be Cartesians.")

    give_header = (
        "[MOLDEN FORMAT]\n"
        + "[N_GEO]\n"
        + str(len(cartesian_list))
        + "\n"
        + "[GEOCONV]\n"
        + "energy\n{energy}"
        + "max-force\n{max_force}"
        + "rms-force\n{rms_force}"
        + "[GEOMETRIES] (XYZ)\n"
    ).format

    values = len(cartesian_list) * "1\n"
    energy = (
        "\n".join([str(m.metadata.get("energy", 1)) for m in cartesian_list]) + "\n"
    )

    header = give_header(energy=energy, max_force=values, rms_force=values)

    coordinates = [
        x.to_xyz(sort_index=sort_index, float_format=float_format)
        for x in cartesian_list
    ]
    output = header + "\n".join(coordinates)

    if buf is not None:
        with open(buf, mode="w" if overwrite else "x") as f:
            f.write(output)
        return None
    else:
        return output


def write_molden(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Deprecated, use :func:`~chemcoord.xyz_functions.to_molden`"""
    message = "Will be removed in the future. Please use to_molden()."
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, DeprecationWarning)
    return to_molden(*args, **kwargs)


def read_molden(inputfile: PathLike, start_index: int = 0) -> list[Cartesian]:
    """Read a molden file.

    Args:
        inputfile (str):
        start_index (int):

    Returns:
        list: A list containing :class:`~chemcoord.Cartesian` is returned.
    """
    with open(inputfile) as f:
        found = False
        while not found:
            line = f.readline()
            if "[N_GEO]" in line:
                found = True
                number_of_molecules = int(f.readline().strip())

        energies = []
        found = False
        while not found:
            line = f.readline()
            if "energy" in line:
                found = True
                for _ in range(number_of_molecules):
                    energies.append(float(f.readline().strip()))

        found = False
        while not found:
            line = f.readline()
            if "[GEOMETRIES] (XYZ)" in line:
                found = True
                current_line = f.tell()
                number_of_atoms = int(f.readline().strip())
                f.seek(current_line)

        cartesians = []
        for energy in energies:
            cartesian = Cartesian.read_xyz(
                f,
                start_index=start_index,
                nrows=number_of_atoms,
                engine="python",
            )
            cartesian.metadata["energy"] = energy
            cartesians.append(cartesian)
    return cartesians


def isclose(
    a: Cartesian,
    b: Cartesian,
    align: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> DataFrame:
    """Compare two molecules for numerical equality.

    Args:
        a :
        b :
        align : a and b are
            prealigned along their principal axes of inertia and moved to their
            barycenters before comparing.
        rtol : Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for further explanation.
        atol : Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for further explanation.
    """
    # The pandas documentation says about the arguments to all(axis=...)
    #   None : reduce all axes, return a scalar
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.all.html
    # but the stubs don't have it.
    if not (
        set(a.index) == set(b.index)
        and (a.loc[:, "atom"] == b.loc[a.index, "atom"]).all(axis=None)  # type: ignore[arg-type]
    ):
        message = "Can only compare molecules with the same atoms and labels"
        raise ValueError(message)

    if align:
        a = a.get_inertia()["transformed_Cartesian"]
        b = b.get_inertia()["transformed_Cartesian"]
    A, B = a.loc[:, COORDS].values, b.loc[a.index, COORDS].values

    out = pd.DataFrame(index=a.index, columns=["atom"] + COORDS, dtype=bool)
    out.loc[:, "atom"] = True
    out.loc[:, COORDS] = np.isclose(A, B, rtol=rtol, atol=atol)
    return out


def allclose(
    a: Cartesian,
    b: Cartesian,
    align: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> bool:
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


def concat(
    cartesians: Sequence[Cartesian],
    ignore_index: bool = False,
    keys: Iterable | None = None,
) -> Cartesian:
    """Join list of cartesians into one molecule.

    Wrapper around the :func:`pandas.concat` function.
    Default values are the same as in the pandas function except for
    ``verify_integrity`` which is set to true in case of this library.

    Args:
        cartesians (sequence): A sequence of :class:`~chemcoord.Cartesian`
            to be concatenated.
        ignore_index (bool): It
            behaves like in the description of
            :meth:`pandas.DataFrame.append`.
        keys (sequence): If multiple levels passed, should contain tuples.
            Construct hierarchical index using the passed keys as
            the outermost level

    Returns:
        Cartesian:
    """
    frames = [molecule._frame for molecule in cartesians]
    if keys is None:
        new = pd.concat(frames, ignore_index=ignore_index, verify_integrity=True)
    else:
        new = pd.concat(
            frames, ignore_index=ignore_index, keys=keys, verify_integrity=True
        )
    return cartesians[0].__class__(new)


def get_rotation_matrix(axis: Sequence[float], angle: float) -> Matrix[np.float64]:
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
    vaxis = normalize(np.asarray(axis))
    if not (vaxis.shape) == (3,):
        raise ValueError("axis.shape has to be 3")
    return _jit_get_rotation_matrix(vaxis, angle)


@njit
def _jit_get_rotation_matrix(
    axis: Vector[np.float64], angle: Real
) -> Matrix[np.float64]:
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
    rot_matrix[0, 1] = 2.0 * (b * c - a * d)
    rot_matrix[0, 2] = 2.0 * (b * d + a * c)
    rot_matrix[1, 0] = 2.0 * (b * c + a * d)
    rot_matrix[1, 1] = a**2 + c**2 - b**2 - d**2
    rot_matrix[1, 2] = 2.0 * (c * d - a * b)
    rot_matrix[2, 0] = 2.0 * (b * d - a * c)
    rot_matrix[2, 1] = 2.0 * (c * d + a * b)
    rot_matrix[2, 2] = a**2 + d**2 - b**2 - c**2
    return rot_matrix


def orthonormalize_righthanded(basis: Matrix[np.floating]) -> Matrix[np.float64]:
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


def get_kabsch_rotation(
    Q: Matrix[np.floating],
    P: Matrix[np.floating],
    weights: Vector[np.floating] | None = None,
) -> Matrix[np.float64]:
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
    V, S, W = np.linalg.svd(A)  # noqa: F841
    W = W.T
    d = np.linalg.det(W @ V.T)
    return W @ np.diag([1.0, 1.0, d]) @ V.T


def apply_grad_zmat_tensor(
    grad_C: Tensor4D, construction_table: DataFrame, cart_dist: Cartesian
) -> Zmat:
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

    dtypes = [
        ("atom", str),
        ("b", str),
        ("bond", float),
        ("a", str),
        ("angle", float),
        ("d", str),
        ("dihedral", float),
    ]

    new = pd.DataFrame(
        np.empty(len(construction_table), dtype=dtypes), index=cart_dist.index
    )

    X_dist = cart_dist.loc[:, COORDS].values.T
    C_dist = np.tensordot(grad_C, X_dist, axes=([3, 2], [0, 1])).T
    if C_dist.dtype == np.dtype("i8"):
        C_dist = C_dist.astype("f8")
    try:
        C_dist[:, [1, 2]] = np.rad2deg(C_dist[:, [1, 2]])
    # Unevaluated symbolic expressions are remaining.
    # catches AttributeError as well, because this was
    # the raised exception before https://github.com/numpy/numpy/issues/13666
    except (AttributeError, TypeError):
        C_dist[:, [1, 2]] = sympy.deg(C_dist[:, [1, 2]])
        new = new.astype({k: "O" for k in ["bond", "angle", "dihedral"]})

    new.loc[:, ["b", "a", "d"]] = construction_table
    new.loc[:, "atom"] = cart_dist.loc[:, "atom"]
    new.loc[:, ["bond", "angle", "dihedral"]] = C_dist
    return Zmat(new, _metadata={"last_valid_cartesian": cart_dist})


def _cart_interpolate(start: Cartesian, end: Cartesian, N: int) -> list[Cartesian]:
    Delta = (end - start) / (N - 1)
    return [start + i * Delta for i in range(N)]


def interpolate(
    start: Cartesian, end: Cartesian, N: int, coord: Literal["cart", "zmat"] = "zmat"
) -> list[Cartesian]:
    """Interpolate between start and end structure.

    Args:
        start :
        end :
        N (int): Number of structures to interpolate between.
        coord :
            Interpolate either in cartesian or zmatrix space.
    """
    if coord == "cart":
        return _cart_interpolate(start, end, N)
    elif coord == "zmat":
        return _zmat_interpolate(start, end, N)
    else:
        assert_never(f"coord must be either 'cart' or 'zmat', not {coord}")
