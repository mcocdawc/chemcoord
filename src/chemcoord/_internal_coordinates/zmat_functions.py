import numpy as np
import sympy

from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord._internal_coordinates.zmat_class_main import Zmat
from chemcoord.typing import Tensor4D


class DummyManipulation:
    """Contextmanager that controls the behaviour of
    :meth:`~chemcoord.Zmat.safe_loc` and
    :meth:`~chemcoord.Zmat.safe_iloc`.

    In the following examples it is assumed, that using the assignment with
    :meth:`~chemcoord.Zmat.safe_loc` would lead to an invalid reference.
    Then there are two possible usecases::

        with DummyManipulation(zmat, True):
            zmat.safe_loc[...] = ...
            # This inserts required dummy atoms and removes them,
            # if they are not needed anymore.
            # Removes only dummy atoms, that were automatically inserted.

        with DummyManipulation(zmat, False):
            zmat.safe_loc[...] = ...
            # This raises an exception
            # :class:`~chemcoord.exceptions.InvalidReference`.
            # which can be handled appropiately.
            # The zmat instance is unmodified, if an exception was raised.
    """

    def __init__(self, dummy_manipulation_allowed, cls=None):
        if cls is None:
            cls = Zmat
        self.cls = cls
        self.dummy_manipulation_allowed = dummy_manipulation_allowed
        self.old_value = self.cls.dummy_manipulation_allowed

    def __enter__(self):
        self.cls.dummy_manipulation_allowed = self.dummy_manipulation_allowed

    def __exit__(self, exc_type, exc_value, traceback):
        self.cls.dummy_manipulation_allowed = self.old_value


class TestOperators:
    """Switch the validity testing of zmatrices resulting from operators.

    The following examples is done with ``+``
    it is assumed, that adding ``zmat_1`` and ``zmat_2``
    leads to a zmatrix with an invalid reference::

        with TestOperators(True):
            zmat_1 + zmat_2
            # Raises InvalidReference Exception

    Since it is on by default, the contextmanager is mostly relevant for switching
    the validity testing off when doing a difference.
    """

    def __init__(self, test_operators, cls=None):
        if cls is None:
            cls = Zmat
        self.cls = cls
        self.test_operators = test_operators
        self.old_value = self.cls.test_operators

    def __enter__(self):
        self.cls.test_operators = self.test_operators

    def __exit__(self, exc_type, exc_value, traceback):
        self.cls.test_operators = self.old_value


class PureInternalMovement:
    """Remove the translational and rotational degrees of freedom.

    When doing assignments to the z-matrix::

        with PureInternalMovement(True):
            zmat_1.loc[1, 'bond'] = value

    the translational and rotational degrees of freedom are automatically projected out.

    For infinitesimal movements this would be done with the Eckhard conditions,
    but in this case we allow large non-infinitesimal movements.
    For the details read [6]_.

    """

    def __init__(self, pure_internal_mov, cls=None):
        if cls is None:
            cls = Zmat
        self.cls = cls
        self.pure_internal_mov = pure_internal_mov
        self.old_value = self.cls.pure_internal_mov

    def __enter__(self):
        self.cls.pure_internal_mov = self.pure_internal_mov

    def __exit__(self, exc_type, exc_value, traceback):
        self.cls.pure_internal_mov = self.old_value


class CleanDihedralOrientation:
    def __init__(self, clean_dihedral_orientation, cls=None):
        if cls is None:
            cls = Zmat
        self.cls = cls
        self.clean_dihedral_orientation = clean_dihedral_orientation
        self.old_value = self.cls.pure_internal_mov

    def __enter__(self):
        self.cls.pure_internal_mov = self.clean_dihedral_orientation

    def __exit__(self, exc_type, exc_value, traceback):
        self.cls.pure_internal_mov = self.old_value


def apply_grad_cartesian_tensor(grad_X: Tensor4D, zmat_dist: Zmat) -> Cartesian:
    """Apply the gradient for transformation to cartesian space onto zmat_dist.

    Args:
        grad_X : A ``(3, n, n, 3)`` array.
            The mathematical details of the index layout is explained in
            :meth:`~chemcoord.Cartesian.get_grad_zmat()`.
        zmat_dist :
            Distortions in Zmatrix space.

    Returns:
        :class:`~chemcoord.Cartesian`: Distortions in cartesian space.
    """
    columns = ["bond", "angle", "dihedral"]
    C_dist = zmat_dist.loc[:, columns].values.T
    try:
        C_dist = C_dist.astype("f8")
        C_dist[[1, 2], :] = np.radians(C_dist[[1, 2], :])
    except (TypeError, AttributeError):
        C_dist[[1, 2], :] = sympy.rad(C_dist[[1, 2], :])
    cart_dist = np.tensordot(grad_X, C_dist, axes=([3, 2], [0, 1])).T
    from chemcoord._cartesian_coordinates.cartesian_class_main import (  # noqa: PLC0415
        Cartesian,
    )

    return Cartesian.set_atom_coords(
        atoms=zmat_dist["atom"], coords=cart_dist, index=zmat_dist.index
    )


def _zmat_interpolate(start: Cartesian, end: Cartesian, N: int) -> list[Cartesian]:
    z_start = start.get_zmat()
    z_end = end.get_zmat(z_start.loc[:, ["b", "a", "d"]])
    with TestOperators(False):
        z_step = (z_end - z_start).minimize_dihedrals() / (N - 1)
    zmatrices = [z_start.copy()]
    with CleanDihedralOrientation(True):
        for i in range(N - 1):
            zmatrices.append(zmatrices[-1] + z_step)
    return [zm.get_cartesian() for zm in zmatrices]
