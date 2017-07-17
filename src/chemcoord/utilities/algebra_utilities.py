from __future__ import division
from __future__ import absolute_import

import math as m
from numba import jit
import numpy as np


@jit(nopython=True)
def _jit_isclose(a, b, atol=1e-5, rtol=1e-8):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


@jit(nopython=True)
def _jit_allclose(a, b, atol=1e-5, rtol=1e-8):
    n, m = a.shape
    for i in range(n):
        for j in range(m):
            if np.abs(a[i, j] - b[i, j]) > (atol + rtol * np.abs(b[i, j])):
                return False
    return True


@jit(nopython=True)
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


@jit(nopython=True)
def _jit_normalize(vector):
    """Normalizes a vector
    """
    normed_vector = vector / np.linalg.norm(vector)
    return normed_vector


def rotation_matrix(axis, angle):
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
    return _jit_rotation_matrix(axis, angle)


@jit(nopython=True)
def _jit_rotation_matrix(axis, angle):
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


def kabsch(P, Q):
    """The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.

    Using the Kabsch algorithm with two sets of paired point P and Q,
    centered around the center-of-mass.
    Each vector set is represented as an NxD matrix, where D is the
    the dimension of the space.

    The algorithm works in three steps:

        - a translation of P and Q
        - the computation of a covariance matrix C
        - computation of the optimal rotation matrix U

    http://en.wikipedia.org/wiki/Kabsch_algorithm


    Args:
        P (numpy.array): ``(N, number of points)*(D, dimension)`` matrix
        Q (numpy.array): ``(N, number of points)*(D, dimension)`` matrix
        ignore_hydrogens (bool): Hydrogens are ignored for the
        RMSD.

    Returns:
        :func:`~numpy.array`: Rotation matrix
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def rotate(P, Q):
    """Rotate matrix P unto matrix Q using Kabsch algorithm
    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P
