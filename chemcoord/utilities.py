from __future__ import division
from __future__ import absolute_import

try:
    import itertools.imap as map
except ImportError:
    pass

import numpy as np
import math as m


def normalize(vector):
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
    a = np.cos(angle/2)
    b, c, d = axis * np.sin(angle/2)
    rot_matrix = np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    return rot_matrix


def give_angle(Vector1, Vector2):
    '''
    Calculate the angle in degrees between two vectors.
    The vectors do not have to be normalized.
    '''
    vector1 = normalize(Vector1)
    vector2 = normalize(Vector2)

    # Is this ok
    scalar_product = np.dot(vector1, vector2)
    if -1.00000000000001 < scalar_product < -1.:
        scalar_product = -1.

    elif 1.00000000000001 > scalar_product > 1.:
        scalar_product = 1.

    angle = m.acos(scalar_product)
    angle = np.degrees(angle)

    return angle


def add_dummy(zmat_frame, xyz_frame, index):
    zmat = zmat_frame.copy()
    p_list = []
    xyz = xyz_frame.copy()

    atom, bond, angle, dihedral = zmat.loc[
        index, ['atom', 'bond', 'angle', 'dihedral']]

    angle, dihedral = map(m.radians, (angle, dihedral))

    bond_with, angle_with, dihedral_with = zmat.loc[index, [
        'bond_with', 'angle_with', 'dihedral_with']]

    bond_with, angle_with, dihedral_with = map(
        int, (bond_with, angle_with, dihedral_with))

    vb = np.array(xyz.loc[bond_with, ['x', 'y', 'z']], dtype=float)
    va = np.array(xyz.loc[angle_with, ['x', 'y', 'z']], dtype=float)
    vd = np.array(xyz.loc[dihedral_with, ['x', 'y', 'z']], dtype=float)

    AB = vb - va
    DA = vd - va

    n1 = normalize(np.cross(DA, AB))
    ab = normalize(AB)

    # Vector of length distance pointing along the x-axis
    d = bond * -ab

    # Rotate d by the angle around the n1 axis
    d = np.dot(rotation_matrix(n1, angle), d)
    d = np.dot(rotation_matrix(ab, dihedral), d)

    # Add d to the position of q to get the new coordinates of the atom
    p = vb + d

    # Change of nonlocal variables
    p_list.append(list(p))
    xyz.loc[index] = [atom] + list(p)
    return xyz


def orthormalize(basis):
    """Orthonormalizes a given basis.

    This functions returns a right handed orthormalized basis.
    Since only the first two vectors in the basis are used, it does not matter
    if you give two or three vectors.

    Right handed means, that:

        - np.cross(e1, e2) = e3
        - np.cross(e2, e3) = e1
        - np.cross(e3, e1) = e2

    Args:
        basis (np.array): An array of shape = (3,2) or (3,3)

    Returns:
        new_basis (np.array): A right handed orthonormalized basis.
    """
    def local_orthonormalize(basis):
        v1, v2 = basis[:, 0], basis[:, 1]
        e1 = normalize(v1)
        e3 = normalize(np.cross(e1, v2))
        e2 = normalize(np.cross(e3, e1))
        basis = np.transpose(np.array([e1, e2, e3]))
        return basis

    for _ in range(3):
        basis = local_orthonormalize(basis)
    return basis


def distance(vector1, vector2):
    """Calculates the distance between vector1 and vector2
    """
    length = np.linalg.norm(vector1 - vector2)
    return length


def give_distance_array(location_array):
    """Returns a xyz_frame with a column for the distance from origin.
    """
    A = np.expand_dims(location_array, axis=1)
    B = np.expand_dims(location_array, axis=0)
    C = A - B
    return_array = np.linalg.norm(C, axis=2)
    return return_array



def kabsch(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
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

    Parameters:
    P -- (N, number of points)x(D, dimension) matrix
    Q -- (N, number of points)x(D, dimension) matrix

    Returns:
    U -- Rotation matrix

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
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm
    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P

#def kabsch(P, Q):
#    """Application of Kabsch algorithm on two matrices.
#
#    The optimal rotation matrix U is calculated and then used to rotate matrix
#    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
#    calculated.
#
#    Using the Kabsch algorithm with two sets of paired point P and Q,
#    centered around the center-of-mass.
#    Each vector set is represented as an NxD matrix, where D is the
#    the dimension of the space.
#
#    The algorithm works in three steps:
#
#    * a translation of P and Q
#    * the computation of a covariance matrix C
#    * computation of the optimal rotation matrix U
#
#    http://en.wikipedia.org/wiki/Kabsch_algorithm
#
#    Args:
#        P (np.array): (N, number of points)x(D, dimension) matrix
#        Q (np.array): (N, number of points)x(D, dimension) matrix
#
#    Returns:
#        new_basis (np.array): A right handed orthonormalized basis.
#        U (np.array): Rotation matrix
#
#    .. [1] Kabsch W. (1976).
#        A solution for the best rotation to relate two sets of vectors.
#        Acta Crystallographica, A32:922-923.
#        `doi:10.1107/S0567739476001873 <http://dx.doi.org/10.1107/S0567739476001873>`_
#
#    .. [2] Jimmy Charnley Kromann ; Casper Steinmann ; larsbratholm ; aandi ; Kasper Primdal Lauritzen (2016).
#        GitHub: Calculate RMSD for two XYZ structures.
#        http://github.com/charnley/rmsd, `doi:10.5281/zenodo.46697 <http://dx.doi.org/10.5281/zenodo.46697>`_
#    """
#
#    # Computation of the covariance matrix
#    C = np.dot(np.transpose(P), Q)
#
#    # Computation of the optimal rotation matrix
#    # This can be done using singular value decomposition (SVD)
#    # Getting the sign of the det(V)*(W) to decide
#    # whether we need to correct our rotation matrix to ensure a
#    # right-handed coordinate system.
#    # And finally calculating the optimal rotation matrix U
#    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
#    V, S, W = np.linalg.svd(C)
#    signature = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
#
#    if signature:
#        S[-1] = -S[-1]
#        V[:, -1] = -V[:, -1]
#
#    # Create Rotation matrix U
#    U = np.dot(V, W)
#    return U
