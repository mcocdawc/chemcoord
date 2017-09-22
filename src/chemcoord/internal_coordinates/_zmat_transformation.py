# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)


import numba as nb
import numpy as np
from numpy import sin, cos
from numba import jit, generated_jit

import chemcoord.constants as constants
from chemcoord.cartesian_coordinates.xyz_functions import (
    _jit_cross, _jit_get_rotation_matrix,
    _jit_isclose, _jit_normalize)
from chemcoord.exceptions import ERR_CODE_OK, ERR_CODE_InvalidReference


@generated_jit(nopython=True)
def get_ref_pos(X, indices):
    if isinstance(indices, nb.types.Array):
        def f(X, indices):
            ref_pos = np.empty((3, len(indices)))
            for col, i in enumerate(indices):
                if i < constants.keys_below_are_abs_refs:
                    ref_pos[:, col] = constants._jit_absolute_refs(i)
                else:
                    ref_pos[:, col] = X[:, i]
            return ref_pos
        return f
    elif isinstance(indices, nb.types.Integer):
        def f(X, indices):
            i = indices
            if i < constants.keys_below_are_abs_refs:
                ref_pos = constants._jit_absolute_refs(i)
            else:
                ref_pos = X[:, i]
            return ref_pos
        return f


@jit(nopython=True)
def get_S(C, j):
    S = np.zeros(3)
    r, alpha, delta = C[:, j]
    if _jit_isclose(alpha, np.pi):
        S[2] = r
    elif _jit_isclose(alpha, 0):
        S[2] = -r
    else:
        S[0] = r * sin(alpha) * cos(delta)
        S[1] = -r * sin(alpha) * sin(delta)
        S[2] = -r * cos(alpha)
    return S


def get_grad_S(C, j):
    grad_S = np.empty(3, 3)
    r, alpha, delta = C[:, j]

    # Derive for r
    grad_S[0, 0] = sin(alpha) * cos(delta)
    grad_S[1, 0] = -sin(alpha) * sin(delta)
    grad_S[2, 0] = -cos(alpha)

    # Derive for alpha
    grad_S[0, 1] = r * cos(alpha) * cos(delta)
    grad_S[1, 1] = -r * sin(delta) * cos(alpha)
    grad_S[2, 1] = r * sin(alpha)

    # Derive for delta
    grad_S[0, 2] = -r * sin(alpha) * sin(delta)
    grad_S[1, 2] = -r * sin(alpha) * cos(delta)
    grad_S[2, 2] = 0
    return grad_S


@jit(nopython=True)
def get_B(X, c_table, j):
    B = np.empty((3, 3))
    ref_pos = get_ref_pos(X, c_table[:, j])
    BA = ref_pos[:, 1] - ref_pos[:, 0]
    if _jit_isclose(BA, 0.).all():
        return (ERR_CODE_InvalidReference, B)
    AD = ref_pos[:, 2] - ref_pos[:, 1]
    B[:, 2] = -_jit_normalize(BA)
    N = _jit_cross(AD, BA)
    if _jit_isclose(N, 0.).all():
        return (ERR_CODE_InvalidReference, B)
    B[:, 1] = _jit_normalize(N)
    B[:, 0] = _jit_cross(B[:, 1], B[:, 2])
    return (ERR_CODE_OK, B)


@jit(nopython=True)
def get_grad_B(X, c_table, j):
    pass


@jit(nopython=True)
def get_X(C, c_table):
    X = np.empty_like(C)
    n_atoms = X.shape[1]
    for j in range(n_atoms):
        err, B = get_B(X, c_table, j)
        if err == ERR_CODE_InvalidReference:
            return (err, j, X)
        X[:, j] = (np.dot(B, get_S(C, j))
                   + get_ref_pos(X, c_table[0, j]))
    return (ERR_CODE_OK, j, X)

# @jit(nopython=True)
# def get_grad_X(C, c_table):
#     n_atoms = X.shape[1]
#     grad_V = np.zeros(3, n_atoms, n_atoms, 3)
#     for j in range(n_atoms):
#         for l in range(j-1):
#             grad_V[:, j, l, :] =


#
#
#
# @jit(nopython=True)
# def b(j):
#     pass
#
# @jit(nopython=True)
# def a(j):
#     pass
#
# @jit(nopython=True)
# def d(j):
#     pass
#
#
# def B(X, )
