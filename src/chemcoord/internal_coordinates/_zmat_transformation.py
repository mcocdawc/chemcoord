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


@jit(nopython=True)
def get_grad_S(C, j):
    grad_S = np.empty((3, 3))
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
    grad_B = np.empty((3, 3, 3, 3))
    ref_pos = get_ref_pos(X, c_table[:, j])
    v_b, v_a, v_d = ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2]
    x_b, y_b, z_b = v_b
    x_a, y_a, z_a = v_a
    x_d, y_d, z_d = v_d
    BA, AD = v_a - v_b, v_d - v_a
    norm_AD_cross_BA = np.linalg.norm(_jit_cross(AD, BA))
    norm_BA = np.linalg.norm(BA)
    grad_B[0, 0, 0, 0] = ((x_a - x_b)*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((y_a - y_b)*(y_a - y_d) + (z_a - z_b)*(z_a - z_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((y_a - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[0, 0, 0, 1] = ((y_a - y_b)*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((-x_a + x_b)*(y_a - y_d) + (2*x_a - 2*x_d)*(y_a - y_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[0, 0, 0, 2] = ((z_a - z_b)*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((-x_a + x_b)*(z_a - z_d) + (2*x_a - 2*x_d)*(z_a - z_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[0, 0, 1, 0] = ((-x_a + x_b)*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((y_a - y_b)*(y_b - y_d) + (z_a - z_b)*(z_b - z_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((y_b - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_b - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[0, 0, 1, 1] = ((-y_a + y_b)*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_b - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_b - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b) - (x_b - x_d)*(y_a - y_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[0, 0, 1, 2] = ((-z_a + z_b)*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_b - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_b - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b) - (x_b - x_d)*(z_a - z_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[0, 0, 2, 0] = (-((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))**2 + ((y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[0, 0, 2, 1] = ((-x_a + x_b)*(y_a - y_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[0, 0, 2, 2] = ((-x_a + x_b)*(z_a - z_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[0, 1, 0, 0] = ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))*((y_a - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 1, 0, 1] = ((-z_a + z_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 1, 0, 2] = ((y_a - y_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 1, 1, 0] = (-(y_a - y_b)*(z_a - z_d) + (y_a - y_d)*(z_a - z_b))*((y_b - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_b - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 1, 1, 1] = ((z_b - z_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_b - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_b - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 1, 1, 2] = ((-y_b + y_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_b - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_b - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 1, 2, 0] = (-(y_a - y_b)*(z_a - z_d) + (y_a - y_d)*(z_a - z_b))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 1, 2, 1] = ((z_a - z_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 1, 2, 2] = ((-y_a + y_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[0, 2, 0, 0] = ((y_a - y_b)**2 + (z_a - z_b)**2)/norm_BA**3
    grad_B[0, 2, 0, 1] = (-x_a + x_b)*(y_a - y_b)/norm_BA**3
    grad_B[0, 2, 0, 2] = (-x_a + x_b)*(z_a - z_b)/norm_BA**3
    grad_B[0, 2, 1, 0] = (-(y_a - y_b)**2 - (z_a - z_b)**2)/norm_BA**3
    grad_B[0, 2, 1, 1] = (x_a - x_b)*(y_a - y_b)/norm_BA**3
    grad_B[0, 2, 1, 2] = (x_a - x_b)*(z_a - z_b)/norm_BA**3
    grad_B[0, 2, 2, 0] = 0
    grad_B[0, 2, 2, 1] = 0
    grad_B[0, 2, 2, 2] = 0
    grad_B[1, 0, 0, 0] = ((-x_a + x_b)*((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + (-(x_a - x_d)*(y_a - y_b) + (2*x_a - 2*x_b)*(y_a - y_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[1, 0, 0, 1] = ((-y_a + y_b)*((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*(x_a - x_d) + (z_a - z_b)*(z_a - z_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[1, 0, 0, 2] = ((-z_a + z_b)*((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + ((-y_a + y_b)*(z_a - z_d) + (2*y_a - 2*y_d)*(z_a - z_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[1, 0, 1, 0] = ((x_a - x_b)*((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_b - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_b - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + ((-x_a + x_b)*(y_a - y_d) - (x_a - x_b)*(y_b - y_d) + (x_a - x_d)*(y_a - y_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[1, 0, 1, 1] = ((y_a - y_b)*((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*(x_b - x_d) + (z_a - z_b)*(z_b - z_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_b - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_b - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[1, 0, 1, 2] = ((z_a - z_b)*((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_b - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_b - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b) - (y_b - y_d)*(z_a - z_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[1, 0, 2, 0] = ((-x_a + x_b)*(y_a - y_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[1, 0, 2, 1] = (-((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))**2 + ((x_a - x_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[1, 0, 2, 2] = (-(y_a - y_b)*(z_a - z_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[1, 1, 0, 0] = ((z_a - z_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))*((y_a - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[1, 1, 0, 1] = ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))*((x_a - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[1, 1, 0, 2] = ((-x_a + x_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))*((x_a - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[1, 1, 1, 0] = ((-z_b + z_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))*((y_b - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_b - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[1, 1, 1, 1] = ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))*((-x_b + x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_b - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[1, 1, 1, 2] = ((x_b - x_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))*((x_b - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_b - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[1, 1, 2, 0] = ((-z_a + z_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[1, 1, 2, 1] = ((-x_a + x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))/norm_AD_cross_BA**3
    grad_B[1, 1, 2, 2] = ((x_a - x_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[1, 2, 0, 0] = (-x_a + x_b)*(y_a - y_b)/norm_BA**3
    grad_B[1, 2, 0, 1] = ((x_a - x_b)**2 + (z_a - z_b)**2)/norm_BA**3
    grad_B[1, 2, 0, 2] = (-y_a + y_b)*(z_a - z_b)/norm_BA**3
    grad_B[1, 2, 1, 0] = (x_a - x_b)*(y_a - y_b)/norm_BA**3
    grad_B[1, 2, 1, 1] = (-(x_a - x_b)**2 - (z_a - z_b)**2)/norm_BA**3
    grad_B[1, 2, 1, 2] = (y_a - y_b)*(z_a - z_b)/norm_BA**3
    grad_B[1, 2, 2, 0] = 0
    grad_B[1, 2, 2, 1] = 0
    grad_B[1, 2, 2, 2] = 0
    grad_B[2, 0, 0, 0] = ((-x_a + x_b)*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + (-(x_a - x_d)*(z_a - z_b) + (2*x_a - 2*x_b)*(z_a - z_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[2, 0, 0, 1] = ((-y_a + y_b)*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + (-(y_a - y_d)*(z_a - z_b) + (2*y_a - 2*y_b)*(z_a - z_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[2, 0, 0, 2] = ((-z_a + z_b)*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*(x_a - x_d) + (y_a - y_b)*(y_a - y_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[2, 0, 1, 0] = ((x_a - x_b)*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_b - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_b - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + ((-x_a + x_b)*(z_a - z_d) - (x_a - x_b)*(z_b - z_d) + (x_a - x_d)*(z_a - z_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[2, 0, 1, 1] = ((y_a - y_b)*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_b - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_b - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) + ((-y_a + y_b)*(z_a - z_d) - (y_a - y_b)*(z_b - z_d) + (y_a - y_d)*(z_a - z_b))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[2, 0, 1, 2] = ((z_a - z_b)*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*(x_b - x_d) + (y_a - y_b)*(y_b - y_d))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_b - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_b - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2))/(norm_AD_cross_BA**3*norm_BA**3)
    grad_B[2, 0, 2, 0] = ((-x_a + x_b)*(z_a - z_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[2, 0, 2, 1] = (-(y_a - y_b)*(z_a - z_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[2, 0, 2, 2] = (-((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))**2 + ((x_a - x_b)**2 + (y_a - y_b)**2)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2))/(norm_AD_cross_BA**3*norm_BA)
    grad_B[2, 1, 0, 0] = ((-y_a + y_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))*((y_a - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[2, 1, 0, 1] = ((x_a - x_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))*((x_a - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[2, 1, 0, 2] = (-(x_a - x_b)*(y_a - y_d) + (x_a - x_d)*(y_a - y_b))*((x_a - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[2, 1, 1, 0] = ((y_b - y_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))*((y_b - y_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_b - z_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[2, 1, 1, 1] = ((-x_b + x_d)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))*((x_b - x_d)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_b - z_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[2, 1, 1, 2] = ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))*((x_b - x_d)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_b - y_d)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[2, 1, 2, 0] = ((y_a - y_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) - ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))*((y_a - y_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) + (z_a - z_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[2, 1, 2, 1] = ((-x_a + x_b)*(((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))**2 + ((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b))**2 + ((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))**2) + ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))*((x_a - x_b)*((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b)) - (z_a - z_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b))))/norm_AD_cross_BA**3
    grad_B[2, 1, 2, 2] = ((x_a - x_b)*(y_a - y_d) - (x_a - x_d)*(y_a - y_b))*((x_a - x_b)*((x_a - x_b)*(z_a - z_d) - (x_a - x_d)*(z_a - z_b)) + (y_a - y_b)*((y_a - y_b)*(z_a - z_d) - (y_a - y_d)*(z_a - z_b)))/norm_AD_cross_BA**3
    grad_B[2, 2, 0, 0] = (-x_a + x_b)*(z_a - z_b)/norm_BA**3
    grad_B[2, 2, 0, 1] = (-y_a + y_b)*(z_a - z_b)/norm_BA**3
    grad_B[2, 2, 0, 2] = ((x_a - x_b)**2 + (y_a - y_b)**2)/norm_BA**3
    grad_B[2, 2, 1, 0] = (x_a - x_b)*(z_a - z_b)/norm_BA**3
    grad_B[2, 2, 1, 1] = (y_a - y_b)*(z_a - z_b)/norm_BA**3
    grad_B[2, 2, 1, 2] = (-(x_a - x_b)**2 - (y_a - y_b)**2)/norm_BA**3
    grad_B[2, 2, 2, 0] = 0
    grad_B[2, 2, 2, 1] = 0
    grad_B[2, 2, 2, 2] = 0
    return grad_B


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


@jit(nopython=True)
def get_new_grad_X(X, grad_X, C, c_table, j, l):
    if j < constants.keys_below_are_abs_refs:
        new_grad_X = np.zeros((3, 3))
    else:
        n_atoms = X.shape[1]
        grad_B = get_grad_B(X, c_table, j)
        S = get_S(C, j)
        new_grad_X = grad_X[:, c_table[0, j], l, :].copy()

        for m_2 in range(3):
            for m_1 in range(3):
                for k in range(3):
                    if k > constants.keys_below_are_abs_refs:
                        new_grad_X += (S[m_2]
                                       * grad_B[:, m_2, k, m_1]
                                       * grad_X[m_1, c_table[k, j], l, :])
    return new_grad_X


@jit(nopython=True)
def get_grad_X(C, c_table):
    n_atoms = C.shape[1]
    grad_X = np.empty((3, n_atoms, n_atoms, 3))
    X = get_X(C, c_table)[2]
    for j in range(n_atoms):
        for l in range(j + 1, n_atoms):
            grad_X[:, j, l, :] = 0.

    for j in range(n_atoms):
        grad_X[:, j, j, :] = get_grad_S(C, j)

    for j in range(n_atoms):
        for l in range(j):
            grad_X[:, j, l, :] = get_new_grad_X(X, grad_X, C, c_table, j, l)
    return grad_X
