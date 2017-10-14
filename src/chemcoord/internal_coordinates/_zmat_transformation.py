# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)


import numba as nb
import numpy as np
from numpy import sin, cos
from numba import jit

import chemcoord.constants as constants
from chemcoord.cartesian_coordinates.xyz_functions import _jit_isclose
from chemcoord.cartesian_coordinates._cart_transformation import (
    get_B, get_grad_B, get_ref_pos)
from chemcoord.exceptions import ERR_CODE_OK, ERR_CODE_InvalidReference


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
def get_grad_S(C, j):
    grad_S = np.empty((3, 3), dtype=nb.f8)
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
    grad_S[2, 2] = 0.
    return grad_S


@jit(nopython=True, cache=True)
def get_X(C, c_table):
    X = np.empty_like(C)
    n_atoms = X.shape[1]
    for j in range(n_atoms):
        err, B = get_B(X, c_table, j)
        if err == ERR_CODE_InvalidReference:
            return (err, j, X)
        X[:, j] = (np.dot(B, get_S(C, j))
                   + get_ref_pos(X, c_table[0, j]))
    return (ERR_CODE_OK, j, X)  # pylint:disable=undefined-loop-variable


@jit(nopython=True, cache=True)
def chain_grad(X, grad_X, C, c_table, j, l):
    if j < constants.keys_below_are_abs_refs:
        new_grad_X = np.zeros((3, 3))
    else:
        grad_B = get_grad_B(X, c_table, j)
        S = get_S(C, j)
        new_grad_X = grad_X[:, c_table[0, j], l, :].copy()

        for m_2 in range(3):
            for m_1 in range(3):
                for k in range(3):
                    if c_table[k, j] > constants.keys_below_are_abs_refs:
                        new_grad_X += (S[m_2]
                                       * grad_B[:, m_2, k, m_1]
                                       * grad_X[m_1, c_table[k, j], l, :])
    return new_grad_X


@jit(nopython=True, cache=True)
def get_grad_X(C, c_table, chain=True):
    n_atoms = C.shape[1]
    grad_X = np.zeros((3, n_atoms, n_atoms, 3))
    X = get_X(C, c_table)[2]
    for j in range(n_atoms):
        grad_X[:, j, j, :] = np.dot(get_B(X, c_table, j)[1], get_grad_S(C, j))
        if chain:
            for l in range(j):
                grad_X[:, j, l, :] = chain_grad(X, grad_X, C, c_table, j, l)
    return grad_X
