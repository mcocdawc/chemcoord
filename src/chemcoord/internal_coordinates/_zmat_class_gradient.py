# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import copy
import warnings

import numba as nb
import numpy as np
import pandas as pd
from numba import jit, generated_jit

import chemcoord.constants as constants
import chemcoord.internal_coordinates._indexers as indexers
from chemcoord._generic_classes.generic_core import GenericCore
from chemcoord.cartesian_coordinates.xyz_functions import (
    _jit_cross, _jit_get_rotation_matrix,
    _jit_isclose, _jit_normalize)
from chemcoord.exceptions import (ERR_CODE_OK, ERR_CODE_InvalidReference,
                                  InvalidReference, PhysicalMeaning)
from chemcoord.utilities import _decorators
from chemcoord.internal_coordinates._zmat_class_pandas_wrapper import \
    PandasWrapper


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
        S[0] = r * np.sin(alpha) * np.cos(delta)
        S[1] = -r * np.sin(alpha) * np.sin(delta)
        S[2] = -r * np.cos(alpha)
    return S


def get_grad_S(C, j):
    pass


@jit(nopython=True)
def get_B(X, c_table, j):
    zeros = np.zeros(3, dtype=nb.f8)
    B = np.empty((3, 3))
    ref_pos = get_ref_pos(X, c_table[:, j])
    BA = ref_pos[:, 1] - ref_pos[:, 0]
    if _jit_isclose(BA, 0.).all():
        return (ERR_CODE_InvalidReference, zeros)
    AD = ref_pos[:, 2] - ref_pos[:, 1]
    B[:, 2] = -_jit_normalize(BA)
    N = _jit_cross(AD, BA)
    if _jit_isclose(N, 0.).all():
        return (ERR_CODE_InvalidReference, zeros)
    B[:, 1] = _jit_normalize(N)
    B[:, 0] = _jit_cross(B[:, 1], B[:, 2])
    return B


@jit(nopython=True)
def get_grad_B(X, c_table, j):
    pass


@jit(nopython=True)
def get_X(C, c_table):
    X = np.empty_like(C)
    n_atoms = X.shape[1]
    for j in range(n_atoms):
        X[:, j] = (np.dot(get_B(X, c_table, j), get_S(C, j))
                   + get_ref_pos(X, c_table[0, j]))
    return X

# @jit(nopython=True)
# def get_grad_V(X, C, c_table):
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
#
class ZmatGradient(PandasWrapper, GenericCore):
    def derive_to_cartesian(self):
        pass
