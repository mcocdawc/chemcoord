import numba as nb
import numpy as np
from numba.core.types import f8
from numba.extending import overload
from numpy import arccos, arctan2, sqrt

import chemcoord.constants as constants
from chemcoord._utilities._decorators import njit
from chemcoord.exceptions import ERR_CODE_OK, ERR_CODE_InvalidReference
from chemcoord.typing import Matrix, Vector


def normalize(vector: Vector[np.floating]) -> Vector[np.float64]:
    """Normalizes a vector"""
    return vector / np.linalg.norm(vector)


@njit
def _jit_normalize(vector: Vector[np.floating]) -> Vector[np.float64]:
    """Normalizes a vector"""
    return vector / np.linalg.norm(vector)


def _stub_get_ref_pos(X, indices):  # noqa: ARG001
    raise AssertionError("Should not call this function unjitted.")


@overload(_stub_get_ref_pos)
def _get_ref_pos_impl(X, indices):  # noqa: ARG001
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
    else:
        raise AssertionError("Should not be here")


@njit
def get_ref_pos(X, indices):
    return _stub_get_ref_pos(X, indices)


@njit
def get_B(X, c_table, j):
    B = np.empty((3, 3))
    ref_pos = get_ref_pos(X, c_table[:, j])
    BA = ref_pos[:, 1] - ref_pos[:, 0]
    if np.allclose(BA, 0.0):
        return (ERR_CODE_InvalidReference, B)
    AD = ref_pos[:, 2] - ref_pos[:, 1]
    B[:, 2] = -_jit_normalize(BA)
    N = np.cross(AD, BA)
    if np.allclose(N, 0.0):
        return (ERR_CODE_InvalidReference, B)
    B[:, 1] = _jit_normalize(N)
    B[:, 0] = np.cross(B[:, 1], B[:, 2])
    return (ERR_CODE_OK, B)


@njit
def get_grad_B(X, c_table, j):
    grad_B = np.empty((3, 3, 3, 3))
    ref_pos = get_ref_pos(X, c_table[:, j])
    v_b, v_a, v_d = ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2]
    x_b, y_b, z_b = v_b
    x_a, y_a, z_a = v_a
    x_d, y_d, z_d = v_d
    BA, AD = v_a - v_b, v_d - v_a
    norm_AD_cross_BA = np.linalg.norm(np.cross(AD, BA))
    norm_BA = np.linalg.norm(BA)
    grad_B[0, 0, 0, 0] = (
        (x_a - x_b)
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((y_a - y_b) * (y_a - y_d) + (z_a - z_b) * (z_a - z_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * (
            (y_a - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[0, 0, 0, 1] = (
        (y_a - y_b)
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((-x_a + x_b) * (y_a - y_d) + (2 * x_a - 2 * x_d) * (y_a - y_b))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[0, 0, 0, 2] = (
        (z_a - z_b)
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((-x_a + x_b) * (z_a - z_d) + (2 * x_a - 2 * x_d) * (z_a - z_b))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[0, 0, 1, 0] = (
        (-x_a + x_b)
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((y_a - y_b) * (y_b - y_d) + (z_a - z_b) * (z_b - z_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * (
            (y_b - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_b - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[0, 0, 1, 1] = (
        (-y_a + y_b)
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_b - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_b - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (
            (x_a - x_b) * (y_a - y_d)
            - (x_a - x_d) * (y_a - y_b)
            - (x_b - x_d) * (y_a - y_b)
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[0, 0, 1, 2] = (
        (-z_a + z_b)
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_b - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_b - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (
            (x_a - x_b) * (z_a - z_d)
            - (x_a - x_d) * (z_a - z_b)
            - (x_b - x_d) * (z_a - z_b)
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[0, 0, 2, 0] = (
        -(
            (
                (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
                + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            )
            ** 2
        )
        + ((y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[0, 0, 2, 1] = (
        (-x_a + x_b)
        * (y_a - y_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[0, 0, 2, 2] = (
        (-x_a + x_b)
        * (z_a - z_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[0, 1, 0, 0] = (
        ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        * (
            (y_a - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        / norm_AD_cross_BA**3
    )
    grad_B[0, 1, 0, 1] = (
        (-z_a + z_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
    ) / norm_AD_cross_BA**3
    grad_B[0, 1, 0, 2] = (
        (y_a - y_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
    ) / norm_AD_cross_BA**3
    grad_B[0, 1, 1, 0] = (
        (-(y_a - y_b) * (z_a - z_d) + (y_a - y_d) * (z_a - z_b))
        * (
            (y_b - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_b - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        / norm_AD_cross_BA**3
    )
    grad_B[0, 1, 1, 1] = (
        (z_b - z_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_b - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_b - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
    ) / norm_AD_cross_BA**3
    grad_B[0, 1, 1, 2] = (
        (-y_b + y_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_b - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_b - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
    ) / norm_AD_cross_BA**3
    grad_B[0, 1, 2, 0] = (
        (-(y_a - y_b) * (z_a - z_d) + (y_a - y_d) * (z_a - z_b))
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        / norm_AD_cross_BA**3
    )
    grad_B[0, 1, 2, 1] = (
        (z_a - z_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
    ) / norm_AD_cross_BA**3
    grad_B[0, 1, 2, 2] = (
        (-y_a + y_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
    ) / norm_AD_cross_BA**3
    grad_B[0, 2, 0, 0] = ((y_a - y_b) ** 2 + (z_a - z_b) ** 2) / norm_BA**3
    grad_B[0, 2, 0, 1] = (-x_a + x_b) * (y_a - y_b) / norm_BA**3
    grad_B[0, 2, 0, 2] = (-x_a + x_b) * (z_a - z_b) / norm_BA**3
    grad_B[0, 2, 1, 0] = (-((y_a - y_b) ** 2) - (z_a - z_b) ** 2) / norm_BA**3
    grad_B[0, 2, 1, 1] = (x_a - x_b) * (y_a - y_b) / norm_BA**3
    grad_B[0, 2, 1, 2] = (x_a - x_b) * (z_a - z_b) / norm_BA**3
    grad_B[0, 2, 2, 0] = 0.0
    grad_B[0, 2, 2, 1] = 0.0
    grad_B[0, 2, 2, 2] = 0.0
    grad_B[1, 0, 0, 0] = (
        (-x_a + x_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (-(x_a - x_d) * (y_a - y_b) + (2 * x_a - 2 * x_b) * (y_a - y_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[1, 0, 0, 1] = (
        (-y_a + y_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((x_a - x_b) * (x_a - x_d) + (z_a - z_b) * (z_a - z_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_a - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[1, 0, 0, 2] = (
        (-z_a + z_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_a - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + ((-y_a + y_b) * (z_a - z_d) + (2 * y_a - 2 * y_d) * (z_a - z_b))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[1, 0, 1, 0] = (
        (x_a - x_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_b - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_b - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (
            (-x_a + x_b) * (y_a - y_d)
            - (x_a - x_b) * (y_b - y_d)
            + (x_a - x_d) * (y_a - y_b)
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[1, 0, 1, 1] = (
        (y_a - y_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((x_a - x_b) * (x_b - x_d) + (z_a - z_b) * (z_b - z_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_b - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_b - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[1, 0, 1, 2] = (
        (z_a - z_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_b - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_b - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (
            (y_a - y_b) * (z_a - z_d)
            - (y_a - y_d) * (z_a - z_b)
            - (y_b - y_d) * (z_a - z_b)
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[1, 0, 2, 0] = (
        (-x_a + x_b)
        * (y_a - y_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[1, 0, 2, 1] = (
        -(
            (
                (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
                - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
            )
            ** 2
        )
        + ((x_a - x_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[1, 0, 2, 2] = (
        -(y_a - y_b)
        * (z_a - z_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[1, 1, 0, 0] = (
        (z_a - z_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        * (
            (y_a - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[1, 1, 0, 1] = (
        ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        * (
            (x_a - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        / norm_AD_cross_BA**3
    )
    grad_B[1, 1, 0, 2] = (
        (-x_a + x_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        * (
            (x_a - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[1, 1, 1, 0] = (
        (-z_b + z_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        * (
            (y_b - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_b - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[1, 1, 1, 1] = (
        ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        * (
            (-x_b + x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_b - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        / norm_AD_cross_BA**3
    )
    grad_B[1, 1, 1, 2] = (
        (x_b - x_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        * (
            (x_b - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_b - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[1, 1, 2, 0] = (
        (-z_a + z_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[1, 1, 2, 1] = (
        (
            (-x_a + x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        / norm_AD_cross_BA**3
    )
    grad_B[1, 1, 2, 2] = (
        (x_a - x_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[1, 2, 0, 0] = (-x_a + x_b) * (y_a - y_b) / norm_BA**3
    grad_B[1, 2, 0, 1] = ((x_a - x_b) ** 2 + (z_a - z_b) ** 2) / norm_BA**3
    grad_B[1, 2, 0, 2] = (-y_a + y_b) * (z_a - z_b) / norm_BA**3
    grad_B[1, 2, 1, 0] = (x_a - x_b) * (y_a - y_b) / norm_BA**3
    grad_B[1, 2, 1, 1] = (-((x_a - x_b) ** 2) - (z_a - z_b) ** 2) / norm_BA**3
    grad_B[1, 2, 1, 2] = (y_a - y_b) * (z_a - z_b) / norm_BA**3
    grad_B[1, 2, 2, 0] = 0.0
    grad_B[1, 2, 2, 1] = 0.0
    grad_B[1, 2, 2, 2] = 0.0
    grad_B[2, 0, 0, 0] = (
        (-x_a + x_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (-(x_a - x_d) * (z_a - z_b) + (2 * x_a - 2 * x_b) * (z_a - z_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[2, 0, 0, 1] = (
        (-y_a + y_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_a - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (-(y_a - y_d) * (z_a - z_b) + (2 * y_a - 2 * y_b) * (z_a - z_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[2, 0, 0, 2] = (
        (-z_a + z_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((x_a - x_b) * (x_a - x_d) + (y_a - y_b) * (y_a - y_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_a - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[2, 0, 1, 0] = (
        (x_a - x_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_b - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_b - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (
            (-x_a + x_b) * (z_a - z_d)
            - (x_a - x_b) * (z_b - z_d)
            + (x_a - x_d) * (z_a - z_b)
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[2, 0, 1, 1] = (
        (y_a - y_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_b - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_b - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        + (
            (-y_a + y_b) * (z_a - z_d)
            - (y_a - y_b) * (z_b - z_d)
            + (y_a - y_d) * (z_a - z_b)
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[2, 0, 1, 2] = (
        (z_a - z_b)
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((x_a - x_b) * (x_b - x_d) + (y_a - y_b) * (y_b - y_d))
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_b - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_b - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    ) / (norm_AD_cross_BA**3 * norm_BA**3)
    grad_B[2, 0, 2, 0] = (
        (-x_a + x_b)
        * (z_a - z_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[2, 0, 2, 1] = (
        -(y_a - y_b)
        * (z_a - z_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[2, 0, 2, 2] = (
        -(
            (
                (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
                + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
            )
            ** 2
        )
        + ((x_a - x_b) ** 2 + (y_a - y_b) ** 2)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
    ) / (norm_AD_cross_BA**3 * norm_BA)
    grad_B[2, 1, 0, 0] = (
        (-y_a + y_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
        * (
            (y_a - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[2, 1, 0, 1] = (
        (x_a - x_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
        * (
            (x_a - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[2, 1, 0, 2] = (
        (-(x_a - x_b) * (y_a - y_d) + (x_a - x_d) * (y_a - y_b))
        * (
            (x_a - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        / norm_AD_cross_BA**3
    )
    grad_B[2, 1, 1, 0] = (
        (y_b - y_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
        * (
            (y_b - y_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_b - z_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[2, 1, 1, 1] = (
        (-x_b + x_d)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
        * (
            (x_b - x_d) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_b - z_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[2, 1, 1, 2] = (
        ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
        * (
            (x_b - x_d) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_b - y_d) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        / norm_AD_cross_BA**3
    )
    grad_B[2, 1, 2, 0] = (
        (y_a - y_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        - ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
        * (
            (y_a - y_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            + (z_a - z_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[2, 1, 2, 1] = (
        (-x_a + x_b)
        * (
            ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b)) ** 2
            + ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b)) ** 2
            + ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b)) ** 2
        )
        + ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
        * (
            (x_a - x_b) * ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
            - (z_a - z_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
    ) / norm_AD_cross_BA**3
    grad_B[2, 1, 2, 2] = (
        ((x_a - x_b) * (y_a - y_d) - (x_a - x_d) * (y_a - y_b))
        * (
            (x_a - x_b) * ((x_a - x_b) * (z_a - z_d) - (x_a - x_d) * (z_a - z_b))
            + (y_a - y_b) * ((y_a - y_b) * (z_a - z_d) - (y_a - y_d) * (z_a - z_b))
        )
        / norm_AD_cross_BA**3
    )
    grad_B[2, 2, 0, 0] = (-x_a + x_b) * (z_a - z_b) / norm_BA**3
    grad_B[2, 2, 0, 1] = (-y_a + y_b) * (z_a - z_b) / norm_BA**3
    grad_B[2, 2, 0, 2] = ((x_a - x_b) ** 2 + (y_a - y_b) ** 2) / norm_BA**3
    grad_B[2, 2, 1, 0] = (x_a - x_b) * (z_a - z_b) / norm_BA**3
    grad_B[2, 2, 1, 1] = (y_a - y_b) * (z_a - z_b) / norm_BA**3
    grad_B[2, 2, 1, 2] = (-((x_a - x_b) ** 2) - (y_a - y_b) ** 2) / norm_BA**3
    grad_B[2, 2, 2, 0] = 0.0
    grad_B[2, 2, 2, 1] = 0.0
    grad_B[2, 2, 2, 2] = 0.0
    return grad_B


@njit(f8[:](f8[:]))
def get_S_inv(v):
    x, y, z = v
    r = np.linalg.norm(v)
    if r == 0:
        return np.array([0.0, 0.0, 0.0])
    alpha = arccos(-z / r)
    delta = arctan2(-y, x)
    return np.array([r, alpha, delta])


@njit(f8[:, :](f8[:]))
def get_grad_S_inv(v):
    x, y, z = v
    grad_S_inv = np.zeros((3, 3))

    r = np.linalg.norm(v)
    if np.isclose(r, 0):
        pass
    elif np.isclose(x**2 + y**2, 0):
        grad_S_inv[0, 0] = 0.0
        grad_S_inv[0, 1] = 0.0
        grad_S_inv[0, 2] = 1
        grad_S_inv[1, 0] = -1 / z
        grad_S_inv[1, 1] = -1 / z
        grad_S_inv[1, 2] = 0.0
        grad_S_inv[2, 0] = 0.0
        grad_S_inv[2, 1] = 0.0
        grad_S_inv[2, 2] = 0.0
    else:
        grad_S_inv[0, 0] = x / r
        grad_S_inv[0, 1] = y / r
        grad_S_inv[0, 2] = z / r
        grad_S_inv[1, 0] = -x * z / (sqrt(x**2 + y**2) * r**2)
        grad_S_inv[1, 1] = -y * z / (sqrt(x**2 + y**2) * r**2)
        grad_S_inv[1, 2] = sqrt(x**2 + y**2) / r**2
        grad_S_inv[2, 0] = y / (x**2 + y**2)
        grad_S_inv[2, 1] = -x / (x**2 + y**2)
        grad_S_inv[2, 2] = 0.0
    return grad_S_inv


@njit
def get_T(X, c_table, j):
    err, B = get_B(X, c_table, j)
    if err == ERR_CODE_OK:
        v_b = get_ref_pos(X, c_table[0, j])
        result = B.T @ (X[:, j] - v_b)
    else:
        result = np.empty(3)
    return err, result


@njit
def get_C(X: Matrix[np.floating], c_table) -> tuple[int, Matrix[np.float64]]:
    C = np.empty((3, c_table.shape[1]))

    for j in range(C.shape[1]):
        err, v = get_T(X, c_table, j)
        if err == ERR_CODE_OK:
            C[:, j] = get_S_inv(v)
        else:
            return (err, C)
    return (ERR_CODE_OK, C)


@njit
def get_grad_C(X, c_table):
    n_atoms = X.shape[1]
    grad_C = np.zeros((3, n_atoms, n_atoms, 3))

    for j in range(X.shape[1]):
        IB = (X[:, j] - get_ref_pos(X, c_table[0, j])).reshape((3, 1, 1))
        grad_S_inv = get_grad_S_inv(get_T(X, c_table, j)[1])
        err, B = get_B(X, c_table, j)
        if err == ERR_CODE_InvalidReference:
            return (err, j, grad_C)
        grad_B = get_grad_B(X, c_table, j)

        # Derive for j
        grad_C[:, j, j, :] = grad_S_inv @ B.T

        # Derive for b(j)
        if c_table[0, j] > constants.keys_below_are_abs_refs:
            A = np.sum(grad_B[:, :, 0, :] * IB, axis=0)
            grad_C[:, j, c_table[0, j], :] = grad_S_inv @ (A - B.T)
        else:
            grad_C[:, j, c_table[0, j], :] = 0.0

        # Derive for a(j)
        if c_table[1, j] > constants.keys_below_are_abs_refs:
            A = np.sum(grad_B[:, :, 1, :] * IB, axis=0)
            grad_C[:, j, c_table[1, j], :] = grad_S_inv @ A
        else:
            grad_C[:, j, c_table[1, j], :] = 0.0

        # Derive for d(j)
        if c_table[2, j] > constants.keys_below_are_abs_refs:
            A = np.sum(grad_B[:, :, 2, :] * IB, axis=0)
            grad_C[:, j, c_table[2, j], :] = grad_S_inv @ A
        else:
            grad_C[:, j, c_table[2, j], :] = 0.0
    return (ERR_CODE_OK, j, grad_C)  # pylint:disable=undefined-loop-variable
