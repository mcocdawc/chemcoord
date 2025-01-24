from numpy import allclose

import chemcoord as cc
from chemcoord.constants import elements


def testRestoreElementData():
    old_radii = elements.loc[:, "atomic_radius_cc"].copy()

    with cc.constants.RestoreElementData():
        # increase by 10 %
        elements.loc[:, "atomic_radius_cc"] *= 3

        assert allclose(
            old_radii * 3, elements.loc[:, "atomic_radius_cc"], equal_nan=True
        )

    assert allclose(old_radii, elements.loc[:, "atomic_radius_cc"], equal_nan=True)
