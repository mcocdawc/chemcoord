from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import chemcoord as cc
from chemcoord.xyz_functions import allclose
import pytest
from chemcoord.exceptions import UndefinedCoordinateSystem
import itertools
import numpy as np
import os
import sys


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_structure_path(script_path):
    test_path = os.path.join(script_path)
    while True:
        structure_path = os.path.join(test_path, 'structures')
        if os.path.exists(structure_path):
            return structure_path
        else:
            test_path = os.path.join(test_path, '..')


STRUCTURES = get_structure_path(get_script_path())


def test_concat():
    path = os.path.join(STRUCTURES, 'MIL53_small.xyz')
    molecule = cc.Cartesian.read_xyz(path)
    cartesians = [molecule + i * 10 for i in range(5)]

    with pytest.raises(ValueError):
        cc.xyz_functions.concat(cartesians)

    joined = cc.xyz_functions.concat(cartesians, ignore_index=True)
    for i, molecule in enumerate(cartesians):
        molecule2 = joined.iloc[(i * len(molecule)):(i + 1) * len(molecule)]
        molecule2.index = molecule.index
        assert allclose(molecule, molecule2)

    key_joined = cc.xyz_functions.concat(cartesians, keys=['a', 'b'])
    assert allclose(key_joined.loc['a'], (key_joined.loc['b'] - 10))


def test_concat_with_zmats():
    path = os.path.join(STRUCTURES, 'MIL53_small.xyz')
    m = cc.Cartesian.read_xyz(path, start_index=1)
    zm = m.get_zmat()
    zm1 = zm.change_numbering(new_index=range(1, len(zm) + 1))
    zm2 = zm.change_numbering(new_index=zm1.index + len(zm1))

    new = cc.xyz_functions.concat([zm1.get_cartesian(),
                                   zm2.get_cartesian() + [0, 0, 10]])

    c_table = zm2.loc[:, ['b', 'a', 'd']]
    c_table.loc[57, ['b', 'a', 'd']] = [4, 1, 2]
    c_table.loc[58, ['a', 'd']] = [1, 2]
    c_table.loc[59, 'd'] = 1

    large_c_table = new.get_construction_table(
        fragment_list=[(zm2.get_cartesian(), c_table)])
    znew = new.get_zmat(large_c_table)

    cc.xyz_functions.allclose(new, znew.get_cartesian())

    znew.safe_loc[57, 'bond'] = 20. - 0.89
    assert allclose(
        zm1.get_cartesian().append(zm2.get_cartesian() + [0, 0, 20]),
        znew.get_cartesian())
