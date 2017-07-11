from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import chemcoord as cc
import pytest
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


def test_water():
    molecule = cc.Cartesian.read_xyz(os.path.join(STRUCTURES, 'water.xyz'),
                                     start_index=1)
    molecule = molecule - molecule.loc[5, ['x', 'y', 'z']]
    expected = {1: {2, 3}, 2: {1}, 3: {1}, 4: {5, 6}, 5: {4}, 6: {4}}
    assert molecule.get_bonds() == expected
