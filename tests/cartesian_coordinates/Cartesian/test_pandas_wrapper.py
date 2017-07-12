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


def get_complete_path(structure):
    STRUCTURES = get_structure_path(get_script_path())
    return os.path.join(STRUCTURES, structure)


molecule = cc.Cartesian.read_xyz(get_complete_path('water.xyz'),
                                 start_index=1)


def test_to_string():
    expected = ('  atom         x    y         z\n'
                '1    O  0.000000  0.0  0.000000\n'
                '2    H  0.758602  0.0  0.504284\n'
                '3    H  0.260455  0.0 -0.872893\n'
                '4    O  3.000000  0.5  0.000000\n'
                '5    H  3.758602  0.5  0.504284\n'
                '6    H  3.260455  0.5 -0.872893')
    assert molecule.to_string() == expected


def test_to_xyz():
    expected = """6
Created by chemcoord http://chemcoord.readthedocs.io/
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893"""
    assert molecule.to_xyz() == expected

    with pytest.warns(DeprecationWarning):
        assert molecule.write_xyz() == expected
