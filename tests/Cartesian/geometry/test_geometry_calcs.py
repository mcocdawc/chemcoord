import chemcoord as cc
import pytest
import numpy as np
from numpy import isclose
import os
import sys


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_structure_path(script_path):
    found, n = False, 0
    while not found:
        parents = ['..' for _ in range(n)]
        structure_path = os.path.join(script_path, *parents, 'structures')
        if os.path.exists(structure_path):
            return structure_path
        else:
            n += 1


STRUCTURES = get_structure_path(get_script_path())

molecule = cc.Cartesian.read_xyz(os.path.join(STRUCTURES, 'MIL53_small.xyz'))


def test_bond_length():
    assert isclose(molecule.bond_lengths([0, 1])[0], 1.3041493563710411)
    assert isclose(molecule.bond_lengths([0, 1])[0], 1.3041493563710411)
