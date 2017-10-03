from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os

import chemcoord as cc
import numpy as np


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

molecule = cc.Cartesian.read_xyz(os.path.join(STRUCTURES, 'MIL53_small.xyz'),
                                 start_index=1)
molecule -= molecule.get_barycenter()


def test_asymmetric_unit():
    a = molecule.get_asymmetric_unit()
    a.loc[8, 'x'] = 10
    a.loc[8, 'y'] = 10
    eq = molecule.get_equivalent_atoms()
    new = a.get_cartesian()
    comparison = np.isclose((new - molecule).loc[:, ['x', 'y', 'z']], 0.)
    new = new[~comparison.all(axis=1)]
    assert set(new.index) == eq['eq_sets'][8]
    assert np.isclose(abs(new.loc[:, ['x', 'y']]), 10).all


def test_point_group_detection_and_symmetrizing():
    np.random.seed(77)
    dist_molecule = molecule.copy()
    assert 'C2v' == dist_molecule.get_pointgroup(tolerance=0.1).sch_symbol
    dist_molecule += np.random.randn(len(dist_molecule), 3) / 25
    assert 'C1' == dist_molecule.get_pointgroup(tolerance=0.1).sch_symbol
    eq = dist_molecule.symmetrize(max_n=25, tolerance=0.3, epsilon=1e-5)
    assert 'C2v' == eq['sym_mol'].get_pointgroup(tolerance=0.1).sch_symbol
    a, b = molecule.align(dist_molecule)
    a, c = molecule.align(eq['sym_mol'])
    d1 = (a - b).get_distance_to()
    d2 = (a - c).get_distance_to()
    assert d1['distance'].sum() > d2['distance'].sum()
