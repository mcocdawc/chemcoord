# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import pymatgen as mg
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from chemcoord.cartesian_coordinates._cartesian_class_core import CartesianCore


class CartesianSymmetry(CartesianCore):
    def _get_mg_molecule(self):
        return mg.Molecule(self['atom'].values,
                           self.loc[:, ['x', 'y', 'z']].values)

    def get_point_group_analyzer(self):
        return PointGroupAnalyzer(self._get_mg_molecule())
