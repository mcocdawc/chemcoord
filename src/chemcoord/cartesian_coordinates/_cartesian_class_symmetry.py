# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pymatgen as mg
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import numpy as np
import pandas as pd
import warnings
import chemcoord.constants as constants
from chemcoord.exceptions import \
    IllegalArgumentCombination, \
    UndefinedCoordinateSystem
from chemcoord.cartesian_coordinates._cartesian_class_core import \
    CartesianCore
from chemcoord.configuration import settings
from chemcoord.internal_coordinates.zmat_class_main import Zmat
from collections import OrderedDict
from itertools import permutations


class CartesianSymmetry(CartesianCore):
    def _give_mg_molecule(self):
        return mg.Molecule(self['atom'].values,
                           self.loc[:, ['x', 'y', 'z']].values)

    def give_point_group_analyzer(self):
        return PointGroupAnalyzer(self._give_mg_molecule())
