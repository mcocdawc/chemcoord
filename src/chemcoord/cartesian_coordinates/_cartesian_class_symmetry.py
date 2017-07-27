# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import libmsym as msym
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
    def _give_msym_elements(self):
        return [msym.Element(name=self.loc[i, 'atom'],
                             coordinates=self.loc[i, ['x', 'y', 'z']])
                for i in self.index]

    def give_symmetry_group(self):
        with msym.Context(elements=self._give_msym_elements()) as ctx:
            sym_group = ctx.find_symmetry()
        return sym_group

    # def symmetrise(self):
