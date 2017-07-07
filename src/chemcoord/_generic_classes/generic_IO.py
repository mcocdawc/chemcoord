# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from chemcoord._exceptions import PhysicalMeaning
from chemcoord.configuration import settings
from chemcoord.utilities import algebra_utilities
from chemcoord.utilities.set_utilities import pick
import chemcoord.constants as constants
import collections
import sympy
from itertools import product
import numba as nb
from numba import jit
import numpy as np
import pandas as pd
from six.moves import zip  # pylint:disable=redefined-builtin
from sortedcontainers import SortedSet


class GenericIO(object):
    def _sympy_formatter(self):
        def formatter(x):
            if (isinstance(x, sympy.Basic)):
                return '${}$'.format(sympy.latex(x))
            else:
                return x
        new = self.copy()
        for col in self.columns.drop('atom'):
            if self[col].dtype == np.dtype('O'):
                new.unsafe_loc[:, col] = self[col].apply(formatter)
        return new
