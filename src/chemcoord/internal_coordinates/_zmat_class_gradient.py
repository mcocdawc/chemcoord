# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import copy
import warnings

import numba as nb
import numpy as np
import pandas as pd
from numba import jit

import chemcoord.constants as constants
import chemcoord.internal_coordinates._indexers as indexers
from chemcoord._generic_classes.generic_core import GenericCore
from chemcoord.cartesian_coordinates.xyz_functions import (
    _jit_cross, _jit_get_rotation_matrix,
    _jit_isclose, _jit_normalize)
from chemcoord.exceptions import (ERR_CODE_OK, ERR_CODE_InvalidReference,
                                  InvalidReference, PhysicalMeaning)
from chemcoord.utilities import _decorators
from chemcoord.internal_coordinates._zmat_class_pandas_wrapper import \
    PandasWrapper


class ZmatGradient(PandasWrapper, GenericCore):
    def derive_to_cartesian(self):
        pass
