from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

try:
    import itertools.imap as map
except ImportError:
    pass

import numpy as np
import pandas as pd
import math as m
from chemcoord._exceptions import PhysicalMeaningError
from chemcoord.algebra_utilities import utilities
from chemcoord import export
from chemcoord.configuration import settings
# from chemcoord.constants import constants


@export
def is_Zmat(possible_Zmat):
    """Tests, if given instance is a Zmat.

    Args:
        possible_Zmat (any type):

    Returns:
        bool:
    """
    try:
        assert type(columns) is not str
        columns = set(columns)
    except (TypeError, AssertionError):
        columns = set([columns])
    is_zmat = {'atom', 'bond_with', 'bond', 'angle_with', 'angle',
               'dihedral_with', 'dihedral'} <= columns
    return is_zmat
