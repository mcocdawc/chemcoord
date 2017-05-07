from __future__ import absolute_import


def export(func):
    if callable(func) and hasattr(func, '__name__'):
        globals()[func.__name__] = func
    try:
        __all__.append(func.__name__)
    except NameError:
        __all__ = [func.__name__]
    return func

import chemcoord.utilities.algebra_utilities
from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian
import chemcoord.cartesian_coordinates.xyz_functions as xyz_functions
from chemcoord.internal_coordinates.zmat_class_main import Zmat
import chemcoord.internal_coordinates.zmat_functions as zmat_functions
import chemcoord.configuration
from chemcoord.configuration import settings
from chemcoord.vibration import vibration

import sys
sys.modules['chemcoord.xyz_functions'] = xyz_functions
sys.modules['chemcoord.zmat_functions'] = zmat_functions


# globals()['xyz_functions'] = cartesian_coordinates.xyz_functions
# globals()['settings'] = configuration.settings
