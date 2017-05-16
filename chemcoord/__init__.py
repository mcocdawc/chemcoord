from __future__ import absolute_import

import os

import pkg_resources  # part of setuptools
__version__ = pkg_resources.get_distribution("chemcoord").version
_git_hash = "8ad2758aad4df6582c8966d7a507d824580e8f31"
_git_branch = "experimental"

def export(func):
    if callable(func) and hasattr(func, '__name__'):
        globals()[func.__name__] = func
    try:
        __all__.append(func.__name__)
    except NameError:
        __all__ = [func.__name__]
    return func

# have to be imported after export definition
import chemcoord.utilities
from chemcoord.utilities._print_versions import show_versions
import chemcoord._generic_classes
from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian
import chemcoord.cartesian_coordinates.xyz_functions as xyz_functions
from chemcoord.internal_coordinates.zmat_class_main import Zmat
import chemcoord.internal_coordinates.zmat_functions as zmat_functions
import chemcoord.configuration as configuration
from chemcoord.configuration import settings
import chemcoord.vibration.vibration as vibration

import sys
sys.modules['chemcoord.xyz_functions'] = xyz_functions
sys.modules['chemcoord.zmat_functions'] = zmat_functions
sys.modules['chemcoord.vibration'] = vibration
