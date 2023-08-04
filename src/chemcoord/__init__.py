import os

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version

__version__ = version("chemcoord")
_git_branch = "master"


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
from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord.cartesian_coordinates.asymmetric_unit_cartesian_class import \
    AsymmetricUnitCartesian
import chemcoord.cartesian_coordinates.xyz_functions as xyz_functions
from chemcoord.internal_coordinates.zmat_class_main import Zmat
import chemcoord.internal_coordinates.zmat_functions as zmat_functions
import chemcoord.configuration as configuration
from chemcoord.configuration import settings
import chemcoord.constants

import sys
sys.modules['chemcoord.xyz_functions'] = xyz_functions
sys.modules['chemcoord.zmat_functions'] = zmat_functions
