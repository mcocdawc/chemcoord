import os

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version

__version__ = version("chemcoord")
_git_branch = "master"


def export(func):
    if callable(func) and hasattr(func, "__name__"):
        globals()[func.__name__] = func
    try:
        __all__.append(func.__name__)
    except NameError:
        __all__ = [func.__name__]
    return func


# have to be imported after export definition
import sys

import chemcoord.cartesian_coordinates.xyz_functions as xyz_functions
import chemcoord.configuration as configuration
import chemcoord.constants
import chemcoord.internal_coordinates.zmat_functions as zmat_functions
import chemcoord.utilities
from chemcoord.cartesian_coordinates.asymmetric_unit_cartesian_class import (
    AsymmetricUnitCartesian as AsymmetricUnitCartesian,
)
from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian as Cartesian
from chemcoord.configuration import settings as settings
from chemcoord.internal_coordinates.zmat_class_main import Zmat as Zmat
from chemcoord.utilities._print_versions import show_versions as show_versions

sys.modules["chemcoord.xyz_functions"] = xyz_functions
sys.modules["chemcoord.zmat_functions"] = zmat_functions
