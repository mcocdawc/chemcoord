# ruff: noqa: E402
# ruff: noqa: PLC0414
# ruff: noqa: F401

from importlib.metadata import version

__version__ = version("chemcoord")
_git_branch = "master"


def export(func):
    if callable(func) and hasattr(func, "__name__"):
        globals()[func.__name__] = func
    try:
        __all__.append(func.__name__)
    except NameError:
        __all__ = [func.__name__]  # noqa: F841
    return func


# have to be imported after export definition
import sys

import chemcoord._cartesian_coordinates.xyz_functions as xyz_functions
import chemcoord._internal_coordinates.zmat_functions as zmat_functions
import chemcoord._utilities
import chemcoord.configuration as configuration
import chemcoord.constants
from chemcoord._cartesian_coordinates.asymmetric_unit_cartesian_class import (
    AsymmetricUnitCartesian as AsymmetricUnitCartesian,
)
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian as Cartesian
from chemcoord._internal_coordinates.zmat_class_main import Zmat as Zmat
from chemcoord._utilities._print_versions import show_versions as show_versions
from chemcoord.configuration import settings as settings

sys.modules["chemcoord.xyz_functions"] = xyz_functions
sys.modules["chemcoord.zmat_functions"] = zmat_functions
