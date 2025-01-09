# ruff: noqa: E402
# ruff: noqa: PLC0414
# ruff: noqa: F401

from importlib.metadata import version

__version__ = version("chemcoord")
_git_branch = "master"


import chemcoord.configuration as configuration
import chemcoord.constants
import chemcoord.xyz_functions
import chemcoord.zmat_functions
from chemcoord._cartesian_coordinates.asymmetric_unit_cartesian_class import (
    AsymmetricUnitCartesian as AsymmetricUnitCartesian,
)
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian as Cartesian
from chemcoord._cartesian_coordinates.point_group import PointGroupOperations
from chemcoord._internal_coordinates.zmat_class_main import Zmat as Zmat
from chemcoord._utilities._print_versions import show_versions as show_versions
from chemcoord.configuration import settings as settings
from chemcoord.zmat_functions import (
    DummyManipulation,
    PureInternalMovement,
    TestOperators,
)
