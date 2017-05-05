from __future__ import absolute_import


def export(func):
    if callable(func) and hasattr(func, '__name__'):
        globals()[func.__name__] = func
    try:
        __all__.append(func.__name__)
    except NameError:
        __all__ = [func.__name__]
    return func

from chemcoord.algebra_utilities import utilities
from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian
from chemcoord.cartesian_coordinates import xyz_functions
from chemcoord.cartesian_coordinates import xyz_functions as pew
from chemcoord.internal_coordinates.zmat_class_main import Zmat
from chemcoord.internal_coordinates import zmat_functions
from chemcoord.configuration import configuration
from chemcoord.configuration.configuration import settings
from chemcoord.vibration import vibration


# globals()['xyz_functions'] = cartesian_coordinates.xyz_functions
# globals()['settings'] = configuration.settings
