from __future__ import absolute_import

from setuptools_scm import get_version
import os

import pkg_resources  # part of setuptools
__version__ = pkg_resources.get_distribution("chemcoord").version


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


# TODO
def show_versions(as_json=False):
    import imp
    import os
    fn = __file__
    this_dir = os.path.dirname(fn)
    sv_path = os.path.join(this_dir, 'chemcoord', 'util')
    mod = imp.load_module('pvmod', *imp.find_module('print_versions', [sv_path]))
    return mod.show_versions(as_json)


# def show_versions():
#     return output
