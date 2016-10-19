from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
try:
    # import itertools.imap as map
    import itertools.izip as zip
except ImportError:
    pass
import numpy as np
import pandas as pd
import copy
import collections
from threading import Thread
import subprocess
import os
import tempfile
import warnings
from . import _common_class
from ._exceptions import PhysicalMeaningError
from . import constants
from . import utilities
from . import zmat_functions
from . import xyz_functions
from . import export
from .configuration import settings
import io
from io import open


def pick(my_set):
    """Returns one element from a set.

    **Do not** make any assumptions about the element to be returned.
    ``pick`` just returns a random element,
    could be the same, could be different.
    """
    assert type(my_set) is set, 'Pick can be applied only on sets.'
    x = my_set.pop()
    my_set.add(x)
    return x

class Mode(_common_class.common_methods):
    """test
    """

    def __init__(self):
        self.minimum = mimimum
        pass

    def __add__(self, other):
        pass

    def __radd___(self, other):
        return self.add(self, other) # use commutation
