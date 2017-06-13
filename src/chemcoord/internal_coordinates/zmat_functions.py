# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord import export


@export
def is_Zmat(possible_Zmat):
    """Tests, if given instance is a Zmat.

    Args:
        possible_Zmat (any type):

    Returns:
        bool:
    """
    columns = possible_Zmat.columns
    try:
        assert type(columns) is not str
        columns = set(columns)
    except (TypeError, AssertionError):
        columns = set([columns])
    is_zmat = {'atom', 'b', 'bond', 'a', 'angle',
               'd', 'dihedral'} <= columns
    return is_zmat
