# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord import export
from chemcoord._exceptions import InvalidReference
from contextlib import contextmanager


@export
@contextmanager
def allow_dummy_insertion():
    try:
        yield
    except InvalidReference as e:
        print("Not Implemented yet")
        print('Inserting dummy atom as dihedral reference for', e.index,
              'because the following are linear', e.references)
