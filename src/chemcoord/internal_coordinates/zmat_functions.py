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
class allow_dummy_insertion(object):
    def __init__(self, zmat, insertion_allowed):
        self.zmat = zmat
        self.insertion_allowed = insertion_allowed
        self.old_value = self.zmat._metadata['insertion_allowed']

    def __enter__(self):
        self.zmat._metadata['insertion_allowed'] = self.insertion_allowed

    def __exit__(self, exc_type, exc_value, traceback):
        self.zmat._metadata['insertion_allowed'] = self.old_value
