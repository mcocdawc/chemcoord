# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord import export


@export
class dummy_manipulation(object):
    """Contextmanager that controls the behaviour of
    :meth:`~chemcoord.Zmat.safe_loc` and
    :meth:`~chemcoord.Zmat.safe_iloc`.

    In the following examples it is assumed, that using the assignment with
    :meth:`~chemcoord.Zmat.safe_loc` would lead to an invalid reference.
    Then there are two possible usecases::

        with dummy_manipulation(zmat, True):
            zmat.safe_loc[...] = ...
            # This inserts required dummy atoms and removes them,
            # if they are not needed anymore.
            # Removes only dummy atoms, that were automatically inserted.

        with dummy_manipulation(zmat, False):
            zmat.safe_loc[...] = ...
            # This raises an exception
            # :class:`~chemcoord.exceptions.InvalidReference`.
            # which can be handled appropiately.
            # The zmat instance is unmodified, if an exception was raised.
    """
    def __init__(self, zmat, insertion_allowed):
        self.zmat = zmat
        self.insertion_allowed = insertion_allowed
        self.old_value = self.zmat._metadata['insertion_allowed']

    def __enter__(self):
        self.zmat._metadata['insertion_allowed'] = self.insertion_allowed

    def __exit__(self, exc_type, exc_value, traceback):
        self.zmat._metadata['insertion_allowed'] = self.old_value
