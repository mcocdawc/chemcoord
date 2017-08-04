# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

from chemcoord import export


@export
class DummyManipulation(object):
    """Contextmanager that controls the behaviour of
    :meth:`~chemcoord.Zmat.safe_loc` and
    :meth:`~chemcoord.Zmat.safe_iloc`.

    In the following examples it is assumed, that using the assignment with
    :meth:`~chemcoord.Zmat.safe_loc` would lead to an invalid reference.
    Then there are two possible usecases::

        with DummyManipulation(zmat, True):
            zmat.safe_loc[...] = ...
            # This inserts required dummy atoms and removes them,
            # if they are not needed anymore.
            # Removes only dummy atoms, that were automatically inserted.

        with DummyManipulation(zmat, False):
            zmat.safe_loc[...] = ...
            # This raises an exception
            # :class:`~chemcoord.exceptions.InvalidReference`.
            # which can be handled appropiately.
            # The zmat instance is unmodified, if an exception was raised.
    """
    def __init__(self, zmat, insertion_allowed):
        self.zmat = zmat
        self.insertion_allowed = insertion_allowed
        self.old_value = self.zmat._metadata['dummy_manipulation_allowed']

    def __enter__(self):
        self.zmat._metadata[
            'dummy_manipulation_allowed'] = self.insertion_allowed

    def __exit__(self, exc_type, exc_value, traceback):
        self.zmat._metadata[
            'dummy_manipulation_allowed'] = self.old_value
