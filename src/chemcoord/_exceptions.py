# -*- coding: utf-8 -*-

# Errorcodes are there for the jit_functions
ERR_CODE_OK = 0


ERR_CODE_PhysicalMeaning = 200
class PhysicalMeaning(Exception):  # noqa
    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return repr(self.message)


ERR_CODE_UndefinedCoordinateSystem = 201
class UndefinedCoordinateSystem(PhysicalMeaning): # noqa
    pass


ERR_CODE_InvalidReference = 202
class InvalidReference(UndefinedCoordinateSystem): # noqa
    def __init__(self, message='', i=None, b=None, a=None, d=None):
        self.message = message
        self.index = i
        self.references = {'b': b, 'a': a, 'd': d}

    def __str__(self):
        return repr(self.message)


class IllegalArgumentCombination(ValueError):
    pass
