# -*- coding: utf-8 -*-

# Errorcodes are there for the jit_functions
ERR_CODE_OK = 0


ERR_CODE_PhysicalMeaning = 200
class PhysicalMeaning(Exception):  # noqa
    def __init__(self, value=''):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


ERR_CODE_UndefinedCoordinateSystem = 201
class UndefinedCoordinateSystem(PhysicalMeaning): # noqa
    pass


ERR_CODE_InvalidReference = 202
class InvalidReference(UndefinedCoordinateSystem): # noqa
    pass


class IllegalArgumentCombination(ValueError):
    pass
