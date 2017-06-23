# -*- coding: utf-8 -*-


class PhysicalMeaning(Exception):
    def __init__(self, value=''):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


class UndefinedCoordinateSystem(PhysicalMeaning):
    pass


class InvalidReference(UndefinedCoordinateSystem):
    pass


class IllegalArgumentCombination(ValueError):
    pass
