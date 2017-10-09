# -*- coding: utf-8 -*-

# Errorcodes are there for the jit_functions
ERR_CODE_OK = 0


ERR_CODE_PhysicalMeaning = 200


class PhysicalMeaning(Exception):
    """Raised when data is corrupted in a way, that it can not carry
    any information of physical meaning.

    """
    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return repr(self.message)


ERR_CODE_UndefinedCoordinateSystem = 201


class UndefinedCoordinateSystem(PhysicalMeaning):
    """Raised when there is no possibility to obtain a defined coordinate
    system for the chosen construction table.

    """
    pass


ERR_CODE_InvalidReference = 202


class InvalidReference(UndefinedCoordinateSystem):
    """Raised when the i-th atom uses an invalid reference.

    May carry several attributes:

    * ``i``: Index of the atom with an invalid refernce.
    * ``b``, ``a``, and ``d``: Indices of reference atoms.
    * ``already_built_cartesian``: The cartesian of all atoms up to (i-1)
    * ``zmat_after_assignment``: Attached information if
      it was raised from the safe assignment methods
      (:meth:`Zmat.safe_loc` and :meth:`Zmat.unsafe_loc`).
    """
    def __init__(self, message=None, i=None, b=None, a=None, d=None,
                 already_built_cartesian=None,
                 zmat_after_assignment=None):
        self.message = message
        if i:
            self.index = i
        references = {'b': b, 'a': a, 'd': d}
        references = {k: v for k, v in references.items() if v is not None}
        if references:
            self.references = references
        if already_built_cartesian:
            self.already_built_cartesian = already_built_cartesian.copy()
        if zmat_after_assignment:
            self.zmat_after_assignment = zmat_after_assignment.copy()

    def __str__(self):
        if self.message is None:
            give_message = ('Atom {i} uses an invalid/linear reference '
                            'spanned by: {r}'.format)
            return give_message(i=self.index, r=self.references)
        else:
            return repr(self.message)


class IllegalArgumentCombination(ValueError):
    """Raised if the combination of correctly typed arguments is invalid.

    """
    pass
