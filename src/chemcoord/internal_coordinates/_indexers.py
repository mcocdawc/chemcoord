# -*- coding: utf-8 -*-

from chemcoord.exceptions import InvalidReference


class _generic_Indexer(object):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.indexer[key[0], key[1]]
        else:
            selected = self.indexer[key]
        return selected


class _Loc(_generic_Indexer):
    def __init__(self, molecule):
        self.molecule = molecule
        self.indexer = self.molecule._frame.loc

class _ILoc(_generic_Indexer):
    def __init__(self, molecule):
        self.molecule = molecule
        self.indexer = self.molecule._frame.iloc

class _Unsafe_base():
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.indexer[key[0], key[1]] = value
        else:
            self.indexer[key] = value

class _SafeBase():
    def __setitem__(self, key, value):
        if self.molecule.dummy_manipulation_allowed:
            molecule = self.molecule
        else:
            molecule = self.molecule.copy()

        if isinstance(key, tuple):
            self.indexer[key[0], key[1]] = value
        else:
            self.indexer[key] = value

        try:
            molecule.get_cartesian()
        # Sympy objects
        # catches AttributeError as well, because this was
        # the raised exception before https://github.com/numpy/numpy/issues/13666
        except (AttributeError, TypeError):
            self.molecule = molecule
        except InvalidReference as exception:
            if molecule.dummy_manipulation_allowed:
                self.molecule._insert_dummy_zmat(exception, inplace=True)
            else:
                exception.zmat_after_assignment = molecule
                raise exception

        if molecule.dummy_manipulation_allowed:
            try:
                self.molecule._remove_dummies(inplace=True)
            # Sympy objects
            # catches AttributeError, because this was
            # the raised exception before https://github.com/numpy/numpy/issues/13666
            except (AttributeError, TypeError):
                pass



class _Unsafe_Loc(_Loc, _Unsafe_base):
    pass


class _Safe_Loc(_Loc, _SafeBase):
    pass


class _Unsafe_ILoc(_ILoc, _Unsafe_base):
    pass

class _Safe_ILoc(_ILoc, _SafeBase):
    pass
