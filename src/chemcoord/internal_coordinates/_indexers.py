# -*- coding: utf-8 -*-

from chemcoord.exceptions import InvalidReference


class _generic_Indexer(object):
    def __init__(self, molecule):
        self.molecule = molecule


class _Loc(_generic_Indexer):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule._frame.loc[key[0], key[1]]
        else:
            selected = self.molecule._frame.loc[key]
        return selected


class _Unsafe_Loc(_Loc):
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule._frame.loc[key[0], key[1]] = value
        else:
            self.molecule._frame.loc[key] = value


class _Safe_Loc(_Loc):
    def __setitem__(self, key, value):
        if self.molecule.dummy_manipulation_allowed:
            molecule = self.molecule
        else:
            molecule = self.molecule.copy()
        if isinstance(key, tuple):
            molecule._frame.loc[key[0], key[1]] = value
        else:
            molecule._frame.loc[key] = value

        try:
            molecule.get_cartesian()
        except AttributeError:
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
            except AttributeError:
                pass


class _ILoc(_generic_Indexer):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule._frame.iloc[key[0], key[1]]
        else:
            selected = self.molecule._frame.iloc[key]
        return selected


class _Unsafe_ILoc(_ILoc):
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule._frame.iloc[key[0], key[1]] = value
        else:
            self.molecule._frame.iloc[key] = value


class _Safe_ILoc(_Unsafe_ILoc):
    def __setitem__(self, key, value):
        if self.molecule.dummy_manipulation_allowed:
            molecule = self.molecule
        else:
            molecule = self.molecule.copy()
        if isinstance(key, tuple):
            molecule._frame.iloc[key[0], key[1]] = value
        else:
            molecule._frame.iloc[key] = value

        try:
            molecule.get_cartesian()
        except AttributeError:
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
            except AttributeError:
                pass
