# -*- coding: utf-8 -*-


class _generic_Indexer(object):
    def __init__(self, molecule):
        self.molecule = molecule


class _Loc(_generic_Indexer):

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule._frame.loc[key[0], key[1]]
        else:
            selected = self.molecule._frame.loc[list(key)]
        try:
            return self.molecule._return_appropiate_type(selected)
        except AttributeError:
            return selected

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule._frame.loc[key[0], key[1]] = value
        else:
            self.molecule._frame.loc[list(key)] = value


class _ILoc(_generic_Indexer):

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule._frame.iloc[key[0], key[1]]
        else:
            selected = self.molecule._frame.iloc[list(key)]
        try:
            return self.molecule._return_appropiate_type(selected)
        except AttributeError:
            return selected

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule._frame.iloc[key[0], key[1]] = value
        else:
            self.molecule._frame.iloc[list(key)] = value