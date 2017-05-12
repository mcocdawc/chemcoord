import pandas as pd


class _generic_Indexer(object):
    def __init__(self, molecule):
        self.molecule = molecule


class _Loc(_generic_Indexer):

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule.frame.loc[key[0], key[1]]
        else:
            selected = self.molecule.frame.loc[key]
        return self.molecule._return_appropiate_type(selected)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule.frame.loc[key[0], key[1]] = value
        else:
            self.molecule.frame.loc[key] = value


class _ILoc(_generic_Indexer):

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule.frame.iloc[key[0], key[1]]
        else:
            selected = self.molecule.frame.iloc[key]
        return self.molecule._return_appropiate_type(selected)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule.frame.iloc[key[0], key[1]] = value
        else:
            self.molecule.frame.iloc[key] = value
