# -*- coding: utf-8 -*-


class _generic_Indexer(object):
    def __init__(self, molecule):
        self.molecule = molecule


class _Loc(_generic_Indexer):

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule._frame.loc[_set_caster(key[0]), _set_caster(key[1])]
        else:
            selected = self.molecule._frame.loc[_set_caster(key)]
        try:
            return self.molecule._return_appropiate_type(selected)
        except AttributeError:
            return selected

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule._frame.loc[_set_caster(key[0]), _set_caster(key[1])] = value
        else:
            self.molecule._frame.loc[_set_caster(key)] = value


class _ILoc(_generic_Indexer):

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.molecule._frame.iloc[_set_caster(key[0]), _set_caster(key[1])]
        else:
            selected = self.molecule._frame.iloc[_set_caster(key)]
        try:
            return self.molecule._return_appropiate_type(selected)
        except AttributeError:
            return selected

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule._frame.iloc[_set_caster(key[0]), _set_caster(key[1])] = value
        else:
            self.molecule._frame.iloc[_set_caster(key)] = value



def _set_caster(x):
    """Pandas removed the possibility to index via sets, which we rely on

    Cast to a list when needed.
    """
    if isinstance(x, set):
        return list(x)
    else:
        return x
