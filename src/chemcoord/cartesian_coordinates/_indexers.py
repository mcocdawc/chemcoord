# -*- coding: utf-8 -*-

import warnings

from chemcoord.utilities._temporary_deprecation_workarounds import is_iterable

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
        df = self.molecule._frame
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=FutureWarning)
                if isinstance(key, tuple):
                    df.loc[_set_caster(key[0]), _set_caster(key[1])] = value
                else:
                    df.loc[_set_caster(key)] = value
        except FutureWarning:
            # We have the situation where value is of different type than
            #  the columns we assign to.
            # This happens for example when assigning sympy objects,
            #  i.e. symbolic variables, to a float column.
            # Currently this is not a problem in pandas and only raises a FutureWarning
            #  (as of version 2.2.), but to be futureproof make an explicit cast.
            # The `except FutureWarning:` has likely to become `except TypeError:`
            #  then in the future.
            if isinstance(key, tuple):
                if type(key[1]) is not str and is_iterable(key[1]):
                    self.molecule._frame = df.astype({k: 'O' for k in key[1]})
                else:
                    self.molecule._frame = df.astype({key[1]: 'O'})
                self.molecule._frame.loc[_set_caster(key[0]), _set_caster(key[1])] = value
            else:
                raise TypeError("Assignment not supported.")


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
        df = self.molecule._frame
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=FutureWarning)
                if isinstance(key, tuple):
                    df.iloc[_set_caster(key[0]), _set_caster(key[1])] = value
                else:
                    df.iloc[_set_caster(key)] = value
        except FutureWarning:
            # We have the situation where value is of different type than
            #  the columns we assign to.
            # This happens for example when assigning sympy objects,
            #  i.e. symbolic variables, to a float column.
            # Currently this is not a problem in pandas and only raises a FutureWarning
            #  (as of version 2.2.), but to be futureproof make an explicit cast.
            # The `except FutureWarning:` has likely to become `except TypeError:`
            #  then in the future.
            if isinstance(key, tuple):
                if type(key[1]) is not str and is_iterable(key[1]):
                    self.molecule._frame = df.astype({df.columns[k]: 'O' for k in key[1]})
                else:
                    self.molecule._frame = df.astype({df.columns[key[1]]: 'O'})
                self.molecule._frame.iloc[_set_caster(key[0]), _set_caster(key[1])] = value
            else:
                raise TypeError("Assignment not supported.")



def _set_caster(x):
    """Pandas removed the possibility to index via sets, which we rely on

    Cast to a list when needed.
    """
    if isinstance(x, set):
        return list(x)
    else:
        return x
