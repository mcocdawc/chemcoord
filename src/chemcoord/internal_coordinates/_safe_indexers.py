# -*- coding: utf-8 -*-

from chemcoord._generic_classes._indexers import _ILoc, _Loc


class _Safe_Loc(_Loc):

    # overwrites existing method
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule.frame.loc[key[0], key[1]] = value
        else:
            self.molecule.frame.loc[key] = value

        self.molecule._test_give_cartesian()


class _Safe_ILoc(_ILoc):

    # overwrites existing method
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.molecule.frame.iloc[key[0], key[1]] = value
        else:
            self.molecule.frame.iloc[key] = value

        self.molecule._test_give_cartesian()
