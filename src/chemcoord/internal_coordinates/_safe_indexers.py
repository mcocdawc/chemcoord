# -*- coding: utf-8 -*-

from chemcoord._generic_classes._indexers import _ILoc, _Loc
from chemcoord._exceptions import InvalidReference


class _Safe_Loc(_Loc):

    # overwrites existing method
    def __setitem__(self, key, value):
        zmat_after_assignment = self.molecule.copy()
        if isinstance(key, tuple):
            zmat_after_assignment._frame.loc[key[0], key[1]] = value
        else:
            zmat_after_assignment._frame.loc[key] = value

        try:
            zmat_after_assignment.give_cartesian()
            if isinstance(key, tuple):
                self.molecule._frame.iloc[key[0], key[1]] = value
            else:
                self.molecule._frame.iloc[key] = value
        except InvalidReference as e:
            e.zmat_after_assignment = zmat_after_assignment
            raise e


class _Safe_ILoc(_ILoc):

    # overwrites existing method
    def __setitem__(self, key, value):
        raise NotImplemented
        # before_assignment = self.molecule.copy()
        # if isinstance(key, tuple):
        #     self.molecule._frame.iloc[key[0], key[1]] = value
        # else:
        #     self.molecule._frame.iloc[key] = value
        #
        # try:
        #     self.molecule._test_give_cartesian()
        # except InvalidReference as e:
        #     e.zmat_before_assignment = zmat_before_assignment
        #     raise e
