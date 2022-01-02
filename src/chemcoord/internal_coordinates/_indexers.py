# -*- coding: utf-8 -*-

from chemcoord.exceptions import InvalidReference


class _generic_Indexer(object):
    def __init__(self, molecule):
        self.molecule = molecule

    def __getitem__(self, key):
        indexer = getattr(self.molecule._frame, self.indexer)
        if isinstance(key, tuple):
            selected = indexer[key[0], key[1]]
        else:
            selected = indexer[key]
        return selected


class _Loc(_generic_Indexer):
    indexer = 'loc'

class _ILoc(_generic_Indexer):
    indexer = 'iloc'

class _Unsafe_base():
    def __setitem__(self, key, value):
        indexer = getattr(self.molecule._frame, self.indexer)
        if isinstance(key, tuple):
            indexer[key[0], key[1]] = value
        else:
            indexer[key] = value

class _SafeBase():
    def __setitem__(self, key, value):
        try:
            self.molecule._metadata['last_valid_cartesian'] = self.molecule.get_cartesian()
        except TypeError:
            # We are here because of Sympy
            pass

        if self.molecule.dummy_manipulation_allowed:
            molecule = self.molecule
        else:
            molecule = self.molecule.copy()
        indexer = getattr(molecule._frame, self.indexer)

        if isinstance(key, tuple):
            indexer[key[0], key[1]] = value
        else:
            indexer[key] = value

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

        if self.molecule.pure_internal_mov:
            ref = self.molecule._metadata['last_valid_cartesian']
            new = self.molecule.get_cartesian()
            # TODO(@Oskar): Ensure that this works with Dummy atoms as well
            rotated = ref.align(new, mass_weight=True)[1]
            c_table = self.molecule.loc[:, ['b', 'a', 'd']]
            self.molecule._frame = rotated.get_zmat(c_table)._frame



class _Unsafe_Loc(_Loc, _Unsafe_base):
    pass


class _Safe_Loc(_Loc, _SafeBase):
    pass


class _Unsafe_ILoc(_ILoc, _Unsafe_base):
    pass

class _Safe_ILoc(_ILoc, _SafeBase):
    pass
