from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

try:
    import itertools.imap as map
except ImportError:
    pass

import numpy as np
import pandas as pd
import math as m
from chemcoord._generic_classes._common_class import _common_class
import chemcoord.internal_coordinates._indexers_for_zmat_class as indexers
from chemcoord._exceptions import PhysicalMeaningError
from chemcoord.configuration import settings


# TODO implement is_physical
class Zmat_core(_common_class):
    """The main class for dealing with internal coordinates.
    """
    _required_cols = frozenset({
        'atom', 'bond_with', 'bond', 'angle_with', 'angle',
        'dihedral_with', 'dihedral'})

    def __init__(self, frame):
        """How to initialize a Zmat instance.

        Args:
            init (pd.DataFrame): A Dataframe with at least the columns
                ``['atom', 'bond_with', 'bond', 'angle_with', 'angle',
                'dihedral_with', 'dihedral']``.
                Where ``'atom'`` is a string for the elementsymbol.

        Returns:
            Zmat: A new zmat instance.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValueError('Need a pd.DataFrame as input')
        if not self._required_cols <= set(frame.columns):
            raise PhysicalMeaningError('There are columns missing for a \
                                       meaningful description of a molecule')
        self.frame = frame.copy()
        self.metadata = {}
        self._metadata = {}

    @property
    def loc(self):
        """pew pew
        """
        return indexers._Loc(self)

    @property
    def iloc(self):
        """pew pew
        """
        return indexers._ILoc(self)

    # def __getitem__(self, key):
    #     # overwrites the method defined in _pandas_wrapper
    #     frame = self.frame.loc[key[0], key[1]]
    #     try:
    #         if self._required_cols <= (frame.columns):
    #             molecule = self.__class__(frame)
    #             # NOTE here is the difference to the _pandas_wrapper definition
    #             # TODO make clear in documentation that metadata is an
    #             # alias/pointer
    #             # TODO persistent attributes have to be inserted here
    #             molecule.metadata = self.metadata
    #             molecule._metadata = self.metadata.copy()
    #             keys_not_to_keep = [
    #                 'bond_dict'  # You could end up with loose ends
    #                 ]
    #             for key in keys_not_to_keep:
    #                 try:
    #                     molecule._metadata.pop(key)
    #                 except KeyError:
    #                     pass
    #             return molecule
    #         else:
    #             return frame
    #     except AttributeError:
    #         # A series and not a DataFrame was returned
    #         return frame
    #
    # def __setitem__(self, key, value):
    #     self.frame.loc[key[0], key[1]] = value

    def __add__(self, other):
        selection = ['atom', 'bond_with', 'angle_with', 'dihedral_with']
        coords = ['bond', 'angle', 'dihedral']
        new = self.copy()
        new._metadata['absolute_zmat'] = (self._metadata['absolute_zmat']
                                          and other._metadata['absolute_zmat'])
        try:
            assert (self.index == other.index).all()
            # TODO default values for _metadata
            if new._metadata['absolute_zmat']:
                assert np.alltrue(self.loc[:, selection] == other.loc[:, selection])
            else:
                self.loc[:, selection].isnull()
                tested_where_equal = (self.loc[:, selection] == other.loc[:, selection])
                tested_where_nan = (self.loc[:, selection].isnull()
                                    | other.loc[:, selection].isnull())
                for column in selection:
                    tested_where_equal[tested_where_nan[column], column] = True
                assert np.alltrue(tested_where_equal)

            new[:, coords] = self.loc[:, coords] + other.loc[:, coords]
        except AssertionError:
            raise PhysicalMeaningError("You can add only those zmatrices that \
have the same index, use the same buildlist, have the same ordering... \
The only allowed difference is ['bond', 'angle', 'dihedral']")
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def _to_Zmat(self):
        return self.copy()

    def get_buildlist(self):
        """Return the buildlist which is necessary to create this Zmat

        Args:
            None

        Returns:
            np.array: Buildlist
        """
        columns = ['temporary_index', 'bond_with', 'angle_with', 'dihedral_with']
        tmp = self.insert(0, 'temporary_index', self.index)
        buildlist = tmp.loc[:, columns].values.astype('int64')
        buildlist[0, 1:] = 0
        buildlist[1, 2:] = 0
        buildlist[2, 3:] = 0
        return buildlist

    def change_numbering(self, new_index=None, inplace=False):
        """Change numbering to a new index.

        Changes the numbering of index and all dependent numbering
            (bond_with...) to a new_index.
        The user has to make sure that the new_index consists of distinct
            elements.

        Args:
            new_index (list): If None the new_index is taken from 1 to the
            number of atoms.

        Returns:
            Zmat: Reindexed version of the zmatrix.
        """
        output = self if inplace else self.copy()
        old_index = output.index

        if (new_index is None):
            new_index = range(1, self.n_atoms + 1)
        else:
            new_index = new_index
        assert len(new_index) == len(old_index)

        output.index = new_index

        cols = ['bond_with', 'angle_with', 'dihedral_with']
        output.loc[:, cols] = output.loc[:, cols].replace(old_index, new_index)

        if not inplace:
            return output

    def has_same_sumformula(self, other):
        same_atoms = True
        for atom in set(self.loc[:, 'atom']):
            own_atom_number = self.loc[self.loc[:, 'atom'] == atom, :].shape[0]
            other_atom_number = other.loc[other.loc[:, 'atom'] == atom, :].shape[0]
            same_atoms = (own_atom_number == other_atom_number)
            if not same_atoms:
                break
        return same_atoms
