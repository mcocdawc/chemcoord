# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
try:
    import itertools.izip as zip
except ImportError:
    pass
import numpy as np
import pandas as pd
import warnings
from sortedcontainers import SortedSet
from collections import OrderedDict
from chemcoord._exceptions import PhysicalMeaningError
from chemcoord.cartesian_coordinates._cartesian_class_core import \
    Cartesian_core
from chemcoord.internal_coordinates.zmat_class_main import Zmat
from chemcoord.utilities.set_utilities import pick
from chemcoord.configuration import settings


class Cartesian_give_zmat(Cartesian_core):
    def _get_buildlist(
            self, use_lookup=settings['defaults']['use_lookup']):
        topologic_center = self.topologic_center()
        molecule = self.distance_to(topologic_center, sort=True)
        bond_dict = molecule._give_val_sorted_bond_dict()

        # The assignment of an arbitrary integer arb_int lateron
        # is just done to preserve the type 'int64' in the DataFrame
        arb_int = 77
        # ['b', 'a', 'd'] is the abbreviation for
        # ['bond_with', 'angle_with', 'dihedral_with']
        buildlist = pd.DataFrame(columns=['b', 'a', 'd'])
        built = set([])

        i = molecule.index[0]
        buildlist.loc[i] = [arb_int, arb_int, arb_int]
        buildlist = buildlist.astype('int64')
        built.add(i)
        if molecule.n_atoms > 1:
            parent = {j: i for j in bond_dict[i]}
            work_bond_dict = OrderedDict([(j, bond_dict[j] - built)
                                          for j in bond_dict[i]])
        else:
            parent, work_bond_dict = {}, {}

        while work_bond_dict:
            new_work_bond_dict = OrderedDict()
            for i in work_bond_dict:
                if i in built:
                    continue
                b = parent[i]
                if b in buildlist.index[:3]:
                    if len(buildlist) == 1:
                        reference = b, arb_int, arb_int
                    elif len(buildlist) == 2:
                        a = (bond_dict[b] & built)[0]
                        reference = b, a, arb_int
                    else:
                        try:
                            a = parent[b]
                        except KeyError:
                            a = (bond_dict[b] & built)[0]
                        try:
                            d = parent[a]
                        except KeyError:
                            try:
                                d = ((bond_dict[a] & built) - set([b, a]))[0]
                            except IndexError:
                                d = ((bond_dict[b] & built) - set([b, a]))[0]
                        reference = b, a, d
                else:
                    a, d = buildlist.loc[b, ['b', 'a']]
                    reference = b, a, d
                buildlist.loc[i] = reference
                built.add(i)
                for j in work_bond_dict[i]:
                    new_work_bond_dict[j] = bond_dict[j] - built
                    parent[j] = i
            work_bond_dict = new_work_bond_dict
        return buildlist

    def _clean_dihedral(self, buildlist_to_check,
                        use_lookup=settings['defaults']['use_lookup']):
        """Reindexe the dihedral defining atom if colinear.

        Args:
            buildlist (np.array):
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            np.array: modified_buildlist
        """
        buildlist = buildlist_to_check.copy()

        bond_dic = self.get_bonds(use_lookup=use_lookup)

        angles = self.angle_degrees(buildlist[3:, 1:])

        test_vector = np.logical_or(170 < angles, angles < 10)
        problematic_indices = np.nonzero(test_vector)[0]

        # look for index + 3 because index start directly at dihedrals
        for index in problematic_indices:
            try:
                already_tested = set([])
                found = False
                while not found:
                    new_dihedral = pick(bond_dic[buildlist[index + 3, 2]]
                                        - set(buildlist[index + 3, [0, 1, 3]])
                                        - set(buildlist[(index + 3):, 0])
                                        - already_tested)
                    already_tested.add(new_dihedral)
                    temp_buildlist = buildlist[index + 3]
                    temp_buildlist[3] = new_dihedral
                    angle = self.angle_degrees(temp_buildlist[1:])
                    found = True if 10 < angle < 170 else False
            except KeyError:
                origin = buildlist[index + 3, 2]
                could_be_reference = set(buildlist[: index + 3, 0])

                sorted_Cartesian = self.distance_to(
                    origin,
                    indices_of_other_atoms=could_be_reference)
                sorted_Cartesian.frame = \
                    sorted_Cartesian.frame.sort_values(by='distance')

                buildlist_for_new_dihedral_with = np.empty(
                    (len(could_be_reference), 3), dtype='int64')
                bond_with, angle_with = buildlist[index + 3, [1, 2]]
                buildlist_for_new_dihedral_with[:, 0] = bond_with
                buildlist_for_new_dihedral_with[:, 1] = angle_with
                dihedral_with = sorted_Cartesian.index
                buildlist_for_new_dihedral_with[:, 2] = dihedral_with
                angles = self.angle_degrees(buildlist_for_new_dihedral_with)

                test_vector = np.logical_and(170 > angles, angles > 10)
                new_dihedral = dihedral_with[np.nonzero(test_vector)[0][0]]

                buildlist[index + 3, 3] = new_dihedral
        return buildlist

    def _build_zmat(self, buildlist):
        """Create the zmatrix from a buildlist.

        Args:
            buildlist (np.array):

        Returns:
            Zmat: A new instance of :class:`zmat_functions.Zmat`.
        """
        indexlist = buildlist[:, 0]

        default_columns = [
            'atom', 'bond_with', 'bond', 'angle_with',
            'angle', 'dihedral_with', 'dihedral']
        additional_columns = list(set(self.columns)
                                  - set(['atom', 'x', 'y', 'z']))

        zmat_frame = pd.DataFrame(
            columns=default_columns + additional_columns,
            dtype='float',
            index=indexlist)

        zmat_frame.loc[:, additional_columns] = self.loc[indexlist,
                                                         additional_columns]

        bonds = self.bond_lengths(buildlist, start_row=1)
        angles = self.angle_degrees(buildlist, start_row=2)
        dihedrals = self.dihedral_degrees(buildlist, start_row=3)

        zmat_frame.loc[indexlist, 'atom'] = self.loc[indexlist, 'atom']
        zmat_frame.loc[indexlist[1:], 'bond_with'] = buildlist[1:, 1]
        zmat_frame.loc[indexlist[1:], 'bond'] = bonds
        zmat_frame.loc[indexlist[2:], 'angle_with'] = buildlist[2:, 2]
        zmat_frame.loc[indexlist[2:], 'angle'] = angles
        zmat_frame.loc[indexlist[3:], 'dihedral_with'] = buildlist[3:, 3]
        zmat_frame.loc[indexlist[3:], 'dihedral'] = dihedrals

        lines = np.full(self.n_atoms, True, dtype='bool')
        lines[:3] = False
        zmat_frame.loc[zmat_frame['dihedral'].isnull() & lines, 'dihedral'] = 0

        # return zmat_frame
        # TODO
        return Zmat(zmat_frame)

    def give_zmat(self, buildlist=None, fragment_list=None,
                  check_linearity=True,
                  use_lookup=settings['defaults']['use_lookup']):
        """Transform to internal coordinates.

        Transforming to internal coordinates involves basically three
        steps:

        1. Define an order of how to build.

        2. Check for problematic local linearity. In this algorithm an
        angle with ``170 < angle < 10`` is assumed to be linear.
        This is not the mathematical definition, but makes it safer
        against "floating point noise"

        3. Calculate the bond lengths, angles and dihedrals using the
        references defined in step 1 and 2.

        In the first two steps a so called ``buildlist`` is created.
        This is basically a ``np.array`` of shape ``(n_atoms, 4)`` and
        integer type.

        The four columns are ``['own_index', 'bond_with', 'angle_with',
        'dihedral_with']``.
        This means that usually the upper right triangle can be any
        number, because for example the first atom has no other
        atom as reference.

        It is important to know, that getting the buildlist is a very
        costly step since the algoritym tries to make some guesses
        based on the connectivity to create a "chemical" zmatrix.

        If you create several zmatrices based on the same references
        you can save the buildlist of a zmatrix with
        :meth:`Zmat.get_buildlist`.
        If you then pass the buildlist as argument to ``to_zmat``,
        then the algorithm directly starts with step 3.


        Another thing is that you can specify fragments.
        For this purpose the function :meth:`Cartesian.get_fragment`
        is quite handy.
        An element of fragment_list looks like::

            (fragment, connections)

        Fragment is a ``Cartesian`` instance and connections is a
        ``(3, 4)`` numpy integer array, that defines how the
        fragment is connected to the molecule.

        Args:
            buildlist (np.array):
            fragment_list (list):
            check_linearity (bool):
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            Zmat: A new instance of :class:`~.zmat_functions.Zmat`.
        """
        if buildlist is None:
            if fragment_list is None:
                buildlist = self._get_buildlist(use_lookup=use_lookup)
            else:
                def create_big_molecule(self, fragment_list):
                    def prepare_variables(self, fragment_list):
                        buildlist = np.empty((self.n_atoms, 4), dtype='int64')
                        fragment_index = set([])
                        for fragment_tpl in fragment_list:
                            fragment_index |= set(
                                fragment_tpl[0].index)
                        big_molecule_index = (
                            set(self.index) - fragment_index)
                        return buildlist, big_molecule_index
                    buildlist, big_molecule_index = prepare_variables(
                        self, fragment_list)
                    big_molecule = self.loc[big_molecule_index, :]
                    row = big_molecule.n_atoms
                    buildlist[: row, :] = big_molecule._get_buildlist(
                        use_lookup=use_lookup)
                    return buildlist, big_molecule, row

                def add_fragment(
                        self, fragment_tpl, big_molecule, buildlist, row):
                    next_row = row + fragment_tpl[0].n_atoms
                    buildlist[row: next_row, :] = \
                        fragment_tpl[0]._get_buildlist(
                            fragment_tpl[1], use_lookup=use_lookup)
                    return buildlist, big_molecule, row

                buildlist, big_molecule, row = create_big_molecule(
                    self, fragment_list)

                for fragment_tpl in fragment_list:
                    buildlist, big_molecule, row = add_fragment(
                        self, fragment_tpl, big_molecule, buildlist, row)

        if check_linearity:
            buildlist = self._clean_dihedral(buildlist, use_lookup=True)

        zmat = self._build_zmat(buildlist)
        zmat.metadata = self.metadata.copy()
        zmat._metadata = self.metadata.copy()
        # Because they don't make **physically** sense
        # for internal coordinates
        keys_not_to_keep = ['bond_dict']
        for key in keys_not_to_keep:
            try:
                zmat._metadata.pop(key)
            except KeyError:
                pass
        return zmat

    def to_zmat(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.give_zmat`
        """
        message = 'Will be removed in the future. Please use give_zmat.'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.give_zmat(*args, **kwargs)
