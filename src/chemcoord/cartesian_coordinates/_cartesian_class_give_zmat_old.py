
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
from chemcoord._exceptions import PhysicalMeaningError
from chemcoord.cartesian_coordinates._cartesian_class_core import \
    Cartesian_core
from chemcoord.internal_coordinates.zmat_class_main import Zmat
from chemcoord.utilities.set_utilities import pick
from chemcoord.configuration import settings


class Cartesian_give_zmat(Cartesian_core):
    def _get_buildlist(self, fixed_buildlist=None,
                       use_lookup=settings['defaults']['use_lookup']):
        """Create a buildlist for a Zmatrix.

        Args:
            fixed_buildlist (np.array): It is possible to provide the
                beginning of the buildlist. The rest is "figured" out
                automatically.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            np.array: buildlist
        """
        buildlist = np.zeros((self.n_atoms, 4)).astype('int64')
        if fixed_buildlist is not None:
            buildlist[:fixed_buildlist.shape[0], :] = fixed_buildlist
            start_row = fixed_buildlist.shape[0]
            already_built = set(fixed_buildlist[:, 0])
            to_be_built = set(self.index) - already_built
            convert_index = dict(zip(buildlist[:, 0], range(start_row)))
        else:
            start_row = 0
            already_built, to_be_built = set([]), set(self.index)
            convert_index = {}

        bond_dic = self.get_bonds(use_lookup=use_lookup)
        topologic_center = self.topologic_center()
        distance_to_topologic_center = self.distance_to(topologic_center)

        def update(already_built, to_be_built, new_atoms_set):
            """NOT SIDEEFFECT FREE.
            """
            for index in new_atoms_set:
                already_built.add(index)
                to_be_built.remove(index)
            return already_built, to_be_built

        def from_topologic_center(
                self,
                row_in_buildlist,
                already_built,
                to_be_built,
                first_time=False,
                third_time=False
                ):
            index_of_new_atom = distance_to_topologic_center.loc[
                to_be_built, 'distance'].idxmin()
            buildlist[row_in_buildlist, 0] = index_of_new_atom
            convert_index[index_of_new_atom] = row_in_buildlist
            if not first_time:
                bond_with = self.distance_to(
                    index_of_new_atom,
                    already_built).loc[:, 'distance'].idxmin()
                angle_with = self.distance_to(
                    bond_with,
                    already_built - set([bond_with]))['distance'].idxmin()
                buildlist[row_in_buildlist, 1:3] = [bond_with, angle_with]
                if not third_time:
                    dihedral_with = self.distance_to(
                        bond_with,
                        already_built - set([bond_with, angle_with])
                        )['distance'].idxmin()
                    buildlist[row_in_buildlist, 1:] = [
                        bond_with, angle_with, dihedral_with]
            new_row_to_modify = row_in_buildlist + 1
            already_built.add(index_of_new_atom)
            to_be_built.remove(index_of_new_atom)
            return new_row_to_modify, already_built, to_be_built

        def second_atom(self, already_built, to_be_built):
            new_atoms_set = bond_dic[buildlist[0, 0]] - already_built
            if new_atoms_set != set([]):
                index_of_new_atom = pick(new_atoms_set)
                convert_index[index_of_new_atom] = 1
                buildlist[1, 0] = index_of_new_atom
                buildlist[1, 1] = pick(already_built)
            else:
                new_row_to_modify, already_built, to_be_built = \
                    from_topologic_center(
                        self, row_in_buildlist, already_built, to_be_built)
            if len(new_atoms_set) > 1:
                use_index = buildlist[0, 0]
            else:
                use_index = buildlist[1, 0]
            new_row_to_modify = 2
            already_built.add(index_of_new_atom)
            to_be_built.remove(index_of_new_atom)
            return new_row_to_modify, already_built, to_be_built, use_index

        # TODO (3, 1) if atoms are not connected to it
        # TODO find nearest atom of 90
        def third_atom(self, already_built, to_be_built, use_index):
            new_atoms_set = bond_dic[use_index] - already_built
            if new_atoms_set != set([]):
                index_of_new_atom = pick(new_atoms_set)
                convert_index[index_of_new_atom] = 2
                buildlist[2, 0] = index_of_new_atom
                buildlist[2, 1] = use_index
                buildlist[2, 2] = pick(already_built - set([use_index]))
                if (
                        self.angle_degrees(buildlist[2, :]) < 10
                        or self.angle_degrees(buildlist[2, :]) > 170):
                    try:
                        index_of_new_atom = pick(
                            new_atoms_set - set([index_of_new_atom]))
                        convert_index[index_of_new_atom] = 2
                        buildlist[2, 0] = index_of_new_atom
                        buildlist[2, 1] = use_index
                        buildlist[2, 2] = pick(
                            already_built - set([use_index]))
                    except KeyError:
                        pass
                already_built.add(index_of_new_atom)
                to_be_built.remove(index_of_new_atom)
            else:
                new_row_to_modify, already_built, to_be_built = \
                    from_topologic_center(
                        # The two is hardcoded because of third atom.
                        self, 2, already_built, to_be_built, third_time=True)

            if len(new_atoms_set) > 1:
                use_index = use_index
            else:
                use_index = buildlist[2, 0]
            new_row_to_modify = 3
            return new_row_to_modify, already_built, to_be_built, use_index

        def pick_new_atoms(
                self,
                row_in_buildlist,
                already_built,
                to_be_built,
                use_given_index=None):
            """Get the indices of new atoms to be put in buildlist.

            Tries to get the atoms bonded to the one, that was last
                inserted into the buildlist. If the last atom is the
                end of a branch, it looks for the index of an atom
                that is the nearest atom to the topologic center and
                not built in yet.

            .. note:: It modifies the buildlist array which is global
                to this function.

            Args:
                row_in_buildlist (int): The row which has to be filled
                    at least.

            Returns:
                list: List of modified rows.
            """
            if use_given_index is None:
                new_atoms_set = (bond_dic[buildlist[row_in_buildlist-1, 0]]
                                 - already_built)

                if new_atoms_set != set([]):
                    update(already_built, to_be_built, new_atoms_set)
                    new_row_to_modify = row_in_buildlist + len(new_atoms_set)
                    new_atoms_list = list(new_atoms_set)
                    bond_with = buildlist[row_in_buildlist - 1, 0]
                    angle_with = buildlist[convert_index[bond_with], 1]
                    dihedral_with = buildlist[convert_index[bond_with], 2]
                    buildlist[
                        row_in_buildlist: new_row_to_modify,
                        0] = new_atoms_list
                    buildlist[
                        row_in_buildlist: new_row_to_modify,
                        1] = bond_with
                    buildlist[
                        row_in_buildlist: new_row_to_modify,
                        2] = angle_with
                    buildlist[
                        row_in_buildlist: new_row_to_modify,
                        3] = dihedral_with
                    for key, value in zip(
                            new_atoms_list,
                            range(row_in_buildlist, new_row_to_modify)):
                        convert_index[key] = value
                else:
                    new_row_to_modify, already_built, to_be_built = \
                        from_topologic_center(
                            self, row_in_buildlist, already_built, to_be_built)

            else:
                new_atoms_set = bond_dic[use_given_index] - already_built
                new_row_to_modify = row_in_buildlist + len(new_atoms_set)
                new_atoms_list = list(new_atoms_set)
                bond_with = use_given_index
                angle_with, dihedral_with = (
                    already_built - set([use_given_index]))
                buildlist[
                    row_in_buildlist: new_row_to_modify, 0] = new_atoms_list
                buildlist[
                    row_in_buildlist: new_row_to_modify, 1] = bond_with
                buildlist[
                    row_in_buildlist: new_row_to_modify, 2] = angle_with
                buildlist[
                    row_in_buildlist: new_row_to_modify, 3] = dihedral_with
                update(already_built, to_be_built, new_atoms_set)
                for key, value in zip(
                        new_atoms_list,
                        range(row_in_buildlist, new_row_to_modify)):
                    convert_index[key] = value

            return new_row_to_modify, already_built, to_be_built

        row = start_row
        if 0 <= row <= 2:
            if row == 0 & 0 < buildlist.shape[0]:
                row, already_built, to_be_built = from_topologic_center(
                    self,
                    row,
                    already_built,
                    to_be_built,
                    first_time=True)
            if row == 1 & 1 < buildlist.shape[0]:
                row, already_built, to_be_built, use_index = second_atom(
                    self, already_built, to_be_built)
            if row == 2 & 2 < buildlist.shape[0]:
                row, already_built, to_be_built, use_index = third_atom(
                    self, already_built, to_be_built, use_index)
            if row < buildlist.shape[0]:
                row, already_built, to_be_built = pick_new_atoms(
                    self, row, already_built, to_be_built, use_index)

        while row < buildlist.shape[0]:
            row, already_built, to_be_built = pick_new_atoms(
                self, row, already_built, to_be_built)

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
        zmat.metadata = self.metadata
        zmat._metadata = self.metadata.copy()
        keys_not_to_keep = []  # Because they don't make **physically** sense
        # for internal coordinates
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
