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
from chemcoord._exceptions import \
    PhysicalMeaning, UndefinedCoordinateSystem, IllegalArgumentCombination
from chemcoord.cartesian_coordinates._cartesian_class_core import \
    Cartesian_core
from chemcoord.internal_coordinates.zmat_class_main import Zmat
from chemcoord.utilities.set_utilities import pick
from chemcoord.configuration import settings


class Cartesian_give_zmat(Cartesian_core):
    @staticmethod
    def _check_construction_table(construction_table):
        """Checks if a construction table uses valid references.
        Raises an exception (UndefinedCoordinateSystem) otherwise.
        """
        c_table = construction_table
        for row, i in enumerate(c_table.index):
            give_message = ("Not a valid construction table. "
                            "The index {i} uses an invalid reference").format
            if row == 0:
                pass
            elif row == 1:
                if c_table.loc[i, 'bond_with'] not in c_table.index[:row]:
                    raise UndefinedCoordinateSystem(give_message(i=i))
            elif row == 2:
                reference = c_table.loc[i, ['bond_with', 'angle_with']]
                if not reference.isin(c_table.index[:row]).all():
                    raise UndefinedCoordinateSystem(give_message(i=i))
            else:
                reference = c_table.loc[i, ['bond_with', 'angle_with']]
                if not reference.isin(c_table.index[:row]).all():
                    raise UndefinedCoordinateSystem(give_message(i=i))

    def _get_construction_table(self,
                                start_atom=None,
                                predefined_table=None,
                                use_lookup=settings['defaults']['use_lookup'],
                                bond_dict=None):
        """Create a construction table for a Zmatrix.

        A construction table is basically a Zmatrix without the values
        for the bond lenghts, angles and dihedrals.
        It contains the whole information about which reference atoms
        are used by each atom in the Zmatrix.

        This method creates a so called "chemical" construction table,
        which makes use of the connectivity table in this molecule.

        By default the first atom is the one nearest to the topologic center.
        (Compare with :meth:`~Cartesian.topologic_center()`)

        Args:
            start_atom (index): An index for the first atom may be provided.
            predefined_table (pd.DataFrame): An uncomplete construction table
                may be provided. The rest is created automatically.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.
            bond_dict (OrderedDict): If a connectivity table is provided, it is
                not recalculated.

        Returns:
            pd.DataFrame: Construction table
        """
        if start_atom is not None and predefined_table is not None:
            raise IllegalArgumentCombination('Either start_atom or '
                                             'predefined_table has to be None')
        if bond_dict is None:
            bond_dict = self._give_val_sorted_bond_dict(use_lookup=use_lookup)
        if predefined_table is not None:
            self._check_construction_table(predefined_table)
            construction_table = predefined_table.copy()
            construction_table.columns = ['b', 'a', 'd']

        if predefined_table is None:
            if start_atom is None:
                molecule = self.distance_to(self.topologic_center())
                i = molecule['distance'].idxmin()
            else:
                i = start_atom
            construction_table = {i: {'b': np.nan, 'a': np.nan, 'd': np.nan}}
            order_of_definition = [i]
            user_defined = set()
        else:
            order_of_definition = list(construction_table.index)
            user_defined = set(construction_table.index)
            i = construction_table.index[0]
            construction_table = construction_table.to_dict(orient='index')
        visited = {i}

        if self.n_atoms > 1:
            parent = {j: i for j in bond_dict[i]}
            work_bond_dict = OrderedDict(
                [(j, bond_dict[j] - visited) for j in bond_dict[i]])
        else:
            parent, work_bond_dict = {}, {}

        while work_bond_dict:
            new_work_bond_dict = OrderedDict()
            for i in work_bond_dict:
                if i in visited:
                    continue
                if i not in user_defined:
                    b = parent[i]
                    if b in order_of_definition[:3]:
                        if len(construction_table) == 1:
                            construction_table[i] = {'b': b}
                        elif len(construction_table) == 2:
                            a = (bond_dict[b] & visited)[0]
                            construction_table[i] = {'b': b, 'a': a}
                        else:
                            try:
                                a = parent[b]
                            except KeyError:
                                a = (bond_dict[b] & visited)[0]
                            try:
                                d = parent[a]
                                if d in set([b, a]):
                                    raise KeyError
                            except KeyError:
                                try:
                                    d = ((bond_dict[a] & visited)
                                         - set([b, a]))[0]
                                except IndexError:
                                    d = ((bond_dict[b] & visited)
                                         - set([b, a]))[0]
                            construction_table[i] = {'b': b, 'a': a, 'd': d}
                    else:
                        a, d = [construction_table[b][k] for k in ['b', 'a']]
                        construction_table[i] = {'b': b, 'a': a, 'd': d}
                    order_of_definition.append(i)

                visited.add(i)
                for j in work_bond_dict[i]:
                    new_work_bond_dict[j] = bond_dict[j] - visited
                    parent[j] = i
            work_bond_dict = new_work_bond_dict

        output = pd.DataFrame.from_dict(construction_table, orient='index')
        output = output.fillna(0).astype('int64')
        output = output.loc[order_of_definition, ['b', 'a', 'd']]
        output.columns = ['bond_with', 'angle_with', 'dihedral_with']
        return output

    def _shortest_distance(self, other):
        coords = ['x', 'y', 'z']
        old_indices = self.index, other.index
        self.index, other.index = range(self.n_atoms), range(other.n_atoms)
        self_positions = self.loc[:, coords].values
        other_positions = other.loc[:, coords].values

        coord1 = self_positions[:, 0]
        coord2 = other_positions[:, 0]
        squared_distances = (coord1 - coord2[:, None])**2
        for i in range(1, 3):
            coord1= self_positions[:, i]
            coord2= other_positions[:, i]
            squared_distances += (coord1 - coord2[:, None])**2
        i, j = np.unravel_index(squared_distances.argmin(),
                                squared_distances.shape)
        distance = np.sqrt(squared_distances[i, j])
        self.index, other.index = old_indices
        return i, j, distance


    def get_construction_table(self,
                               use_lookup=settings['defaults']['use_lookup']):
        """Create a construction table for a Zmatrix.

        A construction table is basically a Zmatrix without the values
        for the bond lenghts, angles and dihedrals.
        It contains the whole information about which reference atoms
        are used by each atom in the Zmatrix.

        This method creates a so called "chemical" construction table,
        which makes use of the connectivity table in this molecule.

        By default the first atom is the one nearest to the topologic center.
        (Compare with :meth:`~Cartesian.topologic_center()`)

        Args:
            start_atom (index): An index for the first atom may be provided.
            predefined_table (pd.DataFrame): An uncomplete construction table
                may be provided. The rest is created automatically.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            pd.DataFrame: Construction table
        """
        bond_dict = self._give_val_sorted_bond_dict(use_lookup=use_lookup)
        fragments = sorted(self.fragmentate(),
                           key=lambda x: len(x), reverse=True)
        molecule = fragments[0]
        full_constr_table = molecule._get_construction_table(use_lookup=True)
        full_constr_table.columns = ['b', 'a', 'd']
        included = list(molecule.index)
        for molecule in fragments[1:]:
            i, j = molecule._shortest_distance(self.loc[included])
            constr_table = molecule._get_construction_table(start_atom=i,
                                                            use_lookup=True)
            constr_table.columns = ['b', 'a', 'd']
            a, d = full_constr_table.loc[j, ['b', 'a']]
            constr_table.loc[i] = j, a, d
            constr_table.iloc[1, ['a', 'd']] = b, a
            constr_table.iloc[2, 'd'] = b
            pd.concat([full_constr_table, constr_table])
        return full_constr_table

    def _clean_dihedral(self, construction_table, bond_dict=None,
                        use_lookup=settings['defaults']['use_lookup']):
        """Reindexe the dihedral defining atom if colinear.

        Args:
            construction_table (pd.DataFrame):
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.
            bond_dict (OrderedDict): If a connectivity table is provided, it is
                not recalculated.

        Returns:
            pd.DataFrame: construction_table
        """
        if bond_dict is None:
            bond_dict = self._give_val_sorted_bond_dict(use_lookup=use_lookup)
        c_table = construction_table.copy()
        c_table.columns = ['b', 'a', 'd']
        angles = self.angle_degrees(c_table.iloc[3:, :])
        problem_index = np.nonzero((175 < angles) | (angles < 5))[0]
        rename = dict(enumerate(c_table.index[3:]))
        problem_index = [rename[i] for i in problem_index]

        print(problem_index)
        for i in problem_index:
            loc_i = c_table.index.get_loc(i)
            b, a, problem_d = c_table.loc[i, ['b', 'a', 'd']]
            try:
                d = (bond_dict[a] - {b, a, problem_d}
                     - set(c_table.index[loc_i:]))[0]
            except IndexError:
                visited = set(c_table.index[loc_i:]) | {b, a, problem_d}
                tmp_bond_dict = OrderedDict([(j, bond_dict[j] - visited)
                                             for j in bond_dict[problem_d]])
                found = False
                while tmp_bond_dict and not found:
                    new_tmp_bond_dict = OrderedDict()
                    for new_d in tmp_bond_dict:
                        if new_d in visited:
                            continue
                        angle = self.angle_degrees([b, a, new_d])[0]
                        if (5 > angle) or (angle > 175):
                            visited.add(new_d)
                            for j in tmp_bond_dict[new_d]:
                                new_tmp_bond_dict[j] = bond_dict[j] - built
                        else:
                            found = True
                            c_table.loc[i, 'd'] = new_d
                    tmp_bond_dict = new_tmp_bond_dict
        c_table.columns = ['bond_with', 'angle_with', 'dihedral_with']
        return c_table

    def _build_zmat(self, construction_table):
        """Create the Zmatrix from a construction table.

        Args:
            Construction table (pd.DataFrame):

        Returns:
            Zmat: A new instance of :class:`Zmat`.
        """
        c_table = construction_table
        index = c_table.index
        default_cols = ['atom', 'bond_with', 'bond', 'angle_with', 'angle',
                        'dihedral_with', 'dihedral']
        optional_cols = list(set(self.columns) - {'atom', 'x', 'y', 'z'})

        zmat_frame = pd.DataFrame(columns=default_cols + optional_cols,
                                  dtype='float', index=index)

        zmat_frame.loc[:, optional_cols] = self.loc[index, optional_cols]

        bonds = self.bond_lengths(c_table, start_row=1)
        angles = self.angle_degrees(c_table, start_row=2)
        dihedrals = self.dihedral_degrees(c_table, start_row=3)

        zmat_frame.loc[index, 'atom'] = self.loc[index, 'atom']
        zmat_frame.loc[index, 'bond_with'] = c_table.iloc[1:, 0]
        zmat_frame.loc[index[1:], 'bond'] = bonds
        zmat_frame.loc[index[2:], 'angle_with'] = c_table.iloc[2:, 1]
        zmat_frame.loc[index[2:], 'angle'] = angles
        zmat_frame.loc[index[3:], 'dihedral_with'] = c_table.iloc[3:, 2]
        zmat_frame.loc[index[3:], 'dihedral'] = dihedrals

        lines = np.full(self.n_atoms, True, dtype='bool')
        lines[:3] = False
        zmat_frame.loc[zmat_frame['dihedral'].isnull() & lines, 'dihedral'] = 0

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
                    row = len(big_molecule)
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

        Zmat = self._build_zmat(buildlist)
        Zmat.metadata = self.metadata.copy()
        keys_to_keep = []
        for key in keys_to_keep:
            Zmat._metadata[key] = self._metadata[key].copy()
        return Zmat

    def to_zmat(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.give_zmat`
        """
        message = 'Will be removed in the future. Please use give_zmat.'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.give_zmat(*args, **kwargs)
