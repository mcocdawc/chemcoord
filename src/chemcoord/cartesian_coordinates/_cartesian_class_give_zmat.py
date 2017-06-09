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
from collections import OrderedDict, deque
from itertools import permutations, cycle
from chemcoord._exceptions import \
    PhysicalMeaning, UndefinedCoordinateSystem, IllegalArgumentCombination, \
    InvalidReference
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
                if c_table.loc[i, 'b'] not in c_table.index[:row]:
                    raise UndefinedCoordinateSystem(give_message(i=i))
            elif row == 2:
                reference = c_table.loc[i, ['b', 'a']]
                if not reference.isin(c_table.index[:row]).all():
                    raise UndefinedCoordinateSystem(give_message(i=i))
            else:
                reference = c_table.loc[i, ['b', 'a', 'd']]
                if not reference.isin(c_table.index[:row]).all():
                    raise UndefinedCoordinateSystem(give_message(i=i))

    def _get_constr_table(self, start_atom=None, predefined_table=None,
                          use_lookup=settings['defaults']['use_lookup'],
                          bond_dict=None):
        """Create a construction table.

        It is written under the assumption that self is one
        connected molecule.
        """
        def modify_priority(bond_dict, user_defined):
            for j in reversed(user_defined):
                try:
                    work_bond_dict.move_to_end(j, last=False)
                except KeyError:
                    pass
        if start_atom is not None and predefined_table is not None:
            raise IllegalArgumentCombination('Either start_atom or '
                                             'predefined_table has to be None')
        if bond_dict is None:
            bond_dict = self._give_val_sorted_bond_dict(use_lookup=use_lookup)
        if predefined_table is not None:
            self._check_construction_table(predefined_table)
            construction_table = predefined_table.copy()

        if predefined_table is None:
            if start_atom is None:
                molecule = self.distance_to(self.topologic_center())
                i = molecule['distance'].idxmin()
            else:
                i = start_atom
            order_of_definition = [i]
            user_defined = []
            construction_table = {i: {'b': -4, 'a': -3, 'd': -1}}
        else:
            i = construction_table.index[0]
            order_of_definition = list(construction_table.index)
            user_defined = list(construction_table.index)
            construction_table = construction_table.to_dict(orient='index')

        visited = {i}
        if self.n_atoms > 1:
            parent = {j: i for j in bond_dict[i]}
            bond_dict[i]
            work_bond_dict = OrderedDict(
                [(j, bond_dict[j] - visited) for j in bond_dict[i]])
            modify_priority(work_bond_dict, user_defined)
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
                        if len(order_of_definition) == 1:
                            construction_table[i] = {'b': b, 'a': -3, 'd': -1}
                        elif len(order_of_definition) == 2:
                            a = (bond_dict[b] & visited)[0]
                            construction_table[i] = {'b': b, 'a': a, 'd': -1}
                        else:
                            try:
                                a = parent[b]
                            except KeyError:
                                a = (bond_dict[b] & visited)[0]
                            try:
                                d = parent[a]
                                if d in set([b, a]):
                                    message = "Don't make tautologic reference"
                                    raise InvalidReference(message)
                            except (KeyError, InvalidReference):
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
            modify_priority(work_bond_dict, user_defined)
        output = pd.DataFrame.from_dict(construction_table, orient='index')
        output = output.fillna(0).astype('int64')
        output = output.loc[order_of_definition, ['b', 'a', 'd']]
        return output

    def check_dihedral(self, construction_table,
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
        bond_dict = self._give_val_sorted_bond_dict(use_lookup=use_lookup)
        c_table = construction_table.copy()
        angles = self.angle_degrees(c_table.iloc[3:, :].values)
        problem_index = np.nonzero((175 < angles) | (angles < 5))[0]
        rename = dict(enumerate(c_table.index[3:]))
        problem_index = [rename[i] for i in problem_index]

        for i in problem_index:
            loc_i = c_table.index.get_loc(i)
            b, a, problem_d = c_table.loc[i, ['b', 'a', 'd']]
            try:
                c_table.loc[i, 'd'] = (bond_dict[a] - {b, a, problem_d}
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
                        if 5 < angle < 175:
                            found = True
                            c_table.loc[i, 'd'] = new_d
                        else:
                            visited.add(new_d)
                            for j in tmp_bond_dict[new_d]:
                                new_tmp_bond_dict[j] = bond_dict[j] - visited
                    tmp_bond_dict = new_tmp_bond_dict
                if not found:
                    molecule = self.distance_to(origin=i, sort=True,
                                                indices_of_other_atoms=visited)
                    k = 0
                    while not found and k < len(molecule):
                        new_d = molecule.index[k]
                        angle = self.angle_degrees([b, a, new_d])[0]
                        if 5 < angle < 175:
                            found = True
                            c_table.loc[i, 'd'] = new_d
                        k = k + 1
                    if not found:
                        message = ('The atom with index {} has no possibility '
                                   'to get nonlinear reference atoms'.format)
                        raise UndefinedCoordinateSystem(message(i))
        return c_table

    def check_absolute_refs(self, construction_table):
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
        c_table = construction_table.copy()
        abs_refs = self._metadata['abs_refs']

        def is_valid_abs_ref(self, c_table, abs_references, row):
            A = np.empty((3, 3))
            for i in range(3):
                if i < row:
                    A[i] = self.loc[c_table.iloc[row, i], ['x', 'y', 'z']]
                else:
                    A[i] = abs_references[c_table.iloc[row, i]][0]
            v1, v2 = A[2] - A[1], A[1] - A[0]
            K = np.cross(v1, v2)
            zero = np.array([0, 0, 0])
            return not (np.isclose(K, zero).all()
                        or np.isclose(v1, zero).all()
                        or np.isclose(v2, zero).all())

        for i in range(min(3, len(c_table))):
            order_of_refs = iter(permutations(abs_refs.keys()))
            finished = False
            while not finished:
                if not is_valid_abs_ref(self, c_table, abs_refs, i):
                    c_table.iloc[i, i:] = next(order_of_refs)[i:3]
                else:
                    finished = True
        return c_table

    def get_construction_table(self, fragment_list=None,
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
        full_bond_dict = self._give_val_sorted_bond_dict(use_lookup=use_lookup)
        if fragment_list is None:
            fragments = sorted(self.fragmentate(), key=lambda x: -len(x))
            # During function execution the bonding situation does not change,
            # so the lookup may be used now.
            use_lookup = True
        else:
            fragments = fragment_list

        def prepend_missing_parts_of_molecule(fragment_list):
            for fragment in fragment_list:
                if pd.api.types.is_list_like(fragment):
                    try:
                        full_index |= fragment[0].index
                    except NameError:
                        full_index = fragment[0].index
                else:
                    try:
                        full_index |= fragment.index
                    except NameError:
                        full_index = fragment.index

            if not self.index.difference(full_index).empty:
                missing_part = self.without(self.loc[full_index],
                                            use_lookup=use_lookup)
                fragment_list = missing_part + fragment_list
            return fragment_list

        fragments = prepend_missing_parts_of_molecule(fragments)

        if pd.api.types.is_list_like(fragments[0]):
            fragment, references = fragments[0]
            full_table = fragment._get_constr_table(
                use_lookup=use_lookup, predefined_table=references)
        else:
            fragment = fragments[0]
            full_table = fragment._get_constr_table(use_lookup=use_lookup)

        for fragment in fragments[1:]:
            finished_part = self.loc[full_table.index]
            bond_dict = finished_part.restrict_bond_dict(full_bond_dict)
            if pd.api.types.is_list_like(fragment):
                fragment, references = fragment
                if len(references) < min(3, len(fragment)):
                    raise ValueError('If you specify references for a '
                                     'fragment, it has to consist of at least'
                                     'min(3, len(fragment)) rows.')
                constr_table = fragment._get_constr_table(
                    predefined_table=references, use_lookup=use_lookup)
            else:
                i, b = fragment._shortest_distance(finished_part)[:2]
                constr_table = fragment._get_constr_table(
                    start_atom=i, use_lookup=use_lookup)
                if len(full_table) == 1:
                    a, d = -3, -1
                elif len(full_table) == 2:
                    if b == full_table.index[0]:
                        a = full_table.index[1]
                    else:
                        a = full_table.index[0]
                    d = -1
                else:
                    if b in full_table.index[:2]:
                        if b == full_table.index[0]:
                            a = full_table.index[2]
                            d = full_table.index[1]
                        else:
                            a = full_table.loc[b, 'b']
                            d = full_table.index[2]
                    else:
                        a, d = full_table.loc[b, ['b', 'a']]

                if len(constr_table) >= 1:
                    constr_table.iloc[0, :] = b, a, d
                if len(constr_table) >= 2:
                    constr_table.iloc[1, [1, 2]] = b, a
                if len(constr_table) >= 3:
                    constr_table.iloc[2, 2] = b

            full_table = pd.concat([full_table, constr_table])
        return full_table

    def _calculate_values(self, construction_table):
        values = np.empty((len(self), 3), dtype='float64')

        def get_position(self, construction_table):
            coords = ['x', 'y', 'z']
            abs_references = self._metadata['abs_refs']

            values = np.empty((self.n_atoms, 3))
            pos = np.empty((self.n_atoms, 3, 4))

            pos[:, :, 0] = self.loc[construction_table.index, coords]

            pos[0, :, 1] = abs_references[construction_table.iloc[0, 0]][0]
            pos[1:, :, 1] = self.loc[construction_table.iloc[1:, 0], coords]

            pos[0, :, 2] = abs_references[construction_table.iloc[0, 1]][0]
            pos[1, :, 2] = abs_references[construction_table.iloc[1, 1]][0]
            pos[2:, :, 2] = self.loc[construction_table.iloc[2:, 1], coords]

            pos[0, :, 3] = abs_references[construction_table.iloc[0, 2]][0]
            pos[1, :, 3] = abs_references[construction_table.iloc[1, 2]][0]
            pos[2, :, 3] = abs_references[construction_table.iloc[2, 2]][0]
            pos[3:, :, 3] = self.loc[construction_table.iloc[3:, 2], coords]

            IB = pos[:, :, 1] - pos[:, :, 0]
            BA = pos[:, :, 2] - pos[:, :, 1]
            AD = pos[:, :, 3] - pos[:, :, 2]
            return IB, BA, AD

        def get_bond_length(IB):
            return np.linalg.norm(IB, axis=1)

        def get_angle(IB, BA):
            ba = BA / np.linalg.norm(BA, axis=1)[:, None]
            bi = -1 * IB / np.linalg.norm(IB, axis=1)[:, None]
            dot_product = np.sum(bi * ba, axis=1)
            dot_product[np.isclose(dot_product, 1)] = 1
            dot_product[np.isclose(dot_product, -1)] = -1
            return np.nan_to_num(np.degrees(np.arccos(dot_product)))

        def get_dihedral(IB, BA, AD):
            N1 = np.cross(IB, BA, axis=1)
            N2 = np.cross(BA, AD, axis=1)
            n1, n2 = [v / np.linalg.norm(v, axis=1)[:, None] for v in (N1, N2)]
            dot_product = np.sum(n1 * n2, axis=1)
            dot_product[np.isclose(dot_product, 1)] = 1
            dot_product[np.isclose(dot_product, -1)] = -1
            dihedrals = np.degrees(np.arccos(dot_product))
            # the next lines are to test the direction of rotation.
            # is a dihedral really 90 or 270 degrees?
            # Equivalent to direction of rotation of dihedral
            where_to_modify = np.sum(BA * np.cross(n1, n2, axis=1), axis=1) > 0
            where_to_modify = np.nonzero(where_to_modify)[0]
            sign = np.full_like(dihedrals, 1)
            to_add = np.full_like(dihedrals, 0)
            sign[where_to_modify] = -1
            to_add[where_to_modify] = 360
            return np.nan_to_num(to_add + sign * dihedrals)

        IB, BA, AD = get_position(self, construction_table)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            values[:, 0] = get_bond_length(IB)
            values[:, 1] = get_angle(IB, BA)
            values[:, 2] = get_dihedral(IB, BA, AD)
        return values

    def _build_zmat(self, construction_table):
        """Create the Zmatrix from a construction table.

        Args:
            Construction table (pd.DataFrame):

        Returns:
            Zmat: A new instance of :class:`Zmat`.
        """
        c_table = construction_table
        default_cols = ['atom', 'b', 'bond', 'a', 'angle', 'd', 'dihedral']
        optional_cols = list(set(self.columns) - {'atom', 'x', 'y', 'z'})

        zmat_frame = pd.DataFrame(columns=default_cols + optional_cols,
                                  dtype='float', index=c_table.index)

        zmat_frame.loc[:, optional_cols] = self.loc[c_table.index,
                                                    optional_cols]

        zmat_frame.loc[:, 'atom'] = self.loc[c_table.index, 'atom']
        zmat_frame.loc[:, ['b', 'a', 'd']] = c_table

        zmat_frame.loc[:, ['bond', 'angle', 'dihedral']] = \
            self._calculate_values(c_table)

        zmatrix = Zmat(zmat_frame)
        keys_to_keep = ['abs_refs']
        for key in keys_to_keep:
            zmatrix._metadata[key] = self._metadata[key].copy()

        return zmatrix

    def give_zmat(self, construction_table=None,
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
        self.get_bonds(use_lookup=use_lookup)
        # During function execution the connectivity situation won't change
        # So use_look=True will be used
        if construction_table is None:
            c_table = self.get_construction_table(use_lookup=True)
            c_table = self.check_dihedral(c_table, use_lookup=True)
            c_table = self.check_absolute_refs(c_table)
        else:
            c_table = construction_table
        return self._build_zmat(c_table)

    def to_zmat(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.give_zmat`
        """
        message = 'Will be removed in the future. Please use give_zmat.'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.give_zmat(*args, **kwargs)
