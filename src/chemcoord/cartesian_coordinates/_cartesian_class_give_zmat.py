# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import warnings
import chemcoord.constants as constants
from chemcoord._exceptions import \
    IllegalArgumentCombination, \
    UndefinedCoordinateSystem
from chemcoord.cartesian_coordinates._cartesian_class_core import \
    CartesianCore
from chemcoord.configuration import settings
from chemcoord.internal_coordinates.zmat_class_main import Zmat
from collections import OrderedDict
from itertools import permutations


class CartesianGiveZmat(CartesianCore):
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

    def _get_frag_constr_table(self, start_atom=None, predefined_table=None,
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
            start_atom: An index for the first atom may be provided.
            predefined_table (pd.DataFrame): An uncomplete construction table
                may be provided. The rest is created automatically.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            pd.DataFrame: Construction table
        """
        int_label = constants.int_label

        def modify_priority(bond_dict, user_defined):
            def move_to_start(dct, key):
                "Due to PY27 compatibility"
                keys = dct.keys()
                if key in keys and key != keys[0]:
                    root = dct._OrderedDict__root
                    first = root[1]
                    link = dct._OrderedDict__map[key]
                    link_prev, link_next, _ = link
                    link_prev[1] = link_next
                    link_next[0] = link_prev
                    link[0] = root
                    link[1] = first
                    root[1] = first[0] = link
                else:
                    raise KeyError

            for j in reversed(user_defined):
                try:
                    try:
                        bond_dict.move_to_end(j, last=False)
                    except AttributeError:
                        # No move_to_end method in python 2.x
                        move_to_start(bond_dict, j)
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
            order_of_def = [i]
            user_defined = []
            construction_table = {i: {'b': int_label['origin'],
                                      'a': int_label['e_z'],
                                      'd': int_label['e_x']}}
        else:
            i = construction_table.index[0]
            order_of_def = list(construction_table.index)
            user_defined = list(construction_table.index)
            construction_table = construction_table.to_dict(orient='index')

        visited = {i}
        if len(self) > 1:
            parent = {j: i for j in bond_dict[i]}
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
                    if b in order_of_def[:3]:
                        if len(order_of_def) == 1:
                            construction_table[i] = {'b': b,
                                                     'a': int_label['e_z'],
                                                     'd': int_label['e_x']}
                        elif len(order_of_def) == 2:
                            a = (bond_dict[b] & set(order_of_def))[0]
                            construction_table[i] = {'b': b, 'a': a,
                                                     'd': int_label['e_x']}
                        else:
                            try:
                                a = parent[b]
                            except KeyError:
                                a = (bond_dict[b] & set(order_of_def))[0]
                            try:
                                d = parent[a]
                                if d in set([b, a]):
                                    message = "Don't make self references"
                                    raise UndefinedCoordinateSystem(message)
                            except (KeyError, UndefinedCoordinateSystem):
                                try:
                                    d = ((bond_dict[a] & set(order_of_def))
                                         - set([b, a]))[0]
                                except IndexError:
                                    d = ((bond_dict[b] & set(order_of_def))
                                         - set([b, a]))[0]
                            construction_table[i] = {'b': b, 'a': a, 'd': d}
                    else:
                        a, d = [construction_table[b][k] for k in ['b', 'a']]
                        construction_table[i] = {'b': b, 'a': a, 'd': d}
                    order_of_def.append(i)

                visited.add(i)
                for j in work_bond_dict[i]:
                    new_work_bond_dict[j] = bond_dict[j] - visited
                    parent[j] = i

            work_bond_dict = new_work_bond_dict
            modify_priority(work_bond_dict, user_defined)
        output = pd.DataFrame.from_dict(construction_table, orient='index')
        output = output.fillna(0).astype('int64')
        output = output.loc[order_of_def, ['b', 'a', 'd']]
        return output

    def get_construction_table(self, fragment_list=None,
                               use_lookup=settings['defaults']['use_lookup'],
                               perform_checks=True):
        """Create a construction table for a Zmatrix.

        A construction table is basically a Zmatrix without the values
        for the bond lengths, angles and dihedrals.
        It contains the whole information about which reference atoms
        are used by each atom in the Zmatrix.

        This method creates a so called "chemical" construction table,
        which makes use of the connectivity table in this molecule.

        Args:
            fragment_list (sequence): There are four possibilities to specify
                the sequence of fragments:

                1. A list of tuples is given. Each tuple contains the fragment
                with its corresponding construction table in the form of::

                    [(frag1, c_table1), (frag2, c_table2)...]

                If the construction table of a fragment is not complete,
                the rest of each fragment's
                construction table is calculated automatically.

                2. It is possible to omit the construction tables for some
                or all fragments as in the following example::

                    [(frag1, c_table1), frag2, (frag3, c_table3)...]

                3. If ``self`` contains more atoms than the union over all
                fragments, the rest of the molecule without the fragments
                is automatically prepended using :meth:`~Cartesian.without`::

                    self.without(fragments) + fragment_list

                4. If fragment_list is ``None`` then fragmentation, etc.
                is done automatically. The fragments are then sorted by
                their number of atoms, in order to use the largest fragment
                as reference for the other ones.

            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.
            perform_checks (bool): The checks for invalid references are
                performed using :meth:`~chemcoord.Cartesian.correct_dihedral`
                and :meth:`~chemcoord.Cartesian.correct_absolute_refs`.

        Returns:
            :class:`pandas.DataFrame`: Construction table
        """
        int_label = constants.int_label
        if fragment_list is None:
            fragments = sorted(self.fragmentate(use_lookup=use_lookup),
                               key=lambda x: len(x), reverse=True)
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
            full_table = fragment._get_frag_constr_table(
                use_lookup=use_lookup, predefined_table=references)
        else:
            fragment = fragments[0]
            full_table = fragment._get_frag_constr_table(use_lookup=use_lookup)

        for fragment in fragments[1:]:
            finished_part = self.loc[full_table.index]
            if pd.api.types.is_list_like(fragment):
                fragment, references = fragment
                if len(references) < min(3, len(fragment)):
                    raise ValueError('If you specify references for a '
                                     'fragment, it has to consist of at least'
                                     'min(3, len(fragment)) rows.')
                constr_table = fragment._get_frag_constr_table(
                    predefined_table=references, use_lookup=use_lookup)
            else:
                i, b = fragment.shortest_distance(finished_part)[:2]
                constr_table = fragment._get_frag_constr_table(
                    start_atom=i, use_lookup=use_lookup)
                if len(full_table) == 1:
                    a, d = int_label['e_z'], int_label['e_x']
                elif len(full_table) == 2:
                    if b == full_table.index[0]:
                        a = full_table.index[1]
                    else:
                        a = full_table.index[0]
                    d = int_label['e_x']
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

        c_table = full_table
        if perform_checks:
            c_table = self.correct_dihedral(c_table)
            c_table = self.correct_dihedral(c_table, use_lookup=use_lookup)
            c_table = self.correct_absolute_refs(c_table)
        return c_table

    def check_dihedral(self, construction_table):
        """Checks, if the dihedral defining atom is colinear.

        Checks for each index starting from the third row of the
        ``construction_table``, if the reference atoms are colinear.

        Args:
            construction_table (pd.DataFrame):

        Returns:
            list: A list of problematic indices.
        """
        c_table = construction_table
        angles = self.angle_degrees(c_table.iloc[3:, :].values)
        problem_index = np.nonzero((175 < angles) | (angles < 5))[0]
        rename = dict(enumerate(c_table.index[3:]))
        problem_index = [rename[i] for i in problem_index]
        return problem_index

    def correct_dihedral(self, construction_table,
                         use_lookup=settings['defaults']['use_lookup']):
        """Reindexe the dihedral defining atom if linear reference is used.

        Uses :meth:`~Cartesian.check_dihedral` to obtain the problematic
        indices.

        Args:
            construction_table (pd.DataFrame):
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            pd.DataFrame: Appropiately renamed construction table.
        """

        problem_index = self.check_dihedral(construction_table)
        bond_dict = self._give_val_sorted_bond_dict(use_lookup=use_lookup)
        c_table = construction_table.copy()
        for i in problem_index:
            loc_i = c_table.index.get_loc(i)
            b, a, problem_d = c_table.loc[i, ['b', 'a', 'd']]
            try:
                c_table.loc[i, 'd'] = (bond_dict[a] - {b, a, problem_d}
                                       - set(c_table.index[loc_i:]))[0]
            except IndexError:
                # TODO(Use only already defined atoms as reference)
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
                    other_atoms = c_table.index[:loc_i].difference({b, a})
                    molecule = self.distance_to(origin=i, sort=True,
                                                other_atoms=other_atoms)
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

    def _has_valid_abs_ref(self, i, construction_table):
        """Checks, if ``i`` uses valid absolute references.

        Checks for each index from first to third row of the
        ``construction_table``, if the references are colinear.
        This case has to be specially treated, because the references
        are not only atoms (to fix internal degrees of freedom) but also points
        in cartesian space called absolute references.
        (to fix translational and rotational degrees of freedom)

        Args:
            i (label): The label has to be in the first three rows.
            construction_table (pd.DataFrame):

        Returns:
            bool:
        """
        c_table = construction_table
        abs_refs = constants.absolute_refs
        A = np.empty((3, 3))
        row = c_table.index.get_loc(i)
        if row > 2:
            message = 'The index {i} is not from the first three, rows'.format
            raise ValueError(message(i=i))
        for k in range(3):
            if k < row:
                A[k] = self.loc[c_table.iloc[row, k], ['x', 'y', 'z']]
            else:
                A[k] = abs_refs[c_table.iloc[row, k]]
        v1, v2 = A[2] - A[1], A[1] - A[0]
        K = np.cross(v1, v2)
        zero = np.full(3, 0.)
        return not (np.allclose(K, zero) or np.allclose(v1, zero)
                    or np.allclose(v2, zero))

    def check_absolute_refs(self, construction_table):
        """Checks first three rows of ``construction_table`` for linear references

        Checks for each index from first to third row of the
        ``construction_table``, if the references are colinear.
        This case has to be specially treated, because the references
        are not only atoms (to fix internal degrees of freedom) but also points
        in cartesian space called absolute references.
        (to fix translational and rotational degrees of freedom)

        Args:
            construction_table (pd.DataFrame):

        Returns:
            list: A list of problematic indices.
        """
        c_table = construction_table
        problem_index = [i for i in c_table.index[:3]
                         if not self._has_valid_abs_ref(i, c_table)]
        return problem_index

    def correct_absolute_refs(self, construction_table):
        """Reindexe construction_table if linear reference in first three rows
        present.

        Uses :meth:`~Cartesian.check_absolute_refs` to obtain the problematic
        indices.

        Args:
            construction_table (pd.DataFrame):

        Returns:
            pd.DataFrame: Appropiately renamed construction table.
        """
        c_table = construction_table.copy()
        abs_refs = constants.absolute_refs
        problem_index = self.check_absolute_refs(c_table)
        for i in problem_index:
            order_of_refs = iter(permutations(abs_refs.keys()))
            finished = False
            while not finished:
                if self._has_valid_abs_ref(i, c_table):
                    finished = True
                else:
                    row = c_table.index.get_loc(i)
                    c_table.iloc[row, row:] = next(order_of_refs)[row:3]
        return c_table

    def _get_bond_vectors(self, construction_table):
        c_table = construction_table
        pos = np.empty((len(c_table), 3, 4))

        if isinstance(c_table, pd.DataFrame):
            pos[:, :, 0] = self._get_positions(c_table.index)
            pos[:, :, 1] = self._get_positions(c_table['b'])
            pos[:, :, 2] = self._get_positions(c_table['a'])
            pos[:, :, 3] = self._get_positions(c_table['d'])
        else:
            c_table = np.array(c_table)
            if len(c_table.shape) == 1:
                c_table = c_table[None, :]
            for col in range(4):
                pos[:, :, col] = self._get_positions(c_table[:, col])

        IB = pos[:, :, 1] - pos[:, :, 0]
        BA = pos[:, :, 2] - pos[:, :, 1]
        AD = pos[:, :, 3] - pos[:, :, 2]
        return IB, BA, AD

    def _calculate_zmat_values(self, construction_table):
        IB, BA, AD = self._get_bond_vectors(construction_table)
        values = np.empty_like(IB, dtype='float64')

        def get_bond_length(IB):
            return np.linalg.norm(IB, axis=1)

        def get_angle(IB, BA):
            ba = BA / np.linalg.norm(BA, axis=1)[:, None]
            bi = -1 * IB / np.linalg.norm(IB, axis=1)[:, None]
            dot_product = np.sum(bi * ba, axis=1)
            dot_product[dot_product > 1] = 1
            dot_product[dot_product < -1] = -1
            return np.nan_to_num(np.degrees(np.arccos(dot_product)))

        def get_dihedral(IB, BA, AD):
            N1 = np.cross(IB, BA, axis=1)
            N2 = np.cross(BA, AD, axis=1)
            n1, n2 = [v / np.linalg.norm(v, axis=1)[:, None] for v in (N1, N2)]
            dot_product = np.sum(n1 * n2, axis=1)
            dot_product[dot_product > 1] = 1
            dot_product[dot_product < -1] = -1
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

        zmat_values = self._calculate_zmat_values(c_table)
        zmat_frame.loc[:, ['bond', 'angle', 'dihedral']] = zmat_values

        zmatrix = Zmat(zmat_frame, metadata=self.metadata)
        zmatrix._metadata['last_valid_cartesian'] = self.copy()
        return zmatrix

    def give_zmat(self, construction_table=None,
                  use_lookup=settings['defaults']['use_lookup']):
        """Transform to internal coordinates.

        Transforming to internal coordinates involves basically three
        steps:

        1. Define an order of how to build and define for each atom
        the used reference atoms.

        2. Check for problematic local linearity. In this algorithm an
        angle with ``170 < angle < 10`` is assumed to be linear.
        This is not the mathematical definition, but makes it safer
        against "floating point noise"

        3. Calculate the bond lengths, angles and dihedrals using the
        references defined in step 1 and 2.

        In the first two steps a so called ``construction_table`` is created.
        This is basically a Zmatrix without the values for the bonds, angles
        and dihedrals hence containing only the information about the used
        references. ChemCoord uses a :class:`pandas.DataFrame` with the columns
        ``['b', 'a', 'd']``. Look into
        :meth:`~chemcoord.Cartesian.get_construction_table` for more
        information.

        It is important to know, that calculating the construction table
        is a very costly step since the algoritym tries to make some guesses
        based on connectivity to create a "chemical" zmatrix.

        If you create several zmatrices based on the same references
        you can obtain the construction table of a zmatrix with
        ``Zmat_instance.loc[:, ['b', 'a', 'd']]``
        If you then pass the buildlist as argument to ``give_zmat``,
        the algorithm directly starts with step 3 (which is much faster).

        If a ``construction_table`` is passed into :meth:`~Cartesian.give_zmat`
        the check for pathological linearity is not performed!
        So if a ``construction_table`` is either manually created,
        or obtained from :meth:`~Cartesian.get_construction_table`
        under the option ``perform_checks = False``, it is recommended to use
        the following methods:

            * :meth:`~Cartesian.correct_dihedral`
            * :meth:`~Cartesian.correct_absolute_refs`

        If you want to check for problematic indices in order to solve the
        invalid references yourself, use the following methods:

            * :meth:`~Cartesian.check_dihedral`
            * :meth:`~Cartesian.check_absolute_refs`

        Args:
            construction_table (pandas.DataFrame):
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            Zmat: A new instance of :class:`~Zmat`.
        """
        self.get_bonds(use_lookup=use_lookup)
        use_lookup = True
        # During function execution the connectivity situation won't change
        # So use_look=True will be used
        if construction_table is None:
            c_table = self.get_construction_table(use_lookup=use_lookup)
            c_table = self.correct_dihedral(c_table, use_lookup=use_lookup)
            c_table = self.correct_absolute_refs(c_table)
        else:
            c_table = construction_table
        return self._build_zmat(c_table)

    def to_zmat(self, *args, **kwargs):
        """Deprecated, use :meth:`~Cartesian.give_zmat`
        """
        message = 'Will be removed in the future. Please use give_zmat.'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.give_zmat(*args, **kwargs)
