# -*- coding: utf-8 -*-
import collections
import copy
import itertools
from functools import partial
from itertools import product

import numba as nb
import numpy as np
import pandas as pd
from numba import jit
from sortedcontainers import SortedSet

import chemcoord.cartesian_coordinates.xyz_functions as xyz_functions
import chemcoord.constants as constants
from chemcoord._generic_classes.generic_core import GenericCore
from chemcoord.cartesian_coordinates._cartesian_class_pandas_wrapper import \
    PandasWrapper
from chemcoord.cartesian_coordinates.xyz_functions import dot
from chemcoord.configuration import settings
from chemcoord.exceptions import IllegalArgumentCombination, PhysicalMeaning
from six.moves import zip  # pylint:disable=redefined-builtin


class CartesianCore(PandasWrapper, GenericCore):

    _required_cols = frozenset({'atom', 'x', 'y', 'z'})

    # Look into the numpy manual for description of __array_priority__:
    # https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.classes.html
    __array_priority__ = 15.0

    # overwrites existing method
    def __init__(self, frame=None, atoms=None, coords=None, index=None,
                 metadata=None, _metadata=None):
        """How to initialize a Cartesian instance.

        Args:
            frame (pd.DataFrame): A Dataframe with at least the
                columns ``['atom', 'x', 'y', 'z']``.
                Where ``'atom'`` is a string for the elementsymbol.
            atoms (sequence): A list of strings. (Elementsymbols)
            coords (sequence): A ``n_atoms * 3`` array containg the positions
                of the atoms. Note that atoms and coords are mutually exclusive
                to frame. Besides atoms and coords have to be both either None
                or not None.

        Returns:
            Cartesian: A new cartesian instance.
        """
        if (bool(atoms is None and coords is None)
                == bool(atoms is not None and coords is not None)):
            message = 'atoms and coords have to be both None or not None'
            raise IllegalArgumentCombination(message)
        elif frame is None and atoms is None and coords is None:
            message = 'Either frame or atoms and coords have to be not None'
            raise IllegalArgumentCombination(message)
        elif atoms is not None and coords is not None:
            dtypes = [('atom', str), ('x', float), ('y', float), ('z', float)]
            frame = pd.DataFrame(np.empty(len(atoms), dtype=dtypes), index=index)
            frame['atom'] = atoms
            frame.loc[:, ['x', 'y', 'z']] = coords
        elif not isinstance(frame, pd.DataFrame):
            raise ValueError('Need a pd.DataFrame as input')
        if not self._required_cols <= set(frame.columns):
            raise PhysicalMeaning('There are columns missing for a '
                                  'meaningful description of a molecule')
        self._frame = frame.copy()
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata.copy()

        if _metadata is None:
            self._metadata = {}
        else:
            self._metadata = copy.deepcopy(_metadata)

    def _return_appropiate_type(self, selected):
        if isinstance(selected, pd.Series):
            frame = pd.DataFrame(selected).T
            if self._required_cols <= set(frame.columns):
                selected = frame.apply(pd.to_numeric, errors='ignore')
            else:
                return selected

        if (isinstance(selected, pd.DataFrame)
                and self._required_cols <= set(selected.columns)):
            molecule = self.__class__(selected)
            molecule.metadata = self.metadata.copy()
            molecule._metadata = copy.deepcopy(self._metadata)
            return molecule
        else:
            return selected

    def _test_if_can_be_added(self, other):
        if not (set(self.index) == set(other.index)
                and (self['atom'] == other.loc[self.index, 'atom']).all(axis=None)):
            message = ("You can add only Cartesians which are indexed in the "
                       "same way and use the same atoms.")
            raise PhysicalMeaning(message)

    def __add__(self, other):
        coords = ['x', 'y', 'z']
        new = self.copy()
        if isinstance(other, CartesianCore):
            self._test_if_can_be_added(other)
            new.loc[:, coords] = self.loc[:, coords] + other.loc[:, coords]
        elif isinstance(other, pd.DataFrame):
            new.loc[:, coords] = self.loc[:, coords] + other.loc[:, coords]
        else:
            try:
                other = np.array(other, dtype='f8')
            except TypeError:
                pass
            new.loc[:, coords] = self.loc[:, coords] + other
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        coords = ['x', 'y', 'z']
        new = self.copy()
        if isinstance(other, CartesianCore):
            self._test_if_can_be_added(other)
            new.loc[:, coords] = self.loc[:, coords] - other.loc[:, coords]
        elif isinstance(other, pd.DataFrame):
            new.loc[:, coords] = self.loc[:, coords] - other.loc[:, coords]
        else:
            try:
                other = np.array(other, dtype='f8')
            except TypeError:
                pass
            new.loc[:, coords] = self.loc[:, coords] - other
        return new

    def __rsub__(self, other):
        coords = ['x', 'y', 'z']
        new = self.copy()
        if isinstance(other, CartesianCore):
            self._test_if_can_be_added(other)
            new.loc[:, coords] = other.loc[:, coords] - self.loc[:, coords]
        elif isinstance(other, pd.DataFrame):
            new.loc[:, coords] = other.loc[:, coords] - self.loc[:, coords]
        else:
            try:
                other = np.array(other, dtype='f8')
            except TypeError:
                pass
            new.loc[:, coords] = other - self.loc[:, coords]
        return new

    def __mul__(self, other):
        coords = ['x', 'y', 'z']
        new = self.copy()
        if isinstance(other, CartesianCore):
            self._test_if_can_be_added(other)
            new.loc[:, coords] = self.loc[:, coords] * other.loc[:, coords]
        elif isinstance(other, pd.DataFrame):
            new.loc[:, coords] = self.loc[:, coords] * other.loc[:, coords]
        else:
            try:
                other = np.array(other, dtype='f8')
            except TypeError:
                pass
            new.loc[:, coords] = self.loc[:, coords] * other
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        coords = ['x', 'y', 'z']
        new = self.copy()
        if isinstance(other, CartesianCore):
            self._test_if_can_be_added(other)
            new.loc[:, coords] = self.loc[:, coords] / other.loc[:, coords]
        elif isinstance(other, pd.DataFrame):
            new.loc[:, coords] = self.loc[:, coords] / other.loc[:, coords]
        else:
            try:
                other = np.array(other, dtype='f8')
            except TypeError:
                pass
            new.loc[:, coords] = self.loc[:, coords] / other
        return new

    def __rtruediv__(self, other):
        coords = ['x', 'y', 'z']
        new = self.copy()
        if isinstance(other, CartesianCore):
            self._test_if_can_be_added(other)
            new.loc[:, coords] = other.loc[:, coords] / self.loc[:, coords]
        elif isinstance(other, pd.DataFrame):
            new.loc[:, coords] = other.loc[:, coords] / self.loc[:, coords]
        else:
            try:
                other = np.array(other, dtype='f8')
            except TypeError:
                pass
            new.loc[:, coords] = other / self.loc[:, coords]
        return new

    def __pow__(self, other):
        coords = ['x', 'y', 'z']
        new = self.copy()
        new.loc[:, coords] = self.loc[:, coords]**other
        return new

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        return -1 * self.copy()

    def __abs__(self):
        coords = ['x', 'y', 'z']
        new = self.copy()
        new.loc[:, coords] = abs(new.loc[:, coords])
        return new

    def __matmul__(self, other):
        return NotImplemented

    def __rmatmul__(self, other):
        coords = ['x', 'y', 'z']
        new = self.copy()
        new.loc[:, coords] = (np.dot(other, new.loc[:, coords].T)).T
        return new

    def __eq__(self, other):
        return self._frame == other._frame

    def __ne__(self, other):
        return self._frame != other._frame

    def copy(self):
        molecule = self.__class__(self._frame)
        molecule.metadata = self.metadata.copy()
        molecule._metadata = copy.deepcopy(self._metadata)
        return molecule

    def subs(self, *args):
        """Substitute a symbolic expression in ``['x', 'y', 'z']``

        This is a wrapper around the substitution mechanism of
        `sympy <http://docs.sympy.org/latest/tutorial/basic_operations.html>`_.
        Any symbolic expression in the columns
        ``['x', 'y', 'z']`` of ``self`` will be substituted
        with value.

        Args:
            symb_expr (sympy expression):
            value :
            perform_checks (bool): If ``perform_checks is True``,
                it is asserted, that the resulting Zmatrix can be converted
                to cartesian coordinates.
                Dummy atoms will be inserted automatically if necessary.

        Returns:
            Cartesian: Cartesian with substituted symbolic expressions.
            If all resulting sympy expressions in a column are numbers,
            the column is recasted to 64bit float.
        """
        cols = ['x', 'y', 'z']
        out = self.copy()

        def get_subs_f(*args):
            def subs_function(x):
                if hasattr(x, 'subs'):
                    x = x.subs(*args)
                    try:
                        x = float(x)
                    except TypeError:
                        pass
                return x
            return subs_function

        for col in cols:
            if out.loc[:, col].dtype is np.dtype('O'):
                out.loc[:, col] = out.loc[:, col].map(get_subs_f(*args))
                try:
                    out._frame = out._frame.astype({col: 'f8'})
                except (SystemError, TypeError):
                    pass
        return out

    @staticmethod
    @jit(nopython=True, cache=True)
    def _jit_give_bond_array(pos, bond_radii, self_bonding_allowed=False):
        """Calculate a boolean array where ``A[i,j] is True`` indicates a
        bond between the i-th and j-th atom.
        """
        n = pos.shape[0]
        bond_array = np.empty((n, n), dtype=nb.boolean)

        for i in range(n):
            for j in range(i, n):
                D = 0
                for h in range(3):
                    D += (pos[i, h] - pos[j, h])**2
                B = (bond_radii[i] + bond_radii[j])**2
                bond_array[i, j] = (B - D) >= 0
                bond_array[j, i] = bond_array[i, j]
        if not self_bonding_allowed:
            for i in range(n):
                bond_array[i, i] = False
        return bond_array

    def _update_bond_dict(self, fragment_indices,
                          positions,
                          bond_radii,
                          bond_dict=None,
                          self_bonding_allowed=False,
                          convert_index=None):
        """If bond_dict is provided, this function is not side effect free
        bond_dict has to be a collections.defaultdict(set)
        """
        assert (isinstance(bond_dict, collections.defaultdict)
                or bond_dict is None)
        fragment_indices = list(fragment_indices)
        if convert_index is None:
            convert_index = dict(enumerate(fragment_indices))
        if bond_dict is None:
            bond_dict = collections.defaultdict(set)

        frag_pos = positions[fragment_indices, :]
        frag_bond_radii = bond_radii[fragment_indices]

        bond_array = self._jit_give_bond_array(
            frag_pos, frag_bond_radii,
            self_bonding_allowed=self_bonding_allowed)
        a, b = bond_array.nonzero()
        a, b = [convert_index[i] for i in a], [convert_index[i] for i in b]
        for row, index in enumerate(a):
            # bond_dict is a collections.defaultdict(set)
            bond_dict[index].add(b[row])
        return bond_dict

    def _divide_et_impera(self, n_atoms_per_set=500, offset=3):
        coords = ['x', 'y', 'z']
        sorted_series = dict(zip(
            coords, [self[axis].sort_values() for axis in coords]))

        def ceil(x):
            return int(np.ceil(x))

        n_sets = len(self) / n_atoms_per_set
        n_sets_along_axis = ceil(n_sets**(1 / 3))
        n_atoms_per_set_along_axis = ceil(len(self) / n_sets_along_axis)

        def give_index(series, i, n_atoms_per_set_along_axis, offset=offset):
            N = n_atoms_per_set_along_axis
            try:
                min_value, max_value = series.iloc[[i * N, (i + 1) * N]]
            except IndexError:
                min_value, max_value = series.iloc[[i * N, -1]]
            selection = series.between(min_value - offset, max_value + offset)
            return set(series[selection].index)

        indices_at_axis = {axis: {} for axis in coords}
        for axis, i in product(coords, range(n_sets_along_axis)):
            indices_at_axis[axis][i] = give_index(sorted_series[axis], i,
                                                  n_atoms_per_set_along_axis)

        array_of_fragments = np.full([n_sets_along_axis] * 3, None, dtype='O')
        for i, j, k in product(*[range(x) for x in array_of_fragments.shape]):
            selection = (indices_at_axis['x'][i]
                         & indices_at_axis['y'][j]
                         & indices_at_axis['z'][k])
            array_of_fragments[i, j, k] = selection
        return array_of_fragments

    def get_bonds(self,
                  self_bonding_allowed=False,
                  offset=3,
                  modified_properties=None,
                  use_lookup=False,
                  set_lookup=True,
                  atomic_radius_data=None
                  ):
        """Return a dictionary representing the bonds.

        .. warning:: This function is **not sideeffect free**, since it
            assigns the output to a variable ``self._metadata['bond_dict']`` if
            ``set_lookup`` is ``True`` (which is the default). This is
            necessary for performance reasons.

        ``.get_bonds()`` will use or not use a lookup
        depending on ``use_lookup``. Greatly increases performance if
        True, but could introduce bugs in certain situations.

        Just imagine a situation where the :class:`~Cartesian` is
        changed manually. If you apply lateron a method e.g.
        :meth:`~get_zmat()` that makes use of :meth:`~get_bonds()`
        the dictionary of the bonds
        may not represent the actual situation anymore.

        You have two possibilities to cope with this problem.
        Either you just re-execute ``get_bonds`` on your specific instance,
        or you change the ``internally_use_lookup`` option in the settings.
        Please note that the internal use of the lookup variable
        greatly improves performance.

        Args:
            modified_properties (dic): If you want to change the van der
                Vaals radius of one or more specific atoms, pass a
                dictionary that looks like::

                    modified_properties = {index1: 1.5}

                For global changes use the constants module.
            offset (float):
            use_lookup (bool):
            set_lookup (bool):
            self_bonding_allowed (bool):
            atomic_radius_data (str): Defines which column of
                :attr:`constants.elements` is used. The default is
                ``atomic_radius_cc`` and can be changed with
                :attr:`settings['defaults']['atomic_radius_data']`.
                Compare with :func:`add_data`.

        Returns:
            dict: Dictionary mapping from an atom index to the set of
            indices of atoms bonded to.
        """
        if atomic_radius_data is None:
            atomic_radius_data = settings['defaults']['atomic_radius_data']

        def complete_calculation():
            old_index = self.index
            self.index = range(len(self))
            fragments = self._divide_et_impera(offset=offset)
            positions = np.array(self.loc[:, ['x', 'y', 'z']], order='F')
            data = self.add_data([atomic_radius_data, 'valency'])
            bond_radii = data[atomic_radius_data]
            if modified_properties is not None:
                bond_radii.update(pd.Series(modified_properties))
            bond_radii = bond_radii.values
            bond_dict = collections.defaultdict(set)
            for i, j, k in product(*[range(x) for x in fragments.shape]):
                # The following call is not side effect free and changes
                # bond_dict
                self._update_bond_dict(
                    fragments[i, j, k], positions, bond_radii,
                    bond_dict=bond_dict,
                    self_bonding_allowed=self_bonding_allowed)

            for i in set(self.index) - set(bond_dict.keys()):
                bond_dict[i] = {}

            self.index = old_index
            rename = dict(enumerate(self.index))
            bond_dict = {rename[key]: {rename[i] for i in bond_dict[key]}
                         for key in bond_dict}
            return bond_dict

        if use_lookup:
            try:
                bond_dict = self._metadata['bond_dict']
            except KeyError:
                bond_dict = complete_calculation()
        else:
            bond_dict = complete_calculation()

        if set_lookup:
            self._metadata['bond_dict'] = bond_dict
        return bond_dict

    def _give_val_sorted_bond_dict(self, use_lookup):
        def complete_calculation():
            bond_dict = self.get_bonds(use_lookup=use_lookup)
            valency = dict(zip(self.index,
                               self.add_data('valency')['valency']))
            val_bond_dict = {key:
                             SortedSet([i for i in bond_dict[key]],
                                       key=lambda x: -valency[x])
                             for key in bond_dict}
            return val_bond_dict
        if use_lookup:
            try:
                val_bond_dict = self._metadata['val_bond_dict']
            except KeyError:
                val_bond_dict = complete_calculation()
        else:
            val_bond_dict = complete_calculation()
        self._metadata['val_bond_dict'] = val_bond_dict
        return val_bond_dict

    def get_coordination_sphere(
            self, index_of_atom, n_sphere=1, give_only_index=False,
            only_surface=True, exclude=None,
            use_lookup=None):
        """Return a Cartesian of atoms in the n-th coordination sphere.

        Connected means that a path along covalent bonds exists.

        Args:
            index_of_atom (int):
            give_only_index (bool): If ``True`` a set of indices is
                returned. Otherwise a new Cartesian instance.
            n_sphere (int): Determines the number of the coordination sphere.
            only_surface (bool): Return only the surface of the coordination
                sphere.
            exclude (set): A set of indices that should be ignored
                for the path finding.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`. The default is
                specified in ``settings['defaults']['use_lookup']``

        Returns:
            A set of indices or a new Cartesian instance.
        """
        if use_lookup is None:
            use_lookup = settings['defaults']['use_lookup']
        exclude = set() if exclude is None else exclude
        bond_dict = self.get_bonds(use_lookup=use_lookup)
        i = index_of_atom
        if n_sphere != 0:
            visited = set([i]) | exclude
            try:
                tmp_bond_dict = {j: (bond_dict[j] - visited)
                                 for j in bond_dict[i]}
            except KeyError:
                tmp_bond_dict = {}
            n = 0
            while tmp_bond_dict and (n + 1) < n_sphere:
                new_tmp_bond_dict = {}
                for i in tmp_bond_dict:
                    if i in visited:
                        continue
                    visited.add(i)
                    for j in tmp_bond_dict[i]:
                        new_tmp_bond_dict[j] = bond_dict[j] - visited
                tmp_bond_dict = new_tmp_bond_dict
                n += 1
            if only_surface:
                index_out = set(tmp_bond_dict.keys())
            else:
                index_out = visited | set(tmp_bond_dict.keys())
        else:
            index_out = {i}

        if give_only_index:
            return index_out - exclude
        else:
            return self.loc[index_out - exclude]

    def _preserve_bonds(self, sliced_cartesian,
                        use_lookup=None):
        """Is called after cutting geometric shapes.

        If you want to change the rules how bonds are preserved, when
            applying e.g. :meth:`Cartesian.cut_sphere` this is the
            function you have to modify.
        It is recommended to inherit from the Cartesian class to
            tailor it for your project, instead of modifying the
            source code of ChemCoord.

        Args:
            sliced_frame (Cartesian):
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`. The default is
                specified in ``settings['defaults']['use_lookup']``

        Returns:
            Cartesian:
        """
        if use_lookup is None:
            use_lookup = settings['defaults']['use_lookup']

        included_atoms_set = set(sliced_cartesian.index)
        assert included_atoms_set.issubset(set(self.index)), \
            'The sliced Cartesian has to be a subset of the bigger frame'
        bond_dic = self.get_bonds(use_lookup=use_lookup)
        new_atoms = set([])
        for atom in included_atoms_set:
            new_atoms = new_atoms | bond_dic[atom]
        new_atoms = new_atoms - included_atoms_set
        while not new_atoms == set([]):
            index_of_interest = new_atoms.pop()
            included_atoms_set = (
                included_atoms_set |
                self.get_coordination_sphere(
                    index_of_interest,
                    n_sphere=float('inf'),
                    only_surface=False,
                    exclude=included_atoms_set,
                    give_only_index=True,
                    use_lookup=use_lookup))
            new_atoms = new_atoms - included_atoms_set
        molecule = self.loc[included_atoms_set, :]
        return molecule

    def cut_sphere(
            self,
            radius=15.,
            origin=None,
            outside_sliced=True,
            preserve_bonds=False):
        """Cut a sphere specified by origin and radius.

        Args:
            radius (float):
            origin (list): Please note that you can also pass an
                integer. In this case it is interpreted as the
                index of the atom which is taken as origin.
            outside_sliced (bool): Atoms outside/inside the sphere
                are cut out.
            preserve_bonds (bool): Do not cut covalent bonds.

        Returns:
            Cartesian:
        """
        if origin is None:
            origin = np.zeros(3)
        elif pd.api.types.is_list_like(origin):
            origin = np.array(origin, dtype='f8')
        else:
            origin = self.loc[origin, ['x', 'y', 'z']]

        molecule = self.get_distance_to(origin)
        if outside_sliced:
            molecule = molecule[molecule['distance'] < radius]
        else:
            molecule = molecule[molecule['distance'] > radius]

        if preserve_bonds:
            molecule = self._preserve_bonds(molecule)

        return molecule

    def cut_cuboid(
            self,
            a=20,
            b=None,
            c=None,
            origin=None,
            outside_sliced=True,
            preserve_bonds=False):
        """Cut a cuboid specified by edge and radius.

        Args:
            a (float): Value of the a edge.
            b (float): Value of the b edge. Takes value of a if None.
            c (float): Value of the c edge. Takes value of a if None.
            origin (list): Please note that you can also pass an
                integer. In this case it is interpreted as the index
                of the atom which is taken as origin.
            outside_sliced (bool): Atoms outside/inside the sphere are
                cut away.
            preserve_bonds (bool): Do not cut covalent bonds.

        Returns:
            Cartesian:
        """
        if origin is None:
            origin = np.zeros(3)
        elif pd.api.types.is_list_like(origin):
            origin = np.array(origin, dtype='f8')
        else:
            origin = self.loc[origin, ['x', 'y', 'z']]
        b = a if b is None else b
        c = a if c is None else c

        sides = np.array([a, b, c])
        pos = self.loc[:, ['x', 'y', 'z']]
        if outside_sliced:
            molecule = self[((pos - origin) / (sides / 2)).max(axis=1) < 1.]
        else:
            molecule = self[((pos - origin) / (sides / 2)).max(axis=1) > 1.]

        if preserve_bonds:
            molecule = self._preserve_bonds(molecule)
        return molecule

    def get_centroid(self):
        """Return the average location.

        Args:
            None

        Returns:
            :class:`numpy.ndarray`:
        """
        return np.mean(self.loc[:, ['x', 'y', 'z']], axis=0)

    def get_barycenter(self):
        """Return the mass weighted average location.

        Args:
            None

        Returns:
            :class:`numpy.ndarray`:
        """
        try:
            mass = self['mass'].values
        except KeyError:
            mass = self.add_data('mass')['mass'].values
        pos = self.loc[:, ['x', 'y', 'z']].values
        return (pos * mass[:, None]).sum(axis=0) / self.get_total_mass()

    def get_bond_lengths(self, indices):
        """Return the distances between given atoms.

        Calculates the distance between the atoms with
        indices ``i`` and ``b``.
        The indices can be given in three ways:

        * As simple list ``[i, b]``
        * As list of lists: ``[[i1, b1], [i2, b2]...]``
        * As :class:`pd.DataFrame` where ``i`` is taken from the index and
          ``b`` from the respective column ``'b'``.

        Args:
            indices (list):

        Returns:
            :class:`numpy.ndarray`: Vector of angles in degrees.
        """
        coords = ['x', 'y', 'z']
        if isinstance(indices, pd.DataFrame):
            i_pos = self.loc[indices.index, coords].values
            b_pos = self.loc[indices.loc[:, 'b'], coords].values
        else:
            indices = np.array(indices)
            if len(indices.shape) == 1:
                indices = indices[None, :]
            i_pos = self.loc[indices[:, 0], coords].values
            b_pos = self.loc[indices[:, 1], coords].values
        return np.linalg.norm(i_pos - b_pos, axis=1)

    def get_angle_degrees(self, indices):
        """Return the angles between given atoms.

        Calculates the angle in degrees between the atoms with
        indices ``i, b, a``.
        The indices can be given in three ways:

        * As simple list ``[i, b, a]``
        * As list of lists: ``[[i1, b1, a1], [i2, b2, a2]...]``
        * As :class:`pd.DataFrame` where ``i`` is taken from the index and
          ``b`` and ``a`` from the respective columns ``'b'`` and ``'a'``.

        Args:
            indices (list):

        Returns:
            :class:`numpy.ndarray`: Vector of angles in degrees.
        """
        coords = ['x', 'y', 'z']
        if isinstance(indices, pd.DataFrame):
            i_pos = self.loc[indices.index, coords].values
            b_pos = self.loc[indices.loc[:, 'b'], coords].values
            a_pos = self.loc[indices.loc[:, 'a'], coords].values
        else:
            indices = np.array(indices)
            if len(indices.shape) == 1:
                indices = indices[None, :]
            i_pos = self.loc[indices[:, 0], coords].values
            b_pos = self.loc[indices[:, 1], coords].values
            a_pos = self.loc[indices[:, 2], coords].values

        BI, BA = i_pos - b_pos, a_pos - b_pos
        bi, ba = [v / np.linalg.norm(v, axis=1)[:, None] for v in (BI, BA)]
        dot_product = np.sum(bi * ba, axis=1)
        dot_product[dot_product > 1] = 1
        dot_product[dot_product < -1] = -1
        angles = np.degrees(np.arccos(dot_product))
        return angles

    def get_dihedral_degrees(self, indices, start_row=0):
        """Return the dihedrals between given atoms.

        Calculates the dihedral angle in degrees between the atoms with
        indices ``i, b, a, d``.
        The indices can be given in three ways:

        * As simple list ``[i, b, a, d]``
        * As list of lists: ``[[i1, b1, a1, d1], [i2, b2, a2, d2]...]``
        * As :class:`pandas.DataFrame` where ``i`` is taken from the index and
          ``b``, ``a`` and ``d``from the respective columns
          ``'b'``, ``'a'`` and ``'d'``.

        Args:
            indices (list):

        Returns:
            :class:`numpy.ndarray`: Vector of angles in degrees.
        """
        coords = ['x', 'y', 'z']
        if isinstance(indices, pd.DataFrame):
            i_pos = self.loc[indices.index, coords].values
            b_pos = self.loc[indices.loc[:, 'b'], coords].values
            a_pos = self.loc[indices.loc[:, 'a'], coords].values
            d_pos = self.loc[indices.loc[:, 'd'], coords].values
        else:
            indices = np.array(indices)
            if len(indices.shape) == 1:
                indices = indices[None, :]
            i_pos = self.loc[indices[:, 0], coords].values
            b_pos = self.loc[indices[:, 1], coords].values
            a_pos = self.loc[indices[:, 2], coords].values
            d_pos = self.loc[indices[:, 3], coords].values

        IB = b_pos - i_pos
        BA = a_pos - b_pos
        AD = d_pos - a_pos

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

        length = indices.shape[0] - start_row
        sign = np.full(length, 1, dtype='float64')
        to_add = np.full(length, 0, dtype='float64')
        sign[where_to_modify] = -1
        to_add[where_to_modify] = 360
        dihedrals = to_add + sign * dihedrals
        return dihedrals

    def fragmentate(self, give_only_index=False,
                    use_lookup=None):
        """Get the indices of non bonded parts in the molecule.

        Args:
            give_only_index (bool): If ``True`` a set of indices is returned.
                Otherwise a new Cartesian instance.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`. The default is
                specified in ``settings['defaults']['use_lookup']``

        Returns:
            list: A list of sets of indices or new Cartesian instances.
        """
        if use_lookup is None:
            use_lookup = settings['defaults']['use_lookup']

        fragments = []
        pending = set(self.index)
        self.get_bonds(use_lookup=use_lookup)

        while pending:
            index = self.get_coordination_sphere(
                pending.pop(), use_lookup=True, n_sphere=float('inf'),
                only_surface=False, give_only_index=True)
            pending = pending - index
            if give_only_index:
                fragments.append(index)
            else:
                fragment = self.loc[index]
                fragment._metadata['bond_dict'] = fragment.restrict_bond_dict(
                    self._metadata['bond_dict'])
                try:
                    fragment._metadata['val_bond_dict'] = (
                        fragment.restrict_bond_dict(
                            self._metadata['val_bond_dict']))
                except KeyError:
                    pass
                fragments.append(fragment)
        return fragments

    def restrict_bond_dict(self, bond_dict):
        """Restrict a bond dictionary to self.

        Args:
            bond_dict (dict): Look into :meth:`~chemcoord.Cartesian.get_bonds`,
                to see examples for a bond_dict.

        Returns:
            bond dictionary
        """
        return {j: bond_dict[j] & set(self.index) for j in self.index}

    def get_fragment(self, list_of_indextuples, give_only_index=False,
                     use_lookup=None):
        """Get the indices of the atoms in a fragment.

        The list_of_indextuples contains all bondings from the
        molecule to the fragment. ``[(1,3), (2,4)]`` means for example that the
        fragment is connected over two bonds. The first bond is from atom 1 in
        the molecule to atom 3 in the fragment. The second bond is from atom
        2 in the molecule to atom 4 in the fragment.

        Args:
            list_of_indextuples (list):
            give_only_index (bool): If ``True`` a set of indices
                is returned. Otherwise a new Cartesian instance.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`. The default is
                specified in ``settings['defaults']['use_lookup']``

        Returns:
            A set of indices or a new Cartesian instance.
        """
        if use_lookup is None:
            use_lookup = settings['defaults']['use_lookup']

        exclude = [tuple[0] for tuple in list_of_indextuples]
        index_of_atom = list_of_indextuples[0][1]
        fragment_index = self.get_coordination_sphere(
            index_of_atom, exclude=set(exclude), n_sphere=float('inf'),
            only_surface=False, give_only_index=True, use_lookup=use_lookup)
        if give_only_index:
            return fragment_index
        else:
            return self.loc[fragment_index, :]

    def get_without(self, fragments,
                    use_lookup=None):
        """Return self without the specified fragments.

        Args:
            fragments: Either a list of :class:`~chemcoord.Cartesian` or a
                :class:`~chemcoord.Cartesian`.
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`. The default is
                specified in ``settings['defaults']['use_lookup']``

        Returns:
            list: List containing :class:`~chemcoord.Cartesian`.
        """
        if use_lookup is None:
            use_lookup = settings['defaults']['use_lookup']

        if pd.api.types.is_list_like(fragments):
            for fragment in fragments:
                try:
                    index_of_all_fragments |= fragment.index
                except NameError:
                    index_of_all_fragments = fragment.index
        else:
            index_of_all_fragments = fragments.index
        missing_part = self.loc[self.index.difference(index_of_all_fragments)]
        missing_part = missing_part.fragmentate(use_lookup=use_lookup)
        return sorted(missing_part, key=len, reverse=True)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _jit_pairwise_distances(pos1, pos2):
        """Optimized function for calculating the distance between each pair
        of points in positions1 and positions2.

        Does use python mode as fallback, if a scalar and not an array is
        given.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        D = np.empty((n1, n2))

        for i in range(n1):
            for j in range(n2):
                D[i, j] = np.sqrt(((pos1[i] - pos2[j])**2).sum())
        return D

    def get_shortest_distance(self, other):
        """Calculate the shortest distance between self and other

        Args:
            Cartesian: other

        Returns:
            tuple: Returns a tuple ``i, j, d`` with the following meaning:

            ``i``:
            The index on self that minimises the pairwise distance.

            ``j``:
            The index on other that minimises the pairwise distance.

            ``d``:
            The distance between self and other. (float)
        """
        coords = ['x', 'y', 'z']
        pos1 = self.loc[:, coords].values
        pos2 = other.loc[:, coords].values
        D = self._jit_pairwise_distances(pos1, pos2)
        i, j = np.unravel_index(D.argmin(), D.shape)
        d = D[i, j]
        i, j = dict(enumerate(self.index))[i], dict(enumerate(other.index))[j]
        return i, j, d

    def get_inertia(self):
        """Calculate the inertia tensor and transforms along
        rotation axes.

        This function calculates the inertia tensor and returns
        a 4-tuple.

        The unit is ``amu * length-unit-of-xyz-file**2``

        Args:
            None

        Returns:
            dict: The returned dictionary has four possible keys:

            ``transformed_Cartesian``:
            A :class:`~chemcoord.Cartesian`
            that is transformed to the basis spanned by
            the eigenvectors of the inertia tensor. The x-axis
            is the axis with the lowest inertia moment, the
            z-axis the one with the highest. Contains also a
            column for the mass

            ``diag_inertia_tensor``:
            A vector containing the ascendingly sorted inertia moments after
            diagonalization.

            ``inertia_tensor``:
            The inertia tensor in the old basis.

            ``eigenvectors``:
            The eigenvectors of the inertia tensor in the old basis.
            Since the inertia_tensor is hermitian, they are orthogonal and
            are returned as an orthonormal righthanded basis.
            The i-th eigenvector corresponds to the i-th eigenvalue in
            ``diag_inertia_tensor``.
        """
        def calculate_inertia_tensor(molecule):
            masses = molecule.loc[:, 'mass'].values
            pos = molecule.loc[:, ['x', 'y', 'z']].values
            inertia = np.sum(
                masses[:, None, None]
                * ((pos**2).sum(axis=1)[:, None, None]
                   * np.identity(3)[None, :, :]
                   - pos[:, :, None] * pos[:, None, :]),
                axis=0)
            diag_inertia, eig_v = np.linalg.eig(inertia)
            sorted_index = np.argsort(diag_inertia)
            diag_inertia = diag_inertia[sorted_index]
            eig_v = eig_v[:, sorted_index]
            return inertia, eig_v, diag_inertia

        molecule = self.add_data('mass')
        molecule = molecule - molecule.get_barycenter()
        inertia, eig_v, diag_inertia = calculate_inertia_tensor(molecule)
        eig_v = xyz_functions.orthonormalize_righthanded(eig_v)
        molecule = molecule.basistransform(eig_v)
        return {'transformed_Cartesian': molecule, 'eigenvectors': eig_v,
                'diag_inertia_tensor': diag_inertia, 'inertia_tensor': inertia}

    def basistransform(self, new_basis, old_basis=None,
                       orthonormalize=True):
        """Transform the frame to a new basis.

        This function transforms the cartesian coordinates from an
        old basis to a new one. Please note that old_basis and
        new_basis are supposed to have full Rank and consist of
        three linear independent vectors. If rotate_only is True,
        it is asserted, that both bases are orthonormal and right
        handed. Besides all involved matrices are transposed
        instead of inverted.
        In some applications this may require the function
        :func:`xyz_functions.orthonormalize` as a previous step.

        Args:
            old_basis (np.array):
            new_basis (np.array):
            rotate_only (bool):

        Returns:
            Cartesian: The transformed molecule.
        """
        if old_basis is None:
            old_basis = np.identity(3)

        is_rotation_matrix = np.isclose(np.linalg.det(new_basis), 1)
        if not is_rotation_matrix and orthonormalize:
            new_basis = xyz_functions.orthonormalize_righthanded(new_basis)
            is_rotation_matrix = True

        if is_rotation_matrix:
            return dot(np.dot(new_basis.T, old_basis), self)
        else:
            return dot(np.dot(np.linalg.inv(new_basis), old_basis), self)

    def _get_positions(self, indices):
        old_index = self.index
        self.index = range(len(self))
        rename = {j: i for i, j in enumerate(old_index)}

        pos = self.loc[:, ['x', 'y', 'z']].values.astype('f8')
        out = np.empty((len(indices), 3))
        indices = np.array([rename.get(i, i) for i in indices], dtype='i8')

        normal = indices > constants.keys_below_are_abs_refs
        out[normal] = pos[indices[normal]]

        for row, i in zip(np.nonzero(~normal), indices[~normal]):
            out[row] = constants.absolute_refs[i]

        self.index = old_index
        return out

    def get_distance_to(self, origin=None, other_atoms=None, sort=False):
        """Return a Cartesian with a column for the distance from origin.
        """
        if origin is None:
            origin = np.zeros(3)
        elif pd.api.types.is_list_like(origin):
            origin = np.array(origin, dtype='f8')
        else:
            origin = self.loc[origin, ['x', 'y', 'z']]

        if other_atoms is None:
            other_atoms = self.index

        new = self.loc[other_atoms, :].copy()
        norm = np.linalg.norm
        try:
            new['distance'] = norm((new - origin).loc[:, ['x', 'y', 'z']],
                                   axis=1)
        except AttributeError:
            # Happens if molecule consists of only one atom
            new['distance'] = norm((new - origin).loc[:, ['x', 'y', 'z']])
        if sort:
            new.sort_values(by='distance', inplace=True)
        return new

    def change_numbering(self, rename_dict, inplace=False):
        """Return the reindexed version of Cartesian.

        Args:
            rename_dict (dict): A dictionary mapping integers on integers.

        Returns:
            Cartesian: A renamed copy according to the dictionary passed.
        """
        output = self if inplace else self.copy()
        new_index = [rename_dict.get(key, key) for key in self.index]
        output.index = new_index
        if not inplace:
            return output

    def partition_chem_env(self, n_sphere=4,
                           use_lookup=None):
        """This function partitions the molecule into subsets of the
        same chemical environment.

        A chemical environment is specified by the number of
        surrounding atoms of a certain kind around an atom with a
        certain atomic number represented by a tuple of a string
        and a frozenset of tuples.
        The ``n_sphere`` option determines how many branches the
        algorithm follows to determine the chemical environment.

        Example:
        A carbon atom in ethane has bonds with three hydrogen (atomic
        number 1) and one carbon atom (atomic number 6).
        If ``n_sphere=1`` these are the only atoms we are
        interested in and the chemical environment is::

        ('C', frozenset([('H', 3), ('C', 1)]))

        If ``n_sphere=2`` we follow every atom in the chemical
        enviromment of ``n_sphere=1`` to their direct neighbours.
        In the case of ethane this gives::

        ('C', frozenset([('H', 6), ('C', 1)]))

        In the special case of ethane this is the whole molecule;
        in other cases you can apply this operation recursively and
        stop after ``n_sphere`` or after reaching the end of
        branches.


        Args:
            n_sphere (int):
            use_lookup (bool): Use a lookup variable for
                :meth:`~chemcoord.Cartesian.get_bonds`. The default is
                specified in ``settings['defaults']['use_lookup']``

        Returns:
            dict: The output will look like this::

                { (element_symbol, frozenset([tuples])) : set([indices]) }

                A dictionary mapping from a chemical environment to
                the set of indices of atoms in this environment.
        """
        if use_lookup is None:
            use_lookup = settings['defaults']['use_lookup']

        def get_chem_env(self, i, n_sphere):
            env_index = self.get_coordination_sphere(
                i, n_sphere=n_sphere, only_surface=False,
                give_only_index=True, use_lookup=use_lookup)
            env_index.remove(i)
            atoms = self.loc[env_index, 'atom']
            environment = frozenset(collections.Counter(atoms).most_common())
            return (self.loc[i, 'atom'], environment)

        chemical_environments = collections.defaultdict(set)
        for i in self.index:
            chemical_environments[get_chem_env(self, i, n_sphere)].add(i)
        return dict(chemical_environments)

    def align(self, other, mass_weight=False):
        """Align two Cartesians.

        Minimize the RMSD (root mean squared deviation) between
        ``self`` and ``other``.
        Returns a tuple of copies of ``self`` and ``other`` where
        both are centered around their centroid and
        ``other`` is rotated unto ``self``.
        The rotation minimises the distances between the
        atom pairs of same label.
        Uses the Kabsch algorithm implemented within
        :func:`~.xyz_functions.get_kabsch_rotation`

        Args:
            other (Cartesian):
            mass_weight (bool): Do a mass weighting to find the best rotation

        Returns:
            tuple:
        """
        if mass_weight:
            m1 = (self - self.get_barycenter()).sort_index()
            m2 = (other - other.get_barycenter()).sort_index()
        else:
            m1 = (self - self.get_centroid()).sort_index()
            m2 = (other - other.get_centroid()).sort_index()

        m2 = m1.get_align_transf(m2, mass_weight, centered=True) @ m2
        return m1, m2


    def get_align_transf(self, other, mass_weight=False, centered=False):
        """Return the rotation matrix that aligns other onto self.

        Minimize the RMSD (root mean squared deviation) between
        ``self`` and ``other``.
        The rotation minimises the distances between the
        atom pairs of same label.
        Uses the Kabsch algorithm implemented within
        :func:`~.xyz_functions.get_kabsch_rotation`.
        If ``mass_weight`` is ``True`` the atoms are weighted by their mass.
        The atoms are moved first to the centroid/barycenter (depending on ``mass_weight``)
        if centered is ``False``.

        Args:
            other (Cartesian):
            mass_weight (bool): Do a mass weighting to find the best rotation
            centered (bool): Assume ``self`` and ``other`` to be centered

        Returns:
            tuple:
        """
        if not centered:
            if mass_weight:
                m1 = (self - self.get_barycenter()).sort_index()
                m2 = (other - other.get_barycenter()).sort_index()
            else:
                m1 = (self - self.get_centroid()).sort_index()
                m2 = (other - other.get_centroid()).sort_index()
        else:
            m1 = self
            m2 = other

        pos1 = m1.loc[:, ['x', 'y', 'z']].values
        pos2 = m2.loc[m1.index, ['x', 'y', 'z']].values
        mass = m1.add_data('mass').loc[:, 'mass'].values if mass_weight else None

        return xyz_functions.get_kabsch_rotation(pos1, pos2, mass)




    def reindex_similar(self, other, n_sphere=4):
        """Reindex ``other`` to be similarly indexed as ``self``.

        Returns a reindexed copy of ``other`` that minimizes the
        distance for each atom to itself in the same chemical environemt
        from ``self`` to ``other``.
        Read more about the definition of the chemical environment in
        :func:`Cartesian.partition_chem_env`

        .. note:: It is necessary to align ``self`` and other before
            applying this method.
            This can be done via :meth:`~Cartesian.align`.

        .. note:: It is probably necessary to improve the result using
            :meth:`~Cartesian.change_numbering()`.

        Args:
            other (Cartesian):
            n_sphere (int): Wrapper around the argument for
                :meth:`~Cartesian.partition_chem_env`.

        Returns:
            Cartesian: Reindexed version of other
        """
        def make_subset_similar(m1, subset1, m2, subset2, index_dct):
            """Changes index_dct INPLACE"""
            coords = ['x', 'y', 'z']
            index1 = list(subset1)
            for m1_i in index1:
                dist_m2_to_m1_i = m2.get_distance_to(m1.loc[m1_i, coords],
                                                     subset2, sort=True)

                m2_i = dist_m2_to_m1_i.index[0]
                dist_new = dist_m2_to_m1_i.loc[m2_i, 'distance']
                m2_pos_i = dist_m2_to_m1_i.loc[m2_i, coords]

                counter = itertools.count()
                found = False
                while not found:
                    if m2_i in index_dct.keys():
                        old_m1_pos = m1.loc[index_dct[m2_i], coords]
                        if dist_new < np.linalg.norm(m2_pos_i - old_m1_pos):
                            index1.append(index_dct[m2_i])
                            index_dct[m2_i] = m1_i
                            found = True
                        else:
                            m2_i = dist_m2_to_m1_i.index[next(counter)]
                            dist_new = dist_m2_to_m1_i.loc[m2_i, 'distance']
                            m2_pos_i = dist_m2_to_m1_i.loc[m2_i, coords]
                    else:
                        index_dct[m2_i] = m1_i
                        found = True
            return index_dct

        molecule1 = self.copy()
        molecule2 = other.copy()

        partition1 = molecule1.partition_chem_env(n_sphere)
        partition2 = molecule2.partition_chem_env(n_sphere)

        index_dct = {}
        for key in partition1:
            message = ('You have chemically different molecules, regarding '
                       'the topology of their connectivity.')
            assert len(partition1[key]) == len(partition2[key]), message
            index_dct = make_subset_similar(molecule1, partition1[key],
                                            molecule2, partition2[key],
                                            index_dct)
        molecule2.index = [index_dct[i] for i in molecule2.index]
        return molecule2.loc[molecule1.index]
