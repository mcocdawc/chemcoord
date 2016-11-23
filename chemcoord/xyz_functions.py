from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
try:
    # import itertools.imap as map
    import itertools.izip as zip
except ImportError:
    pass
import numpy as np
import pandas as pd
import collections
from threading import Thread
import subprocess
import os
import tempfile
import warnings
from . import _common_class
from ._exceptions import PhysicalMeaningError
from . import utilities
from . import zmat_functions
from . import export
from .configuration import settings
import io
from io import open
import re


def pick(my_set):
    """Return one element from a set.

    **Do not** make any assumptions about the element to be returned.
    ``pick`` just returns a random element,
    could be the same, could be different.
    """
    assert type(my_set) is set, 'Pick can be applied only on sets.'
    x = my_set.pop()
    my_set.add(x)
    return x




@export
class Cartesian(_common_class.common_methods):
    """The main class for dealing with cartesian Coordinates.

    **Mathematical Operations**:

    It supports binary operators in the logic of the scipy stack, but you need
    python3.x for using the matrix multiplication operator ``@``.

    The general rule is that mathematical operations using the binary operators
    ``+ - * / @`` and the unary operatos ``+ - abs``
    are only applied to the ``['x', 'y', 'z']`` columns.

    **Addition/Subtraction/Multiplication/Division**:
    If you add a scalar to a Cartesian it is added elementwise onto the
    ``['x', 'y', 'z']`` columns.
    If you add a 3-dimensional vector, list, tuple... the first element of this
    vector is added elementwise to the ``'x'`` column of the
    Cartesian instance and so on.
    The last possibility is to add a matrix with
    ``shape=(Cartesian.n_atoms, 3)`` which is again added elementwise.
    The same rules are true for subtraction, division and multiplication.

    **Matrixmultiplication**:
    Only leftsided multiplication with a matrix of ``shape=(n, 3)``,
    where ``n`` is a natural number, are supported.
    The usual usecase is for example
    ``np.diag([1, 1, -1]) @ cartesian_instance``
    to mirror on the x-y plane.

    **Slicing**:

    Slicing is supported and behaves like the ``.loc`` method of pandas.
    The returned type depends on the remaining columns after the slice.
    If the information of the remaining columns
    is sufficient to describe the geometry
    of a molecule, a Cartesian instance is returned.
    Otherwise a ``pandas.DataFrame`` or in the case of one remaining column
    a ``pandas.Series`` is returned.

    ``molecule[:, ['atom', 'x', 'y', 'z']]`` returns a ``Cartesian``.

    ``molecule[:, ['atom', 'x']]`` returns a ``pandas.DataFrame``.

    ``molecule[:, 'atom']`` returns a ``pandas.Series``.

    **Comparison**:

    Comparison for equality with ``==`` is supported.
    It behaves exactly like the equality comparison of DataFrames in pandas.
    Amongst other things this means that the index has to be the same and
    the comparison of floating point numbers is exact and not numerical.
    For this reason you rarely want to use ``==``.
    Usually the question is "are two given molecules chemically the same".
    For this comparison you have to use the function :func:`isclose`, which
    moves to the barycenter, aligns along the principal axes of inertia and
    compares numerically.
    """

    # Look into the numpy manual for description of __array_priority__
    __array_priority__ = 15.0

    def __init__(self, init):
        """How to initialize a Cartesian instance.

        Args:
            frame (pd.DataFrame): A Dataframe with at least the
                columns ``['atom', 'x', 'y', 'z']``.
                Where ``'atom'`` is a string for the elementsymbol.

        Returns:
            Cartesian: A new cartesian instance.
        """
        try:
            self = init._to_Cartesian()

        except AttributeError:
            # Create from pd.DataFrame
            if not self._is_physical(init.columns):
                error_string = 'There are columns missing for a \
                                meaningful description of a molecule'
                raise PhysicalMeaningError(error_string)
            self.frame = init.copy()
            self.shape = self.frame.shape
            self.n_atoms = self.shape[0]
            self.metadata = {}
            """Test"""
            self._metadata = {}

    def __getitem__(self, key):
        # overwrites the method defined in _pandas_wrapper
        frame = self.frame.loc[key[0], key[1]]
        try:
            if self._is_physical(frame.columns):
                molecule = self.__class__(frame)
                # NOTE here is the difference to the _pandas_wrapper definition
                # TODO make clear in documentation that metadata is an
                # alias/pointer
                # TODO persistent attributes have to be inserted here
                molecule.metadata = self.metadata
                molecule._metadata = self.metadata.copy()
                keys_not_to_keep = [
                    'bond_dict'   # You could end up with loose ends
                    ]
                for key in keys_not_to_keep:
                    try:
                        molecule._metadata.pop(key)
                    except KeyError:
                        pass
                return molecule
            else:
                return frame
        except AttributeError:
            # A series and not a DataFrame was returned
            return frame

    def copy(self):
        molecule = self.__class__(self.frame)
        molecule.metadata = self.metadata
        for key in self._metadata.keys():
            molecule._metadata[key] = self._metadata[key]
        return molecule

    def __add__(self, other):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        try:
            assert set(self.index) == set(other.index)
            assert np.alltrue(self[:, 'atom'] == other[self.index, 'atom'])
            new_molecule[:, coords] = self[:, coords] + other[:, coords]
        except (TypeError, IndexError, AttributeError):
            # It is a shape=3 vector or list
            new_molecule[:, coords] = self[:, coords] + other
        return new_molecule

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        try:
            assert set(self.index) == set(other.index)
            assert np.alltrue(self[:, 'atom'] == other[self.index, 'atom'])
            new_molecule[:, coords] = self[:, coords] - other[:, coords]
        except (TypeError, IndexError, AttributeError):
            # It is a shape=3 vector or list
            new_molecule[:, coords] = self[:, coords] - other
        return new_molecule

    def __rsub__(self, other):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        try:
            assert set(self.index) == set(other.index)
            assert np.alltrue(self[:, 'atom'] == other[self.index, 'atom'])
            new_molecule[:, coords] = other[:, coords] - self[:, coords]
        except (TypeError, IndexError, AttributeError):
            # It is a shape=3 vector or list
            new_molecule[:, coords] = other - self[:, coords]
        return new_molecule

    def __mul__(self, other):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        try:
            assert set(self.index) == set(other.index)
            assert np.alltrue(self[:, 'atom'] == other[self.index, 'atom'])
            new_molecule[:, coords] = self[:, coords] * other[:, coords]
        except (TypeError, IndexError, AttributeError):
            # It is a shape=3 vector or list
            new_molecule[:, coords] = self[:, coords] * other
        return new_molecule

    def __rmul__(self, other):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        try:
            assert set(self.index) == set(other.index)
            assert np.alltrue(self[:, 'atom'] == other[self.index, 'atom'])
            new_molecule[:, coords] = self[:, coords] * other[:, coords]
        except (TypeError, IndexError, AttributeError):
            # It is a shape=3 vector or list
            new_molecule[:, coords] = self[:, coords] * other
        return new_molecule

    def __truediv__(self, other):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        try:
            assert set(self.index) == set(other.index)
            assert np.alltrue(self[:, 'atom'] == other[self.index, 'atom'])
            new_molecule[:, coords] = self[:, coords] / other[:, coords]
        except (TypeError, IndexError, AttributeError):
            # It is a shape=3 vector or list
            new_molecule[:, coords] = self[:, coords] / other
        return new_molecule

    def __rtruediv__(self, other):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        try:
            assert set(self.index) == set(other.index)
            assert np.alltrue(self[:, 'atom'] == other[self.index, 'atom'])
            new_molecule[:, coords] = other[:, coords] / self[:, coords]
        except (TypeError, IndexError, AttributeError):
            # It is a shape=3 vector or list
            new_molecule[:, coords] = other / self[:, coords]
        return new_molecule

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        new_molecule[:, coords] = abs(new_molecule[:, coords])
        return new_molecule

    def __neg__(self):
        return -1 * self.copy()

    def __rmatmul__(self, other):
        coords = ['x', 'y', 'z']
        new_molecule = self.copy()
        new_molecule[:, coords] = (np.dot(other, new_molecule[:, coords].T)).T
        return new_molecule

    def __eq__(self, other):
        return np.alltrue(self.frame == other.frame)

    def _to_Cartesian(self):
        return self.copy()

    def _to_ase_Atoms(self):
        import ase
        atoms = ''.join(self[:, 'atom'])
        positions = self.location()
        # test
        return ase.Atoms(atoms, positions)


    def _give_distance_array(self):
        """Returns a frame with a column for the distance from origin.
        """
        convert_to = {
            'frame': dict(zip(range(self.n_atoms), self.index)),
            'array': dict(zip(self.index, range(self.n_atoms)))}
        location_array = self[:, ['x', 'y', 'z']].values.astype(float)
        A = np.expand_dims(location_array, axis=1)
        B = np.expand_dims(location_array, axis=0)
        C = A - B
        return_array = np.linalg.norm(C, axis=2)
        return return_array, convert_to

    def _overlap(self, bond_size_dic, include=None):
        """Calculates the overlap of van der Vaals radii.

        Do not confuse it with overlap of atomic orbitals.
        This is just a "poor man's overlap" of van der Vaals radii.

        Args:
            bond_size_dic (dic): A dictionary mapping from the
                indices of atoms (integers) to their van der Vaals
                radius.
            include (list): The indices between which the overlap
                should be calculated. If ``None``, the whole index is
                taken.

        Returns:
            tuple: **First element**: overlap_array:
                A (n_atoms, n_atoms) array that contains the overlap
                    between every atom given in the frame.

                **Second element**: convert_to: A nested dictionary
                    that gives the possibility to convert the indices
                    from the frame to the overlap_array and back.
        """
        include = self.index if (include is None) else include

        def summed_bond_size_array(bond_size_dic):
            bond_size = np.array([bond_size_dic[key] for key in include])
            A = np.expand_dims(bond_size, axis=1)
            B = np.expand_dims(bond_size, axis=0)
            C = A + B
            return C

        bond_size_array = summed_bond_size_array(bond_size_dic)
        distance_array, convert_to = self[include, :]._give_distance_array()
        overlap_array = bond_size_array - distance_array
        return overlap_array, convert_to

    def get_bonds(
            self,
            modified_properties=None,
            maximum_edge_length=25,
            difference_edge=6,
            use_valency=False,
            use_lookup=False,
            set_lookup=True,
            divide_et_impera=True,
            atomic_radius_data=settings['defaults']['atomic_radius_data']):
        """Return a dictionary representing the bonds.

        .. warning:: This function is **not sideeffect free**, since it
            assigns the output to a variable ``self._metadata['bond_dict']`` if
            ``set_lookup`` is ``True`` (which is the default). This is
            necessary for performance reasons.

        ``.get_bonds()`` will use or not use a lookup
        depending on ``use_lookup``. Greatly increases performance if
        True, but could introduce bugs in certain situations.

        Just imagine a situation where the ``Cartesian().frame`` is
        changed manually. If you apply lateron a method e.g. ``to_zmat()``
        that makes use of ``get_bonds()`` the dictionary of the bonds
        may not represent the actual situation anymore.

        You have two possibilities to cope with this problem.
        Either you just re-execute ``get_bonds`` on your specific instance,
        or you change the ``internally_use_lookup`` option in the settings.
        Please note that the internal use of the lookup variable
        greatly improves performance.

        Args:
            modified_properties (dic): If you want to change the van der
                Vaals radius or valency of one or more specific atoms, pass a
                dictionary that looks like::

                    modified_properties = {index1 :
                        {'atomic_radius' : 1.5, 'valency' : 8}, ...}

                For global changes use the constants.py module.
            maximum_edge_length (float): Maximum length of one edge of a
            cuboid if ``divide_et_impera`` is ``True``.
            difference_edge (float):
            use_valency (bool): If ``True`` atoms can't have more bonds than
                their valency. This means that the bonds, exceeding the number
                of valency, with lowest overlap will be cut, although the
                van der Waals radii overlap.
            use_lookup (bool):
            set_lookup (bool):
            divide_et_impera (bool): Since the calculation of overlaps or
                distances between atoms scale with :math:`O(n^2)`, it is
                recommended to split the molecule in smaller cuboids and
                calculate the bonds in each cuboid. The scaling becomes
                then :math:`O(n\log(n))`. This approach can lead to problems
                if ``use_valency`` is ``True``. Bonds from one cuboid to
                another can not be counted for the valency.. This means that
                in certain situations some atoms can be oversaturated, although
                ``use_valency`` is ``True``.
            atomic_radius_data (str): Defines which column of
                :attr:`constants.elements` is used. The default is
                ``atomic_radius_cc`` and can be changed with
                :attr:`settings['defaults']['atomic_radius_data']`.
                Compare with :func:`add_data`.

        Returns:
            dict: Dictionary mapping from an atom index to the indices of atoms
            bonded to.
        """
        def preparation_of_variables(modified_properties):
            bond_dic = dict(
                zip(self.index, [set([]) for _ in range(self.n_atoms)]))

            molecule2 = self.add_data(['valency', atomic_radius_data])
            valency_dic = dict(zip(
                molecule2.index, molecule2[:, 'valency'].astype('int64')))

            atomic_radius_dic = dict(zip(molecule2.index,
                                         molecule2[:, atomic_radius_data]))

            if modified_properties is None:
                pass
            else:
                for key in modified_properties:
                    valency_dic[key] = modified_properties[key]['valency']
                    atomic_radius_dic[key] = \
                        modified_properties[key]['atomic_radius']
            return bond_dic, valency_dic, atomic_radius_dic

        def get_bonds_local(
                self,
                bond_dic,
                valency_dic,
                atomic_radius_dic,
                use_valency,
                index_of_cube=self.index):
            overlap_array, convert_to = self._overlap(atomic_radius_dic,
                                                      index_of_cube)
            np.fill_diagonal(overlap_array, -1.)
            bin_overlap_array = overlap_array > 0
            actual_valency = np.sum(bin_overlap_array, axis=1)
            theoretical_valency = np.array([valency_dic[key]
                                            for key in index_of_cube])
            excess_valency = (actual_valency - theoretical_valency)
            indices_of_oversaturated_atoms = np.nonzero(excess_valency > 0)[0]
            oversaturated_converted = [
                convert_to['frame'][index]
                for index in indices_of_oversaturated_atoms]

            if use_valency & (len(indices_of_oversaturated_atoms) > 0):
                if settings['show_warnings']['valency']:
                    warning_string = (
                        'You specified use_valency=True '
                        'and provided a geometry with over saturated '
                        'atoms. This means that the bonds with lowest '
                        'overlap will be cut, although the van der '
                        "Waals radii overlap. If you don't want to see "
                        "this warning execute "
                        "cc.settings['show_warnings']['valency'] = False. "
                        "For a permament change have a look "
                        "at the configuration submodule. "
                        "The problematic indices are:\n") \
                        + oversaturated_converted.__repr__()
                    warnings.warn(warning_string)

                for index in indices_of_oversaturated_atoms:
                    atoms_bonded_to = np.nonzero(
                        bin_overlap_array[index, :])[0]
                    temp_frame = pd.Series(overlap_array[
                        index, atoms_bonded_to], index=atoms_bonded_to)
                    temp_frame.sort_values(inplace=True, ascending=False)
                    cut_bonds_to = temp_frame.iloc[
                        (theoretical_valency[index]):].index
                    overlap_array[index, [cut_bonds_to]] = -1
                    overlap_array[[cut_bonds_to], index] = -1
                    bin_overlap_array = overlap_array > 0

            if (not use_valency) & (len(indices_of_oversaturated_atoms) > 0):
                if settings['show_warnings']['valency']:
                    warning_string = (
                        "You specified use_valency=False (or "
                        "used the default) and provided a geometry with "
                        "over saturated atoms. This means that bonds are "
                        "not cut even if their number exceeds the valency. "
                        "If you don't want to see this warning execute "
                        "cc.settings['show_warnings']['valency'] = False. "
                        "For a permament change have a look "
                        "at the configuration submodule. "
                        "The problematic indices are:\n") \
                        + oversaturated_converted.__repr__()
                    warnings.warn(warning_string)

            def update_dic(bin_overlap_array):
                a, b = np.nonzero(bin_overlap_array)
                a, b = (
                    [convert_to['frame'][key] for key in a],
                    [convert_to['frame'][key] for key in b])
                for row, index in enumerate(a):
                    bond_dic[index] |= set([b[row]])
                return bond_dic

            update_dic(bin_overlap_array)
            return bond_dic

        def complete_calculation(divide_et_impera):
            bond_dic, valency_dic, atomic_radius_dic = \
                preparation_of_variables(modified_properties)
            if divide_et_impera:
                cuboid_dic = self._divide_et_impera(
                    maximum_edge_length, difference_edge)
                for number, key in enumerate(cuboid_dic):
                    get_bonds_local(
                        self, bond_dic, valency_dic,
                        atomic_radius_dic, use_valency,
                        index_of_cube=cuboid_dic[key][1])
            else:
                get_bonds_local(
                    self, bond_dic, valency_dic,
                    atomic_radius_dic, use_valency)
            return bond_dic

        if use_lookup:
            try:
                bond_dic = self._metadata['bond_dict']
            except KeyError:
                bond_dic = complete_calculation(divide_et_impera)
                if set_lookup:
                    self._metadata['bond_dict'] = bond_dic
        else:
            bond_dic = complete_calculation(divide_et_impera)
            if set_lookup:
                self._metadata['bond_dict'] = bond_dic

        return bond_dic

    def _divide_et_impera(self, maximum_edge_length=25., difference_edge=6.):
        """Return a molecule split into cuboids.

        If your algorithm scales with :math:`O(n^2)`.
        You can use this function as a preprocessing step to make it
        scaling with :math:`O(n\log(n))`.

        Args:
            maximum_edge_length (float): Maximum length of one edge
            of a cuboid. difference_edge (float):

        Returns:
            dict: A dictionary mapping from a 3 tuple of integers
            to a 2 tuple of sets. The 3 tuple gives the integer
            numbered coordinates of the cuboids. The first set
            contains the indices of atoms lying in the cube with
            a maximum edge length of ``maximum_edge_length``. They
            are pairwise disjunct and are referred to as small
            cuboids. The second set contains the indices of atoms
            lying in the cube with ``maximum_edge_length +
            difference_edge``. They are a bit larger than the small
            cuboids and overlap with ``difference_edge / 2``.
        """
        coordinates = ['x', 'y', 'z']
        sorted_series = dict(zip(
            coordinates, [
                self[:, axis].sort_values().copy()
                for axis in coordinates]))

        convert = dict(
            (axis, dict(zip(range(self.n_atoms), sorted_series[axis].index)))
            for axis in coordinates)
        sorted_arrays = dict(
            (key, sorted_series[key].values.astype(float))
            for key in coordinates)

        minimum = (np.array([sorted_arrays[key][0] for key in coordinates])
                   - np.array([0.01, 0.01, 0.01]))
        maximum = (np.array([sorted_arrays[key][-1] for key in coordinates])
                   + np.array([0.01, 0.01, 0.01]))
        extent = maximum - minimum
        steps = np.ceil(extent / maximum_edge_length).astype(int)
        cube_dic = {}

        if np.array_equal(steps, np.array([1, 1, 1])):
            small_cube_index = self.index
            big_cube_index = small_cube_index
            cube_dic[(0, 0, 0)] = [small_cube_index, big_cube_index]
        else:
            cuboid_diagonal = extent / steps
            steps = dict((axis, steps[number])
                         for number, axis in enumerate(coordinates))
            edge_small = dict(
                (axis, cuboid_diagonal[number])
                for number, axis in enumerate(coordinates))
            edge_big = dict(
                (axis, (edge_small[axis] + difference_edge))
                for axis in coordinates)
            origin_array = np.empty((steps['x'], steps['y'], steps['z'], 3))

            for x_counter in range(steps['x']):
                for y_counter in range(steps['y']):
                    for z_counter in range(steps['z']):
                        origin_array[x_counter, y_counter, z_counter] = (
                            minimum
                            + cuboid_diagonal / 2
                            + np.dot(
                                np.diag([x_counter, y_counter, z_counter]),
                                cuboid_diagonal))

            origin1D = {}
            origin1D['x'] = dict(
                (counter, origin_array[counter, 0, 0, 0])
                for counter in range(steps['x']))
            origin1D['y'] = dict(
                (counter, origin_array[0, counter, 0, 1])
                for counter in range(steps['y']))
            origin1D['z'] = dict(
                (counter, origin_array[0, 0, counter, 2])
                for counter in range(steps['z']))

            indices = dict(zip(coordinates, [{}, {}, {}]))
            for axis in coordinates:
                for counter in range(steps[axis]):
                    intervall_small = [
                        origin1D[axis][counter] - edge_small[axis] / 2,
                        origin1D[axis][counter] + edge_small[axis] / 2]
                    intervall_big = [
                        origin1D[axis][counter] - edge_big[axis] / 2,
                        origin1D[axis][counter] + edge_big[axis] / 2]
                    bool_vec_small = np.logical_and(
                        intervall_small[0] <= sorted_arrays[axis],
                        sorted_arrays[axis] < intervall_small[1])
                    bool_vec_big = np.logical_and(
                        intervall_big[0] <= sorted_arrays[axis],
                        sorted_arrays[axis] < intervall_big[1])
                    index_small = set(np.nonzero(bool_vec_small)[0])
                    index_small = set(
                        convert[axis][index] for index in index_small)
                    index_big = set(np.nonzero(bool_vec_big)[0])
                    index_big = set(
                        convert[axis][index] for index in index_big)
                    indices[axis][counter] = [index_small, index_big]

            for x_counter in range(steps['x']):
                for y_counter in range(steps['y']):
                    for z_counter in range(steps['z']):
                        small_cube_index = (indices['x'][x_counter][0]
                                            & indices['y'][y_counter][0]
                                            & indices['z'][z_counter][0])
                        big_cube_index = (indices['x'][x_counter][1]
                                          & indices['y'][y_counter][1]
                                          & indices['z'][z_counter][1])
                        cube_dic[(x_counter, y_counter, z_counter)] = (
                            small_cube_index, big_cube_index)

            def test_output(cube_dic):
                for key in cube_dic.keys():
                    try:
                        assert (cube_dic[key][0]
                                & cube_dic[previous_key][0] == set([])), \
                                ('I am sorry Dave. I made a mistake.'
                                    'Report a bug please.')
                    except UnboundLocalError:
                        pass
                    finally:
                        previous_key = key
        # slows down performance too much
        #            test_output(cube_dic)
        return cube_dic

    def connected_to(
            self, index_of_atom,
            exclude=None,
            give_only_index=False,
            follow_bonds=None):
        """Return a Cartesian of atoms connected to the specified
            one.

        Connected means that a path along covalent bonds exists.

        Args:
            index_of_atom (int):
            exclude (list): Indices in this list are omitted.
            give_only_index (bool): If ``True`` a set of indices is
                returned. Otherwise a new Cartesian instance.
            follow_bonds (int): This option determines how many
                branches the algorithm follows. If ``None`` it stops
                after reaching the end in every branch. If you have a
                single molecule this usually means, that the whole
                molecule is recovered.

        Returns:
            A set of indices or a new Cartesian instance.
        """
        bond_dic = self.get_bonds(use_lookup=True)
        exclude = set([]) if (exclude is None) else set(exclude)

        previous_atoms = (
            (set([index_of_atom]) | set(bond_dic[index_of_atom])) - exclude)
        fixed_atoms = set([index_of_atom]) | previous_atoms

        def new_shell(bond_dic, previous_atoms, fixed_atoms, exclude):
            before_inserting = len(fixed_atoms)
            new_atoms = set([])
            for index in previous_atoms:
                new_atoms = new_atoms | (
                    (bond_dic[index] - exclude) - fixed_atoms)
                fixed_atoms |= new_atoms
            after_inserting = len(fixed_atoms)
            changed = (before_inserting != after_inserting)
            return new_atoms, fixed_atoms, changed

        if follow_bonds is None:
            changed = True
            while changed:
                previous_atoms, fixed_atoms, changed = new_shell(
                    bond_dic, previous_atoms, fixed_atoms, exclude)
        else:
            assert follow_bonds >= 0, 'follow_bonds has to be positive'
            fixed_atoms = set(
                [index_of_atom]) if (follow_bonds == 0) else fixed_atoms
            for _ in range(follow_bonds - 1):
                previous_atoms, fixed_atoms, changed = new_shell(
                    bond_dic, previous_atoms, fixed_atoms, exclude)

        if give_only_index:
            to_return = fixed_atoms
        else:
            to_return = self[fixed_atoms, :]
        return to_return

    def _preserve_bonds(self, sliced_cartesian):
        """Is called after cutting geometric shapes.

        If you want to change the rules how bonds are preserved, when
            applying e.g. :meth:`Cartesian.cutsphere` this is the
            function you have to modify.
        It is recommended to inherit from the Cartesian class to
            tailor it for your project, instead of modifying the
            source code of ChemCoord.

        Args:
            sliced_frame (Cartesian):

        Returns:
            Cartesian:
        """
        included_atoms_set = set(sliced_cartesian.index)
        assert included_atoms_set.issubset(set(self.index)), \
            'The sliced Cartesian has to be a subset of the bigger frame'
        bond_dic = self.get_bonds(use_lookup=True)
        new_atoms = set([])
        for atom in included_atoms_set:
            new_atoms = new_atoms | bond_dic[atom]
        new_atoms = new_atoms - included_atoms_set
        while not new_atoms == set([]):
            index_of_interest = new_atoms.pop()
            included_atoms_set = (
                included_atoms_set |
                self.connected_to(
                    index_of_interest,
                    exclude=included_atoms_set,
                    give_only_index=True))
            new_atoms = new_atoms - included_atoms_set
        molecule = self[included_atoms_set, :]
        return molecule

    def cutsphere(
            self,
            radius=15.,
            origin=[0., 0., 0.],
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
        try:
            origin[0]
        except (TypeError, IndexError):
            origin = self.location(int(origin))

        molecule = self.distance_to(origin)
        if outside_sliced:
            molecule = molecule[molecule[:, 'distance'] < radius, :]
        else:
            molecule = molecule[molecule[:, 'distance'] > radius, :]

        if preserve_bonds:
            molecule = self._preserve_bonds(molecule)

        return molecule

    def cutcuboid(
            self,
            a=20,
            b=None,
            c=None,
            origin=[0, 0, 0],
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
                cut out.
            preserve_bonds (bool): Do not cut covalent bonds.

        Returns:
            Cartesian:
        """
        try:
            origin[0]
        except (TypeError, IndexError):
            origin = self.location(int(origin))
        b = a if b is None else b
        c = a if c is None else c

        origin = dict(zip(['x', 'y', 'z'], list(origin)))

        boolean_vector = ((np.abs((self[:, 'x'] - origin['x'])) < a / 2)
                          & (np.abs((self[:, 'y'] - origin['y'])) < b / 2)
                          & (np.abs((self[:, 'z'] - origin['z'])) < c / 2))
        if outside_sliced:
            molecule = self[boolean_vector, :]
        else:
            molecule = self[~boolean_vector, :]

        if preserve_bonds:
            molecule = self._preserve_bonds(molecule)
        return molecule

    def topologic_center(self):
        """Return the average location.

        Args:
            None

        Returns:
            np.array:
        """
        location_array = self.location()
        return np.mean(location_array, axis=0)

    def barycenter(self):
        """Return the mass weighted average location.

        Args:
            None

        Returns:
            np.array:
        """
        try:
            mass_vector = self[:, 'mass'].values.astype('float64')
        except KeyError:
            mass_molecule = self.add_data('mass')
            mass_vector = mass_molecule[:, 'mass'].values.astype('float64')
        location_array = self.location()
        barycenter = (np.sum(location_array * mass_vector[:, None], axis=0)
                      / self.total_mass())
        return barycenter

    def move(
            self,
            vector=[0, 0, 0],
            matrix=np.identity(3),
            matrix_first=True,
            indices=None,
            copy=False):
        """Move a Cartesian.

        The Cartesian is first rotated, mirrored... by the matrix
        and afterwards translated by the vector.

        Args:
            vector (np.array): default is np.zeros(3)
            matrix (np.array): default is np.identity(3)
            matrix_first (bool): If True the multiplication with the matrix
            is the first operation.
            indices (list): Indices to be moved.
            copy (bool): Atoms are copied or translated to the new location.

        Returns:
            Cartesian:
        """
        output = self.copy()

        indices = self.index if (indices is None) else indices
        vectors = output[indices, ['x', 'y', 'z']]

        if matrix_first:
            vectors = np.dot(np.array(matrix), vectors.T).T
            vectors = vectors + np.array(vector)
        else:
            vectors = vectors + np.array(vector)
            vectors = np.dot(np.array(matrix), vectors.T).T

        if copy:
            max_index = self.index.max()
            index_for_copied_atoms = range(
                max_index + 1, max_index + len(indices) + 1
                )
            temp = self[indices, :].copy()
            temp.index = index_for_copied_atoms
            temp[index_for_copied_atoms, ['x', 'y', 'z']] = vectors
            output = output.append(temp)

        else:
            output[indices, ['x', 'y', 'z']] = vectors
        return output

    def bond_lengths(self, buildlist, start_row=0):
        """Return the distances between given atoms.

        In order to know more about the buildlist, go to
            :func:`to_zmat`.

        Args:
            buildlist (np.array):
            start_row (int):

        Returns:
            list: Vector of the distances between the first and second
                atom of every entry in the buildlist.
        """
        # check sanity of input
        buildlist = np.array(buildlist)
        try:
            buildlist.shape[1]
        except IndexError:
            buildlist = buildlist[None, :]
        else:
            pass

        buildlist = np.array(buildlist)
        own_location = self.location(buildlist[start_row:, 0])
        bond_with_location = self.location(buildlist[start_row:, 1])
        distance_vector = own_location - bond_with_location
        distance_list = np.linalg.norm(distance_vector, axis=1)
        return distance_list

    def angle_degrees(self, buildlist, start_row=0):
        """Return the angles between given atoms.

        In order to know more about the buildlist, go to :func:`to_zmat`.

        Args:
            buildlist (list):
            start_row (int):

        Returns:
            list: List of the angle between the first, second and
                third atom of every entry in the buildlist.
        """
        # check sanity of input
        buildlist = np.array(buildlist)
        try:
            buildlist.shape[1]
        except IndexError:
            buildlist = buildlist[None, :]
        else:
            pass

        buildlist = np.array(buildlist)
        own_location = self.location(buildlist[start_row:, 0])
        bond_with_location = self.location(buildlist[start_row:, 1])
        angle_with_location = self.location(buildlist[start_row:, 2])

        BI, BA = (
            own_location - bond_with_location,
            angle_with_location - bond_with_location)
        bi, ba = (
            BI / np.linalg.norm(BI, axis=1)[:, None],
            BA / np.linalg.norm(BA, axis=1)[:, None])
        dot_product = np.sum(bi * ba, axis=1)
        dot_product[np.isclose(dot_product, 1)] = 1
        dot_product[np.isclose(dot_product, -1)] = -1
        angles = np.degrees(np.arccos(dot_product))
        return angles

    def dihedral_degrees(self, buildlist, start_row=0):
        """Return the angles between given atoms.

        In order to know more about the buildlist, go to :func:`to_zmat`.

        Args:
            buildlist (list):
            start_row (int):

        Returns:
            list: List of the dihedral between the first, second,
                third and fourth atom of every entry in the buildlist.
        """
        # check sanity of input
        buildlist = np.array(buildlist)
        try:
            buildlist.shape[1]
        except IndexError:
            buildlist = buildlist[None, :]
        else:
            pass

        own_location = self.location(buildlist[start_row:, 0])
        bond_with_location = self.location(buildlist[start_row:, 1])
        angle_with_location = self.location(buildlist[start_row:, 2])
        dihedral_with_location = self.location(buildlist[start_row:, 3])
        length = buildlist[start_row:].shape[0]

        DA = dihedral_with_location - angle_with_location
        AB = angle_with_location - bond_with_location
        BI = bond_with_location - own_location

        N1 = np.cross(DA, AB, axis=1)
        N2 = np.cross(AB, BI, axis=1)

        n1 = N1 / np.linalg.norm(N1, axis=1)[:, None]
        n2 = N2 / np.linalg.norm(N2, axis=1)[:, None]

        dot_product = np.sum(n1 * n2, axis=1)
        dot_product[np.isclose(dot_product, 1)] = 1
        dot_product[np.isclose(dot_product, -1)] = -1
        dihedrals = np.degrees(np.arccos(dot_product))

        # the next lines are to test the direction of rotation.
        # is a dihedral really 90 or 270 degrees?
        test_where_to_modify = (
            np.sum(AB * np.cross(n1, n2, axis=1), axis=1) > 0)
        where_to_modify = np.nonzero(test_where_to_modify)[0]

        sign = np.full(length, 1, dtype='float64')
        sign[where_to_modify] = -1
        to_add = np.full(length, 0, dtype='float64')
        to_add[where_to_modify] = 360

        dihedrals = to_add + sign * dihedrals
        return dihedrals

    def fragmentate(self, give_only_index=False):
        """Get the indices of non bonded parts in the molecule.

        Args:
            give_only_index (bool): If ``True`` a set of indices is returned.
                Otherwise a new Cartesian instance.

        Returns:
            list: A list of sets of indices or new Cartesian instances.
        """
        list_fragment_indices = []
        still_to_check = set(self.index)
        while still_to_check != set([]):
            indices = self.connected_to(
                pick(still_to_check),
                give_only_index=True)
            still_to_check = still_to_check - indices
            list_fragment_indices.append(indices)

        if give_only_index:
            value_to_return = list_fragment_indices
        else:
            value_to_return = [self[indices, :] for
                               indices in list_fragment_indices]
        return value_to_return

    def get_fragment(self, list_of_indextuples, give_only_index=False):
        """Get the indices of the atoms in a fragment.

        The list_of_indextuples contains all bondings from the
            molecule to the fragment. ``[(1,3), (2,4)]`` means
            for example that the fragment is connected over two
            bonds. The first bond is from atom 1 in the molecule
            to atom 3 in the fragment. The second bond is from atom
            2 in the molecule to atom 4 in the fragment.

        Args:
            list_of_indextuples (list):
            give_only_index (bool): If ``True`` a set of indices
                is returned. Otherwise a new Cartesian instance.

        Returns:
            A set of indices or a new Cartesian instance.
        """
        exclude = [tuple[0] for tuple in list_of_indextuples]
        index_of_atom = list_of_indextuples[0][1]
        fragment_index = self.connected_to(
            index_of_atom, exclude=exclude, give_only_index=True)
        if give_only_index:
            value_to_return = fragment_index
        else:
            value_to_return = self[fragment_index, :]
        return value_to_return

    def _get_buildlist(self, fixed_buildlist=None):
        """Create a buildlist for a Zmatrix.

        Args:
            fixed_buildlist (np.array): It is possible to provide the
                beginning of the buildlist. The rest is "figured" out
                automatically.

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

        bond_dic = self.get_bonds(use_lookup=True)
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
            index_of_new_atom = distance_to_topologic_center[
                to_be_built, 'distance'].idxmin()
            buildlist[row_in_buildlist, 0] = index_of_new_atom
            convert_index[index_of_new_atom] = row_in_buildlist
            if not first_time:
                bond_with = self.distance_to(
                    index_of_new_atom,
                    already_built)[:, 'distance'].idxmin()
                angle_with = self.distance_to(
                    bond_with,
                    already_built - set([bond_with]))[:, 'distance'].idxmin()
                buildlist[row_in_buildlist, 1:3] = [bond_with, angle_with]
                if not third_time:
                    dihedral_with = self.distance_to(
                        bond_with,
                        already_built - set([bond_with, angle_with])
                        )[:, 'distance'].idxmin()
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


    def _clean_dihedral(self, buildlist_to_check):
        """Reindexe the dihedral defining atom if colinear.

        Args:
            buildlist (np.array):

        Returns:
            np.array: modified_buildlist
        """
        buildlist = buildlist_to_check.copy()

        bond_dic = self.get_bonds(use_lookup=True)

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

        zmat_frame.loc[:, additional_columns] = self[indexlist,
                                                     additional_columns]

        bonds = self.bond_lengths(buildlist, start_row=1)
        angles = self.angle_degrees(buildlist, start_row=2)
        dihedrals = self.dihedral_degrees(buildlist, start_row=3)

        zmat_frame.loc[indexlist, 'atom'] = self[indexlist, 'atom']
        zmat_frame.loc[indexlist[1:], 'bond_with'] = buildlist[1:, 1]
        zmat_frame.loc[indexlist[1:], 'bond'] = bonds
        zmat_frame.loc[indexlist[2:], 'angle_with'] = buildlist[2:, 2]
        zmat_frame.loc[indexlist[2:], 'angle'] = angles
        zmat_frame.loc[indexlist[3:], 'dihedral_with'] = buildlist[3:, 3]
        zmat_frame.loc[indexlist[3:], 'dihedral'] = dihedrals

        lines = np.full(self.n_atoms, True, dtype='bool')
        lines[:3] = False
        zmat_frame.loc[zmat_frame['dihedral'].isnull() & lines, 'dihedral'] = 0

        return zmat_functions.Zmat(zmat_frame)

    def to_zmat(
            self,
            buildlist=None,
            fragment_list=None,
            check_linearity=True):
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

        Returns:
            Zmat: A new instance of :class:`~.zmat_functions.Zmat`.
        """
        if buildlist is None:
            if fragment_list is None:
                buildlist = self._get_buildlist()
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
                    big_molecule = self[big_molecule_index, :]
                    row = big_molecule.n_atoms
                    buildlist[: row, :] = big_molecule._get_buildlist()
                    return buildlist, big_molecule, row

                def add_fragment(
                        self, fragment_tpl, big_molecule, buildlist, row):
                    next_row = row + fragment_tpl[0].n_atoms
                    buildlist[row: next_row, :] = \
                        fragment_tpl[0]._get_buildlist(
                            fragment_tpl[1])
                    return buildlist, big_molecule, row

                buildlist, big_molecule, row = create_big_molecule(
                    self, fragment_list)

                for fragment_tpl in fragment_list:
                    buildlist, big_molecule, row = add_fragment(
                        self, fragment_tpl, big_molecule, buildlist, row)

        if check_linearity:
            buildlist = self._clean_dihedral(buildlist)

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

    def inertia(self):
        """Calculate the inertia tensor and transforms along
        rotation axes.

        This function calculates the inertia tensor and returns
        a 4-tuple.

        Args:
            None

        Returns:
            dict: The returned dictionary has four possible keys:

            ``transformed_Cartesian``:
            A frame that is transformed to the basis spanned by
            the eigenvectors of the inertia tensor. The x-axis
            is the axis with the lowest inertia moment, the
            z-axis the one with the highest. Contains also a
            column for the mass

            ``diag_inertia_tensor``:
            A vector containing the sorted inertia moments after
            diagonalization.

            ``inertia_tensor``:
            The inertia tensor in the old basis.

            ``eigenvectors``:
            The eigenvectors of the inertia tensor in the old basis.
        """
        try:
            mass_vector = self[:, 'mass'].values.astype('float64')
            molecule = self.copy()
        except KeyError:
            molecule = self.add_data('mass')
            mass_vector = molecule[:, 'mass'].values.astype('float64')

        molecule = molecule - molecule.barycenter()
        locations = molecule.location()

        diagonals = (np.sum(locations**2, axis=1)[:, None, None]
                     * np.identity(3)[None, :, :])
        dyadic_product = locations[:, :, None] * locations[:, None, :]
        inertia_tensor = (mass_vector[:, None, None]
                          * (diagonals - dyadic_product)).sum(axis=0)
        diag_inertia_tensor, eigenvectors = np.linalg.eig(inertia_tensor)

        # Sort ascending
        sorted_index = np.argsort(diag_inertia_tensor)
        diag_inertia_tensor = diag_inertia_tensor[sorted_index]
        eigenvectors = eigenvectors[:, sorted_index]

        new_basis = eigenvectors
        new_basis = utilities.orthormalize(new_basis)
        old_basis = np.identity(3)
        Cartesian_mass = self.basistransform(new_basis, old_basis)
        Cartesian_mass = Cartesian_mass - Cartesian_mass.barycenter()

        dic_of_values = dict(zip([
            'transformed_Cartesian', 'diag_inertia_tensor',
            'inertia_tensor', 'eigenvectors'],
            [Cartesian_mass, diag_inertia_tensor, inertia_tensor, eigenvectors]
        ))
        return dic_of_values

    def basistransform(
            self, new_basis,
            old_basis=np.identity(3),
            rotate_only=True):
        """Transform the frame to a new basis.

        This function transforms the cartesian coordinates from an
            old basis to a new one. Please note that old_basis and
            new_basis are supposed to have full Rank and consist of
            three linear independent vectors. If rotate_only is True,
            it is asserted, that both bases are orthonormal and right
            handed. Besides all involved matrices are transposed
            instead of inverted.
        In some applications this may require the function
            :func:`utilities.orthonormalize` as a previous step.

        Args:
            old_basis (np.array):
            new_basis (np.array):
            rotate_only (bool):

        Returns:
            Cartesian: The transformed molecule.
        """
        old_basis = np.array(old_basis)
        new_basis = np.array(new_basis)
        # tuples are extracted row wise
        # For this reason you need to transpose e.g. ex is the first column
        # from new_basis
        if rotate_only:
            ex, ey, ez = np.transpose(old_basis)
            v1, v2, v3 = np.transpose(new_basis)
            assert np.allclose(
                np.dot(old_basis, np.transpose(old_basis)),
                np.identity(3)), 'old basis not orthonormal'
            assert np.allclose(
                np.dot(new_basis, np.transpose(new_basis)),
                np.identity(3)), 'new_basis not orthonormal'
            assert np.allclose(
                np.cross(ex, ey), ez), 'old_basis not righthanded'
            assert np.allclose(
                np.cross(v1, v2), v3), 'new_basis not righthanded'

            basistransformation = np.dot(new_basis, np.transpose(old_basis))
            test_basis = np.dot(np.transpose(basistransformation), new_basis)
            new_cartesian = self.move(matrix=np.transpose(basistransformation))
        else:
            basistransformation = np.dot(new_basis, np.linalg.inv(old_basis))
            test_basis = np.dot(np.linalg.inv(basistransformation), new_basis)
            new_cartesian = self.move(
                matrix=np.linalg.inv(basistransformation))

        assert np.isclose(
            test_basis, old_basis).all(), 'transformation did not work'
        return new_cartesian

    def location(self, indexlist=None):
        """Return the location of an atom.

        You can pass an indexlist or an index.

        Args:
            frame (pd.dataframe):
            indexlist (list): If indexlist is None, the complete index
                is used.

        Returns:
            np.array: A matrix of 3D rowvectors of the location of the
            atoms specified by indexlist. In the case of one index
            given a 3D vector is returned one index.
        """
        indexlist = self.index if indexlist is None else indexlist
        array = self[indexlist, ['x', 'y', 'z']].values.astype(float)
        return array

    def distance_to(self,
                    origin=[0, 0, 0],
                    indices_of_other_atoms=None,
                    sort=False):
        """Return a Cartesian with a column for the distance from origin.
        """
        try:
            origin[0]
        except (TypeError, IndexError):
            origin = self.location(int(origin))
        if indices_of_other_atoms is None:
            indices_of_other_atoms = self.index
        origin = np.array(origin, dtype=float)

        output = self[indices_of_other_atoms, :].copy()
        other_locations = output.location()
        output[:, 'distance'] = np.linalg.norm(other_locations
                                               - origin, axis=1)
        if sort:
            output.sort_values(by='distance', inplace=True)
        return output

    def change_numbering(self, rename_dict, inplace=False):
        """Return the reindexed version of Cartesian.

        Args:
            rename_dict (dict): A dictionary mapping integers on integers.

        Returns:
            Cartesian: A renamed copy according to the dictionary passed.
        """
        output = self if inplace else self.copy()

        replace_list = list(rename_dict.keys())
        with_list = [rename_dict[key] for key in replace_list]

        output[:, 'temporary_column'] = output.index
        output[:, 'temporary_column'].replace(replace_list,
                                              with_list, inplace=True)

        output.set_index('temporary_column', drop=True, inplace=True)
        output.sort_index(inplace=True)
        output.index.name = None

        if not inplace:
            return output

    def partition_chem_env(self, follow_bonds=4):
        """This function partitions the molecule into subsets of the
        same chemical environment.

        A chemical environment is specified by the number of
        surrounding atoms of a certain kind around an atom with a
        certain atomic number represented by a tuple of a string
        and a frozenset of tuples.
        The ``follow_bonds`` option determines how many branches the
        algorithm follows to determine the chemical environment.

        Example:
        A carbon atom in ethane has bonds with three hydrogen (atomic
        number 1) and one carbon atom (atomic number 6).
        If ``follow_bonds=1`` these are the only atoms we are
        interested in and the chemical environment is::

        ('C', frozenset([('H', 3), ('C', 1)]))

        If ``follow_bonds=2`` we follow every atom in the chemical
        enviromment of ``follow_bonds=1`` to their direct neighbours.
        In the case of ethane this gives::

        ('C', frozenset([('H', 6), ('C', 1)]))

        In the special case of ethane this is the whole molecule;
        in other cases you can apply this operation recursively and
        stop after ``follow_bonds`` or after reaching the end of
        branches.


        Args:
            follow_bonds (int):

        Returns:
            dict: The output will look like this::

                { (element_symbol, frozenset([tuples])) : set([indices]) }

                A dictionary mapping from a chemical environment to
                the set of indices of atoms in this environment.
    """
        env_dict = {}

        def get_chem_env(self, atomseries, index, follow_bonds):
            indices_of_env_atoms = self.connected_to(
                index, follow_bonds=follow_bonds, give_only_index=True)
            indices_of_env_atoms.remove(index)
            own_symbol, atoms = (
                atomseries[index], atomseries[indices_of_env_atoms])
            environment = collections.Counter(atoms).most_common()
            environment = frozenset(environment)
            return (own_symbol, environment)

        atomseries = self[:, 'atom']

        for index in self.index:
            chem_env = get_chem_env(self, atomseries, index, follow_bonds)
            try:
                env_dict[chem_env].add(index)
            except KeyError:
                env_dict[chem_env] = set([index])
        return env_dict

# TODO still to rewrite
#        U = utilities.kabsch(location2, location1)
    def align(self, Cartesian2, ignore_hydrogens=False):
        """Align two Cartesians.

        Searches for the optimal rotation matrix that minimizes
        the RMSD (root mean squared deviation) of ``self`` to
        Cartesian2.
        Returns a tuple of copies of ``self`` and ``Cartesian2`` where
        both are centered around their topologic center and
        ``Cartesian2`` is aligned along ``self``.
        Uses the Kabsch algorithm implemented within
        :func:`~.utilities.kabsch`

        Args:
            Cartesian2 (Cartesian):
            ignore_hydrogens (bool): Hydrogens are ignored for the
            RMSD.

        Returns:
            tuple:
        """
        # TODO rewrite with C function
        molecule1 = self.sort_index()
        molecule2 = Cartesian2.sort_index()
        molecule1[:, 'x':'z'] = (molecule1[:, 'x':'z']
                                 - molecule1.topologic_center())
        molecule2[:, 'x':'z'] = (molecule2[:, 'x':'z']
                                 - molecule2.topologic_center())

        if ignore_hydrogens:
            location1 = molecule1[molecule1[:, 'atom'] != 'H', :].location()
            location2 = molecule2[molecule2[:, 'atom'] != 'H', :].location()
        else:
            location1 = molecule1.location()
            location2 = molecule2.location()

        molecule2[:, ['x', 'y', 'z']] = utilities.rotate(location2, location1)
        return molecule1, molecule2

    def make_similar(self, Cartesian2, follow_bonds=4, prealign=True):
        """Similarize two Cartesians.

        Returns a reindexed copy of ``Cartesian2`` that minimizes the
        distance for each atom in the same chemical environemt
        from ``self`` to ``Cartesian2``.
        Read more about the definition of the chemical environment in
        :func:`Cartesian.partition_chem_env`

        .. warning:: Please check the result with e.g.
            :func:`Cartesian.move_to()`
            It is probably necessary to use the function
            :func:`Cartesian.change_numbering()`.

        Args:
            Cartesian2 (Cartesian):
            max_follow_bonds (int):
            prealign (bool): The method :func:`Cartesian.align()`
            is applied before reindexing.

        Returns:
            tuple: Aligned copy of ``self`` and aligned + reindexed
            version of ``Cartesian2``
        """
        if prealign:
            molecule1, molecule2_new = self.align(Cartesian2)
        else:
            molecule1 = self.copy()
        # Copy ??
            molecule2_new = Cartesian2.copy()

        partition1 = molecule1.partition_chem_env(follow_bonds)
        partition2 = molecule2_new.partition_chem_env(follow_bonds)
        index_dic = {}

        def make_subset_similar(molecule1, subset1,
                                molecule2, subset2, index_dic):
            indexlist1 = list(subset1)
            for index_on_molecule1 in indexlist1:
                distances_to_atom_on_molecule1 = molecule2_new.distance_to(
                    molecule1.location(index_on_molecule1), subset2, sort=True)

                index_on_molecule2 = \
                    distances_to_atom_on_molecule1.frame.iloc[0].name
                distance_new = distances_to_atom_on_molecule1[
                    index_on_molecule2, 'distance']
                location_of_atom2 = distances_to_atom_on_molecule1.location(
                    index_on_molecule2)

                i = 1
                while True:
                    if index_on_molecule2 in index_dic.keys():
                        location_of_old_atom1 = molecule1.location(
                            index_dic[index_on_molecule2])
                        distance_old = utilities.distance(
                            location_of_old_atom1, location_of_atom2)
                        if distance_new < distance_old:
                            indexlist1.append(index_dic[index_on_molecule2])
                            index_dic[index_on_molecule2] = index_on_molecule1
                            break
                        else:
                            index_on_molecule2 = \
                                distances_to_atom_on_molecule1.frame.iloc[i].name
                            distance_new = \
                                distances_to_atom_on_molecule1[
                                    index_on_molecule2, 'distance']
                            location_of_atom2 = \
                                distances_to_atom_on_molecule1.location(
                                    index_on_molecule2)
                            i = i + 1
                    else:
                        index_dic[index_on_molecule2] = index_on_molecule1
                        break
            return index_dic

        for key in partition1.keys():
            assert len(partition1[key]) == len(partition2[key]), \
                (
                    'You have chemically different molecules, regarding the'
                    'topology of their connectivity. Perhaps'
                    ' get_bonds(use_valency=False) helps.')
            index_dic = make_subset_similar(
                molecule1, partition1[key], molecule2_new,
                partition2[key], index_dic)

        new_index = [
            index_dic[old_index2]
            for old_index2 in molecule2_new.index]
        molecule2_new.index = new_index
        molecule2_new.sort_index(inplace=True)

        return molecule1, molecule2_new

    def move_to(self, Cartesian2, step=5, extrapolate=(0, 0)):
        """Return list of Cartesians for the movement from
        self to Cartesian2.

        Args:
            Cartesian2 (Cartesian):
            step (int):
            extrapolate (tuple):

        Returns:
            list: The list contains ``self`` as first and ``Cartesian2``
            as last element.
            The number of intermediate frames is defined by step.
            Please note, that for this reason: len(list) = (step + 1).
            The numbers in extrapolate define how many frames are
            appended to the left and right of the list continuing
            the movement.
        """
        difference = Cartesian2[:, ['x', 'y', 'z']] - self[:, ['x', 'y', 'z']]

        step_frame = difference.copy() / step

        Cartesian_list = []
        temp_Cartesian = self.copy()

        for t in range(-extrapolate[0], step + 1 + extrapolate[1]):
            temp_Cartesian[:, ['x', 'y', 'z']] = (
                self[:, ['x', 'y', 'z']]
                + step_frame.loc[:, ['x', 'y', 'z']] * t)
            Cartesian_list.append(temp_Cartesian.copy())
        return Cartesian_list

    def write(self, outputfile, sort_index=True, filetype='xyz'):
        """Write the Cartesian into a file.

        .. note:: Since it permamently writes a file, this function
            is strictly speaking **not sideeffect free**.
            The frame to be written is of course not changed.

        Args:
            outputfile (str):
            sort_index (bool): If sort_index is true, the Cartesian
                is sorted by the index before writing.
            filetype (str): The filetype to be used.
                The default is xyz.
                Supported filetypes are: 'xyz'

        Returns:
            None: None
        """
        def write_xyz(self, outputfile, sort_index):
            frame = self.frame[['atom', 'x', 'y', 'z']].copy()
            if sort_index:
                frame = frame.sort_index()
                n_atoms = frame.shape[0]
                with open(outputfile, mode='w') as f:
                    f.write(str(n_atoms) + 2 * '\n')
                frame.to_csv(
                    outputfile,
                    sep=str(' '),
                    index=False,
                    header=False,
                    mode='a')
            else:
                frame = frame.sort_values(by='atom')
                n_atoms = frame.shape[0]
                with open(outputfile, mode='w') as f:
                    f.write(str(n_atoms) + 2 * '\n')
                frame.to_csv(
                    outputfile,
                    sep=str(' '),
                    # https://github.com/pydata/pandas/issues/6035
                    index=False,
                    header=False,
                    mode='a')

        if filetype == 'xyz':
            write_xyz(self, outputfile, sort_index)
        # elif filetype == other filetype:
        else:
            error_string = 'The desired filetype is not implemented'
            raise NotImplementedError(error_string)

    @classmethod
    def read(cls, inputfile, pythonic_index=False,
             get_bonds=True, filetype='auto'):
        """Read a file of coordinate information.

        Args:
            inputfile (str):
            pythonic_index (bool):
            filetype (str): The filetype to be read from.
                The default is xyz.
                Supported filetypes are: xyz and molden.
                'auto' uses the charakters after the last point as filetype.

        Returns:
            Cartesian: Depending on the type of file returns a Cartesian,
            or a list of Cartesians.
        """
        def read_xyz(cls, inputfile, pythonic_index, get_bonds):
            frame = pd.read_table(
                inputfile,
                skiprows=2,
                comment='#',
                delim_whitespace=True,
                names=['atom', 'x', 'y', 'z'])

            if not pythonic_index:
                n_atoms = frame.shape[0]
                frame.index = range(1, n_atoms+1)

            molecule = cls(frame)
            if get_bonds:
                previous_warnings_bool = settings['show_warnings']['valency']
                settings['show_warnings']['valency'] = False
                molecule.get_bonds(
                    use_lookup=False, set_lookup=True, use_valency=False)
                settings['show_warnings']['valency'] = previous_warnings_bool
            return molecule

        def read_molden(cls, inputfile, pythonic_index, get_bonds):
            """Read a molden file.

            Args:
                inputfile (str):
                pythonic_index (bool):

            Returns:
                list: A list containing Cartesian is returned.
            """
            f = open(inputfile, 'r')

            found = False
            while not found:
                line = f.readline()
                if line.strip() == '[N_GEO]':
                    found = True
                    number_of_molecules = int(f.readline().strip())

            found = False
            while not found:
                line = f.readline()
                if line.strip() == '[GEOMETRIES] (XYZ)':
                    found = True
                    current_line = f.tell()
                    number_of_atoms = int(f.readline().strip())
                    f.seek(current_line)

            for i in range(number_of_molecules):
                molecule_in = [f.readline()
                               for j in range(number_of_atoms + 2)]
                molecule_in = ''.join(molecule_in)
                molecule_in = io.StringIO(molecule_in)
                molecule = cls.read(molecule_in, pythonic_index=pythonic_index,
                                    get_bonds=get_bonds, filetype='xyz')
                try:
                    list_of_cartesians.append(molecule)
                except NameError:
                    list_of_cartesians = [molecule]

            f.close()
            return list_of_cartesians

        if filetype == 'auto':
            filetype = re.split('\.', inputfile)[-1]

        if filetype == 'xyz':
            return read_xyz(cls, inputfile, pythonic_index, get_bonds)
        if filetype == 'molden':
            return read_molden(cls, inputfile, pythonic_index, get_bonds)
        else:
            error_string = 'The desired filetype is not implemented'
            raise NotImplementedError(error_string)

    def view(self, viewer=settings['defaults']['viewer'], use_curr_dir=False):
        """View your molecule.

        .. note:: This function writes a temporary file and opens it with
            an external viewer.
            If you modify your molecule afterwards you have to recall view
            in order to see the changes.

        Args:
            viewer (str): The external viewer to use. The default is
                specified in cc.settings.settings['viewer']
            use_curr_dir (bool): If True, the temporary file is written to
                the current diretory. Otherwise it gets written to the
                OS dependendent temporary directory.

        Returns:
            None:
        """
        if use_curr_dir:
            TEMP_DIR = os.path.curdir
        else:
            TEMP_DIR = tempfile.gettempdir()

        def give_filename(i):
            filename = 'ChemCoord_list_' + str(i) + '.molden'
            return os.path.join(TEMP_DIR, filename)

        i = 1
        while os.path.exists(give_filename(i)):
            i = i + 1
        self.write(give_filename(i))

        def open(i):
            """Open file and close after being finished."""
            try:
                subprocess.check_call([viewer, give_filename(i)])
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise
            finally:
                if use_curr_dir:
                    pass
                else:
                    os.remove(give_filename(i))

        Thread(target=open, args=(i,)).start()

    def has_same_sumformula(self, other):
        same_atoms = True
        for atom in set(self[:, 'atom']):
            own_atom_number = self[self[:, 'atom'] == atom, :].shape[0]
            other_atom_number = other[other[:, 'atom'] == atom, :].shape[0]
            same_atoms = (own_atom_number == other_atom_number)
            if not same_atoms:
                break
        return same_atoms


@export
def view(molecule, viewer=settings['defaults']['viewer'], use_curr_dir=False):
    """View your molecule or list of molecules.

    .. note:: This function writes a temporary file and opens it with
        an external viewer.
        If you modify your molecule afterwards you have to recall view
        in order to see the changes.

    Args:
        molecule: Can be a cartesian, or a list of cartesians.
        viewer (str): The external viewer to use. The default is
            specified in settings.viewer
        use_curr_dir (bool): If True, the temporary file is written to
            the current diretory. Otherwise it gets written to the
            OS dependendent temporary directory.

    Returns:
        None:
    """
    try:
        molecule.view(viewer=viewer, use_curr_dir=use_curr_dir)
    except AttributeError:
        if use_curr_dir:
            TEMP_DIR = os.path.curdir
        else:
            TEMP_DIR = tempfile.gettempdir()

        def give_filename(i):
            filename = 'ChemCoord_list_' + str(i) + '.molden'
            return os.path.join(TEMP_DIR, filename)

        i = 1
        while os.path.exists(give_filename(i)):
            i = i + 1

        write(molecule, give_filename(i), filetype='molden')

        def open(i):
            """Open file and close after being finished."""
            try:
                subprocess.check_call([viewer, give_filename(i)])
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise
            finally:
                if use_curr_dir:
                    pass
                else:
                    os.remove(give_filename(i))
        Thread(target=open, args=(i,)).start()


@export
def write(to_be_written, outputfile, sort_index=True, filetype='auto'):
    """Write the coordinates into a file.

    .. note:: Since it permamently writes a file, this function is
        strictly speaking **not sideeffect free**.
        The frame to be written is of course not changed.

    Args:
        to_be_written (Cartesian): This can be a :class:`Cartesian` for
            xyz files
            and a :class:`list` of :class:`Cartesian` for molden files.
        outputfile (str):
        sort_index (bool): If sort_index is true, the Cartesian
            is sorted by the index before writing.
        filetype (str): The filetype to be used.
            The default is auto.
            Supported filetypes are: 'xyz' and 'molden'.
            'auto' uses the charakters after the last point as filetype.

    Returns:
        None:
    """
    def write_xyz(to_be_written, outputfile, sort_index):
        frame = to_be_written.frame[['atom', 'x', 'y', 'z']].copy()
        if sort_index:
            frame = frame.sort_index()
            n_atoms = frame.shape[0]
            with open(outputfile, mode='w') as f:
                f.write(str(n_atoms) + 2 * '\n')
            frame.to_csv(
                outputfile,
                sep=str(' '),
                index=False,
                header=False,
                mode='a')
        else:
            frame = frame.sort_values(by='atom')
            n_atoms = frame.shape[0]
            with open(outputfile, mode='w') as f:
                f.write(str(n_atoms) + 2 * '\n')
            frame.to_csv(
                outputfile,
                sep=str(' '),  # https://github.com/pydata/pandas/issues/6035
                index=False,
                header=False,
                mode='a')

    def write_molden(to_be_written, outputfile, sort_index):
        """Write a list of Cartesians into a molden file.

        .. note:: Since it permamently writes a file, this function
            is strictly speaking **not sideeffect free**.
            The frame to be written is of course not changed.

        Args:
            cartesian_list (list):
            outputfile (str):
            sort_index (bool): If sort_index is true, the Cartesian
                is sorted by the index before writing.

        Returns:
            None:
        """
        if sort_index:
            framelist = [molecule.sort_index().frame
                         for molecule in to_be_written]
        else:
            framelist = [molecule.frame for molecule in to_be_written]
        n_frames = len(framelist)
        n_atoms = to_be_written[0].n_atoms
        values = n_frames * '1\n'
        string = ("[MOLDEN FORMAT]\n"
                  + "[N_GEO]\n"
                  + str(n_frames) + "\n"
                  + '[GEOCONV]\n'
                  + 'energy\n' + values
                  + 'max-force\n' + values
                  + 'rms-force\n' + values
                  + '[GEOMETRIES] (XYZ)\n')

        with open(outputfile, mode='w') as f:
            f.write(string)

        for frame in framelist:
            frame = frame.sort_index()
            n_atoms = frame.shape[0]
            with open(outputfile, mode='a') as f:
                f.write(str(n_atoms) + 2 * '\n')
            frame.to_csv(
                outputfile,
                sep=str(' '),
                index=False,
                header=False,
                mode='a')

    if filetype == 'auto':
        filetype = re.split('\.', outputfile)[-1]

    if filetype == 'xyz':
        write_xyz(to_be_written, outputfile, sort_index)
    elif filetype == 'molden':
        write_molden(to_be_written, outputfile, sort_index)
    else:
        raise NotImplementedError('The desired filetype is not implemented')


@export
def read(inputfile, pythonic_index=False, get_bonds=True, filetype='auto'):
    """Read a file of coordinate information.

    .. note:: This function calls in the background :func:`Cartesian.read`.
        If you inherited from :class:`Cartesian` to tailor it for your project,
        you have to use this method as a constructor.
        Otherwise you can choose.

    Args:
        inputfile (str):
        pythonic_index (bool):
        filetype (str): The filetype to be read from.
            The default is xyz.
            Supported filetypes are: xyz and molden.
            'auto' uses the charakters after the last point as filetype.

    Returns:
        Cartesian: Depending on the type of file returns a Cartesian,
        or a list of Cartesians.
    """
    molecule = Cartesian.read(inputfile, pythonic_index=pythonic_index,
                              get_bonds=get_bonds, filetype=filetype)
    return molecule


@export
def isclose(a, b, align=True, rtol=1.e-5, atol=1.e-8):
    """Compare two molecules for numerical equality.

    Args:
        a (Cartesian):
        b (Cartesian):
        align (bool): a and b are
            prealigned along their principal axes of inertia and moved to their
            barycenters before comparing.
        rtol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for furter explanation.
        atol (float): Relative tolerance for the numerical equality comparison
            look into :func:`numpy.isclose` for furter explanation.

    Returns:
        bool:
    """
    pretest = (set(a.index) == set(b.index)
               and np.alltrue(a[:, 'atom'] == b[a.index, 'atom']))

    if align and pretest:
        A = a.inertia()['transformed_Cartesian'].location()
        B = b.inertia()['transformed_Cartesian'][a.index, :].location()
        return np.allclose(A, B, rtol=rtol, atol=atol)
    elif pretest:
        A = a.location()
        B = b[a.index, :].location()
        return np.allclose(A, B, rtol=rtol, atol=atol)
    else:
        return False



@export
def is_Cartesian(possible_Cartesian):
    """Tests, if given instance is a Cartesian.

    Args:
        is_Cartesian (any type):

    Returns:
        bool:
    """
    columns = possible_Cartesian.columns.copy()
    try:
        assert type(columns) is not str
        columns = set(columns)
    except (TypeError, AssertionError):
        columns = set([columns])
    return {'atom', 'x', 'y', 'z'} <= columns
