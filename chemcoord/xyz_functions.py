
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities
from . import zmat_functions
from . import settings



def pick(my_set):
    """Returns one element from a set.
    """
    x = my_set.pop()
    my_set.add(x)
    return x

class Cartesian:
    def __init__(self, xyz_frame):
        self.xyz_frame = xyz_frame
        self.n_atoms = xyz_frame.shape[0]

    def __repr__(self):
        return self.xyz_frame.__repr__()

    def _repr_html_(self):
        return self.xyz_frame._repr_html_()

# TODO provide wrappers for heavily used pd.DataFrame methods
############################################################################
# From here till shown end pandas wrappers are defined.
############################################################################
#    def give_index(self):
#        """This function returns the index of the underlying pd.DataFrame.
#        Please note that you can't assign a new index. 
#        Instead you need to write::
#
#            cartesian_object.xyz_frame.index = new_index
#
#        """ 
#        return self.xyz_frame.index
#
#    def loc(self, row, column):
#        return Cartesian(self.xyz_frame.loc[row, column])
#
############################################################################
# end of pandas wrapper definition
############################################################################

    def _give_distance_array(self, include):
        """Returns a xyz_frame with a column for the distance from origin.
        """
        frame = self.xyz_frame.loc[include, :]
        convert_to = {
                'frame' : dict(zip(range(len(frame.index)), frame.index)),
                'array' : dict(zip(frame.index, range(len(frame.index))))
                }
        location_array = frame.loc[:, ['x', 'y', 'z']].get_values().astype(float)
        A = np.expand_dims(location_array, axis=1)
        B = np.expand_dims(location_array, axis=0)
        C = A - B
        return_array = np.linalg.norm(C, axis=2)
        return return_array, convert_to

    def _overlap(self, bond_size_dic, include=None):
        """Calculates the overlap of van der Vaals radii.

        Args:
            frame (pd.DataFrame): The indices of atoms for which the overlap should be calculated.
            bond_size_dic (dic): A dictionary mapping from the indices of atoms (integers) to their 
                van der Vaals radius.
    
        Returns:
            tuple: **First element**: overlap_array:
                A (n_atoms, n_atoms) array that contains the overlap between every atom given in the frame.

                **Second element**: convert_to: A nested dictionary that gives the possibility to convert the indices 
                from the frame to the overlap_array and back.
        """
        include = self.xyz_frame.index if (include is None) else include
        def summed_bond_size_array(bond_size_dic):
            bond_size_vector = np.array([bond_size_dic[key] for key in include])
            A = np.expand_dims(bond_size_vector, axis=1)
            B = np.expand_dims(bond_size_vector, axis=0)
            C = A + B
            return C

        bond_size_array = summed_bond_size_array(bond_size_dic)
        distance_array, convert_to = self._give_distance_array(include)
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
                divide_et_impera=True
                ):
        """Returns a dictionary representing the bonds.

        .. warning:: This function is **not sideeffect free**, since it assigns the output to 
            a variable ``self.__bond_dic`` if ``set_lookup`` is ``True`` (which is the default). 
            This is necessary for performance reasons.

        The Cartesian().get_bonds() method will use or not use a lookup depending on ``use_lookup``.
        Greatly increases performance if True, but could introduce bugs in certain situations.

        Just imagine a situation where the ``Cartesian().xyz_frame`` is changed manually. 
        If you apply lateron a method e.g. ``to_zmat()`` that makes use of 
        ``get_bonds()`` the dictionary of the bonds
        may not represent the actual situation anymore.

        You have two possibilities to cope with this problem. 
        Either you just re-execute ``get_bonds`` on your specific instance,
        or you change the ``internally_use_lookup`` option in the settings submodule.
        Please note that the internal use of the lookup variable greatly improves performance.

        Args:
            modified_properties (dic): If you want to change the van der Vaals 
                radius or valency of one or more specific atoms, pass a dictionary that looks like::

                    modified_properties = {index1 : {'bond_size' : 1.5, 'valency' : 8}, ...}

                For global changes use the constants.py module.
            maximum_edge_length (float): Maximum length of one edge of a cuboid if ``divide_et_impera`` is ``True``.
            difference_edge (float): 
            use_valency (bool): If ``True`` atoms can't have more bonds than their valency.
                This means that the bonds, exceeding the number of valency, with lowest overlap will be cut, although the van der Waals radii overlap.
            use_lookup (bool):
            set_lookup (bool):
            divide_et_impera (bool): Since the calculation of overlaps or distances between atoms scale with :math:`O(n^2)`,
                it is recommended to split the molecule in smaller cuboids and calculate the bonds in each cuboid.
                The scaling becomes then :math:`O(n\log(n))`.
                This approach can lead to problems if ``use_valency`` is ``True``. 
                Bonds from one cuboid to another can not be counted for the valency..
                This means that in certain situations some atoms can be oversaturated, although ``use_valency`` is ``True``.
    
        Returns:
            dic: Dictionary mapping from an atom index to the indices of atoms bonded to.

        """
        def preparation_of_variables(modified_properties):
            bond_dic = dict(zip(self.xyz_frame.index, [set([]) for _ in range(self.n_atoms)]))

            # TODO ask Steven and make more efficient 
            valency_dic = dict(zip(
                    self.xyz_frame.index, 
                    [constants.atom_properties[self.xyz_frame.at[index, 'atom']]['valency'] for index in self.xyz_frame.index]
                    ))
            bond_size_dic = dict(zip(
                    self.xyz_frame.index, 
                    [constants.atom_properties[self.xyz_frame.at[index, 'atom']]['bond_size'] for index in self.xyz_frame.index]
                    ))
            if modified_properties is None:
                pass
            else:
                for key in modified_properties:
                    valency_dic[key] = modified_properties[key]['valency']
                    bond_size_dic[key] = modified_properties[key]['bond_size']
            return bond_dic, valency_dic, bond_size_dic


        def get_bonds_local(self, bond_dic, valency_dic, bond_size_dic, use_valency, index_of_cube=self.xyz_frame.index):
            overlap_array, convert_to = self._overlap(bond_size_dic, index_of_cube)
            np.fill_diagonal(overlap_array, -1.)
            bin_overlap_array = overlap_array > 0
            actual_valency = np.sum(bin_overlap_array, axis=1)
            theoretical_valency = np.array([valency_dic[key] for key in index_of_cube])
            excess_valency = (actual_valency - theoretical_valency)
            indices_of_oversaturated_atoms = np.nonzero(excess_valency > 0)[0]
            oversaturated_converted = [convert_to['frame'][index] for index in indices_of_oversaturated_atoms]

            if use_valency & (len(indices_of_oversaturated_atoms) > 0):
                if settings.show_warnings['valency']:
                    warning_string = """Warning: You specified use_valency=True and provided a geometry with over saturated atoms. 
This means that the bonds with lowest overlap will be cut, although the van der Waals radii overlap.
If you don't want to see this warning go to settings.py and edit the dictionary.
The problematic indices are:\n""" + oversaturated_converted.__repr__()
                    print(warning_string)
                select = np.nonzero(overlap_array[indices_of_oversaturated_atoms, :])
                for index in indices_of_oversaturated_atoms:
                    atoms_bonded_to = np.nonzero(bin_overlap_array[index, :])[0]
                    temp_frame = pd.Series(overlap_array[index, atoms_bonded_to], index=atoms_bonded_to)
                    temp_frame.sort_values(inplace=True, ascending=False)
                    cut_bonds_to = temp_frame.iloc[(theoretical_valency[index]) : ].index
                    overlap_array[index, [cut_bonds_to]] = -1
                    overlap_array[[cut_bonds_to], index] = -1
                    bin_overlap_array = overlap_array > 0

            if (not use_valency) & (len(indices_of_oversaturated_atoms) > 0):
                if settings.show_warnings['valency']:
                    warning_string = """Warning: You specified use_valency=False (or used the default) 
and provided a geometry with over saturated atoms. 
This means that bonds are not cut even if their number exceeds the valency.
If you don't want to see this warning go to settings.py and edit the dictionary.
The problematic indices are:\n""" + oversaturated_converted.__repr__()
                    print(warning_string)
            
            def update_dic(bin_overlap_array):
                a,b = np.nonzero(bin_overlap_array)
                a, b = [convert_to['frame'][key] for key in a], [convert_to['frame'][key] for key in b]
                for row, index in enumerate(a):
                    bond_dic[index] |= set([b[row]])
                return bond_dic

            update_dic(bin_overlap_array)
            return bond_dic

        def complete_calculation(divide_et_impera):
            bond_dic, valency_dic, bond_size_dic = preparation_of_variables(modified_properties)
            if divide_et_impera:
                cuboid_dic = self._divide_et_impera(maximum_edge_length, difference_edge)
                for number, key in enumerate(cuboid_dic):
                    get_bonds_local(self, bond_dic, valency_dic, bond_size_dic, use_valency, index_of_cube=cuboid_dic[key][1])
            else:
                get_bonds_local(self, bond_dic, valency_dic, bond_size_dic, use_valency)
            return bond_dic

        if use_lookup:
            try:
                bond_dic = self.__bond_dic
            except AttributeError:
                bond_dic = complete_calculation(divide_et_impera)
                if set_lookup:
                    self.__bond_dic = bond_dic
        else:
            bond_dic = complete_calculation(divide_et_impera)
            if set_lookup:
                self.__bond_dic = bond_dic

        return bond_dic

    def _divide_et_impera(self, maximum_edge_length=25., difference_edge=6.):
        """Returns a molecule split into cuboids.
         
        Args:
            maximum_edge_length (float): Maximum length of one edge of a cuboid. 
            difference_edge (float): 
    
        Returns:
            dict: A dictionary mapping from a 3 tuple of integers to a 2 tuple of sets.
                The 3 tuple gives the integer numbered coordinates of the cuboids.
                The first set contains the indices of atoms lying in the cube with a maximum edge length of ``maximum_edge_length``.
                They are pairwise disjunct and are referred to as small cuboids.
                The second set contains the indices of atoms lying in the cube with ``maximum_edge_length + difference_edge``.
                They are a bit larger than the small cuboids and overlap with ``difference_edge / 2``.
        """
        coordinates = ['x', 'y', 'z']
        sorted_series = dict(zip(
                coordinates, 
                [self.xyz_frame[axis].sort_values().copy() for axis in coordinates]
                ))
        convert = {axis : dict(zip(range(self.n_atoms), sorted_series[axis].index)) for axis in coordinates}
        sorted_arrays = {key : sorted_series[key].get_values().astype(float) for key in coordinates}

        list_of_cuboid_tuples = []
        minimum = np.array([sorted_arrays[key][ 0] for key in coordinates]) - np.array([0.01, 0.01, 0.01])
        maximum = np.array([sorted_arrays[key][-1] for key in coordinates]) + np.array([0.01, 0.01, 0.01])
        extent = maximum - minimum
        steps = np.ceil(extent / maximum_edge_length).astype(int)
        cube_dic = {}

        if np.array_equal(steps, np.array([1, 1, 1])):
            small_cube_index = self.xyz_frame.index
            big_cube_index = small_cube_index
            cube_dic[(0, 0, 0)] = [small_cube_index, big_cube_index]
        else:
            cuboid_diagonal = extent / steps
            steps = {axis : steps[number] for number, axis in enumerate(coordinates)}
            edge_small = {axis : cuboid_diagonal[number] for number, axis in enumerate(coordinates)}
            edge_big = {axis : (edge_small[axis] + difference_edge) for axis in coordinates}
            origin_array = np.empty((steps['x'], steps['y'], steps['z'], 3))

            for x_counter in range(steps['x']):
                for y_counter in range(steps['y']):
                    for z_counter in range(steps['z']):
                        origin_array[x_counter, y_counter, z_counter] = (
                                  minimum + cuboid_diagonal / 2 
                                + np.dot(np.diag([x_counter, y_counter, z_counter]), cuboid_diagonal)
                                )

            origin1D = {}
            origin1D['x'] = {counter : origin_array[counter, 0, 0, 0] for counter in range(steps['x'])}
            origin1D['y'] = {counter : origin_array[0, counter, 0, 1] for counter in range(steps['y'])}
            origin1D['z'] = {counter : origin_array[0, 0, counter, 2] for counter in range(steps['z'])}

            indices = dict(zip(coordinates, [{}, {}, {}]))
            for axis in coordinates:
                for counter in range(steps[axis]):
                    intervall_small = [origin1D[axis][counter] - edge_small[axis] / 2, origin1D[axis][counter] + edge_small[axis] / 2]
                    intervall_big = [origin1D[axis][counter] - edge_big[axis] / 2, origin1D[axis][counter] + edge_big[axis] / 2]
                    bool_vec_small = np.logical_and(intervall_small[0] <= sorted_arrays[axis], sorted_arrays[axis] < intervall_small[1])
                    bool_vec_big = np.logical_and(intervall_big[0] <= sorted_arrays[axis], sorted_arrays[axis] < intervall_big[1])
                    index_small = set(np.nonzero(bool_vec_small)[0])
                    index_small = {convert[axis][index] for index in index_small}
                    index_big = set(np.nonzero(bool_vec_big)[0])
                    index_big = {convert[axis][index] for index in index_big}
                    indices[axis][counter] = [index_small, index_big]

            for x_counter in range(steps['x']):
                for y_counter in range(steps['y']):
                    for z_counter in range(steps['z']):
                        small_cube_index = indices['x'][x_counter][0] & indices['y'][y_counter][0] & indices['z'][z_counter][0]
                        big_cube_index = indices['x'][x_counter][1] & indices['y'][y_counter][1] & indices['z'][z_counter][1]
                        cube_dic[(x_counter, y_counter, z_counter)] = (small_cube_index, big_cube_index) 

            def test_output(cube_dic):
                for key in cube_dic.keys():
                    try:
                        assert cube_dic[key][0] & cube_dic[previous_key][0] == set([]), 'I am sorry Dave. I made a mistake. Report a bug please.'
                    except UnboundLocalError:
                        pass
                    finally:
                        previous_key = key
# slows down performance too much
#            test_output(cube_dic) 
        return cube_dic


    @staticmethod
    def _connected_to(index_of_atom, bond_dic, exclude=None):
        """Gives a set of indices of atoms connected to the specified one.
        """
        exclude = set([]) if (exclude is None) else set(exclude)
    
        included_atoms_set = (set([index_of_atom]) | set(bond_dic[index_of_atom])) - exclude
        included_atoms_list = list(included_atoms_set)
        
        for index in included_atoms_list:
            new_atoms = (bond_dic[index] - included_atoms_set) - exclude
            included_atoms_set = new_atoms | included_atoms_set
            for atom in new_atoms:
                included_atoms_list.append(atom)
        return set(included_atoms_list)



    def connected_to(self, index_of_atom, exclude=None):
        """Returns a Cartesian of atoms connected the specified one.

        Connected means that a path along covalent bonds exists.
    
        Args:
            index_of_atom (int):
            exclude (list): Indices in this list are omitted.
    
        Returns:
            Cartesian: 
        """
        bond_dic = self.get_bonds(use_lookup=True)
        fragment_index = self._connected_to(index_of_atom, bond_dic, exclude=None)
        return Cartesian(self.xyz_frame.loc[fragment_index, :])



    def _preserve_bonds(self, sliced_xyz_frame):
        included_atoms_set = set(sliced_xyz_frame.index)
        assert included_atoms_set.issubset(set(self.xyz_frame.index)), 'The sliced frame has to be a subset of the bigger frame'
        included_atoms_list = list(included_atoms_set)
        bond_dic = self.get_bonds(use_lookup=True)
        new_atoms = set([])
        for atom in included_atoms_set:
            new_atoms = new_atoms | bond_dic[atom]
        new_atoms = new_atoms - included_atoms_set
        while not new_atoms == set([]):
            index_of_interest = new_atoms.pop()
            included_atoms_set = included_atoms_set | self._connected_to(index_of_interest, bond_dic, exclude=included_atoms_set)
            new_atoms = new_atoms - included_atoms_set
        chemical_xyz_frame = self.xyz_frame.loc[included_atoms_set, :].copy()
        return chemical_xyz_frame

    def cutsphere(self, radius=15., origin=[0., 0., 0.], outside_sliced=True, preserve_bonds=False):
        """Cuts a sphere specified by origin and radius.
    
        Args:
            radius (float): 
            origin (list): Please note that you can also pass an integer. 
                In this case it is interpreted as the index of the atom which is taken as origin.
            outside_sliced (bool): Atoms outside/inside the sphere are cut out.
            preserve_bonds (bool): Do not cut covalent bonds.
    
        Returns:
            Cartesian: 
        """
        try:
            origin[0]
        except (TypeError, IndexError):
            origin = self.location(int(origin))

        ordered_molecule = self.distance_to(origin)
        frame = ordered_molecule.xyz_frame
        if outside_sliced:
            sliced_xyz_frame = frame[frame['distance'] < radius]
        else:
            sliced_xyz_frame = frame[frame['distance'] > radius]

        if preserve_bonds:
            sliced_xyz_frame = self._preserve_bonds(sliced_xyz_frame)
        return self.__class__(sliced_xyz_frame)

    def cutcuboid(self, a=20, b=None, c=None, origin=[0, 0, 0], outside_sliced = True, preserve_bonds=False):
        """Cuts a cuboid specified by edge and radius.
    
        Args:
            a (float): Value of the a edge.
            b (float): Value of the b edge. Takes value of a if None.
            c (float): Value of the c edge. Takes value of a if None.
            origin (list): Please note that you can also pass an integer. 
                In this case it is interpreted as the index of the atom which is taken as origin.
            outside_sliced (bool): Atoms outside/inside the sphere are cut out.
            preserve_bonds (bool): Do not cut covalent bonds.
    
        Returns:
            pd.dataframe: Sliced xyz_frame
        """
        try:
            origin[0]
        except (TypeError, IndexError):
            origin = self.location(int(origin))
        b = a if b is None else b
        c = a if c is None else c
        xyz_frame = self.xyz_frame.copy()
        origin = dict(zip(['x', 'y', 'z'], list(origin)))
        if outside_sliced:
            sliced_xyz_frame = xyz_frame[
                   (np.abs((xyz_frame['x'] - origin['x'])) < a / 2)
                 & (np.abs((xyz_frame['y'] - origin['y'])) < b / 2)
                 & (np.abs((xyz_frame['z'] - origin['z'])) < c / 2)
                ].copy()
        else:
            sliced_xyz_frame = xyz_frame[
                   (np.abs((xyz_frame['x'] - origin['x'])) > a / 2)
                 & (np.abs((xyz_frame['y'] - origin['y'])) > b / 2)
                 & (np.abs((xyz_frame['z'] - origin['z'])) > c / 2)
                ].copy()

        if preserve_bonds:
            sliced_xyz_frame = self._preserve_bonds(sliced_xyz_frame)

        return self.__class__(sliced_xyz_frame)


    def mass(self):
        """Gives several properties related to mass.
    
        Args:
    
        Returns:
            dic: The returned dictionary has four possible keys:
            
            ``Cartesian_mass``: Cartesian with an additional column for the masses of each atom.
        
            ``total_mass``: The total mass.
    
            ``barycenter``: The mass weighted average location.
        
            ``topologic_center``: The average location.
        """
        xyz_frame_mass = self.xyz_frame.copy()
        indexlist = list(xyz_frame_mass.index)
        n_atoms = xyz_frame_mass.shape[0]
    
        masses_dic = dict(zip(
                constants.atom_properties.keys(), 
                [constants.atom_properties[atom]['mass'] for atom in constants.atom_properties.keys()]
                ))
        masses = [masses_dic[atom] for atom in xyz_frame_mass['atom']]
        xyz_frame_mass['mass'] = masses
        
        total_mass = xyz_frame_mass['mass'].sum()
    
        location_array = self.location(indexlist)
    
        barycenter = np.zeros([3])
        topologic_center = np.zeros([3])
    
        for row, index in enumerate(indexlist):
            barycenter = barycenter + location_array[row] * xyz_frame_mass.at[index, 'mass']
            topologic_center = topologic_center + location_array[row]
        barycenter = barycenter / total_mass
        topologic_center = topologic_center / n_atoms
        
        dic_of_values = dict(zip(
            ['Cartesian_mass', 'total_mass', 'barycenter', 'topologic_center'], 
            [Cartesian(xyz_frame_mass), total_mass, barycenter, topologic_center]
            ))
        return dic_of_values


    def move(self, vector=[0, 0, 0], matrix=np.identity(3), indices=None, copy=False):
        """Move an xyz_frame.
    
        The xyz_frame is first rotated, mirrored... by the matrix
        and afterwards translated by the vector
    
        Args:
            vector (np.array): default is np.zeros(3)
            matrix (np.array): default is np.identity(3)
            indices (list): Indices to be moved.
            copy (bool): Atoms are copied to the new location or not.
    
        Returns:
            Cartesian: 
        """
        indices = self.xyz_frame.index if (indices is None) else indices
        indices = list(indices)
        frame = self.xyz_frame.copy()
        vectors = frame.loc[indices, ['x', 'y', 'z']].get_values().astype(float)
        vectors = np.transpose(np.dot(np.array(matrix), np.transpose(vectors)))
        vectors = vectors + np.array(vector)
        if copy:
            max_index = frame.index.max()
            index_for_copied_atoms = range(max_index + 1, max_index + len(indices) + 1)
            temp_frame = frame.loc[indices, :].copy()
            temp_frame.index = index_for_copied_atoms
            temp_frame.loc[:, ['x', 'y', 'z']] = vectors
            frame = frame.append(temp_frame) 
            
        else:
            frame.loc[indices, ['x', 'y', 'z']] = vectors
        return Cartesian(frame)

    def bond_lengths(self, buildlist, start_row=0):
        """Return the distances between given atoms.
    
        In order to know more about the buildlist, go to :func:`to_zmat`.
       
        Args:
            buildlist (np.array): 
    
        Returns:
            list: Vector of the distances between the first and second atom of every entry in the buildlist. 
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
        own_location = self.location(buildlist[start_row:,0])
        bond_with_location = self.location(buildlist[start_row:,1])
        distance_vector = own_location - bond_with_location
        distance_list = np.linalg.norm(distance_vector, axis=1)
        return distance_list
    

    def angle_degrees(self, buildlist, start_row=0):
        """Return the angles between given atoms.
    
        In order to know more about the buildlist, go to :func:`to_zmat`.
       
        Args:
            xyz_frame (pd.dataframe): 
            buildlist (list): 
            exclude_first (bool): The exclude_first option excludes the first two rows of the buildlist from calculation.
    
        Returns:
            list: List of the angle between the first, second and third atom of every entry in the buildlist. 
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
        own_location = self.location(buildlist[start_row:,0])
        bond_with_location = self.location(buildlist[start_row:,1])
        angle_with_location = self.location(buildlist[start_row:,2])

        BI, BA = own_location - bond_with_location, angle_with_location - bond_with_location
        bi, ba = BI / np.linalg.norm(BI, axis=1)[:, None], BA / np.linalg.norm(BA, axis=1)[:, None]
        dot_product = np.sum(bi * ba, axis=1)
        angles = np.degrees(np.arccos(dot_product))
        return angles


    def dihedral_degrees(self, buildlist, start_row=0):
        """Return the angles between given atoms.
       
        In order to know more about the buildlist, go to :func:`to_zmat`.
    
        Args:
            xyz_frame (pd.dataframe): 
            buildlist (list): 
            exclude_first (bool): The exclude_first option excludes the first three rows of the buildlist from calculation.
    
        Returns:
            list: List of the dihedral between the first, second, third and fourth atom of every entry in the buildlist. 
        """
        # check sanity of input
        buildlist = np.array(buildlist)
        try:
            buildlist.shape[1]
        except IndexError:
            buildlist = buildlist[None, :]
        else:
            pass

        own_location = self.location(buildlist[start_row:,0])
        bond_with_location = self.location(buildlist[start_row:,1])
        angle_with_location = self.location(buildlist[start_row:,2])
        dihedral_with_location = self.location(buildlist[start_row:,3])
        length = buildlist[start_row:].shape[0]
        
        DA = dihedral_with_location - angle_with_location
        AB = angle_with_location - bond_with_location
        BI = bond_with_location - own_location

        N1 = np.cross(DA, AB, axis=1)
        N2 = np.cross(AB, BI, axis=1)

        n1 = N1 / np.linalg.norm(N1, axis=1)[:, None]
        n2 = N2 / np.linalg.norm(N2, axis=1)[:, None]

        dot_product = np.sum(n1 * n2, axis=1)
        dihedrals = np.degrees(np.arccos(dot_product))

        # the next lines are to test the direction of rotation. 
        # is a dihedral really 90 or 270 degrees?
        test_where_to_modify = (np.sum(AB * np.cross(n1, n2, axis=1), axis=1) > 0)
        where_to_modify = np.nonzero(test_where_to_modify)[0]

        sign = np.full(length, 1, dtype='float64')
        sign[where_to_modify] = -1
        to_add = np.full(length, 0, dtype='float64')
        to_add[where_to_modify] = 360

        dihedrals = to_add + sign * dihedrals
        return dihedrals


    def get_fragment(self, list_of_indextuples, give_only_index=False):
        """Get the indices of the atoms in a fragment.
       
        The list_of_indextuples contains all bondings from the molecule to the fragment.
        ``[(1,3), (2,4)]`` means for example that the fragment is connected over two bonds.
        The first bond is from atom 1 in the molecule to atom 3 in the fragment.
        The second bond is from atom 2 in the molecule to atom 4 in the fragment.
    
        Args:
            xyz_frame (pd.dataframe): 
            list_of_indextuples (list): 
            give_only_index (bool): If ``True`` a set of indices is returned. 
                Otherwise a new Cartesian instance.
    
        Returns:
            A set of indices or a new Cartesian instance.
        """
        exclude = [tuple[0] for tuple in list_of_indextuples]
        index_of_atom = list_of_indextuples[0][1]
        fragment_index = self._connected_to(index_of_atom, self.get_bonds(use_lookup=True), exclude=exclude)

        if give_only_index:
            value_to_return = fragment_index
        else:
            value_to_return = self.__class__(self.xyz_frame.loc[fragment_index, :])
        return value_to_return

    def _get_buildlist(self, fixed_buildlist=None):
        """Create a buildlist for a Zmatrix.
        
        Args:
            fixed_buildlist (np.array): It is possible to provide the beginning of the buildlist.
                The rest is "figured" out automatically.
        
        Returns:
            np.array: buildlist
        """
        buildlist = np.zeros((self.n_atoms, 4)).astype('int64')
        if not fixed_buildlist is None:
            buildlist[:fixed_buildlist.shape[0], :] = fixed_buildlist
            start_row = fixed_buildlist.shape[0]
            already_built = set(fixed_buildlist[:, 0])
            to_be_built = set(self.xyz_frame.index) - already_built
            convert_index = dict(zip(buildlist[:, 0], range(start_row)))
        else:
            start_row = 0
            already_built, to_be_built = set([]), set(self.xyz_frame.index)
            convert_index = {}

        bond_dic = self.get_bonds(use_lookup=True)
        topologic_center = self.mass()['topologic_center']
        distance_to_topologic_center = self.distance_to(topologic_center).xyz_frame.copy()
        
        def update(already_built, to_be_built, new_atoms_set):
            """NOT SIDEEFFECT FREE
            """
            for index in new_atoms_set:
                already_built.add(index)
                to_be_built.remove(index)
            return already_built, to_be_built

        def from_topologic_center(self, row_in_buildlist, already_built, to_be_built, first_time=False):
            index_of_new_atom = distance_to_topologic_center.loc[to_be_built, 'distance'].idxmin()
            buildlist[row_in_buildlist, 0] = index_of_new_atom
            convert_index[index_of_new_atom] = row_in_buildlist
            if not first_time:
                bond_with = self.distance_to(index_of_new_atom, already_built).xyz_frame['distance'].idxmin()
                angle_with = self.distance_to(bond_with, already_built - {bond_with}).xyz_frame['distance'].idxmin()
                dihedral_with = self.distance_to(bond_with, already_built - {bond_with, angle_with}).xyz_frame['distance'].idxmin()
                buildlist[row_in_buildlist, 1:] = [bond_with, angle_with, dihedral_with]
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
                new_row_to_modify, already_built, to_be_built = from_topologic_center(self, row_in_buildlist, already_built, to_be_built)
            if len(new_atoms_set) > 1:
                use_index = buildlist[0, 0]
            else:
                use_index = buildlist[1, 0]
            new_row_to_modify = 2
            already_built.add(index_of_new_atom)
            to_be_built.remove(index_of_new_atom)
            return new_row_to_modify, already_built, to_be_built, use_index

        def third_atom(self, already_built, to_be_built, use_index):
            new_atoms_set = bond_dic[use_index] - already_built
            if new_atoms_set != set([]):
                index_of_new_atom = pick(new_atoms_set)
                convert_index[index_of_new_atom] = 2
                buildlist[2, 0] = index_of_new_atom
                buildlist[2, 1] = use_index
                buildlist[2, 2] = pick(already_built - {use_index})
            else:
                new_row_to_modify, already_built, to_be_built = from_topologic_center(self, row_in_buildlist, already_built, to_be_built)
            if len(new_atoms_set) > 1:
                use_index = use_index
            else:
                use_index = buildlist[2, 0]
            new_row_to_modify = 3
            already_built.add(index_of_new_atom)
            to_be_built.remove(index_of_new_atom)
            return new_row_to_modify, already_built, to_be_built, use_index

        def pick_new_atoms(self, row_in_buildlist, already_built, to_be_built, use_given_index=None):
            """Get the indices of new atoms to be put in buildlist.
            
            Tries to get the atoms bonded to the one, that was last inserted into the buildlist.
            If the last atom is the end of a branch, it looks for the index of an atom that is
            the nearest atom to the topologic center and not built in yet.
    
            .. note:: It modifies the buildlist array which is global to this function.
    
            Args:
                row_in_buildlist (int): The row which has to be filled at least.
        
            Returns:
                list: List of modified rows.
            """
            if use_given_index is None:
                new_atoms_set = bond_dic[buildlist[row_in_buildlist-1, 0]] - already_built

                if new_atoms_set != set([]):
                    update(already_built, to_be_built, new_atoms_set)
                    new_row_to_modify = row_in_buildlist + len(new_atoms_set)
                    new_atoms_list = list(new_atoms_set)
                    bond_with = buildlist[row_in_buildlist - 1, 0]
                    angle_with = buildlist[convert_index[bond_with], 1]
                    dihedral_with = buildlist[convert_index[bond_with], 2]
                    buildlist[row_in_buildlist : new_row_to_modify, 0 ] = new_atoms_list
                    buildlist[row_in_buildlist : new_row_to_modify, 1 ] = bond_with
                    buildlist[row_in_buildlist : new_row_to_modify, 2 ] = angle_with
                    buildlist[row_in_buildlist : new_row_to_modify, 3 ] = dihedral_with
                    for key, value in zip(new_atoms_list, range(row_in_buildlist, new_row_to_modify)):
                        convert_index[key] = value
                else:
                    new_row_to_modify, already_built, to_be_built = from_topologic_center(self, row_in_buildlist, already_built, to_be_built)

            else:
                new_atoms_set = bond_dic[use_given_index] - already_built
                new_row_to_modify = row_in_buildlist + len(new_atoms_set)
                new_atoms_list = list(new_atoms_set)
                bond_with = use_given_index
                angle_with, dihedral_with = (already_built - set([use_given_index]))
                buildlist[row_in_buildlist : new_row_to_modify, 0 ] = new_atoms_list
                buildlist[row_in_buildlist : new_row_to_modify, 1 ] = bond_with
                buildlist[row_in_buildlist : new_row_to_modify, 2 ] = angle_with
                buildlist[row_in_buildlist : new_row_to_modify, 3 ] = dihedral_with
                update(already_built, to_be_built, new_atoms_set)
                for key, value in zip(new_atoms_list, range(row_in_buildlist, new_row_to_modify)):
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
                        first_time=True
                        )
            if row == 1 & 1 < buildlist.shape[0]:
                row, already_built, to_be_built, use_index = second_atom(self, already_built, to_be_built)
            if row == 2 & 2 < buildlist.shape[0]:
                row, already_built, to_be_built, use_index = third_atom(self, already_built, to_be_built, use_index)
            if row < buildlist.shape[0]:
                row, already_built, to_be_built = pick_new_atoms(self, row, already_built, to_be_built, use_index)

           
        while row < buildlist.shape[0]:
            row, already_built, to_be_built = pick_new_atoms(self, row, already_built, to_be_built)

        return buildlist

    def clean_dihedral(self, buildlist_to_check):
        """Reindexes the dihedral defining atom if colinear.

        Args:
            buildlist (np.array): 
        
        Returns:
            np.array: modified_buildlist
        """
        buildlist = buildlist_to_check.copy()

        bond_dic = self.get_bonds(use_lookup = True)

        angles = self.angle_degrees(buildlist[3:,1:])

        test_vector = np.logical_or(170 < angles, angles < 10)
        problematic_indices = np.nonzero(test_vector)[0]

        converged = True if len(problematic_indices) == 0 else False

        for index in problematic_indices: 
            index = index + 3
            try:
                buildlist[index, 3] = pick(bond_dic[buildlist[index, 2]] - set(buildlist[index, [1, 3]]))
            except KeyError:
                origin = buildlist[index, 2]
                could_be_reference = set(buildlist[:index, 0]) - set(buildlist[index, :3])
        
                sorted_Cartesian = self.distance_to(origin, could_be_reference)
                sorted_Cartesian.xyz_frame = sorted_Cartesian.xyz_frame.sort_values(by='distance')
                
                bond_with, angle_with = buildlist[index, [1, 2]]
                dihedral_with = np.array(sorted_Cartesian.xyz_frame.index).astype('int64')
                buildlist_for_new_dihedral_with = np.empty((len(could_be_reference), 3), dtype='int64')
                buildlist_for_new_dihedral_with[:, 0]  = bond_with
                buildlist_for_new_dihedral_with[:, 1]  = angle_with
                buildlist_for_new_dihedral_with[:, 2]  = dihedral_with
                angles = self.angle_degrees(buildlist_for_new_dihedral_with)
        
        
                test_vector = np.logical_or(170 > angles, angles > 10)
                index_to_use = buildlist[np.nonzero(test_vector)[0][0], 0]
            
                buildlist[index, 3] = index_to_use
        return buildlist, converged



    def _build_zmat(self, buildlist):
        indexlist = buildlist[:, 0]

        default_columns = ['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'] 
        additional_columns = list(set(self.xyz_frame.columns) - set(['atom', 'x', 'y', 'z']))

        zmat_frame = pd.DataFrame(columns= default_columns + additional_columns,
                dtype='float',
                index=indexlist
                )

        zmat_frame.loc[:, additional_columns] = self.xyz_frame.loc[indexlist, additional_columns]

        bonds = self.bond_lengths(buildlist, start_row=1)
        angles = self.angle_degrees(buildlist, start_row=2)
        dihedrals = self.dihedral_degrees(buildlist, start_row=3)

        zmat_frame.loc[indexlist, 'atom'] = self.xyz_frame.loc[indexlist, 'atom']
        zmat_frame.loc[indexlist[1:], 'bond_with'] = buildlist[1:, 1]
        zmat_frame.loc[indexlist[1:], 'bond'] = bonds 
        zmat_frame.loc[indexlist[2:], 'angle_with'] = buildlist[2:, 2]
        zmat_frame.loc[indexlist[2:], 'angle'] = angles
        zmat_frame.loc[indexlist[3:], 'dihedral_with'] = buildlist[3:, 3]
        zmat_frame.loc[indexlist[3:], 'dihedral'] = dihedrals
        return zmat_frame


    def to_zmat(self, buildlist=None, fragment_list=None, check_linearity=True):
        if (buildlist is None):
            if (fragment_list is None):
                buildlist = self._get_buildlist()
                if check_linearity:
                    buildlist, converged = self.clean_dihedral(buildlist) 
                    i, max_iter = 0, 5
                    while (not converged) & (i == max_iter):
                        buildlist, converged = self.clean_dihedral(buildlist) 
                        print('iteration', i)
                        i = i + 1
            else:
                pass

        zmat = self._build_zmat(buildlist)
        return zmat_functions.Zmat(zmat)

    def inertia(self):
        """Calculates the inertia tensor and transforms along rotation axes.
    
        This function calculates the inertia tensor and returns a 4-tuple.
    
        Args:
    
        Returns:
            dic: The returned dictionary has four possible keys:
            
            ``transformed_Cartesian``:
            A xyz_frame that is transformed to the basis spanned by the eigenvectors 
            of the inertia tensor. The x-axis is the axis with the lowest inertia moment,
            the z-axis the one with the highest. Contains also a column for the mass
        
            ``diag_inertia_tensor``:
            A vector containing the sorted inertia moments after diagonalization.
    
            ``inertia_tensor``:
            The inertia tensor in the old basis.
        
            ``eigenvectors``:
            The eigenvectors of the inertia tensor in the old basis.
        """
        xyz_frame = self.xyz_frame
        keys = ['Cartesian_mass', 'total_mass', 'barycenter', 'topologic_center']
        temp_dict = self.mass()
        Cartesian_mass, total_mass, barycenter, topologic_center = [temp_dict[key] for key in keys]
    
        Cartesian_mass = Cartesian_mass.move(vector = -barycenter)
        frame_mass = Cartesian_mass.xyz_frame
    
        coordinates = ['x', 'y', 'z']
        
        def kronecker(i, j):
            """Please note, that it also compares e.g. strings.
            """
            if i == j:
                return 1
            else:
                return 0
    
        inertia_tensor = np.zeros([3, 3])
        for row_index, row_coordinate in enumerate(coordinates):
            for column_index, column_coordinate in enumerate(coordinates):
                inertia_tensor[row_index, column_index] = (
                        frame_mass.loc[:, 'mass'] * ( kronecker(row_index, column_index) * (frame_mass.loc[:, coordinates]**2).sum(axis=1)
                        - (frame_mass.loc[:, row_coordinate] * frame_mass.loc[:, column_coordinate])
                        )).sum()
    
        
        diag_inertia_tensor, eigenvectors = np.linalg.eig(inertia_tensor)
    
        # Sort ascending
        sorted_index = np.argsort(diag_inertia_tensor)
        diag_inertia_tensor = diag_inertia_tensor[sorted_index]
        eigenvectors = eigenvectors[:, sorted_index]
    
        new_basis = eigenvectors
        new_basis = utilities.orthormalize(new_basis)
        old_basis = np.identity(3)
        Cartesian_mass = self.basistransform(new_basis, old_basis)
        
        dic_of_values =  dict(zip(
            ['transformed_Cartesian', 'diag_inertia_tensor', 'inertia_tensor', 'eigenvectors'],
            [Cartesian_mass, diag_inertia_tensor, inertia_tensor, eigenvectors]
            ))
        return dic_of_values

    def basistransform(self, new_basis, old_basis=np.identity(3), rotate_only=True):
        """Transforms the xyz_frame to a new basis.
    
        This function transforms the cartesian coordinates from an old basis to a new one.
        Please note that old_basis and new_basis are supposed to have full Rank and consist of 
        three linear independent vectors.
        If rotate_only is True, it is asserted, that both bases are orthonormal and right handed.
        Besides all involved matrices are transposed instead of inverted.
        In some applications this may require the function :func:`utilities.orthonormalize` as a previous step.
    
        Args:
            old_basis (np.array):  
            new_basis (np.array): 
            rotate_only (bool): 
    
        Returns:
            pd.DataFrame: The transformed xyz_frame
        """
        frame = self.xyz_frame.copy()
        old_basis = np.array(old_basis)
        new_basis = np.array(new_basis)
        # tuples are extracted row wise
        # For this reason you need to transpose e.g. ex is the first column from new_basis
        if rotate_only:
            ex, ey, ez = np.transpose(old_basis)
            v1, v2, v3 = np.transpose(new_basis)
            assert np.allclose(np.dot(old_basis, np.transpose(old_basis)), np.identity(3)), 'old basis not orthonormal'
            assert np.allclose(np.dot(new_basis, np.transpose(new_basis)), np.identity(3)), 'new_basis not orthonormal'
            assert np.allclose(np.cross(ex, ey), ez), 'old_basis not righthanded'
            assert np.allclose(np.cross(v1, v2), v3), 'new_basis not righthanded'
    
            basistransformation = np.dot(new_basis, np.transpose(old_basis))
            test_basis = np.dot(np.transpose(basistransformation), new_basis)
            new_cartesian = self.move(matrix=np.transpose(basistransformation))
        else:
            basistransformation = np.dot(new_basis, np.linalg.inv(old_basis))
            test_basis = np.dot(np.linalg.inv(basistransformation), new_basis)
            new_cartesian = self.move(matrix=np.linalg.inv(basistransformation))
    
        assert np.isclose(test_basis, old_basis).all(), 'transformation did not work'
        return new_cartesian



    def location(self, indexlist):
        """Returns the location of an atom.
    
        You can pass an indexlist or an index.
    
        Args:
            xyz_frame (pd.dataframe): 
            index (list): 
    
        Returns:
            np.array: A matrix of 3D rowvectors of the location of the atoms
            specified by indexlist. In the case of one index given a 3D vector is returned one index.
        """
        xyz_frame = self.xyz_frame.copy()
        try:
            if not set(indexlist).issubset(set(xyz_frame.index)):
                raise KeyError('One or more indices in the indexlist are not in the xyz_frame')
        except TypeError:
            if not set([indexlist]).issubset(set(xyz_frame.index)):
                raise KeyError('One or more indices in the indexlist are not in the xyz_frame')
        array = xyz_frame.ix[indexlist, ['x', 'y', 'z']].get_values().astype(float)
        return array
    
    
    def distance_to(self, origin, indices_of_other_atoms=None):
        """Returns a xyz_frame with a column for the distance from origin.
        """
        try:
            origin[0]
        except (TypeError, IndexError):
            origin = self.location(int(origin))
        if indices_of_other_atoms is None:
            indices_of_other_atoms = self.xyz_frame.index
        frame_distance = self.xyz_frame.loc[indices_of_other_atoms, :].copy()
        origin = np.array(origin, dtype=float)
        locations = frame_distance.loc[:, ['x', 'y', 'z']].get_values().astype(float)
        frame_distance['distance'] = np.linalg.norm(locations - origin, axis =1)
        return self.__class__(frame_distance)

    def change_numbering(self, rename_dic):
        """Returns the reindexed version of xyz_frame.
    
        Args:
            xyz_frame (pd.dataframe): 
            rename_dic (dic): A dictionary mapping integers on integers.
    
        Returns:
            pd.dataframe: 
        """
        frame = self.xyz_frame.copy()
        replace_list = list(rename_dic.keys())
        with_list = [rename_dic[key] for key in replace_list]
        frame['temporary_column'] = frame.index
        frame.loc[:, 'temporary_column'].replace(replace_list, with_list, inplace=True)
        frame.set_index('temporary_column', drop=True, inplace=True)
        frame.index.name = None
        return self.__class__(frame)


    def write(self, outputfile, sort_index=True):
        """Writes the xyz_frame into a file.
    
        If sort_index is true, the frame is sorted by the index before writing. 
    
        .. note:: Since it permamently writes a file, this function is strictly speaking **not sideeffect free**.
            The frame to be written is of course not changed.
    
        Args:
            xyz_frame (pd.dataframe): 
            outputfile (str): 
            sort_index (bool):
    
        Returns:
            None: None
        """
        frame = self.xyz_frame[['atom', 'x', 'y','z']].copy()
        if sort_index:
            frame = frame.sort_index()
            n_atoms = frame.shape[0]
            with open(outputfile, mode='w') as f:
                f.write(str(n_atoms) + 2 * '\n')
            frame.to_csv(
                outputfile,
                sep=' ',
                index=False,
                header=False,
                mode='a'
                )
        else:
            frame = frame.sort_values(by='atom')
            n_atoms = frame.shape[0]
            with open(outputfile, mode='w') as f:
                f.write(str(n_atoms) + 2 * '\n')
            frame.to_csv(
                outputfile,
                sep=' ',
                index=False,
                header=False,
                mode='a'
                )

    @classmethod
    def read(cls, inputfile, pythonic_index=False, get_bonds=True):
        """Reads a xyz file.
    
        Args:
            inputfile (str): 
            pythonic_index (bool):
    
        Returns:
            pd.DataFrame: 
        """
        xyz_frame = pd.read_table(
            inputfile,
            skiprows=2,
            comment='#',
            delim_whitespace=True,
            names=['atom', 'x', 'y', 'z']
            )
        if not pythonic_index:
            n_atoms = xyz_frame.shape[0]
            xyz_frame.index = range(1, n_atoms+1)
        molecule = cls(xyz_frame)
        if get_bonds:
            molecule.get_bonds()
        return molecule
