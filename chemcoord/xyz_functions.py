
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities
from . import zmat_functions
from . import settings


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
# From here till shown end panda wrappers are defined.
############################################################################
    def give_index(self):
        """This function returns the index of the underlying pd.DataFrame.
        Please note that you can't assign a new index. 
        Instead you need to write::

            cartesian_object.xyz_frame.index = new_index

        """ 
        return self.xyz_frame.index

#    def loc(self, row, column):
#        return Cartesian(self.xyz_frame.loc[row, column])

############################################################################
# end of pandas wrapper definition
############################################################################


    @staticmethod 
    def _overlap(frame, bond_size_dic):
        """Calculates the overlap of van der Vaals radii.

        Args:
            indices_of_other_atoms (list): The indices of atoms for which the overlap should be calculated.
            modified_properties (dic): If you want to change the van der Vaals 
                radius of one or more specific atoms, pass a dictionary that looks like::

                    modified_properties = {index1 : 1.5}

                For global changes use the constants.py module.
    
        Returns:
            dic: Dictionary mapping from an atom index to the indices of atoms bonded to.

        """
        convert_to = {
                'frame' : dict(zip(range(len(frame.index)), frame.index)),
                'array' : dict(zip(frame.index, range(len(frame.index))))
                }
        location_array = frame.loc[:, ['x', 'y', 'z']].get_values().astype(float)

        def summed_bond_size_array(bond_size_dic):
            """Returns a xyz_frame with a column for the distance from origin.
            """
            bond_size_vector = np.array([bond_size_dic[key] for key in frame.index])
            A = np.expand_dims(bond_size_vector, axis=1)
            B = np.expand_dims(bond_size_vector, axis=0)
            C = A + B
            return C

        bond_size_array = summed_bond_size_array(bond_size_dic)
        distance_array = utilities.give_distance_array(location_array)

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

        Args:
            modified_properties (dic): If you want to change the van der Vaals 
                radius or valency of one or more specific atoms, pass a dictionary that looks like::

                    modified_properties = {index1 : {'bond_size' : 1.5, 'valency' : 8}, ...}

                For global changes use the constants.py module.
    
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
            overlap_array, convert_to = self._overlap(self.xyz_frame.loc[index_of_cube, :], bond_size_dic)
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
                    get_bonds_local(self, bond_dic, valency_dic, bond_size_dic, use_valency, index_of_cube=cuboid_dic[key][0])
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

    def _divide_et_impera(self, maximum_edge_length=25, difference_edge=6):
        """Returns a molecule split into cuboids.
       
         
        Args:
            maximum_edge_length (float): 
            difference_edge (float): 
    
        Returns:
            dict: A dictionary mapping from a 3 tuple of integers to a 2 tuple of sets.
                The 3 tuple gives the integer numbered coordinates of cuboids.
                The first set contains the indices of atoms lying in the cube with an edge length of ``maximum_edge_length``.
                The second set contains the indices of atoms lying in the cube with ``maximum_edge_length + difference_edge``
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



# TODO can be deleted
    @staticmethod
    def _connected_to(index_of_atom, bond_dic, exclude=None):
        """No object overhead.
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



# TODO can be deleted
    def connected_to(self, index_of_atom, exclude=None):
        bond_dic = self.get_bonds(use_lookup=True)
        fragment_index = self._connected_to(index_of_atom, bond_dic, exclude=None)
        return Cartesian(self.xyz_frame.loc[fragment_index, :])

    def cutsphere(self, radius=15., origin=[0., 0., 0.], outside_sliced=True, preserve_bonds=False):
        """Cuts a sphere specified by origin and radius.
    
        Args:
            radius (float): 
            origin (list):
            outside_sliced (bool): Atoms outside/inside the sphere are cut out.
    
        Returns:
            pd.dataframe: Sliced xyz_frame
        """
        try:
            origin[0]
        except (TypeError, IndexError):
            origin = self.location(int(origin))

        ordered_molecule = self.distance_frame(origin)
        frame = ordered_molecule.xyz_frame
        if outside_sliced:
            sliced_xyz_frame = frame[frame['distance'] < radius]
        else:
            sliced_xyz_frame = frame[frame['distance'] > radius]
        

        if preserve_bonds:
            included_atoms_set = set(sliced_xyz_frame.index)
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

            sliced_xyz_frame = self.xyz_frame.loc[included_atoms_set, :]

        return self.__class__(sliced_xyz_frame)




    def cutcube(self, a=20, b=None, c=None, origin=[0, 0, 0], outside_sliced = True):
        """Cuts a cube specified by edge and radius.
    
        Args:
            a (float): Value of the a edge.
            b (float): Value of the b edge. Takes value of a if None.
            c (float): Value of the c edge. Takes value of a if None.
            origin (list):
            outside_sliced (bool): Atoms outside/inside the sphere are cut out.
    
        Returns:
            pd.dataframe: Sliced xyz_frame
        """
        b = a if b is None else b
        c = a if c is None else c
        xyz_frame = self.xyz_frame.copy()
        # Next line changes from python list to dictionary for easy access of the origin values
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
        return self.__class__(sliced_xyz_frame)


    def mass(self):
        """Gives several properties related to mass.
    
        Args:
            xyz_frame (pd.dataframe): 
    
        Returns:
            dic: The returned dictionary has four possible keys:
            
            ``frame_mass``: xyz_DataFrame with an additional column for the masses of each atom.
        
            ``total_mass``: The total mass.
    
            ``barycenter``: The mass weighted average location.
        
            ``topologic_center``: The average location.
        """
        xyz_frame = self.xyz_frame
        indexlist = list(xyz_frame.index)
        n_atoms = xyz_frame.shape[0]
    
        if 'mass' in xyz_frame.columns:
            xyz_frame_mass = xyz_frame
        else:
            masses_dic = dict(zip(
                    constants.atom_properties.keys(), 
                    [constants.atom_properties[atom]['mass'] for atom in constants.atom_properties.keys()]
                    ))
            masses = pd.Series([ masses_dic[atom] for atom in xyz_frame['atom']], name='mass', index=xyz_frame.index)
            xyz_frame_mass = pd.concat([xyz_frame, masses], axis=1, join='outer')
        
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
            ['xyz_frame_mass', 'total_mass', 'barycenter', 'topologic_center'], 
            [Cartesian(xyz_frame_mass), total_mass, barycenter, topologic_center]
            ))
        return dic_of_values


    def move(self, vector=[0, 0, 0], matrix=np.identity(3)):
        """Move an xyz_frame.
    
        The xyz_frame is first rotated, mirrored... by the matrix
        and afterwards translated by the vector
    
        Args:
            xyz_frame (pd.dataframe): 
            vector (np.array): default is np.zeros(3)
            matrix (np.array): default is np.identity(3)
    
        Returns:
            pd.dataframe: Moved xyz_frame
        """
        frame = self.xyz_frame.copy()
        vectors = frame.loc[:, ['x', 'y', 'z']].get_values().astype(float)
        frame.loc[:, ['x', 'y', 'z']] = np.transpose(np.dot(np.array(matrix), np.transpose(vectors)))
        vectors = frame.loc[:, ['x', 'y', 'z']].get_values().astype(float)
        frame.loc[:, ['x', 'y', 'z']] = vectors + np.array(vector)
        return self.__class__(frame)

    def distance(self, index, bond_with):
        """Return the distance between two atoms.
       
        Args:
            xyz_frame (pd.dataframe): 
            index (int): 
            bond_with (int): Index of atom bonding with.
    
        Returns:
            float: distance
        """
        vi, vb = self.location([index, bond_with])
        q = vb - vi
        distance = np.linalg.norm(q)
        return distance

    def _distance_optimized(self, buildlist, exclude_first = True):
        """Return the distances between given atoms.
    
        In order to know more about the buildlist, go to :func:`to_zmat`.
       
        Args:
            xyz_frame (pd.dataframe): 
            buildlist (list): 
            exclude_first (bool): The exclude_first option excludes the first row of the buildlist from calculation.
    
        Returns:
            list: List of the distance between the first and second atom of every entry in the buildlist. 
        """
        if exclude_first:
            temp_buildlist = copy.deepcopy(buildlist[1:])
        else:
            temp_buildlist = copy.deepcopy(buildlist)
    
        index_set = set([])
        for listelement in temp_buildlist:
            index_set |= set(listelement[:2])
        index_list = list(index_set)
        convert = dict(zip(index_list, range(len(index_list))))

        location_array = self.location(index_list)
    
        distance_list = []
        for listelement in temp_buildlist:
            index, bond_with = listelement[:2]
            vi, vb = location_array[[convert[index], convert[bond_with]], :]
            q = vb - vi
            distance = np.linalg.norm(q)
            distance_list.append(distance)
        return distance_list

    def angle_degrees(self, index, bond_with, angle_with):
        """Return the angle in dregrees between three atoms.
       
        Args:
            xyz_frame (pd.dataframe): 
            index (int): 
            bond_with (int): Index of atom bonding with.
            angle_with (int): Index of angle defining atom. 
    
        Returns:
            float: Angle in degrees.
        """
        vi, vb, va = self.location([index, bond_with, angle_with])
        BI, BA = vi - vb, va - vb
        angle = utilities.give_angle(BI, BA)
        return angle


    def _angle_degrees_optimized(self, buildlist, exclude_first = True):
        """Return the angles between given atoms.
    
        In order to know more about the buildlist, go to :func:`to_zmat`.
       
        Args:
            xyz_frame (pd.dataframe): 
            buildlist (list): 
            exclude_first (bool): The exclude_first option excludes the first two rows of the buildlist from calculation.
    
        Returns:
            list: List of the angle between the first, second and third atom of every entry in the buildlist. 
        """
        if exclude_first:
            temp_buildlist = copy.deepcopy(buildlist[2:])
        else:
            temp_buildlist = copy.deepcopy(buildlist)
    
        index_set = set([])
        for listelement in temp_buildlist:
            index_set |= set(listelement[:3])
        index_list = list(index_set)
        convert = dict(zip(index_list, range(len(index_list))))
        location_array = self.location(index_list)
    
        angle_list = []
        for listelement in temp_buildlist:
            index, bond_with, angle_with = listelement[:3]
            vi, vb, va = location_array[[convert[index], convert[bond_with], convert[angle_with]], :]
            BI, BA = vi - vb, va - vb
            angle = utilities.give_angle(BI, BA)
            angle_list.append(angle)
        return angle_list


    def dihedral_degrees(self, index, bond_with, angle_with, dihedral_with):
        """Return the angle in dregrees between three atoms.
       
        Args:
            xyz_frame (pd.dataframe): 
            index (int): 
            bond_with (int): Index of atom bonding with.
            angle_with (int): Index of angle defining atom. 
            dihedral_with (int): Index of dihedral defining atom. 
    
        Returns:
            float: dihedral in degrees.
        """
        vi, vb, va, vd = self.location([index, bond_with, angle_with, dihedral_with])
    
        DA = va - vd
        AB = vb - va
        BI = vi - vb
    
        n1 = utilities.normalize(np.cross(DA, AB))
        n2 = utilities.normalize(np.cross(AB, BI))
    
        dihedral = utilities.give_angle(n1, n2)
        if (dihedral != 0):
            dihedral = dihedral if 0 < np.dot(AB, np.cross(n1, n2)) else 360 - dihedral
        return dihedral


    def _dihedral_degrees_optimized(self, buildlist, exclude_first = True):
        """Return the angles between given atoms.
       
        In order to know more about the buildlist, go to :func:`to_zmat`.
    
        Args:
            xyz_frame (pd.dataframe): 
            buildlist (list): 
            exclude_first (bool): The exclude_first option excludes the first three rows of the buildlist from calculation.
    
        Returns:
            list: List of the dihedral between the first, second, third and fourth atom of every entry in the buildlist. 
        """
        if exclude_first:
            temp_buildlist = copy.deepcopy(buildlist[3:])
        else:
            temp_buildlist = copy.deepcopy(buildlist)
    
        index_set = set([])
        for listelement in temp_buildlist:
            index_set |= set(listelement[:4])
        index_list = list(index_set)
        convert = dict(zip(index_list, range(len(index_list))))
        location_array = self.location(index_list)
    
        dihedral_list = []
        for listelement in temp_buildlist:
            index, bond_with, angle_with, dihedral_with = listelement[:4]
            vi, vb, va, vd = location_array[[convert[index], convert[bond_with], convert[angle_with], convert[dihedral_with]], :]
    
            DA = va - vd
            AB = vb - va
            BI = vi - vb
        
            n1 = utilities.normalize(np.cross(DA, AB))
            n2 = utilities.normalize(np.cross(AB, BI))
        
            dihedral = utilities.give_angle(n1, n2)
            if (dihedral != 0):
                dihedral = dihedral if 0 < np.dot(AB, np.cross(n1, n2)) else 360 - dihedral
            dihedral_list.append(dihedral)
        return dihedral_list


    def get_fragment(self, list_of_indextuples, threshold=2.0):
        """Get the indices of the atoms in a fragment.
       
        The list_of_indextuples contains all bondings from the molecule to the fragment.
        ``[(1,3), (2,4)]`` means for example that the fragment is connected over two bonds.
        The first bond is from atom 1 in the molecule to atom 3 in the fragment.
        The second bond is from atom 2 in the molecule to atom 4 in the fragment.
        The threshold defines the maximum distance between two atoms in order to 
        be considered as connected.
    
        Args:
            xyz_frame (pd.dataframe): 
            list_of_indextuples (list): 
            threshold (float): 
    
        Returns:
            list: A list of the indices of the fragment.
        """
        # Preparation of frame
        prepared_molecule = self
        for tuple in list_of_indextuples:
            va, vb = self.location(tuple[0:2])
    
            BA = va - vb
            bond = np.linalg.norm(BA)
            new_center = vb + 2. * BA
            prepared_molecule = prepared_molecule.cutsphere(
                    radius = (1.5 * bond ),
                    origin= new_center,
                    outside_sliced = False
                    )
    
        fixed = set([])
        previous_found = set([tuple[1] for tuple in list_of_indextuples])
        just_found = set([])
        
        # Please note that "not ... < ... " indicates "previous_found is not a real subset of fixed". 
        # Because of perhaps disjoint elements there is not the possibility to just write:
        # previous_found > fixed
        # Mathematically speaking: sets only define partial ordering (subset relationships)
        while not previous_found < fixed:
            fixed |= previous_found
            for index in previous_found:
                new_center = self.location(index)
                # TODO perhaps use pd Wrapper for index
                just_found |= set(prepared_molecule.cutsphere(radius = threshold, origin = new_center).xyz_frame.index)
    
            previous_found = just_found - fixed
            just_found = set([])
    
        index_of_fragment_list = list(fixed)
        return index_of_fragment_list


# TODO from here on write into object methods using get_bonds
    def _order_of_building(self, to_be_built=None, already_built=None, recursion=2):
        frame = self.xyz_frame.copy()
        n_atoms = frame.shape[0]
    
        if already_built is None:
            already_built = []
    
        if to_be_built is None:
            to_be_built = list(frame.index)
        else:
            to_be_built = list(to_be_built)
    
        try:
            buildlist = copy.deepcopy(already_built)
            already_built = [element[0] for element in already_built]
        except:
            pass
    
        for element in already_built:
            to_be_built.remove(element)
        number_of_atoms_to_add = len(to_be_built)
        already_built = list(already_built)
        topologic_center = mass(frame)['topologic_center']
    
        assert recursion in set([0, 1, 2])
        assert (set(to_be_built) & set(already_built)) == set([])
    
    
        def zero_reference(topologic_center):
            frame_distance = distance_frame(frame.loc[to_be_built, :], topologic_center)
            index = frame_distance['distance'].idxmin()
            return index
    
        def one_reference(previous):
            previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']]
            frame_distance_previous = distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
            index = frame_distance_previous['distance'].idxmin()
            return index, previous
    
        def two_references(previous, before_previous):
            previous_atom, before_previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']], xyz_frame.ix[before_previous, ['x', 'y', 'z']]
            frame_distance_previous = distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
            frame_distance_before_previous = distance_frame(xyz_frame.loc[to_be_built, :], before_previous_atom)
            summed_distance = frame_distance_previous.loc[:, 'distance'] + frame_distance_before_previous.loc[:, 'distance']
            index = summed_distance.idxmin()
            return index, previous
    
    
        if len(already_built) > 1:
            previous, index = already_built[-2:]
    
        elif len(already_built) == 1:
            index = already_built[-1:]
    
        if recursion == 2:
            mode = 0
            for _ in range(number_of_atoms_to_add):
                if (len(already_built) > 1) or mode == 2:
                    index, previous = two_references(index, previous)
                    bond_length = distance(frame, index, previous)
                    if bond_length > 7:
                        index = zero_reference(topologic_center)
                        mode = 1
    
                elif (len(already_built) == 1) or mode == 1:
                    index, previous = one_reference(index)
                    mode = 2
    
                elif already_built == []:
                    index = zero_reference(topologic_center)
                    mode = 1
    
                already_built.append(index)
                to_be_built.remove(index)
    
        if recursion == 1:
            for _ in range(number_of_atoms_to_add):
                if len(already_built) > 0:
                    index, previous = one_reference(index)
                    bond_length = distance(frame, index, previous)
                    if bond_length > 5:
                        index = zero_reference(topologic_center)
    
                elif already_built == []:
                    index = zero_reference(topologic_center)
    
                already_built.append(index)
                to_be_built.remove(index)
    
        elif recursion == 0:
            already_built = already_built + to_be_built
    
        try:
            for index, element in enumerate(buildlist):
                already_built[index] = element
        except:
            pass
    
        already_built = list(already_built)
        return already_built


    # TODO Check for linearity and insert dummy atoms
    def _get_reference(xyz_frame, order_of_building, recursion = 1):
        xyz_frame = self.xyz_frame.copy()
    
        buildlist_given = copy.deepcopy(order_of_building)
        defined_entries = {}
        order_of_building = []
    
        for element in buildlist_given:
            if type(element) is list:
                defined_entries[element[0]] = element
                order_of_building.append(element[0])
            else:
                order_of_building.append(element)
    
        reference_list = []
    
        n_atoms = len(order_of_building)
        to_be_built, already_built = list(order_of_building), []
    
        if recursion > 0:
            def first_atom(index_of_atom):
                reference_list.append([index_of_atom])
                already_built.append(index_of_atom)
                to_be_built.remove(index_of_atom)
    
            def second_atom(index_of_atom):
                try:
                    reference_list.append(defined_entries[index_of_atom])
                    already_built.append(to_be_built.pop(0))
                except KeyError:
                    distances_to_other_atoms = distance_frame(
                            xyz_frame.loc[already_built, :],
                            self.location(index_of_atom)
                            )
                    bond_with = distances_to_other_atoms['distance'].idxmin()
                    reference_list.append([index_of_atom, bond_with])
                    already_built.append(to_be_built.pop(0))
    
            def third_atom(index_of_atom):
                try:
                    reference_list.append(defined_entries[index_of_atom])
                    already_built.append(to_be_built.pop(0))
                except KeyError:
                    distances_to_other_atoms = distance_frame(
                            xyz_frame.loc[already_built, :],
                            self.location(index_of_atom)
                            ).sort_values(by='distance')
                    bond_with, angle_with = list(distances_to_other_atoms.iloc[0:2, :].index)
                    reference_list.append([index_of_atom, bond_with, angle_with])
                    already_built.append(to_be_built.pop(0))
    
            def other_atom(index_of_atom):
                try:
                    reference_list.append(defined_entries[index_of_atom])
                    already_built.append(to_be_built.pop(0))
                except KeyError:
                    distances_to_other_atoms = distance_frame(
                            xyz_frame.loc[already_built, :],
                            self.location(index_of_atom)
                            ).sort_values(by='distance')
                    bond_with, angle_with, dihedral_with = list(distances_to_other_atoms.iloc[0:3, :].index)
                    reference_list.append([index_of_atom, bond_with, angle_with, dihedral_with])
                    already_built.append(to_be_built.pop(0))
    
            if n_atoms == 1:
                first_atom(to_be_built[0])
                second_atom(to_be_built[0])
                third_atom(to_be_built[0])
    
            elif n_atoms == 2:
                first_atom(to_be_built[0])
                second_atom(to_be_built[0])
    
            elif n_atoms == 3:
                first_atom(to_be_built[0])
                second_atom(to_be_built[0])
                third_atom(to_be_built[0])
    
            elif n_atoms > 3:
                first_atom(to_be_built[0])
                second_atom(to_be_built[0])
                third_atom(to_be_built[0])
    
                for _ in range(3, n_atoms):
                    other_atom(to_be_built[0])
    
        else:
            if n_atoms == 1:
                reference_list.append([order_of_building[0]])
    
            elif n_atoms == 2:
                reference_list.append([order_of_building[0]])
                reference_list.append([order_of_building[1], order_of_building[0]])
    
            elif n_atoms == 3:
                reference_list.append([order_of_building[0]])
                reference_list.append([order_of_building[1], order_of_building[0]])
                reference_list.append([order_of_building[2], order_of_building[1], order_of_building[0]])
    
            elif n_atoms > 3:
                reference_list.append([order_of_building[0]])
                reference_list.append([order_of_building[1], order_of_building[0]])
                reference_list.append([order_of_building[2], order_of_building[1], order_of_building[0]])
                for i in range(3, n_atoms):
                    reference_list.append([
                        order_of_building[i],
                        order_of_building[i-1],
                        order_of_building[i-2],
                        order_of_building[i-3]
                        ])
    
        return reference_list





    def _build_zmat(self, buildlist):
        # not necessary
        xyz_frame = self.xyz_frame
        n_atoms = len(buildlist)
        # Taking functions from other namespaces
        indexlist = [element[0] for element in buildlist]
        zmat_frame = pd.DataFrame(columns=['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'],
                dtype='float',
                index=indexlist
                )
        indexlist = [element[0] for element in buildlist]
        bond_with_list = [element[1] for element in buildlist[1:]]
        angle_with_list = [element[2] for element in buildlist[2:]]
        dihedral_with_list = [element[3] for element in buildlist[3:]]
    
        def add_first_atom():
            index = indexlist[0]
            zmat_frame.loc[index, 'atom'] = xyz_frame.loc[index, 'atom']
    
        def add_second_atom():
            index, bond_with = buildlist[1]
            bond_length = distance(xyz_frame, index, bond_with)
            zmat_frame.loc[index, 'atom':'bond'] = [xyz_frame.loc[index, 'atom'], bond_with, bond_length]
    
        def add_third_atom():
            index, bond_with, angle_with = buildlist[2]
            bond_length = distance(xyz_frame, index, bond_with)
            angle = angle_degrees(xyz_frame, index, bond_with, angle_with)
            zmat_frame.loc[index, 'atom':'angle'] = [
                    xyz_frame.loc[index, 'atom'],
                    bond_with, bond_length,
                    angle_with, angle
                    ]
    
    
        def add_atoms():
            distance_list = _distance_optimized(xyz_frame, buildlist)
            angle_list = _angle_degrees_optimized(xyz_frame, buildlist)
            dihedral_list = _dihedral_degrees_optimized(xyz_frame, buildlist)
    
            zmat_frame.loc[indexlist, 'atom'] = xyz_frame.loc[indexlist, 'atom']
            zmat_frame.loc[indexlist[1:], 'bond_with'] = bond_with_list
            zmat_frame.loc[indexlist[1:], 'bond'] = distance_list
            zmat_frame.loc[indexlist[2:], 'angle_with'] = angle_with_list
            zmat_frame.loc[indexlist[2:], 'angle'] = angle_list
            zmat_frame.loc[indexlist[3:], 'dihedral_with'] = dihedral_with_list
            zmat_frame.loc[indexlist[3:], 'dihedral'] = dihedral_list
    
    
        if n_atoms > 3:
            add_atoms()
    
        elif n_atoms == 1:
            add_first_atom()
    
        elif n_atoms == 2:
            add_first_atom()
            add_second_atom()
    
        elif n_atoms == 3:
            add_first_atom()
            add_second_atom()
            add_third_atom()
    
    
        return zmat_frame.loc[[element[0] for element in buildlist], :]


    # TODO Write Docstring and Tutorial
    # TODO Extract columns with additional information and append in the end again
    def to_zmat(xyz_frame, buildlist = None, fragments_list = None, recursion_level = 2):
        """Convert xyz_frame to zmat_frame.
        """
        assert recursion_level in set([0,1,2])
    
        if buildlist is None:
            buildlist = []
        if fragments_list is None:
            fragments_list = []
    
        fragments_old = copy.deepcopy(fragments_list)
    
        # preparing the fragments list
        fragments_new = []
        for fragment in fragments_old:
            for index in [element[0] for element in fragment[0:3]]:
                fragment.remove(index)
            fragment = fragment[0:3] + [[number] for number in fragment[3:]]
            fragments_new.append(fragment)
        fragments = fragments_new
    
        fragment_index_set = set([])
        for fragment in fragments:
            fragment_index = set([element[0] for element in fragment])
            assert fragment_index < set(xyz_frame.index)
            fragment_index_set |= fragment_index
    
        molecule_without_fragments_set = set(xyz_frame.index) - fragment_index_set
    
        assert set([element[0] for element in buildlist]) <= molecule_without_fragments_set
    
        # first build big molecule
        building_order = _order_of_building(
                xyz_frame.loc[molecule_without_fragments_set, :],
                already_built = buildlist,
                recursion = recursion_level
                )
        buildlist_for_big_molecule = _get_reference(
                xyz_frame.loc[molecule_without_fragments_set, :],
                order_of_building = building_order,
                recursion = recursion_level
                )
    
        zmat_big = _build_zmat(
                xyz_frame,
                buildlist_for_big_molecule
                )
    
    
    
        list_of_fragment_zmat = []
        for fragment in fragments:
            temp_buildlist = [fragment[0][:1], fragment[1][:2], fragment[2][:3]]
            fragment_index = [element[0] for element in fragment]
    
            building_order = _order_of_building(
                    xyz_frame.loc[fragment_index, :],
                    already_built = temp_buildlist,
                    recursion = recursion_level
                    )
            buildlist_for_fragment = _get_reference(
                    xyz_frame.loc[fragment_index, :],
                    order_of_building = building_order,
                    recursion = recursion_level
                    )
    
            zmat_fragment = _build_zmat(xyz_frame, buildlist_for_fragment)
    
            list_of_fragment_zmat.append((fragment[0:3], zmat_fragment))
    
        for reference_atoms, fragment_zmat in list_of_fragment_zmat:
            zmat_big = zmat_functions.concatenate(fragment_zmat, zmat_big, reference_atoms, xyz_frame)
    
        return zmat_big


    def inertia(self):
        """Calculates the inertia tensor and transforms along rotation axes.
    
        This function calculates the inertia tensor and returns a 4-tuple.
    
        Args:
            xyz_frame (pd.DataFrame): 
    
        Returns:
            dic: The returned dictionary has four possible keys:
            
            ``transformed_frame``:
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
        my_keys = ['frame_mass', 'total_mass', 'barycenter', 'topologic_center']
        my_dic = self.mass()
        frame_mass, total_mass, barycenter, topologic_center = [my_dic[key] for key in my_keys]
    
        frame_mass = frame_mass.move(vector = -barycenter)
    
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
        frame_mass = self.basistransform(frame_mass, new_basis, old_basis)
        
        dic_of_values =  dict(zip(
            ['transformed_frame', 'diag_inertia_tensor', 'inertia_tensor', 'eigenvectors'],
            [frame_mass, diag_inertia_tensor, inertia_tensor, eigenvectors]
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
            xyz_frame (pd.DataFrame): 
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
            frame = move(frame, matrix=np.transpose(basistransformation))
            test_basis = np.dot(np.transpose(basistransformation), new_basis)
        else:
            basistransformation = np.dot(new_basis, np.linalg.inv(old_basis))
            frame = move(frame, matrix=np.linalg.inv(basistransformation))
            test_basis = np.dot(np.linalg.inv(basistransformation), new_basis)
    
        assert np.isclose(test_basis, old_basis).all(), 'transformation did not work'
        return self.__class__(frame)



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
    
    
    def distance_frame(self, origin, indices_of_other_atoms=None):
        """Returns a xyz_frame with a column for the distance from origin.
        """
        if indices_of_other_atoms is None:
            indices_of_other_atoms = self.xyz_frame.index
        frame_distance = self.xyz_frame.loc[indices_of_other_atoms, :].copy()
        origin = np.array(origin, dtype=float)
        frame_distance['distance'] = np.linalg.norm(frame_distance.loc[:, ['x', 'y', 'z']].get_values().astype(float) - origin, axis =1)
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


#    @staticmethod
#    def _distance_frame(frame, origin, indices_of_other_atoms=None):
#        """Returns a xyz_frame with a column for the distance from origin.
#        """
#        if indices_of_other_atoms is None:
#            indices_of_other_atoms = self.xyz_frame.index
#        frame_distance = frame.copy()
#        origin = np.array(origin, dtype=float)
#        frame_distance['distance'] = np.linalg.norm(frame_distance.loc[:, ['x', 'y', 'z']].get_values().astype(float) - origin, axis =1)
#        return frame_distance

