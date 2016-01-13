'''
A module for manipulating and switching between zmatrices and cartesian coordinates.
'''

#import math as m
import numpy as np
import pandas as pd
import math as m
import copy
import constants


class read:
    @staticmethod
    def zmat(inputfile):
        """
        The input is a filename.
        The output is a zmat_DataFrame.
        """

        zmat_frame = pd.read_table(
            inputfile,
            comment='#',
            delim_whitespace=True,
            names=['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'],
            )
        # Changing to pythonic indexing
        zmat_frame['bond_with'] = zmat_frame['bond_with'] - 1
        zmat_frame['angle_with'] = zmat_frame['angle_with'] - 1
        zmat_frame['dihedral_with'] = zmat_frame['dihedral_with'] - 1
        return zmat_frame

    
    @staticmethod
    def xyz(inputfile):
        """
        The input is a filename.
        The output is a xyz_DataFrame.
        """
        xyz_frame = pd.read_table(
            inputfile,
            skiprows=2,
            comment='#',
            delim_whitespace=True,
            names=['atom', 'x', 'y', 'z']
            )
        return xyz_frame


class xyz_functions:
    """
    A collection of functions that operate on xyz_frames.
    """
    @staticmethod
    def sort(xyz_frame, origin=[0, 0, 0]):
        """
        The required input is a xyz_frame.
        The optional input is a list or np.array with length=3
        for the location of the desired origin. (default=[0, 0, 0])
        The output is a xyz_DataFrame with a column for the distance of each atom
        from the origin.
        The DataFrame is sorted by the values of the distance.
        """
        origin = dict(zip(['x', 'y', 'z'], list(origin)))
        xyz_frame['distance'] = np.sqrt(
            (xyz_frame['x'] - origin['x'])**2 +
            (xyz_frame['y'] - origin['y'])**2 +
            (xyz_frame['z'] - origin['z'])**2
            )
        sortedxyz_frame = xyz_frame.sort_values(by='distance')   #.reset_index(drop=True)
        return sortedxyz_frame


    @staticmethod
    def cutsphere(xyz_frame, radius=15, origin=[0, 0, 0], outside_sliced = True):
        """
        The required input is a xyz_frame.
        The optional input is a number for the radius (default=15) and
        a list or np.array with length=3 for the location of the desired origin (default=[0, 0, 0]).
        The output is a xyz_DataFrame where all atoms outside/inside the sphere defined by radius
        and origin are sliced out.
        """
        # Next line changes from ordinary python list to dictionary for easy access of the origin values
        origin = dict(zip(['x', 'y', 'z'], list(origin)))
        if outside_sliced:
            sliced_xyz_frame = xyz_frame[
                 ((xyz_frame['x'] - origin['x'])**2 +
                  (xyz_frame['y'] - origin['y'])**2 +
                  (xyz_frame['z'] - origin['z'])**2
                 ) < radius**2
                ]
        else:
            sliced_xyz_frame = xyz_frame[
                 ((xyz_frame['x'] - origin['x'])**2 +
                  (xyz_frame['y'] - origin['y'])**2 +
                  (xyz_frame['z'] - origin['z'])**2
                 ) > radius**2
                ]
        sliced_xyz_frame = sliced_xyz_frame
        return sliced_xyz_frame


    @staticmethod
    def cutcube(xyz_frame, edge=20, origin=[0, 0, 0]):
        """
        The required input is a xyz_frame.
        The optional input is a number for the edge (default=20) and
        a list or np.array with length=3 for the location of the desired origin (default=[0, 0, 0]).
        The output is a xyz_DataFrame where all atoms outside the cube defined by edge
        and origin are sliced out.
        """
        # Next line changes from ordinary python list to dictionary for easy access of the origin values
        origin = dict(zip(['x', 'y', 'z'], list(origin)))
        sliced_xyz_frame = xyz_frame[
               (np.abs((xyz_frame['x'] - origin['x'])) < edge / 2)
             & (np.abs((xyz_frame['y'] - origin['y'])) < edge / 2)
             & (np.abs((xyz_frame['z'] - origin['z'])) < edge / 2)
            ]
        sliced_xyz_frame = sliced_xyz_frame
        return sliced_xyz_frame


    @staticmethod
    def mass(xyz_frame):
        """
        The input is a xyz_DataFrame.
        Returns a tuple of four values.
        The first one is the zmat_DataFrame with an additional column for the masses of each atom.
        The second value is the total mass.
        The third value is the location of the barycentrum.
        The forth value is the location of the topologic center.
        """
        frame = xyz_frame.copy()
        indexlist = list(xyz_frame.index)
        n_atoms = frame.shape[0]

        masses_dic = constants.elementary_masses
        masses = pd.Series([ masses_dic[atom] for atom in frame['atom']], name='mass', index=frame.index)
        total_mass = masses.sum()
        frame_mass = pd.concat([frame, masses], axis=1, join='inner')

        location_array = frame_mass.loc[indexlist, ['x', 'y', 'z']].get_values().astype(float)

        baryzentrum = np.zeros([3])
        topologic_center = np.zeros([3])

        for row, index in enumerate(indexlist):
            baryzentrum = baryzentrum + location_array[row] * frame_mass.at[index, 'mass']
            topologic_center = topologic_center + location_array[row]
        baryzentrum = baryzentrum / total_mass
        topologic_center = topologic_center / n_atoms
        return frame_mass, total_mass, baryzentrum, topologic_center


    @staticmethod
    def move(xyz_frame, vector = [0, 0, 0], matrix = np.identity(3)):
        """
        The required input is a xyz_DataFrame.
        Optional input is a vector (default=[0, 0, 0]) and a matrix (default=np.identity(3))
        Returns a xyz_DataFrame that is first rotated, mirrored... by the matrix
        and afterwards moved by the vector
        """
        frame = xyz_frame.copy()
        vectors = frame.loc[:, ['x', 'y', 'z']].get_values().astype(float)
        frame.loc[:, ['x', 'y', 'z']] = np.transpose(np.dot(np.array(matrix), np.transpose(vectors)))
        vectors = frame.loc[:, ['x', 'y', 'z']].get_values().astype(float)
        frame.loc[:, ['x', 'y', 'z']] = vectors + np.array(vector)
        return frame



    @staticmethod
    def distance(frame, index, bond_with):
        """
        The input is the own index, the index of the atom bonding to and a xyz_DataFrame.
        Returns the distance between these atoms.
        """
        vi, vb = frame.ix[[index, bond_with], ['x', 'y', 'z']].get_values().astype(float)
        q = vb - vi
        distance = np.linalg.norm(q)
        return distance


    @staticmethod
    def _distance_optimized(location_array, buildlist, indexlist = None, exclude_first = True):
        """
        The input is a np.array with three columns for the x, y and z coordinates and an arbitrary number of rows.
        Since the np.array does not contain the indices of the pd.DataFrame anymore it is necessary to 
        pass an indexlist with which the location_array was created.
        Usually this is something like: location_array = xyz_frame.ix[indexlist, ['x', 'y', 'z']].get_values().astype(float).
        If the indices contained in the buildlist are the same as the indexlist (default) the indexlist can be omitted.
        The exclude_first options excludes the first row of the buildlist from calculation.
        Returns a list of the distance between the first and second atom of every entry in the buildlist. 
        """
        if indexlist is None:
            indexlist = [element[0] for element in buildlist]

        n_atoms = len(indexlist)
        convert_index = dict(zip(indexlist, range(n_atoms)))
        converted_buildlist = [[convert_index[number] for number in listelement] for listelement in buildlist]
        converted_buildlist = converted_buildlist[1:] if exclude_first else converted_buildlist

        distance_list = []
        if exclude_first:
            for atom in converted_buildlist:
                vi, vb = location_array[atom][:2]
                q = vb - vi
                distance = np.linalg.norm(q)
                distance_list.append(distance)
        else:
            for atom in converted_buildlist:
                vi, vb = location_array[atom][:2]
                q = vb - vi
                distance = np.linalg.norm(q)
                distance_list.append(distance)

        return distance_list

    @staticmethod
    def angle_degrees(frame, index, bond_with, angle_with):
        """
        The input is the own index, the index of the atom bonding to,
        the index of the angle defining atom and a xyz_DataFrame.
        Returns the angle between these atoms in degrees.
        """
        normalize = _utilities.normalize

        vi, vb, va = frame.ix[[index, bond_with, angle_with], ['x', 'y', 'z']].get_values().astype(float)

        BI = vi - vb
        BA = va - vb
        bi = normalize(BI)
        ba = normalize(BA)

        # Is this ok
        scalar_product = np.dot(bi, ba)
        if  -1.00000000000001 < scalar_product < -1.:
            scalar_product = -1.

        elif 1.00000000000001 > scalar_product > 1.:
            scalar_product = 1.

        angle = m.acos(scalar_product)
        angle = np.degrees(angle)
        return angle


    @staticmethod
    def _angle_degrees_optimized(location_array, buildlist, indexlist = None, exclude_first = True):
        """
        The input is a np.array with three columns for the x, y and z coordinates and an arbitrary number of rows.
        Since the np.array does not contain the indices of the pd.DataFrame anymore it is necessary to 
        pass an indexlist with which the location_array was created.
        Usually this is something like: location_array = xyz_frame.ix[indexlist, ['x', 'y', 'z']].get_values().astype(float).
        If the indices contained in the buildlist are the same as the indexlist (default) the indexlist can be omitted.
        The exclude_first options excludes the first two row of the buildlist from calculation.
        Returns a list of the angle between the first, second and third atom of every entry in the buildlist. 
        """
        normalize = _utilities.normalize
        if indexlist is None:
            indexlist = [element[0] for element in buildlist]
        n_atoms = len(indexlist)
        convert_index = dict(zip(indexlist, range(n_atoms)))
        converted_buildlist = [[convert_index[number] for number in listelement] for listelement in buildlist]
        converted_buildlist = converted_buildlist[2:] if exclude_first else converted_buildlist


        angle_list = []

        for atom in converted_buildlist:
            vi, vb, va = location_array[atom][:3]

            BI = vi - vb
            BA = va - vb
            bi = normalize(BI)
            ba = normalize(BA)

            # Is this ok
            scalar_product = np.dot(bi, ba)
            if  -1.00000000000001 < scalar_product < -1.:
                scalar_product = -1.

            elif 1.00000000000001 > scalar_product > 1.:
                scalar_product = 1.

            angle = m.acos(scalar_product)
            angle = np.degrees(angle)

            angle_list.append(angle)

        return angle_list


    @staticmethod
    def dihedral_degrees(frame, index, bond_with, angle_with, dihedral_with):
        """
        The input is the own index, the index of the atom bonding to,
        the index of the angle defining atom, the index of the dihedral defining atom and a xyz_DataFrame.
        Returns the dihedral between these atoms in degrees.
        """
        normalize = _utilities.normalize
        vi, vb, va, vd = frame.ix[[index, bond_with, angle_with, dihedral_with], ['x', 'y', 'z']].get_values().astype(float)

        DA = va - vd
        AB = vb - va
        BI = vi - vb

        n1 = normalize(np.cross(DA, AB))
        n2 = normalize(np.cross(AB, BI))

        # Is this ok
        scalar_product = np.dot(n1, n2)
        if  -1.00000000000001 < scalar_product < -1.:
            scalar_product = -1.

        elif 1.00000000000001 > scalar_product > 1.:
            scalar_product = 1.

        dihedral = m.acos(scalar_product)
        dihedral = np.degrees(dihedral)

        if (dihedral != 0):
            dihedral = dihedral if 0 < np.dot(AB, np.cross(n1, n2)) else 360 - dihedral

        return dihedral


    @staticmethod
    def _dihedral_degrees_optimized(location_array, buildlist, indexlist = None, exclude_first = True):
        """
        The input is a np.array with three columns for the x, y and z coordinates and an arbitrary number of rows.
        Since the np.array does not contain the indices of the pd.DataFrame anymore it is necessary to 
        pass an indexlist with which the location_array was created.
        Usually this is something like: location_array = xyz_frame.ix[indexlist, ['x', 'y', 'z']].get_values().astype(float).
        If the indices contained in the buildlist are the same as the indexlist (default) the indexlist can be omitted.
        The exclude_first options excludes the first three row of the buildlist from calculation.
        Returns a list of the dihedral between the first, second, third and fourth atom of every entry in the buildlist. 
        """
        normalize = _utilities.normalize
        if indexlist is None:
            indexlist = [element[0] for element in buildlist]
        n_atoms = len(indexlist)
        convert_index = dict(zip(indexlist, range(n_atoms)))
        converted_buildlist = [[convert_index[number] for number in listelement] for listelement in buildlist]
        converted_buildlist = converted_buildlist[3:] if exclude_first else converted_buildlist

        dihedral_list = []

        for atom in converted_buildlist:
            vi, vb, va, vd = location_array[atom]

            DA = va - vd
            AB = vb - va
            BI = vi - vb

            n1 = normalize(np.cross(DA, AB))
            n2 = normalize(np.cross(AB, BI))

            # Is this ok
            scalar_product = np.dot(n1, n2)
            if  -1.00000000000001 < scalar_product < -1.:
                scalar_product = -1.

            elif 1.00000000000001 > scalar_product > 1.:
                scalar_product = 1.

            dihedral = m.acos(scalar_product)
            dihedral = np.degrees(dihedral)

            if (dihedral != 0):
                dihedral = dihedral if 0 < np.dot(AB, np.cross(n1, n2)) else 360 - dihedral

            dihedral_list.append(dihedral)
        return dihedral_list





    @staticmethod
    def get_fragment(xyz_frame, list_of_indextuples, threshold = 2.0):
        """
        The input is the xyz_frame of the molecule to be fragmented.
        The list_of_indextuples contains all bondings from the molecule to the fragment.
        [(1,3), (2,4)] means for example that the fragment is connected over two bonds.
        The first bond is from atom 1 in the molecule to atom 3 in the fragment.
        The second bond is from atom 2 in the molecule to atom 4 in the fragment.
        The threshold defines the maximum distance between two atoms in order to 
        be considered as connected.
        Returns a list of the indices of the fragment.
        """
        frame = xyz_frame.copy()
        frame_index = frame.index

        # Preparation of frame
        for tuple in list_of_indextuples:
            va = np.array(xyz_frame.loc[tuple[0], ['x', 'y', 'z']], dtype = float)
            vb = np.array(xyz_frame.loc[tuple[1], ['x', 'y', 'z']], dtype = float)

            BA = va - vb
            bond = np.linalg.norm(BA)
            new_center = vb + 2. * BA
            frame = xyz_functions.cutsphere(frame, radius = (1.5 * bond ), origin= new_center, outside_sliced = False)

        fixed = set([])
        previous_found = set([tuple[1] for tuple in list_of_indextuples])
        just_found = set([])


        # Preparation for search

        while not previous_found < fixed:
            fixed |= previous_found
            for index in previous_found:
                new_center = np.array(frame.loc[index, ['x', 'y', 'z']], dtype = float)
                just_found |= set(xyz_functions.cutsphere(frame, radius = threshold, origin = new_center).index)

            previous_found = just_found - fixed
            just_found = set([])

        index_of_fragment_list = list(fixed)

        return index_of_fragment_list

    @staticmethod
    def _order_of_building(xyz_frame, to_be_built = None, already_built = None, recursion = 2):
        frame = xyz_frame.copy()
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
        topologic_center = xyz_functions.mass(frame)[3]

        assert recursion in set([0, 1, 2])
        assert (set(to_be_built) & set(already_built)) == set([])


        def zero_reference(topologic_center):
            frame_distance = _utilities.distance_frame(frame.loc[to_be_built, :], topologic_center)
            index = frame_distance['distance'].idxmin()
            return index

        def one_reference(previous):
            previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']]
            frame_distance_previous = _utilities.distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
            index = frame_distance_previous['distance'].idxmin()
            return index, previous

        def two_references(previous, before_previous):
            previous_atom, before_previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']], xyz_frame.ix[before_previous, ['x', 'y', 'z']]
            frame_distance_previous = _utilities.distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
            frame_distance_before_previous = _utilities.distance_frame(xyz_frame.loc[to_be_built, :], before_previous_atom)
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
                    distance = xyz_functions.distance(frame, index, previous)
                    if distance > 5:
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
                    distance = xyz_functions.distance(frame, index, previous)
                    if distance > 5:
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

    def _get_reference(xyz_frame, order_of_building, recursion = True):

        buildlist = copy.deepcopy(order_of_building)
        index_of_reference_items = []
        order_of_building = []

        for index, element in enumerate(buildlist):
            if type(element) is list:
                order_of_building.append(element[0])
                index_of_reference_items.append(index)
            else:
                order_of_building.append(element)

        indexlist = []

        n_atoms = len(order_of_building)
        to_be_built, already_built = list(order_of_building), []

        if recursion > 0:

            def first_atom(index_of_atom):
                indexlist.append([index_of_atom])
                already_built.append(index_of_atom)
                to_be_built.remove(index_of_atom)

            def second_atom(index_of_atom):
                distances_to_other_atoms = _utilities.distance_frame(
                        xyz_frame.loc[already_built, :],
                        xyz_frame.loc[index_of_atom, ['x', 'y', 'z']]
                        )
                bond_with = distances_to_other_atoms['distance'].idxmin()
                indexlist.append([index_of_atom, bond_with])
                already_built.append(to_be_built.pop(0))

            def third_atom(index_of_atom):
                distances_to_other_atoms = _utilities.distance_frame(
                        xyz_frame.loc[already_built, :],
                        xyz_frame.loc[index_of_atom, ['x', 'y', 'z']]
                        ).sort_values(by='distance')
                bond_with, angle_with = list(distances_to_other_atoms.iloc[0:2, :].index)
                indexlist.append([index_of_atom, bond_with, angle_with])
                already_built.append(to_be_built.pop(0))

            def other_atom(index_of_atom):
                distances_to_other_atoms = _utilities.distance_frame(
                        xyz_frame.loc[already_built, :],
                        xyz_frame.loc[index_of_atom, ['x', 'y', 'z']]
                        ).sort_values(by='distance')
                bond_with, angle_with, dihedral_with = list(distances_to_other_atoms.iloc[0:3, :].index)
                indexlist.append([index_of_atom, bond_with, angle_with, dihedral_with])
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
                indexlist.append([order_of_building[0]])

            elif n_atoms == 2:
                indexlist.append([order_of_building[0]])
                indexlist.append([order_of_building[1], order_of_building[0]])

            elif n_atoms == 3:
                indexlist.append([order_of_building[0]])
                indexlist.append([order_of_building[1], order_of_building[0]])
                indexlist.append([order_of_building[2], order_of_building[1], order_of_building[0]])

            elif n_atoms > 3:
                indexlist.append([order_of_building[0]])
                indexlist.append([order_of_building[1], order_of_building[0]])
                indexlist.append([order_of_building[2], order_of_building[1], order_of_building[0]])
                for i in range(3, n_atoms):
                    indexlist.append([
                        order_of_building[i],
                        order_of_building[i-1],
                        order_of_building[i-2],
                        order_of_building[i-3]
                        ])


        for index in index_of_reference_items:
                indexlist[index] = buildlist[index]

        return indexlist



    @staticmethod
    def _build_zmat(xyz_frame, buildlist):
        # not necessary
        n_atoms = len(buildlist)
        # Taking functions from other namespaces
        indexlist = [element[0] for element in buildlist]
        distance = xyz_functions.distance
        angle_degrees = xyz_functions.angle_degrees
        dihedral_degrees = xyz_functions.dihedral_degrees
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
            location_array = xyz_frame.ix[indexlist, ['x', 'y', 'z']].get_values().astype(float)
            distance_list = xyz_functions._distance_optimized(location_array, buildlist)
            angle_list = xyz_functions._angle_degrees_optimized(location_array, buildlist)
            dihedral_list = xyz_functions._dihedral_degrees_optimized(location_array, buildlist)

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


    @staticmethod
    def to_zmat(xyz_frame, buildlist = None, fragments_list = None, recursion_level = 2):
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
        building_order = xyz_functions._order_of_building(
                xyz_frame.loc[molecule_without_fragments_set, :],
                already_built = buildlist,
                recursion = recursion_level
                )
        buildlist_for_big_molecule = xyz_functions._get_reference(
                xyz_frame.loc[molecule_without_fragments_set, :],
                order_of_building = building_order,
                recursion = recursion_level
                )

        zmat_big = xyz_functions._build_zmat(
                xyz_frame,
                buildlist_for_big_molecule
                )



        list_of_fragment_zmat = []
        for fragment in fragments:
            temp_buildlist = [fragment[0][:1], fragment[1][:2], fragment[2][:3]]
            fragment_index = [element[0] for element in fragment]

            building_order = xyz_functions._order_of_building(
                    xyz_frame.loc[fragment_index, :],
                    already_built = temp_buildlist,
                    recursion = recursion_level
                    )
            buildlist_for_fragment = xyz_functions._get_reference(
                    xyz_frame.loc[fragment_index, :],
                    order_of_building = building_order,
                    recursion = recursion_level
                    )
            zmat_fragment = xyz_functions._build_zmat(xyz_frame, buildlist_for_fragment)

            list_of_fragment_zmat.append((fragment[0:3], zmat_fragment))

        for reference_atoms, fragment_zmat in list_of_fragment_zmat:
            zmat_big = zmat_functions.concatenate(fragment_zmat, zmat_big, reference_atoms, xyz_frame)

        return zmat_big


    @staticmethod
    def _xyz_to_zmat2(xyz_frame):
        """
        The input is a xyz_DataFrame.
        The output is a zmat_DataFrame.
        """
        n_atoms = xyz_frame.shape[0]
        zmat_frame = pd.DataFrame(columns=['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'],
                dtype='float',
                index=range(n_atoms)
                )
        to_be_built = list(range(n_atoms))
        already_built = []

        # Taking functions from other namespaces
        distance = xyz_functions.distance
        angle_degrees = xyz_functions.angle_degrees
        dihedral_degrees = xyz_functions.dihedral_degrees
        distance_frame = _utilities.distance_frame


        def add_first_atom():
            topologic_center = xyz_functions.mass(xyz_frame)[3]
            frame_distance = distance_frame(xyz_frame, topologic_center)
            index = frame_distance['distance'].idxmin()
            # Change of nonlocal variables
            zmat_frame.loc[index, 'atom'] = xyz_frame.loc[index, 'atom']
            already_built.append(index)
            to_be_built.remove(index)
            return index, topologic_center


        def add_second_atom(previous):
            # Identification of the index of the new atom; depending on previous atom
            previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']]
            frame_distance_previous = distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
            index = frame_distance_previous['distance'].idxmin()

            bond_with = previous
            bond_length = frame_distance_previous.loc[index, 'distance']
            # Change of nonlocal variables
            zmat_frame.loc[index, 'atom':'bond'] = [xyz_frame.loc[index, 'atom'], bond_with, bond_length]
            already_built.append(index)
            to_be_built.remove(index)
            return index, previous

        def add_third_atom(previous, before_previous):
            # Identification of the index of the new atom; depending on previous and before_previous atom
            previous_atom, before_previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']], xyz_frame.ix[before_previous, ['x', 'y', 'z']]
            frame_distance_previous = distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
            frame_distance_before_previous = distance_frame(xyz_frame.loc[to_be_built, :], before_previous_atom)
            summed_distance = frame_distance_previous.loc[:, 'distance'] + frame_distance_before_previous.loc[:, 'distance']
            index = summed_distance.idxmin()
            distances_to_other_atoms = distance_frame(xyz_frame.loc[already_built, :], xyz_frame.loc[index, ['x', 'y', 'z']]).sort_values(by='distance')
            bond_with, angle_with = distances_to_other_atoms.iloc[0:2, :].index

            bond_length = distances_to_other_atoms.at[bond_with, 'distance']
            angle = angle_degrees(xyz_frame, index, bond_with, angle_with)
            # Change of nonlocal variables
            zmat_frame.loc[index, 'atom':'angle'] = [
                    xyz_frame.loc[index, 'atom'],
                    bond_with, bond_length,
                    angle_with, angle
                    ]
            already_built.append(index)
            to_be_built.remove(index)
            return index, previous

        def add_atom(previous, before_previous, topologic_center):
            # Identification of the index of the new atom; depending on previous and before_previous atom
            previous_atom, before_previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']], xyz_frame.ix[before_previous, ['x', 'y', 'z']]
            frame_distance_previous = distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
            frame_distance_before_previous = distance_frame(xyz_frame.loc[to_be_built, :], before_previous_atom)
            summed_distance = frame_distance_previous.loc[:, 'distance'] + frame_distance_before_previous.loc[:, 'distance']
            index = summed_distance.idxmin()
            distances_to_other_atoms = distance_frame(xyz_frame.loc[already_built, :], xyz_frame.loc[index, ['x', 'y', 'z']]).sort_values(by='distance')
            # distances_to_other_atoms = distance_frame(xyz_frame.loc[index, ['x', 'y', 'z']], xyz_frame.loc[already_built, :]).sort_values(by='distance')
            bond_with, angle_with, dihedral_with = distances_to_other_atoms.iloc[0:3, :].index

            # Calculating the variables
            bond_length = distances_to_other_atoms.loc[bond_with, 'distance']
            angle = angle_degrees(xyz_frame, index, bond_with, angle_with)
            dihedral = dihedral_degrees(xyz_frame, index, bond_with, angle_with, dihedral_with)

            # Change of nonlocal variables
            zmat_frame.loc[index, 'atom':'dihedral'] = [
                    xyz_frame.loc[index, 'atom'],
                    bond_with, bond_length,
                    angle_with, angle,
                    dihedral_with, dihedral
                    ]
            already_built.append(index)
            to_be_built.remove(index)
            return index, previous


        if n_atoms == 1:
            previous, topologic_center = add_first_atom()

        elif n_atoms == 2:
            previous, topologic_center = add_first_atom()
            previous, before_previous = add_second_atom(previous)

        elif n_atoms == 3:
            previous, topologic_center = add_first_atom()
            previous, before_previous = add_second_atom(previous)
            previous, before_previous = add_third_atom(previous, before_previous)

        elif n_atoms > 3:
            previous, topologic_center = add_first_atom()
            previous, before_previous = add_second_atom(previous)
            previous, before_previous = add_third_atom(previous, before_previous)
            for _ in range(n_atoms - 3):
                previous, before_previous = add_atom(previous, before_previous, topologic_center)

        zmat_frame = zmat_frame.loc[already_built, : ]
        return zmat_frame

    @staticmethod
    def make_similar(xyz_frame1, xyz_frame2):
        """
        Takes two xyz_DataFrames and returns a reindexed copy of xyz_frame2
        which minimizes the necessary movements to get from xyz_frame1 to xyz_frame2.
        """
        frame1, frame2 = xyz_frame1.copy(), xyz_frame2.copy()
        assert set(frame1['atom']) == set(frame2['atom'])
        atomset = set(frame1['atom'])
        framedic = {}

        distance_frame = _utilities.distance_frame

        for atom in atomset:
            framedic[atom] = (frame1[frame1['atom'] == atom], frame2[frame2['atom'] == atom])
            assert framedic[atom][0].shape[0] == framedic[atom][1].shape[0]

        list_of_new_indexed_frames = []
        for atom in atomset:
            index_dic = {}

            for index1 in framedic[atom][0].index:
                location_of_reference = np.array(framedic[atom][0].loc[index1, ['x', 'y', 'z']], dtype=float)
                new_distance_frame = distance_frame(framedic[atom][1], location_of_reference)
                index2 = new_distance_frame['distance'].idxmin()
                index_dic[index2] = index1

            new_index = [index_dic[old_index2] for old_index2 in framedic[atom][1].index]
            framedic[atom][1].index = new_index
            list_of_new_indexed_frames.append(framedic[atom][1])

            xyz_frame3 = pd.concat(list_of_new_indexed_frames)


        return xyz_frame3.sort_index()

    @staticmethod
    def changes_from_to_xyz(xyz_frame1, xyz_frame2, step=5):
        """
        This function returns a list of xyz_frames with the subsequent 
        movement from xyz_frame1 to xyz_frame2.
        The list contains xyz_frame1 as first and xyz_frame2 as last element.
        Please note, that for this reason: len(list) = (step + 1).
        """
        xyzframe1 = xyzframe1.copy()
        xyzframe2 = xyzframe2.copy()

        difference = xyzframe2.copy()
        difference.loc[:, ['x', 'y', 'z']] = xyzframe2.loc[:, ['x', 'y', 'z']] - (
                xyzframe1.loc[:, ['x', 'y', 'z']]
                )

        step_frame = difference.copy()
        step_frame.loc[:, ['x', 'y', 'z']]  = step_frame.loc[:, ['x', 'y', 'z']] / step

        list_of_xyzframes = []
        temp_xyz = xyzframe1.copy()

        for t in range(steps + 1):
            temp_xyz.loc[:, ['x', 'y', 'z']] = xyzframe1.loc[:, ['x', 'y', 'z']] + (
                        step_frame.loc[:, ['x', 'y', 'z']] * t
                        )
            appendframe = temp_xyz.copy()
            list_of_xyzframes.append(appendframe)
        return list_of_xyzframes




class zmat_functions:
    """
    A collection of functions that operate on zmat_frames.
    """
    @staticmethod
    def mass(frame):
        """
        The input is a zmat_DataFrame.
        Returns a tuple of two values.
        The first one is the zmat_DataFrame with an additional column for the masses of each atom.
        The second value is the total mass.
        """
        zmat_frame = frame.copy()
        masses_dic = constants.elementary_masses
        masses = pd.Series([ masses_dic[atom] for atom in zmat_frame['atom']], name='masses')
        total_mass = masses.sum()
        zmat_frame_mass = pd.concat([zmat_frame, masses], axis=1, join='inner')
        return zmat_frame_mass, total_mass


    @staticmethod
    def change_numbering(frame, new_index = None):
        """
        The input is a zmat_DataFrame.
        Changes the numbering of index and all dependent numbering (bond_with...) to a new_index.
        The user has to make sure that the new_index consists of distinct elements.
        By default the new_index are the natural number ascending from 0.
        """
        zmat_frame = frame.copy()
        old_index = list(zmat_frame.index)
        new_index = list(range(zmat_frame.shape[0])) if (new_index == None) else list(new_index)
        assert len(new_index) == len(old_index)
        zmat_frame.index = new_index
        zmat_frame.loc[:, ['bond_with', 'angle_with', 'dihedral_with']] = zmat_frame.loc[
                :, ['bond_with', 'angle_with', 'dihedral_with']].replace(old_index, new_index)
        return zmat_frame


    @staticmethod
    def build_list(zmat_frame):
        """
        This functions outputs the buildlist required to reproduce
        building of the zmat_frame.
        """
        zmat = zmat_frame.copy()
        n_atoms = zmat.shape[0]


        zmat.insert(0, 'temporary_index', zmat.index)
        array_of_values = zmat.loc[:, ['temporary_index', 'bond_with', 'angle_with', 'dihedral_with']].get_values().astype(int)

        temporary_list1 = [[array_of_values[0, 0]],  list(array_of_values[1, 0:2]), list(array_of_values[2, 0:3])]
        temporary_list2 = [list(vector) for vector in array_of_values[3:]]

        buildlist = temporary_list1 + temporary_list2

        return buildlist


    @staticmethod
    def zmat_to_xyz(zmat):
        """
        The input is a zmat_DataFrame.
        The output is a xyz_DataFrame.
        """
        n_atoms = zmat.shape[0]
        xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'],
                dtype='float',
                index=range(n_atoms)
                )
        to_be_built = list(zmat.index)
        already_built = []

        normalize = _utilities.normalize

        def rotation_matrix(axis, angle):
            '''Euler-Rodrigues formula for rotation matrix. Input angle in radians.'''
            # Normalize the axis
            axis = normalize(np.array(axis))
            a = np.cos( angle/2 )
            b,c,d = - axis * np.sin(angle/2)
            rot_matrix = np.array( [[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]]
                )
            return rot_matrix


        def add_first_atom():
            index = to_be_built[0]
            # Change of nonlocal variables
            xyz_frame.loc[index] = [zmat.at[index, 'atom'], 0., 0., 0.]
            already_built.append(to_be_built.pop(0))

        def add_second_atom():
            index = to_be_built[0]
            atom, bond = zmat.loc[index, ['atom', 'bond']]
            # Change of nonlocal variables
            xyz_frame.loc[index] = [atom, bond, 0., 0. ]
            already_built.append(to_be_built.pop(0))


        def add_third_atom():
            index = to_be_built[0]
            atom, bond, angle = zmat.loc[index, ['atom', 'bond', 'angle']]
            angle = m.radians(angle)
            bond_with, angle_with = zmat.loc[index, ['bond_with', 'angle_with']]
            bond_with, angle_with = map(int, (bond_with, angle_with))


            vb = np.array(xyz_frame.loc[bond_with, ['x', 'y', 'z']], dtype=float)
            va = np.array(xyz_frame.loc[angle_with, ['x', 'y', 'z']], dtype=float)

            # Vector pointing from vb to va
            BA = va - vb

            # Vector of length distance pointing along the x-axis
            d = bond * normalize(BA)

            # Rotate d by the angle around the z-axis
            d = np.dot(rotation_matrix([0, 0, 1], angle), d)

            # Add d to the position of q to get the new coordinates of the atom
            p = vb + d

            # Change of nonlocal variables
            xyz_frame.loc[index] = [atom] + list(p)
            already_built.append(to_be_built.pop(0))


        def add_atom():

            index = to_be_built[0]
            atom, bond, angle, dihedral = zmat.loc[index, ['atom', 'bond', 'angle', 'dihedral']]
            angle, dihedral = map(m.radians, (angle, dihedral))
            bond_with, angle_with, dihedral_with = zmat.loc[index, ['bond_with', 'angle_with', 'dihedral_with']]
            bond_with, angle_with, dihedral_with = map(int, (bond_with, angle_with, dihedral_with))


            vb = np.array(xyz_frame.loc[bond_with, ['x', 'y', 'z']], dtype=float)
            va = np.array(xyz_frame.loc[angle_with, ['x', 'y', 'z']], dtype=float)
            vd = np.array(xyz_frame.loc[dihedral_with, ['x', 'y', 'z']], dtype=float)

            AB = vb - va
            DA = vd - va

            n1 = normalize(np.cross(DA, AB))
            ab = normalize(AB)

            # Vector of length distance pointing along the x-axis
            d = bond * -ab

            # Rotate d by the angle around the n1 axis
            d = np.dot(rotation_matrix(-n1, angle), d)
            d = np.dot(rotation_matrix(-ab, dihedral), d)

            # Add d to the position of q to get the new coordinates of the atom
            p = vb + d

            # Change of nonlocal variables
            xyz_frame.loc[index] = [atom] + list(p)
            already_built.append(to_be_built.pop(0))

        if n_atoms == 1:
            add_first_atom()

        elif n_atoms == 2:
            add_first_atom()
            add_second_atom()

        elif n_atoms == 3:
            add_first_atom()
            add_second_atom()
            add_third_atom()

        elif n_atoms > 3:
            add_first_atom()
            add_second_atom()
            add_third_atom()
            for _ in range(n_atoms - 3):
                add_atom()

        return xyz_frame

    def _zmat_to_xyz_experimental(zmat):
        n_atoms = zmat.shape[0]
        xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'],
                dtype='float',
                index=range(n_atoms)
                )
        to_be_built = list(zmat.index)
        already_built = []

        def add_first_atom():
            index = to_be_built[0]
            # Change of nonlocal variables
            xyz_frame.loc[index] = [zmat.at[index, 'atom'], 0., 0., 0.]
            already_built.append(to_be_built.pop(0))

        def add_second_atom():
            index = to_be_built[0]
            # Change of nonlocal variables
            xyz_frame.loc[index] = [zmat.at[index, 'atom'], zmat.at[index, 'bond'], 0., 0. ]
            already_built.append(to_be_built.pop(0))


        def add_third_atom():
            index = to_be_built[0]
            atom, bond, angle = zmat.loc[index, ['atom', 'bond', 'angle']]
            angle = m.radians(angle)
            x_projection, y_projection = np.cos(angle) * bond, np.sin(angle) * bond
            # Change of nonlocal variables
            xyz_frame.loc[index] = [atom, x_projection, y_projection, 0.]
            already_built.append(to_be_built.pop(0))

        def add_atom():
            normalize = _utilities.normalize

            index = to_be_built[0]
            atom, bond, angle, dihedral = zmat.loc[index, ['atom', 'bond', 'angle', 'dihedral']]
            angle, dihedral = map(m.radians, (angle, dihedral))
            bond_with, angle_with, dihedral_with = zmat.loc[index, ['bond_with', 'angle_with', 'dihedral_with']]
            bond_with, angle_with, dihedral_with = map(int, (bond_with, angle_with, dihedral_with))

            D2 = np.array([
                bond * np.cos(angle),
                bond * np.cos(dihedral) * np.sin(angle),
                bond * np.sin(dihedral) * np.sin(angle)
                ], dtype= float)

            vA = np.array(xyz_frame.loc[dihedral_with, ['x', 'y', 'z']], dtype=float)
            vB = np.array(xyz_frame.loc[angle_with, ['x', 'y', 'z']], dtype=float)
            vC = np.array(xyz_frame.loc[bond_with, ['x', 'y', 'z']], dtype=float)

            bc = normalize(vC - vB)
            ba = (vA - vB)
            n = normalize(np.cross(ba, bc))

            M = np.array([
                    bc,
                    np.cross(n, bc),
                    n
                    ])

            D = np.dot(M, D2) + vC


            xyz_frame.loc[index] = [atom] + list(D)
            already_built.append(to_be_built.pop(0))
            return M , np.linalg.norm(np.array([bc, ba, n]), axis = 1)


        if n_atoms == 1:
            add_first_atom()

        elif n_atoms == 2:
            add_first_atom()
            add_second_atom()

        elif n_atoms == 3:
            add_first_atom()
            add_second_atom()
            add_third_atom()

        elif n_atoms > 3:
            add_first_atom()
            add_second_atom()
            add_third_atom()

        return xyz_frame

    @staticmethod
    def concatenate(fragment_zmat_frame, zmat_frame, reference_atoms, xyz_frame = 'default'):
        """
        This function binds the fragment_zmat_frame onto the molecule defined by zmat_frame.
        The reference atoms is a list/matrix of three rows and four columns and contains the
        first atoms of fragment_zmat_frame and each their reference atoms.
        If xyz_frame is specified the values of the bond, angles and dihedrals are calculated.
        Otherwise the values 2, 90 and 90 are used.
        """
        fragment = fragment_zmat_frame.copy()
        distance = xyz_functions.distance
        angle_degrees = xyz_functions.angle_degrees
        dihedral_degrees = xyz_functions.dihedral_degrees


        assert (set(fragment.index) & set(zmat_frame)) == set([])
        assert len(reference_atoms) == 3

        all_reference_atoms_set = set([])
        for element in reference_atoms:
                all_reference_atoms_set |= set(element)

        all_reference_atoms = list(all_reference_atoms_set)
        fragment_atoms_index = [element[0] for element in reference_atoms]
        bond_with_list = [element[1] for element in reference_atoms]
        angle_with_list = [element[2] for element in reference_atoms]
        dihedral_with_list = [element[3] for element in reference_atoms]


        if type(xyz_frame) == str:
            bond = 1
            angle_list = [90 for _ in range(2)]
            dihedral_list = [90 for _ in range(3)]

        else:
            location_array = xyz_frame.loc[all_reference_atoms, ['x', 'y', 'z']].get_values().astype(float)

            bond_list = xyz_functions._distance_optimized(
                    location_array,
                    buildlist = reference_atoms,
                    indexlist = all_reference_atoms,
                    exclude_first = False
                    )
            angle_list = xyz_functions._angle_degrees_optimized(
                    location_array,
                    reference_atoms,
                    indexlist = all_reference_atoms,
                    exclude_first = False
                    )
            dihedral_list = xyz_functions._dihedral_degrees_optimized(
                    location_array,
                    reference_atoms,
                    indexlist = all_reference_atoms,
                    exclude_first = False
                    )

        fragment.loc[fragment_atoms_index, ['dihedral_with']] = dihedral_with_list
        fragment.loc[fragment_atoms_index, ['dihedral']] = dihedral_list
        fragment.loc[fragment_atoms_index, ['angle_with']] = angle_with_list
        fragment.loc[fragment_atoms_index, ['angle']] = angle_list
        fragment.loc[fragment_atoms_index, ['bond_with']] = bond_with_list
        fragment.loc[fragment_atoms_index, ['bond']] = bond_list


        out_frame = pd.concat([zmat_frame, fragment])
        return out_frame


    @staticmethod
    def changes_from_to_zmat(zmat_frame1, zmat_frame2, steps = 5):
        """
        This function returns a list of zmat_frames with the subsequent 
        movement from zmat_frame1 to zmat_frame2.
        The list contains zmat_frame1 as first and zmat_frame2 as last element.
        Please note, that for this reason: len(list) = (step + 1).
        """
        zmatframe1 = zmat_frame1.copy()
        zmatframe2 = zmat_frame2.copy()

        difference = zmat_frame2.copy()
        difference.loc[:, ['bond', 'angle', 'dihedral']] = zmat_frame2.loc[
                :, ['bond', 'angle', 'dihedral']
                ] - zmatframe1.loc[:, ['bond', 'angle', 'dihedral']]
        step_frame = difference.copy()

        step_frame.loc[:, ['bond', 'angle', 'dihedral']] = step_frame.loc[
                :, ['bond', 'angle', 'dihedral']] / steps

        list_of_zmatrices = []
        temp_zmat = zmatframe1.copy()

        for t in range(steps + 1):
            temp_zmat.loc[:, ['bond', 'angle', 'dihedral']] = zmatframe1.loc[
                    :, ['bond', 'angle', 'dihedral']] + (
                        step_frame.loc[:, ['bond', 'angle', 'dihedral']]  * t
                        )
            appendframe = temp_zmat.copy()
            list_of_zmatrices.append(appendframe)
        return list_of_zmatrices






class write:
    @staticmethod
    def zmat(zmat, outputfile, reset_numbering=True):
        """
        Writes the zmatrix into a file.
        """
        # The following functions are necessary to deal with the fact, that pandas does not support "NaN" for integers.
        EPSILON = 1e-9

        def _lost_precision(s):
            """
            The total amount of precision lost over Series `s`
            during conversion to int64 dtype
            """
            try:
                return (s - s.fillna(0).astype(np.int64)).sum()
            except ValueError:
                return np.nan

        def _nansafe_integer_convert(s):
            """
            Convert Series `s` to an object type with `np.nan`
            represented as an empty string ""
            """
            if _lost_precision(s) < EPSILON:
                # Here's where the magic happens
                as_object = s.fillna(0).astype(np.int64).astype(np.object)
                as_object[s.isnull()] = ""
                return as_object
            else:
                return s


        def nansafe_to_csv(df, *args, **kwargs):
            """
            Write `df` to a csv file, allowing for missing values
            in integer columns

            Uses `_lost_precision` to test whether a column can be
            converted to an integer data type without losing precision.
            Missing values in integer columns are represented as empty
            fields in the resulting csv.
            """
            df.apply(_nansafe_integer_convert).to_csv(*args, **kwargs)


        if reset_numbering:
            zmat_frame = zmat_functions.change_numbering(zmat)
            nansafe_to_csv(zmat_frame.loc[:, 'atom':], outputfile,
                sep=' ',
                index=False,
                header=False,
                mode='w'
                )
        else:
            nansafe_to_csv(zmat, outputfile,
                sep=' ',
                index=False,
                header=False,
                mode='w'
                )


    @staticmethod
    def xyz(xyz_frame, outputfile, sort = True):
        """
        Writes the xyz_DataFrame into a file.
        If (sort = True) the DataFrame is sorted by the index.
        """
        frame = xyz_frame[['atom', 'x', 'y','z']].copy()
        frame = frame.sort_index() if sort else frame
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


class _utilities:
    """
    A collection of recurring functions that should not be in the global namespace.
    """
    def distance_frame(xyz_frame, origin):
        origin = np.array(origin, dtype=float)
        frame_distance = xyz_frame.copy()
        frame_distance['distance'] = np.linalg.norm(np.array(frame_distance.loc[:,'x':'z']) - origin, axis =1)
        return frame_distance

    def normalize(vector):
        normed_vector = vector / np.linalg.norm(vector)
        return normed_vector




class _dev_utilities:
    @staticmethod
    def _test_conversion(how_often, xyz_frame):
        first_xyz = xyz_frame.copy()
        first_zmat = xyz_functions.xyz_to_zmat(first_xyz)
        zmat, xyz = first_zmat, first_xyz
        step_list=[(first_xyz, first_zmat)]
        for _ in range(how_often):
            xyz = zmat_functions.zmat_to_xyz(zmat)
            zmat = xyz_functions.xyz_to_zmat(xyz)
            step_list.append((xyz, zmat))
        return step_list
