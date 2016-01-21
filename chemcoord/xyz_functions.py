
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities


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

    if 'mass' in frame.columns:
        frame_mass = frame
    else:
        masses_dic = constants.elementary_masses
        masses = pd.Series([ masses_dic[atom] for atom in frame['atom']], name='mass', index=frame.index)
        # masses = pd.Series([ 5. for atom in frame['atom']], name='mass', index=frame.index)
        frame_mass = pd.concat([frame, masses], axis=1, join='outer')
    
    total_mass = frame_mass['mass'].sum()

    location_array = frame_mass.loc[indexlist, ['x', 'y', 'z']].get_values().astype(float)

    baryzentrum = np.zeros([3])
    topologic_center = np.zeros([3])

    for row, index in enumerate(indexlist):
        baryzentrum = baryzentrum + location_array[row] * frame_mass.at[index, 'mass']
        topologic_center = topologic_center + location_array[row]
    baryzentrum = baryzentrum / total_mass
    topologic_center = topologic_center / n_atoms
    return frame_mass, total_mass, baryzentrum, topologic_center


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

def modify(buildlist, entries):
    indexlist = [element[0] for element in buildlist]
    buildlist_modified = copy.deepcopy(buildlist)
    # changing_index = [element[0] for element in entries]
    for entry in entries:
        position = indexlist.index(entry[0])
        buildlist_modified[position] = entry

    return buildlist_modified





def distance(frame, index, bond_with):
    """
    The input is the own index, the index of the atom bonding to and a xyz_DataFrame.
    Returns the distance between these atoms.
    """
    vi, vb = frame.ix[[index, bond_with], ['x', 'y', 'z']].get_values().astype(float)
    q = vb - vi
    distance = np.linalg.norm(q)
    return distance


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

def angle_degrees(frame, index, bond_with, angle_with):
    """
    The input is the own index, the index of the atom bonding to,
    the index of the angle defining atom and a xyz_DataFrame.
    Returns the angle between these atoms in degrees.
    """
    normalize = utilities.normalize

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
    normalize = utilities.normalize
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


def dihedral_degrees(frame, index, bond_with, angle_with, dihedral_with):
    """
    The input is the own index, the index of the atom bonding to,
    the index of the angle defining atom, the index of the dihedral defining atom and a xyz_DataFrame.
    Returns the dihedral between these atoms in degrees.
    """
    normalize = utilities.normalize
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
    normalize = utilities.normalize
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

        scalar_product = np.dot(n1, n2)
        # Is this ok
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
        frame =cutsphere(frame, radius = (1.5 * bond ), origin= new_center, outside_sliced = False)

    fixed = set([])
    previous_found = set([tuple[1] for tuple in list_of_indextuples])
    just_found = set([])


    # Preparation for search

    while not previous_found < fixed:
        fixed |= previous_found
        for index in previous_found:
            new_center = np.array(frame.loc[index, ['x', 'y', 'z']], dtype = float)
            just_found |= set(cutsphere(frame, radius = threshold, origin = new_center).index)

        previous_found = just_found - fixed
        just_found = set([])

    index_of_fragment_list = list(fixed)

    return index_of_fragment_list

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
    topologic_center = mass(frame)[3]

    assert recursion in set([0, 1, 2])
    assert (set(to_be_built) & set(already_built)) == set([])


    def zero_reference(topologic_center):
        frame_distance = utilities.distance_frame(frame.loc[to_be_built, :], topologic_center)
        index = frame_distance['distance'].idxmin()
        return index

    def one_reference(previous):
        previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']]
        frame_distance_previous = utilities.distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
        index = frame_distance_previous['distance'].idxmin()
        return index, previous

    def two_references(previous, before_previous):
        previous_atom, before_previous_atom = xyz_frame.ix[previous, ['x', 'y', 'z']], xyz_frame.ix[before_previous, ['x', 'y', 'z']]
        frame_distance_previous = utilities.distance_frame(xyz_frame.loc[to_be_built, :], previous_atom)
        frame_distance_before_previous = utilities.distance_frame(xyz_frame.loc[to_be_built, :], before_previous_atom)
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
                distance = distance(frame, index, previous)
                if distance > 7:
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
                distance = distance(frame, index, previous)
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

def _get_reference(xyz_frame, order_of_building, recursion = 1):

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
            distances_to_other_atoms = utilities.distance_frame(
                    xyz_frame.loc[already_built, :],
                    xyz_frame.loc[index_of_atom, ['x', 'y', 'z']]
                    )
            bond_with = distances_to_other_atoms['distance'].idxmin()
            indexlist.append([index_of_atom, bond_with])
            already_built.append(to_be_built.pop(0))

        def third_atom(index_of_atom):
            distances_to_other_atoms = utilities.distance_frame(
                    xyz_frame.loc[already_built, :],
                    xyz_frame.loc[index_of_atom, ['x', 'y', 'z']]
                    ).sort_values(by='distance')
            bond_with, angle_with = list(distances_to_other_atoms.iloc[0:2, :].index)
            indexlist.append([index_of_atom, bond_with, angle_with])
            already_built.append(to_be_built.pop(0))

        def other_atom(index_of_atom):
            distances_to_other_atoms = utilities.distance_frame(
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



def _build_zmat(xyz_frame, buildlist):
    # not necessary
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
        location_array = xyz_frame.ix[indexlist, ['x', 'y', 'z']].get_values().astype(float)
        distance_list = _distance_optimized(location_array, buildlist)
        angle_list = _angle_degrees_optimized(location_array, buildlist)
        dihedral_list = _dihedral_degrees_optimized(location_array, buildlist)

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



def to_zmat(xyz_frame, buildlist = None, fragments_list = None, recursion_level = 2):
    """
    Takes a xyz_frame and converts to a zmat_frame.
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
    distance_frame = utilities.distance_frame


    def add_first_atom():
        topologic_center = mass(xyz_frame)[3]
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

def inertia(xyz_frame):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """
    rotation_matrix = utilities.rotation_matrix
    frame_mass, total_mass, baryzentrum, topologic_center = mass(xyz_frame)
    frame_mass = move(frame_mass, vector = -baryzentrum)

    coordinates = ['x', 'y', 'z']
    
    def kronecker(i, j):
        """
        Please note, that it also compares e.g. strings.
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

    v1, v2, v3 = np.transpose(eigenvectors)
    I1, I2, I3 = diag_inertia_tensor
    ex, ey, ez = np.identity(3)

    # Map v3 on ez
    axis = np.cross(ez, v3)
    angle = utilities.give_angle(ez, v3)
    rotationmatrix = utilities.rotation_matrix(axis, m.radians(angle))
    new_axes = rotationmatrix @ eigenvectors
    frame_mass = move(frame_mass, matrix = rotationmatrix)
    v1, v2, v3 = np.transpose(new_axes)

    # Map v1 on ex
    axis = ez
    angle = utilities.give_angle(ex, v1)
    if (angle != 0):
        angle = angle if 0 < np.dot(ez, np.cross(ex, v1)) else 360 - angle
    rotationmatrix = utilities.rotation_matrix(axis, m.radians(angle))
    new_axes = rotationmatrix @ new_axes
    frame_mass = move(frame_mass, matrix = rotationmatrix)
    v1, v2, v3 = np.transpose(new_axes)

    # Assert that new axes is right handed.
    if new_axes[1, 1] < 0:
        mirrormatrix = np.array([[1, 0, 0], [0,-1, 0],[0, 0, 1]])
        new_axes = mirrormatrix @ new_axes
        frame_mass = move(frame_mass, matrix = mirrormatrix)
        v1, v2, v3 = np.transpose(new_axes)

    return frame_mass, diag_inertia_tensor, inertia_tensor, eigenvectors



def make_similar(xyz_frame1, xyz_frame2, prealign = True):
    """
    Takes two xyz_DataFrames and returns a reindexed copy of xyz_frame2
    which minimizes the necessary movements to get from xyz_frame1 to xyz_frame2.
    """
    if prealign:
        frame1 = inertia(xyz_frame1)[0][['atom', 'x', 'y', 'z']]
        frame2 = inertia(xyz_frame2)[0][['atom', 'x', 'y', 'z']]
    else:
        frame1 = xyz_frame1.copy()
        frame2 = xyz_frame2.copy()

    assert set(frame1['atom']) == set(frame2['atom'])
    atomset = set(frame1['atom'])
    framedic = {}

    distance_frame = utilities.distance_frame

    for atom in atomset:
        framedic[atom] = (frame1[frame1['atom'] == atom], frame2[frame2['atom'] == atom])
        assert framedic[atom][0].shape[0] == framedic[atom][1].shape[0]

    list_of_new_indexed_frames = []

    def _distance(vector1, vector2):
        length = np.sqrt(np.linalg.norm(vector1 - vector2))
        return length


    for atom in atomset:
        index_dic = {}
        distance_frame_dic = {}

        frame1_indexlist = list(framedic[atom][0].index)
        for index_on_frame1 in frame1_indexlist:
            location_atom_frame1 = framedic[atom][0].loc[index_on_frame1, ['x', 'y', 'z']].get_values().astype(float)
            distances_to_atom_on_frame1 = distance_frame(framedic[atom][1], location_atom_frame1)
            
            
            distances_to_atom_on_frame1 = distances_to_atom_on_frame1.sort_values(by='distance')
            index_on_frame2 = distances_to_atom_on_frame1.iloc[0].name
            distance_new = distances_to_atom_on_frame1.at[index_on_frame2, 'distance']
            location_of_atom2 = distances_to_atom_on_frame1.loc[index_on_frame2, ['x', 'y', 'z']].get_values().astype(float)


            i = 1
            while True:
                if index_on_frame2 in index_dic.keys():
                    location_of_old_atom1 = framedic[atom][0].loc[index_dic[index_on_frame2], ['x', 'y', 'z']].get_values().astype(float)  
                    distance_old = _distance(location_of_old_atom1, location_of_atom2)
                    if distance_new < distance_old:
                        frame1_indexlist.append(index_dic[index_on_frame2])
                        index_dic[index_on_frame2] = index_on_frame1
                        break
                    else:
                        index_on_frame2 = distances_to_atom_on_frame1.iloc[i].name


                        distance_new = distances_to_atom_on_frame1.at[index_on_frame2, 'distance']
                        location_of_atom2 = distances_to_atom_on_frame1.loc[index_on_frame2, ['x', 'y', 'z']].get_values().astype(float)
                        i = i + 1

                else:
                    index_dic[index_on_frame2] = index_on_frame1
                    break


        new_index = [index_dic[old_index2] for old_index2 in framedic[atom][1].index]
        framedic[atom][1].index = new_index
        list_of_new_indexed_frames.append(framedic[atom][1])
    
    xyz_frame3 = pd.concat(list_of_new_indexed_frames)
   
   
    return xyz_frame3.sort_index()

def from_to(xyz_frame1, xyz_frame2, step=5):
    """
    This function returns a list of xyz_frames with the subsequent 
    movement from xyz_frame1 to xyz_frame2.
    The list contains xyz_frame1 as first and xyz_frame2 as last element.
    Please note, that for this reason: len(list) = (step + 1).
    """
    xyzframe1 = xyz_frame1.copy()
    xyzframe2 = xyz_frame2.copy()

    difference = xyzframe2.copy()
    difference.loc[:, ['x', 'y', 'z']] = xyzframe2.loc[:, ['x', 'y', 'z']] - (
            xyzframe1.loc[:, ['x', 'y', 'z']]
            )

    step_frame = difference.copy()
    step_frame.loc[:, ['x', 'y', 'z']]  = step_frame.loc[:, ['x', 'y', 'z']] / step

    list_of_xyzframes = []
    temp_xyz = xyzframe1.copy()

    for t in range(step + 1):
        temp_xyz.loc[:, ['x', 'y', 'z']] = xyzframe1.loc[:, ['x', 'y', 'z']] + (
                    step_frame.loc[:, ['x', 'y', 'z']] * t
                    )
        appendframe = temp_xyz.copy()
        list_of_xyzframes.append(appendframe)
    return list_of_xyzframes


