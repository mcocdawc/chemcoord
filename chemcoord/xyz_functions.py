
import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities
from . import zmat_functions



def cutsphere(xyz_frame, radius=15., origin=[0., 0., 0.], outside_sliced=True):
    """Cuts a sphere specified by origin and radius.

    Args:
        xyz_frame (pd.dataframe): 
        radius (float): 
        origin (list):
        outside_sliced (bool): Atoms outside/inside the sphere are cut out.

    Returns:
        pd.dataframe: Sliced xyz_frame
    """
    frame = distance_frame(xyz_frame, origin)
    if outside_sliced:
        sliced_xyz_frame = frame[frame['distance'] < radius]
    else:
        sliced_xyz_frame = frame[frame['distance'] > radius]
    return sliced_xyz_frame


def cutcube(xyz_frame, edge=20, origin=[0, 0, 0], outside_sliced = True):
    """Cuts a cube specified by edge and radius.

    Args:
        xyz_frame (pd.dataframe): 
        edge (float): 
        origin (list):
        outside_sliced (bool): Atoms outside/inside the sphere are cut out.

    Returns:
        pd.dataframe: Sliced xyz_frame
    """
    # Next line changes from ordinary python list to dictionary for easy access of the origin values
    origin = dict(zip(['x', 'y', 'z'], list(origin)))
    if outside_sliced:
        sliced_xyz_frame = xyz_frame[
               (np.abs((xyz_frame['x'] - origin['x'])) < edge / 2)
             & (np.abs((xyz_frame['y'] - origin['y'])) < edge / 2)
             & (np.abs((xyz_frame['z'] - origin['z'])) < edge / 2)
            ].copy()
    else:
        sliced_xyz_frame = xyz_frame[
               (np.abs((xyz_frame['x'] - origin['x'])) > edge / 2)
             & (np.abs((xyz_frame['y'] - origin['y'])) > edge / 2)
             & (np.abs((xyz_frame['z'] - origin['z'])) > edge / 2)
            ].copy()
    return sliced_xyz_frame


def mass(xyz_frame):
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

    barycenter = np.zeros([3])
    topologic_center = np.zeros([3])

    for row, index in enumerate(indexlist):
        barycenter = barycenter + location_array[row] * frame_mass.at[index, 'mass']
        topologic_center = topologic_center + location_array[row]
    barycenter = barycenter / total_mass
    topologic_center = topologic_center / n_atoms
    
    dic_of_values = dict(zip(['frame_mass', 'total_mass', 'barycenter', 'topologic_center'], [frame_mass, total_mass, barycenter, topologic_center]))
    return dic_of_values


def move(xyz_frame, vector=[0, 0, 0], matrix=np.identity(3)):
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
    frame = xyz_frame.copy()
    vectors = frame.loc[:, ['x', 'y', 'z']].get_values().astype(float)
    frame.loc[:, ['x', 'y', 'z']] = np.transpose(np.dot(np.array(matrix), np.transpose(vectors)))
    vectors = frame.loc[:, ['x', 'y', 'z']].get_values().astype(float)
    frame.loc[:, ['x', 'y', 'z']] = vectors + np.array(vector)
    return frame

def modify(buildlist, entries):
    """Modify a buildlist for constructing zmatrices.
   
    In order to know about the meaning of the buildlist, go to :func:`to_zmat`.

    .. warning:: The user has to make sure himself that the modified buildlist is still valid. 
        This means that atoms use only other atoms as reference, that are in previous rows of the buildlist.

    Here is an example::

        In: buildlist_test = [[1], [2, 1], [3, 2, 1], [4, 3, 2, 1]]
        In: modify(buildlist_test, [[3, 1, 2], [4, 1, 2, 3]])
        Out: [[1], [2, 1], [3, 1, 2], [4, 1, 2, 3]]

    Args:
        buildlist (list): 
        entries (list): Entries you want to change.

    Returns:
        list: Modified buildlist. 
    """
    buildlist_modified = copy.deepcopy(buildlist)
    if type(entries[0]) == list:
        indexlist = [element[0] for element in buildlist]
        for entry in entries:
            position = indexlist.index(entry[0])
            buildlist_modified[position] = entry
    elif type(entries[0]) == int:
        position = indexlist.index(entries[0])
        buildlist_modified[position] = entries
    return buildlist_modified





def distance(xyz_frame, index, bond_with):
    """Return the distance between two atoms.
   
    Args:
        xyz_frame (pd.dataframe): 
        index (int): 
        bond_with (int): Index of atom bonding with.

    Returns:
        float: distance
    """
    vi, vb = location(xyz_frame, [index, bond_with])
    q = vb - vi
    distance = np.linalg.norm(q)
    return distance



def _distance_optimized(xyz_frame, buildlist, exclude_first = True):
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
    location_array = location(xyz_frame, index_list)

    distance_list = []
    for listelement in temp_buildlist:
        index, bond_with = listelement[:2]
        vi, vb = location_array[[convert[index], convert[bond_with]], :]
        q = vb - vi
        distance = np.linalg.norm(q)
        distance_list.append(distance)
    return distance_list

def angle_degrees(frame, index, bond_with, angle_with):
    """Return the angle in dregrees between three atoms.
   
    Args:
        xyz_frame (pd.dataframe): 
        index (int): 
        bond_with (int): Index of atom bonding with.
        angle_with (int): Index of angle defining atom. 

    Returns:
        float: Angle in degrees.
    """
    vi, vb, va = location(frame, [index, bond_with, angle_with])
    BI, BA = vi - vb, va - vb
    angle = utilities.give_angle(BI, BA)
    return angle


def _angle_degrees_optimized(xyz_frame, buildlist, exclude_first = True):
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
    location_array = location(xyz_frame, index_list)

    angle_list = []
    for listelement in temp_buildlist:
        index, bond_with, angle_with = listelement[:3]
        vi, vb, va = location_array[[convert[index], convert[bond_with], convert[angle_with]], :]
        BI, BA = vi - vb, va - vb
        angle = utilities.give_angle(BI, BA)
        angle_list.append(angle)
    return angle_list


def dihedral_degrees(frame, index, bond_with, angle_with, dihedral_with):
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
    normalize = utilities.normalize
    vi, vb, va, vd = location(frame, [index, bond_with, angle_with, dihedral_with])

    DA = va - vd
    AB = vb - va
    BI = vi - vb

    n1 = normalize(np.cross(DA, AB))
    n2 = normalize(np.cross(AB, BI))

    dihedral = utilities.give_angle(n1, n2)
    if (dihedral != 0):
        dihedral = dihedral if 0 < np.dot(AB, np.cross(n1, n2)) else 360 - dihedral

    return dihedral


def _dihedral_degrees_optimized(xyz_frame, buildlist, exclude_first = True):
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
    location_array = location(xyz_frame, index_list)

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


def get_fragment(xyz_frame, list_of_indextuples, threshold=2.0):
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
            new_center = location(xyz_frame, index)
            just_found |= set(cutsphere(frame, radius = threshold, origin = new_center).index)

        previous_found = just_found - fixed
        just_found = set([])

    index_of_fragment_list = list(fixed)

    return index_of_fragment_list

def _order_of_building(xyz_frame, to_be_built=None, already_built=None, recursion=2):
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


# TODO Dont repeat if full building list is given
# TODO Check for linearity and insert dummy atoms
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
            distances_to_other_atoms = distance_frame(
                    xyz_frame.loc[already_built, :],
                    location(xyz_frame, index_of_atom)
                    )
            bond_with = distances_to_other_atoms['distance'].idxmin()
            indexlist.append([index_of_atom, bond_with])
            already_built.append(to_be_built.pop(0))

        def third_atom(index_of_atom):
            distances_to_other_atoms = distance_frame(
                    xyz_frame.loc[already_built, :],
                    location(xyz_frame, index_of_atom)
                    ).sort_values(by='distance')
            bond_with, angle_with = list(distances_to_other_atoms.iloc[0:2, :].index)
            indexlist.append([index_of_atom, bond_with, angle_with])
            already_built.append(to_be_built.pop(0))

        def other_atom(index_of_atom):
            distances_to_other_atoms = distance_frame(
                    xyz_frame.loc[already_built, :],
                    location(xyz_frame, index_of_atom)
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


def inertia(xyz_frame):
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
    my_keys = ['frame_mass', 'total_mass', 'barycenter', 'topologic_center']
    my_dic = mass(xyz_frame)
    frame_mass, total_mass, barycenter, topologic_center = [my_dic[key] for key in my_keys]

    frame_mass = move(frame_mass, vector = -barycenter)

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
    frame_mass = basistransform(frame_mass, new_basis, old_basis)
    
    dic_of_values =  dict(zip(['transformed_frame', 'diag_inertia_tensor', 'inertia_tensor', 'eigenvectors'], [frame_mass, diag_inertia_tensor, inertia_tensor, eigenvectors]))
    return dic_of_values



def make_similar(xyz_frame1, xyz_frame2, prealign = True):
    """Similarizes two xyz_frames.

    Returns a reindexed copy of xyz_frame2 that minimizes the distance 
    for each atom from xyz_frame1 to itself on xyz_frame2.

    .. warning:: The algorithm is still very basic, so it is important, to have a good
        prealignment and quite similar frames.

    Args:
        xyz_frame1 (pd.DataFrame): 
        xyz_frame2 (pd.DataFrame): 
        prealign (bool): If True both frames are moved to their barycenters and aligned along 
            their principal axes of inertia before reindexing.

    Returns:
        pd.DataFrame: Reindexed copy of xyz_frame2.
    """
    if prealign:
        frame1 = inertia(xyz_frame1)['transformed_frame']
        frame2 = inertia(xyz_frame2)['transformed_frame']
    else:
        frame1 = xyz_frame1.copy()
        frame2 = xyz_frame2.copy()

    assert set(frame1['atom']) == set(frame2['atom'])
    atomset = set(frame1['atom'])
    framedic = {}


    for atom in atomset:
        framedic[atom] = (frame1[frame1['atom'] == atom], frame2[frame2['atom'] == atom])
        assert framedic[atom][0].shape[0] == framedic[atom][1].shape[0]

    list_of_new_indexed_frames = []

    def _distance(vector1, vector2):
        length = np.linalg.norm(vector1 - vector2)
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
                    distance_old = utilities.distance(location_of_old_atom1, location_of_atom2)
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
    """Returns list of xyz_frames for the movement from xyz_frame1 to xyz_frame2.

    Args:
        xyz_frame1 (pd.DataFrame): 
        xyz_frame2 (pd.DataFrame): 
        step (int): 

    Returns:
        list: The list contains xyz_frame1 as first and xyz_frame2 as last element.
        The number of intermediate frames is defined by step.
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


def basistransform(xyz_frame, new_basis, old_basis=np.identity(3), rotate_only=True):
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
    frame = xyz_frame.copy()
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
    return frame



def location(xyz_frame, indexlist):
    """Returns the location of an atom.

    You can pass an indexlist or an index.

    Args:
        xyz_frame (pd.dataframe): 
        index (list): 

    Returns:
        np.array: A matrix of 3D rowvectors of the location of the atoms
        specified by indexlist. In the case of one index given a 3D vector is returned one index.
    """
    matrix = xyz_frame.ix[indexlist, ['x', 'y', 'z']].get_values().astype(float)
    return matrix


def distance_frame(xyz_frame, origin):
    """Returns a xyz_frame with a column for the distance from origin.
    """
    origin = np.array(origin, dtype=float)
    frame_distance = xyz_frame.copy()
    frame_distance['distance'] = np.linalg.norm(frame_distance.loc[:, ['x', 'y', 'z']].get_values().astype(float) - origin, axis =1)
    return frame_distance

def change_numbering(xyz_frame, rename_dic):
    """Returns the reindexed version of xyz_frame.

    Args:
        xyz_frame (pd.dataframe): 
        rename_dic (dic): A dictionary mapping integers on integers.

    Returns:
        pd.dataframe: 
    """
    frame = xyz_frame.copy()
    replace_list = list(rename_dic.keys())
    with_list = [rename_dic[key] for key in replace_list]
    frame['temporary_column'] = frame.index
    frame.loc[:, 'temporary_column'].replace(replace_list, with_list, inplace=True)
    frame.set_index('temporary_column', drop=True, inplace=True)
    frame.index.name = None
    return frame


def test_conversion(xyz_frame, steps=10, **kwargs):
    temp_xyz = xyz_frame.copy()
    framelist = [temp_xyz]
    for _ in range(steps):
        temp_zmat = to_zmat(temp_xyz)
        temp_xyz = zmat_functions.to_xyz(temp_zmat, **kwargs)
        framelist.append(temp_xyz)
    return framelist





