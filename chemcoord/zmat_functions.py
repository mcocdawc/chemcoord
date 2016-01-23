import numpy as np
import pandas as pd
import math as m
import copy
from . import constants
from . import utilities
from . import xyz_functions

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


def build_list(zmat_frame, complete=False):
    """
    This functions outputs the buildlist required to reproduce
    building of the zmat_frame.
    """
    zmat = zmat_frame.copy()
    n_atoms = zmat.shape[0]
    zmat.insert(0, 'temporary_index', zmat.index)

    if complete:
        array_of_values = zmat.loc[:, ['temporary_index', 'bond_with', 'angle_with', 'dihedral_with']].get_values().astype(int)
        buildlist =  [list(vector) for vector in array_of_values]
        
    else:
        array_of_values = zmat.loc[:, ['temporary_index', 'bond_with', 'angle_with', 'dihedral_with']].get_values().astype(int)

        temporary_list1 = [[array_of_values[0, 0]],  list(array_of_values[1, 0:2]), list(array_of_values[2, 0:3])]
        temporary_list2 = [list(vector) for vector in array_of_values[3:]]

        buildlist = temporary_list1 + temporary_list2

    return buildlist


def to_xyz(zmat, SN_NeRF=False):
    """
    The input is a zmat_DataFrame.
    The output is a xyz_DataFrame.
    If SN_NeRF is True the algorithm is used.
    """
    n_atoms = zmat.shape[0]
    xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'],
            dtype='float',
            index=zmat.index
            )
    to_be_built = list(zmat.index)
    already_built = []

    normalize = utilities.normalize
    rotation_matrix = utilities.rotation_matrix

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

        # vb is the vector of the atom bonding to, 
        # va is the vector of the angle defining atom,
        vb, va = xyz_functions.location(xyz_frame, [bond_with, angle_with])

        # Vector pointing from vb to va
        BA = va - vb

        # Vector of length distance 
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

        # vb is the vector of the atom bonding to, 
        # va is the vector of the angle defining atom,
        # vd is the vector of the dihedral defining atom
        vb, va, vd = xyz_functions.location(xyz_frame, [bond_with, angle_with, dihedral_with])

        if (m.radians(179.9999999) < angle < m.radians(180.0000001)):
            AB = vb - va
            ab = normalize(AB)
            d = bond * ab

            p = vb + d
            xyz_frame.loc[index] = [atom] + list(p)
            already_built.append(to_be_built.pop(0))

        else:
            AB = vb - va
            DA = vd - va

            n1 = normalize(np.cross(DA, AB))
            ab = normalize(AB)

            # Vector of length distance pointing along the x-axis
            d = bond * -ab

            # Rotate d by the angle around the n1 axis
            d = np.dot(rotation_matrix(n1, angle), d)
            d = np.dot(rotation_matrix(ab, dihedral), d)

            # Add d to the position of q to get the new coordinates of the atom
            p = vb + d

            # Change of nonlocal variables
            xyz_frame.loc[index] = [atom] + list(p)
            already_built.append(to_be_built.pop(0))

    def add_atom_SN_NeRF():
        normalize = utilities.normalize

        index = to_be_built[0]
        atom, bond, angle, dihedral = zmat.loc[index, ['atom', 'bond', 'angle', 'dihedral']]
        angle, dihedral = map(m.radians, (angle, dihedral))
        bond_with, angle_with, dihedral_with = zmat.loc[index, ['bond_with', 'angle_with', 'dihedral_with']]
        bond_with, angle_with, dihedral_with = map(int, (bond_with, angle_with, dihedral_with))

        vb, va, vd = xyz_functions.location(xyz_frame, [bond_with, angle_with, dihedral_with])

        # The next steps implements the so called SN-NeRF algorithm. 
        # In their paper they use a different definition of the angle.
        # This means, that I use sometimes cos instead of sin and other minor changes
        # Compare with the paper:
        # Parsons J, Holmes JB, Rojas JM, Tsai J, Strauss CE.: 
        # Practical conversion from torsion space to Cartesian space for in silico protein synthesis. 
        # J Comput Chem.  2005 Jul 30;26(10):1063-8. 
        # PubMed PMID: 15898109


        D2 = bond * np.array([
                - np.cos(angle),
                np.cos(dihedral) * np.sin(angle),
                np.sin(dihedral) * np.sin(angle)
                ], dtype= float)

        ab = normalize(vb - va)
        da = (va - vd)
        n = normalize(np.cross(da, ab))

        M = np.array([
                ab,
                np.cross(n, ab),
                n
                ])
        D = np.dot(np.transpose(M), D2) + vb

        xyz_frame.loc[index] = [atom] + list(D)
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
        if SN_NeRF:
            for _ in range(n_atoms - 3):
                add_atom_SN_NeRF()
        else:
            for _ in range(n_atoms - 3):
                add_atom()

    return xyz_frame


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


def from_to(zmat_frame1, zmat_frame2, steps = 5):
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
