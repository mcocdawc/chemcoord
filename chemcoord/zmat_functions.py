import numpy as np
import pandas as pd
import math as m
import copy
from . import export
from . import constants
from . import utilities
from . import xyz_functions

@export
class Zmat:
    def __init__(self, zmat_frame):
        self.zmat_frame = zmat_frame.copy()
        self.n_atoms = zmat_frame.shape[0]

    def __repr__(self):
        return self.zmat_frame.__repr__()

    def _repr_html_(self):
        return self.zmat_frame._repr_html_()

    def mass(self):
        """Gives several properties related to mass.
    
        Args:
    
        Returns:
            dic: The returned dictionary has four possible keys:
            
            ``Zmat_mass``: Zmatrix with an additional column for the masses of each atom.
        
            ``total_mass``: The total mass.
        """
        zmat_mass = self.zmat_frame.copy()
        masses_dic = dict(zip(
                constants.atom_properties.keys(), 
                [constants.atom_properties[atom]['mass'] for atom in constants.atom_properties.keys()]
                ))
        masses = [masses_dic[atom] for atom in zmat_mass['atom']]
        zmat_mass['mass'] = masses
        total_mass = zmat_mass['mass'].sum()
        dic_of_values = dict(zip(['Zmat_mass', 'total_mass'], [Zmat(zmat_mass), total_mass]))
        return dic_of_values


    def build_list(self):
        """Return the buildlist which is necessary to create this Zmat
    
        Args:
    
        Returns:
            np.array: Buildlist
        """
        zmat = self.zmat_frame.copy()
        n_atoms = zmat.shape[0]
        zmat.insert(0, 'temporary_index', zmat.index)
    
        buildlist = zmat.loc[:, ['temporary_index', 'bond_with', 'angle_with', 'dihedral_with']].get_values().astype('int64')
        buildlist[0,1:] = 0
        buildlist[1,2:] = 0
        buildlist[2,3:] = 0
        return buildlist


    def change_numbering(self, new_index = None):
        """
        The input is a zmat_DataFrame.
        Changes the numbering of index and all dependent numbering (bond_with...) to a new_index.
        The user has to make sure that the new_index consists of distinct elements.
        By default the new_index are the natural number ascending from 0.
        """
        zmat_frame = self.zmat_frame.copy()
        old_index = list(zmat_frame.index)
        new_index = list(range(1, zmat_frame.shape[0]+1)) if (new_index == None) else list(new_index)
        assert len(new_index) == len(old_index)
        zmat_frame.index = new_index
        zmat_frame.loc[:, ['bond_with', 'angle_with', 'dihedral_with']] = zmat_frame.loc[
                :, ['bond_with', 'angle_with', 'dihedral_with']].replace(old_index, new_index)
        return self.__class__(zmat_frame)


    # TODO performing better
    def movement_to(self, Zmat, steps = 5):
        """
        This function returns a list of zmat_frames with the subsequent 
        movement from zmat_frame1 to zmat_frame2.
        The list contains zmat_frame1 as first and zmat_frame2 as last element.
        Please note, that for this reason: len(list) = (step + 1).
        """
        zmatframe1 = self.zmat_frame.copy()
        zmatframe2 = Zmat.zmat_frame.copy()
        difference = zmatframe2.copy()
    
        difference.loc[:, ['bond', 'angle', 'dihedral']] = zmatframe2.loc[
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
            
            list_of_zmatrices.append(self.__class__(temp_zmat))
        return list_of_zmatrices


    def to_xyz(self, SN_NeRF=False):
        """
        The input is a zmat_DataFrame.
        The output is a xyz_DataFrame.
        If SN_NeRF is True the algorithm is used.
        """
        zmat = self.zmat_frame.copy()
        n_atoms = zmat.shape[0]
        xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'],
                dtype='float',
                index=zmat.index
                )
        molecule = xyz_functions.Cartesian(xyz_frame)
        buildlist = self.build_list()
        to_be_built = list(zmat.index)
        already_built = []
    
        normalize = utilities.normalize
        rotation_matrix = utilities.rotation_matrix
        tolerance_for_linearity=1e-04
    
        def add_first_atom():
            index = to_be_built[0]
            # Change of nonlocal variables
            molecule.xyz_frame.loc[index] = [zmat.at[index, 'atom'], 0., 0., 0.]
            already_built.append(to_be_built.pop(0))
    
        def add_second_atom():
            index = to_be_built[0]
            atom, bond = zmat.loc[index, ['atom', 'bond']]
            # Change of nonlocal variables
            molecule.xyz_frame.loc[index] = [atom, bond, 0., 0. ]
            already_built.append(to_be_built.pop(0))
    
    
        def add_third_atom():
            index = to_be_built[0]
            atom, bond, angle = zmat.loc[index, ['atom', 'bond', 'angle']]
            angle = m.radians(angle)
            bond_with, angle_with = buildlist[2, 1:3]
#            bond_with, angle_with = zmat.loc[index, ['bond_with', 'angle_with']]
#            bond_with, angle_with = map(int, (bond_with, angle_with))
    
            # vb is the vector of the atom bonding to, 
            # va is the vector of the angle defining atom,
            vb, va = molecule.location([bond_with, angle_with])
    
            # Vector pointing from vb to va
            BA = va - vb
    
            # Vector of length distance 
            d = bond * normalize(BA)
    
            # Rotate d by the angle around the z-axis
            d = np.dot(rotation_matrix([0, 0, 1], angle), d)
    
            # Add d to the position of q to get the new coordinates of the atom
            p = vb + d
    
            # Change of nonlocal variables
            molecule.xyz_frame.loc[index] = [atom] + list(p)
            already_built.append(to_be_built.pop(0))
    
    
        def add_atom(row):
            index = to_be_built[0]
            atom, bond, angle, dihedral = zmat.loc[index, ['atom', 'bond', 'angle', 'dihedral']]
            angle, dihedral = map(m.radians, (angle, dihedral))
            bond_with, angle_with, dihedral_with = buildlist[row, 1:]
#            bond_with, angle_with, dihedral_with = zmat.loc[index, ['bond_with', 'angle_with', 'dihedral_with']]
#            bond_with, angle_with, dihedral_with = map(int, (bond_with, angle_with, dihedral_with))
    
            # vb is the vector of the atom bonding to, 
            # va is the vector of the angle defining atom,
            # vd is the vector of the dihedral defining atom
            vb, va, vd = molecule.location([bond_with, angle_with, dihedral_with])
            if np.isclose(m.degrees(angle), 180., atol=tolerance_for_linearity):
                AB = vb - va
                ab = normalize(AB)
                d = bond * ab
    
                p = vb + d
                molecule.xyz_frame.loc[index] = [atom] + list(p)
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
                molecule.xyz_frame.loc[index] = [atom] + list(p)
                already_built.append(to_be_built.pop(0))
    
        def add_atom_SN_NeRF(row):
            normalize = utilities.normalize
    
            index = to_be_built[0]
            atom, bond, angle, dihedral = zmat.loc[index, ['atom', 'bond', 'angle', 'dihedral']]
            angle, dihedral = map(m.radians, (angle, dihedral))
            bond_with, angle_with, dihedral_with = buildlist[row, 1:]
#            bond_with, angle_with, dihedral_with = zmat.loc[index, ['bond_with', 'angle_with', 'dihedral_with']]
#            bond_with, angle_with, dihedral_with = map(int, (bond_with, angle_with, dihedral_with))
    
            vb, va, vd = molecule.location([bond_with, angle_with, dihedral_with])
    
            # The next steps implements the so called SN-NeRF algorithm. 
            # In their paper they use a different definition of the angle.
            # This means, that I use sometimes cos instead of sin and other minor changes
            # Compare with the paper:
            # Parsons J, Holmes JB, Rojas JM, Tsai J, Strauss CE.: 
            # Practical conversion from torsion space to Cartesian space for in silico protein synthesis. 
            # J Comput Chem.  2005 Jul 30;26(10):1063-8. 
            # PubMed PMID: 15898109

            # Theoretically it uses 30 % less floating point operations. 
            # Since the python overhead is the limiting step, you won't see any difference.
            # But it is more elegant ;).

            if np.isclose(m.degrees(angle), 180., atol=tolerance_for_linearity):
                AB = vb - va
                ab = normalize(AB)
                d = bond * ab
    
                p = vb + d
                molecule.xyz_frame.loc[index] = [atom] + list(p)
                already_built.append(to_be_built.pop(0))
    
            else:
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
        
                molecule.xyz_frame.loc[index] = [atom] + list(D)
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
                for row in range(3, n_atoms):
                    add_atom_SN_NeRF(row)
            else:
                for row in range(3, n_atoms):
                    add_atom(row)
        return molecule

    @classmethod
    def read_zmat(cls, inputfile, implicit_index=True):
        """Reads a zmat file.
        
        Lines beginning with ``#`` are ignored.
    
        Args:
            inputfile (str): 
            implicit_index (bool): If this option is true the first column has to be the element symbols for the atoms.
                The row number is used to determine the index.
    
        Returns:
            pd.DataFrame: 
        """
        if implicit_index:
            zmat_frame = pd.read_table(
                inputfile,
                comment='#',
                delim_whitespace=True,
                names=['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'],
                )
            n_atoms = zmat_frame.shape[0]
            zmat_frame.index = range(1, n_atoms+1)
        else:
            zmat_frame = pd.read_table(
                inputfile,
                comment='#',
                delim_whitespace=True,
                names=['temp_index', 'atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'],
                )
            zmat_frame.set_index('temp_index', drop=True, inplace=True)
            zmat_frame.index.name = None
        return cls(zmat_frame)


    def write(self, outputfile, implicit_index=True):
        """Writes the zmatrix into a file.
    
        .. note:: Since it permamently writes a file, this function is strictly speaking **not sideeffect free**.
            The frame to be written is of course not changed.
    
        Args:
            zmat (pd.dataframe): 
            outputfile (str): 
            implicit_index (bool): If implicit_index is set, the zmat indexing is changed to range(1, number_atoms+1)
                Besides the index is omitted while writing which means, that the index is given implicitly by the row number.
    
        Returns:
            None: None
        """
        # The following functions are necessary to deal with the fact, that pandas does not support "NaN" for integers.
        # It was written by the user LondonRob at StackExchange:
        # http://stackoverflow.com/questions/25789354/exporting-ints-with-missing-values-to-csv-in-pandas/31208873#31208873
        # Begin of the copied code snippet
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
        # End of the copied code snippet
    
             
        if implicit_index:
            zmat_frame = self.change_numbering().zmat_frame
            nansafe_to_csv(zmat_frame.loc[:, 'atom':], outputfile,
                sep=' ',
                index=False,
                header=False,
                mode='w'
                )
        else:
            nansafe_to_csv(self.zmat_frame, outputfile,
                sep=' ',
                index=True,
                header=False,
                mode='w'
                )

#def concatenate(self, zmat_frame, reference_atoms, xyz_frame=None):
#    """
#    This function binds the fragment_zmat_frame onto the molecule defined by zmat_frame.
#    The reference atoms is a list/matrix of three rows and four columns and contains the
#    first atoms of fragment_zmat_frame and each their reference atoms.
#    If xyz_frame is specified the values of the bond, angles and dihedrals are calculated.
#    Otherwise the values 2, 90 and 90 are used.
#    """
#    fragment = fragment_zmat_frame.copy()
#    distance = xyz_functions.distance
#    angle_degrees = xyz_functions.angle_degrees
#    dihedral_degrees = xyz_functions.dihedral_degrees
#
#
#    assert (set(fragment.index) & set(zmat_frame)) == set([])
#    assert len(reference_atoms) == 3
#
#    all_reference_atoms_set = set([])
#    for element in reference_atoms:
#            all_reference_atoms_set |= set(element)
#
#    all_reference_atoms = list(all_reference_atoms_set)
#    fragment_atoms_index = [element[0] for element in reference_atoms]
#    bond_with_list = [element[1] for element in reference_atoms]
#    angle_with_list = [element[2] for element in reference_atoms]
#    dihedral_with_list = [element[3] for element in reference_atoms]
#
#
#    if xyz_frame is None:
#        bond = 1
#        angle_list = [90 for _ in range(2)]
#        dihedral_list = [90 for _ in range(3)]
#
#    else:
#        bond_list = xyz_functions._distance_optimized(
#                xyz_frame,
#                buildlist=reference_atoms,
#                exclude_first = False
#                )
#        angle_list = xyz_functions._angle_degrees_optimized(
#                xyz_frame,
#                buildlist=reference_atoms,
#                exclude_first = False
#                )
#        dihedral_list = xyz_functions._dihedral_degrees_optimized(
#                xyz_frame,
#                buildlist=reference_atoms,
#                exclude_first = False
#                )
#
#    fragment.loc[fragment_atoms_index, ['dihedral_with']] = dihedral_with_list
#    fragment.loc[fragment_atoms_index, ['dihedral']] = dihedral_list
#    fragment.loc[fragment_atoms_index, ['angle_with']] = angle_with_list
#    fragment.loc[fragment_atoms_index, ['angle']] = angle_list
#    fragment.loc[fragment_atoms_index, ['bond_with']] = bond_with_list
#    fragment.loc[fragment_atoms_index, ['bond']] = bond_list
#
#
#    out_frame = pd.concat([zmat_frame, fragment])
#    return out_frame
