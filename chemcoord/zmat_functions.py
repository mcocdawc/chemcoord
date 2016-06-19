from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

try:
    import itertools.imap as map
except ImportError:
    pass

import numpy as np
import pandas as pd
import math as m
from . import _common_class
from . import export
from . import constants
from . import utilities
from ._exceptions import PhysicalMeaningError

@export
class Zmat(_common_class.common_methods):
    """The main class for dealing with internal coordinates.
    """
    def __init__(self, init):
        """How to initialize a Zmat instance.

        Args:
            init (pd.DataFrame): A Dataframe with at least the columns
                ``['atom', 'bond_with', 'bond', 'angle_with', 'angle',
                'dihedral_with', 'dihedral']``.
                Where ``'atom'`` is a string for the elementsymbol.

        Returns:
            Zmat: A new zmat instance.
        """
        try:
            tmp = init._to_Zmat()
            self.frame = tmp.frame.copy()
            self.shape = self.frame.shape
            self.n_atoms = self.shape[0]
            try:
                # self.__bond_dic = tmp.__bond_dic
                pass
            except AttributeError:
                pass

        except AttributeError:
            # Create from pd.DataFrame
            if not self._is_physical(init.columns):
                raise PhysicalMeaningError('There are columns missing for a meaningful description of a molecule')
            self.frame = init.copy()
            self.shape = self.frame.shape
            self.n_atoms = self.shape[0]


    def copy(self):
        molecule = self.__class__(self.frame)
        try:
            # molecule.__bond_dic = self.__bond_dic
            pass
        except AttributeError:
            pass
        return molecule


    def _to_Zmat(self):
        return self.copy()


    def build_list(self):
        """Return the buildlist which is necessary to create this Zmat

        Args:
            None

        Returns:
            np.array: Buildlist
        """
        columns = ['temporary_index', 'bond_with', 'angle_with', 'dihedral_with']
        tmp = self.insert(0, 'temporary_index', self.index)
        buildlist = tmp[:, columns].values.astype('int64')
        buildlist[0, 1:] = 0
        buildlist[1, 2:] = 0
        buildlist[2, 3:] = 0
        return buildlist

    def change_numbering(self, new_index=None, inplace=False):
        """Change numbering to a new index.

        Changes the numbering of index and all dependent numbering
            (bond_with...) to a new_index.
        The user has to make sure that the new_index consists of distinct
            elements.

        Args:
            new_index (list): If None the new_index is taken from 1 to the
            number of atoms.

        Returns:
            Zmat: Reindexed version of the zmatrix.
        """
        output = self if inplace else self.copy()
        old_index = output.index

        if (new_index is None):
            new_index = range(1, zmat_frame.shape[0]+1) 
        else:
            new_index = new_index
        assert len(new_index) == len(old_index)

        output.index = new_index

        output[:, ['bond_with', 'angle_with', 'dihedral_with']] = \
            output[:, ['bond_with', 'angle_with', 'dihedral_with']].replace(old_index, new_index)
        
        if not inplace:
            return output

    def to_xyz(self, SN_NeRF=False):
        """Transforms to cartesian space.

        Args:
            SN_NeRF (bool): Use the **Self-Normalizing Natural
            Extension Reference Frame** algorithm [1]_. In theory this
            means 30 % less floating point operations, but since
            this module is in python, floating point operations are
            not the rate determining step. Nevertheless it is a more
            elegant method than the "intuitive" conversion. Could make
            a difference in the future when certain functions will be
            implemented in ``Fortran``.

        Returns:
            Zmat: Reindexed version of the zmatrix.

        .. [1] Parsons J, Holmes JB, Rojas JM, Tsai J, Strauss CE (2005).
            Practical conversion from torsion space to Cartesian space for in
            silico protein synthesis.
            J Comput Chem. 26(10) , 1063-8.
            `doi:10.1002/jcc.20237 <http://dx.doi.org/10.1002/jcc.20237>`_
        """
        # zmat = self.zmat_frame.copy()
        # n_atoms = self.n_atoms
        xyz_frame = pd.DataFrame(
            columns=['atom', 'x', 'y', 'z'],
            dtype='float',
            index=self.index)

        # Cannot import globally in python 2, so we will only import here.
        # It is not a beautiful hack, but works for now!
        # See:
        # stackoverflow.com/questions/17226016/simple-cross-import-in-python
        from . import xyz_functions

        molecule = xyz_functions.Cartesian(xyz_frame)
        buildlist = self.build_list()

        normalize = utilities.normalize
        rotation_matrix = utilities.rotation_matrix

        def add_first_atom():
            index = buildlist[0, 0]
            # Change of nonlocal variables
            molecule[index, :] = [self[index, 'atom'], 0., 0., 0.]

        def add_second_atom():
            index = buildlist[1, 0]
            atom, bond = self[index, ['atom', 'bond']]
            # Change of nonlocal variables
            molecule[index, :] = [atom, bond, 0., 0.]

        def add_third_atom():
            index, bond_with, angle_with = buildlist[2, :3]
            atom, bond, angle = self[index, ['atom', 'bond', 'angle']]
            angle = m.radians(angle)

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
            molecule[index, :] = [atom] + list(p)

        def add_atom(row):
            index, bond_with, angle_with, dihedral_with = buildlist[row, :]
            atom, bond, angle, dihedral = self[index, ['atom', 'bond', 'angle', 'dihedral']]

            angle, dihedral = map(m.radians, (angle, dihedral))

            # vb is the vector of the atom bonding to,
            # va is the vector of the angle defining atom,
            # vd is the vector of the dihedral defining atom
            vb, va, vd = molecule.location(
                [bond_with, angle_with, dihedral_with])
            if np.isclose(m.degrees(angle), 180.):
                AB = vb - va
                ab = normalize(AB)
                d = bond * ab

                p = vb + d
                molecule[index, :] = [atom] + list(p)

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

                # Add d to the position of q to get the new coordinates
                # of the atom
                p = vb + d

                # Change of nonlocal variables
                molecule[index, :] = [atom] + list(p)

        def add_atom_SN_NeRF(row):
            normalize = utilities.normalize

            # TODO python2 compatibility
#            raise NotImplementedError(
#                "This functionality has not been implemented yet!")
#            index = None  # Should be added

            index, bond_with, angle_with, dihedral_with = buildlist[row, :]
            atom, bond, angle, dihedral = self[index, ['atom', 'bond', 'angle', 'dihedral']]
            angle, dihedral = map(m.radians, (angle, dihedral))
            bond_with, angle_with, dihedral_with = buildlist[row, 1:]

            vb, va, vd = molecule.location([bond_with, angle_with, dihedral_with])

            # The next steps implements the so called SN-NeRF algorithm.
            # In their paper they use a different definition of the angle.
            # This means, that I use sometimes cos instead of sin and other
            # minor changes
            # Compare with the paper:
            # Parsons J, Holmes JB, Rojas JM, Tsai J, Strauss CE.:
            # Practical conversion from torsion space to Cartesian space for
            # in silico protein synthesis.
            # J Comput Chem.  2005 Jul 30;26(10):1063-8.
            # PubMed PMID: 15898109

            # Theoretically it uses 30 % less floating point operations.
            # Since the python overhead is the limiting step, you won't see
            # any difference. But it is more elegant ;).

            if np.isclose(m.degrees(angle), 180.):
                AB = vb - va
                ab = normalize(AB)
                d = bond * ab

                p = vb + d
                molecule[index, :] = [atom] + list(p)

            else:
                D2 = bond * np.array([
                    - np.cos(angle),
                    np.cos(dihedral) * np.sin(angle),
                    np.sin(dihedral) * np.sin(angle)
                ], dtype=float)

                ab = normalize(vb - va)
                da = (va - vd)
                n = normalize(np.cross(da, ab))

                M = np.array([
                    ab,
                    np.cross(n, ab),
                    n])
                D = np.dot(np.transpose(M), D2) + vb

                molecule[index, :] = [atom] + list(D)

        if self.n_atoms == 1:
            add_first_atom()

        elif self.n_atoms == 2:
            add_first_atom()
            add_second_atom()

        elif self.n_atoms == 3:
            add_first_atom()
            add_second_atom()
            add_third_atom()

        elif self.n_atoms > 3:
            add_first_atom()
            add_second_atom()
            add_third_atom()
            if SN_NeRF:
                for row in range(3, self.n_atoms):
                    add_atom_SN_NeRF(row)
            else:
                for row in range(3, self.n_atoms):
                    add_atom(row)

        assert not molecule.frame.isnull().values.any(), \
            ('Serious bug while converting, please report an error'
                'on the Github page with your coordinate files')
        return molecule

    @classmethod
    def read_zmat(cls, inputfile, implicit_index=True):
        """Reads a zmat file.

        Lines beginning with ``#`` are ignored.

        Args:
            inputfile (str):
            implicit_index (bool): If this option is true the first column
            has to be the element symbols for the atoms.
                The row number is used to determine the index.

        Returns:
            Zmat:
        """
        if implicit_index:
            zmat_frame = pd.read_table(
                inputfile,
                comment='#',
                delim_whitespace=True,
                names=[
                    'atom', 'bond_with', 'bond', 'angle_with',
                    'angle', 'dihedral_with', 'dihedral'], )

            n_atoms = zmat_frame.shape[0]
            zmat_frame.index = range(1, n_atoms+1)
        else:
            zmat_frame = pd.read_table(
                inputfile,
                comment='#',
                delim_whitespace=True,
                names=[
                    'temp_index', 'atom', 'bond_with',
                    'bond', 'angle_with', 'angle',
                    'dihedral_with', 'dihedral'],
            )
            zmat_frame.set_index('temp_index', drop=True, inplace=True)
            zmat_frame.index.name = None
        return cls(zmat_frame)

    def write(self, outputfile, implicit_index=True):
        """Writes the zmatrix into a file.

        .. note:: Since it permamently writes a file, this function is
            strictly speaking **not sideeffect free**.
            The frame to be written is of course not changed.

        Args:
            outputfile (str):
            implicit_index (bool): If implicit_index is set, the zmat indexing
                is changed to range(1, number_atoms+1). Besides the index is
                omitted while writing which means, that the index is given
                implicitly by the row number.

        Returns:
            None: None
        """
        # The following functions are necessary to deal with the fact,
        # that pandas does not support "NaN" for integers.
        # It was written by the user LondonRob at StackExchange:
        # http://stackoverflow.com/questions/25789354/
        # exporting-ints-with-missing-values-to-csv-in-pandas/31208873#31208873
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
            nansafe_to_csv(
                zmat_frame.loc[:, 'atom':],
                outputfile,
                sep=str(' '),
                index=False,
                header=False,
                mode='w'
            )
        else:
            nansafe_to_csv(
                self.zmat_frame,
                outputfile,
                sep=str(' '),
                index=True,
                header=False,
                mode='w'
            )

