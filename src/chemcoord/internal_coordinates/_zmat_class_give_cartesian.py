# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import math as m
import warnings
from chemcoord.internal_coordinates._zmat_class_core import Zmat_core
from chemcoord.utilities import algebra_utilities
from chemcoord.configuration import settings


class Zmat_give_cartesian(Zmat_core):
    """The main class for dealing with internal coordinates.
    """
    def give_cartesian(self, SN_NeRF=False):
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
            Cartesian:

        .. [1] Parsons J, Holmes JB, Rojas JM, Tsai J, Strauss CE (2005).
            Practical conversion from torsion space to Cartesian space for in
            silico protein synthesis.
            J Comput Chem. 26(10) , 1063-8.
            `doi:10.1002/jcc.20237 <http://dx.doi.org/10.1002/jcc.20237>`_
        """
        # zmat = self.zmat_frame.copy()
        # n_atoms = self.n_atoms
        xyz_frame = pd.DataFrame(columns=['atom', 'x', 'y', 'z'],
                                 dtype='float', index=self.index)

        # TODO correct
        # Cannot import globally in python 2, so we will only import here.
        # It is not a beautiful hack, but works for now!
        # See:
        # stackoverflow.com/questions/17226016/simple-cross-import-in-python
        # from . import xyz_functions
        from chemcoord.cartesian_coordinates.cartesian_class_main \
            import Cartesian

        molecule = Cartesian(xyz_frame)
        buildlist = self.get_buildlist()

        normalize = algebra_utilities.normalize
        rotation_matrix = algebra_utilities.rotation_matrix

        def add_first_atom():
            index = buildlist[0, 0]
            # Change of nonlocal variables
            molecule.loc[index, :] = [self.loc[index, 'atom'], 0., 0., 0.]

        def add_second_atom():
            index = buildlist[1, 0]
            atom, bond = self.loc[index, ['atom', 'bond']]
            # Change of nonlocal variables
            molecule.loc[index, :] = [atom, bond, 0., 0.]

        def add_third_atom():
            index, bond_with, angle_with = buildlist[2, :3]
            atom, bond, angle = self.loc[index, ['atom', 'bond', 'angle']]
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
            molecule.loc[index, :] = [atom] + list(p)

        def add_atom(row):
            index, bond_with, angle_with, dihedral_with = buildlist[row, :]
            atom, bond, angle, dihedral = self.loc[
                index, ['atom', 'bond', 'angle', 'dihedral']]

            angle, dihedral = [m.radians(x) for x in (angle, dihedral)]

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
                molecule.loc[index, :] = [atom] + list(p)

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
                molecule.loc[index, :] = [atom] + list(p)

        def add_atom_SN_NeRF(row):
            normalize = algebra_utilities.normalize

            # TODO python2 compatibility
#            raise NotImplementedError(
#                "This functionality has not been implemented yet!")
#            index = None  # Should be added

            index, bond_with, angle_with, dihedral_with = buildlist[row, :]
            atom, bond, angle, dihedral = self.loc[
                index, ['atom', 'bond', 'angle', 'dihedral']]
            angle, dihedral = [m.radians(x) for x in (angle, dihedral)]
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
                molecule.loc[index, :] = [atom] + list(p)

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

                molecule.loc[index, :] = [atom] + list(D)

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

        molecule.metadata = self.metadata
        return molecule

    def to_xyz(self, *args, **kwargs):
        """Deprecated, use :meth:`~chemcoord.Zmat.give_cartesian`
        """
        message = 'Will be removed in the future. Please use give_cartesian.'
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.give_cartesian(*args, **kwargs)
