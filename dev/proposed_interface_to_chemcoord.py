"""This is a proposed interface to chemcoord and will currently not work!
"""
import os

from ase.structure import molecule
from ase.lattice.surface import fcc111, add_adsorbate
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations
import ase.io
import numpy as np

import chemcoord as cc

if os.path.exists('CH3_Al.traj'):
    slab = ase.io.read('CH3_Al.traj')
    slab.set_calculator(EMT())  # Need to reset when loading from traj file
else:
    slab = fcc111('Al', size=(2, 2, 2), vacuum=3.0)

    CH3 = molecule('CH3')
    add_adsorbate(slab, CH3, 2.5, 'ontop')

    constraint = FixAtoms(mask=[a.symbol == 'Al' for a in slab])
    slab.set_constraint(constraint)

    slab.set_calculator(EMT())

    dyn = QuasiNewton(slab, trajectory='QN_slab.traj')
    dyn.run(fmax=0.05)

    ase.io.write('CH3_Al.traj', slab)

# Running vibrational analysis
vib = Vibrations(slab, indices=[8, 9, 10, 11])
vib.run()
vib.summary()


# Initialize the intermal coordinates representation of the modes
InternalCoordinates = cc.vibrations_to_internal_coordinates(
    vib,
    restrict_to_bending=1,  # exclude all vibrational extensions if this is on
    bonding_to_environment=[[7, 8]]  # list of pairs of atomic indices.
    # This is suppose to be the indices of the carbon atom (8) and
    # the Al atom that it is bonded to (7).
    )

# See the infitisimal movement vector of mode 0, which is the mode with the
# lowest eigenvalue in the eigenvalue spectrum of the harmonic normal mode
# analysis carried out by ase.vibrations
infinitesimal_mode_vector = InternalCoordinates.get_mode_infitisimal(0)

# Move along the internal coordinates of mode 0
# Make displacement here has the side effect of changing the internal
# coordinates for the InternalCoordinates object. That might not be the best
# way to do it. We could instead return a new object when doing
# make_displacemnet and run get_cartesian() on that.
# Displacement in radia
InternalCoordinates.make_displacement(-0.1, mode=0)
cartesian_coordinates = InternalCoordinates.get_cartesian()

# ASE calculation
slab.set_positions(cartesian_coordinates)
e = slab.get_potential_energy()
f = slab.get_forces()

# Now I can see what the projection vector is for the displacement
infinitesimal_mode_vector = InternalCoordinates.get_mode_infitisimal(0)
# project forces on the mode vector
forces_projected = np.dot(infinitesimal_mode_vector, f)

# Reset the displacements so that I can do a displacement along a different
# internal mode
InternalCoordinates.reset_displacements()
