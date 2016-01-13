Introduction and General Structure
==================================


This module uses DataFrames of the pandas module to represent cartesian and internal coordinates.
I will refer to them as xyz_frames and zmat_frames.

The functions operating on them are all **sideeffect free unless otherwise stated**.

The xyz_frame has at least four columns ``['atom', 'x', 'y', 'z']``.

The zmat_frame has at least seven columns ``['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral']``.

Since they are normal pandas DataFrames you can do everything with them as long as you respect this structure.
This means it is possible to append e.g. a column for the masses of each atom.
Besides you can use all the capabilities of pandas. 
If you want e.g. to get only the oxygen atoms of a xyz_frame you can slice with ``xyz_frame[xyz_frame['atom'] == 'O']``.
