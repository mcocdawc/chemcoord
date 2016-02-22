Introduction and General Structure
==================================

What you need to know
+++++++++++++++++++++

I assume that you know `python <https://docs.python.org/3/tutorial/index.html>`_.

You can use chemcoord without knowing Pandas, but it gives you a great advantage.
If you invest 1h for their `tutorial <http://pandas.pydata.org/pandas-docs/stable/tutorials.html>`_ 
you will greatly increase your productivity in scientific data analysis.

It also helps to know about `numpy <https://docs.scipy.org/doc/numpy-dev/user/quickstart.html>`_.

Internal representation of Data
+++++++++++++++++++++++++++++++
This module uses pandas DataFrames to represent cartesian and internal coordinates.
(I will refer to them in lab slang as xyz and zmat)

The xyz_frame has at least four columns ``['atom', 'x', 'y', 'z']``.

The zmat_frame has at least seven columns ``['atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral']``.

Since they are normal pandas DataFrames you can do everything with them as long as you respect this structure.
This means it is possible to append e.g. a column for the masses of each atom.
Besides you can use all the capabilities of pandas. 

If you want for example to get only the oxygen atoms of a xyz_frame you can use boolean slicing::
    
    xyz_frame[xyz_frame['atom'] == 'O']


Main classes of this module
++++++++++++++++++++++++++++

The "working horses" of this module are the ``Cartesian`` and the ``Zmat`` class.
The have the methods to operate on their coordinates.

An methods of an instance of the ``Cartesian`` class usually return new instances of `Cartesian`.
Besides all methods are **sideeffect free unless otherwise stated**.

Let's assume you have a ``molecule1`` and you want to cut a sphere around the origin which gives you ``molecule2``::

    molecule2 = molecule1.cutsphere()

If you try this, you will see that:
* molecule2 is a ``Cartesian`` instance.
* molecule1 remains unchanged.

