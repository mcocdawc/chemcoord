Cartesian coordinates
===================================

Cartesian
-------------

The :class:`~chemcoord.Cartesian` class which is used to represent
a molecule in cartesian coordinates.

.. currentmodule:: chemcoord

.. autosummary::
    :toctree: src_Cartesian

    ~Cartesian



xyz_functions
---------------

A collection of functions operating on instances of
:class:`~chemcoord.Cartesian`.


.. currentmodule:: chemcoord

.. autosummary::
    :toctree: src_xyz_functions

    ~xyz_functions.isclose
    ~xyz_functions.allclose
    ~xyz_functions.concat
    ~xyz_functions.write_molden
    ~xyz_functions.to_molden
    ~xyz_functions.read_molden
    ~xyz_functions.view
    ~xyz_functions.dot
    ~xyz_functions.apply_grad_zmat_tensor

Symmetry
---------

.. currentmodule:: chemcoord

.. autosummary::
    :toctree: src_PointGroupOperations

    ~PointGroupOperations

.. autosummary::
    :toctree: src_AsymmetricUnitCartesian

    ~AsymmetricUnitCartesian
