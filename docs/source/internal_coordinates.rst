Internal coordinates
===================================

Zmat
-------------

The :class:`~chemcoord.Zmat` class which is used to represent
a molecule in non redundant, internal coordinates.

.. currentmodule:: chemcoord

.. autosummary::
    :toctree: src_Zmat

    ~Zmat



zmat_functions
---------------

A collection of functions operating on instances of :class:`~chemcoord.Zmat`.


.. currentmodule:: chemcoord.zmat_functions

.. rubric:: Functions

.. autosummary::
    :toctree: src_zmat_functions

    ~apply_grad_cartesian_tensor


.. rubric:: Contextmanagers

.. autosummary::
    :toctree: src_zmat_functions

    ~DummyManipulation
    ~TestOperators
    ~PureInternalMovement
