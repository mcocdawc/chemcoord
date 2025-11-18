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
    ~CleanDihedralOrientation


Redundant Internal Coordinates (RICs)
-------------------------------------

The :class:`~chemcoord.RedundantInternalCoordinates` class which is used to represent
a molecule in redundant internal coordinates, along with the related methods.
And the :class:`~chemcoord.DeltaRedundantInternalCoordinates` class which is used
to represent differences of RICs.

.. currentmodule:: chemcoord

.. autosummary::
    :toctree: src_RIC

    ~RedundantInternalCoordinates
    ~DeltaRedundantInternalCoordinates


RIC functions
---------------

A collection of functions operating on instances of
:class:`~chemcoord.RedundantInternalCoordinates`
or :class:`~chemcoord.DeltaRedundantInternalCoordinates`.


.. currentmodule:: chemcoord.ric_functions

.. autosummary::
    :toctree: src_RIC_functions

    ~RIC_interpolate
    ~get_primitives_idx
    ~DefaultWeights
