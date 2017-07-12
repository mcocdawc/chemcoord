chemcoord\.Cartesian
====================

.. currentmodule:: chemcoord

.. autoclass:: Cartesian


    .. rubric:: Chemical Methods

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.__init__
         ~Cartesian.get_bonds
         ~Cartesian.restrict_bond_dict
         ~Cartesian.get_fragment
         ~Cartesian.fragmentate
         ~Cartesian.without
         ~Cartesian.add_data
         ~Cartesian.total_mass
         ~Cartesian.connected_to
         ~Cartesian.get_movement_to
         ~Cartesian.partition_chem_env


    .. rubric:: Manipulate

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.cutcuboid
         ~Cartesian.cutsphere
         ~Cartesian.move
         ~Cartesian.basistransform
         ~Cartesian.align
         ~Cartesian.change_numbering
         ~Cartesian.subs

    .. rubric:: Geometry

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.location
         ~Cartesian.bond_lengths
         ~Cartesian.angle_degrees
         ~Cartesian.dihedral_degrees
         ~Cartesian.barycenter
         ~Cartesian.inertia
         ~Cartesian.topologic_center
         ~Cartesian.distance_to
         ~Cartesian.shortest_distance

    .. rubric:: Conversion to internal coordinates

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.give_zmat
         ~Cartesian.get_construction_table
         ~Cartesian.check_dihedral
         ~Cartesian.correct_dihedral
         ~Cartesian.check_absolute_refs
         ~Cartesian.correct_absolute_refs
         ~Cartesian.to_zmat

    .. rubric:: IO

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.write_xyz
         ~Cartesian.to_xyz
         ~Cartesian.read_xyz
         ~Cartesian.view
         ~Cartesian.to_string
         ~Cartesian.to_latex


    .. rubric:: Pandas DataFrame Wrapper

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.copy
         ~Cartesian.index
         ~Cartesian.columns
         ~Cartesian.replace
         ~Cartesian.sort_index
         ~Cartesian.set_index
         ~Cartesian.append
         ~Cartesian.insert
         ~Cartesian.sort_values
         ~Cartesian.loc
         ~Cartesian.iloc


    .. rubric:: Advanced methods

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian._divide_et_impera
         ~Cartesian._preserve_bonds



   .. rubric:: Attributes

   .. autosummary::
         :toctree: src_Cartesian

      ~Cartesian.columns
      ~Cartesian.index
