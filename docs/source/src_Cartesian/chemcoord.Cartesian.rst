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
         ~Cartesian.get_without
         ~Cartesian.add_data
         ~Cartesian.get_total_mass
         ~Cartesian.get_electron_number
         ~Cartesian.get_coordination_sphere
         ~Cartesian.partition_chem_env


    .. rubric:: Manipulate

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.cut_cuboid
         ~Cartesian.cut_sphere
         ~Cartesian.basistransform
         ~Cartesian.align
         ~Cartesian.reindex_similar
         ~Cartesian.change_numbering
         ~Cartesian.subs

    .. rubric:: Geometry

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.get_bond_lengths
         ~Cartesian.get_angle_degrees
         ~Cartesian.get_dihedral_degrees
         ~Cartesian.get_barycenter
         ~Cartesian.get_inertia
         ~Cartesian.get_centroid
         ~Cartesian.get_distance_to
         ~Cartesian.get_shortest_distance

    .. rubric:: Conversion to internal coordinates

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.get_zmat
         ~Cartesian.get_grad_zmat
         ~Cartesian.get_construction_table
         ~Cartesian.check_dihedral
         ~Cartesian.correct_dihedral
         ~Cartesian.check_absolute_refs
         ~Cartesian.correct_absolute_refs
         ~Cartesian.to_zmat

    .. rubric:: Symmetry

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.get_pointgroup
         ~Cartesian.get_equivalent_atoms
         ~Cartesian.symmetrize
         ~Cartesian.get_asymmetric_unit



    .. rubric:: IO

    .. autosummary::
         :toctree: src_Cartesian

         ~Cartesian.to_xyz
         ~Cartesian.write_xyz
         ~Cartesian.read_xyz
         ~Cartesian.to_cjson
         ~Cartesian.read_cjson
         ~Cartesian.view
         ~Cartesian.to_string
         ~Cartesian.to_latex
         ~Cartesian.get_pymatgen_molecule
         ~Cartesian.from_pymatgen_molecule
         ~Cartesian.get_ase_atoms
         ~Cartesian.from_ase_atoms


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
