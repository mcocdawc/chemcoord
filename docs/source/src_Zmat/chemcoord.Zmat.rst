chemcoord\.Zmat
===============

.. currentmodule:: chemcoord

.. autoclass:: Zmat


   .. rubric:: Chemical Methods

   .. autosummary::
      :toctree: src_Zmat

      ~Zmat.__init__
      ~Zmat.add_data
      ~Zmat.change_numbering
      ~Zmat.has_same_sumformula
      ~Zmat.get_cartesian
      ~Zmat.get_grad_cartesian
      ~Zmat.to_xyz
      ~Zmat.get_total_mass
      ~Zmat.get_electron_number
      ~Zmat.subs
      ~Zmat.iupacify
      ~Zmat.minimize_dihedrals


   .. rubric:: Selection of data

   .. autosummary::
      :toctree: src_Zmat

      ~Zmat.loc
      ~Zmat.safe_loc
      ~Zmat.unsafe_loc
      ~Zmat.iloc
      ~Zmat.safe_iloc
      ~Zmat.unsafe_iloc

   .. rubric:: Pandas DataFrame Wrapper

   .. autosummary::
      :toctree: src_Zmat

      ~Zmat.copy
      ~Zmat.index
      ~Zmat.columns
      Zmat.sort_index
      ~Zmat.insert
      ~Zmat.sort_values


   .. rubric:: IO

   .. autosummary::
      :toctree: src_Zmat

      ~Zmat.to_zmat
      ~Zmat.write
      ~Zmat.read_zmat
      ~Zmat.to_string
      ~Zmat.to_latex


   .. rubric:: Attributes

   .. autosummary::
      :toctree: src_Zmat

      ~Zmat.columns
      ~Zmat.index
      ~Zmat.shape
      ~Zmat.dtypes
