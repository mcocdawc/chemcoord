Installation guide
==================
A working python 3 installation is required (3.7 <= version <= 3.11 are tested).

It is highly recommended to use this module in combination with
`Ipython <http://ipython.org/>`_ and `jupyter <http://jupyter.org/>`_.
They come shipped by default with the
`anaconda3 installer <https://www.continuum.io/downloads/>`_.

Unix
++++


There are packaged versions on PyPi and conda-forge.

For the packaged version from `PyPi <https://pypi.org/project/chemcoord/>`__, the following commands have to be executed::

   pip install chemcoord


For the packaged version from `conda-forge <https://anaconda.org/conda-forge/chemcoord>`__, the following commands have to be executed::

   conda install -c conda-forge chemcoord

For the up to date development version (experimental)::

   git clone https://github.com/mcocdawc/chemcoord.git
   cd chemcoord
   pip install .

Windows
+++++++

Neither installation nor running the module are tested on windows.
To the best of my knowledge it should work there as well.
Just use the same steps as for UNIX.
