Configuration of settings
=========================

.. currentmodule:: chemcoord.configuration

The current settings of ``chemcoord`` can be seen with ``cc.settings``.
If it is necessary to change these settings permamently there is
the possibility to write a configuration file of the current settings,
that is read automatically while importing the module.
The configuration file is in the yaml format and
can be changed with any text editor.

The possible settings and their defaults are:


.. autoclass:: Settings

   .. rubric:: Methods

   .. autosummary::
    :toctree: src_Settings

    ~Settings.write

   .. rubric:: Attributes

   .. autosummary::
    :toctree: src_Settings

    ~Settings.defaults


.. autoclass:: Defaults

   .. rubric:: Attributes

   .. autosummary::
    :toctree: src_Defaults

    ~Defaults.atomic_radius_data
    ~Defaults.viewer
    ~Defaults.list_viewer_file