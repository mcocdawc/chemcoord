Configuration of settings
=========================


.. currentmodule:: chemcoord

The current settings of ``chemcoord`` can be seen with ``cc.settings``.
This is a dictionary that can be changed in place.
If it is necessary to change these settings permamently there is
the possibility to write a configuration file of the current settings,
that is read automatically while importing the module.
The configuration file is in the INI format and
can be changed with any text editor.

The possible settings and their defaults are:

``['defaults']``

  ``['atomic_radius_data'] = 'atomic_radius_cc'``
    Determines which atomic radius is used for calculating if atoms are bonded
  ``['use_lookup_internally'] = True``
    Look into :meth:`~chemcoord.Cartesian.get_bonds()` for an explanation
  ``['viewer'] = 'gv.exe'``
    Which one is the default viewer used in :meth:`chemcoord.Cartesian.view`
    and :func:`chemcoord.xyz_functions.view`.


.. autosummary::
    :toctree: src_configuration

    ~configuration.write_configuration_file
    ~configuration.read_configuration_file
