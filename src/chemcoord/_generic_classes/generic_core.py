# -*- coding: utf-8 -*-
import chemcoord.constants as constants
import pandas as pd


class GenericCore(object):
    def add_data(self, new_cols=None):
        """Adds a column with the requested data.

        If you want to see for example the mass, the colormap used in
        jmol and the block of the element, just use::

            ['mass', 'jmol_color', 'block']

        The underlying ``pd.DataFrame`` can be accessed with
        ``constants.elements``.
        To see all available keys use ``constants.elements.info()``.

        The data comes from the module `mendeleev
        <http://mendeleev.readthedocs.org/en/latest/>`_ written
        by Lukasz Mentel.

        Please note that I added three columns to the mendeleev data::

            ['atomic_radius_cc', 'atomic_radius_gv', 'gv_color',
                'valency']

        The ``atomic_radius_cc`` is used by default by this module
        for determining bond lengths.
        The three others are taken from the MOLCAS grid viewer written
        by Valera Veryazov.

        Args:
            new_cols (str): You can pass also just one value.
                E.g. ``'mass'`` is equivalent to ``['mass']``. If
                ``new_cols`` is ``None`` all available data
                is returned.
            inplace (bool):

        Returns:
            Cartesian:
        """
        atoms = self['atom']
        data = constants.elements
        if pd.api.types.is_list_like(new_cols):
            new_cols = set(new_cols)
        elif new_cols is None:
            new_cols = set(data.columns)
        else:
            new_cols = [new_cols]
        new_frame = data.loc[atoms, list(set(new_cols) - set(self.columns))]
        new_frame.index = self.index
        return self.__class__(pd.concat([self._frame, new_frame], axis=1))

    def get_total_mass(self):
        """Returns the total mass in g/mol.

        Args:
            None

        Returns:
            float:
        """
        try:
            mass = self.loc[:, 'mass'].sum()
        except KeyError:
            mass_molecule = self.add_data('mass')
            mass = mass_molecule.loc[:, 'mass'].sum()
        return mass

    def has_same_sumformula(self, other):
        """Determines if ``other``  has the same sumformula

        Args:
            other (molecule):

        Returns:
            bool:
        """
        same_atoms = True
        for atom in set(self['atom']):
            own_atom_number = len(self[self['atom'] == atom])
            other_atom_number = len(other[other['atom'] == atom])
            same_atoms = (own_atom_number == other_atom_number)
            if not same_atoms:
                break
        return same_atoms

    def get_electron_number(self, charge=0):
        """Return the number of electrons.

        Args:
            charge (int): Charge of the molecule.

        Returns:
            int:
        """
        atomic_number = constants.elements['atomic_number'].to_dict()
        return sum([atomic_number[atom] for atom in self['atom']]) - charge
