from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
try:
    # import itertools.imap as map
    import itertools.izip as zip
except ImportError:
    pass
import numpy as np
import pandas as pd
from ._exceptions import PhysicalMeaningError
from . import _pandas_wrapper
from . import constants
from . import utilities
from . import settings


class common_methods(_pandas_wrapper.core):
    def add_data(self, list_of_columns=None, inplace=False):
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
            list_of_columns (str): You can pass also just one value.
                E.g. ``'mass'`` is equivalent to ``['mass']``. If
                ``list_of_columns`` is ``None`` all available data
                is returned.
            inplace (bool):

        Returns:
            Cartesian:
        """
        data = constants.elements
        
        frame = self.frame.copy()

        list_of_columns = (
            data.columns if (list_of_columns is None) else list_of_columns)

        atom_symbols = frame['atom']
        new_columns = data.loc[atom_symbols, list_of_columns]
        new_columns.index = frame.index
        frame = pd.concat([frame, new_columns], axis=1)

        if inplace:
            self.frame = frame
        else:
            return self.__class__(frame)


    def total_mass(self):
        """Returns the total mass in g/mol.

        Args:
            None

        Returns:
            float:
        """
        mass_molecule = self.add_data('mass')
        mass = mass_molecule[:, 'mass'].sum()
        return mass
