from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import six
import pandas as pd
import numpy as np
from chemcoord._generic_classes._pandas_wrapper import _pandas_wrapper
import chemcoord.constants as constants


class _common_class(_pandas_wrapper):
    """This class provides methods which are used by Zmat and Cartesian.
    """

    _required_cols = set(['atom'])

    def _return_appropiate_type(self, selected):
        if isinstance(selected, pd.Series):
            frame = pd.DataFrame(selected).T
            if self._required_cols <= set(frame.columns):
                selected = frame
            else:
                return selected
        if (isinstance(selected, pd.DataFrame)
                and self._required_cols <= set(selected.columns)):
            return self.__class__(selected)
        else:
            return selected

    def loc_set_copy(self, key, value):
        """asdf
        """
        new = self.copy()
        if pd.api.types.is_list_like(key):
            new.loc[key[0], key[1]] = value
        else:
            new.loc[key] = value
        return new

    def iloc_set_copy(self, key, value):
        """asdf
        """
        new = self.copy()
        if pd.api.types.is_list_like(key):
            new.iloc[key[0], key[1]] = value
        else:
            new.iloc[key] = value
        return new

    @property
    def n_atoms(self):
        return self.shape[0]

    def add_data(self, list_of_columns=None):
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

        list_of_columns = (
            data.columns if (list_of_columns is None) else list_of_columns)

        if isinstance(list_of_columns, six.string_types):
            assert list_of_columns not in set(self.columns), \
                'Column is already present'
        else:
            for column in list_of_columns:
                assert column not in set(self.columns), \
                    'Column is already present'

        atom_symbols = self.loc[:, 'atom']
        new_columns = data.loc[atom_symbols, list_of_columns]
        new_columns.index = self.index

        return self.__class__(pd.concat([self.frame, new_columns], axis=1))

    def total_mass(self):
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

    def _convert_nan_int(self):
        """ The following functions are necessary to deal with the fact,
        that pandas does not support "NaN" for integers.
        It was written by the user LondonRob at StackExchange:
        http://stackoverflow.com/questions/25789354/
        exporting-ints-with-missing-values-to-csv-in-pandas/31208873#31208873
        Begin of the copied code snippet
        """
        COULD_BE_ANY_INTEGER = 0

        def _lost_precision(s):
            """
            The total amount of precision lost over Series `s`
            during conversion to int64 dtype
            """
            try:
                diff = (s - s.fillna(COULD_BE_ANY_INTEGER).astype(np.int64))
                return diff.sum()
            except ValueError:
                return np.nan

        def _nansafe_integer_convert(s, epsilon=1e-9):
            """
            Convert Series `s` to an object type with `np.nan`
            represented as an empty string ""
            """
            if _lost_precision(s) < epsilon:
                # Here's where the magic happens
                as_object = s.fillna(COULD_BE_ANY_INTEGER)
                as_object = as_object.astype(np.int64).astype(np.object)
                as_object[s.isnull()] = "nan"
                return as_object
            else:
                return s
        return self.apply(_nansafe_integer_convert)
