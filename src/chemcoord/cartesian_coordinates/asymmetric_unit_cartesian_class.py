# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian


class AsymmetricUnitCartesian(Cartesian):
    """Manipulate cartesian coordinates while preserving the point group.

    This class has all the methods of a :class:`~Cartesian`, with
    one additional :meth:`~AsymmetricUnitCartesian.get_cartesian` method
    and contains only one member of each symmetry equivalence class.
    """

    def get_cartesian(self):
        """Return a :class:`~Cartesian` where all
        members of a symmetry equivalence class are inserted back in.

        Args:
            None

        Returns:
            Cartesian: A new cartesian instance.
        """
        coords = ['x', 'y', 'z']
        eq_sets = self._metadata['eq']['eq_sets']
        sym_ops = self._metadata['eq']['sym_ops']
        frame = pd.DataFrame(index=[i for v in eq_sets.values() for i in v],
                             columns=['atom', 'x', 'y', 'z'], dtype='f8')
        frame['atom'] = pd.Series(
            {i: self.loc[k, 'atom'] for k, v in eq_sets.items() for i in v})
        frame.loc[self.index, coords] = self.loc[:, coords]
        for i in eq_sets:
            for j in eq_sets[i]:
                frame.loc[j, coords] = np.dot(sym_ops[i][j],
                                              frame.loc[i, coords])
        return Cartesian(frame)
