# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

import numpy as np
import pandas as pd

from chemcoord.cartesian_coordinates.cartesian_class_main import Cartesian


class AsymmetricUnitCartesian(Cartesian):
    """The main class for manipulating cartesian coordinates while preserving
    the point group

    This class has all the methods of a :class:`~Cartesian`, with an additional
    :meth:`~AssymetricUnitCartesian.get_cartesian` method, that returns.
    """
    def get_cartesian(self):
        coords = ['x', 'y', 'z']
        eq_sets = self._metadata['eq']['eq_sets']
        sym_ops = self._metadata['eq']['sym_ops']
        frame = pd.DataFrame(index=[i for v in eq_sets.values() for i in v],
                             columns=['atom', 'x', 'y', 'z'])
        frame['atom'] = pd.Series(
            {i: self.loc[k, 'atom'] for k, v in eq_sets.items() for i in v})
        frame.loc[self.index, coords] = self.loc[:, coords]
        for i in eq_sets:
            for j in eq_sets[i]:
                frame.loc[j, coords] = np.dot(sym_ops[i][j],
                                              frame.loc[i, coords])
        return Cartesian(frame)
