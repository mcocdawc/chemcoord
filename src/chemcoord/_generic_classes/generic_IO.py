# -*- coding: utf-8 -*-
import sympy
import numpy as np


class GenericIO(object):
    def _sympy_formatter(self):
        def formatter(x):
            if (isinstance(x, sympy.Basic)):
                return '${}$'.format(sympy.latex(x))
            else:
                return x
        new = self.copy()
        for col in self.columns.drop('atom'):
            if self[col].dtype == np.dtype('O'):
                new._frame.loc[:, col] = self[col].apply(formatter)
        return new
