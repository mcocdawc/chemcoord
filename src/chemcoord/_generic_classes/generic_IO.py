# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
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
