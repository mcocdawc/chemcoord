import numpy as np
import sympy
from typing_extensions import Self


class GenericIO:
    def _sympy_formatter(self) -> Self:
        def formatter(x):
            if isinstance(x, sympy.Basic):
                return f"${sympy.latex(x)}$"
            else:
                return x

        new = self.copy()  # type: ignore[attr-defined]
        for col in self.columns.drop("atom"):  # type: ignore[attr-defined]
            if self[col].dtype == np.dtype("O"):  # type: ignore[index]
                new._frame.loc[:, col] = self[col].apply(formatter)  # type: ignore[index]
        return new
