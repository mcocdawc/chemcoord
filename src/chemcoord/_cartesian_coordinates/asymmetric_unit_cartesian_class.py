import pandas as pd

from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import COORDS
from chemcoord._cartesian_coordinates.cartesian_class_main import Cartesian


class AsymmetricUnitCartesian(Cartesian):
    """Manipulate cartesian coordinates while preserving the point group.

    This class has all the methods of a :class:`~Cartesian`, with
    one additional :meth:`~AsymmetricUnitCartesian.get_cartesian` method
    and contains only one member of each symmetry equivalence class.
    """

    def get_cartesian(self) -> Cartesian:
        """Return a :class:`~Cartesian` where all
        members of a symmetry equivalence class are inserted back in.

        Args:
            None

        Returns:
            Cartesian: A new cartesian instance.
        """
        eq_sets = self._metadata["eq"]["eq_sets"]
        sym_ops = self._metadata["eq"]["sym_ops"]
        frame = pd.DataFrame(
            index=[i for v in eq_sets.values() for i in v],
            columns=["atom", "x", "y", "z"],
            dtype="f8",
        )
        frame["atom"] = pd.Series(
            {i: self.loc[k, "atom"] for k, v in eq_sets.items() for i in v}
        )
        frame.loc[self.index, COORDS] = self.loc[:, COORDS].values
        for i in eq_sets:
            for j in eq_sets[i]:
                frame.loc[j, COORDS] = sym_ops[i][j] @ frame.loc[i, COORDS]
        return Cartesian(frame)
