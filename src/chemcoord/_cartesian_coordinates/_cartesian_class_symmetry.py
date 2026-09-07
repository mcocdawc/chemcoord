import warnings
from contextlib import contextmanager

from numpy.exceptions import ComplexWarning

from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord._cartesian_coordinates.point_group import PointGroupOperations


@contextmanager
def _tolerate_complex_roundoff():
    """Silence :class:`numpy.exceptions.ComplexWarning` raised inside pymatgen.

    :class:`pymatgen.symmetry.analyzer.PointGroupAnalyzer` diagonalises the (real
    symmetric) inertia tensor with :func:`numpy.linalg.eig`, i.e. with the general
    LAPACK driver. Depending on the BLAS/LAPACK implementation this returns complex
    arrays whose imaginary parts are pure round-off noise, and pymatgen then warns
    while casting the principal axes back into its real affine matrices. Discarding
    those imaginary parts is exactly the right thing to do here, so the warning is
    noise as well -- but it aborts the symmetry detection for anyone who runs with
    warnings turned into errors.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        yield


class CartesianSymmetry(CartesianCore):
    def _get_point_group_analyzer(self, tolerance=0.3):
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer  # noqa: PLC0415

        with _tolerate_complex_roundoff():
            return PointGroupAnalyzer(self.get_pymatgen_molecule(), tolerance=tolerance)

    def _convert_eq(self, eq):
        """WORKS INPLACE on eq"""
        rename = dict(enumerate(self.index))
        eq["eq_sets"] = {
            rename[k]: {rename[x] for x in v} for k, v in eq["eq_sets"].items()
        }
        eq["sym_ops"] = {
            rename[k]: {rename[x]: v[x] for x in v} for k, v in eq["sym_ops"].items()
        }
        try:
            sym_mol = self.from_pymatgen_molecule(eq["sym_mol"])
            sym_mol.index = self.index
            eq["sym_mol"] = sym_mol
        except KeyError:
            pass

    def get_pointgroup(self, tolerance=0.3):
        """Returns a PointGroup object for the molecule.

        Args:
            tolerance (float): Tolerance to generate the full set of symmetry
                operations.

        Returns:
            :class:`~PointGroupOperations`

        """
        PA = self._get_point_group_analyzer(tolerance=tolerance)
        return PointGroupOperations(PA.sch_symbol, PA.symmops)

    def get_equivalent_atoms(self, tolerance=0.3):
        """Returns sets of equivalent atoms with symmetry operations

        Args:
            tolerance (float): Tolerance to generate the full set of symmetry
                operations.

        Returns:
            dict: The returned dictionary has two possible keys:

            ``eq_sets``:
            A dictionary of indices mapping to sets of indices,
            each key maps to indices of all equivalent atoms.
            The keys are guaranteed to be not equivalent.

            ``sym_ops``:
            Twofold nested dictionary.
            ``operations[i][j]`` gives the symmetry operation
            that maps atom ``i`` unto ``j``.
        """
        PA = self._get_point_group_analyzer(tolerance=tolerance)
        eq = PA.get_equivalent_atoms()
        self._convert_eq(eq)
        return eq

    def symmetrize(self, max_n=10, tolerance=0.3, epsilon=1e-3):
        """Returns a symmetrized molecule

        The equivalent atoms obtained via
        :meth:`~Cartesian.get_equivalent_atoms`
        are rotated, mirrored... unto one position.
        Then the average position is calculated.
        The average position is rotated, mirrored... back with the inverse
        of the previous symmetry operations, which gives the
        symmetrized molecule.
        This operation is repeated iteratively ``max_n`` times at maximum
        until the difference between subsequently symmetrized structures is
        smaller than ``epsilon``.

        Args:
            max_n (int): Maximum number of iterations.
            tolerance (float): Tolerance for detecting symmetry.
                Gets passed as Argument into
                :class:`pymatgen.analyzer.symmetry.PointGroupAnalyzer`.
            epsilon (float): If the elementwise absolute difference of two
                subsequently symmetrized structures is smaller epsilon,
                the iteration stops before ``max_n`` is reached.

        Returns:
            dict: The returned dictionary has three possible keys:

            ``sym_mol``:
            A symmetrized molecule :class:`~Cartesian`

            ``eq_sets``:
            A dictionary of indices mapping to sets of indices,
            each key maps to indices of all equivalent atoms.
            The keys are guaranteed to be not symmetry-equivalent.

            ``sym_ops``:
            Twofold nested dictionary.
            ``operations[i][j]`` gives the symmetry operation
            that maps atom ``i`` unto ``j``.
        """
        from pymatgen.symmetry.analyzer import iterative_symmetrize  # noqa: PLC0415

        mg_mol = self.get_pymatgen_molecule()
        with _tolerate_complex_roundoff():
            eq = iterative_symmetrize(
                mg_mol, max_n=max_n, tolerance=tolerance, epsilon=epsilon
            )
        self._convert_eq(eq)
        return eq

    def get_asymmetric_unit(self, eq=None):
        eq = self.get_equivalent_atoms() if (eq is None) else eq
        new_frame = self.loc[eq["eq_sets"].keys(), :]._frame
        from chemcoord._cartesian_coordinates.asymmetric_unit_cartesian_class import (  # noqa: PLC0415
            AsymmetricUnitCartesian,
        )

        return AsymmetricUnitCartesian(new_frame, _metadata={"eq": eq})
