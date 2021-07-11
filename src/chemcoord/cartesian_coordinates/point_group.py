# -*- coding: utf-8 -*-
from chemcoord import export

@export
class PointGroupOperations(list):
    """Defines a point group as sequence of symmetry operations.

    Args:
        sch_symbol (str): Schoenflies symbol of the point group.
        operations (:class:`numpy.ndarray`):
            Initial set of symmetry operations. It is
            sufficient to provide only just enough operations to generate
            the full set of symmetries.
        tolerance (float): Tolerance to generate the full set of symmetry
            operations.
    """
    def __init__(self, sch_symbol, operations, tolerance=0.1):
        from pymatgen.symmetry.analyzer import generate_full_symmops
        self.sch_symbol = sch_symbol
        super(PointGroupOperations, self).__init__(
            [op.rotation_matrix
             for op in generate_full_symmops(operations, tolerance)])

    def __str__(self):
        return self.sch_symbol

    def __repr__(self):
        return self.__str__()
