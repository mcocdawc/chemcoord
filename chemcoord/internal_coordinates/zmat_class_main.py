from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from chemcoord.internal_coordinates.zmat_class_io import Zmat_io
from chemcoord.internal_coordinates.zmat_class_to_cartesian \
    import Zmat_to_cartesian


class Zmat(Zmat_io, Zmat_to_cartesian):
    """The main class for dealing with internal coordinates.
    """
