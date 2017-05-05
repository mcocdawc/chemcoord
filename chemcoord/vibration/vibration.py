from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import six
try:
    # import itertools.imap as map
    import itertools.izip as zip
except ImportError:
    pass
import numpy as np
import pandas as pd
import collections
from threading import Thread
import subprocess
import os
import tempfile
import warnings
from . import _common_class
from ._exceptions import PhysicalMeaningError
from . import utilities
from . import zmat_functions
from . import xyz_functions
from . import export
from .configuration import settings
import io
from io import open
import re

# TODO Change perhaps the _give_displacement representation to np.array
# and rely on same indices.
# Physically this would make sense.


def through(functions, value, recurse_level=np.infty):
    """Calls each function in a list

    This function is passing the value as argument to each function
    in a list.
    For a one dimensional list consisting of
    only functions it is equivalent to::

    [f(value) for f in list]

    If an element is not callable and not iterable, it is not called.
    An iterable is either directly appended or will be recursed through
    depending on recurse_level.

    Args:
        functions (list):
        value (any type):
        recurse_level (int):

    Returns:
        list:
    """
    if callable(functions):
        return functions(value)
    elif (recurse_level < 0 or not hasattr(functions, '__iter__')
          or isinstance(functions, six.string_types)):
        return functions
    else:
        return [through(f, value, recurse_level - 1) for f in functions]


class mode(object):
    def __init__(self, equilibrium_structure, calculate_structure):
        self.eq_structure = {}
        if xyz_functions.is_Cartesian(equilibrium_structure):
            self.eq_structure['xyz'] = equilibrium_structure
            self.eq_structure['zmat'] = equilibrium_structure.to_zmat()
        else:
            self.eq_structure['xyz'] = equilibrium_structure.to_xyz()
            self.eq_structure['zmat'] = equilibrium_structure
        self.buildlist = self.eq_structure['zmat'].get_buildlist()

        self._give_displacement = calculate_structure

    def _repr_html_(self):
        try:
            columns = ('bond', 'angle', 'dihedral')
            frame = self._give_displacement
            selection = (~(frame.loc[:, columns] == 0)
                         & ~frame.loc[:, columns].isnull()).any(axis=1)
            return self._give_displacement[selection]._repr_html_()
        except AttributeError:
            pass

    @classmethod
    def interpolate(cls, eq_strct_zmat, displaced_zmats,
                    fit_function='default'):
        """Interpolate a vibration mode.

        Args:
            eq_strct_zmat (Zmat): This is the equilibrium structure from which
                the vibration starts.
            displaced_zmats (list): The displaced_zmats are a list of tuples.
                The first element of the tuple is a parameter $t \in [-1, 1]$.
                the second element is a Zmat.
                The variable $t$ is implicitly assumed to be zero for the
                equilibrium structure and parametrises the vibrational mode.
                This means that the most distorted structure in one direction
                corresponds to $t=1$ and the most distorted structure in the
                other direction corresponds to $t=-1$.
                The values in between $t \in (-1, 1)$  define the exact
                parametrisation; depending on the fit function this may lead
                to the same curves.
            fit_function (fit_function): The default fit_function is a
                polynomial fit with the length of displaced_zmats as degree.
        Returns:
            mode:
        """

        def check_index(eq_strct_zmat, displaced_zmats):
            for t, zmat in displaced_zmats:
                if not set(eq_strct_zmat.index) == set(zmat.index):
                    return False
            return True

        def check_input(eq_strct_zmat, displaced_zmats):
            index = eq_strct_zmat.index
            buildlist = eq_strct_zmat.get_buildlist()
            for t, zmat in displaced_zmats:
                if not (buildlist == zmat[index, :].get_buildlist()).all():
                    return False
            return True

        def give_arrays(eq_strct_zmat, displaced_zmats):
            X = np.array([0] + [item[0] for item in displaced_zmats])
            columns = ('bond', 'angle', 'dihedral')
            Y = ([eq_strct_zmat[:, columns].values]
                 + [item[1][eq_strct_zmat.index, columns].values for
                    item in displaced_zmats]
                 )
            Y = np.concatenate([A[None, :, :] for A in Y], axis=0)
            return X, Y

        def default_fit_function(eq_strct_zmat, displaced_zmats):
            """Interpolate a vibration mode.

            It is assumed, that eq_strct_zmat and displaced_zmats contain
            physically valid input.

            Args:
                eq_strct_zmat (Zmat): This is the equilibrium structure
                    from which the vibration starts.
                displaced_zmats (list): The displaced_zmats are a list of
                    tuples.
                    The first element of the tuple is a parameter
                    $t \in [-1, 1]$.
                    The second element is a Zmat.
                    The variable $t$ is implicitly assumed to
                    be zero for the equilibrium structure and
                    parametrises the vibrational mode.
                    This means that the most distorted structure in one
                    direction corresponds to $t=1$ and the most distorted
                    structure in the other direction corresponds
                    to $t=-1$. The values in between $t \in (-1, 1)$
                    define the exact
                    parametrisation; depending on the fit function
                    this may lead
                    to the same curves.

            Returns:
                mode:
            """
            fit = np.polynomial.Polynomial.fit
            columns = ('bond', 'angle', 'dihedral')

            X, Y = give_arrays(eq_strct_zmat, displaced_zmats)
            degree = Y.shape[0] - 1

            n_coord = Y.shape[2]
            assert (n_coord == 3)
            functions = np.empty((eq_strct_zmat.n_atoms, n_coord), dtype='O')
            for i in range(eq_strct_zmat.n_atoms):
                for j in range(n_coord):
                    if (~np.isnan(Y[:, i, j])).all():
                        if settings['show_warnings']['polynomial_fit']:
                            P = fit(X, Y[:, i, j], degree)
                            if np.isclose(P.deriv().coef, 0).all():
                                P = 0
                            else:
                                P = P - P.coef[0]
                            functions[i, j] = P
                        else:
                            with warnings.catch_warnings():
                                match = "The fit may be poorly"
                                warnings.filterwarnings("ignore", match)
                                P = fit(X, Y[:, i, j], degree)
                                if np.isclose(P.deriv().coef, 0).all():
                                    P = 0
                                else:
                                    P = P - P(0)
                                functions[i, j] = P
                    else:
                        functions[i, j] = np.nan
            # TODO write nicer
            tmp = eq_strct_zmat.frame.copy()
            tmp.loc[:, columns] = functions
            functions = tmp
            return functions

        if not check_index(eq_strct_zmat, displaced_zmats):
            error_message = 'All indices have to match'
            raise PhysicalMeaningError(error_message)

        if not check_input(eq_strct_zmat, displaced_zmats):
            error_message = ('All Zmatrices require the same index and '
                             + 'the atom of each index number has to match.')
            raise PhysicalMeaningError(error_message)

        if fit_function == 'default':
            calculate_structure = default_fit_function(eq_strct_zmat,
                                                       displaced_zmats)

        return cls(eq_strct_zmat, calculate_structure)

    def give_structure(self, t):
        """Calculates a concrete structure.

        Args:
            t (float): A real number that parametrises the movement along
                the mode.
                The equilibrium structure corresponds to :math:`t=0`,
                the rightmost distortion to :math:`t=1` and
                the leftmost distortion to :math:`t=-1`.

        Returns:
            Zmat:
        """
        columns = ('bond', 'angle', 'dihedral')
        new_structure = self.eq_structure['zmat'].copy()
        give_displacements = self._give_displacement.loc[:, columns].values
        displacements = np.array(through(give_displacements, t))
        new_structure[:, columns] = new_structure[:, columns] + displacements
        return new_structure

    def copy(self):
        pass

    def __mul__(self, other):
        new_mode = self._give_displacement.copy()
        columns = ('bond', 'angle', 'dihedral')
        new_mode.loc[:, columns] = (self._give_displacement.loc[:, columns]
                                    * other)
        return self.__class__(self.eq_structure['zmat'].copy(), new_mode)

    def __rmul__(self, other):
        new_mode = self._give_displacement.copy()
        columns = ('bond', 'angle', 'dihedral')
        new_mode.loc[:, columns] = (self._give_displacement.loc[:, columns]
                                    * other)
        return self.__class__(self.eq_structure['zmat'].copy(), new_mode)

    def __add__(self, other):
        if self.eq_structure['zmat'] != other.eq_structure['zmat']:
            message = 'Only defined for the exact same equilibrium structure'
            raise PhysicalMeaningError(message)
        new_mode = self._give_displacement.copy()
        columns = ('bond', 'angle', 'dihedral')
        new_mode.loc[:, columns] = (self._give_displacement.loc[:, columns]
                                    + other._give_displacement.loc[:, columns])
        return self.__class__(self.eq_structure['zmat'].copy(), new_mode)

    def __radd__(self, other):
        if self.eq_structure['zmat'] != other.eq_structure['zmat']:
            message = 'Only defined for the exact same equilibrium structure'
            raise PhysicalMeaningError(message)
        new_mode = self._give_displacement.copy()
        columns = ('bond', 'angle', 'dihedral')
        new_mode.loc[:, columns] = (self._give_displacement.loc[:, columns]
                                    + other._give_displacement.loc[:, columns])
        return self.__class__(self.eq_structure['zmat'].copy(), new_mode)


def give_distortions(vib, screen_modes=True, step_size=1e-4):
    """Returns a dictionary of distorted structures for each mode.

    Args:
        vib (:class:`ase.vibration.Vibrations`):
        screen_modes (bool): Translations and rotations are filtered out.
        step_size (float): Defines how far to follow the mode in Cartesian
            coordinates in each direction to make the distortions that will
            be used by chemcoord. Smaller should be better, as long as we
            avoid numerical fluctuations.
    Returns:
        dict: The keys of the dictionary are integers, corresponding to each
        mode in the vib instance.
        The value for each key is a 2-tuple consisting of a
        :class:`~.xyz_functions.Cartesian` instance and a list of 2-tuples.
        The first element of the tuples in the list
        is a float denoting the mode parameter :math:`t`,
        the second element is the distorted structure corresponding to the
        value of :math:`t`.

        See an example which gives the equilibrium structure and distorted
        structures for the 5th mode in a given vib instance::

            In: give_distortions(vib)[5]

            Out: equilibrium_structure, [
                (1, leftmost_distorted_structure),
                (-1, rightmost_distorted_structure)]
    """

    # Positions of all the atoms in the molecule used for the
    # vibrational analysis given as
    # np.array([[x_1, y_1, z_1], [x_2, y_2, z_2], ...])
    groundstate_positions = vib.atoms.get_positions()

    mode_dict = {}
    for n in range(len(vib.hnu)):
        # Temperature * ase.units.kB
        kT = 300 * 8.617330337217213e-05

        # Mode from ase.vibrations object
        mode = vib.get_mode(n) * np.sqrt(kT / abs(vib.hnu[n]))
        mode_dict[n] = [
            groundstate_positions,
            (-step_size, groundstate_positions - step_size*mode),
            (step_size, groundstate_positions + step_size*mode)]

    return mode_dict


# class vibration(object):
#     def __init__(equilibrium_structure):
#         self.equilibrium_structure =
#         pass
#
#     def get_mode(self):
#         pass
#
#     def add_mode(self):
#         pass
