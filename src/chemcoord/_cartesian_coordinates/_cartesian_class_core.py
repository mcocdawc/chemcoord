import copy
import itertools
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from itertools import product
from typing import Any, Final, Literal, TypeVar, cast, overload

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from sortedcontainers import SortedSet
from typing_extensions import Self, assert_never

import chemcoord._cartesian_coordinates.xyz_functions as xyz_functions
import chemcoord.constants as constants
from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import (
    COORDS,
    PandasWrapper,
)
from chemcoord._cartesian_coordinates._indexers import QueryFunction
from chemcoord._generic_classes.generic_core import GenericCore
from chemcoord._utilities._decorators import njit
from chemcoord.configuration import settings
from chemcoord.constants import RestoreElementData, elements
from chemcoord.exceptions import PhysicalMeaning
from chemcoord.typing import (
    ArithmeticOther,
    AtomIdx,
    Axes,
    BondDict,
    Integral,
    Matrix,
    Real,
    SequenceNotStr,
    T,
    Tensor3D,
    Vector,
)


class CartesianCore(PandasWrapper, GenericCore):  # noqa: PLW1641
    # Look into the numpy manual for description of __array_priority__:
    # https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.classes.html
    __array_priority__ = 15.0

    @classmethod
    def set_atom_coords(
        cls,
        atoms: Sequence[str],
        coords: Matrix,
        index: Axes | None = None,
    ) -> Self:
        dtypes = [("atom", str), ("x", float), ("y", float), ("z", float)]
        frame = DataFrame(np.empty(len(atoms), dtype=dtypes), index=index)
        frame["atom"] = atoms
        frame.loc[:, COORDS] = coords
        return cls(frame)

    def _return_appropiate_type(
        self, selected: Series | DataFrame
    ) -> Self | Series | DataFrame:
        if isinstance(selected, Series):
            frame = DataFrame(selected).T
            if self._required_cols <= set(frame.columns):
                selected = frame.apply(pd.to_numeric, errors="ignore")
            else:
                return selected

        if isinstance(selected, DataFrame) and self._required_cols <= set(
            selected.columns
        ):
            molecule = self.__class__(selected)
            molecule.metadata = self.metadata.copy()
            molecule._metadata = copy.deepcopy(self._metadata)
            return molecule
        else:
            return selected

    def _test_if_can_be_added(self, other: Self) -> None:
        if not (
            set(self.index) == set(other.index)
            and (self["atom"] == other.loc[self.index, "atom"]).all(axis=None)
        ):
            message = (
                "You can add only Cartesians which are indexed in the "
                "same way and use the same atoms."
            )
            raise PhysicalMeaning(message)

    def __add__(self, other: Self | ArithmeticOther) -> Self:
        new = self.copy()
        if isinstance(other, self.__class__):
            self._test_if_can_be_added(other)
            new.loc[:, COORDS] = self.loc[:, COORDS] + other.loc[:, COORDS]
        elif isinstance(other, DataFrame):
            new.loc[:, COORDS] = self.loc[:, COORDS] + other.loc[:, COORDS]
        else:
            try:
                other = np.array(other, dtype="f8")
            except TypeError:
                pass
            new.loc[:, COORDS] = self.loc[:, COORDS] + other
        return new

    def __radd__(self, other: Self | ArithmeticOther) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Self | ArithmeticOther) -> Self:
        new = self.copy()
        if isinstance(other, self.__class__):
            self._test_if_can_be_added(other)
            new.loc[:, COORDS] = self.loc[:, COORDS] - other.loc[:, COORDS]
        elif isinstance(other, DataFrame):
            new.loc[:, COORDS] = self.loc[:, COORDS] - other.loc[:, COORDS]
        else:
            try:
                other = np.array(other, dtype="f8")
            except TypeError:
                pass
            new.loc[:, COORDS] = self.loc[:, COORDS] - other
        return new

    def __rsub__(self, other: Self | ArithmeticOther) -> Self:
        new = self.copy()
        if isinstance(other, self.__class__):
            self._test_if_can_be_added(other)
            new.loc[:, COORDS] = other.loc[:, COORDS] - self.loc[:, COORDS]
        elif isinstance(other, DataFrame):
            new.loc[:, COORDS] = other.loc[:, COORDS] - self.loc[:, COORDS]
        else:
            try:
                other = np.array(other, dtype="f8")
            except TypeError:
                pass
            new.loc[:, COORDS] = other - self.loc[:, COORDS]
        return new

    def __mul__(self, other: Self | ArithmeticOther) -> Self:
        new = self.copy()
        if isinstance(other, self.__class__):
            self._test_if_can_be_added(other)
            new.loc[:, COORDS] = self.loc[:, COORDS] * other.loc[:, COORDS]
        elif isinstance(other, DataFrame):
            new.loc[:, COORDS] = self.loc[:, COORDS] * other.loc[:, COORDS]
        else:
            try:
                other = np.array(other, dtype="f8")
            except TypeError:
                pass
            new.loc[:, COORDS] = self.loc[:, COORDS] * other
        return new

    def __rmul__(self, other: Self | ArithmeticOther) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: Self | ArithmeticOther) -> Self:
        new = self.copy()
        if isinstance(other, self.__class__):
            self._test_if_can_be_added(other)
            new.loc[:, COORDS] = (
                self.loc[:, COORDS].values / other.loc[:, COORDS].values
            )
        elif isinstance(other, DataFrame):
            new.loc[:, COORDS] = self.loc[:, COORDS] / other.loc[:, COORDS]
        else:
            try:
                other = np.array(other, dtype="f8")
            except TypeError:
                pass
            new.loc[:, COORDS] = self.loc[:, COORDS].values / other
        return new

    def __rtruediv__(self, other: Self | ArithmeticOther) -> Self:
        new = self.copy()
        if isinstance(other, self.__class__):
            self._test_if_can_be_added(other)
            new.loc[:, COORDS] = (
                other.loc[:, COORDS].values / self.loc[:, COORDS].values
            )
        elif isinstance(other, DataFrame):
            new.loc[:, COORDS] = other.loc[:, COORDS] / self.loc[:, COORDS]
        else:
            try:
                other = np.array(other, dtype="f8")
            except TypeError:
                pass
            new.loc[:, COORDS] = other / self.loc[:, COORDS].values
        return new

    def __pow__(self, other: Self | ArithmeticOther) -> Self:
        new = self.copy()
        new.loc[:, COORDS] = self.loc[:, COORDS] ** other
        return new

    def __pos__(self) -> Self:
        return self.copy()

    def __neg__(self) -> Self:
        return -1 * self.copy()

    def __abs__(self) -> Self:
        new = self.copy()
        new.loc[:, COORDS] = abs(new.loc[:, COORDS])
        return new

    def __matmul__(self, other: Matrix) -> Self:
        return NotImplemented

    # Here we get a type error, which is fine and expected. Ignore it.
    #   Signatures of "__rmatmul__" of "CartesianCore" and "__matmul__" of
    #   "ndarray[tuple[int, ...], dtype[Any]]" are unsafely overlapping
    def __rmatmul__(self, other: Matrix) -> Self:  # type: ignore[misc]
        new = self.copy()
        new.loc[:, COORDS] = (np.dot(other, new.loc[:, COORDS].values.T)).T
        return new

    # Somehow the base class `object` expects the return type to be `bool``
    #  but the correct type hint is DataFrame.
    # Ignore this override error.
    def __eq__(self, other: Self) -> DataFrame:  # type: ignore[override]
        return self._frame == other._frame

    def __ne__(self, other: Self) -> DataFrame:  # type: ignore[override]
        return self._frame != other._frame

    def copy(self) -> Self:
        molecule = self.__class__(self._frame)
        molecule.metadata = self.metadata.copy()
        molecule._metadata = copy.deepcopy(self._metadata)
        return molecule

    def subs(self, *args) -> Self:  # type: ignore[no-untyped-def]
        """Substitute a symbolic expression in ``['x', 'y', 'z']``

        This is a wrapper around the substitution mechanism of
        `sympy <http://docs.sympy.org/latest/tutorial/basic_operations.html>`_.
        Any symbolic expression in the columns
        ``['x', 'y', 'z']`` of ``self`` will be substituted
        with value.

        Args:
            symb_expr (sympy expression):
            value :
            perform_checks (bool): If ``perform_checks is True``,
                it is asserted, that the resulting Zmatrix can be converted
                to cartesian coordinates.
                Dummy atoms will be inserted automatically if necessary.

        Returns:
            Cartesian: Cartesian with substituted symbolic expressions.
            If all resulting sympy expressions in a column are numbers,
            the column is recasted to 64bit float.
        """
        out = self.copy()

        def get_subs_f(*args) -> Callable[[T], T | float]:  # type: ignore[no-untyped-def]
            def subs_function(x: T) -> T | float:
                if hasattr(x, "subs"):
                    x = x.subs(*args)
                    try:
                        # We try to convert the expression to a float and return it,
                        # when failing. Hence, we can ignore the following type error.
                        return float(x)  # type: ignore[arg-type]
                    except TypeError:
                        pass
                return x

            return subs_function

        for col in COORDS:
            if out.loc[:, col].dtype is np.dtype("O"):
                out.loc[:, col] = out.loc[:, col].map(get_subs_f(*args))
                try:
                    out._frame = out._frame.astype({col: "f8"})
                except (SystemError, TypeError):
                    pass
        return out

    def assign(
        self,
        idx: Integral
        | Index
        | Set[Integral]
        | Vector
        | SequenceNotStr[Integral]
        | slice
        | Series
        | QueryFunction,
        col: Literal["x", "y", "z"],
        val: Real,
    ) -> Self:
        """Return a copy where the value is assigned.

        Args:
            idx :
            col :
            val :
        """
        new = self.copy()
        new.loc[idx, col] = val
        return new

    @staticmethod
    @njit
    def _jit_give_bond_array(
        pos: Matrix[np.floating],
        bond_radii: Vector[np.floating],
        self_bonding_allowed: bool = False,
    ) -> Matrix[np.bool_]:
        """Calculate a boolean array where ``A[i, j] == True`` indicates a
        bond between the i-th and j-th atom.
        """
        n = pos.shape[0]
        bond_array = np.empty((n, n), dtype=np.bool_)

        for i in range(n):
            for j in range(i, n):
                D = 0
                for h in range(3):
                    D += (pos[i, h] - pos[j, h]) ** 2
                B = (bond_radii[i] + bond_radii[j]) ** 2
                bond_array[i, j] = (B - D) >= 0
                bond_array[j, i] = bond_array[i, j]
        if not self_bonding_allowed:
            for i in range(n):
                bond_array[i, i] = False
        return bond_array

    def _update_bond_dict(
        self,
        fragment_indices: Sequence[AtomIdx],
        positions: Matrix[np.floating],
        bond_radii: Vector[np.floating],
        bond_dict: defaultdict[AtomIdx, set[AtomIdx]],
        self_bonding_allowed: bool = False,
        convert_index: Mapping[AtomIdx, AtomIdx] | None = None,
    ) -> None:
        """This function has side effects and bond_dict has to be a defaultdict(set)"""
        assert isinstance(bond_dict, defaultdict) or bond_dict is None
        fragment_indices = list(fragment_indices)
        if convert_index is None:
            convert_index = dict(
                cast(Iterable[tuple[AtomIdx, AtomIdx]], enumerate(fragment_indices))
            )

        frag_pos = positions[fragment_indices, :]
        frag_bond_radii = bond_radii[fragment_indices]

        bond_array = self._jit_give_bond_array(
            frag_pos, frag_bond_radii, self_bonding_allowed=self_bonding_allowed
        )
        a, b = ([convert_index[i] for i in a] for a in bond_array.nonzero())
        for row, index in enumerate(a):
            bond_dict[index].add(b[row])

    def _divide_et_impera(
        self, n_atoms_per_set: int = 500, offset: float = 3.0
    ) -> Tensor3D[set[AtomIdx]]:  # type: ignore[type-var]
        sorted_series = dict(zip(COORDS, [self[axis].sort_values() for axis in COORDS]))

        def ceil(x: Real) -> int:
            return int(np.ceil(x))

        n_sets = len(self) / n_atoms_per_set
        n_sets_along_axis = ceil(n_sets ** (1 / 3))
        n_atoms_per_set_along_axis = ceil(len(self) / n_sets_along_axis)

        def give_index(
            series: pd.Series,
            i: int,
            n_atoms_per_set_along_axis: int,
            offset: float = offset,
        ) -> set[AtomIdx]:
            N = n_atoms_per_set_along_axis
            try:
                min_value, max_value = series.iloc[[i * N, (i + 1) * N]]  # type: ignore[call-overload]
            except IndexError:
                min_value, max_value = series.iloc[[i * N, -1]]  # type: ignore[call-overload]
            selection = series.between(min_value - offset, max_value + offset)
            return set(series[selection].index)

        indices_at_axis = {
            axis: {
                i: give_index(sorted_series[axis], i, n_atoms_per_set_along_axis)
                for i in range(n_sets_along_axis)
            }
            for axis in COORDS
        }

        array_of_fragments = np.full([n_sets_along_axis] * 3, None, dtype="O")
        for i, j, k in product(*[range(x) for x in array_of_fragments.shape]):
            selection = (
                indices_at_axis["x"][i]
                & indices_at_axis["y"][j]
                & indices_at_axis["z"][k]
            )
            array_of_fragments[i, j, k] = selection
        return array_of_fragments

    def _get_atom_radii(
        self,
        modify_element_data: Real
        | Callable[[Real], Real]
        | Mapping[str, Real]
        | None = None,
        modify_atom_data: Mapping[int, Real] | None = None,
        data_col: str | None = None,
    ) -> Vector[np.float64]:
        if data_col is None:
            data_col = settings.defaults.atomic_radius_data

        with RestoreElementData():
            used_vdW_r = elements.loc[:, data_col]
            if isinstance(modify_element_data, Real):
                elements.loc[:, data_col] = used_vdW_r.map(
                    lambda _: float(modify_element_data)
                )
            elif callable(modify_element_data):
                elements.loc[:, data_col] = used_vdW_r.map(modify_element_data)  # type: ignore[arg-type]
            elif isinstance(modify_element_data, Mapping):
                elements.loc[:, data_col].update(modify_element_data)  # type: ignore[arg-type]
                # assert False, elements.loc["C", atomic_radius_data]
            elif modify_element_data is None:
                pass
            else:
                assert_never(modify_element_data)

            atom_radii = self.add_data(data_col)[data_col]
            if isinstance(modify_atom_data, Mapping):
                atom_radii.update(modify_atom_data)
            elif modify_atom_data is None:
                pass
            else:
                assert_never(modify_atom_data)

        return atom_radii.values

    def get_bonds(
        self,
        *,
        self_bonding_allowed: bool = False,
        offset: float | None = None,
        modify_atom_data: Mapping[int, float] | None = None,
        modify_element_data: Real
        | Callable[[Real], Real]
        | Mapping[str, Real]
        | None = None,
        atomic_radius_data_col: str | None = None,
    ) -> dict[AtomIdx, set[AtomIdx]]:
        """Return a dictionary representing the bonds.

        Args:
            modify_atom_data : If you want to change the van der
                Vaals radius of one or more specific atoms, pass a
                dictionary that looks like
                :python:`modified_properties = {index1: 1.5}`.
                For global changes use the constants module.
            modify_element_data : If you want to temporarily change the global
                tabulated data of the van der Waals radii.
                It is possible to pass:

                * a single number which is used as radius for all atoms,
                * a callable which is applied to all radii and
                  can be used to e.g. scale via :python:`lambda r: r * 1.1`,
                * a dictionary which maps the element symbol to the van der Waals
                  radius, to change the radius of individual elements,
                  e.g. :python:`{"C": 1.5}`.
            offset :
                The offset used to determine the overlap between bins in the
                divide et impera function that is used to avoid the quadratic scaling
                when calculating pair-wise distances.
                If :python:`None`, it will be chosen slightly larger than twice
                the maximum atom radius, which guarantees that no bond is missed because
                of binning.
            self_bonding_allowed (bool):
            atomic_radius_data (str): Defines which column of
                :attr:`constants.elements` is used. The default is
                ``atomic_radius_cc`` and can be changed with
                :attr:`settings.defaults.atomic_radius_data`.
                Compare with :func:`add_data`.

        Returns:
            dict: Dictionary mapping from an atom index to the set of
            indices of atoms bonded to.
        """

        old_index = self.index
        atom_radii = self._get_atom_radii(
            modify_element_data, modify_atom_data, atomic_radius_data_col
        )
        # From now on, we assume a 0,...,n indexed molecule
        # and can use plain numpy integer arrays
        self.index = range(len(self))  # type: ignore[assignment]
        # Choose the offset such that even with maximum van der Waals radius r
        # both atoms (denoted as stars) end up in one bin
        #    *_________|_________*
        #         r         r
        #     ____________________|   left bin
        #   |____________________     right bin
        fragments = self._divide_et_impera(
            offset=2.1 * atom_radii.max() if offset is None else offset
        )
        positions = np.array(self.loc[:, COORDS], order="F")
        bond_dict: defaultdict[AtomIdx, set[AtomIdx]] = defaultdict(set)
        for i, j, k in product(*[range(x) for x in fragments.shape]):
            # The following call is not side effect free and changes
            # bond_dict
            self._update_bond_dict(
                fragments[i, j, k],
                positions,
                atom_radii,
                bond_dict=bond_dict,
                self_bonding_allowed=self_bonding_allowed,
            )

        for i in set(self.index) - set(bond_dict.keys()):
            bond_dict[AtomIdx(i)] = set()

        self.index = old_index
        rename = self.index
        return {
            AtomIdx(int(rename[key])): {AtomIdx(int(rename[i])) for i in bond_dict[key]}
            for key in bond_dict
        }

    def _sort_by_valency(self, bond_dict: BondDict) -> dict[AtomIdx, SortedSet]:
        valency: Final = dict(zip(self.index, self.add_data("valency")["valency"]))
        return {
            key: SortedSet([i for i in bond_dict[key]], key=lambda x: -valency[x])
            for key in bond_dict
        }

    @overload
    def get_coordination_sphere(
        self,
        index_of_atom: AtomIdx,
        *,
        n_sphere: float = ...,
        give_only_index: Literal[False] = False,
        only_surface: bool = ...,
        exclude: set[AtomIdx] | None = ...,
        bond_dict: BondDict | None = ...,
    ) -> Self: ...

    @overload
    def get_coordination_sphere(
        self,
        index_of_atom: AtomIdx,
        *,
        n_sphere: float = ...,
        give_only_index: Literal[True],
        only_surface: bool = ...,
        exclude: set[AtomIdx] | None = ...,
        bond_dict: BondDict | None = ...,
    ) -> set[AtomIdx]: ...

    def get_coordination_sphere(
        self,
        index_of_atom: AtomIdx,
        *,
        n_sphere: float = 1,
        give_only_index: bool = False,
        only_surface: bool = True,
        exclude: set[AtomIdx] | None = None,
        bond_dict: BondDict | None = None,
    ) -> Self | set[AtomIdx]:
        """Return a Cartesian of atoms in the n-th coordination sphere.

        Connected means that a path along covalent bonds exists.

        Args:
            index_of_atom (int):
            n_sphere (float): Determines the number of the coordination sphere.
                Is just a float to allow infinity as input;
                in this case it returns all connected atoms.
            give_only_index (bool): If ``True`` a set of indices is
                returned. Otherwise a new Cartesian instance.
            only_surface (bool): Return only the surface of the coordination
                sphere.
            exclude (set): A set of indices that should be ignored
                for the path finding.

        Returns:
            A set of indices or a new Cartesian instance.
        """
        exclude = set() if exclude is None else exclude
        if bond_dict is None:
            bond_dict = self.get_bonds()
        i = index_of_atom
        if (n_sphere < 0) or (n_sphere < float("inf") and n_sphere != int(n_sphere)):
            raise ValueError(
                "n_sphere must be a non-negative integer or infinity, but is "
                f"{n_sphere}."
            )
        if n_sphere > 0:
            visited = {i} | exclude
            try:
                tmp_bond_dict = {j: (bond_dict[j] - visited) for j in bond_dict[i]}
            except KeyError:
                tmp_bond_dict = {}
            n = 0
            while tmp_bond_dict and (n + 1) < n_sphere:
                new_tmp_bond_dict = {}
                for i in tmp_bond_dict:
                    if i in visited:
                        continue
                    visited.add(i)
                    for j in tmp_bond_dict[i]:
                        new_tmp_bond_dict[j] = bond_dict[j] - visited
                tmp_bond_dict = new_tmp_bond_dict
                n += 1
            if only_surface:
                index_out = set(tmp_bond_dict.keys())
            else:
                index_out = visited | set(tmp_bond_dict.keys())
        else:
            index_out = {i}

        if give_only_index:
            return index_out - exclude
        else:
            return self.loc[index_out - exclude]

    def _preserve_bonds(
        self,
        sliced_cartesian: Self,
        bond_dict: BondDict | None = None,
    ) -> Self:
        """Is called after cutting geometric shapes.

        If you want to change the rules how bonds are preserved, when
            applying e.g. :meth:`Cartesian.cut_sphere` this is the
            function you have to modify.
        It is recommended to inherit from the Cartesian class to
            tailor it for your project, instead of modifying the
            source code of ChemCoord.

        Args:
            sliced_frame (Cartesian):
            bond_dict : A bond dictionary computed via
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            Cartesian:
        """
        if bond_dict is None:
            bond_dict = self.get_bonds()

        included_atoms_set = set(sliced_cartesian.index)
        assert included_atoms_set.issubset(set(self.index)), (
            "The sliced Cartesian has to be a subset of the bigger frame"
        )
        new_atoms: set[AtomIdx] = set()
        for atom in included_atoms_set:
            new_atoms = new_atoms | bond_dict[atom]
        new_atoms = new_atoms - included_atoms_set
        while not new_atoms == set():
            index_of_interest = new_atoms.pop()
            included_atoms_set = included_atoms_set | self.get_coordination_sphere(
                index_of_interest,
                n_sphere=float("inf"),
                only_surface=False,
                exclude=included_atoms_set,
                give_only_index=True,
                bond_dict=bond_dict,
            )
            new_atoms = new_atoms - included_atoms_set
        molecule = self.loc[included_atoms_set, :]
        return molecule

    def _get_origin(
        self, origin: Vector[np.floating] | Series | int | Sequence[Real] | None
    ) -> Vector[np.float64]:
        if origin is None:
            return np.zeros(3)
        elif isinstance(origin, (int, np.integer)):
            return cast(Vector[np.float64], self.loc[origin, COORDS].values)
        else:
            return np.asarray(origin, dtype="f8")

    def cut_sphere(
        self,
        radius: Real = 15.0,
        origin: Vector[np.floating] | AtomIdx | Sequence[Real] | None = None,
        outside_sliced: bool = True,
        preserve_bonds: bool = False,
    ) -> Self:
        """Cut a sphere specified by origin and radius.

        Args:
            radius (float):
            origin (list): By default it is :python:`[0.0, 0.0, 0.0]`,
                you can pass an alternative position or a single atom index.
            outside_sliced (bool): Atoms outside/inside the sphere
                are cut out.
            preserve_bonds (bool): Do not cut covalent bonds.

        Returns:
            Cartesian:
        """
        origin = self._get_origin(origin)

        molecule = self.get_distance_to(origin)
        if outside_sliced:
            molecule = molecule[molecule["distance"] < radius]
        else:
            molecule = molecule[molecule["distance"] > radius]

        if preserve_bonds:
            molecule = self._preserve_bonds(molecule)

        return molecule

    def cut_cuboid(
        self,
        a: float = 20.0,
        b: float | None = None,
        c: float | None = None,
        origin: AtomIdx | Vector[np.floating] | Sequence[Real] | None = None,
        outside_sliced: bool = True,
        preserve_bonds: bool = False,
    ) -> Self:
        """Cut a cuboid specified by edge and radius.

        Args:
            a (float): Value of the a edge.
            b (float): Value of the b edge. Takes value of a if None.
            c (float): Value of the c edge. Takes value of a if None.
            origin (list): Please note that you can also pass an
                integer. In this case it is interpreted as the index
                of the atom which is taken as origin.
            outside_sliced (bool): Atoms outside/inside the sphere are
                cut away.
            preserve_bonds (bool): Do not cut covalent bonds.

        Returns:
            Cartesian:
        """
        origin = self._get_origin(origin)
        b = a if b is None else b
        c = a if c is None else c

        sides = np.array([a, b, c])
        pos = self.loc[:, COORDS].values
        if outside_sliced:
            molecule = self[((pos - origin) / (sides / 2)).max(axis=1) < 1.0]
        else:
            molecule = self[((pos - origin) / (sides / 2)).max(axis=1) > 1.0]

        if preserve_bonds:
            molecule = self._preserve_bonds(molecule)
        return molecule

    def get_centroid(self) -> Vector[np.float64]:
        """Return the average location.

        Args:
            None

        Returns:
            :class:`numpy.ndarray`:
        """
        return np.mean(self.loc[:, COORDS].values, axis=0)

    def get_barycenter(self) -> Vector[np.float64]:
        """Return the mass weighted average location.

        Args:
            None

        Returns:
            :class:`numpy.ndarray`:
        """
        try:
            mass = self["mass"].values
        except KeyError:
            mass = self.add_data("mass")["mass"].values
        pos = self.loc[:, COORDS].values
        return (pos * mass[:, None]).sum(axis=0) / self.get_total_mass()

    def get_bond_lengths(
        self,
        indices: (
            Matrix
            | Sequence[tuple[Integral, Integral] | Sequence[Integral]]
            | DataFrame
        ),
    ) -> Vector[np.float64]:
        """Return the distances between given atoms.

        Calculates the distance between the atoms with
        indices ``i`` and ``b``.
        The indices can be given in two ways:

        * As list of lists: ``[[i1, b1], [i2, b2]...]``
        * As :class:`pd.DataFrame` where ``i`` is taken from the index and
          ``b`` from the respective column ``'b'``.

        Args:
            indices :

        Returns:
            :class:`numpy.ndarray`: Vector of angles in degrees.
        """
        if isinstance(indices, DataFrame):
            i_pos = self.loc[indices.index, COORDS].values
            b_pos = self.loc[indices.loc[:, "b"], COORDS].values
        else:
            indices = np.array(indices)
            if len(indices.shape) == 1:
                indices = indices[None, :]
            i_pos = self.loc[indices[:, 0], COORDS].values
            b_pos = self.loc[indices[:, 1], COORDS].values
        return np.linalg.norm(i_pos - b_pos, axis=1)

    def get_angle_degrees(
        self,
        indices: (
            Matrix
            | Sequence[tuple[Integral, Integral, Integral] | Sequence[Integral]]
            | DataFrame
        ),
    ) -> Vector[np.float64]:
        """Return the angles between given atoms.

        Calculates the angle in degrees between the atoms with
        indices ``i, b, a``.
        The indices can be given in two ways:

        * As list of lists: ``[[i1, b1, a1], [i2, b2, a2]...]``
        * As :class:`pd.DataFrame` where ``i`` is taken from the index and
          ``b`` and ``a`` from the respective columns ``'b'`` and ``'a'``.

        Args:
            indices (list):

        Returns:
            :class:`numpy.ndarray`: Vector of angles in degrees.
        """
        if isinstance(indices, DataFrame):
            i_pos = self.loc[indices.index, COORDS].values
            b_pos = self.loc[indices.loc[:, "b"], COORDS].values
            a_pos = self.loc[indices.loc[:, "a"], COORDS].values
        else:
            indices = np.array(indices)
            if len(indices.shape) == 1:
                indices = indices[None, :]
            i_pos = self.loc[indices[:, 0], COORDS].values
            b_pos = self.loc[indices[:, 1], COORDS].values
            a_pos = self.loc[indices[:, 2], COORDS].values

        BI, BA = i_pos - b_pos, a_pos - b_pos
        bi, ba = (v / np.linalg.norm(v, axis=1)[:, None] for v in (BI, BA))
        dot_product = np.sum(bi * ba, axis=1)
        dot_product[dot_product > 1] = 1
        dot_product[dot_product < -1] = -1
        return np.degrees(np.arccos(dot_product))

    def get_dihedral_degrees(
        self,
        indices: (
            Matrix[np.integer]
            | Sequence[tuple[AtomIdx, AtomIdx, AtomIdx, AtomIdx] | Sequence[AtomIdx]]
            | DataFrame
        ),
        start_row: int = 0,
    ) -> Vector[np.float64]:
        """Return the dihedrals between given atoms.

        Calculates the dihedral angle in degrees between the atoms with
        indices ``i, b, a, d``.
        The indices can be given in two ways:

        * As list of lists: ``[[i1, b1, a1, d1], [i2, b2, a2, d2]...]``
        * As :class:`pandas.DataFrame` where ``i`` is taken from the index and
          ``b``, ``a`` and ``d``from the respective columns
          ``'b'``, ``'a'`` and ``'d'``.

        Args:
            indices (list):

        Returns:
            :class:`numpy.ndarray`: Vector of angles in degrees.
        """
        if isinstance(indices, DataFrame):
            i_pos = self.loc[indices.index, COORDS].values
            b_pos = self.loc[indices.loc[:, "b"], COORDS].values
            a_pos = self.loc[indices.loc[:, "a"], COORDS].values
            d_pos = self.loc[indices.loc[:, "d"], COORDS].values
        else:
            indices = np.array(indices)
            if len(indices.shape) == 1:
                indices = indices[None, :]
            i_pos = self.loc[indices[:, 0], COORDS].values
            b_pos = self.loc[indices[:, 1], COORDS].values
            a_pos = self.loc[indices[:, 2], COORDS].values
            d_pos = self.loc[indices[:, 3], COORDS].values

        IB = b_pos - i_pos
        BA = a_pos - b_pos
        AD = d_pos - a_pos

        N1 = np.cross(IB, BA, axis=1)
        N2 = np.cross(BA, AD, axis=1)
        n1, n2 = (v / np.linalg.norm(v, axis=1)[:, None] for v in (N1, N2))

        dot_product = np.sum(n1 * n2, axis=1)
        dot_product[dot_product > 1] = 1
        dot_product[dot_product < -1] = -1
        dihedrals = np.degrees(np.arccos(dot_product))

        # the next lines are to test the direction of rotation.
        # is a dihedral really 90 or 270 degrees?
        # Equivalent to direction of rotation of dihedral
        where_to_modify = np.sum(BA * np.cross(n1, n2, axis=1), axis=1) > 0
        where_to_modify = np.nonzero(where_to_modify)[0]

        length = indices.shape[0] - start_row
        sign = np.full(length, 1, dtype="float64")
        to_add = np.full(length, 0, dtype="float64")
        sign[where_to_modify] = -1
        to_add[where_to_modify] = 360
        return to_add + sign * dihedrals

    @overload
    def fragmentate(
        self,
        give_only_index: Literal[False] = ...,
        bond_dict: BondDict | None = ...,
    ) -> list[Self]: ...

    @overload
    def fragmentate(
        self,
        give_only_index: Literal[True],
        bond_dict: BondDict | None = ...,
    ) -> list[set[AtomIdx]]: ...

    def fragmentate(
        self,
        give_only_index: bool = False,
        bond_dict: BondDict | None = None,
    ) -> list[Self] | list[set[AtomIdx]]:
        """Get the indices of non bonded parts in the molecule.

        Args:
            give_only_index (bool): If ``True`` a set of indices is returned.
                Otherwise a new Cartesian instance.
            bond_dict : A bond dictionary computed via
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            list: A list of sets of indices or new Cartesian instances.
        """
        if bond_dict is None:
            bond_dict = self.get_bonds()

        fragments: list[set[AtomIdx]] | list[Self] = []
        pending = set(self.index)

        while pending:
            index = self.get_coordination_sphere(
                pending.pop(),
                bond_dict=bond_dict,
                n_sphere=float("inf"),
                only_surface=False,
                give_only_index=True,
            )
            pending = pending - index
            if give_only_index:
                fragments.append(index)  # type: ignore[arg-type]
            else:
                fragment = self.loc[index]
                fragments.append(fragment)  # type: ignore[arg-type]
        return fragments

    _T = TypeVar("_T", Set, SortedSet)

    def restrict_bond_dict(self, bond_dict: Mapping[AtomIdx, _T]) -> dict[AtomIdx, _T]:
        """Restrict a bond dictionary to self.

        Args:
            bond_dict (dict): Look into :meth:`~chemcoord.Cartesian.get_bonds`,
                to see examples for a bond_dict.

        Returns:
            bond dictionary
        """
        return {j: bond_dict[j] & set(self.index) for j in self.index}

    @overload
    def get_fragment(
        self,
        list_of_indextuples: Sequence[tuple[AtomIdx, AtomIdx]],
        give_only_index: Literal[False] = False,
        bond_dict: BondDict | None = None,
    ) -> Self: ...

    @overload
    def get_fragment(
        self,
        list_of_indextuples: Sequence[tuple[AtomIdx, AtomIdx]],
        give_only_index: Literal[True],
        bond_dict: BondDict | None = None,
    ) -> set[AtomIdx]: ...

    def get_fragment(
        self,
        list_of_indextuples: Sequence[tuple[AtomIdx, AtomIdx]],
        give_only_index: bool = False,
        bond_dict: BondDict | None = None,
    ) -> Self | set[AtomIdx]:
        """Get the indices of the atoms in a fragment.

        The list_of_indextuples contains all bondings from the
        molecule to the fragment. ``[(1,3), (2,4)]`` means for example that the
        fragment is connected over two bonds. The first bond is from atom 1 in
        the molecule to atom 3 in the fragment. The second bond is from atom
        2 in the molecule to atom 4 in the fragment.

        Args:
            list_of_indextuples (list):
            give_only_index (bool): If ``True`` a set of indices
                is returned. Otherwise a new Cartesian instance.
            bond_dict : A bond dictionary computed via
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            A set of indices or a new Cartesian instance.
        """
        if bond_dict is None:
            bond_dict = self.get_bonds()

        exclude = [tuple[0] for tuple in list_of_indextuples]

        # we just need one index that is definitely in the fragment
        #  and then find all connected atoms (minus the excluded ones)
        index_of_atom = list_of_indextuples[0][1]
        fragment_index = self.get_coordination_sphere(
            index_of_atom,
            exclude=set(exclude),
            n_sphere=float("inf"),
            only_surface=False,
            give_only_index=True,
            bond_dict=bond_dict,
        )
        if give_only_index:
            return fragment_index
        else:
            return self.loc[fragment_index, :]

    def get_without(
        self,
        fragments: Self | Sequence[Self],
        bond_dict: BondDict | None = None,
    ) -> list[Self]:
        """Return self without the specified fragments.

        Args:
            fragments: Either a list of :class:`~chemcoord.Cartesian` or a
                :class:`~chemcoord.Cartesian`.
            bond_dict : A bond dictionary computed via
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            list: List containing :class:`~chemcoord.Cartesian`.
        """
        if isinstance(fragments, Sequence):
            index_of_all_fragments = OrderedSet(fragments[0].index)
            for fragment in fragments[1:]:
                index_of_all_fragments |= OrderedSet(fragment.index)
        else:
            index_of_all_fragments = fragments.index  # type: ignore[assignment]
        missing_part = self.loc[OrderedSet(self.index) - index_of_all_fragments]
        if bond_dict is None:
            bond_dict = missing_part.get_bonds()
        else:
            bond_dict = missing_part.restrict_bond_dict(bond_dict)
        return sorted(
            missing_part.fragmentate(bond_dict=bond_dict), key=len, reverse=True
        )

    @staticmethod
    @njit
    def _jit_pairwise_distances(
        pos1: Matrix[np.floating], pos2: Matrix[np.floating]
    ) -> Matrix[np.float64]:
        """Optimized function for calculating the distance between each pair
        of points in positions1 and positions2.

        Does use python mode as fallback, if a scalar and not an array is
        given.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        D = np.empty((n1, n2))

        for i in range(n1):
            for j in range(n2):
                D[i, j] = np.sqrt(((pos1[i] - pos2[j]) ** 2).sum())
        return D

    def get_shortest_distance(self, other: Self) -> tuple[AtomIdx, AtomIdx, float]:
        """Calculate the shortest distance between self and other

        Args:
            Cartesian: other

        Returns:
            tuple: Returns a tuple ``i, j, d`` with the following meaning:

            ``i``:
            The index on self that minimises the pairwise distance.

            ``j``:
            The index on other that minimises the pairwise distance.

            ``d``:
            The distance between self and other. (float)
        """
        pos1 = self.loc[:, COORDS].values
        pos2 = other.loc[:, COORDS].values
        D = self._jit_pairwise_distances(pos1, pos2)
        i, j = np.unravel_index(D.argmin(), D.shape)
        return AtomIdx(self.index[i]), AtomIdx(other.index[j]), float(D[i, j])  # type: ignore[call-overload]

    def get_inertia(self) -> dict[str, Any]:
        """Calculate the inertia tensor and transforms along
        rotation axes.

        This function calculates the inertia tensor and returns
        a 4-tuple.

        The unit is ``amu * length-unit-of-xyz-file**2``

        Args:
            None

        Returns:
            dict: The returned dictionary has four possible keys:

            ``transformed_Cartesian``:
            A :class:`~chemcoord.Cartesian`
            that is transformed to the basis spanned by
            the eigenvectors of the inertia tensor. The x-axis
            is the axis with the lowest inertia moment, the
            z-axis the one with the highest. Contains also a
            column for the mass

            ``diag_inertia_tensor``:
            A vector containing the ascendingly sorted inertia moments after
            diagonalization.

            ``inertia_tensor``:
            The inertia tensor in the old basis.

            ``eigenvectors``:
            The eigenvectors of the inertia tensor in the old basis.
            Since the inertia_tensor is hermitian, they are orthogonal and
            are returned as an orthonormal righthanded basis.
            The i-th eigenvector corresponds to the i-th eigenvalue in
            ``diag_inertia_tensor``.
        """

        def calculate_inertia_tensor(molecule: Self) -> tuple[Matrix, Matrix, Matrix]:
            masses = molecule.loc[:, "mass"].values
            pos = molecule.loc[:, COORDS].values
            inertia = np.sum(
                masses[:, None, None]
                * (
                    (pos**2).sum(axis=1)[:, None, None] * np.identity(3)[None, :, :]
                    - pos[:, :, None] * pos[:, None, :]
                ),
                axis=0,
            )
            diag_inertia, eig_v = np.linalg.eig(inertia)
            sorted_index = np.argsort(diag_inertia)
            diag_inertia = diag_inertia[sorted_index]
            eig_v = eig_v[:, sorted_index]
            return inertia, eig_v, diag_inertia

        molecule = self.add_data("mass")
        molecule = molecule - molecule.get_barycenter()
        inertia, eig_v, diag_inertia = calculate_inertia_tensor(molecule)
        eig_v = xyz_functions.orthonormalize_righthanded(eig_v)
        molecule = molecule.basistransform(eig_v)
        return {
            "transformed_Cartesian": molecule,
            "eigenvectors": eig_v,
            "diag_inertia_tensor": diag_inertia,
            "inertia_tensor": inertia,
        }

    def basistransform(
        self,
        new_basis: Matrix,
        old_basis: Matrix | None = None,
        orthonormalize: bool = True,
    ) -> Self:
        """Transform the frame to a new basis.

        This function transforms the cartesian coordinates from an
        old basis to a new one. Please note that old_basis and
        new_basis are supposed to have full Rank and consist of
        three linear independent vectors. If rotate_only is True,
        it is asserted, that both bases are orthonormal and right
        handed. Besides all involved matrices are transposed
        instead of inverted.
        In some applications this may require the function
        :func:`xyz_functions.orthonormalize` as a previous step.

        Args:
            old_basis :
            new_basis :
            orthonormalize :

        Returns:
            Cartesian: The transformed molecule.
        """
        if old_basis is None:
            old_basis = np.identity(3)

        is_rotation_matrix = np.isclose(np.linalg.det(new_basis), 1)
        if not is_rotation_matrix and orthonormalize:
            new_basis = xyz_functions.orthonormalize_righthanded(new_basis)
            is_rotation_matrix = True

        # We know that `new_basis @ Self` is of type Self, mypy does not know that,
        # because of the overlapping resolution with `__matmul__` of numpy.
        # We have to explicitly cast it to Self.
        if is_rotation_matrix:
            return cast(Self, new_basis.T @ old_basis @ self)
        else:
            return cast(Self, np.linalg.inv(new_basis) @ old_basis @ self)

    def _get_positions(
        self, indices: Vector[np.integer] | Sequence[int]
    ) -> Matrix[np.float64]:
        old_index = self.index
        self.index = range(len(self))  # type: ignore[assignment]
        rename = {j: i for i, j in enumerate(old_index)}

        pos = self.loc[:, COORDS].values.astype("f8")
        out = np.empty((len(indices), 3))
        indices = np.array([rename.get(i, i) for i in indices], dtype="i8")

        normal = indices > constants.keys_below_are_abs_refs
        out[normal] = pos[indices[normal]]

        for row, i in zip(np.nonzero(~normal), indices[~normal]):
            out[row] = constants.absolute_refs[i]

        self.index = old_index
        return out

    def get_distance_to(
        self,
        origin: Vector[np.floating] | AtomIdx | Sequence[Real] | None = None,
        other_atoms: Index | SequenceNotStr[int] | Set[int] | None = None,
        sort: bool = False,
    ) -> Self:
        """Return a Cartesian with a column for the distance from origin."""
        origin = self._get_origin(origin)

        if other_atoms is None:
            other_atoms = self.index

        new = self.loc[other_atoms, :].copy()
        norm = np.linalg.norm
        try:
            new["distance"] = norm((new - origin).loc[:, COORDS].values, axis=1)
        except AttributeError:
            # Happens if molecule consists of only one atom
            new["distance"] = norm((new - origin).loc[:, COORDS].values)
        if sort:
            new.sort_values(by="distance", inplace=True)
        return new

    @overload
    def change_numbering(
        self,
        rename_dict: dict[AtomIdx, AtomIdx],
        inplace: Literal[False] = False,
    ) -> Self: ...

    @overload
    def change_numbering(
        self,
        rename_dict: dict[AtomIdx, AtomIdx],
        inplace: Literal[True],
    ) -> None: ...

    def change_numbering(
        self, rename_dict: dict[AtomIdx, AtomIdx], inplace: bool = False
    ) -> Self | None:
        """Return the reindexed version of Cartesian.

        Args:
            rename_dict (dict): A dictionary mapping integers on integers.

        Returns:
            Cartesian: A renamed copy according to the dictionary passed.
        """
        output = self if inplace else self.copy()
        new_index = [rename_dict.get(key, key) for key in self.index]
        output.index = new_index  # type: ignore[assignment]
        if not inplace:
            return output
        else:
            return None

    def partition_chem_env(
        self, n_sphere: int = 4, bond_dict: BondDict | None = None
    ) -> dict[tuple[str, frozenset[tuple[str, int]]], set[AtomIdx]]:
        """This function partitions the molecule into subsets of the
        same chemical environment.

        A chemical environment is specified by the number of
        surrounding atoms of a certain kind around an atom with a
        certain atomic number represented by a tuple of a string
        and a frozenset of tuples.
        The ``n_sphere`` option determines how many branches the
        algorithm follows to determine the chemical environment.

        Example:
        A carbon atom in ethane has bonds with three hydrogen (atomic
        number 1) and one carbon atom (atomic number 6).
        If ``n_sphere=1`` these are the only atoms we are
        interested in and the chemical environment is::

        ('C', frozenset([('H', 3), ('C', 1)]))

        If ``n_sphere=2`` we follow every atom in the chemical
        enviromment of ``n_sphere=1`` to their direct neighbours.
        In the case of ethane this gives::

        ('C', frozenset([('H', 6), ('C', 1)]))

        In the special case of ethane this is the whole molecule;
        in other cases you can apply this operation recursively and
        stop after ``n_sphere`` or after reaching the end of
        branches.


        Args:
            n_sphere (int):
            bond_dict : A bond dictionary computed via
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            dict: The output will look like this::

                { (element_symbol, frozenset([tuples])) : set([indices]) }

                A dictionary mapping from a chemical environment to
                the set of indices of atoms in this environment.
        """
        if bond_dict is None:
            bond_dict = self.get_bonds()

        def get_chem_env(
            self: Self, i: AtomIdx, n_sphere: float
        ) -> tuple[str, frozenset[tuple[str, int]]]:
            env_index = self.get_coordination_sphere(
                i,
                n_sphere=n_sphere,
                only_surface=False,
                give_only_index=True,
                bond_dict=bond_dict,
            )
            env_index.remove(i)
            atoms = self.loc[env_index, "atom"]
            environment = frozenset(Counter(atoms).most_common())
            return (cast(str, self.loc[i, "atom"]), environment)

        chemical_environments = defaultdict(set)
        for i in self.index:
            chemical_environments[get_chem_env(self, i, n_sphere)].add(i)
        return dict(chemical_environments)

    def align(self, other: Self, mass_weight: bool = False) -> tuple[Self, Self]:
        """Align two Cartesians.

        Minimize the RMSD (root mean squared deviation) between
        ``self`` and ``other``.
        Returns a tuple of copies of ``self`` and ``other`` where
        both are centered around their centroid and
        ``other`` is rotated unto ``self``.
        The rotation minimises the distances between the
        atom pairs of same label.
        Uses the Kabsch algorithm implemented within
        :func:`~.xyz_functions.get_kabsch_rotation`

        Args:
            other (Cartesian):
            mass_weight (bool): Do a mass weighting to find the best rotation

        Returns:
            tuple:
        """
        if mass_weight:
            m1 = (self - self.get_barycenter()).sort_index()
            m2 = (other - other.get_barycenter()).sort_index()
        else:
            m1 = (self - self.get_centroid()).sort_index()
            m2 = (other - other.get_centroid()).sort_index()

        m2 = cast(Self, m1.get_align_transf(m2, mass_weight, centered=True) @ m2)
        return m1, m2

    def get_align_transf(
        self, other: Self, mass_weight: bool = False, centered: bool = False
    ) -> Matrix[np.float64]:
        """Return the rotation matrix that aligns other onto self.

        Minimize the RMSD (root mean squared deviation) between
        ``self`` and ``other``.
        The rotation minimises the distances between the
        atom pairs of same label.
        Uses the Kabsch algorithm implemented within
        :func:`~.xyz_functions.get_kabsch_rotation`.
        If ``mass_weight`` is ``True`` the atoms are weighted by their mass.
        The atoms are moved first to the centroid/barycenter
        (depending on ``mass_weight``) if centered is ``False``.

        Args:
            other (Cartesian):
            mass_weight (bool): Do a mass weighting to find the best rotation
            centered (bool): Assume ``self`` and ``other`` to be centered

        Returns:
            tuple:
        """
        if not centered:
            if mass_weight:
                m1 = (self - self.get_barycenter()).sort_index()
                m2 = (other - other.get_barycenter()).sort_index()
            else:
                m1 = (self - self.get_centroid()).sort_index()
                m2 = (other - other.get_centroid()).sort_index()
        else:
            m1 = self
            m2 = other

        pos1 = m1.loc[:, COORDS].values
        pos2 = m2.loc[m1.index, COORDS].values
        mass = m1.add_data("mass").loc[:, "mass"].values if mass_weight else None

        return xyz_functions.get_kabsch_rotation(pos1, pos2, mass)

    def reindex_similar(self, other: Self, n_sphere: int = 4) -> Self:
        """Reindex ``other`` to be similarly indexed as ``self``.

        Returns a reindexed copy of ``other`` that minimizes the
        distance for each atom to itself in the same chemical environemt
        from ``self`` to ``other``.
        Read more about the definition of the chemical environment in
        :func:`Cartesian.partition_chem_env`

        .. note:: It is necessary to align ``self`` and other before
            applying this method.
            This can be done via :meth:`~Cartesian.align`.

        .. note:: It is probably necessary to improve the result using
            :meth:`~Cartesian.change_numbering()`.

        Args:
            other (Cartesian):
            n_sphere (int): Wrapper around the argument for
                :meth:`~Cartesian.partition_chem_env`.

        Returns:
            Cartesian: Reindexed version of other
        """

        def make_subset_similar(
            m1: Self,
            subset1: set[AtomIdx],
            m2: Self,
            subset2: set[AtomIdx],
            index_dct: dict[AtomIdx, AtomIdx],
        ) -> None:
            """Changes index_dct INPLACE"""
            index1 = list(subset1)
            for m1_i in index1:
                dist_m2_to_m1_i = m2.get_distance_to(
                    cast(Vector[np.float64], m1.loc[m1_i, COORDS].values),
                    subset2,
                    sort=True,
                )

                m2_i = dist_m2_to_m1_i.index[0]
                dist_new = dist_m2_to_m1_i.loc[m2_i, "distance"]
                m2_pos_i = dist_m2_to_m1_i.loc[m2_i, COORDS]

                counter = itertools.count()
                found = False
                while not found:
                    if m2_i in index_dct.keys():
                        old_m1_pos = m1.loc[index_dct[m2_i], COORDS]
                        if dist_new < np.linalg.norm(m2_pos_i - old_m1_pos):
                            index1.append(index_dct[m2_i])
                            index_dct[m2_i] = m1_i
                            found = True
                        else:
                            m2_i = dist_m2_to_m1_i.index[next(counter)]
                            dist_new = dist_m2_to_m1_i.loc[m2_i, "distance"]
                            m2_pos_i = dist_m2_to_m1_i.loc[m2_i, COORDS]
                    else:
                        index_dct[m2_i] = m1_i
                        found = True

        molecule1 = self.copy()
        molecule2 = other.copy()

        partition1 = molecule1.partition_chem_env(n_sphere)
        partition2 = molecule2.partition_chem_env(n_sphere)

        index_dct: dict[AtomIdx, AtomIdx] = {}
        for key in partition1:
            message = (
                "You have chemically different molecules, regarding "
                "the topology of their connectivity."
            )
            assert len(partition1[key]) == len(partition2[key]), message
            make_subset_similar(
                molecule1, partition1[key], molecule2, partition2[key], index_dct
            )
        molecule2.index = [index_dct[i] for i in molecule2.index]  # type: ignore[assignment]
        return molecule2.loc[molecule1.index]
