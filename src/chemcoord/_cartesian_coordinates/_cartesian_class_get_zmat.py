import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from itertools import permutations
from typing import Literal, cast, overload

import numpy as np
import pandas as pd
from numba.core.errors import NumbaPerformanceWarning
from numpy import float64
from pandas.core.frame import DataFrame
from sortedcontainers import SortedSet
from typing_extensions import Self

import chemcoord._cartesian_coordinates._cart_transformation as transformation
import chemcoord._cartesian_coordinates.xyz_functions as xyz_functions
import chemcoord.constants as constants
from chemcoord._cartesian_coordinates._cartesian_class_core import CartesianCore
from chemcoord._cartesian_coordinates._cartesian_class_pandas_wrapper import (
    COORDS,
)
from chemcoord._internal_coordinates.zmat_class_main import Zmat
from chemcoord._utilities._temporary_deprecation_workarounds import replace_without_warn
from chemcoord.exceptions import (
    ERR_CODE_OK,
    ERR_CODE_InvalidReference,
    IllegalArgumentCombination,
    InvalidReference,
    UndefinedCoordinateSystem,
)
from chemcoord.typing import AtomIdx, BondDict, Matrix, Tensor4D, Vector


class CartesianGetZmat(CartesianCore):
    @staticmethod
    def _check_construction_table(construction_table: DataFrame) -> None:
        """Checks if a construction table uses valid references.
        Raises an exception (UndefinedCoordinateSystem) otherwise.
        """
        c_table = construction_table
        for row, i in enumerate(c_table.index):
            give_message = (
                "Not a valid construction table. "
                "The index {i} uses an invalid reference"
            ).format
            if row == 0:
                pass
            elif row == 1:
                if c_table.loc[i, "b"] not in c_table.index[:row]:
                    raise UndefinedCoordinateSystem(give_message(i=i))
            elif row == 2:
                reference = c_table.loc[i, ["b", "a"]]
                if not reference.isin(c_table.index[:row]).all():
                    raise UndefinedCoordinateSystem(give_message(i=i))
            else:
                reference = c_table.loc[i, ["b", "a", "d"]]
                if not reference.isin(c_table.index[:row]).all():
                    raise UndefinedCoordinateSystem(give_message(i=i))

    def _get_frag_constr_table(
        self,
        *,
        sorted_bond_dict: Mapping[AtomIdx, SortedSet],
        start_atom: AtomIdx | None = None,
        predefined_table: DataFrame | None = None,
    ) -> DataFrame:
        """Create a construction table for a Zmatrix.

        A construction table is basically a Zmatrix without the values
        for the bond lenghts, angles and dihedrals.
        It contains the whole information about which reference atoms
        are used by each atom in the Zmatrix.

        This method creates a so called "chemical" construction table,
        which makes use of the connectivity table in this molecule.

        By default the first atom is the one nearest to the centroid.
        (Compare with :meth:`~Cartesian.get_centroid()`)

        Args:
            start_atom: An index for the first atom may be provided.
            predefined_table (pd.DataFrame): An uncomplete construction table
                may be provided. The rest is created automatically.

        Returns:
            pd.DataFrame: Construction table
        """
        if start_atom is not None and predefined_table is not None:
            raise IllegalArgumentCombination(
                "Either start_atom or predefined_table has to be None"
            )
        if predefined_table is not None:
            self._check_construction_table(predefined_table)
            i = predefined_table.index[0]
            order_of_def = list(predefined_table.index)
            user_defined = list(predefined_table.index)
            construction_table = predefined_table.to_dict(orient="index")
        else:
            if start_atom is None:
                molecule = self.get_distance_to(self.get_centroid())
                i = molecule["distance"].idxmin()
            else:
                i = start_atom
            order_of_def = [i]
            user_defined = []
            construction_table = {i: {"b": "origin", "a": "e_z", "d": "e_x"}}

        visited = {i}
        if len(self) > 1:
            parent = {j: i for j in sorted_bond_dict[i]}
            work_bond_dict = OrderedDict(
                [(j, sorted_bond_dict[j] - visited) for j in sorted_bond_dict[i]]
            )
            _modify_priority(work_bond_dict, user_defined)
        else:
            parent, work_bond_dict = OrderedDict(), OrderedDict()

        while work_bond_dict:
            new_work_bond_dict = OrderedDict()
            for i in work_bond_dict:
                if i in visited:
                    continue
                if i not in user_defined:
                    b = parent[i]
                    if b in order_of_def[:3]:
                        if len(order_of_def) == 1:
                            construction_table[i] = {"b": b, "a": "e_z", "d": "e_x"}
                        elif len(order_of_def) == 2:
                            a = (sorted_bond_dict[b] & set(order_of_def))[0]
                            construction_table[i] = {"b": b, "a": a, "d": "e_x"}
                        else:
                            try:
                                a = parent[b]
                            except KeyError:
                                a = (sorted_bond_dict[b] & set(order_of_def))[0]
                            try:
                                d = parent[a]
                                if d in {b, a}:
                                    message = "Don't make self references"
                                    raise UndefinedCoordinateSystem(message)
                            except (KeyError, UndefinedCoordinateSystem):
                                try:
                                    d = (
                                        (sorted_bond_dict[a] & set(order_of_def))
                                        - {b, a}
                                    )[0]
                                except IndexError:
                                    d = (
                                        (sorted_bond_dict[b] & set(order_of_def))
                                        - {b, a}
                                    )[0]
                            construction_table[i] = {"b": b, "a": a, "d": d}
                    else:
                        a, d = (construction_table[b][k] for k in ["b", "a"])
                        construction_table[i] = {"b": b, "a": a, "d": d}
                    order_of_def.append(i)

                visited.add(i)
                for j in work_bond_dict[i]:
                    new_work_bond_dict[j] = sorted_bond_dict[j] - visited
                    parent[j] = i

            work_bond_dict = new_work_bond_dict
            _modify_priority(work_bond_dict, user_defined)
        output = pd.DataFrame.from_dict(construction_table, orient="index")
        output = output.loc[order_of_def, ["b", "a", "d"]]
        return output

    def get_construction_table(
        self,
        fragment_list: (Sequence[Self | tuple[Self, DataFrame]] | None) = None,
        bond_dict: BondDict | None = None,
        perform_checks: bool = True,
    ) -> DataFrame:
        """Create a construction table for a Zmatrix.

        A construction table is basically a Zmatrix without the values
        for the bond lengths, angles and dihedrals.
        It contains the whole information about which reference atoms
        are used by each atom in the Zmatrix.

        The absolute references in cartesian space are one of the following
        magic strings::

            ['origin', 'e_x', 'e_y', 'e_z']

        This method creates a so called "chemical" construction table,
        which makes use of the connectivity table in this molecule.

        Args:
            fragment_list (sequence): There are four possibilities to specify
                the sequence of fragments:

                1. A list of tuples is given. Each tuple contains the fragment
                with its corresponding construction table in the form of::

                    [(frag1, c_table1), (frag2, c_table2)...]

                If the construction table of a fragment is not complete,
                the rest of each fragment's
                construction table is calculated automatically.

                2. It is possible to omit the construction tables for some
                or all fragments as in the following example::

                    [(frag1, c_table1), frag2, (frag3, c_table3)...]

                3. If ``self`` contains more atoms than the union over all
                fragments, the rest of the molecule without the fragments
                is automatically prepended using
                :meth:`~Cartesian.get_without`::

                    self.get_without(fragments) + fragment_list

                4. If fragment_list is ``None`` then fragmentation, etc.
                is done automatically. The fragments are then sorted by
                their number of atoms, in order to use the largest fragment
                as reference for the other ones.

            bond_dict : A bond dictionary computed via
                :meth:`~chemcoord.Cartesian.get_bonds`.
            perform_checks (bool): The checks for invalid references are
                performed using :meth:`~chemcoord.Cartesian.correct_dihedral`
                and :meth:`~chemcoord.Cartesian.correct_absolute_refs`.

        Returns:
            :class:`pandas.DataFrame`: Construction table
        """
        if bond_dict is None:
            bond_dict = self.get_bonds()
        sorted_bond_dict = self._sort_by_valency(bond_dict)

        if fragment_list is not None:
            fragments = fragment_list
        else:
            fragments = sorted(
                self.fragmentate(bond_dict=sorted_bond_dict), key=len, reverse=True
            )

        def prepend_missing_parts_of_molecule(
            fragment_list: Sequence[Self | tuple[Self, DataFrame]],
        ) -> list[Self | tuple[Self, DataFrame]]:
            full_index: set[AtomIdx] = set()
            for fragment in fragment_list:
                if isinstance(fragment, tuple):
                    full_index = full_index | set(fragment[0].index)
                else:
                    full_index = full_index | set(fragment.index)

            if set(self.index) - set(full_index):
                missing_part = self.get_without(
                    self.loc[full_index], bond_dict=sorted_bond_dict
                )
                fragment_list = missing_part + list(fragment_list)
            return list(fragment_list)

        fragments = prepend_missing_parts_of_molecule(fragments)

        if isinstance(fragments[0], tuple):
            fragment, references = fragments[0]
            full_table = fragment._get_frag_constr_table(
                sorted_bond_dict=fragment.restrict_bond_dict(sorted_bond_dict),
                predefined_table=references,
            )
        else:
            fragment = fragments[0]
            full_table = fragment._get_frag_constr_table(
                sorted_bond_dict=fragment.restrict_bond_dict(sorted_bond_dict)
            )

        for specified in fragments[1:]:
            finished_part = self.loc[full_table.index]
            if isinstance(specified, tuple):
                fragment, references = specified  # noqa: PLW2901
                if len(references) < min(3, len(fragment)):
                    raise ValueError(
                        "If you specify references for a "
                        "fragment, it has to consist of at least"
                        "min(3, len(fragment)) rows."
                    )
                constr_table = fragment._get_frag_constr_table(
                    predefined_table=references,
                    sorted_bond_dict=fragment.restrict_bond_dict(sorted_bond_dict),
                )
            else:
                fragment = specified
                i, b = fragment.get_shortest_distance(finished_part)[:2]
                constr_table = fragment._get_frag_constr_table(
                    start_atom=i,
                    sorted_bond_dict=sorted_bond_dict,
                )
                if len(full_table) == 1:
                    a, d = "e_z", "e_x"
                elif len(full_table) == 2:
                    if b == full_table.index[0]:
                        a = full_table.index[1]
                    else:
                        a = full_table.index[0]
                    d = "e_x"
                else:
                    if b in full_table.index[:2]:
                        if b == full_table.index[0]:
                            a = full_table.index[2]
                            d = full_table.index[1]
                        else:
                            a = full_table.loc[b, "b"]  # type: ignore[assignment]
                            d = full_table.index[2]
                    else:
                        a, d = full_table.loc[b, ["b", "a"]]  # type: ignore[assignment,index,list-item]

                if len(constr_table) >= 1:
                    constr_table.iloc[0, :] = b, a, d  # type: ignore[assignment]
                if len(constr_table) >= 2:
                    constr_table.iloc[1, [1, 2]] = b, a  # type: ignore[assignment]
                if len(constr_table) >= 3:
                    constr_table.iloc[2, 2] = b

            full_table = pd.concat([full_table, constr_table])

        c_table = full_table
        if perform_checks:
            c_table = self.correct_dihedral(c_table, sorted_bond_dict)
            c_table = self.correct_absolute_refs(c_table)
        return c_table

    def check_dihedral(self, construction_table: DataFrame) -> list[AtomIdx]:
        """Checks, if the dihedral defining atom is colinear.

        Checks for each index starting from the third row of the
        ``construction_table``, if the reference atoms are colinear.

        Args:
            construction_table (pd.DataFrame):

        Returns:
            list: A list of problematic indices.
        """
        c_table = construction_table
        angles = self.get_angle_degrees(c_table.iloc[3:, :].values)
        problem_index = np.nonzero((175 < angles) | (angles < 5))[0]
        rename = c_table.index[3:]
        return [rename[i] for i in problem_index]

    def correct_dihedral(
        self,
        construction_table: DataFrame,
        bond_dict: BondDict | None = None,
    ) -> DataFrame:
        """Reindexe the dihedral defining atom if linear reference is used.

        Uses :meth:`~Cartesian.check_dihedral` to obtain the problematic
        indices.

        Args:
            construction_table (pd.DataFrame):
            bond_dict : A bond dictionary computed via
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            pd.DataFrame: Appropiately renamed construction table.
        """
        if bond_dict is None:
            bond_dict = self.get_bonds()
        sorted_bond_dict = cast(
            dict[int, SortedSet], self._sort_by_valency(bond_dict=bond_dict)
        )
        problem_index = self.check_dihedral(construction_table)
        c_table = construction_table.copy()
        for i in problem_index:
            loc_i = c_table.index.get_loc(i)
            b, a, problem_d = c_table.loc[i, ["b", "a", "d"]]  # type: ignore[list-item,index]
            try:
                c_table.loc[i, "d"] = (
                    sorted_bond_dict[a] - {b, a, problem_d} - set(c_table.index[loc_i:])  # type: ignore[index,misc]
                )[0]
            except IndexError:
                visited = set(c_table.index[loc_i:]) | {b, a, problem_d}  # type: ignore[misc]
                tmp_bond_dict = OrderedDict(
                    [
                        (j, sorted_bond_dict[j] - visited)
                        for j in sorted_bond_dict[problem_d]  # type: ignore[index]
                    ]
                )
                found = False
                while tmp_bond_dict and not found:
                    new_tmp_bond_dict = OrderedDict()
                    for new_d in tmp_bond_dict:
                        if new_d in visited:
                            continue
                        angle = self.get_angle_degrees([[b, a, new_d]])[0]  # type: ignore[list-item]
                        if 5 < angle < 175:
                            found = True
                            c_table.loc[i, "d"] = new_d
                        else:
                            visited.add(new_d)
                            for j in tmp_bond_dict[new_d]:
                                new_tmp_bond_dict[j] = sorted_bond_dict[j] - visited
                    tmp_bond_dict = new_tmp_bond_dict
                if not found:
                    other_atoms = c_table.index[:loc_i].difference([b, a])  # type: ignore[misc]
                    molecule = self.get_distance_to(
                        origin=i, sort=True, other_atoms=other_atoms
                    )
                    k = 0
                    while not found and k < len(molecule):
                        new_d = molecule.index[k]
                        angle = self.get_angle_degrees([[b, a, new_d]])[0]  # type: ignore[list-item]
                        if 5 < angle < 175:
                            found = True
                            c_table.loc[i, "d"] = new_d
                        k = k + 1
                    if not found:
                        message = (
                            "The atom with index {} has no possibility "
                            "to get nonlinear reference atoms".format
                        )
                        raise UndefinedCoordinateSystem(message(i))
        return c_table

    def _has_valid_abs_ref(self, i: AtomIdx, construction_table: DataFrame) -> bool:
        """Checks, if ``i`` uses valid absolute references.

        Checks for each index from first to third row of the
        ``construction_table``, if the references are colinear.
        This case has to be specially treated, because the references
        are not only atoms (to fix internal degrees of freedom) but also points
        in cartesian space called absolute references.
        (to fix translational and rotational degrees of freedom)

        Args:
            i (label): The label has to be in the first three rows.
            construction_table (pd.DataFrame):

        Returns:
            bool:
        """
        c_table = construction_table
        abs_refs = constants.absolute_refs
        A = np.empty((3, 3))
        row = c_table.index.get_loc(i)
        assert isinstance(row, int)
        if row > 2:
            message = "The index {i} is not from the first three, rows".format
            raise ValueError(message(i=i))
        for k in range(3):
            if k < row:
                A[k] = self.loc[c_table.iloc[row, k], COORDS]  # type: ignore[index]
            else:
                A[k] = abs_refs[c_table.iloc[row, k]]  # type: ignore[index]
        v1, v2 = A[2] - A[1], A[1] - A[0]
        K = np.cross(v1, v2)
        zero = np.full(3, 0.0)
        return not (
            np.allclose(K, zero) or np.allclose(v1, zero) or np.allclose(v2, zero)
        )

    def check_absolute_refs(self, construction_table: DataFrame) -> list[AtomIdx]:
        """Checks first three rows of ``construction_table`` for linear references

        Checks for each index from first to third row of the
        ``construction_table``, if the references are colinear.
        This case has to be specially treated, because the references
        are not only atoms (to fix internal degrees of freedom) but also points
        in cartesian space called absolute references.
        (to fix translational and rotational degrees of freedom)

        Args:
            construction_table (pd.DataFrame):

        Returns:
            list: A list of problematic indices.
        """
        c_table = construction_table
        return [i for i in c_table.index[:3] if not self._has_valid_abs_ref(i, c_table)]

    def correct_absolute_refs(self, construction_table: DataFrame) -> DataFrame:
        """Reindexe construction_table if linear reference in first three rows
        present.

        Uses :meth:`~Cartesian.check_absolute_refs` to obtain the problematic
        indices.

        Args:
            construction_table (pd.DataFrame):

        Returns:
            pd.DataFrame: Appropiately renamed construction table.
        """
        c_table = construction_table.copy()
        abs_refs = constants.absolute_refs
        problem_index = self.check_absolute_refs(c_table)
        for i in problem_index:
            order_of_refs = iter(permutations(abs_refs.keys()))
            finished = False
            while not finished:
                if self._has_valid_abs_ref(i, c_table):
                    finished = True
                else:
                    row = c_table.index.get_loc(i)
                    c_table.iloc[row, row:] = next(order_of_refs)[row:3]  # type: ignore[index,misc,assignment]
        return c_table

    def _calculate_zmat_values(
        self,
        construction_table: DataFrame | pd.Series | Matrix | Vector,
    ) -> Matrix[float64]:
        if isinstance(construction_table, pd.DataFrame):
            c_table = construction_table
        elif isinstance(construction_table, pd.Series):
            c_table = pd.DataFrame(construction_table).T
        else:
            tmp_arr = np.asarray(construction_table)
            if len(tmp_arr.shape) == 1:
                tmp_arr = tmp_arr[None, :]
            c_table = pd.DataFrame(
                data=tmp_arr[:, 1:], index=tmp_arr[:, 0], columns=["b", "a", "d"]
            )

        c_table = replace_without_warn(c_table, constants.int_label).astype("i8")
        c_table.index = c_table.index.astype("i8")

        new_index = c_table.index.append(self.index.difference(c_table.index))
        X = self.loc[new_index, COORDS].values.astype("f8").T
        c_table = c_table.replace(dict(zip(new_index, range(len(self)))))

        err, C = transformation.get_C(X, c_table.values.T)
        if err != ERR_CODE_OK:
            raise ValueError
        C[[1, 2], :] = np.rad2deg(C[[1, 2], :])
        return C.T

    def _build_zmat(self, construction_table: DataFrame) -> Zmat:
        """Create the Zmatrix from a construction table.

        Args:
            Construction table (pd.DataFrame):

        Returns:
            Zmat: A new instance of :class:`Zmat`.
        """
        c_table = construction_table
        dtypes = [
            ("atom", str),
            ("b", str),
            ("bond", float),
            ("a", str),
            ("angle", float),
            ("d", str),
            ("dihedral", float),
        ]

        zmat_frame = pd.DataFrame(
            np.empty(len(c_table), dtype=dtypes), index=c_table.index
        )

        zmat_frame.loc[:, "atom"] = self.loc[c_table.index, "atom"]
        zmat_frame.loc[:, ["b", "a", "d"]] = c_table

        zmat_values = self._calculate_zmat_values(c_table)
        zmat_frame.loc[:, ["bond", "angle", "dihedral"]] = zmat_values

        zmat_frame = zmat_frame.join(
            self._frame.loc[:, list(set(self.columns) - {"atom", "x", "y", "z"})]
        )

        zmatrix = Zmat(
            zmat_frame,
            metadata=self.metadata,
            _metadata={"last_valid_cartesian": self.copy()},
        )
        return zmatrix

    def get_zmat(
        self,
        construction_table: DataFrame | None = None,
        bond_dict: BondDict | None = None,
    ) -> Zmat:
        """Transform to internal coordinates.

        Transforming to internal coordinates involves basically three
        steps:

        1. Define an order of how to build and define for each atom
        the used reference atoms.

        2. Check for problematic local linearity. In this algorithm an
        angle with ``170 < angle < 10`` is assumed to be linear.
        This is not the mathematical definition, but makes it safer
        against "floating point noise"

        3. Calculate the bond lengths, angles and dihedrals using the
        references defined in step 1 and 2.

        In the first two steps a so called ``construction_table`` is created.
        This is basically a Zmatrix without the values for the bonds, angles
        and dihedrals hence containing only the information about the used
        references. ChemCoord uses a :class:`pandas.DataFrame` with the columns
        ``['b', 'a', 'd']``. Look into
        :meth:`~chemcoord.Cartesian.get_construction_table` for more
        information.

        It is important to know, that calculating the construction table
        is a very costly step since the algoritym tries to make some guesses
        based on connectivity to create a "chemical" zmatrix.

        If you create several zmatrices based on the same references
        you can obtain the construction table of a zmatrix with
        ``Zmat_instance.loc[:, ['b', 'a', 'd']]``
        If you then pass the buildlist as argument to ``give_zmat``,
        the algorithm directly starts with step 3 (which is much faster).

        If a ``construction_table`` is passed into :meth:`~Cartesian.get_zmat`
        the check for pathological linearity is not performed!
        So if a ``construction_table`` is either manually created,
        or obtained from :meth:`~Cartesian.get_construction_table`
        under the option ``perform_checks = False``, it is recommended to use
        the following methods:

            * :meth:`~Cartesian.correct_dihedral`
            * :meth:`~Cartesian.correct_absolute_refs`

        If you want to check for problematic indices in order to solve the
        invalid references yourself, use the following methods:

            * :meth:`~Cartesian.check_dihedral`
            * :meth:`~Cartesian.check_absolute_refs`

        Args:
            construction_table (pandas.DataFrame):
            bond_dict : A bond dictionary computed via
                :meth:`~chemcoord.Cartesian.get_bonds`.

        Returns:
            Zmat: A new instance of :class:`~Zmat`.
        """
        sorted_bond_dict = self._sort_by_valency(
            self.get_bonds() if bond_dict is None else bond_dict
        )
        if construction_table is None:
            c_table = self.get_construction_table(bond_dict=sorted_bond_dict)
            c_table = self.correct_dihedral(c_table, bond_dict=sorted_bond_dict)
            c_table = self.correct_absolute_refs(c_table)
        else:
            c_table = construction_table
        return self._build_zmat(c_table)

    @overload
    def get_grad_zmat(
        self, construction_table: DataFrame, as_function: Literal[True] = True
    ) -> Callable[[Self], Zmat]: ...

    @overload
    def get_grad_zmat(
        self, construction_table: DataFrame, as_function: Literal[False]
    ) -> Tensor4D: ...

    def get_grad_zmat(
        self, construction_table: DataFrame, as_function: bool = True
    ) -> Tensor4D | Callable[[Self], Zmat]:
        r"""Return the gradient for the transformation to a Zmatrix.

        If ``as_function`` is True, a function is returned that can be directly
        applied onto instances of :class:`~Cartesian`, which contain the
        applied distortions in cartesian space.
        In this case the user does not have to worry about indexing and
        correct application of the tensor product.
        Basically this is the function
        :func:`xyz_functions.apply_grad_zmat_tensor` with partially replaced
        arguments.

        If ``as_function`` is False, a ``(3, n, n, 3)`` tensor is returned,
        which contains the values of the derivatives.

        Since a ``n * 3`` matrix is deriven after a ``n * 3``
        matrix, it is important to specify the used rules for indexing the
        resulting tensor.

        The rule is very simple: The indices of the numerator are used first
        then the indices of the denominator get swapped and appended:

        .. math::
            \left(
                \frac{\partial \mathbf{Y}}{\partial \mathbf{X}}
            \right)_{i, j, k, l}
            =
            \frac{\partial \mathbf{Y}_{i, j}}{\partial \mathbf{X}_{l, k}}

        Applying this rule to an example function:

        .. math::
            f \colon \mathbb{R}^3 \rightarrow \mathbb{R}

        Gives as derivative the known row-vector gradient:

        .. math::
                (\nabla f)_{1, i}
            =
                \frac{\partial f}{\partial x_i} \qquad i \in \{1, 2, 3\}

        .. note::
            The row wise alignment of the XYZ files makes sense for these
            CSV like files.
            But it is mathematically advantageous and
            sometimes (depending on the memory layout) numerically better
            to use a column wise alignment of the coordinates.
            In this function the resulting tensor assumes a ``3 * n`` array
            for the coordinates.

        If

        .. math::

            \mathbf{X}_{i, j} &\qquad 1 \leq i \leq 3, \quad 1 \leq j \leq n \\
            \mathbf{C}_{i, j} &\qquad 1 \leq i \leq 3, \quad 1 \leq j \leq n

        denote the positions in cartesian and Zmatrix space,

        The complete tensor may be written as:

        .. math::

            \left(
                \frac{\partial \mathbf{C}}{\partial \mathbf{X}}
            \right)_{i, j, k, l}
            =
            \frac{\partial \mathbf{C}_{i, j}}{\partial \mathbf{X}_{l, k}}

        Args:
            construction_table (pandas.DataFrame):
            as_function (bool): Return a tensor or
                :func:`xyz_functions.apply_grad_zmat_tensor`
                with partially replaced arguments.

        Returns:
            (func, np.array): Depending on ``as_function`` return a tensor or
            :func:`~chemcoord.xyz_functions.apply_grad_zmat_tensor`
            with partially replaced arguments.
        """
        if (construction_table.index != self.index).any():
            message = "construction_table and self must use the same index"
            raise ValueError(message)
        c_table = construction_table.loc[:, ["b", "a", "d"]]

        c_table = (
            replace_without_warn(c_table, constants.int_label)
            .astype("i8")
            .replace({k: v for v, k in enumerate(c_table.index)})
            .values.T
        )
        X = self.loc[:, COORDS].values.T
        if X.dtype == np.dtype("i8"):
            X = X.astype("f8")

        with warnings.catch_warnings():
            # There were some performance warnings about non-contiguos arrays.
            # Unfortunately we cannot do anything about it, on a conceptional level.
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            err, row, grad_C = transformation.get_grad_C(X, c_table)

        if err == ERR_CODE_InvalidReference:
            rename = dict(enumerate(self.index))
            i = rename[row]
            b, a, d = construction_table.loc[i, ["b", "a", "d"]]
            raise InvalidReference(i=i, b=b, a=a, d=d)

        if as_function:
            return partial(
                xyz_functions.apply_grad_zmat_tensor, grad_C, construction_table
            )
        else:
            return grad_C

    def to_zmat(self, *args, **kwargs) -> Zmat:  # type: ignore[no-untyped-def]
        """Deprecated, use :meth:`~Cartesian.get_zmat`"""
        message = "Will be removed in the future. Please use give_zmat."
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(message, DeprecationWarning)
        return self.get_zmat(*args, **kwargs)


def _modify_priority(bond_dict: OrderedDict, user_defined: Sequence) -> None:
    for j in reversed(user_defined):
        if j in bond_dict:
            bond_dict.move_to_end(j, last=False)
