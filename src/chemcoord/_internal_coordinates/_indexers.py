from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Callable, Set
from typing import Generic, TypeAlias, TypeVar, overload

from attrs import define
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.series import Series

from chemcoord._generic_classes.generic_core import GenericCore
from chemcoord._utilities._temporary_deprecation_workarounds import is_iterable
from chemcoord.exceptions import InvalidReference
from chemcoord.typing import Integral, SequenceNotStr, Vector

# Unlike the Cartesian, the Zmatrix does never return a Zmatrix upon indexing.
# This is because removing a row sometimes results in an undefined Z-matrix
# and I want to guarantee that the user does not accidentally
# think they have a valid Z-matrix, while they actually don't.
# Compare this situation with the corresponding use of `GenericCore` for `Cartesian`.
# There a Union[Self, Series, DataFrame] is returned, while here only a
# Union[Series, DataFrame] is returned.

T = TypeVar("T", bound=GenericCore)

IntIdx: TypeAlias = Integral | Set[Integral] | Vector | SequenceNotStr[Integral]
StrIdx: TypeAlias = str | Set[str] | SequenceNotStr[str]
QueryFunction: TypeAlias = Callable[[DataFrame], Series]


@define
class _generic_Indexer(Generic[T]):
    molecule: T

    @classmethod
    @abstractmethod
    def _get_idxer(cls) -> str: ...


class _Loc(_generic_Indexer):
    @classmethod
    def _get_idxer(cls) -> str:
        return "loc"

    @overload
    def __getitem__(
        self,
        key: Integral,
    ) -> Series: ...

    @overload
    def __getitem__(
        self,
        key: (
            Index
            | Set[Integral]
            | Vector
            | SequenceNotStr[Integral]
            | slice
            | Series
            | QueryFunction
        ),
    ) -> DataFrame: ...

    @overload
    def __getitem__(
        self,
        key: tuple[
            (
                Index
                | Set[Integral]
                | Vector
                | SequenceNotStr[Integral]
                | slice
                | Series
                | QueryFunction
            ),
            Index | Set[str] | Vector | SequenceNotStr[str] | slice | Series,
        ],
    ) -> DataFrame: ...

    @overload
    def __getitem__(
        self,
        key: tuple[
            (
                Index
                | Set[Integral]
                | Vector
                | SequenceNotStr[Integral]
                | slice
                | Series
                | QueryFunction
            ),
            str,
        ],
    ) -> Series: ...

    @overload
    def __getitem__(
        self,
        key: tuple[
            Integral,
            Index | Set[str] | Vector | SequenceNotStr[str] | slice | Series,
        ],
    ) -> Series: ...

    @overload
    def __getitem__(
        self,
        key: tuple[Integral, str],
    ) -> float | str: ...

    def __getitem__(
        self,
        key: (
            (
                Integral
                | Index
                | Set[Integral]
                | Vector
                | SequenceNotStr[Integral]
                | slice
                | Series
                | QueryFunction
            )
            | tuple[
                (
                    Integral
                    | Index
                    | Set[Integral]
                    | Vector
                    | SequenceNotStr[Integral]
                    | slice
                    | Series
                    | QueryFunction
                ),
                str | Index | Set[str] | Vector | SequenceNotStr[str] | slice | Series,
            ]
        ),
    ) -> Series | DataFrame | float | str:
        indexer = getattr(self.molecule._frame, self._get_idxer())
        if isinstance(key, tuple):
            selected = indexer[key[0], key[1]]
        else:
            selected = indexer[key]
        return selected


class _ILoc(_generic_Indexer):
    @classmethod
    def _get_idxer(cls) -> str:
        return "iloc"

    @overload
    def __getitem__(
        self,
        key: Integral,
    ) -> Series: ...

    @overload
    def __getitem__(
        self,
        key: (
            Index | Set[Integral] | Vector | SequenceNotStr[Integral] | slice | Series
        ),
    ) -> DataFrame: ...

    @overload
    def __getitem__(
        self,
        key: tuple[
            (
                Index
                | Set[Integral]
                | Vector
                | SequenceNotStr[Integral]
                | slice
                | Series
            ),
            (
                Index
                | Set[Integral]
                | Vector
                | SequenceNotStr[Integral]
                | slice
                | Series
            ),
        ],
    ) -> DataFrame: ...

    @overload
    def __getitem__(
        self,
        key: tuple[
            (
                Index
                | Set[Integral]
                | Vector
                | SequenceNotStr[Integral]
                | slice
                | Series
            ),
            Integral,
        ],
    ) -> Series: ...

    @overload
    def __getitem__(
        self,
        key: tuple[
            Integral,
            (
                Index
                | Set[Integral]
                | Vector
                | SequenceNotStr[Integral]
                | slice
                | Series
            ),
        ],
    ) -> Series: ...

    @overload
    def __getitem__(
        self,
        key: tuple[Integral, Integral],
    ) -> float | str: ...

    def __getitem__(
        self,
        key: (
            (
                Integral
                | Index
                | Set[Integral]
                | Vector
                | SequenceNotStr[Integral]
                | slice
                | Series
            )
            | tuple[
                (
                    Integral
                    | Index
                    | Set[Integral]
                    | Vector
                    | SequenceNotStr[Integral]
                    | slice
                    | Series
                ),
                (
                    Integral
                    | Index
                    | Set[Integral]
                    | Vector
                    | SequenceNotStr[Integral]
                    | slice
                    | Series
                ),
            ]
        ),
    ) -> Series | DataFrame | float | str:
        indexer = getattr(self.molecule._frame, self._get_idxer())
        if isinstance(key, tuple):
            selected = indexer[key[0], key[1]]
        else:
            selected = indexer[key]
        return selected


class _Unsafe_base:
    def __setitem__(self, key, value):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=FutureWarning)
                indexer = getattr(self.molecule._frame, self._get_idxer())
                if isinstance(key, tuple):
                    indexer[key[0], key[1]] = value
                else:
                    indexer[key] = value
        except FutureWarning:
            # We have the situation where value is of different type than
            #  the columns we assign to.
            # This happens for example when assigning sympy objects,
            #  i.e. symbolic variables, to a float column.
            # Currently this is not a problem in pandas and only raises a FutureWarning
            #  (as of version 2.2.), but to be futureproof make an explicit cast.
            # The `except FutureWarning:` has likely to become `except TypeError:` then.
            if isinstance(key, tuple):
                if type(key[1]) is not str and is_iterable(key[1]):
                    self.molecule._frame = self.molecule._frame.astype(
                        {k: "O" for k in key[1]}
                    )
                else:
                    self.molecule._frame = self.molecule._frame.astype({key[1]: "O"})
                indexer = getattr(self.molecule._frame, self._get_idxer())
                indexer[key[0], key[1]] = value
            else:
                raise TypeError("Assignment not supported.")


class _SafeBase:
    def __setitem__(self, key, value):
        if self.molecule.dummy_manipulation_allowed:
            molecule = self.molecule
        else:
            molecule = self.molecule.copy()

        indexer = getattr(molecule, f"unsafe_{self._get_idxer()}")
        indexer[key] = value

        can_convert_at_all = True
        try:
            molecule.get_cartesian()
        # Sympy objects
        # catches AttributeError as well, because this was
        # the raised exception before https://github.com/numpy/numpy/issues/13666
        except (AttributeError, TypeError):
            self.molecule = molecule
            can_convert_at_all = False
        except InvalidReference as exception:
            if molecule.dummy_manipulation_allowed:
                self.molecule._insert_dummy_zmat(exception, inplace=True)
            else:
                exception.zmat_after_assignment = molecule
                raise exception

        if can_convert_at_all:
            self.molecule._frame = (
                self.molecule._clean_different_dihedral_orientation()._frame
            )

            if molecule.dummy_manipulation_allowed:
                self.molecule._remove_dummies(inplace=True)

            if self.molecule.pure_internal_mov:
                ref = self.molecule._metadata["last_valid_cartesian"]
                new = self.molecule.get_cartesian()
                # TODO(@Oskar): Ensure that this works with Dummy atoms as well
                rotated = ref.align(new, mass_weight=True)[1]
                c_table = self.molecule.loc[:, ["b", "a", "d"]]
                self.molecule._frame = rotated.get_zmat(c_table)._frame

            self.molecule._metadata["last_valid_cartesian"] = (
                self.molecule.get_cartesian()
            )
        else:
            self.molecule = molecule


class _Unsafe_Loc(_Loc, _Unsafe_base):
    pass


class _Safe_Loc(_Loc, _SafeBase):
    pass


class _Unsafe_ILoc(_ILoc, _Unsafe_base):
    pass


class _Safe_ILoc(_ILoc, _SafeBase):
    pass
