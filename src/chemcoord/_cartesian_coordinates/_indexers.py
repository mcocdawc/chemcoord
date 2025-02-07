import warnings
from collections.abc import Set
from typing import Generic, Protocol, TypeVar, Union, overload

from attrs import define
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import Self, TypeAlias

from chemcoord._utilities._temporary_deprecation_workarounds import is_iterable
from chemcoord.typing import Integral, SequenceNotStr, Vector


class Molecule(Protocol):
    _frame: DataFrame
    metadata: dict
    _metadata: dict

    def _return_appropiate_type(
        self, selected: Union[Series, DataFrame]
    ) -> Union[Self, Series, DataFrame]: ...


T = TypeVar("T", bound=Molecule)


@define
class _generic_Indexer(Generic[T]):
    molecule: T


IntIdx: TypeAlias = Union[Integral, Set[Integral], Vector, SequenceNotStr[Integral]]
StrIdx: TypeAlias = Union[str, Set[str], SequenceNotStr[str], slice]


class _Loc(_generic_Indexer, Generic[T]):
    @overload
    def __getitem__(
        self, key: tuple[Union[Index, IntIdx, slice, Series], str]
    ) -> Series: ...

    @overload
    def __getitem__(
        self,
        key: tuple[
            Union[Index, IntIdx, slice, Series],
            Union[Series, Set[str], SequenceNotStr[str]],
        ],
    ) -> Union[T, DataFrame]: ...

    @overload
    def __getitem__(
        self, key: tuple[Union[IntIdx, slice, Index, Series], slice]
    ) -> T: ...
    @overload
    def __getitem__(self, key: Union[Index, IntIdx, slice, Series]) -> T: ...

    def __getitem__(
        self,
        key: Union[
            IntIdx,
            slice,
            Series,
            Index,
            tuple[Union[Series, IntIdx, slice, Index], Union[Series, StrIdx]],
        ],
    ) -> Union[T, DataFrame, Series]:
        if isinstance(key, tuple):
            selected = self.molecule._frame.loc[
                _set_caster(key[0]), _set_caster(key[1])
            ]
        else:
            selected = self.molecule._frame.loc[_set_caster(key)]

        try:
            return self.molecule._return_appropiate_type(selected)
        except AttributeError:
            return selected

    def __setitem__(
        self,
        key: Union[
            IntIdx,
            slice,
            Series,
            Index,
            tuple[Union[Series, IntIdx, slice, Index], Union[Series, StrIdx]],
        ],
        value,
    ) -> None:
        df = self.molecule._frame
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=FutureWarning)
                if isinstance(key, tuple):
                    df.loc[_set_caster(key[0]), _set_caster(key[1])] = value
                else:
                    df.loc[_set_caster(key)] = value
        except FutureWarning:
            # We have the situation where value is of different type than
            #  the columns we assign to.
            # This happens for example when assigning sympy objects,
            #  i.e. symbolic variables, to a float column.
            # Currently this is not a problem in pandas and only raises a FutureWarning
            #  (as of version 2.2.), but to be futureproof make an explicit cast.
            # The `except FutureWarning:` has likely to become `except TypeError:`
            #  then in the future.
            if isinstance(key, tuple):
                if type(key[1]) is not str and is_iterable(key[1]):
                    self.molecule._frame = df.astype({k: "O" for k in key[1]})
                else:
                    self.molecule._frame = df.astype({key[1]: "O"})
                self.molecule._frame.loc[_set_caster(key[0]), _set_caster(key[1])] = (
                    value
                )
            else:
                raise TypeError("Assignment not supported.")


class _ILoc(_generic_Indexer, Generic[T]):
    @overload
    def __getitem__(
        self, key: tuple[Union[Series, IntIdx, slice], Integral]
    ) -> Series: ...

    @overload
    def __getitem__(
        self,
        key: tuple[Union[IntIdx, slice, Series], Union[IntIdx, Series]],
    ) -> Union[T, DataFrame]: ...

    @overload
    def __getitem__(self, key: tuple[Union[IntIdx, slice, Series], slice]) -> T: ...
    @overload
    def __getitem__(self, key: Union[IntIdx, slice, Series]) -> T: ...

    def __getitem__(
        self,
        key: Union[
            IntIdx,
            slice,
            Series,
            tuple[Union[Series, IntIdx, slice], Union[Series, IntIdx, slice]],
        ],
    ) -> Union[T, DataFrame, Series]:
        if isinstance(key, tuple):
            selected = self.molecule._frame.iloc[
                _set_caster(key[0]), _set_caster(key[1])
            ]
        else:
            selected = self.molecule._frame.iloc[_set_caster(key)]
        try:
            return self.molecule._return_appropiate_type(selected)
        except AttributeError:
            return selected

    def __setitem__(
        self,
        key: Union[
            IntIdx,
            slice,
            Series,
            tuple[Union[Series, IntIdx, slice], Union[Series, IntIdx, slice]],
        ],
        value,
    ) -> None:
        df = self.molecule._frame
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=FutureWarning)
                if isinstance(key, tuple):
                    df.iloc[_set_caster(key[0]), _set_caster(key[1])] = value
                else:
                    df.iloc[_set_caster(key)] = value
        except FutureWarning:
            # We have the situation where value is of different type than
            #  the columns we assign to.
            # This happens for example when assigning sympy objects,
            #  i.e. symbolic variables, to a float column.
            # Currently this is not a problem in pandas and only raises a FutureWarning
            #  (as of version 2.2.), but to be futureproof make an explicit cast.
            # The `except FutureWarning:` has likely to become `except TypeError:`
            #  then in the future.
            if isinstance(key, tuple):
                if type(key[1]) is not str and is_iterable(key[1]):
                    self.molecule._frame = df.astype(
                        {df.columns[k]: "O" for k in key[1]}
                    )
                else:
                    self.molecule._frame = df.astype({df.columns[key[1]]: "O"})
                self.molecule._frame.iloc[_set_caster(key[0]), _set_caster(key[1])] = (
                    value
                )
            else:
                raise TypeError("Assignment not supported.")


def _set_caster(x):
    """Pandas removed the possibility to index via sets, which we rely on

    Cast to a list when needed.
    """
    if isinstance(x, set):
        return list(x)
    else:
        return x
