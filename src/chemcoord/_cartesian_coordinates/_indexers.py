import warnings
from abc import abstractmethod
from collections.abc import Callable, Set
from typing import Generic, TypeAlias, TypeVar, overload

from attrs import define
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import Self

from chemcoord._generic_classes.generic_core import GenericCore
from chemcoord._utilities._temporary_deprecation_workarounds import is_iterable
from chemcoord.typing import Integral, SequenceNotStr, Vector


# The Cartesian should know if to return a Cartesian, Series or DataFrame
# after indexing. Force this with an abstract method.
class Molecule(GenericCore):
    @abstractmethod
    def _return_appropiate_type(
        self, selected: Series | DataFrame
    ) -> Self | Series | DataFrame: ...


T = TypeVar("T", bound=Molecule)


@define
class _generic_Indexer(Generic[T]):
    molecule: T


IntIdx: TypeAlias = Integral | Set[Integral] | Vector | SequenceNotStr[Integral]
StrIdx: TypeAlias = str | Set[str] | SequenceNotStr[str]
QueryFunction: TypeAlias = Callable[[DataFrame], Series]


class _Loc(_generic_Indexer, Generic[T]):
    @overload
    def __getitem__(
        self,
        key: Integral,
    ) -> T: ...

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
    ) -> T: ...

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
            Index | Set[str] | Vector | SequenceNotStr[str] | Series,
        ],
    ) -> T | DataFrame: ...

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
            slice,
        ],
    ) -> T: ...

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
    ) -> T | Series: ...

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
    ) -> T | DataFrame | Series | float | str:
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
        key: (
            IntIdx
            | slice
            | Series
            | Index
            | tuple[Series | IntIdx | slice | Index, Series | StrIdx | slice]
        ),
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
        self,
        key: Integral,
    ) -> T: ...

    @overload
    def __getitem__(
        self,
        key: (
            Index | Set[Integral] | Vector | SequenceNotStr[Integral] | slice | Series
        ),
    ) -> T: ...

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
            Index | Set[Integral] | Vector | SequenceNotStr[Integral] | Series,
        ],
    ) -> T | DataFrame: ...

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
            slice,
        ],
    ) -> T: ...

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
    ) -> T | Series: ...

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
    ) -> T | DataFrame | Series | float | str:
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
        key: (
            IntIdx
            | slice
            | Series
            | tuple[Series | IntIdx | slice, Series | IntIdx | slice]
        ),
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
