import warnings
from collections.abc import Sequence, Set
from typing import Generic, Protocol, Self, TypeAlias, TypeVar, Union

from attrs import define

from chemcoord._utilities._temporary_deprecation_workarounds import is_iterable
from chemcoord.typing import DataFrame, Series, Vector


@define(init=False)
class Molecule(Protocol):
    _frame: DataFrame

    def _return_appropiate_type(
        self, selected: Union[Series, DataFrame]
    ) -> Union[Self, Series, DataFrame]: ...


T = TypeVar("T", bound=Molecule)


@define
class _generic_Indexer(Generic[T]):
    molecule: T


IntIdx: TypeAlias = Union[int, Set[int], Vector, Sequence[int], slice]
StrIdx: TypeAlias = Union[str, Set[str], Sequence[str], slice]


class _Loc(_generic_Indexer, Generic[T]):
    # @overload
    # def __getitem__(self, key: tuple[IntIdx, str]) -> Series: ...

    # @overload
    # def __getitem__(
    #     self, key: tuple[IntIdx, Union[Set[str], Sequence[str], slice]]
    # ) -> Union[T, DataFrame]: ...

    # @overload
    # def __getitem__(self, key: IntIdx) -> T: ...

    def __getitem__(self, key: Union[IntIdx, tuple[IntIdx, StrIdx]]) -> DataFrame:
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

    def __setitem__(self, key, value):
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


class _ILoc(_generic_Indexer):
    def __getitem__(self, key):
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

    def __setitem__(self, key, value):
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
