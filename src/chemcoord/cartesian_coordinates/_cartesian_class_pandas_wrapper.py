# -*- coding: utf-8 -*-
import copy

import chemcoord.cartesian_coordinates._indexers as indexers
from chemcoord.exceptions import PhysicalMeaning


class PandasWrapper(object):
    """This class provides wrappers for :class:`pandas.DataFrame` methods.

    It has the same behaviour as the :class:`~pandas.DataFrame`
    with two exceptions:

    Slicing
        The slicing operations try to call the method
        :method:`_return_appropiate_type`.
        This means that a class that inherited from :class:`_pandas_wrapper`
        may control the type which is returned when a slicing is done.
        Look into :class:`_common_class` for an example.

    Metadata
        There are two dictionaris as attributes
        called `metadata` and `_metadata`
        which are passed on when doing slices...
    """
    def __len__(self):
        return self.shape[0]

    @property
    def empty(self):
        return self._frame.empty

    @property
    def loc(self):
        """Label based indexing

        The indexing behaves like Indexing and Selecting data in
        `Pandas <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_.
        You can slice with :meth:`~chemcoord.Cartesian.loc`,
        :meth:`~chemcoord.Cartesian.iloc`
        and ``Cartesian[...]``.
        The only question is about the return type.
        If the information in the columns is enough to draw a molecule,
        an instance of the own class (e.g. :class:`~chemcoord.Cartesian`)
        is returned.
        If the information in the columns is not enough to draw a molecule,
        there are two cases to consider:

            * A :class:`~pandas.Series` instance is
              returned for one dimensional slices.
            * A :class:`~pandas.DataFrame` instance is returned
              in all other cases.

        This means that:

            ``molecule.loc[:, ['atom', 'x', 'y', 'z']]`` returns a
            :class:`~chemcoord.Cartesian`.

            ``molecule.loc[:, ['atom', 'x']]`` returns a
            :class:`pandas.DataFrame`.

            ``molecule.loc[:, 'atom']`` returns a
            :class:`pandas.Series`.
        """
        return indexers._Loc(self)

    @property
    def iloc(self):
        """Label based indexing

        The indexing behaves like Indexing and Selecting data in
        `Pandas <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_.
        You can slice with :meth:`~chemcoord.Cartesian.loc`,
        :meth:`~chemcoord.Cartesian.iloc`
        and ``Cartesian[...]``.
        The only question is about the return type.
        If the information in the columns is enough to draw a molecule,
        an instance of the own class (e.g. :class:`~chemcoord.Cartesian`)
        is returned.
        If the information in the columns is not enough to draw a molecule,
        there are two cases to consider:

            * A :class:`~pandas.Series` instance is
              returned for one dimensional slices.
            * A :class:`~pandas.DataFrame` instance is returned
              in all other cases.

        This means that:

            ``molecule.loc[:, ['atom', 'x', 'y', 'z']]`` returns a
            :class:`~chemcoord.Cartesian`.

            ``molecule.loc[:, ['atom', 'x']]`` returns a
            :class:`pandas.DataFrame`.

            ``molecule.loc[:, 'atom']`` returns a
            :class:`pandas.Series`.
        """
        return indexers._ILoc(self)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self._frame[key[0], key[1]]
        else:
            # print(len(self), key)
            selected = self._frame[key]
        try:
            return self._return_appropiate_type(selected)
        except AttributeError:
            return selected

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self._frame[key[0], key[1]] = value
        else:
            self._frame[key] = value

    def __getattr__(self, name: str):
        """
        After regular attribute access, try looking up the name
        This allows simpler access to columns for interactive use.
        """
        # Note: obj.x will always call obj.__getattribute__('x') prior to
        # calling obj.__getattr__('x').

        if name.startswith('__'):
            # See here, why we do this
            # https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
            raise AttributeError()
        if (name in self._frame.columns):
            return self[name]
        return object.__getattribute__(self, name)


    @property
    def index(self):
        """Returns the index.

        Assigning a value to it changes the index.
        """
        return self._frame.index

    @index.setter
    def index(self, value):
        self._frame.index = value

    @property
    def columns(self):
        """Returns the columns.

        Assigning a value to it changes the columns.
        """
        return self._frame.columns

    @columns.setter
    def columns(self, value):
        if not self._required_cols <= set(value):
            raise PhysicalMeaning('There are columns missing for a '
                                  'meaningful description of a molecule')
        self._frame.columns = value

    @property
    def shape(self):
        return self._frame.shape

    @property
    def dtypes(self):
        return self._frame.dtypes

    def sort_values(self, by, axis=0, ascending=True, inplace=False,
                    kind='quicksort', na_position='last'):
        """Sort by the values along either axis

        Wrapper around the :meth:`pandas.DataFrame.sort_values` method.
        """
        if inplace:
            self._frame.sort_values(
                by, axis=axis, ascending=ascending,
                inplace=inplace, kind=kind, na_position=na_position)
        else:
            new = self.__class__(self._frame.sort_values(
                by, axis=axis, ascending=ascending, inplace=inplace,
                kind=kind, na_position=na_position))
            new.metadata = self.metadata.copy()
            new._metadata = copy.deepcopy(self._metadata)
            return new

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False,
                   kind='quicksort', na_position='last',
                   sort_remaining=True):
        """Sort object by labels (along an axis)

        Wrapper around the :meth:`pandas.DataFrame.sort_index` method.
        """
        if inplace:
            self._frame.sort_index(
                axis=axis, level=level, ascending=ascending, inplace=inplace,
                kind=kind, na_position=na_position,
                sort_remaining=sort_remaining)
        else:
            new = self.__class__(self._frame.sort_index(
                axis=axis, level=level, ascending=ascending,
                inplace=inplace, kind=kind, na_position=na_position,
                sort_remaining=sort_remaining))
            new.metadata = self.metadata.copy()
            new._metadata = copy.deepcopy(self._metadata)
            return new

    def replace(self, to_replace=None, value=None, inplace=False,
                limit=None, regex=False, method='pad', axis=None):
        """Replace values given in 'to_replace' with 'value'.

        Wrapper around the :meth:`pandas.DataFrame.replace` method.
        """
        if inplace:
            self._frame.replace(to_replace=to_replace, value=value,
                                inplace=inplace, limit=limit, regex=regex,
                                method=method, axis=axis)
        else:
            new = self.__class__(self._frame.replace(
                to_replace=to_replace, value=value, inplace=inplace,
                limit=limit, regex=regex, method=method, axis=axis))
            new.metadata = self.metadata.copy()
            new._metadata = copy.deepcopy(self._metadata)
            return new

    def set_index(self, keys, drop=True, append=False,
                  inplace=False, verify_integrity=False):
        """Set the DataFrame index (row labels) using one or more existing
        columns.

        Wrapper around the :meth:`pandas.DataFrame.set_index` method.
        """

        if drop is True:
            try:
                assert type(keys) is not str
                dropped_cols = set(keys)
            except (TypeError, AssertionError):
                dropped_cols = set([keys])

        if not self._required_cols <= (set(self.columns) - set(dropped_cols)):
            raise PhysicalMeaning('You drop a column that is needed to '
                                  'be a physical meaningful description '
                                  'of a molecule.')

        if inplace:
            self._frame.set_index(keys, drop=drop, append=append,
                                  inplace=inplace,
                                  verify_integrity=verify_integrity)
        else:
            new = self._frame.set_index(keys, drop=drop, append=append,
                                        inplace=inplace,
                                        verify_integrity=verify_integrity)
            return self.__class__(new, _metadata=self._metadata,
                                  metadata=self.metadata)


    def reset_index(self):
        """Resets the index to 0...n
        """
        return self.__class__(self._frame.reset_index(drop=True))


    def insert(self, loc, column, value, allow_duplicates=False,
               inplace=False):
        """Insert column into molecule at specified location.

        Wrapper around the :meth:`pandas.DataFrame.insert` method.
        """
        out = self if inplace else self.copy()
        out._frame.insert(loc, column, value,
                          allow_duplicates=allow_duplicates)
        if not inplace:
            return out

    def apply(self, *args, **kwargs):
        """Applies function along input axis of DataFrame.

        Wrapper around the :meth:`pandas.DataFrame.apply` method.
        """
        return self.__class__(self._frame.apply(*args, **kwargs),
                              metadata=self.metadata,
                              _metadata=self._metadata)

    def applymap(self, *args, **kwargs):
        """Applies function elementwise

        Wrapper around the :meth:`pandas.DataFrame.applymap` method.
        """
        return self.__class__(self._frame.applymap(*args, **kwargs),
                              metadata=self.metadata,
                              _metadata=self._metadata)
