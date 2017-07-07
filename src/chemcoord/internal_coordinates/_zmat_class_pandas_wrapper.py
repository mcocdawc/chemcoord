# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


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
    def index(self):
        """Returns the index.

        Assigning a value to it changes the index.
        """
        return self._frame.index

    @property
    def columns(self):
        """Returns the columns.

        Assigning a value to it changes the columns.
        """
        return self._frame.columns

    @property
    def shape(self):
        return self._frame.shape

    @property
    def dtypes(self):
        return self._frame.dtypes

    def __repr__(self):
        return self._frame.__repr__()

    def sort_values(self, by, axis=0, ascending=True,
                    kind='quicksort', na_position='last'):
        """Sort by the values along either axis

        Wrapper around the :meth:`pandas.DataFrame.sort_values` method.
        """
        return self._frame.sort_values(by, axis=axis, ascending=ascending,
                                       inplace=False, kind=kind,
                                       na_position=na_position)

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False,
                   kind='quicksort', na_position='last',
                   sort_remaining=True, by=None):
        """Sort object by labels (along an axis)

        Wrapper around the :meth:`pandas.DataFrame.sort_index` method.
        """
        return self._frame.sort_index(axis=axis, level=level,
                                      ascending=ascending, inplace=inplace,
                                      kind=kind, na_position=na_position,
                                      sort_remaining=sort_remaining, by=by)

    # TODO(think about removal)
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
            new._metadata = self._metadata.copy()
            return new

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
