# -*- coding: utf-8 -*-
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

        Wrapper around the :meth:`pandas.DataFrame.index` property.
        """
        return self._frame.index

    @property
    def columns(self):
        """Returns the columns.

        Wrapper around the :meth:`pandas.DataFrame.columns` property.
        """
        return self._frame.columns

    @property
    def shape(self):
        """Returns the shape.

        Wrapper around the :meth:`pandas.DataFrame.shape` property.
        """
        return self._frame.shape

    @property
    def dtypes(self):
        """Returns the dtypes.

        Wrapper around the :meth:`pandas.DataFrame.dtypes` property.
        """
        return self._frame.dtypes

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
                   sort_remaining=True):
        """Sort object by labels (along an axis)

        Wrapper around the :meth:`pandas.DataFrame.sort_index` method.
        """
        return self._frame.sort_index(axis=axis, level=level,
                                      ascending=ascending, inplace=inplace,
                                      kind=kind, na_position=na_position,
                                      sort_remaining=sort_remaining)

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
