
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pandas as pd
from chemcoord._exceptions import PhysicalMeaningError
import chemcoord._generic_classes._indexers as indexers


class _pandas_wrapper(object):
    """This class provides wrappers for pd.DataFrame methods.
    """
    def __init__(self, frame):
        self.frame = frame.copy()
        self.metadata = {}
        self._metadata = {}

    def __len__(self):
        return self.shape[0]

    @property
    def loc(self):
        """pew pew
        """
        return indexers._Loc(self)

    @property
    def iloc(self):
        """pew pew
        """
        return indexers._ILoc(self)

    @property
    def index(self):
        """Returns the index.

        Assigning a value to it changes the index.
        """
        return self.frame.index

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self.frame[key[0], key[1]]
        else:
            selected = self.frame[key]
        try:
            return self._return_appropiate_type(selected)
        except AttributeError:
            return selected

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.frame[key[0], key[1]] = value
        else:
            self.frame[key] = value

    @index.setter
    def index(self, value):
        self.frame.index = value

    @property
    def columns(self):
        """Returns the columns.

        Assigning a value to it changes the columns.
        """
        return self.frame.columns

    @columns.setter
    def columns(self, value):
        if not self._required_cols <= set(value):
            raise PhysicalMeaningError('There are columns missing for a '
                                       'meaningful description of a molecule')
        self.frame.columns = value

    @property
    def shape(self):
        return self.frame.shape

    def __repr__(self):
        return self.frame.__repr__()

    def _repr_html_(self):
        try:
            return self.frame._repr_html_()
        except AttributeError:
            pass

    def copy(self):
        molecule = self.__class__(self.frame)
        molecule.metadata = self.metadata.copy()
        molecule._metadata = self._metadata.copy()
        return molecule

    def sort_values(self, by, axis=0, ascending=True, inplace=False,
                    kind='quicksort', na_position='last'):
        """Sort by the values along either axis

        Wrapper around the :meth:`pandas.DataFrame.sort_values` method.
        """
        if inplace:
                self.frame.sort_values(
                    by, axis=axis, ascending=ascending,
                    inplace=inplace, kind=kind, na_position=na_position)
        else:
            return self.__class__(self.frame.sort_values(
                    by, axis=axis, ascending=ascending,
                    inplace=inplace, kind=kind, na_position=na_position))

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False,
                   kind='quicksort', na_position='last',
                   sort_remaining=True, by=None):
        """Sort object by labels (along an axis)

        Wrapper around the :meth:`pandas.DataFrame.sort_index` method.
        """
        if inplace:
            self.frame.sort_index(
                axis=axis, level=level, ascending=ascending, inplace=inplace,
                kind=kind, na_position=na_position,
                sort_remaining=sort_remaining, by=by)
        else:
            return self.__class__(self.frame.sort_index(
                    axis=axis, level=level, ascending=ascending,
                    inplace=inplace, kind=kind, na_position=na_position,
                    sort_remaining=sort_remaining, by=by))

    def replace(self, to_replace=None, value=None, inplace=False,
                limit=None, regex=False, method='pad', axis=None):
        """Replace values given in 'to_replace' with 'value'.

        Wrapper around the :meth:`pandas.DataFrame.replace` method.
        """
        if inplace:
            self.frame.replace(to_replace=to_replace, value=value,
                               inplace=inplace, limit=limit,
                               regex=regex, method=method, axis=axis)
        else:
            return self.__class__(self.frame.replace(
                to_replace=to_replace, value=value, inplace=inplace,
                limit=limit, regex=regex, method=method, axis=axis))

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
            raise PhysicalMeaningError('You drop a column that is needed to '
                                       'be a physical meaningful description '
                                       'of a molecule.')

        if inplace:
            self.frame.set_index(keys, drop=drop, append=append,
                                 inplace=inplace,
                                 verify_integrity=verify_integrity)
        else:
            return self.__class__(
                self.frame.set_index(keys, drop=drop, append=append,
                                     inplace=inplace,
                                     verify_integrity=verify_integrity)
                )

    def append(self, other, ignore_index=False, verify_integrity=False):
        """Append rows of `other` to the end of this frame, returning a new object.

        Wrapper around the :meth:`pandas.DataFrame.append` method.
        """
        if not isinstance(other, self.__class__):
            raise ValueError('May only append instances of same type.')
        new_frame = self.frame.append(other.frame,
                                      ignore_index=ignore_index,
                                      verify_integrity=verify_integrity)
        return self.__class__(new_frame)

    def insert(self, loc, column, value, allow_duplicates=False,
               inplace=False):
        """Insert column into molecule at specified location.

        Wrapper around the :meth:`pandas.DataFrame.insert` method.
        """
        if inplace:
            self.frame.insert(loc, column, value,
                              allow_duplicates=allow_duplicates)
        else:
            output = self.copy()
            output.frame.insert(loc, column, value,
                                allow_duplicates=allow_duplicates)
            return output

    def apply(self, *args, **kwargs):
        """Applies function along input axis of DataFrame.

        Wrapper around the :meth:`pandas.DataFrame.apply` method.
        """
        return self.__class__(self.frame.apply(*args, **kwargs))

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None,
                  float_format=None, sparsify=None, index_names=True,
                  justify=None, line_width=None, max_rows=None,
                  max_cols=None, show_dimensions=False):
        """Render a DataFrame to a console-friendly tabular output.

        Wrapper around the :meth:`pandas.DataFrame.to_string` method.
        """
        return self.frame.to_string(buf=buf,
                                    columns=columns,
                                    col_space=col_space,
                                    header=header,
                                    index=index,
                                    na_rep=na_rep,
                                    formatters=formatters,
                                    float_format=float_format,
                                    sparsify=sparsify,
                                    index_names=index_names,
                                    justify=justify,
                                    line_width=line_width,
                                    max_rows=max_rows,
                                    max_cols=max_cols,
                                    show_dimensions=show_dimensions)

    def to_latex(self, buf=None, columns=None, col_space=None, header=True,
                 index=True, na_rep='NaN', formatters=None, float_format=None,
                 sparsify=None, index_names=True, bold_rows=True,
                 column_format=None, longtable=None, escape=None,
                 encoding=None, decimal='.', multicolumn=None,
                 multicolumn_format=None, multirow=None):
        """ Render a DataFrame to a tabular environment table.

        You can splice this into a LaTeX document.
        Requires ``\\usepackage{booktabs}``.
        Wrapper around the :meth:`pandas.DataFrame.to_latex` method.
        """
        return self.frame.to_latex(buf=buf, columns=columns,
                                   col_space=col_space, header=header,
                                   index=index, na_rep=na_rep,
                                   formatters=formatters,
                                   float_format=float_format,
                                   sparsify=sparsify, index_names=index_names,
                                   bold_rows=bold_rows,
                                   column_format=column_format,
                                   longtable=longtable, escape=escape,
                                   encoding=encoding, decimal=decimal,
                                   multicolumn=multicolumn,
                                   multicolumn_format=multicolumn_format,
                                   multirow=multirow)
