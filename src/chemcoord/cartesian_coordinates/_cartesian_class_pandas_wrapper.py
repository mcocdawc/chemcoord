# -*- coding: utf-8 -*-
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from chemcoord._exceptions import PhysicalMeaning
import chemcoord.cartesian_coordinates._indexers as indexers
import pandas as pd
import sympy


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
    def __init__(self, frame):
        self._frame = frame.copy()
        self.metadata = {}
        self._metadata = {}

    def __len__(self):
        return self.shape[0]

    @property
    def empty(self):
        return self._frame.empty

    @property
    def loc(self):
        """Label based indexing

        The indexing behaves like
        `Indexing and Selecting data in Pandas <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_
        The only question is about the return type.
        If the information in the columns is enough to draw a molecule,
        an instance of the own class (e.g. :class:`~chemcoord.Cartesian`)
        is returned.
        If the information in the columns is not enough to draw a molecule
        a :class:`~pandas.Series` instance is returned for one dimensional
        slices and a :class:`~pandas.DataFrame` instance in all other cases.

        In the case of a :class:`~chemcoord.Cartesian` class this means:

            ``molecule.loc[:, ['atom', 'x', 'y', 'z']]`` returns a
            :class:`~chemcoord.Cartesian`.

            ``molecule.loc[:, ['atom', 'x']]`` returns a
            :class:`~pandas.DataFrame`.

            ``molecule.loc[:, 'atom']`` returns a
            :class:`~pandas.Series`.
        """
        return indexers._Loc(self)

    @property
    def iloc(self):
        """Row based indexing

        The indexing behaves like
        `Indexing and Selecting data in Pandas <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_
        The only question is about the return type.
        If the information in the columns is enough to draw a molecule,
        an instance of the own class (e.g. :class:`~chemcoord.Cartesian`)
        is returned.
        If the information in the columns is not enough to draw a molecule
        a :class:`~pandas.Series` instance is returned for one dimensional
        slices and a :class:`~pandas.DataFrame` instance in all other cases.

        In the case of a :class:`~chemcoord.Cartesian` class this means:

            ``molecule.iloc[:, ['atom', 'x', 'y', 'z']]`` returns a
            :class:`~chemcoord.Cartesian`.

            ``molecule.iloc[:, ['atom', 'x']]`` returns a
            :class:`~pandas.DataFrame`.

            ``molecule.iloc[:, 'atom']`` returns a
            :class:`~pandas.Series`.
        """
        return indexers._ILoc(self)

    def loc_set_copy(self, key, value):
        """Labelbased assignment on a copy.

        Allows the use of list comprehensions::

            [molecule.loc_set_copy([1, iloc], i) for i in range(10)]

        In the case of slicing operations, it is necessary to pass
        manually a slicing object::

            [molecule.loc_set_copy([slice(None), bond], i) for i in range(10)]

        A workaround is the definition of a slicer class::

            class SliceMaker(object):
                def __getitem__(self, key):
                    return key
            slicer = SliceMaker()
            [molecule.loc_set_copy([slicer[:], bond], i) for i in range(10)]
        """
        new = self.copy()
        if pd.api.types.is_list_like(key):
            new.loc[key[0], key[1]] = value
        else:
            new.loc[key] = value
        return new

    def iloc_set_copy(self, key, value):
        """Rowbased assignment on a copy.

        Allows the use of list comprehensions::

            [molecule.iloc_set_copy([1, bond], i) for i in range(10)]

        In the case of slicing operations, it is necessary to pass
        manually a slicing object::

            [molecule.iloc_set_copy([slice(None), bond], i) for i in range(10)]

        A workaround is the definition of a slicer class::

            class SliceMaker(object):
                def __getitem__(self, key):
                    return key
            slicer = SliceMaker()
            [molecule.iloc_set_copy([slicer[:], bond], i) for i in range(10)]
        """
        new = self.copy()
        if pd.api.types.is_list_like(key):
            new.iloc[key[0], key[1]] = value
        else:
            new.iloc[key] = value
        return new

    def __getitem__(self, key):
        if isinstance(key, tuple):
            selected = self._frame[key[0], key[1]]
        else:
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

    def __repr__(self):
        return self._frame.__repr__()

    def _repr_html_(self):
        def formatter(x):
            if (isinstance(x, sympy.Basic)):
                return '${}$'.format(sympy.latex(x))
            else:
                return x
        new = self.applymap(formatter)

        def insert_before_substring(insert_txt, substr, txt):
            "Under the assumption that substr only appears once."
            return (insert_txt + substr).join(txt.split(substr))
        html_txt = new._frame._repr_html_()
        insert_txt = '<caption>{}</caption>\n'.format(self.__class__.__name__)
        return insert_before_substring(insert_txt, '<thead>', html_txt)

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
            new._metadata = self._metadata.copy()
            return new

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False,
                   kind='quicksort', na_position='last',
                   sort_remaining=True, by=None):
        """Sort object by labels (along an axis)

        Wrapper around the :meth:`pandas.DataFrame.sort_index` method.
        """
        if inplace:
            self._frame.sort_index(
                axis=axis, level=level, ascending=ascending, inplace=inplace,
                kind=kind, na_position=na_position,
                sort_remaining=sort_remaining, by=by)
        else:
            new = self.__class__(self._frame.sort_index(
                axis=axis, level=level, ascending=ascending,
                inplace=inplace, kind=kind, na_position=na_position,
                sort_remaining=sort_remaining, by=by))
            new.metadata = self.metadata.copy()
            new._metadata = self._metadata.copy()
            return new

    def replace(self, to_replace=None, value=None, inplace=False,
                limit=None, regex=False, method='pad', axis=None):
        """Replace values given in 'to_replace' with 'value'.

        Wrapper around the :meth:`pandas.DataFrame.replace` method.
        """
        if inplace:
            self._frame.replace(to_replace=to_replace, value=value,
                               inplace=inplace, limit=limit,
                               regex=regex, method=method, axis=axis)
        else:
            new = self.__class__(self._frame.replace(
                to_replace=to_replace, value=value, inplace=inplace,
                limit=limit, regex=regex, method=method, axis=axis))
            new.metadata = self.metadata.copy()
            new._metadata = self._metadata.copy()
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
            new = self.__class__(new)
            new.metadata = self.metadata.copy()
            new._metadata = self._metadata.copy()
            return new

    def append(self, other, ignore_index=False, verify_integrity=False):
        """Append rows of `other` to the end of this frame, returning a new object.

        Wrapper around the :meth:`pandas.DataFrame.append` method.
        """
        # TODO(think about metadata)
        if not isinstance(other, self.__class__):
            raise ValueError('May only append instances of same type.')
        new_frame = self._frame.append(other._frame,
                                       ignore_index=ignore_index,
                                       verify_integrity=verify_integrity)
        return self.__class__(new_frame)

    def insert(self, loc, column, value, allow_duplicates=False,
               inplace=False):
        """Insert column into molecule at specified location.

        Wrapper around the :meth:`pandas.DataFrame.insert` method.
        """
        self._frame.insert(loc, column, value,
                           allow_duplicates=allow_duplicates)

    def apply(self, *args, **kwargs):
        """Applies function along input axis of DataFrame.

        Wrapper around the :meth:`pandas.DataFrame.apply` method.
        """
        new = self.__class__(self._frame.apply(*args, **kwargs))
        new.metadata = self.metadata.copy()
        new._metadata = self._metadata.copy()
        return new

    def applymap(self, *args, **kwargs):
        """Applies function elementwise

        Wrapper around the :meth:`pandas.DataFrame.applymap` method.
        """
        new = self.__class__(self._frame.applymap(*args, **kwargs))
        new.metadata = self.metadata.copy()
        new._metadata = self._metadata.copy()
        return new

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None,
                  float_format=None, sparsify=None, index_names=True,
                  justify=None, line_width=None, max_rows=None,
                  max_cols=None, show_dimensions=False):
        """Render a DataFrame to a console-friendly tabular output.

        Wrapper around the :meth:`pandas.DataFrame.to_string` method.
        """
        return self._frame.to_string(
            buf=buf, columns=columns, col_space=col_space, header=header,
            index=index, na_rep=na_rep, formatters=formatters,
            float_format=float_format, sparsify=sparsify,
            index_names=index_names, justify=justify, line_width=line_width,
            max_rows=max_rows, max_cols=max_cols,
            show_dimensions=show_dimensions)

    def to_latex(self, buf=None, columns=None, col_space=None, header=True,
                 index=True, na_rep='NaN', formatters=None, float_format=None,
                 sparsify=None, index_names=True, bold_rows=True,
                 column_format=None, longtable=None, escape=None,
                 encoding=None, decimal='.', multicolumn=None,
                 multicolumn_format=None, multirow=None):
        """Render a DataFrame to a tabular environment table.

        You can splice this into a LaTeX document.
        Requires ``\\usepackage{booktabs}``.
        Wrapper around the :meth:`pandas.DataFrame.to_latex` method.
        """
        return self._frame.to_latex(
            buf=buf, columns=columns, col_space=col_space, header=header,
            index=index, na_rep=na_rep, formatters=formatters,
            float_format=float_format, sparsify=sparsify,
            index_names=index_names, bold_rows=bold_rows,
            column_format=column_format, longtable=longtable, escape=escape,
            encoding=encoding, decimal=decimal, multicolumn=multicolumn,
            multicolumn_format=multicolumn_format, multirow=multirow)
