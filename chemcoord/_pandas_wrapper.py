
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
#try:
#    # import itertools.imap as map
#    import itertools.izip as zip
#except ImportError:
#    pass
import numpy as np
import pandas as pd
# import collections
# import copy
from . import constants
from . import utilities
from . import export
from . import settings
from ._exceptions import PhysicalMeaningError
#from io import open


# TODO replace all *kwargs
class core(object):
    """This class provides wrappers for pd.DataFrame methods.
    """
    # PLEASE NOTE: It is written under the assumption that there exists an
    # attribute self.frame and self.n_atoms. 
    # So you have to provide it in the __init__ of an inheriting class.
    # Look into ./xyz_functions.py for an example.

    def __len__(self):
        return self.n_atoms

    @property
    def index(self):
        """Returns the index.

        Assigning a value to it changes the index.
        """
        return self.frame.index

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
        self.frame.columns = value


    def __repr__(self):
        return self.frame.__repr__()

    def _repr_html_(self):
        try:
            return self.frame._repr_html_()
        except AttributeError:
            pass

    def _is_physical(self, columns):
        try:
            assert type(columns) is not str
            columns = set(columns)
        except (TypeError, AssertionError):
            columns = set([columns])

        is_cartesian = {'atom', 'x', 'y', 'z'} <= columns
        is_zmat = {'atom', 'bond_with', 'bond', 'angle_with', 'angle', 'dihedral_with', 'dihedral'} <= columns
        return (is_cartesian or is_zmat)

    def __getitem__(self, key):
        frame = self.frame.loc[key[0], key[1]]

        try:
            if self._is_physical(frame.columns):
                return self.__class__(frame)
            else:
                return frame
        except AttributeError:
            # A series and not a DataFrame was returne
            return frame


    def __setitem__(self, key, value):
        self.frame.loc[key[0], key[1]] = value

    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
        """Sort by the values along either axis

        The description is taken from the pandas project.
        
        Parameters
        ----------
        by : string name or list of names which refer to the axis items
        axis : index, columns to direct sorting
        ascending : bool or list of bool
             Sort ascending vs. descending. Specify list for multiple sort
             orders.  If this is a list of bools, must match the length of
             the by.
        inplace : bool
             if True, perform operation in-place
        kind : {`quicksort`, `mergesort`, `heapsort`}
             Choice of sorting algorithm. See also ndarray.np.sort for more
             information.  `mergesort` is the only stable algorithm. For
             DataFrames, this option is only applied when sorting on a single
             column or label.
        na_position : {'first', 'last'}
             `first` puts NaNs at the beginning, `last` puts NaNs at the end
        
        Returns
        -------
        sorted_obj : Cartesian
        """
        if inplace:
                self.frame.sort_values(by, axis=axis, ascending=ascending,
                    inplace=inplace, kind=kind, na_position=na_position)
        else:
            return self.__class__(
                self.frame.sort_values(by, axis=axis, ascending=ascending,
                    inplace=inplace, kind=kind, na_position=na_position))


    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None):
        """Sort object by labels (along an axis)

        The description is taken from the pandas project.

        Parameters
        ----------
        axis : index, columns to direct sorting
        level : int or level name or list of ints or list of level names
            if not None, sort on values in specified index level(s)
        ascending : boolean, default True
            Sort ascending vs. descending
        inplace : bool
            if True, perform operation in-place
        kind : {`quicksort`, `mergesort`, `heapsort`}
             Choice of sorting algorithm. See also ndarray.np.sort for more
             information.  `mergesort` is the only stable algorithm. For
             DataFrames, this option is only applied when sorting on a single
             column or label.
        na_position : {'first', 'last'}
             `first` puts NaNs at the beginning, `last` puts NaNs at the end
        sort_remaining : bool
            if true and sorting by level and index is multilevel, sort by other
            levels too (in order) after sorting by specified level
        
        Returns
        -------
        sorted_obj : Cartesian
        """
        if inplace:
            self.frame.sort_index(
                axis=axis, level=level, ascending=ascending, inplace=inplace,
                kind=kind, na_position=na_position, sort_remaining=sort_remaining, by=by)
        else:
            return self.__class__(
                self.frame.sort_index(
                    axis=axis, level=level, ascending=ascending, inplace=inplace,
                    kind=kind, na_position=na_position, sort_remaining=sort_remaining, by=by))


    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad', axis=None):
        """Replace values given in 'to_replace' with 'value'.

        The description is taken from the pandas project.
    
        Parameters
        ----------
        to_replace : str, regex, list, dict, Series, numeric, or None
        
            * str or regex:
        
                - str: string exactly matching `to_replace` will be replaced
                  with `value`
                - regex: regexs matching `to_replace` will be replaced with
                  `value`
        
            * list of str, regex, or numeric:
        
                - First, if `to_replace` and `value` are both lists, they
                  **must** be the same length.
                - Second, if ``regex=True`` then all of the strings in **both**
                  lists will be interpreted as regexs otherwise they will match
                  directly. This doesn't matter much for `value` since there
                  are only a few possible substitution regexes you can use.
                - str and regex rules apply as above.
        
            * dict:
        
                - Nested dictionaries, e.g., {'a': {'b': nan}}, are read as
                  follows: look in column 'a' for the value 'b' and replace it
                  with nan. You can nest regular expressions as well. Note that
                  column names (the top-level dictionary keys in a nested
                  dictionary) **cannot** be regular expressions.
                - Keys map to column names and values map to substitution
                  values. You can treat this as a special case of passing two
                  lists except that you are specifying the column to search in.
        
            * None:
        
                - This means that the ``regex`` argument must be a string,
                  compiled regular expression, or list, dict, ndarray or Series
                  of such elements. If `value` is also ``None`` then this
                  **must** be a nested dictionary or ``Series``.
        
            See the examples section for examples of each of these.
        value : scalar, dict, list, str, regex, default None
            Value to use to fill holes (e.g. 0), alternately a dict of values
            specifying which value to use for each column (columns not in the
            dict will not be filled). Regular expressions, strings and lists or
            dicts of such objects are also allowed.
        inplace : boolean, default False
            If True, in place. Note: this will modify any
            other views on this object (e.g. a column form a DataFrame).
            Returns the caller if this is True.
        limit : int, default None
            Maximum size gap to forward or backward fill
        regex : bool or same types as `to_replace`, default False
            Whether to interpret `to_replace` and/or `value` as regular
            expressions. If this is ``True`` then `to_replace` *must* be a
            string. Otherwise, `to_replace` must be ``None`` because this
            parameter will be interpreted as a regular expression or a list,
            dict, or array of regular expressions.
        method : string, optional, {'pad', 'ffill', 'bfill'}
            The method to use when for replacement, when ``to_replace`` is a
            ``list``.
        
        
        Returns
        -------
        filled : Cartesian
        
        Raises
        ------
        AssertionError
            * If `regex` is not a ``bool`` and `to_replace` is not ``None``.
        TypeError
            * If `to_replace` is a ``dict`` and `value` is not a ``list``,
              ``dict``, ``ndarray``, or ``Series``
            * If `to_replace` is ``None`` and `regex` is not compilable into a
              regular expression or is a list, dict, ndarray, or Series.
        ValueError
            * If `to_replace` and `value` are ``list`` s or ``ndarray`` s, but
              they are not the same length.
        
        Notes
        -----
        * Regex substitution is performed under the hood with ``re.sub``. The
          rules for substitution for ``re.sub`` are the same.
        * Regular expressions will only substitute on strings, meaning you
          cannot provide, for example, a regular expression matching floating
          point numbers and expect the columns in your frame that have a
          numeric dtype to be matched. However, if those floating point numbers
          *are* strings, then you can do this.
        * This method has *a lot* of options. You are encouraged to experiment
          and play with this method to gain intuition about how it works.
        """
        if inplace:
            self.frame.replace(to_replace=to_replace, value=value, inplace=inplace, limit=limit, regex=regex, method=method, axis=axis)
        else:
            return self.__class__(self.frame.replace(to_replace=to_replace, value=value, inplace=inplace, limit=limit, regex=regex, method=method, axis=axis))


    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        """Set the DataFrame index (row labels) using one or more existing
        columns. By default yields a new object.
        
        The description is taken from the pandas project.

        Parameters
        ----------
        keys : column label or list of column labels / arrays
        drop : boolean, default True
            Delete columns to be used as the new index
        append : boolean, default False
            Whether to append columns to existing index
        inplace : boolean, default False
            Modify the DataFrame in place (do not create a new object)
        verify_integrity : boolean, default False
            Check the new index for duplicates. Otherwise defer the check until
            necessary. Setting to False will improve the performance of this
            method
        
        Examples
        --------
        >>> indexed_df = df.set_index(['A', 'B'])
        >>> indexed_df2 = df.set_index(['A', [0, 1, 2, 0, 1, 2]])
        >>> indexed_df3 = df.set_index([[0, 1, 2, 0, 1, 2]])
        
        Returns
        -------
        Cartesian : Cartesian
        """

        if drop == True:
            try:
                assert type(keys) is not str
                dropped_columns = set(keys)
            except (TypeError, AssertionError):
                dropped_columns = set([keys])

            if not self._is_physical(set(self.columns) - set(dropped_columns)):
                raise PhysicalMeaningError('You drop a column that is needed to be a physical meaningful description of a molecule.')

        if inplace:
            self.frame.set_index(keys, drop=drop, append=append, inplace=inplace, verify_integrity=verify_integrity)
        else:
            return self.__class__(self.frame.set_index(keys, drop=drop, append=append, inplace=inplace, verify_integrity=verify_integrity))


    def append(self, other, ignore_index=False, verify_integrity=False):
        """Append rows of `other` to the end of this frame, returning a new object. 
        
        Columns not in this frame are added as new columns.
        The description is taken from the pandas project.

        Parameters
        ----------
        other : DataFrame or Series/dict-like object, or list of these
            The data to append.
        ignore_index : boolean, default False
            If True, do not use the index labels.
        verify_integrity : boolean, default False
            If True, raise ValueError on creating index with duplicates.

        Returns
        -------
        appended : Cartesian

        Notes
        -----
        If a list of dict/series is passed and the keys are all contained in
        the DataFrame's index, the order of the columns in the resulting
        DataFrame will be unchanged.

        See also
        --------
        pandas.concat : General function to concatenate DataFrame, Series
            or Panel objects

        Examples
        --------

        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
        >>> df
           A  B
        0  1  2
        1  3  4
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
        >>> df.append(df2)
           A  B
        0  1  2
        1  3  4
        0  5  6
        1  7  8

        With `ignore_index` set to True:

        >>> df.append(df2, ignore_index=True)
           A  B
        0  1  2
        1  3  4
        2  5  6
        3  7  8
        """
        new_frame = self.frame.append(other.frame, ignore_index=ignore_index,
                verify_integrity=verify_integrity)
        return self.__class__(new_frame)


    def insert(self, loc, column, value, allow_duplicates=False, inplace=False):
        """Insert column into DataFrame at specified location.

        If `allow_duplicates` is False, raises Exception if column
        is already contained in the DataFrame.

        Parameters
        ----------
        loc : int
            Must have 0 <= loc <= len(columns)
        column : object
        value : int, Series, or array-like
        inplace : bool
        """
        if inplace:
            self.frame.insert(loc, column, value, allow_duplicates=allow_duplicates)
        else:
            output = self.copy()
            output.frame.insert(loc, column, value, allow_duplicates=allow_duplicates)
            return output
