class PandasWrapper:
    """Wrappers for pandas.DataFrame methods with custom slicing and metadata.

    Slicing:
        Slicing operations try to call the method `_return_appropiate_type`.
        This allows subclasses to control the type returned when slicing.
        See `_common_class` for an example.

    Metadata:
        There are two dictionaries as attributes, `metadata` and `_metadata`,
        which are passed on when doing slices.
    """

    def __len__(self):
        return self.shape[0]

    @property
    def empty(self):
        return self._frame.empty

    @property
    def index(self):
        """Return the index (wrapper for pandas.DataFrame.index)."""
        return self._frame.index

    @property
    def columns(self):
        """Return the columns (wrapper for pandas.DataFrame.columns)."""
        return self._frame.columns

    @property
    def shape(self):
        """Return the shape (wrapper for pandas.DataFrame.shape)."""
        return self._frame.shape

    @property
    def dtypes(self):
        """Return the dtypes (wrapper for pandas.DataFrame.dtypes)."""
        return self._frame.dtypes

    def sort_values(
        self, by, axis=0, ascending=True, kind="quicksort", na_position="last"
    ):
        """Sort by the values along either axis
        (wrapper for :meth:`pandas.DataFrame.sort_values`)."""
        return self._frame.sort_values(
            by,
            axis=axis,
            ascending=ascending,
            inplace=False,
            kind=kind,
            na_position=na_position,
        )

    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        sort_remaining=True,
    ):
        """Sort object by labels (wrapper for pandas.DataFrame.sort_index)."""
        return self._frame.sort_index(
            axis=axis,
            level=level,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            sort_remaining=sort_remaining,
        )

    def insert(self, loc, column, value, allow_duplicates=False, inplace=False):
        """Insert column into molecule at specified location.

        Wrapper around the :meth:`pandas.DataFrame.insert` method.

        Args:
            loc: Insertion index.
            column: Column label.
            value: Value to insert.
            allow_duplicates: Whether to allow duplicate column labels.
            inplace: If True, modify in place. If False, return a copy.

        Returns:
            If inplace is False, returns a new object with the column inserted.
        """
        out = self if inplace else self.copy()
        out._frame.insert(loc, column, value, allow_duplicates=allow_duplicates)
        if not inplace:
            return out
