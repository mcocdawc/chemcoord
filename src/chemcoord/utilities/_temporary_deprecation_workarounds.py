# -*- coding: utf-8 -*-

import warnings

def replace_without_warn(df, *replace_args, **replace_kwargs):
        """Call the `replace` method of a dataframe while filtering `FutureWarning`

        The warning about:
        [...] Downcasting behavior in `replace` is deprecated [...]
        is already fixed in the code.
        We can remove this warning filter in the future.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return df.replace(*replace_args, **replace_kwargs)


def is_iterable(x):
    """Check if `x` is iterable

    The test via exceptions is actually recommended
    https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    """
    try:
        iter(x)
        return True
    except TypeError:
        return False
