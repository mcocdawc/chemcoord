# The following code was taken from the MIT licensed pandas project
# and modified. http://pandas.pydata.org/
from collections.abc import Callable
from textwrap import dedent
from typing import TypeVar, Union, overload

import numba as nb

# # Substitution and Appender are derived from matplotlib.docstring (1.1.0)
# # module http://matplotlib.org/users/license.html
#

Function = TypeVar("Function", bound=Callable)


class Substitution:
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, func):
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    def update(self, *args, **kwargs):
        "Assume self.params is a dict and update it with supplied args"
        self.params.update(*args, **kwargs)

    @classmethod
    def from_params(cls, params):
        """
        In the case where the params is a mutable sequence (list or dictionary)
        and it may change before this class is called, one may explicitly use a
        reference to the params rather than using *args or **kwargs which will
        copy the values and not reference them.
        """
        result = cls()
        result.params = params
        return result


class Appender:
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    def __init__(self, addendum, join="", indents=0):
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func: Function) -> Function:
        func.__doc__ = func.__doc__ if func.__doc__ else ""
        self.addendum = self.addendum if self.addendum else ""
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func


def indent(text, indents=1):
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\n"] + ["    "] * indents)
    return jointext.join(text.split("\n"))


@overload
def njit(f: Function, **kwargs) -> Function: ...
@overload
def njit(**kwargs) -> Callable[[Function], Function]: ...


def njit(
    f: Union[Function, None] = None, **kwargs
) -> Union[Function, Callable[[Function], Function]]:
    """Type-safe jit wrapper that caches the compiled function

    With this jit wrapper, you can actually use static typing together with numba.
    The crucial declaration is that the decorated function's interface is preserved,
    i.e. mapping :class:`Function` to :class:`Function`.
    Otherwise the following example would not raise a type error:

    .. code-block:: python

        @numba.njit
        def f(x: int) -> int:
            return x

        f(2.0)   # No type error

    While the same example, using this custom :func:`njit` would raise a type error.

    In addition to type safety, this wrapper also sets :code:`cache=True` by default.
    """
    if f is None:
        return nb.njit(cache=True, **kwargs)
    else:
        return nb.njit(f, cache=True, **kwargs)
