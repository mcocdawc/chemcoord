from __future__ import division
from __future__ import absolute_import


def pick(my_set):
    """Return one element from a set.

    **Do not** make any assumptions about the element to be returned.
    ``pick`` just returns a random element,
    could be the same, could be different.
    """
    # assert type(my_set) is set, 'Pick can be applied only on sets.'
    x = my_set.pop()
    my_set.add(x)
    return x


# NOTE probably own module necessary
# class SliceMaker(object):
#     def __getitem__(self, key):
#         return key
#
#
# make_slice = SliceMaker()
