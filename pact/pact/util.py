from functools import reduce  # Required in Python 3
import operator
import more_itertools


def partitions_mit(S):
    for P in more_itertools.set_partitions(S):
        yield P


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]
