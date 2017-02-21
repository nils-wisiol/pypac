import itertools
import numpy


def dist(a, b):
    """
    compute the distance of two function a, b by evaluating all inputs.
    a, b needs to have eval() method and input_length member.
    :return: probability (randomly uniform x) for a.eval(x) != b.eval(x)
    """
    assert a.input_length == b.input_length
    d = 0
    for x in all_inputs(a.input_length):
        d += 1 if a.eval(x) != b.eval(x) else 0
    return d/(2**a.input_length)


def approx_dist(a, b, num):
    """
    approximate the distance of two function a, b by evaluating a random set of inputs.
    a, b needs to have eval() method and input_length member.
    :return: probability (randomly uniform x) for a.eval(x) != b.eval(x)
    """
    assert a.input_length == b.input_length
    d = 0
    for x in random_inputs(a.input_length, num):
        d += 1 if a.eval(x) != b.eval(x) else 0
    return d/num


def random_input(n):
    """
    returns a random {-1,1}-vector of length `n`.
    """
    return numpy.random.choice((-1, +1), n)


def all_inputs(n):
    """
    returns an iterator for all {-1,1}-vectors of length `n`.
    """
    return itertools.product((-1, +1), repeat=n)


def random_inputs(n, num):
    """
    returns an iterator for a random sample of {-1,1}-vectors of length `n` (with replacement).
    """
    for i in range(num):
        yield random_input(n)


def sample_inputs(n, num):
    """
    returns an iterator for either random samples of {-1,1}-vectors of length `n` if `num` < 2^n,
    and an iterator for all {-1,1}-vectors of length `n` otherwise.
    Note that we return only 2^n vectors even with `num` > 2^n.
    In other words, the output of this function is deterministic if and only if num >= 2^n.
    """
    return random_inputs(n, num) if num < 2**n else all_inputs(n)


def chi(s, x):
    """
    returns chi_s(x) = prod_(i \in s) x_i
    """
    assert len(s) == len(x)
    return numpy.prod([x[i] if s[i] == 1 else 1 for i in range(len(x))])


class ValueTable:
    """
    Represents a value table for `func`. Usage: print(str(ValueTable(my_func)))
    """

    def __init__(self, func):
        self.func = func

    def __str__(self):
        s = ''
        for x in all_inputs(self.func.input_length):
            s += str(x) + ' -> ' + str(self.func.eval(x)) + '\n'
        return s
