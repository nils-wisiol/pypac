import itertools
import numpy
import math


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


def approx_fourier_coefficient(f, s, sample_size):
    """
    approximate the Fourier coefficient of `f` on `s` by evaluating the function on `sample_size`
    random inputs.
    """
    return numpy.mean([f.eval(x) * chi(s, x) for x in random_inputs(
                   f.input_length,
                   sample_size)])


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


def ncr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


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


class GoldreichLevin:
    """
    Probabilistic algorithm that with probability 1 - `delta` returns a list of sets for the `instance` Boolean function
    using query access.
    If the magnitude of a coefficient is greater or equal than `tau` its set is guaranteed to be in the output list.
    On the other hand all sets in the output list are guaranteed to have Fourier coefficient magnitude greater or equal
    than 1/2 `tau`.
    """

    def __init__(self, instance, tau, delta):
        self.instance = instance
        self.tau = tau
        epsilon = tau ** 2 / 4
        self.delta = tau ** 2 / (8 * self.instance.input_length * (1 - delta))
        self.sample_size = int(math.ceil(12 * math.log(2.0 / self.delta) / (epsilon ** 2)))

    def find_heavy_monomials(self):
        return self.recursive_find((0, numpy.zeros(self.instance.input_length)))

    def recursive_find(self, bucket):
        k = bucket[0]
        s = bucket[1]
        if k == self.instance.input_length:
            return [s]

        extended_s = numpy.copy(s)
        extended_s[k] = 1
        next_buckets = [(k + 1, s), (k + 1, extended_s)]

        return_sets = []
        for new_bucket in next_buckets:
            weight = self.sample_weight(new_bucket)
            if weight > self.tau ** 2 / 2:
                return_sets += self.recursive_find(new_bucket)

        return return_sets

    def sample_weight(self, bucket):
        k = bucket[0]
        s = bucket[1]
        j = numpy.array([1 if i < k else 0 for i in range(self.instance.input_length)])
        estimate = 0.0
        for i in range(self.sample_size):
            z = random_input(self.instance.input_length - k)
            x1 = numpy.append(random_input(k), z)
            x2 = numpy.append(random_input(k), z)
            estimate += self.instance.eval(x1) * chi(numpy.multiply(s, j), x1) * \
                        self.instance.eval(x2) * chi(numpy.multiply(s, j), x2)
        estimate /= self.sample_size
        return estimate
