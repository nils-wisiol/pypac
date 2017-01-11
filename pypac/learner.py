import numpy
import math
from collections import namedtuple
from itertools import combinations
from pypac import concepts, tools


class LowDegreeAlgorithm:
    """
    Probabilistic algorithm to create a model for given `instance` Boolean function from random examples using the
    low degree algorithm for degree up to `degree` with accuracy 1-`epsilon` and confidence 1-`delta`.
    """

    def __init__(self, instance, degree, epsilon, delta):
        self.instance = instance
        self.degree = degree
        self.epsilon = epsilon
        self.delta = delta
        self.fourier_coefficients = []
        self.coefficient = namedtuple('coefficient', ['val', 's'])

    def learn(self):
        for i in range(self.degree + 1):
            for s in self.low_degree_chi(i):
                self.fourier_coefficients.append(self.approx_fourier_coefficient(s))

        return concepts.FourierExpansionSign(self.fourier_coefficients, self.instance.input_length)

    def exact_fourier_coefficient(self, s):
        """
        exactly determine the Fourier coefficient of `self.instance` on `s` by evaluating the function on all inputs.
        Warning: exponential runtime in `self.instance.input_length`.
        """
        return self.coefficient(
            val=numpy.mean(
                [self.instance.eval(x) * tools.chi(s, x) for x in tools.all_inputs(self.instance.input_length)]
            ),
            s=s,
        )

    def approx_fourier_coefficient(self, s):
        """
        approximate the Fourier coefficient of `self.instance` on `s` by evaluating the function on a number of
        random inputs.
        """
        # TODO check exact example set size, Prop 3.30
        return self.coefficient(
            val=numpy.mean([self.instance.eval(x) * tools.chi(s, x) for x in tools.random_inputs(
                self.instance.input_length,
                math.ceil(numpy.log(1 / self.delta) / (self.epsilon ** 2))
            )]),
            s=s
        )

    def low_degree_chi(self, degree):
        """
        Returns an iterator for the sets s (represented as {0,1}-arrays that represent monomials with degree exactly
        `degree`.
        """
        for indices in combinations(range(self.instance.input_length), degree):
            yield [1 if i in indices else 0 for i in range(self.instance.input_length)]
