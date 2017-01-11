import numpy
from pypac import tools


class FourierExpansion:
    """
    Boolean Function {-1,1}^n -> (real numbers) defined by its Fourier coefficients, given as an array of the form
    [ coefficient, coefficient, coefficient, ... ] where each coefficient has to have members coefficient.val and
    coefficient.s, val being the value and s being a {0,1}-array indicating the character.
    """

    def __init__(self, fourier_coefficients, input_length):
        self.fourier_coefficients = fourier_coefficients
        self.input_length = input_length

    def eval(self, x):
        return sum([tools.chi(coefficient.s, x) * coefficient.val for coefficient in self.fourier_coefficients])


class FourierExpansionSign(FourierExpansion):
    """
    Same as FourierExpansion, but only return the sign of the function value. Use val() to access the real number value.
    """

    def eval(self, x):
        return -1 if super(FourierExpansionSign, self).eval(x) < 0 else 1

    def val(self, x):
        return super(FourierExpansionSign, self).eval(x)


class Ltf:
    """
    A Boolean function {-1,1}^n -> {-1,1} defined by the linear threshold parameters.
    """

    def __init__(self, a):
        self.a = a
        self.input_length = len(a)

    def eval(self, x):
        return -1 if self.val(x) < 0 else 1

    def val(self, x):
        return numpy.dot(self.a, x)


class GaussianLtf(Ltf):
    """
    A linear threshold function with parameters chosen by Gaussian distribution with given parameters.
    """

    def __init__(self, n, mu=0, sigma=1):
        super(GaussianLtf, self).__init__(numpy.random.normal(mu, sigma, n))


class NoisyGaussianLtf(GaussianLtf):
    """
    Same as GaussianLtf, but each evaluation will have simulated noise by adding a Gaussian random variate with given
    mean and standard deviation.
    """

    def __init__(self, n, mu=0, sigma=1, mu_noise=0, sigma_noise=1):
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise
        super(NoisyGaussianLtf, self).__init__(n, mu, sigma)

    def val(self, x):
        return super(NoisyGaussianLtf, self).val(x) + numpy.random.normal(self.mu_noise, self.sigma_noise)
