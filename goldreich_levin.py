from pypac import concepts, learner, tools
import numpy

class Dictator:
    """
    The dictator function returns the i'th input bit as result.
    """
    def __init__(self, n, i):
        self.input_length = n
        self.i = i

    def eval(self, x):
        return x[self.i]

instance = Dictator(8, 3)

gl = tools.GoldreichLevin(instance, 0.8, 0.1)

# Find monomials
monomials = gl.find_heavy_monomials()

# Print monomials with approximated Fourier coefficients
for monomial in monomials:
    print("%s: %f" % (monomial, tools.approx_fourier_coefficient(instance, monomial, 256)))
