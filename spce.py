import numpy as np
import chaospy as cp

class SPCE():

    def __init__(self, p, y_values, sigma, x):
        self.p = p
        self.y_values = y_values # the calculated pdf values
        self.sigma = sigma
        self.x = x

    def generate_expansion(self):
        expansion = cp.monomial(start=0, stop=self.p, dimensions=self.x.shape[0] + 1, graded=True)

        return expansion