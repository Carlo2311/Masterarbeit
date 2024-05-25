import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp


n_samples = 800
dist_X = cp.Uniform(0, 1)
p = 1
dist_Z = cp.Normal(0, 1)
dist_joint = cp.J(dist_X, dist_Z)
poly = cp.generate_expansion(p, dist_joint)

a = np.array([1,2])
b = np.array([4, 5, 6, 7])
c = np.array([2, 2, 2])





pol = poly(a,b)
test = c[:, np.newaxis] * poly(a,b)


x = np.linspace(0, 6, 1000)
y = np.sin(x)
surrogate = cp.fit_regression(poly, [x, np.zeros(len(x))], y)

print(poly)
coeffs = poly.coefficients
exponents = poly.exponents



mask = (exponents[:, 1] == 0)
coeffs_q0 = [coeffs[i] for i in range(len(coeffs)) if mask[i]]
exponents_q0 = [exponents[i] for i in range(len(exponents)) if mask[i]]
q0, q1 = cp.variable(2)
poly_q0 = sum(c * q0**e[0] for c, e in zip(coeffs_q0, exponents_q0))
surrogate1 = cp.fit_regression(poly_q0, x, y)
print(poly_q0)
