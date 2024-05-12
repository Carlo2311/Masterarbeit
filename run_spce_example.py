import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example import AnalyticalExample
from scipy.optimize import minimize

n_samples = 800
dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples, rule="H") 
samples_x = [0.5]
y = np.linspace(-4, 8, 1000)

example = AnalyticalExample(n_samples, y)
pdf, mean_1, mean_2 = example.calculate_pdf(samples_x)
# samples_y = example.create_data_points(pdf, n_samples)

# example.plot_example(samples_x, samples_y, mean_1, mean_2, pdf)
# example.plot_pdf(pdf, samples_x)





# SPCE

dist_Z = cp.Normal()
dist_joint = dist_Z.pdf(y) * pdf
p = 1
poly = cp.generate_expansion(p, dist_joint)


c = np.array([3, 4, 5])
a = poly * c
test = poly(samples_x[0], 0.4)
test1 = a(0.2, 0.4)

# quadrature_points, quadrature_weights = cp.generate_quadrature(8, dist_Z, 'gaussian')
# z = dist_Z.sample(size=quadrature_points.shape[1], rule="H") 
# pce_xz = np.zeros((len(z), poly.shape[0]))

# for j, z_j in enumerate(z):
#         pce_xz[j, :] = c * poly(samples_x[0], z_j)

#expansion_XZ = cp.monomial(start=0, stop=p, dimensions=samples_x.shape[0] + 1, graded=True)

def likelihood_function(N_q, c, x):
    quadrature_points, quadrature_weights = cp.generate_quadrature(N_q, dist_Z, 'gaussian')
    z = dist_Z.sample(size=quadrature_points.shape[1], rule="H") 
    pce_xz = np.zeros((len(z), poly.shape[0]))
    for j, z_j in enumerate(z):
        pce_xz[j, :] = c * poly(samples_x[0], z_j)
    likelihood = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((pdf - pce_xz) ** 2) / (sigma ** 2)) 


def likelihood_function(c, y, sigma, X, N_q):
    def integrand(pce_xz):
        likelihood_point = np.exp(-0.5 * (y - pce_xz) ** 2 / sigma ** 2) / np.sqrt(2 * np.pi) * sigma
        return likelihood_point

    # quadrature
    quadrature_points, quadrature_weights = cp.generate_quadrature('gaussian', N_q)
    integral = np.sum(integrand(quadrature_points) * quadrature_weights)
    return -integral  # negative because we maximize

# initial coefficients c
initial_guess = np.zeros(pce.shape[1])

# input parameters
y = pdf
sigma = 1 
X_value = 0.2  
N_q = 10
additional_args = (y, sigma, X_value, N_q)

# optimization
result = minimize(likelihood_function, initial_guess, args=additional_args, method='BFGS')
optimized_c = result.x

print(optimized_c)
