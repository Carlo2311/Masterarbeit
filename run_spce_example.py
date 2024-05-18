import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example import AnalyticalExample
# from test import AnalyticalExample
from scipy.optimize import minimize

n_samples = 800
dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples, rule="H") 
# samples_x = np.array([0.2, 0.5, 0.75, 0.9])
y = np.linspace(-4, 8, 1000)

example = AnalyticalExample(n_samples, y)
pdf, mean_1, mean_2, sigma_1, sigma_2 = example.calculate_pdf(samples_x)
samples_y = example.create_data_points(pdf, n_samples)
# example.plot_example(samples_x, samples_y, mean_1, mean_2, pdf)
# example.plot_pdf(pdf, samples_x)


dist_1 = cp.Normal(mean_1, sigma_1)
dist_2 = cp.Normal(mean_2, sigma_2)
dist_joint_12 = cp.J(dist_1, dist_2) # falsch 
pdf_val = dist_joint_12.pdf([y, y])


### SPCE

dist_Z = cp.Normal(0, 1)
dist_joint = cp.J(dist_X, dist_Z)
sigma_noise = 0.75
p = 3
poly = cp.generate_expansion(p, dist_joint)

N_q = 10
c_initial = np.ones(poly.shape[0])

def likelihood_function(c, N_q, sigma_noise):
    quadrature_points, quadrature_weights = cp.generate_quadrature(N_q, dist_Z, 'gaussian')
    likelihood = 0
    for i in range(n_samples):
        for j in range(N_q):
            z_j = quadrature_points[0][j]
            w_j = quadrature_weights[j]
            pce = np.sum(c * poly(samples_x[i], z_j))
            likelihood += np.sum((1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-0.5 * ((samples_y[i] - pce) ** 2) / (sigma_noise ** 2))) * w_j)
    return -np.sum(np.log(likelihood))

result = minimize(likelihood_function, c_initial, args=(N_q, sigma_noise), method='BFGS', options={'maxiter': 1})
optimized_c = result.x

dist_eps = cp.Normal(0, sigma_noise)

n_samples_test = 1000
# samples_x_test = dist_X.sample(n_samples_test, rule='H')
samples_x_test = [0.2, 0.5, 0.75, 0.9]
samples_z_test = dist_Z.sample(n_samples_test, rule='H')
samples_eps_test = dist_eps.sample(n_samples_test, rule='H')

dist_spce = np.zeros((len(samples_x_test), n_samples_test))
for x, sample in enumerate(samples_x_test):
    for i in range(n_samples_test):
        # dist_spce[i] = np.sum(optimized_c * poly(samples_x_test[i], samples_z_test[i])) + samples_eps_test[i]
        dist_spce[x, i] = np.sum(optimized_c * poly(sample, samples_z_test[i])) + samples_eps_test[i] # fixed x

    bin_edges = np.arange(-5, 5, 0.1)
    plt.figure()
    plt.hist(dist_spce[x, :], bins=bin_edges)
    plt.title(f'x = {sample}')
plt.show()
