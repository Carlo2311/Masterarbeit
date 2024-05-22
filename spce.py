import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SPCE():

    def __init__(self, n_samples, p, y_values, sigma, x, dist_joint):
        self.n_samples = n_samples
        self.p = p
        self.y_values = y_values
        self.sigma = sigma
        self.samples_x = x
        self.dist_joint = dist_joint
        self.poly = cp.generate_expansion(self.p, self.dist_joint)
        self.optimized_c = None

    def compute_liklihood(self, dist_Z, sigma_noise, N_q):
        c_initial = np.ones(self.poly.shape[0])

        def likelihood_function(c, N_q, sigma_noise):
            quadrature_points, quadrature_weights = cp.generate_quadrature(N_q, dist_Z, 'gaussian')
            likelihood_sum = 0
            for i in range(self.n_samples):
                likelihood = 0
                for j in range(N_q):
                    z_j = quadrature_points[0][j]
                    w_j = quadrature_weights[j]
                    pce = np.sum(c * self.poly(self.samples_x[i], z_j))
                    likelihood += (1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-0.5 * ((self.y_values[i] - pce) ** 2) / (sigma_noise ** 2))) * w_j
                likelihood_sum += np.log(likelihood)
            return -likelihood_sum

        result = minimize(likelihood_function, c_initial, args=(N_q, sigma_noise), method='BFGS', options={'maxiter': 1})
        self.optimized_c = result.x
        
        return self.optimized_c


    def generate_dist_spce(self, n_samples_test, samples_x_test, samples_z_test, samples_eps_test):
        dist_spce = np.zeros((len(samples_x_test), n_samples_test))
        for x, sample in enumerate(samples_x_test):
            for i in range(n_samples_test):
                # dist_spce[x, i] = np.sum(optimized_c * poly(samples_x_test[i], samples_z_test[i])) + samples_eps_test[i]
                dist_spce[x, i] = np.sum(self.optimized_c * self.poly(sample, samples_z_test[i])) #+ samples_eps_test[i] # fixed x

            bin_edges = np.arange(-4, 8, 0.1)
            plt.figure()
            plt.hist(dist_spce[x, :], bins=bin_edges)
            plt.title(f'x = {sample}')
        # plt.show()

    def start_c(self):
        surrogate = cp.fit_regression(self.poly, [self.samples_x, np.zeros(len(self.samples_x))], self.y_values)

        coeffs = self.poly.coefficients
        exponents = self.poly.exponents

        mask = (exponents[:, 1] == 0)
        coeffs_q0 = [coeffs[i] for i in range(len(coeffs)) if mask[i]]
        exponents_q0 = [exponents[i] for i in range(len(exponents)) if mask[i]]
        q0, q1 = cp.variable(2)
        poly_q0 = sum(c * q0**e[0] for c, e in zip(coeffs_q0, exponents_q0))

        surrogate_q0 = cp.fit_regression(poly_q0, self.samples_x, self.y_values)

        return surrogate_q0