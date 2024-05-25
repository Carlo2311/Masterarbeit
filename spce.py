import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import numpoly

class SPCE():

    def __init__(self, n_samples, p, y_values, sigma, x, dist_joint):
        self.n_samples = n_samples
        self.p = p
        self.y_values = y_values
        self.sigma = sigma
        self.samples_x = x
        self.dist_joint = dist_joint
        self.poly = cp.generate_expansion(self.p, self.dist_joint)

    def compute_liklihood(self, dist_Z, sigma_noise, N_q):
        c_initial = np.ones(self.poly.shape[0])

        def likelihood_function(c, N_q, sigma_noise):
            quadrature_points, quadrature_weights = cp.generate_quadrature(N_q, dist_Z, 'gaussian')

            likelihood = 0
            z_j = quadrature_points[0]
            w_j = quadrature_weights

            # test = numpoly.sum([c[:, np.newaxis] * self.poly(self.samples_x[i], z_j) for i in range(self.n_samples)], axis=1)

            pce = np.array([c[:, np.newaxis] * self.poly(self.samples_x[i], z_j) for i in range(self.n_samples)])
            pce_sum = np.sum(pce, axis=1)
            likelihood = np.sum((1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-0.5 * ((self.y_values[:, np.newaxis] - pce_sum) ** 2) / (sigma_noise ** 2))) * w_j, axis=1)
            likelihood_sum = np.sum(np.log(likelihood))

            return -likelihood_sum

        start1 = time.time()
        result = minimize(likelihood_function, c_initial, args=(N_q, sigma_noise), method='BFGS', options={'maxiter': 50}) #, options={'maxiter': 1}
        end1 = time.time()
        time1 = end1 - start1
        print(time1)
        optimized_c = result.x
        
        return optimized_c


    def generate_dist_spce(self, n_samples_test, samples_x_test, samples_z_test, samples_eps_test, optimized_c):
        dist_spce = np.zeros((len(samples_x_test), n_samples_test))
        for x, sample in enumerate(samples_x_test):
            for i in range(n_samples_test):
                # dist_spce[x, i] = np.sum(optimized_c * poly(samples_x_test[i], samples_z_test[i])) + samples_eps_test[i]
                dist_spce[x, i] = np.sum(optimized_c * self.poly(sample, samples_z_test[i])) + samples_eps_test[i] # fixed x

            bin_edges = np.arange(-4, 8, 0.1)
            plt.figure()
            plt.hist(dist_spce[x, :], bins=bin_edges)
            plt.title(f'x = {sample}')
        # plt.show()

    def start_c(self):
        surrogate = cp.fit_regression(self.poly, [self.samples_x, np.zeros(len(self.samples_x))], self.y_values)
        coeffs = self.poly.coefficients
        exponents = self.poly.exponents
        terms_q0 = exponents[:, 1] == 0
        coeffs_new = [coeffs[i] for i in range(len(coeffs)) if terms_q0[i]]
        exp_new = exponents[terms_q0]
        q0 = cp.variable(1)
        poly_q0 = sum(c * q0**e[0] for c, e in zip(coeffs_new, exp_new))
        surrogate_q0 = cp.fit_regression(poly_q0, self.samples_x, self.y_values)
        coef_q0 = surrogate_q0.coefficients

        sorted_coeffs = [0] * len(coeffs)
        sorted_coeffs2 = [0] * len(coeffs)
        for i, exp in enumerate(exp_new):
            idx = np.where((exponents == exp).all(axis=1))[0][0]
            sorted_coeffs[idx] = coef_q0[i]
            sorted_coeffs2[idx] = coef_q0[i] * coeffs[idx]
        
        modified_poly = cp.polynomial(sorted_coeffs2)


        return coef_q0