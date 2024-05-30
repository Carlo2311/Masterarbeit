import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import pandas as pd
import numpoly
from scipy.stats import gaussian_kde
import random
from sklearn.model_selection import cross_val_score

class SPCE():

    def __init__(self, n_samples, p, y_values, sigma, x, dist_joint):
        self.n_samples = n_samples
        self.p = p
        self.y_values = y_values
        self.sigma = sigma
        self.samples_x = x
        self.dist_joint = dist_joint
        self.poly = cp.generate_expansion(self.p, self.dist_joint)

    def compute_optimal_c(self, dist_Z, sigma_noise, N_q, c_initial):

        # c_initial = np.random.uniform(-10, 10, size=self.poly.shape[0])

        def likelihood_function(c, N_q, sigma_noise, initial_likelihood):
            quadrature_points, quadrature_weights = cp.generate_quadrature(N_q, dist_Z, 'gaussian')

            likelihood = 0
            z_j = quadrature_points[0]
            w_j = quadrature_weights
            
            poly_matrix = self.poly(self.samples_x[:, np.newaxis], z_j)
            pce = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0)

            # pce = np.array([c[:, np.newaxis] * self.poly(self.samples_x[i], z_j) for i in range(self.n_samples)])
            # pce_sum = np.sum(pce, axis=1)

            likelihood = np.sum((1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-((self.y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * w_j), axis=1)
            likelihood_sum = np.sum(np.log(likelihood))

            normalized_likelihood = -likelihood_sum / initial_likelihood

            return normalized_likelihood

        initial_likelihood = likelihood_function(c_initial, N_q, sigma_noise, 1)
        start1 = time.time()
        result = minimize(likelihood_function, c_initial, args=(N_q, sigma_noise, initial_likelihood), method='BFGS') #, options={'maxiter': 1}
        end1 = time.time()
        time1 = end1 - start1
        print(time1)
        optimized_c = result.x
        
        return optimized_c


    def generate_dist_spce(self, n_samples, samples_x, samples_z, samples_eps, c):
        
        poly_matrix = self.poly(samples_x[:, np.newaxis], samples_z) 
        dist_spce = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0) + samples_eps

        # dist_spce1 = np.zeros((len(samples_x), n_samples))
        # for x, sample in enumerate(samples_x):
        #     dist_spce1[x,:] = np.array([(np.sum(c[:, np.newaxis].T * self.poly(sample, samples_z[i]), axis=1) + samples_eps[i]) for i in range(n_samples)]).T

        return dist_spce

    def plot_distribution(self, dist_spce, y, pdf, indices, samples_x):
            
        for x, sample in enumerate(samples_x):
            kde = gaussian_kde(dist_spce[x,:])
            x_values_spce = np.linspace(min(dist_spce[x,:]), max(dist_spce[x,:]), 1000)  # Points on the x-axis
            dist_spce_pdf_values = kde(x_values_spce)

            # df=pd.DataFrame(dist_spce[x,:], columns=['SPCE'])
            bin_edges = np.arange(-4, 8, 0.2)

            plt.figure()            
            plt.plot(y, pdf[indices[x],:], label='reference')
            plt.plot(x_values_spce, dist_spce_pdf_values, label='SPCE')
            # df.plot(kind='density', ax=plt.gca())
            plt.hist(dist_spce[x, :], bins=bin_edges, density=True, label='distribution SPCE')
            plt.xlim(-4, 8)
            plt.title(f'x = {sample}')
            plt.legend()            


    def start_c(self):
        coeffs = self.poly.coefficients
        exponents = self.poly.exponents
        terms_q0 = exponents[:, 1] == 0
        coeffs_new = [coeffs[i] for i in range(len(coeffs)) if terms_q0[i]]
        exp_new = exponents[terms_q0]
        q0 = cp.variable(1)
        poly_q0 = sum(c * q0**e[0] for c, e in zip(coeffs_new, exp_new))
        surrogate_q0 = cp.fit_regression(poly_q0, self.samples_x, self.y_values)
        coef_q0 = np.array(surrogate_q0.coefficients)

        c = np.zeros(self.poly.shape[0])
        coef_q0_index = 0
        for i, term in enumerate(self.poly):
            term_str = str(term)
            if "q1" in term_str:
                c[i] = np.random.uniform(-10, 10)  
            else:
                c[i] = coef_q0[coef_q0_index]
                coef_q0_index += 1

        return c
    

    def compute_error(self):
        n_test = 1000
        dist_X_test = cp.Uniform(0, 1)
        samples_x_test = dist_X_test.sample(size=n_test)     


    
    def compute_optimal_sigma(self):

        c_k = self.compute_optimal_c()
    
