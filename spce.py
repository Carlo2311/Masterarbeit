import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import time
import pandas as pd
import numpoly
from scipy import stats
from scipy.stats import gaussian_kde
import random
from sklearn.model_selection import KFold
from scipy.stats import norm
from scipy.integrate import quad
from bayes_opt import BayesianOptimization

class SPCE():

    def __init__(self, n_samples, p, y_values, sigma, x, dist_joint):
        self.n_samples = n_samples
        self.p = p
        self.y_values = y_values
        self.sigma = sigma
        self.samples_x = x
        self.dist_joint = dist_joint
        self.poly = cp.generate_expansion(self.p, self.dist_joint)

    def likelihood_function(self, c, samples_x, y_values, N_q, sigma_noise, dist_Z, initial_likelihood, normalized_likelihood):

        quadrature_points, quadrature_weights = cp.generate_quadrature(N_q, dist_Z, 'gaussian')

        # likelihood = 0
        z_j = quadrature_points[0]
        w_j = quadrature_weights
        
        poly_matrix = self.poly(samples_x[:, np.newaxis], z_j)
        pce = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0)

        # pce = np.array([c[:, np.newaxis] * self.poly(self.samples_x[i], z_j) for i in range(self.n_samples)])
        # pce_sum = np.sum(pce, axis=1)

        likelihood_quadrature = (1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * w_j)
        likelihood = np.sum(likelihood_quadrature, axis=1)
        likelihood_sum = np.sum(np.log(likelihood))

        normalized_likelihood_i = -likelihood_sum / initial_likelihood

        normalized_likelihood.append(normalized_likelihood_i)

        return normalized_likelihood_i

    def compute_optimal_c(self, samples_x, y_values, dist_Z, sigma_noise, N_q, c_initial):

        normalized_likelihood = []
        initial_likelihood = self.likelihood_function(c_initial, samples_x, y_values, N_q, sigma_noise, dist_Z, 1, normalized_likelihood)
        start = time.time()
        result = minimize(self.likelihood_function, c_initial, args=(samples_x, y_values, N_q, sigma_noise, dist_Z, initial_likelihood, normalized_likelihood), method='BFGS') #, options={'maxiter': 1}
        print('time = ', time.time() - start)
        optimized_c = result.x
        print(result.message)

        plt.figure()
        plt.plot(normalized_likelihood[1:])
        plt.xlabel('iteration')
        plt.ylabel('normalized likelihood')
        plt.yscale('log')
        plt.show()
        
        return optimized_c


    def generate_dist_spce(self, samples_x, samples_z, samples_eps, c):
        
        poly_matrix = self.poly(samples_x[:, np.newaxis], samples_z) 
        dist_spce = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0) + samples_eps

        # dist_spce1 = np.zeros((len(samples_x), n_samples))
        # for x, sample in enumerate(samples_x):
        #     dist_spce1[x,:] = np.array([(np.sum(c[:, np.newaxis].T * self.poly(sample, samples_z[i]), axis=1) + samples_eps[i]) for i in range(n_samples)]).T

        return dist_spce

    def plot_distribution(self, dist_spce, y, pdf, samples_x, samples_y_test): #, samples_y_test

        samples_x_i = samples_x[:5]
        indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
            
        for x, sample in enumerate(samples_x_i):
            kde = gaussian_kde(dist_spce[x,:])
            x_values_spce = np.linspace(min(dist_spce[x,:]), max(dist_spce[x,:]), 1000)  # Points on the x-axis
            dist_spce_pdf_values = kde(x_values_spce)

            # df=pd.DataFrame(dist_spce[x,:], columns=['SPCE'])
            bin_edges = np.arange(-4, 8, 0.2)

            plt.figure()            
            plt.plot(y, pdf[indices[x],:], label='reference')
            plt.plot(x_values_spce, dist_spce_pdf_values, label='SPCE')
            # df.plot(kind='density', ax=plt.gca())
            plt.hist(samples_y_test[x,:], bins=bin_edges, density=True, alpha=0.5, label='distribution reference')
            plt.hist(dist_spce[x, :], bins=bin_edges, density=True, alpha=0.5, label='distribution SPCE')
            plt.xlabel('y')
            plt.ylabel('pdf')
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
    

    def compute_error(self, dist_spce, samples_y, samples_y_all):

        u = np.linspace(0, 1, 1000)
        squared_diff = (np.quantile(dist_spce, u, axis=1) - np.quantile(samples_y, u, axis=1)) ** 2
        d_ws_i = np.trapz(squared_diff, u, axis=1)
        d_ws = np.sum(d_ws_i) / d_ws_i.shape[0]
        variance = np.var(samples_y_all)
        error = d_ws / variance

        return error

    def cross_validation(self, sigma_noise, dist_Z, N_q, c_initial):

        if self.n_samples < 200:
            n_cv = 10
        if self.n_samples >= 200 and self.n_samples < 1000:
            n_cv = 5
        if self.n_samples >= 1000:
            n_cv = 3

        kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        cv_scores = []

        for train_index, val_index in kf.split(self.samples_x):
            train_x = self.samples_x[train_index]
            train_y = self.y_values[train_index]
            c_opt = self.compute_optimal_c(train_x, train_y, dist_Z, sigma_noise, N_q, c_initial)
            print('sigma = ', sigma_noise)
            val_x = self.samples_x[val_index]
            val_y = self.y_values[val_index]
            normalized_likelihood=[]
            likelihood = self.likelihood_function(c_opt, val_x, val_y, N_q, sigma_noise, dist_Z, 1, normalized_likelihood)
            cv_scores.append(likelihood)
            # print('cv_score = ', cv_scores[-1])

        total_cv_score = np.sum(cv_scores)
        print('total_cv_score = ', total_cv_score)
        return total_cv_score
    

    def compute_optimal_sigma(self, dist_Z, N_q, c_initial):

        def cv_score(sigma):
            return -self.cross_validation(sigma, dist_Z, N_q, c_initial)
 
        sigma_bounds = (0.15, 1)
        optimizer = differential_evolution(cv_score, bounds=[sigma_bounds], strategy='best1bin', disp=True, maxiter=1)

        optimal_sigma = optimizer.x[0]
        return optimal_sigma

    
