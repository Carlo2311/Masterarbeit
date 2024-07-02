import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import time
import pandas as pd
import numpoly
from scipy import stats
from scipy.stats import gaussian_kde
import random
from sklearn.model_selection import KFold
from scipy.stats import norm
from scipy.integrate import trapz
from bayes_opt import BayesianOptimization
from gaussian_process import Gaussian_Process

class SPCE():

    def __init__(self, n_samples, p, y_values, sigma, x, dist_joint, N_q, dist_Z):
        self.n_samples = n_samples
        self.p = p
        self.y_values = y_values
        self.sigma = sigma
        self.samples_x = x
        self.dist_joint = dist_joint
        self.poly = cp.generate_expansion(self.p, self.dist_joint, cross_truncation=0.75)
        self.quadrature_points, self.quadrature_weights = cp.generate_quadrature(N_q, dist_Z, 'gaussian')
        self.z_j = self.quadrature_points
        self.w_j = self.quadrature_weights

    def likelihood_function(self, c, samples_x, y_values, sigma_noise, initial_likelihood, normalized_likelihood):
        
        poly_matrix = self.poly(samples_x[:, np.newaxis], self.z_j[0,:], self.z_j[1,:])
        pce = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0)

        likelihood_quadrature = (1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j)
        likelihood = np.sum(likelihood_quadrature, axis=1)
        likelihood_sum = - np.sum(np.log(likelihood))

        normalized_likelihood_i = likelihood_sum / initial_likelihood

        normalized_likelihood.append(likelihood_sum)

        return likelihood_sum
    
    def optimize_sigma(self, samples_x, y_values, sigma_initial, c_initial):
        
        def objective(sigma):
            return self.likelihood_function(c_initial, samples_x, y_values, sigma, initial_likelihood, [])
        
        initial_likelihood = self.likelihood_function(c_initial, samples_x, y_values, sigma_initial, 1, [])
        result = minimize_scalar(objective, bounds=(1e-5, 10), method='bounded')
        optimized_sigma = result.x
        print("Optimized sigma:", optimized_sigma)
        
        return optimized_sigma
    
    def plot_sigma(self, samples_x, y_values, sigma_range, c_initial):
        initial_likelihood = self.likelihood_function(c_initial, samples_x, y_values, sigma_range[0], 1, [])
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], 100)
        likelihoods = [self.likelihood_function(c_initial, samples_x, y_values,sigma, initial_likelihood, []) for sigma in sigma_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sigma_values, likelihoods)
        plt.xlabel('sigma')
        plt.ylabel('likelihood')
        plt.grid()
        plt.show()
    

    def compute_optimal_c(self, samples_x, y_values, sigma_noise, c_initial):

        def gradient_function(c, samples_x, y_values, sigma_noise, initial_likelihood, normalized_likelihood):

            poly_matrix = self.poly(samples_x[:, np.newaxis], self.z_j[0], self.z_j[1])
            pce = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0)

            nominator = y_values[:, np.newaxis] - pce
            likelihood_quadrature =  (1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j)

            grad_like =  self.poly(samples_x[:, np.newaxis], self.z_j[0], self.z_j[1]) * nominator / (np.sqrt(2 * np.pi) * sigma_noise ** 3) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j
            grad_like_sum = np.sum(grad_like, axis=2)
            like = np.sum(likelihood_quadrature, axis=1)
            grad_test = np.sum((1 / (like)) * (- grad_like_sum), axis=1)

            return grad_test
        
        normalized_likelihood = []
        initial_likelihood = self.likelihood_function(c_initial, samples_x, y_values, sigma_noise,  1, normalized_likelihood)
        
        start = time.time()
        # result = minimize(self.likelihood_function, c_initial, args=(samples_x, y_values, sigma_noise, initial_likelihood, normalized_likelihood), method='BFGS') #, jac=gradient_function)
        result = minimize(self.likelihood_function, c_initial, args=(samples_x, y_values, sigma_noise, initial_likelihood, normalized_likelihood), method='BFGS', jac=gradient_function)
        print('time = ', time.time() - start)
        optimized_c = result.x
        print(result.message)

        # plt.figure()
        # plt.plot(normalized_likelihood[1:])
        # plt.xlabel('iteration')
        # plt.ylabel('likelihood')
        # plt.yscale('log')
        # plt.show()
        
        return optimized_c


    def generate_dist_spce(self, samples_x, samples_z, samples_eps, c):
        
        poly_matrix = self.poly(samples_x[:, np.newaxis], samples_z[0,:], samples_z[1,:]) 
        dist_spce = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0) + samples_eps

        return dist_spce

    def plot_distribution(self, dist_spce, y, pdf, samples_x, samples_y, mean_test, sigma_test): #, samples_y_test

        samples_x_i = samples_x[:5]
        indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]

        gpr_test = Gaussian_Process(samples_x, samples_y, mean_test, sigma_test)
        mean_prediction_gpr, std_prediction_gpr = gpr_test.run()
        gpr_test.plot_gpr()
            
        for x, sample in enumerate(samples_x_i):
            ''' KDE for SPCE '''
            kde = gaussian_kde(dist_spce[x,:])
            x_values_spce = np.linspace(min(dist_spce[x,:]), max(dist_spce[x,:]), 1000) 
            dist_spce_pdf_values = kde(x_values_spce)

            ''' KDE for GPR '''
            dist_gpr = cp.Normal(mean_prediction_gpr[x], std_prediction_gpr[x])
            samples_gpr = dist_gpr.sample(size=10000)
            kde = gaussian_kde(samples_gpr)
            x_values_gpr = np.linspace(min(samples_gpr), max(samples_gpr), 1000) 
            dist_gpr_pdf_values = kde(x_values_gpr)

            # area_reference = trapz(pdf[indices[x],:], y)
            # area_spce = trapz(dist_spce_pdf_values, x_values_spce)
            # area_gpr = trapz(dist_gpr_pdf_values, x_values_gpr)
            # print('area reference = ', area_reference)
            # print('area spce = ', area_spce)
            # print('area gpr = ', area_gpr)

            bin_edges = np.arange(-4, 8, 0.2)

            plt.figure()            
            plt.plot(y, pdf[indices[x],:], label='reference')
            plt.plot(x_values_spce, dist_spce_pdf_values, label='SPCE')
            plt.plot(x_values_gpr, dist_gpr_pdf_values, label='GPR')
            # df.plot(kind='density', ax=plt.gca())
            # plt.hist(samples_y_test[x,:], bins=bin_edges, density=True, alpha=0.5, label='distribution reference')
            # plt.hist(dist_spce[x, :], bins=bin_edges, density=True, alpha=0.5, label='distribution SPCE')
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
            if "q1" in term_str or "q2" in term_str:
                c[i] = np.random.normal(-10, 10)  
            else:
                c[i] = coef_q0[coef_q0_index]
                coef_q0_index += 1

        return c
    

    def compute_error(self, dist_spce, samples_y):

        u = np.linspace(0, 1, 1000)
        squared_diff = (np.quantile(dist_spce, u, axis=1) - np.quantile(samples_y, u, axis=1)) ** 2
        d_ws_i = np.trapz(squared_diff, u, axis=1)
        d_ws = np.sum(d_ws_i) / d_ws_i.shape[0]
        variance = np.var(samples_y)
        error = d_ws / variance

        return error

    def cross_validation(self, sigma_noise, c_initial):

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
            c_opt = self.compute_optimal_c(train_x, train_y, sigma_noise, c_initial)
            print('sigma = ', sigma_noise)
            val_x = self.samples_x[val_index]
            val_y = self.y_values[val_index]
            normalized_likelihood=[]
            likelihood = self.likelihood_function(c_opt, val_x, val_y, sigma_noise, 1, normalized_likelihood)
            cv_scores.append(likelihood)
            print('cv_score = ', cv_scores[-1])

        total_cv_score = np.sum(cv_scores)
        print('total_cv_score = ', total_cv_score)
        return total_cv_score
    

    def compute_optimal_sigma(self, c_initial):

        def cv_score(sigma):
            return -self.cross_validation(sigma, c_initial)
 
        sigma_bounds = (0.15, 1)
        optimizer = differential_evolution(cv_score, bounds=[sigma_bounds], strategy='best1bin', disp=True, maxiter=1)

        optimal_sigma = optimizer.x[0]
        return optimal_sigma
    

    
