import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
import time
import numpoly
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from gaussian_process import Gaussian_Process

'''
Class to perform the SPCE method
Author: Carlotta Hilscher
Date: October 2024
'''

class SPCE():

    def __init__(self, n_samples, p, y_values, x, dist_joint, N_q, dist_Z, q):
        self.n_samples = n_samples
        self.p = p
        self.y_values = y_values
        self.samples_x = x
        self.dist_joint = dist_joint
        self.poly = cp.generate_expansion(self.p, self.dist_joint, cross_truncation=q)
        self.quadrature_points, self.quadrature_weights = cp.generate_quadrature(N_q, dist_Z, 'gaussian')
        self.z_j = self.quadrature_points[0]
        self.w_j = self.quadrature_weights

    ### function to compute the likelihood function ###
    def likelihood_function(self, c, y_values, sigma_noise, poly, input_x):
        
        terms = c * poly
        poly_opt = cp.sum(terms, axis=0)
        pce = poly_opt(*input_x, self.z_j)
        
        likelihood_quadrature = (1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j)
        likelihood = np.sum(likelihood_quadrature, axis=1)
        likelihood_sum = - np.sum(np.log(likelihood))

        return likelihood_sum
    

    ### function to optimize the coefficients c using MLE ###
    def compute_optimal_c(self, y_values, sigma_noise, c_initial, poly, input_x):
        
        ### function to compute the gradient of the likelihood function ###
        def gradient_function(c, y_values, sigma_noise, poly, input_x):

            terms = c * poly
            poly_opt = cp.sum(terms, axis=0)
            pce = poly_opt(*input_x, self.z_j)

            polynomials_eq = poly
            polynomials = polynomials_eq(*input_x, self.z_j)

            nominator = y_values[:, np.newaxis] - pce
            likelihood_quadrature =  (1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j)

            grad_like =  polynomials * nominator / (np.sqrt(2 * np.pi) * sigma_noise ** 3) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j
            grad_like_sum = np.sum(grad_like, axis=2)
            like = np.sum(likelihood_quadrature, axis=1)
            grad = -np.sum((1 / (like)) * (grad_like_sum), axis=1)

            return grad

        if len(input_x.shape) == 1:
            input_x = input_x[np.newaxis, :]

        input_x = [input_x[i, :, np.newaxis] for i in range(input_x.shape[0])]

        ### check if analytical gradient is correct ###
        # from scipy.optimize import check_grad
        # print(check_grad(self.likelihood_function, gradient_function, c_initial, y_values, sigma_noise, poly, input_x))

        start = time.time() 
        result = minimize(self.likelihood_function, c_initial, args=(y_values, sigma_noise, poly, input_x), method='BFGS', jac=gradient_function) #, options={'disp': True})
        # print('time = ', time.time() - start) # check time for optimization
        optimized_c = result.x
        # print(result.message) # check if optimization was successful
        
        return optimized_c, result.message
    

    ### function to optimize sigma using MLE ###
    def optimize_sigma(self, y_values, c_initial, poly, input_x, sigma_range):
        
        def objective(sigma):
            return self.likelihood_function(c_initial, y_values, sigma, poly, input_x)
        
        if len(input_x.shape) == 1:
            input_x = input_x[np.newaxis, :]
        input_x = [input_x[i, :, np.newaxis] for i in range(input_x.shape[0])]

        result = minimize_scalar(objective, bounds=(sigma_range), method='bounded')
        optimized_sigma = result.x
        print("Optimized sigma:", optimized_sigma)
        
        return optimized_sigma
    
    ### function to plot the likelihood function for sigma ###
    def plot_sigma(self, y_values, sigma_range, c_initial, poly, input_x):
        
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], 100)
        likelihoods = [self.likelihood_function(c_initial, y_values, sigma, poly, input_x) for sigma in sigma_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sigma_values, likelihoods)
        plt.xlabel('sigma')
        plt.ylabel('likelihood')
        plt.grid()
        plt.show()


    ### function to generate the distribution of the SPCE surrogate ###
    def generate_dist_spce(self, samples_z, samples_eps, c, poly, input_x):

        if len(input_x.shape) == 1:
            input_x = input_x[np.newaxis, :]
        input_x = [input_x[i, :, np.newaxis] for i in range(input_x.shape[0])]
        
        terms = c * poly
        poly_opt = cp.sum(terms, axis=0)
        pce = poly_opt(*input_x, samples_z)
        dist_spce = pce + samples_eps

        return dist_spce
    

    ### function to generate the distribution of the GPR surrogate ###
    def generate_dist_gpr(self, samples_x_test, mean_test, sigma_test):

        gpr_test = Gaussian_Process(samples_x_test, mean_test, sigma_test)
        mean_prediction_gpr, std_prediction_gpr = gpr_test.run(self.samples_x, self.y_values)

        if len(self.samples_x.shape) == 1:
            samples_gpr_all = np.random.normal(mean_prediction_gpr[:, np.newaxis], std_prediction_gpr[:, np.newaxis], (samples_x_test.shape[0], 10000))
        else:
            samples_gpr_all = np.random.normal(mean_prediction_gpr[:, np.newaxis], std_prediction_gpr[:, np.newaxis], (samples_x_test.shape[1], 10000))
        
        return mean_prediction_gpr, std_prediction_gpr, samples_gpr_all


    ### function to plot the distribution of the surrogates ###
    def plot_distribution(self, distributions, labels, y, pdf, samples_x):
        samples_x_i = samples_x[:5]
        indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]

        for x, sample in enumerate(samples_x_i):
            plt.figure()
            plt.plot(y, pdf[indices[x], :], label='reference')

            for dist, label in zip(distributions, labels):
                kde = gaussian_kde(dist[x, :]) 
                x_values = np.linspace(min(dist[x, :]), max(dist[x, :]), 1000)  
                dist_pdf_values = kde(x_values)
                plt.plot(x_values, dist_pdf_values, label=label)

            plt.xlabel('y')
            plt.ylabel('pdf')
            plt.xlim(-4, 8)
            plt.title(f'x = {sample}')
            plt.legend()

        plt.show()


    ### function to compute the starting coefficients c ###
    def start_c(self, input_x):

        if len(input_x.shape) == 1:
            input_x = input_x[np.newaxis, :]
        else:
            input_x = input_x

        input_x = [input_x[j, :] for j in range(input_x.shape[0])]

        poly_initial_q0 = self.poly.copy()
        poly_without_q0 = []
        q = str(poly_initial_q0.indeterminants[-1])
        for i, x in enumerate(poly_initial_q0): 
            if q in str(x):
                poly_without_q0.append(np.random.normal(0, 1) * x)
                poly_initial_q0[i] = 0
                

        poly_without_q0 = cp.polynomial(poly_without_q0)
        self.surrogate_q0 = cp.fit_regression(poly_initial_q0, (*input_x,), self.y_values)
        self.coef_q0 = np.array(self.surrogate_q0.coefficients)
        poly_initial = self.surrogate_q0 + numpoly.sum(poly_without_q0)

        return self.surrogate_q0, poly_initial
    

    ### function to compute the LOO error ###
    def loo_error(self, mean_ref, surrogate_q0, input_x):

        if len(input_x.shape) == 1:
            input_x = input_x[np.newaxis, :]
        else:
            input_x = input_x

        input_x = [input_x[j, :] for j in range(input_x.shape[0])]

        mean_poly = surrogate_q0(*input_x)
        error_loo = np.mean((mean_ref - mean_poly) ** 2) + np.std(self.y_values)

        return error_loo
    

    ### function to compute the error using the Wasserstein distance ###
    def compute_error(self, dist_spce, samples_y):

        u = np.linspace(0, 1, 1000)
        squared_diff = (np.quantile(dist_spce, u, axis=1) - np.quantile(samples_y, u, axis=1)) ** 2
        d_ws = np.trapz(squared_diff, u, axis=0)
        variance = np.var(samples_y)
        error_spce = np.mean(d_ws) / variance

        return error_spce 


    ### function of the cross-validation ###	
    def cross_validation(self, sigma_noise, c_initial, poly):

        if len(self.samples_x.shape) == 1:
            samples_x = self.samples_x[np.newaxis, :]
        else:
            samples_x = self.samples_x

        self.n_samples = samples_x.shape[1]
        
        if self.n_samples < 200:
            n_cv = 10
        elif self.n_samples < 1000:
            n_cv = 5
        else:
            n_cv = 3

        kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        cv_scores = []

        for train_index, val_index in kf.split(samples_x[0, :]):  
            train_x = samples_x[:, train_index]
            train_y = self.y_values[train_index]
            input_x_train = [train_x[i, :, np.newaxis] for i in range(train_x.shape[0])]

            c_opt, message = self.compute_optimal_c(train_y, sigma_noise, c_initial, poly, train_x)

            val_x = samples_x[:, val_index]
            val_y = self.y_values[val_index]
            input_x_val = [val_x[i, :, np.newaxis] for i in range(val_x.shape[0])]

            likelihood = - self.likelihood_function(c_opt, val_y, sigma_noise, poly, input_x_val)
            cv_scores.append(likelihood)

        total_cv_score = np.sum(cv_scores)

        return total_cv_score
    

    ### function to optimize sigma using CV ###
    def compute_optimal_sigma(self, c_initial, poly, sigma_range):

        def objective(sigma):
            sigma_noise = sigma[0]  
            return self.cross_validation(sigma_noise, c_initial, poly)

        start = time.time()	
        optimizer = BayesianOptimization(f=lambda sigma: objective([sigma]), pbounds={'sigma': sigma_range}, random_state=42, allow_duplicate_points=True)
        optimizer.maximize(init_points=10, n_iter=20)

        # print('Time: ', time.time() - start)
        optimal_sigma = optimizer.max['params']['sigma']
        print(f'Optimal sigma: {optimal_sigma}')
        return optimal_sigma
    

    ### function to generate the standard PCE surrogate ###
    def standard_pce(self, dist_X, samples_x, y, samples_tot, q):
        
        if samples_tot <=300: 
            x = [samples_x[i, :] for i in range(samples_x.shape[0])]
        else:
            x = [samples_x[i, :300] for i in range(samples_x.shape[0])]

        poly_pce = cp.generate_expansion(self.p, dist_X, cross_truncation=q)
        surrogate = cp.fit_regression(poly_pce, (*x,), y) 
        
        return surrogate

    