import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, minimize_scalar, OptimizeResult
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
import tikzplotlib
from mpl_toolkits.mplot3d import Axes3D

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

    def get_params(self):
        return self.poly, self.z_j

    def likelihood_function(self, c, samples_x, y_values, sigma_noise, poly, input_x, initial_likelihood, normalized_likelihood):
        
        # poly_matrix = self.poly(samples_x[:, np.newaxis], self.z_j)
        # pce2 = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0)

        terms = c * poly
        poly_opt = cp.sum(terms, axis=0)
        # input_x = [samples_x[0,:, np.newaxis], samples_x[1,:, np.newaxis], samples_x[2,:, np.newaxis], samples_x[3,:, np.newaxis]]
        pce = poly_opt(*input_x, self.z_j)
        

        # plt.figure()
        # plt.scatter(samples_x[0,:], y_values, label='reference')
        # plt.scatter(samples_x[0,:], pce[:,4], label='poly')
        # plt.legend()
        # plt.show()
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # sc = ax.scatter(samples_x[0,:], samples_x[1,:], y_values, c=y_values, cmap='viridis', marker='o', label='reference')
        # sc2 = ax.scatter(samples_x[0,:], samples_x[1,:], pce[:,4], c=pce[:,4], cmap='plasma', marker='^', label='SPCE')
        # ax.set_xlabel('Wind Speed (m/s)')
        # ax.set_ylabel('Turbulence Intensity')
        # ax.set_zlabel('Blade Load')
        # cb_actual = plt.colorbar(sc, ax=ax, pad=0.1)
        # cb_actual.set_label('Actual Blade Load')
        # cb_predicted = plt.colorbar(sc2, ax=ax, pad=0.2)
        # cb_predicted.set_label('Predicted Blade Load')
        # plt.show()

        likelihood_quadrature = (1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j)
        likelihood = np.sum(likelihood_quadrature, axis=1)
        likelihood_sum = - np.sum(np.log(likelihood))

        # normalized_likelihood_i = likelihood_sum / initial_likelihood
        normalized_likelihood.append(likelihood_sum)

        return likelihood_sum
    

    def compute_optimal_c(self, samples_x, y_values, sigma_noise, c_initial, poly, input_x):

        def gradient_function(c, samples_x, y_values, sigma_noise, poly_initial, input_x, initial_likelihood, normalized_likelihood):

            # poly_matrix = self.poly(samples_x[:, np.newaxis], self.z_j)
            # pce_test = np.sum(c[:, np.newaxis, np.newaxis] * poly_matrix, axis=0)
            # pce = poly_initial(samples_x[:, np.newaxis], self.z_j)

            # terms = c * cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)
            terms = c * poly
            poly_opt = cp.sum(terms, axis=0)
            pce = poly_opt(*input_x, self.z_j)
            # pce = poly_opt(samples_x[0,:, np.newaxis], samples_x[1,:, np.newaxis], samples_x[2,:, np.newaxis], samples_x[3,:, np.newaxis], self.z_j)

            polynomials_eq = poly
            polynomials = polynomials_eq(*input_x, self.z_j)
            # polynomials = polynomials_eq(samples_x[0,:, np.newaxis], samples_x[1,:, np.newaxis], samples_x[2,:, np.newaxis], samples_x[3,:, np.newaxis], self.z_j)

            nominator = y_values[:, np.newaxis] - pce
            likelihood_quadrature =  (1 / (np.sqrt(2 * np.pi) * sigma_noise) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j)

            grad_like =  polynomials * nominator / (np.sqrt(2 * np.pi) * sigma_noise ** 3) * np.exp(-((y_values[:, np.newaxis] - pce) ** 2) / (2 * sigma_noise ** 2)) * self.w_j
            grad_like_sum = np.sum(grad_like, axis=2)
            like = np.sum(likelihood_quadrature, axis=1)
            grad = -np.sum((1 / (like)) * (grad_like_sum), axis=1)
            # print('grad = ', grad)

            return grad
        
        normalized_likelihood = []
        initial_likelihood = self.likelihood_function(c_initial, samples_x, y_values, sigma_noise, poly, input_x, 1, normalized_likelihood)

        # from scipy.optimize import check_grad
        # print(check_grad(self.likelihood_function, gradient_function, c_initial, samples_x, y_values, sigma_noise, poly, input_x, initial_likelihood, normalized_likelihood))

        start = time.time()
        # result = minimize(self.likelihood_function, c_initial, args=(samples_x, y_values, sigma_noise, poly, input_x, initial_likelihood, normalized_likelihood), method='BFGS', options={'disp': True})
        result = minimize(self.likelihood_function, c_initial, args=(samples_x, y_values, sigma_noise, poly, input_x, initial_likelihood, normalized_likelihood), method='BFGS', jac=gradient_function) #, options={'disp': True})
        print('time = ', time.time() - start)
        optimized_c = result.x
        # print(result.message)

        # plt.figure()
        # plt.plot(normalized_likelihood[1:])
        # plt.xlabel('iteration')
        # plt.ylabel('likelihood')
        # plt.yscale('log')
        # # tikzplotlib.save(rf"tex_files\bimodal\iteration_analytical.tex")
        # plt.show()
        
        return optimized_c, result.message
    
    def optimize_sigma(self, samples_x, y_values, sigma_initial, c_initial, poly, input_x, sigma_range):
        
        def objective(sigma):
            return self.likelihood_function(c_initial, samples_x, y_values, sigma, poly, input_x, initial_likelihood, [])
        
        initial_likelihood = self.likelihood_function(c_initial, samples_x, y_values, sigma_initial, poly, input_x, 1, [])
        result = minimize_scalar(objective, bounds=(sigma_range), method='bounded')
        optimized_sigma = result.x
        print("Optimized sigma:", optimized_sigma)
        
        return optimized_sigma
    
    def plot_sigma(self, samples_x, y_values, sigma_range, c_initial, poly, input_x):
        initial_likelihood = self.likelihood_function(c_initial, samples_x, y_values, sigma_range[0], poly, input_x, 1, [])
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], 100)
        likelihoods = [self.likelihood_function(c_initial, samples_x, y_values, sigma, poly, input_x, initial_likelihood, []) for sigma in sigma_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sigma_values, likelihoods)
        plt.xlabel('sigma')
        plt.ylabel('likelihood')
        plt.grid()
        # tikzplotlib.save(rf"tex_files\bimodal\MLE_simga_likelihood.tex")  
        plt.show()


    def generate_dist_spce(self, samples_x, samples_z, samples_eps, c, poly, input_x):
        
        terms = c * poly
        poly_opt = cp.sum(terms, axis=0)
        pce = poly_opt(*input_x, samples_z)
        dist_spce = pce + samples_eps

        # plt.figure()
        # plt.hist(pce[0,:], bins=60, alpha=0.5, label='SPCE ohne ')
        # plt.hist(dist_spce[0,:], bins=60, alpha=0.5, label='SPCE mit ')
        # plt.legend()
        # plt.show()

        return dist_spce
    
    def generate_dist_gpr(self, samples_x_test, samples_y_test, mean_test, sigma_test):
        gpr_test = Gaussian_Process(samples_x_test, samples_y_test, mean_test, sigma_test)
        mean_prediction_gpr, std_prediction_gpr = gpr_test.run(self.samples_x, self.y_values)
        # gpr_test.plot_gpr()
        samples_gpr_all = np.random.normal(mean_prediction_gpr[:, np.newaxis], std_prediction_gpr[:, np.newaxis], (samples_x_test.shape[0], 10000)) # 1 input
        # samples_gpr_all = np.random.normal(mean_prediction_gpr[:, np.newaxis], std_prediction_gpr[:, np.newaxis], (samples_x_test.shape[1], 10000)) # 4 inputs

        return mean_prediction_gpr, std_prediction_gpr, samples_gpr_all

    def plot_distribution(self, dist_spce, y, pdf, samples_x, samples_y, mean_prediction_gpr, std_prediction_gpr, dist_gpr): 

        samples_x_i = samples_x[:5]
        indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
            
        for x, sample in enumerate(samples_x_i):
            ''' KDE for SPCE '''
            kde = gaussian_kde(dist_spce[x,:])
            x_values_spce = np.linspace(min(dist_spce[x,:]), max(dist_spce[x,:]), 1000) 
            dist_spce_pdf_values = kde(x_values_spce)

            ''' KDE for GPR '''
            kde = gaussian_kde(dist_gpr[x,:])
            x_values_gpr = np.linspace(min(dist_gpr[x,:]), max(dist_gpr[x,:]), 1000) 
            dist_gpr_pdf_values = kde(x_values_gpr)

            
            bin_edges = np.arange(-4, 8, 0.2)

            plt.figure()            
            plt.plot(y, pdf[indices[x],:], label='reference')
            plt.plot(x_values_spce, dist_spce_pdf_values, label='SPCE')
            plt.plot(x_values_gpr, dist_gpr_pdf_values, label='GPR')
            # plt.hist(samples_y_test[x,:], bins=bin_edges, density=True, alpha=0.5, label='distribution reference')
            # plt.hist(dist_spce[x, :], bins=bin_edges, density=True, alpha=0.5, label='distribution SPCE')
            # plt.hist(samples_gpr, bins=bin_edges, density=True, alpha=0.5, label='distribution GPR')
            plt.xlabel('y')
            plt.ylabel('pdf')
            plt.xlim(-4, 8)
            plt.title(f'x = {sample}')
            plt.legend()  
            # tikzplotlib.save(rf"tex_files\bimodal\spce_gpr_{sample}.tex")  

        plt.show()   

    def plot_distribution_2(self, dist_spce, y, pdf, samples_x, samples_y, mean_prediction_gpr, std_prediction_gpr, dist_gpr, dist_pce): 

        samples_x_i = samples_x[:5]
        indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
            
        for x, sample in enumerate(samples_x_i):
            ''' KDE for SPCE '''
            kde = gaussian_kde(dist_spce[x,:])
            x_values_spce = np.linspace(min(dist_spce[x,:]), max(dist_spce[x,:]), 1000) 
            dist_spce_pdf_values = kde(x_values_spce)

            ''' KDE for GPR '''
            kde = gaussian_kde(dist_gpr[x,:])
            x_values_gpr = np.linspace(min(dist_gpr[x,:]), max(dist_gpr[x,:]), 1000) 
            dist_gpr_pdf_values = kde(x_values_gpr)

            ''' KDE for PCE '''
            kde = gaussian_kde(dist_pce[x,:])
            x_values_pce = np.linspace(min(dist_pce[x,:]), max(dist_pce[x,:]), 1000)
            dist_pce_pdf_values = kde(x_values_pce)


            # dist_gpr = cp.Normal(mean_prediction_gpr[x], std_prediction_gpr[x])
            # samples_gpr = dist_gpr.sample(size=samples_y.shape[1])
            # kde = gaussian_kde(samples_gpr)
            # x_values_gpr = np.linspace(min(samples_gpr), max(samples_gpr), 1000) 
            # dist_gpr_pdf_values = kde(x_values_gpr)
            
            bin_edges = np.arange(-4, 8, 0.2)

            plt.figure()            
            plt.plot(y, pdf[indices[x],:], label='reference')
            plt.plot(x_values_spce, dist_spce_pdf_values, label='SPCE')
            plt.plot(x_values_gpr, dist_gpr_pdf_values, label='GPR')
            plt.plot(x_values_pce, dist_pce_pdf_values, label='PCE')
            # plt.hist(samples_y_test[x,:], bins=bin_edges, density=True, alpha=0.5, label='distribution reference')
            # plt.hist(dist_spce[x, :], bins=bin_edges, density=True, alpha=0.5, label='distribution SPCE')
            # plt.hist(samples_gpr, bins=bin_edges, density=True, alpha=0.5, label='distribution GPR')
            plt.xlabel('y')
            plt.ylabel('pdf')
            plt.xlim(-4, 8)
            plt.title(f'x = {sample}')
            plt.legend()  
            # tikzplotlib.save(rf"tex_files\unimodal\spce_gpr_pce_{sample}.tex")  

        plt.show()   


    def start_c(self, input_x):

        poly_initial_q0 = self.poly.copy()
        poly_without_q0 = []
        q = str(poly_initial_q0.indeterminants[-1])
        for i, x in enumerate(poly_initial_q0): 
            if q in str(x):
                poly_without_q0.append(np.random.normal(0, 1) * x)
                poly_initial_q0[i] = 0
                

        poly_without_q0 = cp.polynomial(poly_without_q0)
        # poly_without_q0 = self.poly.copy()
        # q_z = str(poly_without_q0.indeterminants[-1])
        # for i, x in enumerate(poly_without_q0): 
        #     if q_z not in str(x):
        #         poly_without_q0[i] = 0

        # surrogate_q0 = cp.fit_regression(poly_initial_q0, (self.samples_x[0,:], self.samples_x[1,:], self.samples_x[2,:], self.samples_x[3,:]), self.y_values)
        self.surrogate_q0 = cp.fit_regression(poly_initial_q0, (*input_x,), self.y_values)
        self.coef_q0 = np.array(self.surrogate_q0.coefficients)

        poly_initial = self.surrogate_q0 + numpoly.sum(poly_without_q0)

       
        # plt.figure()
        # plt.scatter(self.samples_x[0,:], self.y_values, label='reference')
        # plt.scatter(self.samples_x[0,:], (self.surrogate_q0(self.samples_x[0,:], self.samples_x[1,:], self.samples_x[2,:], self.samples_x[3,:])), label='poly')
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.scatter(self.samples_x, self.y_values, label='reference')
        # plt.scatter(self.samples_x, (surrogate_q0(self.samples_x[:])), label='poly')
        # plt.legend()
        # plt.show()

        return self.surrogate_q0, poly_initial
    
    def loo_error(self, mean_ref, surrogate_q0, input_x):

        mean_poly = surrogate_q0(*input_x)
        error_loo = np.mean((mean_ref - mean_poly) ** 2) + np.std(self.y_values)

        return error_loo
    

    def compute_error(self, dist_spce, samples_y):

        u = np.linspace(0, 1, 1000)
        squared_diff = (np.quantile(dist_spce, u, axis=1) - np.quantile(samples_y, u, axis=1)) ** 2
        d_ws = np.trapz(squared_diff, u, axis=0)
        variance = np.var(samples_y)
        error_spce = np.mean(d_ws) / variance

        return error_spce 

    def cross_validation(self, sigma_noise, c_initial, poly):
        self.n_samples = self.samples_x.shape[1]
        if self.n_samples < 200:
            n_cv = 10
        if self.n_samples >= 200 and self.n_samples < 1000:
            n_cv = 5
        if self.n_samples >= 1000:
            n_cv = 3

        kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        cv_scores = []

        for train_index, val_index in kf.split(self.samples_x[0,:]): # 4 inputs
        # for train_index, val_index in kf.split(self.samples_x): # 1 input
            train_x = self.samples_x[:,train_index] # 4 inputs
            # train_x = self.samples_x[train_index] # 1 inpute
            train_y = self.y_values[train_index]
            input_x_train = [train_x[0,:, np.newaxis], train_x[1,:, np.newaxis], train_x[2,:, np.newaxis], train_x[3,:, np.newaxis]] # 4 inputs
            # input_x_train = [train_x[:, np.newaxis]] # 1 input
            c_opt, message = self.compute_optimal_c(train_x, train_y, sigma_noise, c_initial, poly, input_x_train)
            val_x = self.samples_x[:,val_index] # 4 inputs
            # val_x = self.samples_x[val_index] # 1 input
            val_y = self.y_values[val_index]
            input_x_val = [val_x[0,:, np.newaxis], val_x[1,:, np.newaxis], val_x[2,:, np.newaxis], val_x[3,:, np.newaxis]] # 4 inputs
            # input_x_val = [val_x[:, np.newaxis]] # 1 input
            normalized_likelihood=[]
            likelihood = - self.likelihood_function(c_opt, val_x, val_y, sigma_noise, poly, input_x_val, 1, normalized_likelihood)
            cv_scores.append(likelihood)
            # print('cv_score = ', cv_scores[-1])

        total_cv_score = np.sum(cv_scores)
        # print('total_cv_score = ', total_cv_score)
        return total_cv_score
    

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
    
    def standard_pce(self, dist_X, x, y, q):
        
        poly_pce = cp.generate_expansion(self.p, dist_X) #, cross_truncation=q)
        surrogate = cp.fit_regression(poly_pce, (*x,), y)

        # mean = cp.E(surrogate, dist_X) 
        # std = cp.Std(surrogate, dist_X)   
        
        return surrogate

    
