import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example_unimodal import AnalyticalExample
from spce import SPCE
from gaussian_process import Gaussian_Process
import time

##### initialize input data #############################################################################
n_samples_all = 1500
replications = 30 
n_samples = int(n_samples_all / replications)

dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples) 
samples_x = np.repeat(samples_x, replications)
samples_x_resized = samples_x.reshape(n_samples, replications)

samples_x_i = np.array([0.1, 0.35, 0.6, 0.9]) # specified x-values for which the pdf is plotted
indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
samples_plot = 1
y = np.linspace(-4, 8, 1000)

example = AnalyticalExample(n_samples, y)
pdf, mean, sigma = example.calculate_pdf(samples_x)
samples_y = example.create_data_points(mean, sigma, samples_plot, samples_x, pdf).reshape(-1)
samples_y_resized = samples_y.reshape(n_samples, replications)

### plot example and pdfs ###
# example.plot_example(samples_x, samples_y, mean, pdf, indices)
# example.plot_pdf(pdf, samples_x_i, indices)

##### initilize parameters for the surrogates ###########################################################
p = 5
N_q = 5
q = 0.5
dist_Z = cp.Uniform(-1, 1) # or cp.Normal(0, 1)
dist_joint = cp.J(dist_X, dist_Z)

##### SPCE ##############################################################################################
spce = SPCE(n_samples, p, samples_y.T, samples_x, dist_joint, N_q, dist_Z, q)

### compute initial coefficients ###
surrogate_q0, poly_initial = spce.start_c(samples_x)
optimized_c = poly_initial.coefficients
polynomials = cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)

### compute range of sigma_noise ###
error_loo = spce.loo_error(mean, surrogate_q0, samples_x)
sigma_range = (0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo))
sigma_noise_range = np.linspace(sigma_range[0], sigma_range[-1], 5)
sigma_noise_sorted = sorted(np.exp(sigma_noise_range), reverse=True)

### warm-up strategy ###
for sigma_noise_i in sigma_noise_sorted:
    optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise_i, optimized_c, polynomials, samples_x)

###### choose MLE or CV for estimating sigma with subsequently optimization of c 
### MLE ###
for i in range(10):
    sigma_noise = spce.optimize_sigma(samples_y, optimized_c, polynomials, samples_x, sigma_range)
    optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise, optimized_c, polynomials, samples_x)

### CV ###
# sigma_noise = spce.compute_optimal_sigma(optimized_c, polynomials, sigma_range) 
# optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise, optimized_c, polynomials, samples_x)

##### test surrogates ####################################################################################
dist_eps = cp.Normal(0, sigma_noise)
n_x = 1000
n_samples_test = 10000
samples_x_test = dist_X.sample(n_x, rule='H')
samples_z_test = dist_Z.sample(n_samples_test)
samples_eps_test = dist_eps.sample(n_samples_test)
pdf_test, mean_test, sigma_test = example.calculate_pdf(samples_x_test)
samples_y_test = example.create_data_points(mean_test, sigma_test, n_samples_test, samples_x_test, pdf_test)

### generate distribution of the SPCE ###
dist_spce = spce.generate_dist_spce(samples_z_test, samples_eps_test, optimized_c, polynomials, samples_x_test)

##### PCE ######################
### input data for PCE ###
samples_pce_x = [samples_x_resized[:,0]]
samples_pce_mean_y = np.mean(samples_y_resized, axis=1)
samples_pce_std_y = np.std(samples_y_resized, axis=1)

### generating surrogates ###
surrogate_pce_mean = spce.standard_pce(dist_X, samples_pce_x, samples_pce_mean_y, q)
surrogate_pce_std = spce.standard_pce(dist_X, samples_pce_x, samples_pce_std_y, q)

### generate distribution of the PCE ###
pce_mean_dist = surrogate_pce_mean(samples_x_test)
pce_std_dist = np.abs(surrogate_pce_std(samples_x_test))
dist_pce = np.random.normal(pce_mean_dist[:, np.newaxis], pce_std_dist[:, np.newaxis], (samples_x_test.shape[0], n_samples_test))

##### GPR ######################
mean_prediction_gpr, std_prediction_gpr, dist_gpr = spce.generate_dist_gpr(samples_x_test, mean, sigma)

### plot distributions ###
spce.plot_distribution([dist_spce, dist_gpr, dist_pce], ['SPCE', 'GPR', 'PCE'], y, pdf_test, samples_x_test)

##### computation of the errors #########################################################################
samples_y_mean = np.mean(samples_y_test, axis=1)

### error using the Wasserstein distance ###
error_spce = spce.compute_error(dist_spce, samples_y_test)
error_gpr = spce.compute_error(dist_gpr, samples_y_test)
error_pce = spce.compute_error(dist_pce, samples_y_test)

### error of the mean value estimation ###
mean_spce = np.mean(dist_spce, axis=1)
mean_pce = np.mean(dist_pce, axis=1)
mean_gpr = np.mean(dist_gpr, axis=1)

nrmse_spce = np.sqrt(np.mean((mean_spce - samples_y_mean)**2)) / np.mean(samples_y_mean)
nrmse_pce = np.sqrt(np.mean((mean_pce - samples_y_mean)**2)) / np.mean(samples_y_mean)
nrmse_gpr = np.sqrt(np.mean((mean_gpr - samples_y_mean)**2)) / np.mean(samples_y_mean)

