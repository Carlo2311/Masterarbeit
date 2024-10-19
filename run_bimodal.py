import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
from analytical_example_bimodal import AnalyticalExample
from spce import SPCE
import time

##### initialize input data #############################################################################
n_samples = 1600
dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples, rule='H') 
samples_x_i = np.array([0.2, 0.5, 0.75, 0.9])
indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
y = np.linspace(-4, 8, 1000)

example = AnalyticalExample(n_samples, y)
pdf, mean_1, mean_2, sigma_1, sigma_2, mean_12, sigma_12 = example.calculate_pdf(samples_x)
samples_y = example.create_data_points(mean_1, mean_2, sigma_1, sigma_2, 1, samples_x).reshape(-1)

### plot example and pdfs ###
# example.plot_example(samples_x, samples_y, mean_1, mean_2, pdf, indices)
# example.plot_pdf(pdf, samples_x_i, indices)

##### initilize parameters for the surrogates ###########################################################
p = 5
dist_Z = cp.Uniform(-1, 1)
dist_joint = cp.J(dist_X, dist_Z)
N_q = 10
q = 0.5

##### SPCE ##############################################################################################

spce = SPCE(n_samples, p, samples_y.T, samples_x, dist_joint, N_q, dist_Z, q)

### compute initial coefficients ###
surrogate_q0, poly_initial = spce.start_c(samples_x)
optimized_c = poly_initial.coefficients
polynomials = cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)

### compute range of sigma_noise ###
error_loo = spce.loo_error(mean_12, surrogate_q0, samples_x)
sigma_range = (0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo))
sigma_noise_range = np.linspace(sigma_range[0], sigma_range[-1], 5)
sigma_noise_sorted = sorted(np.exp(sigma_noise_range), reverse=True)

### warm-up strategy ###
for sigma_noise_i in sigma_noise_sorted:
    optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise_i, optimized_c, polynomials, samples_x)

###### choose MLE or CV for estimating sigma with subsequently optimization of c 
### MLE ###
for j in range(15):
    sigma_noise = spce.optimize_sigma(samples_y, optimized_c, polynomials, samples_x, sigma_range)
    optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise, optimized_c, polynomials, samples_x)

### CV ###
# sigma_noise = spce.compute_optimal_sigma(optimized_c, polynomials, sigma_range) 
# optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise, optimized_c, polynomials, samples_x)

##### test surrogate ####################################################################################
dist_eps = cp.Normal(0, sigma_noise)
n_x = 1000
n_samples_test = 10000
samples_x_test = dist_X.sample(n_x, rule='H')
# samples_x_test = np.array([0.2, 0.5, 0.7, 0.9])
samples_z_test = dist_Z.sample(n_samples_test)
samples_eps_test = dist_eps.sample(n_samples_test)
pdf_test, mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, mean_12_test, sigma_12_test = example.calculate_pdf(samples_x_test)
samples_y_test = example.create_data_points(mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, n_samples_test, samples_x_test)

### generate distribution of the SPCE ###
dist_spce = spce.generate_dist_spce(samples_z_test, samples_eps_test, optimized_c, polynomials, samples_x_test)

##### GPR ######################
mean_prediction_gpr, std_prediction_gpr, dist_gpr = spce.generate_dist_gpr(samples_x_test, mean_12_test, sigma_12_test)

### plot distribution ###
spce.plot_distribution([dist_spce, dist_gpr], ['SPCE', 'GPR'], y, pdf_test, samples_x_test)

### computation of the error using the Wasserstein distance ###
error_n = spce.compute_error(dist_spce, samples_y_test)
error_gpr = spce.compute_error(dist_gpr, samples_y_test)


