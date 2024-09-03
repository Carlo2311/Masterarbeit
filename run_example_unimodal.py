import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
# import tikzplotlib
from analytical_example_unimodal import AnalyticalExample
from spce import SPCE
from gaussian_process import Gaussian_Process
import time

n_samples = 50
samples_x_repeat = 30
dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples, rule='H') 
samples_x = np.repeat(samples_x, samples_x_repeat)
samples_x_i = np.array([0.1, 0.35, 0.6, 0.9])
indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
y = np.linspace(-4, 8, 1000)

samples_plot = 1
example = AnalyticalExample(n_samples, y)
pdf, mean, sigma = example.calculate_pdf(samples_x)
samples_y = example.create_data_points(mean, sigma, samples_plot, samples_x, pdf).reshape(-1)
# example.plot_example(samples_x, samples_y, mean, pdf, indices)
# example.plot_pdf(pdf, samples_x_i, indices)


########################################################################################################################
# GPR

# gpr_example = Gaussian_Process(samples_x, samples_y, mean, sigma)
# mean_prediction_gpr, std_prediction_gpr = gpr_example.run()
# gpr_example.plot_gpr()

########################################################################################################################

### SPCE
p = 5
sigma_noise = 0.7
# dist_Z = cp.Normal(0, 1)
dist_Z = cp.Uniform(-1, 1)
dist_joint = cp.J(dist_X, dist_Z)
N_q = 10
q = 0.8

spce = SPCE(n_samples, p, samples_y.T, samples_x, dist_joint, N_q, dist_Z, q)

poly, z_j = spce.get_params()
input_x_start = [samples_x]
input_x = [samples_x[:, np.newaxis]]

surrogate_q0, poly_initial = spce.start_c(input_x_start)

optimized_c = poly_initial.coefficients
polynomials = cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)

error_loo = spce.loo_error(mean, surrogate_q0, input_x_start)

sigma_range = (0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo))
sigma_noise_range = np.linspace(np.log(np.sqrt(error_loo)), np.log(1), 4)
sigma_noise_sorted = sorted(np.exp(sigma_noise_range), reverse=True)

for sigma_noise_i in sigma_noise_sorted:
    print('sigma = ', sigma_noise_i)
    optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise_i, optimized_c, polynomials, input_x)
    print(optimized_c)
    print(message)


sigma_noise = spce.optimize_sigma(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x, sigma_range)
spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c, polynomials, input_x)
optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)
sigma_noise = spce.optimize_sigma(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x, sigma_range)
spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c, polynomials, input_x)
optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)
sigma_noise = spce.optimize_sigma(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x, sigma_range)
spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c, polynomials, input_x)
optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)

# sigma_noise = spce.compute_optimal_sigma(optimized_c, polynomials, sigma_range) # cross validation
# optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)

# np.save('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/c_50_30_cv.npy', optimized_c)
# np.save('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/sigma_50_30_cv.npy', sigma_noise)

optimized_c = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/c_50_30.npy')
sigma_noise = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/sigma_50_30.npy')
print('sigma_noise 50 30 = ', sigma_noise)


##################### test surrogate ###############################################################
dist_eps = cp.Normal(0, sigma_noise)
n_x = 1000
n_samples_test = 10000
samples_x_test = dist_X.sample(n_x, rule='H')
# samples_x_test = np.array([0.2, 0.5, 0.7, 0.9])
samples_z_test = dist_Z.sample(n_samples_test)
samples_eps_test = dist_eps.sample(n_samples_test)
input_x_test = [samples_x_test[:, np.newaxis]]
dist_spce = spce.generate_dist_spce(samples_x_test, samples_z_test, samples_eps_test, optimized_c, polynomials, input_x_test)

#### calculate Y of analytical model 
pdf_test, mean_test, sigma_test = example.calculate_pdf(samples_x_test)
samples_y_test = example.create_data_points(mean_test, sigma_test, n_samples_test, samples_x_test, pdf_test)

mean_prediction_gpr, std_prediction_gpr, dist_gpr = spce.generate_dist_gpr(samples_x_test, samples_y_test, mean_test, sigma_test)
spce.plot_distribution(dist_spce, y, pdf_test, samples_x_test, samples_y_test, mean_prediction_gpr, std_prediction_gpr)

error_n = spce.compute_error(dist_spce, samples_y_test)
error_gpr = spce.compute_error(dist_gpr, samples_y_test)
print('error = ', error_n)



plt.show()


