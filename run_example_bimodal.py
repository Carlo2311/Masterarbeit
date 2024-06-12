import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
# import tikzplotlib
from analytical_example_bimodal import AnalyticalExample
from spce import SPCE
import time

n_samples = 800
dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples,) 
samples_x_i = np.array([0.2, 0.5, 0.75, 0.9])
indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
y = np.linspace(-4, 8, 1000)

example = AnalyticalExample(n_samples, y)
pdf, mean_1, mean_2, sigma_1, sigma_2, mean_12, sigma_12 = example.calculate_pdf(samples_x)
samples_y = example.create_data_points(mean_1, mean_2, sigma_1, sigma_2, 1, samples_x).reshape(-1)
# example.plot_example(samples_x, samples_y, mean_1, mean_2, pdf, indices)
# example.plot_pdf(pdf, samples_x_i, indices)


############## SPCE ########################################
p = 4
sigma_noise = 0.7
# dist_Z = cp.Normal(0, 1)
dist_Z = cp.Uniform(-1, 1)
dist_joint = cp.J(dist_X, dist_Z)
N_q = 15

print('sigma = ', sigma_noise)

spce = SPCE(n_samples, p, samples_y.T, sigma_noise, samples_x, dist_joint)

c_initial = spce.start_c()

# optimized_c = spce.compute_optimal_c(samples_x, samples_y, dist_Z, sigma_noise, N_q, c_initial)
# optimized_c = np.load(fr'solutions_example_1/c_D2_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy') 
# sigma_noise = spce.optimize_sigma(samples_x, samples_y, dist_Z, N_q, sigma_noise, optimized_c)
# sigma_noise = 0.7123380806222996
# sigma_noise = spce.compute_optimal_sigma(dist_Z, N_q, c_initial)
# np.save(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_example_1/sigma_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy', sigma_noise)
# c_initial = np.random.uniform(-10, 10, size=spce.poly.shape[0])
# optimized_c_new = spce.compute_optimal_c(samples_x, samples_y, dist_Z, sigma_noise, N_q, optimized_c)
# np.save(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_example_1/c_D2_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy', optimized_c)
optimized_c_new = np.load(fr'solutions_example_1/c_D2_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy') 


############# test surrogate #########################
dist_eps = cp.Normal(0, sigma_noise)
n_x = 1000
n_samples_test = 10000
samples_x_test = dist_X.sample(n_x, rule='H')
# samples_x_test = np.array([0.1, 0.5, 0.75, 0.9])
samples_z_test = dist_Z.sample(n_samples_test)
samples_eps_test = dist_eps.sample(n_samples_test)
dist_spce = spce.generate_dist_spce(samples_x_test, samples_z_test, samples_eps_test, optimized_c_new)

# calculate Y of analytical model 
pdf_test, mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, mean_12_test, sigma_12_test = example.calculate_pdf(samples_x_test)
samples_y_test = example.create_data_points(mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, n_samples_test, samples_x_test)

error_n = spce.compute_error(dist_spce, samples_y_test)
print('error = ', error_n)

spce.plot_distribution(dist_spce, y, pdf_test, samples_x_test, samples_y_test, mean_12_test, sigma_12_test)


plt.show()


