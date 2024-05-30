import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
# import tikzplotlib
from analytical_example import AnalyticalExample
from spce import SPCE
import time

n_samples = 5000
dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples) 
samples_x_i = np.array([0.2, 0.5, 0.75, 0.9])
indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
y = np.linspace(-4, 8, 1000)

example = AnalyticalExample(n_samples, y)
pdf, mean_1, mean_2, sigma_1, sigma_2 = example.calculate_pdf(samples_x)
samples_y = example.create_data_points(pdf)
# samples_y = example.create_data_points1(mean_1, mean_2, sigma_1, sigma_2)
# example.plot_example(samples_x, samples_y, mean_1, mean_2, pdf, indices)
# example.plot_pdf(pdf, samples_x_i, indices)


### SPCE
p = 3
sigma_noise = 1
dist_Z = cp.Normal(0, 1)
# dist_Z = cp.Uniform(-1, 1)
dist_joint = cp.J(dist_X, dist_Z)
N_q = 10
spce = SPCE(n_samples, p, samples_y, sigma_noise, samples_x, dist_joint)

c_initial = spce.start_c()

# optimized_c = spce.compute_optimal_c(dist_Z, sigma_noise, N_q, c_initial)
# np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions/5000_p3_nq10_sigma1.npy', optimized_c)
optimized_c = np.load(f'solutions/test_p3_sigma1.npy')
print('inital c = ', c_initial)
print('optimized c = ', optimized_c)

dist_eps = cp.Normal(0, sigma_noise)
n_samples_test = 20000
# samples_x_test = dist_X.sample(n_samples_test)
samples_x_test = np.array([0.2, 0.5, 0.75, 0.9])
samples_z_test = dist_Z.sample(n_samples_test)
samples_eps_test = dist_eps.sample(n_samples_test)
dist_spce = spce.generate_dist_spce(n_samples_test, samples_x_test, samples_z_test, samples_eps_test, optimized_c)
spce.plot_distribution(dist_spce, y, pdf, indices, samples_x_test)


plt.show()


