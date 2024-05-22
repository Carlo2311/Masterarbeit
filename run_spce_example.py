import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example import AnalyticalExample
from spce import SPCE

n_samples = 800
dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples, rule="H") 
samples_x_i = np.array([0.2, 0.5, 0.75, 0.9])
y = np.linspace(-4, 8, 1000)

example = AnalyticalExample(n_samples, y)
pdf, mean_1, mean_2, sigma_1, sigma_2 = example.calculate_pdf(samples_x)
samples_y = example.create_data_points(pdf, n_samples)
# example.plot_example(samples_x, samples_y, mean_1, mean_2, pdf)
# example.plot_pdf(pdf, samples_x_i)


### SPCE
p = 3
sigma_noise = 1
dist_Z = cp.Normal(0, 1)
dist_joint = cp.J(dist_X, dist_Z)
N_q = 10
spce = SPCE(n_samples, p, samples_y, sigma_noise, samples_x, dist_joint)

# c_initial = spce.start_c()
optimized_c = spce.compute_liklihood(dist_Z, sigma_noise, N_q)
# np.save('C:\\Users\\carlo\\Masterarbeit\\optimzized_c.npy', optimized_c)
# optimized_c = np.load('optimzized_c.npy')

dist_eps = cp.Normal(0, sigma_noise)

n_samples_test = 1000
# samples_x_test = dist_X.sample(n_samples_test, rule='H')
samples_x_test = [0.2, 0.5, 0.75, 0.9]
samples_z_test = dist_Z.sample(n_samples_test, rule='H')
samples_eps_test = dist_eps.sample(n_samples_test, rule='H')
spce.generate_dist_spce(n_samples_test, samples_x_test, samples_z_test, samples_eps_test, optimized_c)

plt.show()


