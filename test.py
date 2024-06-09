import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
# import tikzplotlib
from analytical_example_bimodal import AnalyticalExample
from spce import SPCE
import time

n_samples = 800
dist_X = cp.Uniform(0, 1)
samples_x = dist_X.sample(size=n_samples, rule='H') 
samples_x_i = np.array([0.2, 0.5, 0.75, 0.9])
indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
y = np.linspace(-4, 8, 1000)

example = AnalyticalExample(n_samples, y)
pdf, mean_1, mean_2, sigma_1, sigma_2 = example.calculate_pdf(samples_x)
samples_y = example.create_data_points(mean_1, mean_2, sigma_1, sigma_2, 1, samples_x).reshape(-1)
# example.plot_example(samples_x, samples_y, mean_1, mean_2, pdf, indices)
# example.plot_pdf(pdf, samples_x_i, indices)
variance = np.var(samples_y)


### SPCE
p = 1
sigma_noise = 0.3
dist_Z = cp.Normal(0, 1)
# dist_Z = cp.Uniform(-1, 1)
dist_joint = cp.J(dist_X, dist_Z)
N_q = 10

spce = SPCE(n_samples, p, samples_y.T, sigma_noise, samples_x, dist_joint)

c_initial = spce.start_c()
# sigma_noise = spce.compute_optimal_sigma(dist_Z, N_q, c_initial)
# np.save(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions/sigma_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy', sigma_noise)
# c_initial = np.random.uniform(-10, 10, size=spce.poly.shape[0])
optimized_c = spce.compute_optimal_c(samples_x, samples_y, dist_Z, sigma_noise, N_q, c_initial)
np.save(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions/c_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy', optimized_c)
# optimized_c = np.load(fr'solutions/800_p3_nq15_sigma0.35.npy')

# list_n = np.array([100,200,400,800,1600])
# error_n = np.zeros(list_n.shape[0])
# i = 0

# for n in list_n:
#     print(i)
#     optimized_c = np.load(fr'solutions/800_p3_nq15_sigma0.35.npy')
#     print('inital c = ', c_initial)
#     print('optimized c = ', optimized_c)


# test surrogate SPCE
dist_eps = cp.Normal(0, sigma_noise)
n_x = 1000
n_samples_test = 5000
samples_x_test = dist_X.sample(n_x)
# samples_x_test = np.array([0.1, 0.5, 0.75, 0.9])
samples_z_test = dist_Z.sample(n_samples_test)
samples_eps_test = dist_eps.sample(n_samples_test)
dist_spce = spce.generate_dist_spce(samples_x_test, samples_z_test, samples_eps_test, optimized_c)

# calculate Y of analytical model 
pdf_test, mean_1_test, mean_2_test, sigma_1_test, sigma_2_test = example.calculate_pdf(samples_x_test)
samples_y_test = example.create_data_points(mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, n_samples_test, samples_x_test)
samples_y_test_all = example.create_data_points(mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, 1, samples_x_test)

error_n = spce.compute_error(dist_spce, samples_y_test, samples_y_test_all)
print(error_n)
#     # print(error)
#     print('error nq i = ', error_n[i])
#     i += 1
    

# np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions/error_nq.npy', error_n)
# error_n = np.load(fr'solutions/error_nq.npy')

# plt.figure()
# plt.plot(list_n, error_n)
# plt.yscale('log')
# plt.title('Error sample number')
# plt.xlabel('number samples')
# plt.ylabel('error')

spce.plot_distribution(dist_spce, y, pdf_test, samples_x_test, samples_y_test)


plt.show()


