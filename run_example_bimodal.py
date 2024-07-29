import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example_bimodal import AnalyticalExample
from spce import SPCE
import time


n_samples_all = [1600]

error_n = np.zeros((1, len(n_samples_all)))
error_gpr = np.zeros((1, len(n_samples_all)))

for i in range(error_n.shape[0]):
    for n_i, n_samples in enumerate(n_samples_all):
        print('i = ', i, 'n_samples = ', n_samples)

        repeat = True  
        while repeat:

            # n_samples = 1600
            dist_X = cp.Uniform(0, 1)
            samples_x = dist_X.sample(size=n_samples, rule='H') 
            samples_x_i = np.array([0.2, 0.5, 0.75, 0.9])
            indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
            y = np.linspace(-4, 8, 1000)

            example = AnalyticalExample(n_samples, y)
            pdf, mean_1, mean_2, sigma_1, sigma_2, mean_12, sigma_12 = example.calculate_pdf(samples_x)
            samples_y = example.create_data_points(mean_1, mean_2, sigma_1, sigma_2, 1, samples_x).reshape(-1)
            # example.plot_example(samples_x, samples_y, mean_1, mean_2, pdf, indices)
            # example.plot_pdf(pdf, samples_x_i, indices)

            p = 5
            sigma_noise = 0.6406371832484791 # 0.7530520509657908 
            # dist_Z = cp.Normal(0, 1)
            dist_Z = cp.Uniform(-1, 1)
            dist_joint = cp.J(dist_X, dist_Z)
            N_q = 10
            q = 0.8

            ############## SPCE #################################################################

            spce = SPCE(n_samples, p, samples_y.T, samples_x, dist_joint, N_q, dist_Z, q)

            poly, z_j = spce.get_params()
            # poly_matrix = poly(samples_x[:, np.newaxis], z_j)
            input_x_start = [samples_x]
            input_x = [samples_x[:, np.newaxis]]

            surrogate_q0, poly_initial = spce.start_c(input_x_start)

            optimized_c = poly_initial.coefficients
            polynomials = cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)

            error_loo = spce.loo_error(mean_12, surrogate_q0, input_x_start)
            
            sigma_range = (0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo))
            # # spce.plot_sigma(samples_x, samples_y, sigma_range, c_initial)
            sigma_noise_range = np.linspace(np.log(np.sqrt(error_loo)), np.log(1), 2)
            sigma_noise_sorted = sorted(np.exp(sigma_noise_range), reverse=True)

            for sigma_noise_i in sigma_noise_sorted:
                print('sigma = ', sigma_noise_i)
                optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise_i, optimized_c, polynomials, input_x)
                print(optimized_c)
                print(message)
            if message == 'Optimization terminated successfully.':
                repeat = False  

        # optimized_c = np.load(fr'solutions_example_1/c_D2_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy') 
        # print('c = ', optimized_c)
        # sigma_noise = spce.optimize_sigma(samples_x, samples_y, sigma_noise, optimized_c)
        # spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c)

        # sigma_noise = spce.compute_optimal_sigma(optimized_c, poly_matrix, sigma_range) # cross validation
        # np.save(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_example_1/sigma_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}_CV.npy', sigma_noise)

        # optimized_c = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c)
        # np.save(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_example_1/c_normal_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy', optimized_c)
        # optimized_c_new = np.load(fr'solutions_example_1/c_D2_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}.npy') 
        # sigma_noise = spce.optimize_sigma(samples_x, samples_y, sigma_noise, optimized_c)
        # spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c_new)
        # optimized_c = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c)
        # sigma_noise = spce.optimize_sigma(samples_x, samples_y, sigma_noise, optimized_c)
        # spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c_new)

        # repeat = True  
        # while repeat:
        #     optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c)
        #     if message != 'NaN result encountered.':
        #         repeat = False  

        # np.save(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_example_1/c_D2_{n_samples}_p{p}_nq{N_q}_sigma{sigma_noise}_CV.npy', optimized_c)
        # print('last sigma = ', sigma_noise)

        sigma_noise = 0.6406371832484791
        ############# test surrogate ##########################################################
        dist_eps = cp.Normal(0, sigma_noise)
        n_x = 1000
        n_samples_test = 10000
        samples_x_test = dist_X.sample(n_x, rule='H')
        # samples_x_test = np.array([0.2, 0.5, 0.7, 0.9])
        samples_z_test = dist_Z.sample(n_samples_test)
        samples_eps_test = dist_eps.sample(n_samples_test)
        input_x_test = [samples_x_test[:, np.newaxis]]
        dist_spce = spce.generate_dist_spce(samples_x_test, samples_z_test, samples_eps_test, optimized_c, polynomials, input_x_test)

        # calculate Y of analytical model 
        pdf_test, mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, mean_12_test, sigma_12_test = example.calculate_pdf(samples_x_test)
        samples_y_test = example.create_data_points(mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, n_samples_test, samples_x_test)

        mean_prediction_gpr, std_prediction_gpr, dist_gpr = spce.generate_dist_gpr(samples_x_test, samples_y_test, mean_12_test, sigma_12_test)
        spce.plot_distribution(dist_spce, y, pdf_test, samples_x_test, samples_y_test, mean_prediction_gpr, std_prediction_gpr)

        error_n[i, n_i], error_gpr[i, n_i] = spce.compute_error(dist_spce, samples_y_test, dist_gpr)
        print('error spce = ', error_n[i, n_i])
        print('error spce = ', error_gpr[i, n_i])


print(error_n, error_gpr)
error_spce = np.mean(error_n, axis=0)
error_gpr_mean = np.mean(error_gpr, axis=0)
print('error mean spce =', error_spce)   
print('error mean gpr =', error_gpr_mean)   

# plt.figure()
# plt.plot(n_samples_all, error_spce, label='SPCE')
# plt.plot(n_samples_all, error_gpr_mean, label='GPR')
# plt.xlabel(f'N')
# plt.ylabel('error')
# plt.grid()
# plt.yscale('log')
# tikzplotlib.save(rf"tex_files\bimodal\bimodal_error_gpr_2.tex")

plt.show()
