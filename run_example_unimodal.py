import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example_unimodal import AnalyticalExample
from spce import SPCE
from gaussian_process import Gaussian_Process
import time

n_samples = 50 #[50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
replications = [8] #[ 1,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
runs = 1
error_n = np.zeros((runs, len(replications)))
error_gpr = np.zeros((runs, len(replications)))
error_pce =np.zeros((runs, len(replications)))
nrmse_spce = np.zeros((runs, len(replications)))
nrmse_gpr = np.zeros((runs, len(replications)))
nrmse_pce = np.zeros((runs, len(replications)))


for r in range(runs):
    print('run = ', r)
    # for n, n_samples in enumerate(n_samples_all):
    for n, repli in enumerate(replications):
        print('n_samples = ', n_samples)
        samples_x_repeat = repli # 30
        dist_X = cp.Uniform(0, 1)
        samples_x = dist_X.sample(size=n_samples, rule='H') 
        samples_x = np.repeat(samples_x, samples_x_repeat)
        samples_x_i = np.array([0.1, 0.35, 0.6, 0.9])
        indices = [np.abs(samples_x - value).argmin() for value in samples_x_i]
        y = np.linspace(-4, 8, 1000)

        samples_x_resized = samples_x.reshape(n_samples, samples_x_repeat)

        samples_plot = 1
        example = AnalyticalExample(n_samples, y)
        pdf, mean, sigma = example.calculate_pdf(samples_x)
        samples_y = example.create_data_points(mean, sigma, samples_plot, samples_x, pdf).reshape(-1)
        # example.plot_example(samples_x, samples_y, mean, pdf, indices)
        # example.plot_pdf(pdf, samples_x_i, indices)

        samples_y_resized = samples_y.reshape(n_samples, samples_x_repeat)


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

        #### PCE #################################################################################################
        ## input sim data
        samples_pce_x = [samples_x_resized[:,0]]
        samples_pce_mean_y = np.mean(samples_y_resized, axis=1)
        samples_pce_std_y = np.std(samples_y_resized, axis=1)

        ### surrogate
        surrogate_pce_mean = spce.standard_pce(dist_X, samples_pce_x, samples_pce_mean_y, q)
        surrogate_pce_std = spce.standard_pce(dist_X, samples_pce_x, samples_pce_std_y, q)

        ###########################################################################################################

        poly, z_j = spce.get_params()
        input_x_start = [samples_x]
        input_x = [samples_x[:, np.newaxis]]

        surrogate_q0, poly_initial = spce.start_c(input_x_start)

        optimized_c = poly_initial.coefficients
        polynomials = cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)

        error_loo = spce.loo_error(mean, surrogate_q0, input_x_start)
        print('error_loo = ', error_loo)

        sigma_range = (0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo))
        sigma_noise_range = np.linspace(0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo), 4)
        sigma_noise_sorted = sorted(np.exp(sigma_noise_range), reverse=True)

        for sigma_noise_i in sigma_noise_sorted:
            print('sigma = ', sigma_noise_i)
            optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise_i, optimized_c, polynomials, input_x)
            # print(optimized_c)
            print(message)


        # ###### MLE ###################################
        # sigma_noise = sigma_noise_sorted[-1]

        for i in range(10):
            print('iteration = ', i)
            sigma_noise = spce.optimize_sigma(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x, sigma_range)
            # spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c, polynomials, input_x)
            optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)

        # initial_sigma_noise = sigma_noise_sorted[-1]
        # sigma_noise = []
        # sigma_diff = 1
        # i = 0
        # while sigma_diff > 0.01:
        #     print('iteration = ', i)
        #     if i == 0:
        #         sigma_noise_current = initial_sigma_noise
        #     else:
        #         sigma_noise_current = sigma_noise[i-1]
        #     sigma_noise.append(spce.optimize_sigma(samples_x, samples_y, sigma_noise_current, optimized_c, polynomials, input_x, sigma_range)) 
        #     # spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c, polynomials, input_x)
        #     optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise[i], optimized_c, polynomials, input_x)
        #     if i > 0:
        #         sigma_diff = abs(sigma_noise[i] - sigma_noise[i-1])
        #     i += 1
        # sigma_noise = sigma_noise[-1]

        ##### CV #####################################

        # sigma_noise = spce.compute_optimal_sigma(optimized_c, polynomials, sigma_range) # cross validation
        # optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)

        #####################################

        # np.save('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/c_test_50.npy', optimized_c)
        # np.save('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/sigma_test_50.npy', sigma_noise)

        # optimized_c = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/c_test.npy')
        # sigma_noise = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/sigma_test.npy')
        # print('sigma_noise 50 30 = ', sigma_noise)


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

        pce_mean_dist = surrogate_pce_mean(samples_x_test)
        pce_std_dist = np.abs(surrogate_pce_std(samples_x_test))
        dist_pce = np.random.normal(pce_mean_dist[:, np.newaxis], pce_std_dist[:, np.newaxis], (samples_x_test.shape[0], n_samples_test))

        #### calculate Y of analytical model 
        pdf_test, mean_test, sigma_test = example.calculate_pdf(samples_x_test)
        samples_y_test = example.create_data_points(mean_test, sigma_test, n_samples_test, samples_x_test, pdf_test)

        mean_prediction_gpr, std_prediction_gpr, dist_gpr = spce.generate_dist_gpr(samples_x_test, samples_y_test, mean, sigma)
        # spce.plot_distribution_2(dist_spce, y, pdf_test, samples_x_test, samples_y_test, mean_prediction_gpr, std_prediction_gpr, dist_gpr, dist_pce)
        # spce.plot_distribution(dist_spce, y, pdf_test, samples_x_test, samples_y_test, mean_prediction_gpr, std_prediction_gpr, dist_gpr) # without PCE


        error_n[r,n] = spce.compute_error(dist_spce, samples_y_test)
        error_gpr[r,n] = spce.compute_error(dist_gpr, samples_y_test)
        error_pce[r,n] = spce.compute_error(dist_pce, samples_y_test)

        samples_y_mean = np.mean(samples_y_test, axis=1)
        mean_spce = np.mean(dist_spce, axis=1)
        mean_pce = np.mean(dist_pce, axis=1)
        mean_gpr = np.mean(dist_gpr, axis=1)

        nrmse_spce[r,n] = np.sqrt(np.mean((mean_spce - samples_y_mean)**2)) / np.mean(samples_y_mean)
        nrmse_pce[r,n] = np.sqrt(np.mean((mean_pce - samples_y_mean)**2)) / np.mean(samples_y_mean)
        nrmse_gpr[r,n] = np.sqrt(np.mean((mean_gpr - samples_y_mean)**2)) / np.mean(samples_y_mean)
        print('nrmse spce i = ', nrmse_spce[r,n])
        print('nrmse gpr i = ', nrmse_gpr[r,n])

print('nrmse spce = ', nrmse_spce)
print('nrmse gpr = ', nrmse_gpr)
print('nrmse pce = ', nrmse_pce)

print('nrmse mean spce = ', np.mean(nrmse_spce, axis=0))
print('nrmse mean gpr = ', np.mean(nrmse_gpr, axis=0))
print('nrmse mean pce = ', np.mean(nrmse_pce, axis=0))

print('error spce = ', error_n)
print('error gpr = ', error_gpr)
print('error pce = ', error_pce)

# np.save('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/error_spce.npy', error_n)
# np.save('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/error_gpr.npy', error_gpr)

plt.figure('RMSE')
plt.plot(replications, np.mean(nrmse_spce, axis=0), label='SPCE')
plt.plot(replications, np.mean(nrmse_gpr, axis=0), label='GPR')
plt.plot(replications, np.mean(nrmse_pce, axis=0), label='PCE')
plt.xlabel(f'replications')
plt.ylabel('nrmse')
plt.grid()
# plt.legend()
plt.yscale('log')
# tikzplotlib.save(rf"tex_files\unimodal\unimodal_nrmse_spce_gpr_pce.tex")

plt.figure('Error')
plt.plot(replications, np.mean(error_n, axis=0), label='SPCE')
plt.plot(replications, np.mean(error_gpr, axis=0), label='GPR')
plt.plot(replications, np.mean(error_pce, axis=0), label='PCE')
plt.xlabel(f'replications')
plt.ylabel('error')
plt.grid()
# plt.legend()
plt.yscale('log')
# tikzplotlib.save(rf"tex_files\unimodal\unimodal_error_spce_gpr_pce.tex")

plt.show()


