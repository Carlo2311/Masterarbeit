import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import chaospy as cp
from spce import SPCE
from scipy import stats
import math


####### input data simulation #################################################################
file_path_input = 'simulation_data/casematrix.csv'
df = pd.read_csv(file_path_input)

windspeed = df['Windspeed [m/s]'].to_numpy()
turbulence_intensity = df['TI [-]'].to_numpy()
rho = df['Rho [kg/m3]'].to_numpy()
yaw_angle = df['Yaw Angle [°]'].to_numpy()
seed1 = df['Seed 1'].to_numpy()
seed2 = df['Seed 2'].to_numpy()

# samples_total = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400, 5700, 6000, 6300, 6600, 6900, 7200, 7500, 7800, 8100, 8400, 8700, 9000]
samples_total = [9000,10,30,50,70,100,150,200,250,300,600,900]
# rep = int(samples_total[0] / 300) # wenn über 300 samples
# rep = 1 # wenn unter 300 samples
iteration = 1


error_spce = np.zeros((iteration, len(samples_total)))
error_gpr = np.zeros((iteration, len(samples_total)))
error_pce = np.zeros((iteration, len(samples_total)))
nrmse_spce = np.zeros((iteration, len(samples_total)))
nrmse_gpr = np.zeros((iteration, len(samples_total)))
nrmse_pce = np.zeros((iteration, len(samples_total)))
nrmse_spce_std = np.zeros((iteration, len(samples_total)))
nrmse_pce_std = np.zeros((iteration, len(samples_total)))
nrmse_spce_pred = np.zeros((iteration, len(samples_total)))
nrmse_pce_pred = np.zeros((iteration, len(samples_total)))
p_value_spce_mean = np.zeros((iteration, len(samples_total)))
p_value_gpr_mean = np.zeros((iteration, len(samples_total)))
p_value_pce_mean = np.zeros((iteration, len(samples_total)))
test_statistic_spce_mean = np.zeros((iteration, len(samples_total)))
test_statistic_gpr_mean = np.zeros((iteration, len(samples_total)))
test_statistic_pce_mean = np.zeros((iteration, len(samples_total)))


for iter in range(iteration):
    for s_i, samples_tot in enumerate(samples_total):
        print(s_i, samples_tot)
        if samples_tot <=300: 
            rep = 1
        else:
            rep = int(samples_tot / 300)

        dist_windspeed = cp.Beta(1.02, 3, 3, 25)
        dist_turbulence_intensity = cp.Uniform(min(turbulence_intensity), max(turbulence_intensity)) 
        dist_rho = cp.Uniform(min(rho), max(rho))
        dist_yaw_angle = cp.Uniform(min(yaw_angle), max(yaw_angle))

        dist_X = cp.J(dist_windspeed, dist_turbulence_intensity, dist_rho, dist_yaw_angle)

        ######### Rosenblatt #################################################

        def rosenblatt_transformation(multi_distribution_standard, multi_distribution_physical, samples_physical):
            samples_standard = multi_distribution_standard.inv(multi_distribution_physical.fwd(samples_physical))
            return samples_standard 

        dist_standard = cp.J(cp.Uniform(-1, 1), cp.Uniform(-1, 1), cp.Uniform(-1, 1), cp.Uniform(-1, 1))

        samples_x_all_physical = np.array([windspeed, turbulence_intensity, rho, yaw_angle])
        samples_x_all = rosenblatt_transformation(dist_standard, dist_X, samples_x_all_physical)

        samples_x = samples_x_all[:,:9000]
        samples_x_resized = np.reshape(samples_x, (4, 30, 300))

        ###
        index_samples = np.random.choice(300, size=int(samples_tot/rep), replace=False)
        index_rep = np.random.choice(30, size=rep, replace=False)
        samples_x = samples_x_resized[:,index_rep,:]
        samples_x = samples_x[:,:,index_samples]

        ###

        # samples_x = samples_x_resized[:,:rep,:samples_tot]
        # samples_x = samples_x_resized[:,:rep,10*iter:samples_tot+(10*iter)]

        # # reshape_vec = 30 * samples_tot # when all replications are used
        # reshape_vec = rep * 300 # when all samples are used

        samples_x = np.reshape(samples_x, (4, samples_tot))

        samples_x_test = samples_x_all[:,9000:]
        samples_x_test_resized = np.reshape(samples_x_test, (4, 30, 30))


        ######## output data simulation #############################################################

        file_path_output = 'simulation_data/surrogate_data_v2.csv'
        df = pd.read_csv(file_path_output)
        df_sorted = df.sort_values(by='Case')

        case = df_sorted['Case'].to_numpy()
        root_myb_mean = df_sorted['RootMyb_[kN-m]_mean'].to_numpy()
        twh_tx_mean = df_sorted['TwHtALxt_[m/s^]_mean'].to_numpy() *1000
        twh_ty_mean = df_sorted['TwHtALyt_[m/s^]_mean'].to_numpy() 
        twh_tz_mean = df_sorted['TwHtALzt_[m/s^]_mean'].to_numpy() *1000
        root_myb_sdv = df_sorted['RootMyb_[kN-m]_sdv'].to_numpy()
        twh_tx_sdv = df_sorted['TwHtALxt_[m/s^]_sdv'].to_numpy() 
        twh_ty_sdv = df_sorted['TwHtALyt_[m/s^]_sdv'].to_numpy() 
        twh_tz_sdv = df_sorted['TwHtALzt_[m/s^]_sdv'].to_numpy() 
        root_myb_min = df_sorted['RootMyb_[kN-m]_min'].to_numpy()
        twh_tx_min = df_sorted['TwHtALxt_[m/s^]_min'].to_numpy()
        twh_ty_min = df_sorted['TwHtALyt_[m/s^]_min'].to_numpy()
        twh_tz_min = df_sorted['TwHtALzt_[m/s^]_min'].to_numpy()
        root_myb_max = df_sorted['RootMyb_[kN-m]_max'].to_numpy()
        twh_tx_max = df_sorted['TwHtALxt_[m/s^]_max'].to_numpy()
        twh_ty_max = df_sorted['TwHtALyt_[m/s^]_max'].to_numpy()
        twh_tz_max = df_sorted['TwHtALzt_[m/s^]_max'].to_numpy()

        root_myb_mean_normalized = root_myb_mean / max(root_myb_mean)
        root_myb_sdv_normalized = root_myb_sdv / max(root_myb_mean)
        root_myb_max_normalized = root_myb_max / max(root_myb_max)
        twh_tx_mean_normalized = twh_tx_mean / max(twh_tx_mean) 
        twh_ty_mean_normalized = twh_ty_mean / max(twh_ty_mean) 
        twh_tz_mean_normalized = twh_tz_mean / max(twh_tz_mean) 
        twh_tx_sdv_normalized = twh_tx_sdv / max(twh_tx_sdv) 
        twh_ty_sdv_normalized = twh_ty_sdv / max(twh_ty_sdv) 
        twh_tz_sdv_normalized = twh_tz_sdv / max(twh_tz_sdv) 

        ###standard
        # samples_y_all = root_myb_mean[:9000]
        # samples_y_resized = np.reshape(samples_y, (30, 300))

        # samples_y_test = root_myb_mean[9000:]
        # samples_y_test_resized = np.reshape(samples_y_test, (30, 30))

        #### normalized
        samples_y_all = root_myb_mean_normalized[:9000] 
        # div = int(samples_tot / 300)
        # samples_y_resized = np.reshape(samples_y_all, (div, 300))
        samples_y_resized = np.reshape(samples_y_all, (30, 300))

        ###
        samples_y_resized = samples_y_resized[index_rep,:]
        samples_y_resized = samples_y_resized[:,index_samples]



        # samples_y_resized = samples_y_resized[:rep,10*iter:samples_tot+(10*iter)]
        samples_y = samples_y_resized.reshape(samples_tot)

        samples_y_test = root_myb_mean_normalized[9000:]
        samples_y_test_resized = np.reshape(samples_y_test, (30, 30))


        mean_samples_y = np.mean(samples_y_resized, axis=0)
        std_samples_y = np.std(samples_y_resized, axis=0)
        mean_samples_y_test = np.mean(samples_y_test_resized, axis=0)
        std_samples_y_test = np.std(samples_y_test_resized, axis=0)

        plt.figure()
        plt.scatter(samples_x[0,:], samples_y)
        plt.scatter(samples_x_test[0,:], samples_y_test)
        plt.xlabel('windspeed [m/s]')
        plt.ylabel('blade load [mm/s^2]')
        plt.grid()
        plt.figure()
        plt.scatter(samples_x[1,:], samples_y)
        plt.scatter(samples_x_test[1,:], samples_y_test)
        plt.xlabel('turbulence intensity')
        plt.ylabel('blade load [mm/s^2]')
        plt.grid()
        plt.figure()
        plt.scatter(samples_x[2,:], samples_y)
        plt.scatter(samples_x_test[2,:], samples_y_test)
        plt.xlabel('rho')
        plt.ylabel('blade load [mm/s^2]')
        plt.grid()
        plt.figure()
        plt.scatter(samples_x[3,:], samples_y)
        plt.scatter(samples_x_test[3,:], samples_y_test)
        plt.xlabel('yaw angle')
        plt.ylabel('blade load [mm/s^2]')
        plt.grid()
        plt.show()

        # plt.figure()
        # # plt.scatter(samples_x[0,:], samples_y, s=1)
        # plt.errorbar(samples_x[0,:samples_tot], mean_samples_y, yerr=std_samples_y, fmt='o', capsize=5, color='g')
        # # plt.scatter(samples_x_test[0,:30], samples_y_test, s=1)
        # plt.errorbar(samples_x_test[0,:30], mean_samples_y_test, yerr=std_samples_y_test, fmt='o', capsize=5, color='r')
        # plt.xlabel('windspeed [m/s]')
        # plt.ylabel('blade load [mm/s^2]')
        # plt.grid()
        # plt.figure()
        # # plt.scatter(samples_x[1,:300], samples_y, s=1)
        # plt.errorbar(samples_x[1,:samples_tot], mean_samples_y, yerr=std_samples_y, fmt='o', capsize=5, color='g')
        # # plt.scatter(samples_x_test[1,:30], samples_y_test, s=1)
        # plt.errorbar(samples_x_test[1,:30], mean_samples_y_test, yerr=std_samples_y_test, fmt='o', capsize=5, color='r')
        # plt.xlabel('turbulence intensity')
        # plt.ylabel('blade load [mm/s^2]')
        # plt.grid()
        # plt.figure()
        # # plt.scatter(samples_x[2,:300], samples_y, s=1)
        # plt.errorbar(samples_x[2,:samples_tot], mean_samples_y, yerr=std_samples_y, fmt='o', capsize=5, color='g')
        # # plt.scatter(samples_x_test[2,:30], samples_y_test, s=1)
        # plt.errorbar(samples_x_test[2,:30], mean_samples_y_test, yerr=std_samples_y_test, fmt='o', capsize=5, color='r')
        # plt.xlabel('rho')
        # plt.ylabel('blade load [mm/s^2]')
        # plt.grid()
        # plt.figure()
        # # plt.scatter(samples_x[3,:300], samples_y, s=1)
        # plt.errorbar(samples_x[3,:samples_tot], mean_samples_y, yerr=std_samples_y, fmt='o', capsize=5, color='g')
        # # plt.scatter(samples_x_test[3,:30], samples_y_test, s=1)
        # plt.errorbar(samples_x_test[3,:30], mean_samples_y_test, yerr=std_samples_y_test, fmt='o', capsize=5, color='r')
        # plt.xlabel('yaw angle')
        # plt.ylabel('blade load [mm/s^2]')
        # plt.grid()
        # plt.show()


        ############## SPCE #################################################################

        n_samples = samples_x.shape[1]
        p = 5
        # dist_Z = cp.Normal(0, 1)
        dist_Z = cp.Uniform(-1, 1)
        dist_joint = cp.J(dist_standard, dist_Z)
        N_q = 5 
        q = 0.5
       
        spce = SPCE(n_samples, p, samples_y.T, samples_x, dist_joint, N_q, dist_Z, q)

        poly, z_j = spce.get_params()
        input_x_start = [samples_x[0,:], samples_x[1,:], samples_x[2,:], samples_x[3,:]]
        input_x = [samples_x[0,:, np.newaxis], samples_x[1,:, np.newaxis], samples_x[2,:, np.newaxis], samples_x[3,:, np.newaxis]]

        surrogate_q0, poly_initial = spce.start_c(input_x_start)

        optimized_c = poly_initial.coefficients
        polynomials = cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)

        #### PCE #####
        ### input sim data
        # samples_pce_x = [samples_x[0,:samples_tot], samples_x[1,:samples_tot], samples_x[2,:samples_tot], samples_x[3,:samples_tot]] # all replications
        # samples_pce_x = [samples_x[0,:], samples_x[1,:], samples_x[2,:], samples_x[3,:]] # all samples
        samples_pce_x = [samples_x[0,:300], samples_x[1,:300], samples_x[2,:300], samples_x[3,:300]] # all samples
        samples_pce_mean_y = np.mean(samples_y_resized, axis=0)
        samples_pce_std_y = np.std(samples_y_resized, axis=0)

        ### surrogate
        surrogate_pce_mean = spce.standard_pce(dist_X, samples_pce_x, samples_pce_mean_y, q)
        surrogate_pce_std = spce.standard_pce(dist_X, samples_pce_x, samples_pce_std_y, q)
        ######################################################################

        error_loo = spce.loo_error(samples_y, surrogate_q0, input_x_start)

        sigma_range = (0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo))
        # print(sigma_range)
        # sigma_range = (0.005 , 0.5) 
        print('sigma range = ', sigma_range)
        sigma_noise_range = np.linspace(np.log(sigma_range[0]) , np.log(sigma_range[1]), 5)
        sigma_noise_sorted = sorted(np.exp(sigma_noise_range), reverse=True)

        for sigma_noise_i in sigma_noise_sorted:
            print('sigma_i = ', sigma_noise_i)
            optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise_i, optimized_c, polynomials, input_x)
            print(optimized_c)
            print(message)

        ######### CV ####################################################################

        # sigma_noise = spce.compute_optimal_sigma(optimized_c, polynomials, sigma_range) # cross validation

        # optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)

        # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/load_c_q{q}_p{p}_Nq{N_q}_cv_standardized.npy', optimized_c)      
        # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/load_sigma_q{q}_p{p}_Nq{N_q}_cv_standardized.npy', sigma_noise)

        # optimized_c = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/good_solutions/very_good/c_q{q}_p{p}_Nq{N_q}_cv_standardized.npy')
        # sigma_noise = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/good_solutions/very_good/sigma_q{q}_p{p}_Nq{N_q}_cv_standardized.npy')

        ######### MLE ####################################################################

        sigma_noise = sigma_noise_sorted[-1]
        
        for i in range(7):
            print('iteration = ', i)
            sigma_noise = spce.optimize_sigma(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x, sigma_range)
            # spce.plot_sigma(samples_x, samples_y, sigma_range, optimized_c, polynomials, input_x)
            optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)

        # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_pq/c_q{q}_p{p}_Nq{N_q}_mle.npy', optimized_c)
        # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_pq/sigma_q{q}_p{p}_Nq{N_q}_mle.npy', sigma_noise)

        # sigma_noise = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_q/sigma_q0.9_p5_Nq5_mle_standardized_norm.npy')
        # optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)

        # optimized_c = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_q/c_q{q}_p{p}_Nq{N_q}_mle_standardized_norm.npy')
        # sigma_noise = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_q/sigma_q{q}_p{p}_Nq{N_q}_mle_standardized_norm.npy')

        print('sigma = ', sigma_noise)
        print('c = ', optimized_c)

        ############# test surrogate ##########################################################
        dist_eps = cp.Normal(0, sigma_noise)
        n_samples_test = 10000
        input_x_test = [samples_x_test[0,:, np.newaxis], samples_x_test[1,:, np.newaxis], samples_x_test[2,:, np.newaxis], samples_x_test[3,:, np.newaxis]]
        # input_x_test = [samples_x_test[0,:, np.newaxis], samples_x_test[1,10, np.newaxis], samples_x_test[2,10, np.newaxis], samples_x_test[3,10, np.newaxis]] # 3 parameters fixed
        samples_z_test = dist_Z.sample(n_samples_test)
        samples_eps_test = dist_eps.sample(n_samples_test)
        dist_spce = spce.generate_dist_spce(samples_x_test, samples_z_test, samples_eps_test, optimized_c, polynomials, input_x_test)

        pce_mean_dist = surrogate_pce_mean(samples_x_test[0,:], samples_x_test[1,:], samples_x_test[2,:], samples_x_test[3,:])
        pce_std_dist = np.abs(surrogate_pce_std(samples_x_test[0,:], samples_x_test[1,:], samples_x_test[2,:], samples_x_test[3,:]))
        dist_pce = np.random.normal(pce_mean_dist[:, np.newaxis], pce_std_dist[:, np.newaxis], (samples_x_test.shape[1], n_samples_test))

        ### GPR
        # mean_gpr, std_gpr, dist_gpr = spce.generate_dist_gpr(samples_x_test, samples_y_test, samples_pce_mean_y, samples_pce_std_y)
        #########

        size_y_test = samples_y_test_resized.shape[1]
        error_spce[iter,s_i] = spce.compute_error(dist_spce[:size_y_test,:], samples_y_test_resized.T)
        error_pce[iter,s_i] = spce.compute_error(dist_pce[:size_y_test,:], samples_y_test_resized.T)
        # error_gpr[iter,s_i] = spce.compute_error(dist_gpr[:size_y_test,:], samples_y_test_resized.T)
        print('spce wasserstein distance = ', error_spce)
        print('pce wasserstein distance = ', error_pce)
        # print('gpr wasserstein distance = ', error_gpr)

        mean_spce = np.mean(dist_spce, axis=1)
        std_spce = np.std(dist_spce, axis=1)

        ####### test simulation data #############################################################
        mean_sim = np.mean(samples_y_test_resized, axis=0)
        std_sim = np.std(samples_y_test_resized, axis=0)
        print('mean sim = ', np.mean(mean_sim))

        # rmse_spce[iter,s_i] = (np.sqrt(np.mean((mean_spce[:size_y_test] - mean_sim)**2))) #/ (np.mean(mean_sim))
        # rmse_pce[iter,s_i] = (np.sqrt(np.mean((pce_mean_dist[:size_y_test] - mean_sim)**2))) #/ (np.mean(mean_sim))
        # # nrmse_gpr[iter,s_i] = (np.sqrt(np.mean((mean_prediction_gpr[:size_y_test] - mean_sim)**2))) / (np.mean(mean_sim))
        # print('spce rmse = ', rmse_spce)
        # print('pce rmse = ', rmse_pce)
        # print('gpr nrmse = ', nrmse_gpr)
        nrmse_spce[iter,s_i] = (np.sqrt(np.mean((mean_spce[:size_y_test] - mean_sim)**2))) / (np.mean(mean_sim))
        nrmse_spce_std[iter,s_i] = np.sqrt(np.mean((std_spce[:size_y_test] - std_sim)**2)) / (np.mean(std_sim))
        nrmse_pce[iter,s_i] = (np.sqrt(np.mean((pce_mean_dist[:size_y_test] - mean_sim)**2))) / (np.mean(mean_sim))
        nrmse_pce_std[iter,s_i] = np.sqrt(np.mean((pce_std_dist[:size_y_test] - std_sim)**2)) / (np.mean(std_sim))
        nrmse_spce_pred[iter,s_i] = np.sqrt(np.mean((std_spce[:size_y_test] - root_myb_sdv_normalized[9000:9030])**2)) / (np.mean(root_myb_sdv_normalized[9000:9030]))
        nrmse_pce_pred[iter,s_i] = np.sqrt(np.mean((pce_std_dist[:size_y_test] - root_myb_sdv_normalized[9000:9030])**2)) / (np.mean(root_myb_sdv_normalized[9000:9030]))
        print('spce nrmse = ', nrmse_spce)
        print('pce nrmse = ', nrmse_pce)
        print('spce nrmse std = ', nrmse_spce_std)
        print('pce nrmse std = ', nrmse_pce_std)
        print('spce nrmse pred = ', nrmse_spce_pred)
        print('pce nrmse pred = ', nrmse_pce_pred)


        index_input_x = 0
        sorted_indices = np.argsort(samples_x_test[index_input_x,:size_y_test])
        x_index_plot = sorted_indices[13]

        mean_spce_plot = mean_spce[:size_y_test]#[sorted_indices]
        mean_sim_plot = mean_sim#[sorted_indices]
        mean_pce_plot = pce_mean_dist[:size_y_test]#[sorted_indices]
        #mean_gpr_plot = mean_gpr[:size_y_test]
        samples_plot = samples_x_test[index_input_x,:size_y_test]#[sorted_indices]
        y_samples_plot = samples_y_test_resized[:,x_index_plot]
        dist_spce_plot = dist_spce[x_index_plot,:]
        #dist_gpr_plot = dist_gpr[x_index_plot,:]
        std_spce_plot = std_spce[:size_y_test]#[sorted_indices]
        std_sim_plot = std_sim#[sorted_indices] 
        std_pce = pce_std_dist[:size_y_test]#[sorted_indices]
        # std_gpr = std_gpr[:size_y_test]
        dist_pce_x_plot = dist_pce[x_index_plot,:]


        ###### KS test ####################################################################
        spce_dist_ks = dist_spce[:size_y_test,:]
        pce_dist_ks = dist_pce[:size_y_test,:]
        #gpr_dist_ks = dist_gpr[:size_y_test,:]
        test_statistic_spce = np.zeros(spce_dist_ks.shape[0])
        p_value_spce = np.zeros(spce_dist_ks.shape[0])
        test_statistic_pce = np.zeros(pce_dist_ks.shape[0])
        p_value_pce = np.zeros(pce_dist_ks.shape[0])
        #test_statistic_gpr = np.zeros(gpr_dist_ks.shape[0])
        #p_value_gpr = np.zeros(gpr_dist_ks.shape[0])


        for i in range(spce_dist_ks.shape[0]):
            test_statistic_spce[i], p_value_spce[i] = stats.ks_2samp(spce_dist_ks[i,:], samples_y_test_resized[:,i])
            test_statistic_pce[i], p_value_pce[i] = stats.ks_2samp(pce_dist_ks[i,:], samples_y_test_resized[:,i])
        #    test_statistic_gpr[i], p_value_gpr[i] = stats.ks_2samp(gpr_dist_ks[i,:], samples_y_test_resized[:,i])

        test_statistic_spce_mean[iter,s_i] = np.mean(test_statistic_spce)
        p_value_spce_mean[iter,s_i] = np.mean(p_value_spce)
        test_statistic_pce_mean[iter,s_i] = np.mean(test_statistic_pce)
        p_value_pce_mean[iter,s_i] = np.mean(p_value_pce)
        #test_statistic_gpr_mean[s_i] = np.mean(test_statistic_gpr)
        #p_value_gpr_mean[s_i] = np.mean(p_value_gpr)


    print('spce p value mean = ', p_value_spce_mean)
    print('pce p value mean = ', p_value_pce_mean)  
    #print('gpr p value mean = ', p_value_gpr_mean)
    print('spce test statistic mean = ', test_statistic_spce_mean)
    print('pce test statistic mean = ', test_statistic_pce_mean)
    #print('gpr test statistic mean = ', test_statistic_gpr_mean)

error_spce = np.mean(error_spce, axis=0)
error_pce = np.mean(error_pce, axis=0)
#error_gpr = np.mean(error_gpr, axis=0)
nrmse_spce = np.mean(nrmse_spce, axis=0)
nrmse_pce = np.mean(nrmse_pce, axis=0)
nrmse_spce_std = np.mean(nrmse_spce_std, axis=0)
nrmse_pce_std = np.mean(nrmse_pce_std, axis=0)
nrmse_spce_pred = np.mean(nrmse_spce_pred, axis=0)
nrmse_pce_pred = np.mean(nrmse_pce_pred, axis=0)
#nrmse_gpr = np.mean(nrmse_gpr, axis=0)
p_value_spce_mean = np.mean(p_value_spce_mean, axis=0)
p_value_pce_mean = np.mean(p_value_pce_mean, axis=0)
#p_value_gpr_mean = np.mean(p_value_gpr_mean, axis=0)
test_statistic_spce_mean = np.mean(test_statistic_spce_mean, axis=0)
test_statistic_pce_mean = np.mean(test_statistic_pce_mean, axis=0)
#test_statistic_gpr_mean = np.mean(test_statistic_gpr_mean, axis=0)


# np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/samples_total_{samples_total[0]}_{samples_total[-1]}.npy', samples_total)
# # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/error_spce_{samples_total[0]}_{samples_total[-1]}_q1.npy', error_spce)
# # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/error_pce_{samples_total[0]}_{samples_total[-1]}_q1.npy', error_pce)
# #np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/error_gpr.npy', error_gpr)
# # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/nrmse_spce_{samples_total[0]}_{samples_total[-1]}.npy', nrmse_spce)
# # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/nrmse_pce_{samples_total[0]}_{samples_total[-1]}.npy', nrmse_pce)
# np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/nrmse_std_spce_{samples_total[0]}_{samples_total[-1]}.npy', nrmse_spce_std)
# np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/nrmse_std_pce_{samples_total[0]}_{samples_total[-1]}.npy', nrmse_pce_std)
# #np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/nrmse_gpr.npy', nrmse_gpr)
# # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/p_value_spce_mean_{samples_total[0]}_{samples_total[-1]}_q1.npy', p_value_spce_mean)
# # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/p_value_pce_mean_{samples_total[0]}_{samples_total[-1]}_q1.npy', p_value_pce_mean)
# #np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/p_value_gpr_mean.npy', p_value_gpr_mean)
# # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/test_statistic_spce_mean_{samples_total[0]}_{samples_total[-1]}_q1.npy', test_statistic_spce_mean)
# # np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/test_statistic_pce_mean_{samples_total[0]}_{samples_total[-1]}_q1.npy', test_statistic_pce_mean)
# #np.save(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples/test_statistic_gpr_mean.npy', test_statistic_gpr_mean)


###### plot ####################################################################

plt.figure() 
plt.plot(samples_total, nrmse_spce_pred, label='SPCE')
plt.plot(samples_total, nrmse_pce_pred, label='PCE')
plt.xlabel(r'$N$')
plt.ylabel('nRMSE std')
plt.grid()
plt.yscale('log')
tikzplotlib.save(rf"tex_files\wind_data\samples\sdv_nrmse_sdv_test{samples_total[0]}_{samples_total[-1]}.tex")
plt.show()

print('x = ', samples_x_test[index_input_x,:size_y_test][x_index_plot])

plt.figure()
plt.hist(dist_spce_plot, bins=60, density=True, label='distribution SPCE')
plt.hist(y_samples_plot, bins=30, density=True, alpha=0.5, label='distribution reference')
plt.hist(dist_pce_x_plot, bins=60, density=True, alpha=0.5, label='distribution PCE')
#plt.hist(dist_gpr_plot, bins=60, density=True, alpha=0.5, label='distribution GPR')
plt.legend()
plt.xlabel('blade load')
plt.ylabel('pdf')

plt.figure()
plt.scatter(samples_plot, mean_pce_plot, color='g', label='predicted mean PCE')
plt.errorbar(samples_plot, mean_pce_plot, yerr=std_pce, fmt='o', capsize=5, color='g', label='std PCE')
plt.scatter(samples_plot, mean_spce_plot, label='predicted mean SPCE')
plt.errorbar(samples_plot, mean_spce_plot, yerr=std_spce_plot, fmt='o', capsize=5, label='std SPCE')
# plt.fill_between(samples_plot, mean_spce_plot - std_spce_plot, mean_spce_plot + std_spce_plot, alpha=0.5)
plt.scatter(samples_plot, mean_sim_plot, label='mean simulation')
plt.errorbar(samples_plot, mean_sim_plot, yerr=std_sim_plot, fmt='o', capsize=5, label='std simulation')
#plt.scatter(samples_plot, mean_gpr_plot, color='r', label='predicted mean GPR')
#plt.errorbar(samples_plot, mean_gpr_plot, yerr=std_gpr, fmt='o', capsize=5, color='r', label='std GPR')
# plt.fill_between(samples_plot, mean_sim_plot - std_sim_plot, mean_sim_plot + std_sim_plot, alpha=0.5)
plt.xlabel('Windspeed [m/s]') # Windspeed [m/s], Turbulence Intensity [-], Rho [kg/m3], Yaw Angle [°]
plt.ylabel('Blade Load [kN-m]')
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\predicted_mean_std.tex")
plt.legend()

fig5 = plt.figure()
ax5 = plt.subplot()
res = stats.ecdf(dist_spce_plot)
res2 = stats.ecdf(y_samples_plot)
res3 = stats.ecdf(dist_pce_x_plot)
#res4 = stats.ecdf(dist_gpr_plot)
res.cdf.plot(ax5, label='SPCE')
res2.cdf.plot(ax5, label='reference')
res3.cdf.plot(ax5, label='PCE')
#res4.cdf.plot(ax5, label='GPR')
ax5.legend()
ax5.grid()
ax5.set_xlabel('blade load')
ax5.set_ylabel('ECDF')

plt.show()