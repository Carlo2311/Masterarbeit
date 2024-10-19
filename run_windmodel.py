import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import chaospy as cp
from spce import SPCE
from scipy import stats
import math


##### input data simulation #################################################################
file_path_input = 'simulation_data/casematrix.csv'
df = pd.read_csv(file_path_input)

windspeed = df['Windspeed [m/s]'].to_numpy()
turbulence_intensity = df['TI [-]'].to_numpy()
rho = df['Rho [kg/m3]'].to_numpy()
yaw_angle = df['Yaw Angle [°]'].to_numpy()
seed1 = df['Seed 1'].to_numpy()
seed2 = df['Seed 2'].to_numpy()

##### initialize input variables ############################################################################
samples_tot = 900

if samples_tot <=300: 
    rep = 1
else:
    rep = int(samples_tot / 300)

dist_windspeed = cp.Beta(1.02, 3, 3, 25)
dist_turbulence_intensity = cp.Uniform(min(turbulence_intensity), max(turbulence_intensity)) 
dist_rho = cp.Uniform(min(rho), max(rho))
dist_yaw_angle = cp.Uniform(min(yaw_angle), max(yaw_angle))

dist_X = cp.J(dist_windspeed, dist_turbulence_intensity, dist_rho, dist_yaw_angle)

### Rosenblatt transformation ###
def rosenblatt_transformation(multi_distribution_standard, multi_distribution_physical, samples_physical):
    samples_standard = multi_distribution_standard.inv(multi_distribution_physical.fwd(samples_physical))
    return samples_standard 

dist_standard = cp.J(cp.Uniform(-1, 1), cp.Uniform(-1, 1), cp.Uniform(-1, 1), cp.Uniform(-1, 1))
samples_x_all_physical = np.array([windspeed, turbulence_intensity, rho, yaw_angle])
samples_x_all = rosenblatt_transformation(dist_standard, dist_X, samples_x_all_physical)

### training and test input data ###
index_samples = np.random.choice(300, size=int(samples_tot/rep), replace=False)
index_rep = np.random.choice(30, size=rep, replace=False)
samples_x = samples_x_all[:,:9000]
samples_x_resized = np.reshape(samples_x, (4, 30, 300))
samples_x = samples_x_resized[:,index_rep,:]
samples_x = samples_x[:,:,index_samples]
samples_x = np.reshape(samples_x, (4, samples_tot)) # training input data

samples_x_test = samples_x_all[:,9000:]
samples_x_test_resized = np.reshape(samples_x_test, (4, 30, 30)) # test input data

######## output data simulation #############################################################################

file_path_output = 'simulation_data/surrogate_data_v2.csv'
df = pd.read_csv(file_path_output)
df_sorted = df.sort_values(by='Case')

case = df_sorted['Case'].to_numpy()
root_myb_mean = df_sorted['RootMyb_[kN-m]_mean'].to_numpy()
# twh_tx_mean = df_sorted['TwHtALxt_[m/s^]_mean'].to_numpy() *1000
# twh_ty_mean = df_sorted['TwHtALyt_[m/s^]_mean'].to_numpy() 
# twh_tz_mean = df_sorted['TwHtALzt_[m/s^]_mean'].to_numpy() *1000
# root_myb_sdv = df_sorted['RootMyb_[kN-m]_sdv'].to_numpy()
# twh_tx_sdv = df_sorted['TwHtALxt_[m/s^]_sdv'].to_numpy() 
# twh_ty_sdv = df_sorted['TwHtALyt_[m/s^]_sdv'].to_numpy() 
# twh_tz_sdv = df_sorted['TwHtALzt_[m/s^]_sdv'].to_numpy() 
# root_myb_min = df_sorted['RootMyb_[kN-m]_min'].to_numpy()
# twh_tx_min = df_sorted['TwHtALxt_[m/s^]_min'].to_numpy()
# twh_ty_min = df_sorted['TwHtALyt_[m/s^]_min'].to_numpy()
# twh_tz_min = df_sorted['TwHtALzt_[m/s^]_min'].to_numpy()
# root_myb_max = df_sorted['RootMyb_[kN-m]_max'].to_numpy()
# twh_tx_max = df_sorted['TwHtALxt_[m/s^]_max'].to_numpy()
# twh_ty_max = df_sorted['TwHtALyt_[m/s^]_max'].to_numpy()
# twh_tz_max = df_sorted['TwHtALzt_[m/s^]_max'].to_numpy()

#### normalized output data
root_myb_mean_normalized = root_myb_mean / max(root_myb_mean)
samples_y_all = root_myb_mean_normalized[:9000] 
samples_y_resized = np.reshape(samples_y_all, (30, 300))

samples_y_resized = samples_y_resized[index_rep,:]
samples_y_resized = samples_y_resized[:,index_samples]
samples_y = samples_y_resized.reshape(samples_tot) # training output data

samples_y_test = root_myb_mean_normalized[9000:]
samples_y_test_resized = np.reshape(samples_y_test, (30, 30)) # test output data

##### initilize parameters for the surrogates ################################################################
n_samples = samples_x.shape[1]
p = 5
dist_Z = cp.Uniform(-1, 1)
dist_joint = cp.J(dist_standard, dist_Z)
N_q = 5 
q = 0.5

##### SPCE ###################################################################################################
spce = SPCE(n_samples, p, samples_y.T, samples_x, dist_joint, N_q, dist_Z, q)

### compute initial coefficients ###
surrogate_q0, poly_initial = spce.start_c(samples_x)
optimized_c = poly_initial.coefficients
polynomials = cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)

### compute range of sigma_noise ###
error_loo = spce.loo_error(samples_y, surrogate_q0, samples_x)
sigma_range = (0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo))
sigma_noise_range = np.linspace(np.log(sigma_range[0]) , np.log(sigma_range[-1]), 5)
sigma_noise_sorted = sorted(np.exp(sigma_noise_range), reverse=True)

### warm-up strategy ###
for sigma_noise_i in sigma_noise_sorted:
    optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise_i, optimized_c, polynomials, samples_x)

###### choose MLE or CV for estimating sigma with subsequently optimization of c 
### MLE ###
for i in range(7):
    sigma_noise = spce.optimize_sigma(samples_y, optimized_c, polynomials, samples_x, sigma_range)
    optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise, optimized_c, polynomials, samples_x)

### CV ###
# sigma_noise = spce.compute_optimal_sigma(optimized_c, polynomials, sigma_range) 
# optimized_c, message = spce.compute_optimal_c(samples_y, sigma_noise, optimized_c, polynomials, samples_x)

##### test surrogate #########################################################################################
dist_eps = cp.Normal(0, sigma_noise)
n_samples_test = 10000
samples_z_test = dist_Z.sample(n_samples_test)
samples_eps_test = dist_eps.sample(n_samples_test)

### generate distribution of the SPCE ###
dist_spce = spce.generate_dist_spce(samples_z_test, samples_eps_test, optimized_c, polynomials, samples_x_test)

##### PCE ######################
### input data for PCE ###
samples_pce_mean_y = np.mean(samples_y_resized, axis=0)
samples_pce_std_y = np.std(samples_y_resized, axis=0)

### generating surrogates ###
surrogate_pce_mean = spce.standard_pce(dist_X, samples_x, samples_pce_mean_y, samples_tot, q)
surrogate_pce_std = spce.standard_pce(dist_X, samples_x, samples_pce_std_y, samples_tot, q)

### generate distribution of the PCE ###
pce_mean_dist = surrogate_pce_mean(samples_x_test[0,:], samples_x_test[1,:], samples_x_test[2,:], samples_x_test[3,:])
pce_std_dist = np.abs(surrogate_pce_std(samples_x_test[0,:], samples_x_test[1,:], samples_x_test[2,:], samples_x_test[3,:]))
dist_pce = np.random.normal(pce_mean_dist[:, np.newaxis], pce_std_dist[:, np.newaxis], (samples_x_test.shape[1], n_samples_test))

##### GPR ######################
mean_gpr, std_gpr, dist_gpr = spce.generate_dist_gpr(samples_x_test, samples_pce_mean_y, samples_pce_std_y)

##### computation of the errors ##############################################################################
size_y_test = samples_y_test_resized.shape[1]

### error using the Wasserstein distance ###
error_spce = spce.compute_error(dist_spce[:size_y_test,:], samples_y_test_resized.T)
error_pce = spce.compute_error(dist_pce[:size_y_test,:], samples_y_test_resized.T)
error_gpr = spce.compute_error(dist_gpr[:size_y_test,:], samples_y_test_resized.T)

### error of the mean value estimation ###
mean_spce = np.mean(dist_spce, axis=1)
std_spce = np.std(dist_spce, axis=1)
mean_sim = np.mean(samples_y_test_resized, axis=0)
std_sim = np.std(samples_y_test_resized, axis=0)

nrmse_spce = (np.sqrt(np.mean((mean_spce[:size_y_test] - mean_sim)**2))) / (np.mean(mean_sim))
nrmse_pce = (np.sqrt(np.mean((pce_mean_dist[:size_y_test] - mean_sim)**2))) / (np.mean(mean_sim))
nrmse_gpr = (np.sqrt(np.mean((mean_gpr[:size_y_test] - mean_sim)**2))) / (np.mean(mean_sim))


###### KS test ################################################################################################
spce_dist_ks = dist_spce[:size_y_test,:]
pce_dist_ks = dist_pce[:size_y_test,:]
gpr_dist_ks = dist_gpr[:size_y_test,:]
test_statistic_spce = np.zeros(spce_dist_ks.shape[0])
test_statistic_pce = np.zeros(pce_dist_ks.shape[0])
test_statistic_gpr = np.zeros(gpr_dist_ks.shape[0])
p_value_spce = np.zeros(spce_dist_ks.shape[0])
p_value_pce = np.zeros(pce_dist_ks.shape[0])
p_value_gpr = np.zeros(gpr_dist_ks.shape[0])


for i in range(spce_dist_ks.shape[0]):
    test_statistic_spce[i], p_value_spce[i] = stats.ks_2samp(spce_dist_ks[i,:], samples_y_test_resized[:,i])
    test_statistic_pce[i], p_value_pce[i] = stats.ks_2samp(pce_dist_ks[i,:], samples_y_test_resized[:,i])
    test_statistic_gpr[i], p_value_gpr[i] = stats.ks_2samp(gpr_dist_ks[i,:], samples_y_test_resized[:,i])

test_statistic_spce_mean = np.mean(test_statistic_spce)
p_value_spce_mean = np.mean(p_value_spce)
test_statistic_pce_mean = np.mean(test_statistic_pce)
p_value_pce_mean = np.mean(p_value_pce)
test_statistic_gpr_mean = np.mean(test_statistic_gpr)
p_value_gpr_mean = np.mean(p_value_gpr)


###### plots ###########################################################################################

index_input_x = 0
sorted_indices = np.argsort(samples_x_test[index_input_x,:size_y_test])
x_index_plot = sorted_indices[13]

mean_spce_plot = mean_spce[:size_y_test]
mean_sim_plot = mean_sim
mean_pce_plot = pce_mean_dist[:size_y_test]
mean_gpr_plot = mean_gpr[:size_y_test]
samples_plot = samples_x_test[index_input_x,:size_y_test]
y_samples_plot = samples_y_test_resized[:,x_index_plot]
dist_spce_plot = dist_spce[x_index_plot,:]
dist_gpr_plot = dist_gpr[x_index_plot,:]
std_spce_plot = std_spce[:size_y_test]
std_sim_plot = std_sim 
std_pce = pce_std_dist[:size_y_test]
std_gpr = std_gpr[:size_y_test]
dist_pce_x_plot = dist_pce[x_index_plot,:]


plt.figure()
plt.hist(dist_spce_plot, bins=60, density=True, label='distribution SPCE')
plt.hist(y_samples_plot, bins=30, density=True, alpha=0.5, label='distribution reference')
plt.hist(dist_pce_x_plot, bins=60, density=True, alpha=0.5, label='distribution PCE')
plt.hist(dist_gpr_plot, bins=60, density=True, alpha=0.5, label='distribution GPR')
plt.legend()
plt.xlabel('blade load')
plt.ylabel('pdf')

plt.figure()
plt.scatter(samples_plot, mean_pce_plot, color='g', label='predicted mean PCE')
plt.errorbar(samples_plot, mean_pce_plot, yerr=std_pce, fmt='o', capsize=5, color='g', label='std PCE')
plt.scatter(samples_plot, mean_spce_plot, label='predicted mean SPCE')
plt.errorbar(samples_plot, mean_spce_plot, yerr=std_spce_plot, fmt='o', capsize=5, label='std SPCE')#
plt.scatter(samples_plot, mean_sim_plot, label='mean simulation')
plt.errorbar(samples_plot, mean_sim_plot, yerr=std_sim_plot, fmt='o', capsize=5, label='std simulation')
plt.scatter(samples_plot, mean_gpr_plot, color='r', label='predicted mean GPR')
plt.errorbar(samples_plot, mean_gpr_plot, yerr=std_gpr, fmt='o', capsize=5, color='r', label='std GPR')
plt.xlabel('Windspeed [m/s]') # Windspeed [m/s], Turbulence Intensity [-], Rho [kg/m3], Yaw Angle [°]
plt.ylabel('Blade Load [kN-m]')
plt.grid()
plt.legend()

fig5 = plt.figure()
ax5 = plt.subplot()
res = stats.ecdf(dist_spce_plot)
res2 = stats.ecdf(y_samples_plot)
res3 = stats.ecdf(dist_pce_x_plot)
res4 = stats.ecdf(dist_gpr_plot)
res.cdf.plot(ax5, label='SPCE')
res2.cdf.plot(ax5, label='reference')
res3.cdf.plot(ax5, label='PCE')
res4.cdf.plot(ax5, label='GPR')
ax5.legend()
ax5.grid()
ax5.set_xlabel('blade load')
ax5.set_ylabel('ECDF')

plt.show()