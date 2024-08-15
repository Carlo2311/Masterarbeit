import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import chaospy as cp
from spce import SPCE
from scipy import stats

####### input data simulation #################################################################
file_path_input = 'simulation_data/casematrix.csv'
df = pd.read_csv(file_path_input)

windspeed = df['Windspeed [m/s]'].to_numpy()
turbulence_intensity = df['TI [-]'].to_numpy()
rho = df['Rho [kg/m3]'].to_numpy()
yaw_angle = df['Yaw Angle [°]'].to_numpy()
seed1 = df['Seed 1'].to_numpy()
seed2 = df['Seed 2'].to_numpy()

windspeed_normalized = windspeed / max(windspeed)
turbulence_intensity_normalized = turbulence_intensity / max(turbulence_intensity)
rho_normalized = rho / max(rho)
yaw_angle_normalized = yaw_angle / max(yaw_angle)

# plt.figure()   
# plt.scatter(windspeed_normalized, turbulence_intensity_normalized)
# plt.show()

# samples_x = np.array([windspeed[:9000], turbulence_intensity[:9000], rho[:9000], yaw_angle[:9000]])
samples_x = np.array([windspeed_normalized[:9000], turbulence_intensity_normalized[:9000], rho_normalized[:9000], yaw_angle_normalized[:9000]])
samples_x_resized = np.reshape(samples_x, (4, 30, 300))
# samples_x = np.array([windspeed_normalized[300:600], turbulence_intensity_normalized[300:600], rho_normalized[300:600], yaw_angle_normalized[300:600]])
# samples_x_resized = np.reshape(samples_x, (4, 300))

# samples_x_test = np.array([windspeed[9000:], turbulence_intensity[9000:], rho[9000:], yaw_angle[9000:]])
samples_x_test = np.array([windspeed_normalized[9000:], turbulence_intensity_normalized[9000:], rho_normalized[9000:], yaw_angle_normalized[9000:]])
samples_x_test_resized = np.reshape(samples_x_test, (4, 30, 30))
# samples_x_test = np.array([windspeed_normalized[9000:9030], turbulence_intensity_normalized[9000:9030], rho_normalized[9000:9030], yaw_angle_normalized[9000:9030]])
# samples_x_test_resized = np.reshape(samples_x_test, (4, 30))

# test with simulation data
# samples_x_test = np.array([windspeed[:9000], turbulence_intensity[:9000], rho[:9000], yaw_angle[:9000]])
# samples_x_test = np.array([windspeed_normalized[:9000], turbulence_intensity_normalized[:9000], rho_normalized[:9000], yaw_angle_normalized[:9000]])
# samples_x_test_resized = np.reshape(samples_x_test, (4, 30, 300))


dist_windspeed = cp.Beta(1.02, 3, 3, 25)
# dist_turbulence_intensity = cp.Uniform(min(turbulence_intensity), max(turbulence_intensity)) # noch ändern
# dist_rho = cp.Uniform(min(rho), max(rho))
# dist_yaw_angle = cp.Uniform(min(yaw_angle), max(yaw_angle))
# normalized
dist_turbulence_intensity = cp.Uniform(min(turbulence_intensity_normalized), max(turbulence_intensity_normalized)) # noch ändern
dist_rho = cp.Uniform(min(rho_normalized), max(rho_normalized))
dist_yaw_angle = cp.Uniform(min(yaw_angle_normalized), max(yaw_angle_normalized))

dist_X = cp.J(dist_windspeed, dist_turbulence_intensity, dist_rho, dist_yaw_angle)


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
twh_tx_mean_normalized = twh_tx_mean / max(twh_tx_mean) 
twh_ty_mean_normalized = twh_ty_mean / max(twh_ty_mean) 
twh_tz_mean_normalized = twh_tz_mean / max(twh_tz_mean) 
twh_tx_sdv_normalized = twh_tx_sdv / max(twh_tx_sdv) 
twh_ty_sdv_normalized = twh_ty_sdv / max(twh_ty_sdv) 
twh_tz_sdv_normalized = twh_tz_sdv / max(twh_tz_sdv) 

####standard
# samples_y = root_myb_mean[:9000]
# samples_y_test = root_myb_mean[9000:]

#### normalized
samples_y = root_myb_mean_normalized[:9000] 
samples_y_resized = np.reshape(samples_y, (30, 300))

samples_y_test = root_myb_mean_normalized[9000:]
samples_y_test_resized = np.reshape(samples_y_test, (30, 30))

# samples_y = root_myb_mean_normalized[300:600] 
# samples_y_resized = np.reshape(samples_y, (1, 300))

# samples_y_test = root_myb_mean_normalized[9000:9030]
# samples_y_test_resized = np.reshape(samples_y_test, (1, 30))

#### test with simulation data
# samples_y_test = root_myb_mean[:9000]
# samples_y_test = root_myb_mean_normalized[:9000]
# samples_y_test_resized = np.reshape(samples_y_test, (30, 300))

plt.figure()
plt.scatter(samples_x[0,:], samples_y, s=1)
plt.xlabel('windspeed [m/s]')
plt.ylabel('tower acceleration [mm/s^2]')
plt.grid()
plt.draw()

####### test data SCADA #######################################################

file_path_output = 'simulation_data/SCADA_Data_2017-2022_with_TI_rho_wsp_binned.csv'
columns_to_read = ['Wind speed (avg.) [m/s]', 'Load on blade 1 (avg.) []', 'Load on blade 2 (avg.) []', 'Load on blade 3 (avg.) []', 'is_Abnormal', 'turbine','TI_mes','wsp_binned','Air Density [kg/m3]', 'Model', 'Relative wind direction (wind shear) (avg.) [°]']
df = pd.read_csv(file_path_output, usecols=columns_to_read)

filtered_df = df[(df['turbine'] == 1) & (df['is_Abnormal'] == False)]

windspeed_scada = filtered_df['Wind speed (avg.) [m/s]'].to_numpy()
turbulence_intensity_scada = filtered_df['TI_mes'].to_numpy()
rho_scada = filtered_df['Air Density [kg/m3]'].to_numpy()
yaw_angle_scada = filtered_df['Relative wind direction (wind shear) (avg.) [°]'].to_numpy()
load_b1_mean = -filtered_df['Load on blade 1 (avg.) []'].to_numpy() / 1000
load_b2_mean = -filtered_df['Load on blade 2 (avg.) []'].to_numpy() / 1000
load_b3_mean = -filtered_df['Load on blade 3 (avg.) []'].to_numpy() / 1000

indices_random = np.random.choice(len(windspeed_scada), 30, replace=False)

windspeed_normalized = windspeed_scada[indices_random] / max(windspeed)
turbulence_intensity_normalized = turbulence_intensity_scada[indices_random] / max(turbulence_intensity)
rho_normalized = rho_scada[indices_random] / max(rho)
yaw_angle_normalized = yaw_angle_scada[indices_random] / max(yaw_angle)

load_mean = (load_b1_mean + load_b2_mean + load_b3_mean) / 3
load_normalized = load_mean[indices_random] / max(root_myb_mean)

# samples_x_test = np.array([windspeed_normalized, turbulence_intensity_normalized, rho_normalized, yaw_angle_normalized])
# samples_y_test = load_normalized

# plt.figure()
# plt.scatter(samples_x_test[0,:], samples_y_test)
# plt.scatter(samples_x[0,:], samples_y)

# plt.figure()
# plt.scatter(samples_x_test[1,:], samples_y_test)

# plt.figure()
# plt.scatter(samples_x_test[2,:], samples_y_test)

# plt.figure()
# plt.scatter(samples_x_test[3,:], samples_y_test)

# plt.figure()
# plt.scatter(windspeed_scada, turbulence_intensity_scada)

# plt.figure()
# plt.scatter(windspeed_scada, rho_scada)

# plt.figure()
# plt.scatter(windspeed_scada, yaw_angle_scada)


############## SPCE #################################################################

n_samples = windspeed.shape[0]
p = 3
# dist_Z = cp.Normal(0, 1)
dist_Z = cp.Uniform(-1, 1)
dist_joint = cp.J(dist_X, dist_Z)
N_q = 10
q = 0.75

spce = SPCE(n_samples, p, samples_y.T, samples_x, dist_joint, N_q, dist_Z, q)
poly, z_j = spce.get_params()
input_x_start = [samples_x[0,:], samples_x[1,:], samples_x[2,:], samples_x[3,:]]
input_x = [samples_x[0,:, np.newaxis], samples_x[1,:, np.newaxis], samples_x[2,:, np.newaxis], samples_x[3,:, np.newaxis]]

surrogate_q0, poly_initial = spce.start_c(input_x_start)
polynomials = cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)

#### PCE #####
# input sim data
samples_pce_x = [samples_x[0,:300], samples_x[1,:300], samples_x[2,:300], samples_x[3,:300]]
samples_pce_mean_y = np.mean(samples_y_resized, axis=0)
samples_pce_std_y = np.std(samples_y_resized, axis=0)

# surrogate
surrogate_pce_mean = spce.standard_pce(dist_X, samples_pce_x, samples_pce_mean_y, q)
surrogate_pce_std = spce.standard_pce(dist_X, samples_pce_x, samples_pce_std_y, q)
##############

# error_loo = spce.loo_error(samples_y, surrogate_q0, input_x_start)

# optimized_c = poly_initial.coefficients

# # sigma_range = (0.1 * np.sqrt(error_loo), 1 * np.sqrt(error_loo))
# sigma_range = (0.005 , 0.1)
# print(sigma_range)
# # # spce.plot_sigma(samples_x, samples_y, sigma_range, c_initial)
# sigma_noise_range = np.linspace(np.log(0.005) , np.log(0.1), 5)
# sigma_noise_sorted = sorted(np.exp(sigma_noise_range), reverse=True)

# for sigma_noise_i in sigma_noise_sorted:
#     print('sigma_i = ', sigma_noise_i)
#     optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise_i, optimized_c, polynomials, input_x)
#     print(optimized_c)
#     print(message)


# sigma_noise = spce.compute_optimal_sigma(optimized_c, polynomials, sigma_range) # cross validation

# optimized_c, message = spce.compute_optimal_c(samples_x, samples_y, sigma_noise, optimized_c, polynomials, input_x)

# np.save('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/c_q075_norm_p3_300_test.npy', optimized_c)
# np.save('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/sigma_q075_norm_p3_300_test.npy', sigma_noise)

optimized_c = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/c_q075_norm_p3_300_test.npy')
sigma_noise = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/sigma_q075_norm_p3_300_test.npy')
print('sigma = ', sigma_noise)
print('c = ', optimized_c)


# terms = optimized_c * cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)
# poly_opt = cp.sum(terms, axis=0)
# pce = poly_opt(samples_x[0,:, np.newaxis], samples_x[1,:, np.newaxis], samples_x[2,:, np.newaxis], samples_x[3,:, np.newaxis], z_j[4])
# plt.figure()
# plt.scatter(samples_x[1,:], samples_y, label='reference')
# plt.scatter(samples_x[1,:], pce, label='poly')
# plt.legend()
# plt.show()

# terms = optimized_c * cp.prod(poly_initial.indeterminants**poly_initial.exponents, axis=-1)
# poly_opt = cp.sum(terms, axis=0)
# pce = poly_opt(samples_x[0,:, np.newaxis], samples_x[1,:, np.newaxis], z_j)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(samples_x[0,:], samples_x[1,:], samples_y, c=samples_y, cmap='viridis', marker='o', label='reference')
# sc2 = ax.scatter(samples_x[0,:], samples_x[1,:], pce[:,4], c=pce[:,4], cmap='plasma', marker='^', label='SPCE')
# ax.set_xlabel('Wind Speed (m/s)')
# ax.set_ylabel('Turbulence Intensity')
# ax.set_zlabel('Blade Load')
# cb_actual = plt.colorbar(sc, ax=ax, pad=0.1)
# cb_actual.set_label('Actual Blade Load')
# cb_predicted = plt.colorbar(sc2, ax=ax, pad=0.2)
# cb_predicted.set_label('Predicted Blade Load')
# plt.show()


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

size_y_test = samples_y_test_resized.shape[1]
error = spce.compute_error(dist_spce[:size_y_test,:], samples_y_test_resized.T)
error_pce = spce.compute_error(dist_pce[:size_y_test,:], samples_y_test_resized.T)
print('spce wasserstein distance = ', error)
print('pce wasserstein distance = ', error_pce)

mean_spce = np.mean(dist_spce, axis=1)
std_spce = np.std(dist_spce, axis=1)

####### test simulation data #############################################################
mean_sim = np.mean(samples_y_test_resized, axis=0)
std_sim = np.std(samples_y_test_resized, axis=0)

nrmse_spce = (np.sqrt(np.mean((mean_spce[:size_y_test] - mean_sim)**2))) / (np.mean(mean_sim))
nrmse_pce = (np.sqrt(np.mean((pce_mean_dist[:size_y_test] - mean_sim)**2))) / (np.mean(mean_sim))
print('spce nrmse = ', nrmse_spce)
print('pce nrmse = ', nrmse_pce)

index_input_x = 0
sorted_indices = np.argsort(samples_x_test[index_input_x,:size_y_test])
x_index_plot = sorted_indices[10]

mean_spce_plot = mean_spce[:size_y_test]#[sorted_indices]
mean_sim_plot = mean_sim#[sorted_indices]
mean_pce_plot = pce_mean_dist[:size_y_test]#[sorted_indices]
samples_plot = samples_x_test[index_input_x,:size_y_test]#[sorted_indices]
y_samples_plot = samples_y_test_resized[:,x_index_plot]
dist_spce_plot = dist_spce[x_index_plot,:]
std_spce_plot = std_spce[:size_y_test]#[sorted_indices]
std_sim_plot = std_sim#[sorted_indices] 
std_pce = pce_std_dist[:size_y_test]#[sorted_indices]
dist_pce_x_plot = dist_pce[x_index_plot,:]

### test scada data #####################################################################
# mean_sim = samples_y_test
# std_sim = np.zeros(mean_sim.shape)

# nrmse_spce = (np.sqrt(np.mean((mean_spce - mean_sim)**2))) / (np.mean(mean_sim))
# nrmse_pce = (np.sqrt(np.mean((pce_mean_dist - mean_sim)**2))) / (np.mean(mean_sim))
# print('nrmse spce = ', nrmse_spce)
# print('nrmse pce = ', nrmse_pce)

# index_input_x = 0
# sorted_indices = np.argsort(samples_x_test[index_input_x,:size_y_test])
# x_index_plot = sorted_indices[17]

# mean_spce_plot = mean_spce#[sorted_indices]
# mean_sim_plot = mean_sim#[sorted_indices]
# mean_pce_plot = pce_mean_dist#[sorted_indices]
# samples_plot = samples_x_test[index_input_x,:]#[sorted_indices]
# # y_samples_plot = samples_y_test[x_index_plot]
# dist_spce_plot = dist_spce[x_index_plot,:]
# std_spce_plot = std_spce#[sorted_indices]
# std_sim_plot = std_sim#[sorted_indices] 
# std_pce = pce_std_dist#[sorted_indices]
# dist_pce_x_plot = dist_pce[x_index_plot,:]


###### KS test ####################################################################
spce_dist_ks = dist_spce[:size_y_test,:]
pce_dist_ks = dist_pce[:size_y_test,:]
test_statistic_spce = np.zeros(spce_dist_ks.shape[0])
p_value_spce = np.zeros(spce_dist_ks.shape[0])
test_statistic_pce = np.zeros(pce_dist_ks.shape[0])
p_value_pce = np.zeros(pce_dist_ks.shape[0])

for i in range(spce_dist_ks.shape[0]):
    test_statistic_spce[i], p_value_spce[i] = stats.ks_2samp(spce_dist_ks[i,:], samples_y_test_resized[:,i])
    test_statistic_pce[i], p_value_pce[i] = stats.ks_2samp(pce_dist_ks[i,:], samples_y_test_resized[:,i])

test_statistic_spce_mean = np.mean(test_statistic_spce)
p_value_spce_mean = np.mean(p_value_spce)
test_statistic_pce_mean = np.mean(test_statistic_pce)
p_value_pce_mean = np.mean(p_value_pce)

print('spce p value mean = ', p_value_spce_mean)
print('pce p value mean = ', p_value_pce_mean)  
print('spce test statistic mean = ', test_statistic_spce_mean)
print('pce test statistic mean = ', test_statistic_pce_mean)


###### plot ####################################################################

print('x = ', samples_x_test[index_input_x,:size_y_test][x_index_plot])

plt.figure()
plt.hist(dist_spce_plot, bins=60, density=True, label='distribution SPCE')
plt.hist(y_samples_plot, bins=15, density=True, alpha=0.5, label='distribution reference')
plt.hist(dist_pce_x_plot, bins=60, density=True, alpha=0.5, label='distribution PCE')
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
# plt.fill_between(samples_plot, mean_sim_plot - std_sim_plot, mean_sim_plot + std_sim_plot, alpha=0.5)
plt.xlabel('Windspeed [m/s]')
plt.ylabel('Blade Load [kN-m]')
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\predicted_mean_std.tex")
plt.legend()

fig5 = plt.figure()
ax5 = plt.subplot()
res = stats.ecdf(dist_spce_plot)
res2 = stats.ecdf(y_samples_plot)
res3 = stats.ecdf(dist_pce_x_plot)
res.cdf.plot(ax5, label='SPCE')
res2.cdf.plot(ax5, label='reference')
res3.cdf.plot(ax5, label='PCE')
ax5.legend()
ax5.grid()
ax5.set_xlabel('blade load')
ax5.set_ylabel('ECDF')

plt.show()



# from scipy.stats import gaussian_kde
# # print(samples_x_test[0])
# kde = gaussian_kde(dist_spce[0,:])
# x_values_spce = np.linspace(min(dist_spce[0,:]), max(dist_spce[0,:]), 1000) 
# dist_spce_pdf_values = kde(x_values_spce)

# plt.figure()            
# plt.plot(x_values_spce, dist_spce_pdf_values, label='SPCE')
# plt.hist(samples_y_test[x,:], bins=bin_edges, density=True, alpha=0.5, label='distribution reference')
# plt.hist(dist_spce[x, :], bins=bin_edges, density=True, alpha=0.5, label='distribution SPCE')
# plt.hist(samples_gpr, bins=bin_edges, density=True, alpha=0.5, label='distribution GPR')
# plt.xlabel('y')
# plt.ylabel('pdf')
# plt.show()




# # calculate Y of analytical model 
# pdf_test, mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, mean_12_test, sigma_12_test = example.calculate_pdf(samples_x_test)
# samples_y_test = example.create_data_points(mean_1_test, mean_2_test, sigma_1_test, sigma_2_test, n_samples_test, samples_x_test)

# mean_prediction_gpr, std_prediction_gpr, dist_gpr = spce.generate_dist_gpr(samples_x_test, samples_y_test, mean_12_test, sigma_12_test)
# spce.plot_distribution(dist_spce, y, pdf_test, samples_x_test, samples_y_test, mean_prediction_gpr, std_prediction_gpr)

# error_n[i, n_i], error_gpr[i, n_i] = spce.compute_error(dist_spce, samples_y_test, dist_gpr)
# print('error spce = ', error_n[i, n_i])
# print('error spce = ', error_gpr[i, n_i])



# print(error_n, error_gpr)
# error_spce = np.mean(error_n, axis=0)
# error_gpr_mean = np.mean(error_gpr, axis=0)
# print('error mean spce =', error_spce)   
# print('error mean gpr =', error_gpr_mean)   

# # plt.figure()
# # plt.plot(n_samples_all, error_spce, label='SPCE')
# # plt.plot(n_samples_all, error_gpr_mean, label='GPR')
# # plt.xlabel(f'N')
# # plt.ylabel('error')
# # plt.grid()
# # plt.yscale('log')
# # tikzplotlib.save(rf"tex_files\bimodal\bimodal_error_gpr_2.tex")

# plt.show()