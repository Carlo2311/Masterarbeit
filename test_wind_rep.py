import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example_unimodal import AnalyticalExample
from spce import SPCE
from gaussian_process import Gaussian_Process
import time

samples = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_samples_total_10_170.npy')
error_spce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_error_spce_10_170.npy')
error_pce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_error_pce_10_170.npy')
nrmse_spce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_nrmse_spce_10_170.npy')
nrmse_pce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_nrmse_pce_10_170.npy')

plt.figure()
plt.plot(samples, error_spce, label='SPCE')
plt.plot(samples, error_pce, label='PCE')
plt.xlabel('Samples')
plt.ylabel('Error')
plt.grid()
plt.yscale('log')
plt.legend()
# tikzplotlib.save(rf"tex_files\wind_data\rep\rep_error_10_170.tex")

plt.figure()
plt.plot(samples, nrmse_spce, label='SPCE')
plt.plot(samples, nrmse_pce, label='PCE')
plt.xlabel('Samples')
plt.ylabel('NRMSE')
plt.grid()
plt.yscale('log')
plt.legend()
# tikzplotlib.save(rf"tex_files\wind_data\rep\rep_nrmse_10_170.tex")
plt.show()

print('nrmse spce = ', nrmse_spce)
print('nrmse pce = ', nrmse_pce)
print('error spce = ', error_spce)
print('error pce = ', error_pce)
 


p_value_spce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_p_value_spce_mean_200_250.npy')
p_value_pce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_p_value_pce_mean_200_250.npy')
test_statistic_spce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_test_statistic_spce_mean_200_250.npy')
test_statistic_pce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_test_statistic_pce_mean_200_250.npy')

print('p_value_spce = ', p_value_spce)
print('p value pce', p_value_pce)
print('Dks spce = ', test_statistic_spce)
print('DKs = ', test_statistic_pce)

plt.figure()
plt.plot(samples, p_value_spce, label='SPCE')
plt.plot(samples, p_value_pce, label='PCE')
plt.xlabel('Samples')
plt.ylabel('p-value')
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\rep\rep_p_value_10_100.tex")

plt.figure()
plt.plot(samples, test_statistic_spce, label='SPCE')
plt.plot(samples, test_statistic_pce, label='PCE')
plt.xlabel('Samples')
plt.ylabel('Test statistic')
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\rep\rep_test_statistic_10_100.tex")

print('hallo')