import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example_unimodal import AnalyticalExample
from spce import SPCE
from gaussian_process import Gaussian_Process
import time

samples = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_samples_total_10_100.npy')
p_value_spce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_p_value_spce_mean_10_100.npy')
p_value_pce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_p_value_pce_mean_10_100.npy')
test_statistic_spce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_test_statistic_spce_mean_10_100.npy')
test_statistic_pce = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/rep_test_statistic_pce_mean_10_100.npy')

plt.figure()
plt.plot(samples, p_value_spce, label='SPCE')
plt.plot(samples, p_value_pce, label='PCE')
plt.xlabel('Samples')
plt.ylabel('p-value')
plt.grid()
tikzplotlib.save(rf"tex_files\wind_data\rep\rep_p_value_10_100.tex")

plt.figure()
plt.plot(samples, test_statistic_spce, label='SPCE')
plt.plot(samples, test_statistic_pce, label='PCE')
plt.xlabel('Samples')
plt.ylabel('Test statistic')
plt.grid()
tikzplotlib.save(rf"tex_files\wind_data\rep\rep_test_statistic_10_100.tex")

print('hallo')