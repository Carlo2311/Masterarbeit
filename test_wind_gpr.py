import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example_unimodal import AnalyticalExample
from spce import SPCE
from gaussian_process import Gaussian_Process
import time

n_samples = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/gpr/samples_total_gpr_10_1500.npy')
error_gpr = np.load(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/gpr/error_gpr_{n_samples[0]}_{n_samples[-1]}.npy')
nrmse_gpr = np.load(f'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/gpr/nrmse_gpr_{n_samples[0]}_{n_samples[-1]}.npy')

plt.figure()
plt.plot(n_samples, nrmse_gpr, label='GPR')
plt.xlabel('Samples')
plt.ylabel('nRMSE')
plt.grid()
plt.yscale('log')
tikzplotlib.save(rf"tex_files\wind_data\gpr\nrmse_gpr_{n_samples[0]}_{n_samples[-1]}.tex")

plt.figure()
plt.plot(n_samples, error_gpr, label='GPR')
plt.xlabel('Samples')
plt.ylabel('Error')
plt.grid()
plt.yscale('log')
tikzplotlib.save(rf"tex_files\wind_data\gpr\error_gpr_{n_samples[0]}_{n_samples[-1]}.tex")
plt.show()