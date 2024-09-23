import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example_unimodal import AnalyticalExample
from spce import SPCE
from gaussian_process import Gaussian_Process
import time

n_samples = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/samples_total_rep.npy')
spce_wd = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/error_spce_rep.npy')
pce_wd = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/error_pce_rep.npy')
spce_rmse = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/nrmse_spce_rep.npy')
pce_rmse = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/nrmse_pce_rep.npy')

plt.figure()
plt.plot(n_samples, spce_wd, label='SPCE')
plt.plot(n_samples, pce_wd, label='PCE')
plt.xlabel(r'$N$')
plt.ylabel(r'$\varepsilon$')
plt.grid()
plt.yscale('log')
# plt.legend()
# tikzplotlib.save(rf"tex_files\wind_data\rep_samples_error.tex")

plt.figure()
plt.plot(n_samples, spce_rmse, label='SPCE')
plt.plot(n_samples, pce_rmse, label='PCE')
plt.xlabel(r'$N$')
plt.ylabel('nRMSE')
plt.grid()
plt.yscale('log')
# plt.legend()
# tikzplotlib.save(rf"tex_files\wind_data\rep_samples_nrmse.tex")

spce_kstest = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/test_statistic_spce_mean_rep.npy')
pce_kstest = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/test_statistic_pce_mean_rep.npy')
spce_pvalue = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/p_value_spce_mean_rep.npy')
pce_pvalue = np.load('C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_wind/load/convergence_samples/p_value_pce_mean_rep.npy')

plt.figure()
plt.plot(n_samples, spce_kstest, label='SPCE')
plt.plot(n_samples, pce_kstest, label='PCE')
plt.xlabel(r'$N$')
plt.ylabel('Test statistic')
plt.grid()
# plt.legend()
# tikzplotlib.save(rf"tex_files\wind_data\rep_samples_test_statistic.tex")

plt.figure()
plt.plot(n_samples, spce_pvalue, label='SPCE')
plt.plot(n_samples, pce_pvalue, label='PCE')
plt.xlabel(r'$N$')
plt.ylabel('p-value')
plt.grid()
# plt.legend()
# tikzplotlib.save(rf"tex_files\wind_data\rep_samples_p_value.tex")

plt.show()




samples_total = [300, 600, 900, 1200, 1500, 1800, 2100, 3000, 4200, 5400, 6900, 9000]

spce_error_wd = [0.06328152, 0.02459415, 0.01805697, 0.01860088, 0.01883854, 0.01861698, 0.01770017, 0.02086027, 0.01816087, 0.01773228, 0.01994122, 0.01773067]
pce_error_wd = [0.02204589, 0.02038833, 0.02011091, 0.02001836, 0.02007136, 0.01998168, 0.01993576, 0.01999994, 0.01987629, 0.01987411, 0.01990309, 0.01988431]
# gpr_error_wd = [0.00715253, 0.00550755, 0.00598421, 0.00559952, 0.004525, 0.00436205, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
spce_nrmse = [0.06054964, 0.04628836, 0.04412346, 0.0448257,  0.04429637, 0.04502493, 0.04382958, 0.0444615,  0.0441404,  0.0441606,  0.04392781, 0.04419561]
pce_nrmse = [0.06297041, 0.06261878, 0.0624478,  0.0623497,  0.06242741, 0.06232251, 0.06225858, 0.06237868, 0.06212454, 0.06211788, 0.06211449, 0.06213608]
# gpr_nrmse = [0.02554459, 0.02274732, 0.02566219, 0.0239631,  0.01840164, 0.01723574, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

plt.figure()
plt.plot(samples_total, spce_error_wd, label='SPCE')
plt.plot(samples_total, pce_error_wd, label='PCE')
plt.xlabel('Number of samples')
plt.ylabel(r'$\varepsilon$')
plt.yscale('log')
# plt.legend()
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\samples_error_wd.tex")

plt.figure()
plt.plot(samples_total, spce_nrmse, label='SPCE')
plt.plot(samples_total, pce_nrmse, label='PCE')
plt.xlabel('Number of samples')
plt.ylabel('nRMSE')
plt.yscale('log')
# plt.legend()
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\samples_nrmse.tex")
# plt.show()

samples_total = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400, 5700, 6000, 6300, 6600, 6900, 7200, 7500]#, 7800, 8100, 8400, 8700, 9000]

p_300_spce = np.mean([0.04257211, 0.03124912, 0.02702674, 0.0103032])
p_900_spce = np.mean([0.02806748, 0.01388193, 0.03561353, 0.02913098])
p_2100_spce = np.mean([0.04242161, 0.01374983, 0.01552263, 0.04058024])
p_300_pce = np.mean([5.99253107e-10, 5.99253107e-10, 5.99253107e-10, 5.99253107e-10])
p_900_pce = np.mean([1.95031781e-02, 1.93709937e-02, 1.92492157e-02, 1.85331660e-02])
p_2100_pce = np.mean([2.96162123e-02, 2.95267658e-02, 2.85627424e-02, 2.93448702e-02])
p_3600_spce = np.mean([0.02818263, 0.02963844, 0.03092491, 0.02843291])
p_3600_pce = np.mean([2.10781100e-02, 2.32854300e-02, 2.01446800e-02, 1.83239500e-02])
p_5100_spce = np.mean([2.39860638e-02, 2.04647778e-02, 2.17733963e-02])
p_5100_pce = np.mean([0.01986811, 0.02384927, 0.02341367, 0.02182201])
p_7500_spce = np.mean([ 1.75039753e-02, 3.13160002e-02, 1.72656456e-02, 3.99885731e-02])
p_7500_pce = np.mean([0.02150062, 0.0209191, 0.01940509, 0.02270611])
p_1500_spce = np.mean([0.02942154, 0.01026895, 0.0329596, 0.03090015, 0.0127082])
p_2400_spce = np.mean([0.01738728, 0.02822492, 0.04056994, 0.03107853, 0.04086372])
p_4200_spce = np.mean([0.0456466, 0.0308839, 0.04036308, 0.04957381, 0.05307872])
p_6000_spce = np.mean([0.03396937, 0.03791093, 0.0234972, 0.05210322, 0.03969887])
p_1500_pce = np.mean([0.03279778, 0.03089119, 0.03378019, 0.03250602, 0.03352797])
p_2400_pce = np.mean([0.02024942, 0.02067086, 0.0153113, 0.01895367, 0.01625704])
p_4200_pce = np.mean([0.0193356, 0.0168729, 0.0207942, 0.01995187, 0.01935727])
p_6000_pce = np.mean([0.02089444, 0.02080059, 0.02098207, 0.0193495, 0.01824872])

print('spce: ', p_300_spce, p_900_spce, p_1500_spce, p_2100_spce, p_2400_spce, p_3600_spce, p_4200_spce, p_5100_spce, p_6000_spce, p_7500_spce)
print('pce: ', p_300_pce, p_900_pce, p_1500_pce, p_2100_pce, p_2400_pce, p_3600_pce, p_4200_pce, p_5100_pce, p_6000_pce, p_7500_pce)

samples_total_p = [300, 900, 1500, 2100, 2400, 3600, 4200, 5100, 6000, 7500]
spce_p_value_mean = [p_300_spce, p_900_spce, p_1500_spce, p_2100_spce, p_2400_spce, p_3600_spce, p_4200_spce, p_5100_spce, p_6000_spce, p_7500_spce]
pce_p_value_mean = [p_300_pce, p_900_pce, p_1500_pce, p_2100_pce, p_2400_pce, p_3600_pce, p_4200_pce, p_5100_pce, p_6000_pce, p_7500_pce]

plt.figure()
plt.plot(samples_total_p, spce_p_value_mean, label='SPCE')
plt.plot(samples_total_p, pce_p_value_mean, label='PCE')
plt.xlabel('Number of samples')
plt.ylabel('p-value')
plt.legend()
# plt.yscale('log')
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\samples_p_value.tex")
plt.show()

spce_p_value_mean = [0.01329773, 0.01153447, 0.04066753, 0.01368928, 0.04392677, 0.02158209, 0.00601303, 0.0081864, 0.04550582, 0.03202478, 0.04312251, 0.04400251, 0.02936292, 0.02851675, 0.04892072, 0.04345737, 0.02413347, 0.02497953, 0.02763573, 0.02272663, 0.04799637, 0.04813078, 0.04772049, 0.04692122, 0.02656592]
pce_p_value_mean = [4.75212460e-11, 6.83074483e-03, 1.80184076e-02, 2.51318320e-02, 3.14817822e-02, 2.23097139e-02, 2.10110990e-02, 1.84235736e-02, 2.40052591e-02, 2.13562294e-02, 1.54957264e-02, 1.96822865e-02, 2.34234863e-02, 2.24166388e-02, 2.38971892e-02, 2.32102718e-02, 2.20969862e-02, 2.20498052e-02, 2.01204031e-02, 1.73343952e-02, 2.03094877e-02, 1.88668579e-02, 1.93512581e-02, 2.22554958e-02, 1.99598641e-02]
gpr_p_value_mean = [0.0632232, 0.07592228, 0.06795022, 0.0689609, 0.06335856, 0.06326439, 0.09728622, 0.09560136, 0.08175088, 0.06098912, 0.06020056, 0.07970203, 0.04914874, 0.05232081, 0.06628447, 0.05635221, 0.07097275, 0.07198363, 0.06949462, 0.08184301, 0.08029191, 0.07588082, 0.0890883, 0.08520161, 0.08586609]
spce_test_statistic_mean = [0.55966222, 0.53691, 0.56128222, 0.55486667, 0.56112556, 0.55761222, 0.54585667, 0.55854111, 0.55584111, 0.55864556, 0.55987556, 0.55981778, 0.55789444, 0.55721222, 0.55491778, 0.55788222, 0.55954667, 0.55796333, 0.55826111, 0.55996, 0.55623667, 0.55737111, 0.55710222, 0.55502, 0.55954667]
pce_test_statistic_mean = [0.92333333, 0.80709667, 0.78866667, 0.77632778, 0.76991667, 0.76870556, 0.76633778, 0.76710333, 0.76535667, 0.76573444, 0.76774556, 0.76416111, 0.76105444, 0.76026111, 0.75908222, 0.75846444, 0.75904444, 0.75664222, 0.75772444, 0.75936778, 0.75817667, 0.75893556, 0.75747556, 0.75719111, 0.75699222]
gpr_test_statistic_mean = [0.48165111, 0.46118333, 0.44898111, 0.42941111, 0.41476, 0.40869444, 0.39527667, 0.40224444, 0.40949556, 0.41111778, 0.41237889, 0.41135333, 0.41746, 0.41945444, 0.41765667, 0.42350222, 0.41686444, 0.41351889, 0.41652778, 0.41100444, 0.41735, 0.41879667, 0.41762111, 0.41564444, 0.41288889]

plt.figure()
plt.plot(samples_total, spce_p_value_mean, label='SPCE')
plt.plot(samples_total, pce_p_value_mean, label='PCE')
plt.plot(samples_total, gpr_p_value_mean, label='GPR')
plt.xlabel('Number of samples')
plt.ylabel('p-value')
# plt.legend()
# plt.yscale('log')
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\samples_p_value.tex")

plt.figure()
plt.plot(samples_total, spce_test_statistic_mean, label='SPCE')
plt.plot(samples_total, pce_test_statistic_mean, label='PCE')
plt.plot(samples_total, gpr_test_statistic_mean, label='GPR')
plt.xlabel('Number of samples')
plt.ylabel('Test statistic')
# plt.legend()
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\samples_test_statistic.tex")
plt.show()




p = [1,2,3,4,5,6,7,8]
p_error_wd = [1.1691700933331, 0.30553940452050227, 0.04853854780289139, 0.054581921392655657, 0.022077139652204668, 0.015748459968552862, 0.017518640037474803, 0.0077304474347982014]
p_nrmse_p = [0.3320667315335488, 0.17634548551060678, 0.061922306630671026, 0.06667076695901708, 0.04413009618674215, 0.03318245939560011, 0.04210942030753135, 0.028934929991208235]

Nq = [1,2,3,4,5,6,7,8,9,10]
Nq_error_wd = [1.7767419112292782, 1.1350553193788804, 0.3209643094358516, 0.08725895848010799, 0.024428452240607544, 0.02180009023570803, 0.022077139652204668, 0.023093354274915285]
Nq_nrmse_p = [0.044308220184083195, 0.0441745476814875, 0.043563721574084786, 0.044142935219345356, 0.04465783453425025, 0.04442821637240173, 0.04413009618674215, 0.04464932500755689]

q = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9]
q_error_wd = [0.0330474763598668, 0.03129009779776156, 0.024428452240607544, 0.021675801660861867, 0.021965662940106875, 0.011532714735787718]
q_nrmse_p = [0.05013328326587912, 0.04978013355019077, 0.04465783453425025, 0.043714799333605385, 0.04689043837292038, 0.03387931252555877]

error_wd = q_error_wd
nrmse = q_nrmse_p

plt.figure()
plt.plot(q, error_wd)
plt.xlabel(r'$q$')
plt.ylabel(r'$\varepsilon$')
plt.yscale('log')
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\Nq_error_wd.tex")


plt.figure()
plt.plot(q, nrmse)
plt.xlabel(r'$q$')
plt.ylabel('nRMSE')
plt.yscale('log')
plt.grid()
# tikzplotlib.save(rf"tex_files\wind_data\Nq_nrmse.tex")
# plt.show()


# samples_total = [10,20,30,40]
# pce wasserstein distance =  [[1.4759334  2.36538735 1.74259023 2.77398253]
#  [1.47609713 2.36561055 1.74320406 2.77215865]
#  [1.47582973 2.36620172 1.74302092 2.77142956]
#  [1.47579462 2.36627716 1.74283587 2.77079196]
#  [1.47576289 2.36604155 1.74294052 2.77173914]]
# pce nrmse =  [[0.53627102 0.67873619 0.58266332 0.73462756]
#  [0.53627102 0.67873619 0.58266332 0.73462756]
#  [0.53627102 0.67873619 0.58266332 0.73462756]
#  [0.53627102 0.67873619 0.58266332 0.73462756]
#  [0.53627102 0.67873619 0.58266332 0.73462756]]
# pce p value mean =  [[1.82452202e-02 1.71055765e-06 1.74800764e-02 3.38952817e-02]
#  [1.67159278e-02 1.33459439e-06 1.61805814e-02 3.29116896e-02]
#  [1.72386296e-02 2.92994008e-06 1.86990629e-02 3.29672122e-02]
#  [1.51617324e-02 2.66841368e-06 1.65304654e-02 3.23946078e-02]
#  [1.61985629e-02 2.43185378e-06 1.83636602e-02 3.27846456e-02]]
# pce test statistic mean =  [[0.93935556 0.92672778 0.82161889 0.90632889]
#  [0.93919556 0.92665333 0.82215222 0.90724889]
#  [0.93939556 0.92601667 0.82125556 0.90678222]
#  [0.93950556 0.92622222 0.82160556 0.90713778]
#  [0.93895556 0.92617889 0.82123556 0.90707667]]