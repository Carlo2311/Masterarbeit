import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from analytical_example_unimodal import AnalyticalExample
from spce import SPCE
from gaussian_process import Gaussian_Process
import time

n_samples = 50 #[50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
replications = [ 1,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
runs = 7
error_n = np.zeros((runs, len(replications)))
error_gpr = np.zeros((runs, len(replications)))
error_pce = np.zeros((runs, len(replications)))
nrmse_spce = np.zeros((runs, len(replications)))
nrmse_gpr = np.zeros((runs, len(replications)))
nrmse_pce = np.zeros((runs, len(replications)))


for r in range(runs):
    print('run = ', r)
    # for n, n_samples in enumerate(n_samples_all):
    for n, repli in enumerate(replications):
        print('repli = ', repli)
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

        n_samples_test = 10000
        n_x = 1000
        samples_x_test = dist_X.sample(n_x, rule='H')
        pce_mean_dist = surrogate_pce_mean(samples_x_test)
        pce_std_dist = np.abs(surrogate_pce_std(samples_x_test))
        dist_pce = np.random.normal(pce_mean_dist[:, np.newaxis], pce_std_dist[:, np.newaxis], (samples_x_test.shape[0], n_samples_test))

        pdf_test, mean_test, sigma_test = example.calculate_pdf(samples_x_test)
        samples_y_test = example.create_data_points(mean_test, sigma_test, n_samples_test, samples_x_test, pdf_test)

        error_pce[r,n] = spce.compute_error(dist_pce, samples_y_test)
        samples_y_mean = np.mean(samples_y_test, axis=1)
        mean_pce = np.mean(dist_pce, axis=1)
        nrmse_pce[r,n] = np.sqrt(np.mean((mean_pce - samples_y_mean)**2)) / np.mean(samples_y_mean)


print("error_pce = ", np.mean(error_pce, axis=0))
print("nrmse_pce = ", np.mean(nrmse_pce, axis=0))

plt.figure('RMSE')
plt.plot(replications, np.mean(nrmse_pce, axis=0), label='PCE')
plt.xlabel(f'replications')
plt.ylabel('nrmse')
plt.grid()
# plt.legend()
plt.yscale('log')
tikzplotlib.save(rf"tex_files\unimodal\unimodal_nrmse_pce.tex")

plt.figure('Error')
plt.plot(replications, np.mean(error_pce, axis=0), label='PCE')
plt.xlabel(f'replications')
plt.ylabel('error')
plt.grid()
# plt.legend()
plt.yscale('log')
tikzplotlib.save(rf"tex_files\unimodal\unimodal_error_pce.tex")

plt.show()



# a = np.array([50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500])
# b = 50 
# c = a/b
# print(c)

# error_spce =  [0.0025500003435142992, 0.001629529060027717, 0.0021237293339699654, 0.0014197864356587007, 0.00132943716284374, 0.00132943716284374, 0.0009324419297750125, 0.0008206147408165991]
# error_gpr =  [0.0007624079463735228, 0.0011518216618631382, 0.0006028589440239252, 0.0008510013023305934, 0.0007580966353888655, 0.0007580966353888655, 0.0011232334612469484, 0.0007394843434893262]
# error_pce =  [0.0020987828976634193, 0.002016920551706348, 0.0022198355960978693, 0.0017974167895561638, 0.0015950557117407934, 0.0015950557117407934, 0.0011802180326976196, 0.0016609056097424432]

# error_spce_1500 = [0.0019469701281922896, 0.0012104741641299956,0.0009818587458472723,0.0008034950227264338 , 0.0008573477835996824,0.0014574842460668826,0.000756345441738161,0.0011832445231859435]
# error_gpr_1500 = [0.001387964144287682, 0.001071335324353848, 0.0009713408035375278,0.0008650883074379538,0.001056990610736413,0.0009669427377679534,0.0005758052460233123,0.0012128262036443568]

# n_samples_all = [50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
# error_spce_samples =  [0.04243022, 0.01582406, 0.01201694, 0.00375494, 0.00532685, 0.00376756,
#  0.00372564, 0.00188208, 0.00234441, 0.00346637, 0.00133803, 0.00143895,
#  0.00190486, 0.00135425, 0.00126038, 0.00105598]
# error_gpr_samples =  [0.01583295, 0.01009484, 0.0069064,  0.00287697, 0.00393923, 0.00402161,
#  0.00113745, 0.00085226, 0.00209405, 0.0029831,  0.00097502, 0.00131458,
#  0.00111018, 0.00095042, 0.0008536,  0.00116727]

# nrmse_mean_spce =  [0.19523562, 0.11246637, 0.07012278, 0.05629622, 0.05177737, 0.0466912,
#  0.0424363,  0.04256411, 0.03960066, 0.03754044, 0.03730846, 0.03189046,
#  0.03121015, 0.03258987, 0.03090773, 0.03496799]
# nrmse_mean_gpr =  [0.14030619, 0.10164697, 0.06662755, 0.05217925, 0.04876567, 0.04518005,
#  0.04026104, 0.04144379, 0.03931617, 0.03583278, 0.0366923,  0.03019419,
#  0.02819257, 0.02986075, 0.02842366, 0.03126942]

# nrmse_mean_spce1 =  [0.0331545] # 50 30
# nrmse_mean_gpr1 =  [0.027592965] # 50 30
# nrmse_mean_pce1 =  [0.03281577] # 50 30
# nrmse_mean_spce =  [0.02940413] # 1500
# nrmse_mean_gpr =  [0.02567586] # 1500
# print(np.mean(nrmse_mean_spce))
# print(np.mean(nrmse_mean_gpr))
# print(np.mean(nrmse_mean_pce))

# error_spce_samples_all = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/error_spce.npy')
# error_gpr_samples_all = np.load(fr'C:/Users/carlo/Masterarbeit/Masterarbeit/solutions_unimodal/error_gpr.npy')
# n_samples_all = [50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]

# error_spce_mean = np.mean(error_spce_samples_all, axis=0)
# error_gpr_mean = np.mean(error_gpr_samples_all, axis=0)
# print("error spce = ", error_spce_mean)
# print("error gpr = ", error_gpr_mean)

# plt.figure()
# plt.plot(n_samples_all, nrmse_mean_spce, label='SPCE')
# plt.plot(n_samples_all, nrmse_mean_gpr, label='GPR')
# plt.xlabel(f'N')
# plt.ylabel('error')
# plt.grid()
# plt.legend()
# plt.yscale('log')
# # tikzplotlib.save(rf"tex_files\unimodal\unimodal_nrmse_spce_gpr.tex")
# plt.show()

# error spce =  [[0.09451874 0.00926381 0.00834126 0.0071673  0.00361055 0.00538644
#   0.00272806 0.00133215 0.00331597 0.00319061 0.00218324 0.00134065
#   0.00247214 0.00111238 0.00205669 0.00087492]
#  [0.07149827 0.00637123 0.00727161 0.00528728 0.00640771 0.00316983
#   0.00257411 0.00140864 0.00180433 0.00183957 0.00066264 0.00140469
#   0.00135395 0.00115656 0.00092633 0.0010748 ]
#  [0.0255601  0.01421873 0.00314769 0.00603351 0.00312754 0.0012455
#   0.00273243 0.00339348 0.00167233 0.00230208 0.0021829  0.00094745
#   0.00136858 0.00197238 0.00145105 0.0011349 ]
#  [0.08479451 0.01856667 0.01565479 0.01080792 0.00292166 0.00564
#   0.00134808 0.00278523 0.00195087 0.00242058 0.00249949 0.00110689
#   0.00087089 0.00143626 0.0013518  0.00091598]
#  [0.02991062 0.0483904  0.00632466 0.00428249 0.00381545 0.00264759
#   0.00320379 0.00462512 0.00280086 0.00233469 0.00168913 0.00059252
#   0.0016631  0.00078813 0.00087664 0.0012109 ]
#  [0.03967188 0.01590494 0.00729745 0.00675933 0.00588118 0.0011102
#   0.00170069 0.00165621 0.00210381 0.00079011 0.00097816 0.00178277
#   0.00198607 0.00136313 0.00090136 0.00097873]
#  [0.02421958 0.03539271 0.00942464 0.00716285 0.005724   0.00456761
#   0.00255977 0.00297975 0.00198399 0.00075783 0.00073544 0.00077653
#   0.00135583 0.00095899 0.00154637 0.00162861]]
# error gpr =  [[0.02627097 0.00599633 0.00780054 0.0050549  0.00360525 0.00631753
#   0.00214267 0.0017881  0.00138882 0.00165489 0.001599   0.00089318
#   0.00201066 0.00078619 0.00108467 0.00070043]
#  [0.01808572 0.00556562 0.00738992 0.0024517  0.00578798 0.0025342
#   0.00295452 0.00115854 0.00102347 0.00091054 0.00042811 0.00095243
#   0.00187898 0.00131023 0.00099688 0.00091324]
#  [0.03160157 0.00660546 0.00381261 0.00257408 0.00375515 0.00118679
#   0.00067055 0.00306946 0.00129119 0.00200515 0.00148466 0.00093056
#   0.00110132 0.00102134 0.00099297 0.00081984]
#  [0.03895528 0.01857427 0.01235883 0.00954256 0.00264497 0.0046557
#   0.00163646 0.00178501 0.00165978 0.0026003  0.0009385  0.00081785
#   0.00077324 0.00085399 0.00107406 0.00060942]
#  [0.00844379 0.02205531 0.00588264 0.00325748 0.00502796 0.00370977
#   0.0024982  0.00359454 0.001021   0.00366511 0.00147116 0.00061155
#   0.00131852 0.00097257 0.00076832 0.00081536]
#  [0.01772053 0.00750191 0.00171946 0.006454   0.00209706 0.00084386
#   0.00264152 0.00163242 0.00188317 0.00080988 0.00092345 0.0010751
#   0.00174232 0.0007946  0.00074451 0.00076853]
#  [0.00971742 0.01225192 0.00639053 0.00404644 0.0024373  0.00279104
#   0.00255835 0.00240288 0.00148407 0.00082012 0.0005658  0.00079839
#   0.00098219 0.0007422  0.00128382 0.00160145]]



# print("error_spce_mean = ", np.mean(error_spce))
# print("error_gpr_mean = ", np.mean(error_gpr))
# print("error_pce_mean = ", np.mean(error_pce))
# print("error_spce_1500_mean = ", np.mean(error_spce_1500))
# print("error_gpr_1500_mean = ", np.mean(error_gpr_1500))

# Nq = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]

# e_wasser = [ 1.1091833057153924, 0.1451418510670338,  0.09455602241156448, 0.023677829103978972, 0.02261651115118024, 0.025679368512708273, 0.02231660802671113, 0.026677512869446657, 0.023845299757542616, 0.023605768820760133]
# rmse = [0.0457621603072, 0.04406933960367994, 0.0484570617919083, 0.04406729451470503,  0.04363864252117107, 0.04431546350656331, 0.045506616838585985, 0.04520678499775289,  0.043895097850468516, 0.04412393010094407]

# p = [6]
# ewasser = [0.01558822212634445]
# rmse = [0.03303458605237961]

# plt.figure()
# plt.plot(Nq, e_wasser)
# plt.xlabel("p")
# plt.ylabel("nRMSE")
# plt.yscale("log") 
# plt.show()


# für plot error und rmse über replications
# nrmse mean spce =  [0.16521264 0.14625732 0.10270257 0.06945739 0.0598631  0.04203771
#  0.04377587 0.03206475 0.04163113 0.04522027 0.03653647 0.02873255
#  0.02070515 0.03345097 0.03298804 0.03331998]
# nrmse mean gpr =  [0.12244294 0.12997083 0.06065084 0.048103   0.06228617 0.04827453
#  0.06170099 0.04038532 0.03544768 0.04236256 0.0299283  0.02863407
#  0.02800075 0.02663852 0.03540972 0.02256019]
# nrmse mean pce =  [0.17052143 0.14010098 0.08453658 0.05062428 0.05743987 0.04116364
#  0.04810613 0.0328441  0.04178435 0.04505782 0.03630812 0.02840621
#  0.0213812  0.03254321 0.03297184 0.03260982]
# error spce =  [[0.04101852 0.02778846 0.01594702 0.00971551 0.00538765 0.0054005
#   0.00243217 0.00166646 0.00250782 0.00277233 0.00143591 0.00126749
#   0.00137355 0.00126118 0.00147211 0.00168518]]
# error gpr =  [[0.01784145 0.01726425 0.00467051 0.00261305 0.00416746 0.00268202
#   0.00404773 0.00190519 0.00166129 0.00198229 0.00102438 0.00101882
#   0.00110097 0.00084358 0.0014969  0.00069727]]
# error pce =  [[0.19195555 0.06341658 0.0124736  0.00581904 0.00548452 0.00319068
#   0.00316318 0.00204867 0.00364786 0.00316547 0.0020261  0.00119002
#   0.00089425 0.00200074 0.00134506 0.00173085]]