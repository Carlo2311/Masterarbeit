import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

####### input data simulation ############
file_path_input = 'simulation_data/casematrix.csv'
df = pd.read_csv(file_path_input)

windspeed = df['Windspeed [m/s]'].to_numpy()
turbulence_intensity = df['TI [-]'].to_numpy()
rho = df['Rho [kg/m3]'].to_numpy()
yaw_angle = df['Yaw Angle [°]'].to_numpy()
seed1 = df['Seed 1'].to_numpy()
seed2 = df['Seed 2'].to_numpy()

######## output data simulation ##############
file_path_output = 'simulation_data/surrogate_data_v2.csv'
df = pd.read_csv(file_path_output)
df_sorted = df.sort_values(by='Case')

case = df_sorted['Case'].to_numpy()
root_myb_mean = df_sorted['RootMyb_[kN-m]_mean'].to_numpy()
twh_tx_mean = df_sorted['TwHtALxt_[m/s^]_mean'].to_numpy()
twh_ty_mean = df_sorted['TwHtALyt_[m/s^]_mean'].to_numpy()
twh_tz_mean = df_sorted['TwHtALzt_[m/s^]_mean'].to_numpy()
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

plt.figure()
plt.hist(windspeed[:300], bins=30)
tikzplotlib.save(rf"tex_files\windspeed.tex")

plt.figure()
plt.scatter(windspeed[:300], turbulence_intensity[:300])

plt.figure()
plt.scatter(windspeed[:300], rho[:300])  

plt.figure()
plt.scatter(windspeed[:300], yaw_angle[:300])

plt.show()