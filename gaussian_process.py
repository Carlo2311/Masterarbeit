import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import chaospy as cp
from scipy.stats import gaussian_kde

'''
Class to create a Gaussian Process
Author: Carlotta Hilscher
Date: October 2024
'''

class Gaussian_Process():

    ### initialize the Gaussian Process ###
    def __init__(self, samples_x, mean, sigma):
        self.samples_x = samples_x
        self.mean = mean
        self.sigma = sigma
        pass

    ### function to run the Gaussian Process ###
    def run(self, x_train, y_train):

        if len(self.samples_x.shape) == 1:
            self.X = self.samples_x.reshape(-1, 1)
            self.X_train = x_train.reshape(-1, 1)
        else:
            self.X = self.samples_x.T
            self.X_train = x_train.T
        
        self.y_train_noisy = y_train
        self.noise_std = np.mean(self.sigma)


        kernel = RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=self.noise_std, noise_level_bounds=(1e-5, 1e1))
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=9
        )
        gaussian_process.fit(self.X_train, self.y_train_noisy)
        self.mean_prediction, self.std_prediction = gaussian_process.predict(self.X, return_std=True)

        return self.mean_prediction, self.std_prediction

    