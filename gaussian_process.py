import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import chaospy as cp
from scipy.stats import gaussian_kde

class Gaussian_Process():

    def __init__(self, samples_x, samples_y, mean, sigma):
        self.samples_x = samples_x
        self.samples_y = samples_y
        self.mean = mean
        self.sigma = sigma
        pass

    def run(self):

        self.X = self.samples_x.reshape(-1, 1)
        self.y = self.mean

        rng = np.random.RandomState(1)
        training_indices = rng.choice(np.arange(self.y.size), size=100, replace=False)
        self.X_train, self.y_train = self.X[training_indices], self.y[training_indices]

        self.noise_std = np.mean(self.sigma)

        self.y_train_noisy = self.y_train + rng.normal(loc=0.0, scale=self.noise_std, size=self.y_train.shape)
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=self.noise_std, noise_level_bounds=(1e-5, 1e1))
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=9
        )
        gaussian_process.fit(self.X_train, self.y_train_noisy)
        self.mean_prediction, self.std_prediction = gaussian_process.predict(self.X, return_std=True)

        return self.mean_prediction, self.std_prediction

    def plot_gpr(self): 

        i = np.argsort(self.X.reshape(-1))

        plt.figure()
        plt.plot(self.X[i], self.y[i], label="reference", linestyle="dotted")
        plt.errorbar(
            self.X_train,
            self.y_train_noisy,
            self.noise_std,
            linestyle="None",
            color="tab:blue",
            marker=".",
            markersize=10,
            label="Observations",
        )
        plt.plot(self.X[i], self.mean_prediction[i], label="Mean prediction")
        plt.fill_between(
            self.X[i].ravel(),
            self.mean_prediction[i] - 1.96 * self.std_prediction[i],
            self.mean_prediction[i] + 1.96 * self.std_prediction[i],
            color="tab:orange",
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        _ = plt.title("GPR")
        # plt.show()

