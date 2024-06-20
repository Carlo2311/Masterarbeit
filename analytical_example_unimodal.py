import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
# import tikzplotlib

class AnalyticalExample():

    def __init__(self, n_samples, y):
        self.n_samples = n_samples
        self.y = y
        self.normal_distribution = cp.Normal(0, 1)

    def calculate_pdf(self, samples_x):
        samples_x_np = np.asarray(samples_x)[:, np.newaxis]
        component_1 = 1.25 * self.y - (5 * np.sin(np.pi * samples_x_np) ** 2 + 5 * samples_x_np - 2.5)
        pdf_values_1 = self.normal_distribution.pdf(component_1)
        pdf = 0.5 * pdf_values_1 / 0.4
        pdf_norm = pdf / np.sum(pdf, axis=1)[:, np.newaxis]
        mean = np.sum(self.y * pdf_norm, axis=1)
        sigma = np.sqrt(np.sum(pdf_norm * (self.y - mean[:, np.newaxis]) ** 2, axis=1))

        return pdf, mean, sigma
    
    def create_data_points(self, mean, sigma, samples_plot, samples_x):

        samples_y = np.zeros((samples_x.shape[0], samples_plot))

        for i in range(samples_x.shape[0]):
            dist_1 = cp.Normal(mean[i], sigma[i])
            samples_1 = dist_1.sample(samples_plot)
            samples_y[i, :] = samples_1[:]

        return samples_y

    
    def plot_example(self, samples_x, samples_y, mean_1, pdf, indices):

        sorted_indices = np.argsort(samples_x)

        plt.figure(figsize=(10,6))
        plt.plot(samples_x[sorted_indices], mean_1[sorted_indices], 'r', label='Component 1')
        plt.axvline(samples_x[indices[0]], color='black', linewidth=0.5)
        plt.plot(0.25*pdf[indices[0],:] + samples_x[indices[0]], self.y, 'b-.', linewidth=0.6, label='Response PDFs')
        plt.axvline(samples_x[indices[1]], color='black', linewidth=0.5)
        plt.plot(0.25*pdf[indices[1],:] + samples_x[indices[1]], self.y, 'b-.', linewidth=0.6)
        plt.axvline(samples_x[indices[2]], color='black', linewidth=0.5)
        plt.plot(0.25*pdf[indices[2],:] + samples_x[indices[2]], self.y, 'b-.', linewidth=0.6)
        plt.axvline(samples_x[indices[3]], color='black', linewidth=0.5)
        plt.plot(0.25*pdf[indices[3],:] + samples_x[indices[3]], self.y, 'b-.', linewidth=0.6)
        plt.scatter(samples_x, samples_y, s=1, label='Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        # tikzplotlib.save(rf"tex_files\analtical_example2.tex")
        plt.show()
    
    def plot_pdf(self, pdf, samples_x, indices):
        for i, x in enumerate(samples_x):
            plt.figure()
            plt.plot(self.y, pdf[indices[i],:])
            plt.title(f'x = {x}')
            plt.xlabel('y')
            plt.ylabel('PDF')
            plt.grid()
            #tikzplotlib.save(rf"tex_files\example_pdf_{x}.tex")
        plt.show()