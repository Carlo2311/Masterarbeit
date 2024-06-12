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
        component_2 = 1.25 * self.y - (5 * np.sin(np.pi * samples_x_np) ** 2 - 5 * samples_x_np + 2.5)

        pdf_values_1 = self.normal_distribution.pdf(component_1)
        pdf_values_2 = self.normal_distribution.pdf(component_2)

        pdf = 0.5 * pdf_values_1 + 0.75 * pdf_values_2

        pdf_norm_1 = pdf_values_1 / np.sum(pdf_values_1, axis=1)[:, np.newaxis]
        pdf_norm_2 = pdf_values_2 / np.sum(pdf_values_2, axis=1)[:, np.newaxis]
        pdf_norm = pdf / np.sum(pdf, axis=1)[:, np.newaxis]

        mean_1 = np.sum(self.y * pdf_norm_1, axis=1)
        mean_2 = np.sum(self.y * pdf_norm_2, axis=1)
        mean_12 = np.sum(self.y * pdf_norm, axis=1)

        sigma_1 = np.sqrt(np.sum(pdf_norm_1 * (self.y - mean_1[:, np.newaxis]) ** 2, axis=1))
        sigma_2 = np.sqrt(np.sum(pdf_norm_2 * (self.y - mean_2[:, np.newaxis]) ** 2, axis=1))
        sigma_12 = np.sqrt(np.sum(pdf_norm * (self.y - mean_12[:, np.newaxis]) ** 2, axis=1))

        return pdf, mean_1, mean_2, sigma_1, sigma_2, mean_12, sigma_12
    
    def create_data_points(self, mean_1, mean_2, sigma_1, sigma_2, samples_plot, samples_x):

        dist_samples_uni = cp.Uniform(0, 1)
        
        samples_y = np.zeros((samples_x.shape[0], samples_plot))

        for i in range(samples_x.shape[0]):
            samples_uni = dist_samples_uni.sample(size=samples_plot) 
            mask = samples_uni <= 0.4
            dist_1 = cp.Normal(mean_1[i], sigma_1[i])
            dist_2 = cp.Normal(mean_2[i], sigma_2[i])
            samples_1 = dist_1.sample(samples_plot)
            samples_2 = dist_2.sample(samples_plot)
            samples_y[i, mask] = samples_1[mask]
            samples_y[i, ~mask] = samples_2[~mask]

        # dist_samples = cp.Uniform(0, 1)
        # samples = dist_samples.sample(size=samples_x.shape[0]) 
        # samples_y = np.zeros(samples_x.shape[0])

        # for i, sample in enumerate(samples):
        #     if sample <= 0.4:
        #         dist = cp.Normal(mean_1[i], sigma_1[i])
        #         samples_y[i] = dist.sample(samples_plot)
        #     else:
        #         dist = cp.Normal(mean_2[i], sigma_2[i])
        #         samples_y[i] = dist.sample(samples_plot)

        return samples_y


    def create_data_points1(self, pdf):
        pdf_normalized = pdf / np.sum(pdf, axis=1, keepdims=True) # normalize PDF
        cdf = np.cumsum(pdf_normalized, axis=1) # CDF
        random_numbers = np.random.rand(self.n_samples) # uniform numbers between 0 and 1
        samples_y = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            samples_y[i] = np.interp(random_numbers[i], cdf[i], self.y) # inverse of the CDF to map uniform random numbers to values

        return samples_y
    
    def plot_example(self, samples_x, samples_y, mean_1, mean_2, pdf, indices):

        sorted_indices = np.argsort(samples_x)

        plt.figure(figsize=(10,6))
        plt.plot(samples_x[sorted_indices], mean_1[sorted_indices], 'r', label='Component 1')
        plt.plot(samples_x[sorted_indices], mean_2[sorted_indices], 'y', label='Component 2')
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