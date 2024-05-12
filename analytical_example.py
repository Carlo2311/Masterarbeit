import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib

class AnalyticalExample():

    def __init__(self, n_samples, y):
        self.n_samples = n_samples
        self.y = y
        self.normal_distribution = cp.Normal(0, 1)

    def calculate_pdf(self, samples_x):
        component_1 = np.zeros((len(samples_x), len(self.y)))
        component_2 = np.zeros((len(samples_x), len(self.y)))
        pdf_values_1 = np.zeros((len(samples_x), len(self.y)))
        pdf_values_2 = np.zeros((len(samples_x), len(self.y)))
        pdf = np.zeros((len(samples_x), len(self.y)))
        mean_1 = np.zeros(len(samples_x))
        mean_2 = np.zeros(len(samples_x))

        for i, x in enumerate(samples_x):
            component_1[i, :] = 1.25 * self.y - (5 * np.sin(np.pi * x) ** 2 + 5 * x - 2.5)
            component_2[i, :] = 1.25 * self.y - (5 * np.sin(np.pi * x) ** 2 - 5 * x + 2.5)
            pdf_values_1[i, :] = 0.5 * self.normal_distribution.pdf(component_1[i, :])
            pdf_values_2[i, :] = 0.75 * self.normal_distribution.pdf(component_2[i, :])
            pdf[i, :] = pdf_values_1[i, :] + pdf_values_2[i, :]
            mean_1[i] = self.y[np.argmax(pdf_values_1[i, :])]
            mean_2[i] = self.y[np.argmax(pdf_values_2[i, :])]

        return pdf, mean_1, mean_2

    def create_data_points(self, pdf, n_samples):
        pdf_normalized = pdf / np.sum(pdf, axis=1, keepdims=True) # normalize PDF
        cdf = np.cumsum(pdf_normalized, axis=1) # CDF
        random_numbers = np.random.rand(n_samples) # uniform numbers between 0 and 1
        samples_y = np.zeros(n_samples)
        for i in range(n_samples):
            samples_y[i] = np.interp(random_numbers[i], cdf[i], self.y) # inverse of the CDF to map uniform random numbers to values

        return samples_y
    
    def plot_example(self, samples_x, samples_y, mean_1, mean_2, pdf):

        sorted_indices = np.argsort(samples_x)

        plt.figure(figsize=(10,6))
        plt.plot(samples_x[sorted_indices], mean_1[sorted_indices], 'r', label='Component 1')
        plt.plot(samples_x[sorted_indices], mean_2[sorted_indices], 'y', label='Component 2')
        plt.axvline(samples_x[sorted_indices][161], color='black', linewidth=0.5)
        plt.plot(0.25*pdf[sorted_indices][161,:] + samples_x[sorted_indices][161], self.y, 'b-.', linewidth=0.6, label='Response PDFs')
        plt.axvline(samples_x[sorted_indices][400], color='black', linewidth=0.5)
        plt.plot(0.25*pdf[sorted_indices][400,:] + samples_x[sorted_indices][400], self.y, 'b-.', linewidth=0.6)
        plt.axvline(samples_x[sorted_indices][561], color='black', linewidth=0.5)
        plt.plot(0.25*pdf[sorted_indices][561,:] + samples_x[sorted_indices][561], self.y, 'b-.', linewidth=0.6)
        plt.axvline(samples_x[sorted_indices][721], color='black', linewidth=0.5)
        plt.plot(0.25*pdf[sorted_indices][721,:] + samples_x[sorted_indices][721], self.y, 'b-.', linewidth=0.6)
        plt.scatter(samples_x, samples_y, s=1, label='Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        # tikzplotlib.save(rf"tex_files\analtical_example2.tex")
        plt.show()
    
    def plot_pdf(self, pdf, samples_x):
        for i, x in enumerate(samples_x):
            plt.figure(i)
            plt.plot(self.y, pdf[i,:])
            plt.title(f'x = {x}')
            plt.xlabel('y')
            plt.ylabel('PDF')
            plt.grid()
            #tikzplotlib.save(rf"tex_files\example_pdf_{x}.tex")
        plt.show()