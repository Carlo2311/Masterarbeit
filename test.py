import chaospy as cp
import numpy as np

# Parameters for the two normal distributions (modes)
mean_1 = 0
sigma_1 = 1
mean_2 = 4
sigma_2 = 1

# Create normal distributions for the two modes
dist_1 = cp.Normal(mean_1, sigma_1)
dist_2 = cp.Normal(mean_2, sigma_2)

# Define weights for the two modes
weight_1 = 0.5
weight_2 = 0.5

# Create a bimodal distribution by combining the two modes
dist_bimodal = cp.J(dist_1, dist_2)

# Define a custom function to calculate the PDF of the bimodal distribution
def bimodal_pdf(x):
    return weight_1 * dist_1.pdf(x[0]) + weight_2 * dist_2.pdf(x[1])

# Assign the custom PDF function to the bimodal distribution
dist_bimodal._pdf = bimodal_pdf

# Generate some samples from the bimodal distribution
samples = dist_bimodal.sample(1000)

# Calculate the probability density function (PDF) of the bimodal distribution
pdf_values = bimodal_pdf(samples)

# Plot the histogram of the samples and compare it with the PDF
import matplotlib.pyplot as plt
plt.hist(samples[0], bins=50, density=True, alpha=0.5, label='Sampled PDF')
plt.scatter(samples[0], pdf_values, label='True PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Bimodal Probability Density Function')
plt.legend()
plt.show()