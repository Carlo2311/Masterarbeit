import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib
from scipy.stats import norm
from scipy import stats

from scipy.integrate import quad


# Example usage:
distribution_x = np.random.normal(0, 1, size=10000)  # Example distribution x
distribution_y = np.random.normal(0, 1, size=10000)  # Example distribution y

def integrand(u, dist1, dist2):
    d_ws = (np.quantile(dist1, u) -  np.quantile(dist2, u))**2
    return d_ws

d_ws, _ = quad(integrand, 0, 1, args=(distribution_x, distribution_y))

print("Squared Wasserstein distance:", d_ws)

plt.figure()
plt.hist(distribution_x, bins=50)
plt.hist(distribution_y, bins=50)
plt.show()
