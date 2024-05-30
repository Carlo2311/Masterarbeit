import numpy as np 
import matplotlib.pyplot as plt
import chaospy as cp
import tikzplotlib


a = np.array([[[1,2,1], [3,4,3]], [[5,6,5], [7,8,7]]])
b = np.array([[4, 5, 6, 7], [1,2,3,4]])
c = np.array([2, 1, 3, 4])

test = b * c
print(test)