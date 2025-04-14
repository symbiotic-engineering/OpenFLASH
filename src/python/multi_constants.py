from numpy import inf
# Constants
h = 100
d = [29, 7, 4]
a = [3, 5, 10]
heaving = [1, 1, 1]
# 0/false if not heaving, 1/true if yes heaving
NMK = [100, 100, 100, 100] # Number of terms in approximation of each region (including e).
# All computations assume at least 2 regions.

m0 = 1
g = 9.81
rho = 1023
# n = 3 # These variables are just here but unused, inherited from the MEEM constants.py
# z = 6 # Why are they here?
# omega = 2 -> calculate omega from m0, g


####for RM3 slant study ###
# import numpy as np
# h = 50.0
# d = np.array([29.0, 7.0, 5.5, 4.0])
# a = np.array([3.0, 5.0, 7.5, 10.0])
# heaving = [0, 1, 1, 1] # 0/false if not heaving, 1/true if yes heaving
# slant = [0, 1, 1, 1] # 0/false if not slanted, 1/true if yes slanted
# n = 3
# z = 6
# omega = 0.4
# # omega = np.linspace(0.1, 1.5, 15)
# m0 = omega**2/9.81
