from numpy import inf
# Constants
h = 100
d = [29, 7, 4]
a = [3, 5, 10]
heaving = [0, 1, 1]
# 0/false if not heaving, 1/true if yes heaving
NMK = [100, 100, 100, 100] # Number of terms in approximation of each region (including e).
# All computations assume at least 2 regions.

m0 = 1
g = 9.81
rho = 1023
