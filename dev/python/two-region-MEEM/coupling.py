import numpy as np
from scipy.special import hankel1 as besselh
from scipy.special import iv as besseli
from scipy.special import kv as besselk
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from math import sqrt, cosh, cos, sinh, sin, pi
from scipy.optimize import newton, minimize_scalar
import scipy as sp
from constants import *
from equations import *



#############################################
# Coupling integrals: (m_k is a function)
def sq(n): 
    return n ** 2

# Order is flipped
def A_nm(n, m):
    if n == 0 and m == 0:
        return h - d1
    if m == 0 and 1 <= n:
        return 0
    sigma = sin((pi * m * (d1 - h)) / (d2 - h))
    if 1 <= m and n == 0:
        return (-sqrt(2) * sigma * (d2 - h)) / (m * pi)
    if 1 <= m and 1 <= n:
        top = -2 * ((-1) ** n) * m * sigma * sq(d1 - h) * (d2 - h)
        bottom = pi * ((sq(d1) * sq(m)) - (2 * d1 * h * sq(m)) - (sq(d2) * sq(n)) + (2 * d2 * h * sq(n)) + (sq(h) * sq(m)) - (sq(h) * sq(n)))
        return top / bottom     
    else:
        raise ValueError("Invalid values for j and n")


def A_nm2(j, n):
    if j == 0 and n == 0:
        return h - d2
    sigma = (pi * n * (d2 - h)) / (d1 - h)
    if j == 0 and 1 <= n:
        return (-sqrt(2) * sin(sigma) * (d1 - h)) / (n * pi)
    if 1 <= j and n == 0:
        return (-sqrt(2) * sin(pi * j) * (d2 - h)) / (j * pi)
    if 1 <= j and 1 <= n:
        top = -2 * (j * (d1 - h) * (d2 - h) * (d1 * sin(pi * j) * cos(sigma) - h * sin(pi * j) * cos(sigma)) - n * (d1 - h) * (d2 - h) * (d2 * sin(sigma) * cos(pi * j) - h * sin(sigma) * cos(pi * j)))
        bottom = pi * ((sq(d1) * sq(j)) - (2 * d1 * h * sq(j)) - (sq(d2) * sq(n)) + (2 * d2 * h * sq(n)) + (sq(h) * sq(j)) - (sq(h) * sq(n)))
        return top / bottom     
    else:
        raise ValueError("Invalid values for j and n")

def A_nj(n, j):
    if j == 0 and n == 0:
        return h - d1
    if 1 <= j and n == 0:
        return (-sqrt(2) * sin(pi * j) * (d1 - h)) / (j * pi)
    sigma = (pi * n * (d1 - h)) / (d2 - h)
    if j == 0 and 1 <= n:
        return (-sqrt(2) * sin(sigma) * (d2 - h)) / (n * pi)
    if 1 <= j and 1 <= n:
        top = (2 * (j * (d1 - h) * (d2 - h) * (d2 * sin(pi * j) * cos(sigma) - h * sin(pi * j) * cos(sigma)) - n * (d1 - h) * (d2 - h) * (d2 * sin(sigma) * cos(pi * j) - h * sin(sigma) * cos(pi * j))))
        bottom = pi * ((-sq(d1) * sq(n)) + (2 * d1 * h * sq(n)) + (sq(d2) * sq(j)) - (2 * d2 * h * sq(j)) + (sq(h) * sq(j)) - (sq(h) * sq(n)))
        return -(top / bottom)     
    else:
        raise ValueError("Invalid values for n and j")

def A_nj2(n, j):
    if j == 0 and n == 0:
        return h - d2
    sigma = (pi * j * (d2 - h)) / (d1 - h)
    if 1 <= j and n == 0:
        return -(sqrt(2) * sin(sigma) * (d1 - h)) / (j * pi)
    if j == 0 and 1 <= n:
        return -(sqrt(2) * sin(pi * n) * (d2 - h)) / (n * pi)
    if 1 <= j and 1 <= n:
        top = -(2 * (j * (d1 - h) * (d2 - h) * (d2 * sin(sigma) * cos(pi * n) - h * sin(sigma) * cos(pi * n)) - n * (d1 - h) * (d2 - h) * (d1 * sin(pi * n) * cos(sigma) - h * sin(pi * n) * cos(sigma))))
        bottom = pi * ((-sq(d1) * sq(n)) + (2 * d1 * h * sq(n)) + (sq(d2) * sq(j)) - (2 * d2 * h * sq(j)) + (sq(h) * sq(j)) - (sq(h) * sq(n)))
        return top / bottom     
    else:
        raise ValueError("Invalid values for n and j")

def nk_sigma_helper(mk, k, m):
    sigma1 = sqrt((sinh(2 * h * m0) + 2 * h * m0) / h)
    sigma2 = sin(mk * (d2 - h))
    sigma3 = pi ** 2 * m ** 2
    sigma4 = sinh(m0 * (d2 - h))
    sigma6 = 2 * h * mk
    sigma5 = sqrt(sin(sigma6) / sigma6 + 1)
    
    return sigma1, sigma2, sigma3, sigma4, sigma5

def A_mk(m, k):
    mk = m_k(k)
    sigma1, sigma2, sigma3, sigma4, sigma5 = nk_sigma_helper(mk, k, m)

    if k == 0 and m == 0:
        C_mk = -2 * sigma4 / (sqrt(m0) * sigma1)
    elif 1 <= k and m == 0:
        C_mk = -sqrt(2) * sigma2 / (mk * sigma5)
    elif k == 0 and 1 <= m:
        C_mk = -(2 * (-1)**m * sqrt(2) * m0**(3/2) * sigma4 * (d2 - h)**2) / \
               (sigma1 * (d2**2 * m0**2 - 2 * d2 * h * m0**2 + h**2 * m0**2 + sigma3))
    elif 1 <= k and 1 <= m:
        C_mk = -(2 * (-1)**m * sigma2 * mk * (d2 - h)**2) / \
               (sigma5 * (d2**2 * mk**2 - 2 * d2 * h * mk**2 + h**2 * mk**2 - sigma3))
    else: 
        raise ValueError("Invalid values for m and k")
    
    return C_mk


# def nk_sigma_helper(mk): 
#     top = sin(2 * h * mk)
#     bottom = 4 * h * mk
#     sigma1 = sqrt(top/bottom + 1/2)
#     sigma2 = sinh(m0 * (d2 - h)) 
#     sigma3 = mk * (d2 - h)
#     sigma4 = sq(pi) * sq(n)
#     sigma5 = sinh(2 * h * m0)
#     return sigma1, sigma2, sigma3, sigma4, sigma5

# def A_km(k, n):
#     mk = m_k(k)
#     sigma1, sigma2, sigma3, sigma4, sigma5 = nk_sigma_helper(mk)
#     if k == 0 and n == 0:
#         return (-2 * sqrt(h) * sigma2) / (sqrt(m0) * sqrt(sigma5 + 2 * h * m0))
#     elif 1 <= k and n == 0:
#         return -sin(sigma3) / (mk * sigma1)
#     elif k == 0 and 1 <= n:
#         top = -sqrt(2) * (m0 * (d2 * cos(pi * n) * sigma2 - h * cos(pi * n) * sigma2) * (d2 - h) + pi * n * sin(pi * n) * cosh(m0 * (d2 - h)) * (d2 - h))
#         bottom = sqrt((sigma5 / (4 * h * m0)) + 1/2) * (sq(d2) * sq(m0) - 2 * d2 * h * sq(m0) + sq(h) * sq(m0) + sigma4)
#         return top / bottom
#     elif 1 <= k and 1 <= n:
#         top = -sqrt(2) * (mk * (d2 * sin(sigma3) * cos(pi * n) - h * sin(sigma3) * cos(pi * n)) * (d2 - h) - pi * n * cos(sigma3) * sin(pi * n) * (d2 - h))
#         bottom = sigma1 * (sq(d2) * sq(mk) - 2 * d2 * h * sq(mk) + sq(h) * sq(mk) - sigma4)
#         return top / bottom
#     else:
#         raise ValueError("Invalid values for n and k")

def nk2_sigma_helper(mk): 
    top = sin(2 * h * mk)
    bottom = 4 * h * mk
    sigma1 = sqrt(top/bottom + 1/2)
    sigma2 = sin(h * mk) 
    sigma3 = (pi * h * n) / (d2 - h)
    sigma4 = sq(pi) * sq(n)
    sigma5 = sinh(2 * h * m0)
    return sigma1, sigma2, sigma3, sigma4, sigma5


def A_km2(n, k):
    mk = m_k(k)
    sigma1, sigma2, sigma3, sigma4, sigma5 = nk2_sigma_helper(mk)
    if k == 0 and n == 0:
        return (-2 * sqrt(h) * sinh(h * m0)) / (sqrt(m0) * sqrt(sigma5 + 2 * h * m0))
    elif 1 <= k and n == 0:
        return sigma2 / (mk * sigma1)
    elif k == 0 and 1 <= n:
        top = sqrt(2) * (m0 * (d2 - h) * (d2 * sinh(h * m0) * cos(sigma3) - h * sinh(h * m0) * cos(sigma3)) + pi * n * cosh(h * m0) * sin(sigma3) * (d2 - h))
        bottom = sqrt((sigma5 / (4 * h * m0)) + 1/2) * (sq(d2) * sq(m0) - 2 * d2 * h * sq(m0) + sq(h) * sq(m0) + sigma4)
        return top / bottom
    elif 1 <= k and 1 <= n:
        top = sqrt(2) * (mk * (d2 - h) * (d2 * sigma2 * cos(sigma3) - h * sigma2 * cos(sigma3)) - pi * n * cos(h * mk) * sin(sigma3) * (d2 - h))
        bottom = sigma1 * (sq(d2) * sq(mk) - 2 * d2 * h * sq(mk) + sq(h) * sq(mk) - sigma4)
        return top / bottom
    else:
        raise ValueError("Invalid values for n and k")
    
