# equation.py
import numpy as np
from scipy.special import hankel1 as besselh
from scipy.special import iv as besseli
from scipy.special import kv as besselk
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from numpy import sqrt, cosh, cos, sinh, sin, pi
from scipy.optimize import newton, minimize_scalar, root_scalar
import scipy as sp

# Defining m_k function that will be used later on
def m_k(k, m0, h):
    # m_k_mat = np.zeros((len(m0_vec), 1))

    m_k_h_err = (
        lambda m_k_h: (m_k_h * np.tan(m_k_h) + m0 * h * np.tanh(m0 * h))
    )
    k_idx = k
    m_k_h_lower = pi * (k_idx - 1/2) + np.finfo(float).eps
    m_k_h_upper = pi * k_idx - np.finfo(float).eps
    # x_0 =  (m_k_upper - m_k_lower) / 2
    
    m_k_initial_guess = pi * (k_idx - 1/2) + np.finfo(float).eps
    result = root_scalar(m_k_h_err, x0=m_k_initial_guess, method="newton")
    # result = minimize_scalar(
        # m_k_h_err, bounds=(m_k_h_lower, m_k_h_upper), method="bounded"
    # )
    

    m_k_val = result.root / h

    shouldnt_be_int = np.round(m0 * m_k_val / np.pi - 0.5, 4)
    # not_repeated = np.unique(m_k_val) == m_k_val
    assert np.all(shouldnt_be_int != np.floor(shouldnt_be_int))

        # m_k_mat[freq_idx, :] = m_k_vec
    return m_k_val

# Equation 4:
def lambda_n1(n, h, d1):
    return n * pi / (h - d1)


def lambda_n2(n, h, d2):
    return n * pi / (h - d2)


#############################################
# Equation 5
def phi_p_a1(z, a1):
    return phi_p_i1(a1, z)

def phi_p_a2(z, a2, h, d2):
    return phi_p_i2(a2, z, h, d2)

def phi_p_i1(r, z, h, d1): 
    return (1 / (2* (h - d1))) * ((z + h) ** 2 - (r**2) / 2)

def phi_p_i2(r, z, h, d2): 
    return (1 / (2* (h - d2))) * ((z + h) ** 2 - (r**2) / 2)

def phi_p_i1_i2_a1(z, h, a1, d1, d2):
    res = ((h + z) ** 2 - a1**2 / 2) / (2 * d1 - 2 * h) - (
        (h + z) ** 2 - a1**2 / 2
    ) / (2 * d2 - 2 * h)
    return res

def diff_phi_p_i2_a2(h, a2, d2):
  return a2/(2*d2 - 2*h)

def diff_phi_p_i1_i2_a1(z, h, a1, d1, d2): #differentiation of difference of particular solution
  return ((h + z)**2 - a1**2/2)/(2*d1 - 2*h) - ((h + z)**2 - a1**2/2)/(2*d2 - 2*h) #flux/velocity at a2

def diff_phi_helper(r, di, h): 
    return -r / (2 * (h - di))

def diff_phi_i1(r, d1, h): 
    return diff_phi_helper(r, d1, h)

def diff_phi_i2(r, d2, h): 
    return diff_phi_helper(r, d2, h)


#############################################
# Equation 7: (r specifies the raidus, use a1/a2 for the radius of the cylinder you want)
def R_1n_1(n, r, a2, h, d1):
    if n == 0:
        return 0.5
    elif n >= 1:
        return besseli(0, lambda_n1(n, h, d1) * r) / besseli(0, lambda_n1(n, h, d1) * a2)
    else: 
        raise ValueError("Invalid value for n")

def R_1n_2(n, r, a2, h, d2):
    if n == 0:
        return 0.5
    elif n >= 1:
        return besseli(0, lambda_n2(n, h, d2) * r) / besseli(0, lambda_n2(n, h, d2) * a2)
    else: 
        raise ValueError("Invalid value for n")

# Differentiating equation 7: (Once again look at changing the r's here
def diff_R_1n_1(n, r, d1, h, a2):
    if n == 0:
        return 0
    else:
        top = n * np.pi * besseli(1, np.pi * n * r / (d1 - h))
        bottom = (d1 - h) * besseli(0, np.pi * a2 * n / (d1 - h))
        return top / bottom


def diff_R_1n_2(n, r, d2, h, a2):
    if n == 0:
        return 0
    else:
        top = n * np.pi * besseli(1, np.pi * n * r / (d2 - h))
        bottom = (d2 - h) * besseli(0, np.pi * a2 * n / (d2 - h))
        return top / bottom


#############################################
# Equation 8:


# My original definition
def R_2n_2(n, r, a2, h, d2):
    if n == 0:
        return 0.5 * np.log(r / a2)
    else:
        return besselk(0, lambda_n2(n, h, d2) * r) / besselk(0, lambda_n2(n, h, d2) * a2)


# Differentiating equation 8:



def diff_R_2n_2(n, r, d2, h, a2):
    if n == 0:
        return 1 / (2 * r)
    else:
        top = n * np.pi * besselk(1, -(np.pi * n * r) / (d2 - h))
        bottom = (d2 - h) * besselk(0, -(np.pi * n * a2) / (d2 - h))
        return top / bottom


#############################################
# Equation 9:
def Z_n_i1(n, z, h, d1):
    if n == 0:
        return 1
    else:
        return np.sqrt(2) * np.cos(lambda_n1(n, h, d1) * (z + h))


def Z_n_i2(n, z, h, d2):
    if n == 0:
        return 1
    else:
        return np.sqrt(2) * np.cos(lambda_n2(n, h, d2) * (z + h))


#############################################
# Equation 13: (m_k is a function)
def Lambda_k_r(k, r, m0, a2, h):
    if k == 0:
        return besselh(0, m0 * r) / besselh(0, m0 * a2)
    else:
        return besselk(0, m_k(k, m0, h) * r) / besselk(0, m_k(k, m0, h) * a2)


def diff_Lambda_k_a2(n, m0, a2, h):
    if n == 0:
        numerator = -(m0 * besselh(1, m0 * a2))
        denominator = besselh(0, a2 * m0)
    else:
        numerator = -(m_k(n, m0, h) * besselk(1, a2 * m_k(n, m0, h)))
        denominator = besselk(0, a2 * m_k(n, m0, h))
    return numerator / denominator


#############################################
# Equation 2.34 in analytical methods book, also eq 16 in Seah and Yeung 2006:
def N_k(k, m0, h):
    if k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * m_k(k, m0, h) * h) / (2 * m_k(k, m0, h) * h))


#############################################
# Equation 14: (m_k is a function)
def Z_n_e(k, z, m0, h):
    if k == 0:
        return 1 / sqrt(N_k(k, m0, h)) * cosh(m0 * (z + h))
    else:
        return 1 / sqrt(N_k(k, m0, h)) * cos(m_k(k, m0, h) * (z + h))

#############################################
# To calculate hydrocoefficients

#differentiate with respect to z
def diff_phi_p_i1_dz(z, h, d1):
    return (h+z)/(h-d1)

#differentiate with respect to z
def diff_phi_p_i2_dz(z, h, d2):
    return (h+z)/(h-d2)


#integrating R_1n_1
def int_R_1n_1(n, a1, a2, h, d1):
    if n == 0:
        return a1**2/4
    else:
        top = a1*besseli(1, lambda_n1(n, h, d1)*a1)
        bottom = lambda_n1(n, h, d1)*besseli(0, lambda_n1(n, h, d1)*a2)
        return top/bottom

#integrating R_1n_2
def int_R_1n_2(n, a2, a1, h, d2):
    if n == 0:
        return a2**2/4 - a1**2/4
    else:
        top = a2*besseli(1, lambda_n2(n, h, d2)*a2)-a1*besseli(1, lambda_n2(n, h, d2)*a1)
        bottom = lambda_n2(n, h, d2)*besseli(0, lambda_n2(n, h, d2)*a2)
        return top / bottom

#integrating R_2n_2
def int_R_2n_2(n, a1, a2, h, d2):
    if n == 0:
        return (a1**2*(2*np.log(a2)-2*np.log(a1)+1)-a2**2)/8
    else:
        top = a2*besselk(1, lambda_n2(n, h, d2)*a2)-a1*besselk(1, lambda_n2(n, h, d2)*a1)
        bottom = -lambda_n2(n, h, d2)*besselk(0, lambda_n2(n, h, d2)*a2)
        return top / bottom
    
#integrating phi_p_i1 * d_phi_p_i1/dz * r *d_r at z=d1
def int_phi_p_i1_no_coef(a1, h, d1):
    return a1**2*(4*(h-d1)**2-a1**2) / (16*(h-d1))

#integrating phi_p_i2 * d_phi_p_i2/dz * r *d_r at z=d1
def int_phi_p_i2_no_coef(a1, a2, h, d2):
    return (a2**2*(4*(h-d2)**2-a2**2) - a1**2*(4*(h-d2)**2-a1**2)) / (16*(h-d2))


def z_n_d1_d2(n):
    if n ==0:
        return 1
    else:
        return sqrt(2)*(-1)**n