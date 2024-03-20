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


# Constants
h = 10
a1 = 5
a2 = 10
d1 = 5
d2 = 2.5
m0 = 1
n = 3
z = 6
omega = 2


# Defining m_k function that will be used later on
def m_k(h_mat):
    m0_vec = np.array([m0])  # Define the m0_vec array
    m_k_mat = np.zeros((len(m0_vec), 1))

    for freq_idx in range(len(m0_vec)):
        m_k_vec = np.zeros(1)
        m0_i = m0_vec[freq_idx]
        m_k_sq_err = (
            lambda m_k: (m_k * np.tan(m_k * h_mat) - m0_i * np.tanh(m0_i * h_mat)) ** 2
        )
        for k_idx in range(len(m0_vec)):
            m_k_lower = (np.pi * (k_idx) + np.pi / 2) / m0_i + np.finfo(float).eps
            m_k_upper = (np.pi * (k_idx) + np.pi) / m0_i - np.finfo(float).eps
            result = minimize_scalar(
                m_k_sq_err, bounds=(m_k_lower, m_k_upper), method="bounded"
            )
            m_k_vec[k_idx] = result.x

        shouldnt_be_int = np.round(m0_i * m_k_vec / np.pi - 0.5, 4)
        not_repeated = len(np.unique(m_k_vec)) == len(m_k_vec)
        assert np.all(shouldnt_be_int != np.floor(shouldnt_be_int)) and not_repeated

        m_k_mat[freq_idx, :] = m_k_vec
    return m_k_mat[0][0]


def m_k_newton(h):
    res = newton(lambda k: k * np.tanh(k * h) - m0**2 / 9.8, x0=1.0, tol=10 ** (-10))
    return res


# Equation 4:
def lambda_n1(n):
    return n * np.pi / (h - d1)


def lambda_n2(n):
    return n * np.pi / (h - d2)


#############################################
# Equation 5
def phi_p_a1(z):
    return (1 / (2 * (h - d1))) * ((z + h) ** 2 - (a1**2) / 2)


def phi_p_a2(z):
    return (1 / (2 * (h - d2))) * ((z + h) ** 2 - (a2**2) / 2)


def phi_p_i1_i2_a1(z):
    res = ((h + z) ** 2 - a1**2 / 2) / (2 * d1 - 2 * h) - (
        (h + z) ** 2 - a1**2 / 2
    ) / (2 * d2 - 2 * h)
    return res


#############################################
# Equation 7: (r specifies the raidus, use a1/a2 for the radius of the cylinder you want)
def R_1n_1(n, r):
    if n == 0:
        return 0.5
    else:
        return besseli(0, lambda_n1(n) * r) / besseli(0, lambda_n1(n) * a2)


def R_1n_2(n, r):
    if n == 0:
        return 0.5
    else:
        return besseli(0, lambda_n2(n) * r) / besseli(0, lambda_n2(n) * a2)


# Differentiating equation 7: (Once again look at changing the r's here
def diff_R_1n_1(n, r):
    if n == 0:
        return 0
    else:
        top = n * np.pi * besseli(1, np.pi * n * r / (d1 - h))
        bottom = (d1 - h) * besseli(0, np.pi * a2 * n / (d1 - h))
        return top / bottom


def diff_R_1n_2(n, r):
    if n == 0:
        return 0
    else:
        top = n * np.pi * besseli(1, np.pi * n * r / (d2 - h))
        bottom = (d2 - h) * besseli(0, np.pi * a2 * n / (d2 - h))
        return top / bottom


#############################################
# Equation 8:
# This function is always 0 regardless of the output
def R_2n_1(n):
    return 0.0


# My original definition
def R_2n_2(n, r):
    if n == 0:
        return 0.5 * np.log(r / a2)
    else:
        return besselk(0, lambda_n2(n) * r) / besselk(0, lambda_n2(n) * a2)


# Differentiating equation 8:
def diff_R_2n_1(n):
    return 0.0


def diff_R_2n_2(n, r):
    if n == 0:
        return 1 / (2 * r)
    else:
        top = n * np.pi * besselk(1, -(np.pi * n * r) / (d2 - h))
        bottom = (d2 - h) * besselk(0, -(np.pi * n * r) / (d2 - h))
        return top / bottom


#############################################
# Equation 9:
def Z_n_i1(n):
    if n == 0:
        return 1
    else:
        return np.sqrt(2) * np.cos(lambda_n1(n) * (z + h))


def Z_n_i2(n):
    if n == 0:
        return 1
    else:
        return np.sqrt(2) * np.cos(lambda_n2(n) * (z + h))


#############################################
# Equation 13: (m_k is a function)
def Lambda_k_a2(k):
    if k == 0:
        return besselh(0, m0 * a2) / besselh(0, m0 * a2)
    else:
        return besselk(0, m_k(k) * a2) / besselk(0, m_k(k) * a2)


def diff_Lambda_k_a2(n):
    if n == 0:
        numerator = -(m0 * besselh(1, m0 * a2))
        denominator = besselh(0, a2 * m0)
    else:
        numerator = -(m_k(n) * besselk(1, a2 * m_k(n)))
        denominator = besselk(0, a2 * m_k(n))
    return numerator / denominator


#############################################
# Equation 2.34 in analytical methods book, also eq 16 in Seah and Yeung 2006:
def N_k(k):
    if k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * m_k(k) * h) / (2 * m_k(k) * h))


#############################################
# Equation 14: (m_k is a function)
def Z_k_e(k):
    if k == 0:
        return 1 / sqrt(N_k(k)) * cosh(m0 * (z + h))
    else:
        return 1 / sqrt(N_k(k)) * cos(m_k(k) * (z + h))


#############################################
# Coupling integrals: (m_k is a function)
def sq(n): 
    return n ** 2

def A_jn(j, n):
    if j == 0 and n == 0:
        return h - d1
    if j == 0 and 1 <= n:
        return (-sqrt(2) * sin(pi * n) * (d1 - h)) / (n * pi)
    sigma = (pi * j * (d1 - h)) / (d2 - h)
    if 1 <= j and n == 0:
        return (-sqrt(2) * sin(sigma) * (d2 - h)) / (j * pi)
    if 1 <= j and 1 <= n:
        top = -2 * (j * (d1 - h) * (d2 - h) * (d1 * sin(sigma) * cos(pi * n) - h * sin(sigma) * cos(pi * n)) - n * (d1 - h) * (d2 - h) * (d2 * sin(pi * n) * cos(sigma) - h * sin(pi * n) * cos(sigma)))
        bottom = pi * ((sq(d1) * sq(j)) - (2 * d1 * h * sq(j)) - (sq(d2) * sq(n)) + (2 * d2 * h * sq(n)) + (sq(h) * sq(j)) - (sq(h) * sq(n)))
        return top / bottom     
    else:
        raise ValueError("Invalid values for j and n")


def A_jn2(j, n):
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
        top = -2 * (j * (d1 - h) * (d2 - h) * (d2 * sin(pi * j) * cos(sigma) - h * sin(pi * j) * cos(sigma)) - n * (d1 - h) * (d2 - h) * (d2 * sin(sigma) * cos(pi * j) - h * sin(sigma) * cos(pi * j)))
        bottom = pi * ((-sq(d1) * sq(n)) + (2 * d1 * h * sq(n)) + (sq(d2) * sq(j)) - (2 * d2 * h * sq(j)) + (sq(h) * sq(j)) - (sq(h) * sq(n)))
        return top / bottom     
    else:
        raise ValueError("Invalid values for n and j")

def A_nj2(n, j):
    if j == 0 and n == 0:
        return h - d2
    sigma = (pi * j * (d2 - h)) / (d1 - h)
    if 1 <= j and n == 0:
        return (-sqrt(2) * sin(sigma) * (d1 - h)) / (j * pi)
    if j == 0 and 1 <= n:
        return (-sqrt(2) * sin(pi * n) * (d2 - h)) / (n * pi)
    if 1 <= j and 1 <= n:
        top = -2 * (j * (d1 - h) * (d2 - h) * (d2 * sin(sigma) * cos(pi * n) - h * sin(sigma) * cos(pi * n)) - n * (d1 - h) * (d2 - h) * (d1 * sin(pi * n) * cos(sigma) - h * sin(pi * n) * cos(sigma)))
        bottom = pi * ((-sq(d1) * sq(n)) + (2 * d1 * h * sq(n)) + (sq(d2) * sq(j)) - (2 * d2 * h * sq(j)) + (sq(h) * sq(j)) - (sq(h) * sq(n)))
        return top / bottom     
    else:
        raise ValueError("Invalid values for n and j")

def nk_sigma_helper(mk): 
    top = sin(2 * h * mk)
    bottom = 4 * h * mk
    sigma1 = sqrt(top/bottom + 1/2)
    sigma2 = sinh(m0 * (d2 - h)) 
    sigma3 = mk * (d2 - h)
    sigma4 = sq(pi) * sq(n)
    sigma5 = sinh(2 * h * m0)
    return sigma1, sigma2, sigma3, sigma4, sigma5

def A_nk(n, k):
    mk = m_k(k)
    sigma1, sigma2, sigma3, sigma4, sigma5 = nk_sigma_helper(mk)
    if k == 0 and n == 0:
        return (-2 * sqrt(h) * sigma2) / (sqrt(m0) * sqrt(sigma5 + 2 * h * m0))
    elif 1 <= k and n == 0:
        return -sin(sigma3) / (mk * sigma1)
    elif k == 0 and 1 <= n:
        top = -sqrt(2) * (m0 * (d2 * cos(pi * n) * sigma2 - h * cos(pi * n) * sigma2) * (d2 - h) + pi * n * sin(pi * n) * cosh(m0 * (d2 - h)) * (d2 - h))
        bottom = sqrt((sigma5 / (4 * h * m0)) + 1/2) * (sq(d2) * sq(m0) - 2 * d2 * h * sq(m0) + sq(h) * sq(m0) + sigma4)
        return top / bottom
    elif 1 <= k and 1 <= n:
        top = -sqrt(2) * (mk * (d2 * sin(sigma3) * cos(pi * n) - h * sin(sigma3) * cos(pi * n)) * (d2 - h) - pi * n * cos(sigma3) * sin(pi * n) * (d2 - h))
        bottom = sigma1 * (sq(d2) * sq(mk) - 2 * d2 * h * sq(mk) + sq(h) * sq(mk) - sigma4)
        return top / bottom
    else:
        raise ValueError("Invalid values for n and k")

def nk2_sigma_helper(mk): 
    top = sin(2 * h * mk)
    bottom = 4 * h * mk
    sigma1 = sqrt(top/bottom + 1/2)
    sigma2 = sin(h * mk) 
    sigma3 = (pi * h * n) / (d2 - h)
    sigma4 = sq(pi) * sq(n)
    sigma5 = sinh(2 * h * m0)
    return sigma1, sigma2, sigma3, sigma4, sigma5


def A_nk2(n, k):
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



potential_small_small = True
velocity_small_small = False
velocity_large_small = False

assert not (velocity_large_small and velocity_small_small)

# Potential matching
# Equation 22 in old 1981 paper, applied to boundary 2-e
dz_1 = h - d1
dz_2 = h - d2


'''
if potential_small_small
    match_2e_potential = dz_2 * (C_1n_2(n)*subs(R_1n_2(n),r,a2) ...
        + C_2n_2(n)*subs(R_2n_2(n),r,a2)) + int( subs(phi_p_i2,r,a2) * Z_n_i2(n), z, -h, -d2) == ...
        symsum( B_k(k) * subs(Lambda_k,r,a2) * C_nk(n,k), k, 0, N_num) %good
    var2 = vpa(match_2e_potential)
else
    match_2e_potential = symsum((C_1n_2(n) * subs(R_1n_2(n),r,a2) + ...
        C_2n_2(n) * subs(R_2n_2(n),r,a2)) * C_nk(n,k),n,0,N_num)  == ...
        B_k(k) * subs(Lambda_k(k),r,a2) * h - ...
        int(subs(phi_p_i2,r,a2) * Z_k_e(k), z, -h, 0)
end
'''


# if n == 0:
#     # First piece of the piecewise function
#     result = d2**2 / 6 - (C_1n_2(0) * (d2 - h)) / 2 - a2**2 / 4 - (d2 * h) / 3 + h**2 / 6
#     result += -sum([
#         (B_k(k) * np.sin(m_k(k) * (d2 - h))) / 
#         (m_k(k) * np.sqrt(np.sin(2 * h * m_k(k)) / (4 * h * m_k(k)) + 1/2))
#         for k in range(1, 7)
#     ])
#     result += -(2 * np.sqrt(h) * B_k(0) * np.sinh(m0 * (d2 - h))) / (np.sqrt(m0) * np.sqrt(np.sinh(2 * h * m0) + 2 * h * m0))
# else:
#     # Second piece of the piecewise function
#     result = (np.sqrt(2) * (d2 - h)**3 * (n**2 * np.pi**2 * np.sin(np.pi * n) - 2 * np.sin(np.pi * n) + 2 * np.pi * n * np.cos(np.pi * n))) / (n**3 * np.pi**3 * (2 * d2 - 2 * h))
#     result -= (d2 - h) * (C_1n_2(n) + C_2n_2(n))
#     result += (np.sqrt(2) * a2**2 * np.sin(np.pi * n) * (d2 - h)) / (2 * n * np.pi * (2 * d2 - 2 * h))
#     result += sum([
#         (np.sqrt(2) * B_k(k) * (
#             m_k(k) * (d2 * np.cos(np.pi * n) * np.sin(m_k(k) * (d2 - h)) - h * np.cos(np.pi * n) * np.sin(m_k(k) * (d2 - h))) * (d2 - h) -
#             np.pi * n * np.sin(np.pi * n) * np.cos(m_k(k) * (d2 - h)) * (d2 - h)
#         )) / (np.sqrt(np.sin(2 * h * m_k(k)) / (4 * h * m_k(k)) + 1/2) * (-m_k(k)**2 * d2**2 + 2 * m_k(k)**2 * d2 * h - m_k(k)**2 * h**2 + np.pi**2 * n**2))
#         for k in range(1, 7)
#     ])
#     result -= (np.sqrt(2) * B_k(0) * (
#         m0 * (d2 * np.cos(np.pi * n) * np.sinh(m0 * (d2 - h)) - h * np.cos(np.pi * n) * np.sinh(m0 * (d2 - h))) * (d2 - h) +
#         np.pi * n * np.sin(np.pi * n) * np.cosh(m0 * (d2 - h)) * (d2 - h)
#     )) / (np.sqrt(np.sinh(2 * h * m0) / (4 * h * m0) + 1/2) * (d2**2 * m0**2 - 2 * d2 * h * m0**2 + h**2 * m0**2 + np.pi**2 * n**2))

























# Helper function to perform numerical integration
def numerical_integrate(func, lower, upper):
    result, _ = sp.quad(func, lower, upper)
    return result

if potential_small_small:
    # Define the integrands and perform numerical integration and summation as needed
    integrand_2e = lambda z: phi_p_i2(a2, z) * Z_n_i2(n)
    integral_2e = numerical_integrate(integrand_2e, -h, -d2)
    
    summation_2e = sum(B_k(k) * Lambda_k(a2) * C_nk(n, k) for k in range(N_num + 1))
    
    match_2e_potential = dz_2 * (C_1n_2(n) * R_1n_2(n, a2) + C_2n_2(n) * R_2n_2(n, a2)) + integral_2e - summation_2e

    # Assuming `var2` is for evaluation/display
    var2 = match_2e_potential
else:
    summation_else_2e = sum((C_1n_2(n) * R_1n_2(n, a2) + C_2n_2(n) * R_2n_2(n, a2)) * C_nk(n, k) for n in range(N_num + 1))
    
    integrand_else_2e = lambda z: phi_p_i2(a2, z) * Z_k_e(k)
    integral_else_2e = numerical_integrate(integrand_else_2e, -h, 0)
    
    match_2e_potential = summation_else_2e - B_k(k) * Lambda_k(k, a2) * h + integral_else_2e

# The logic for match_12_potential follows a similar pattern
if potential_small_small:
    integrand_12 = lambda z: (phi_p_i2(a1, z) * Z_n_i1(n) - phi_p_i1(a1, z) * Z_n_i1(n))
    integral_12 = numerical_integrate(integrand_12, -h, -d1)
    
    summation_12 = sum((C_1n_2(j) * R_1n_2(j, a1) + C_2n_2(j) * R_2n_2(j, a1) * C_jn(j, n)) for j in range(N_num + 1))
    
    match_12_potential = C_1n_1(n) * R_1n_1(n, a1) * dz_1 - summation_12 + integral_12
else:
    integrand_else_12 = lambda z: (phi_p_i1(a1, z) * Z_n_i2(n) - phi_p_i2(a1, z) * Z_n_i2(n))
    integral_else_12 = numerical_integrate(integrand_else_12, -h, -d2)
    
    summation_else_12 = sum((C_1n_1(j) * R_1n_1(j, a1) + C_2n_1(j) * R_2n_1(j, a1) * C_nj2(n, j)) for j in range(N_num + 1))
    
    match_12_potential = (C_1n_2(n) * R_1n_2(n, a1) + C_2n_2(n) * R_2n_2(n, a1)) * dz_2 - summation_else_12 + integral_else_12
