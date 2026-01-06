import numpy as np
from scipy.special import hankel1 as besselh
from scipy.special import iv as besseli
from scipy.special import kv as besselk
from scipy.special import ive as besselie
from scipy.special import kve as besselke
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from numpy import sqrt, cosh, cos, sinh, sin, pi, exp, inf
from scipy.optimize import newton, minimize_scalar, root_scalar
import scipy as sp
from multi_constants import *

def m0_to_omega(m0):
    if m0 == inf:
        return inf
    else:
        return sqrt(m0 * np.tanh(m0 * h) * g)

omega = m0_to_omega(m0)

def wavenumber(omega):
    m0_err = (lambda m0: (m0 * np.tanh(h * m0) - omega ** 2 / g))
    return (root_scalar(m0_err, x0 = 2, method="newton")).root

scale = a #np.append((np.mean([[0]+a[0:-1], a], axis = 0)), a[-1])

# After which the k = 0 e-region eigenfunction is well approximated by its limiting form.
# Empirically the true and approximating form differ by a fraction of < 1e-10 after this.
# The limiting form is used to prevent inf/inf errors.
LARGE_M0H = 14

def lambda_ni(n, i): # factor used often in calculations
    return n * pi / (h - d[i])

#############################################
# creating a m_k function, used often in calculations
def m_k_entry(k):
    if k == 0: return m0
    elif m0 == inf:
        return ((k - 1/2) * pi)/h

    m_k_h_err = (
        lambda m_k_h: (m_k_h * np.tan(m_k_h) + m0 * h * np.tanh(m0 * h))
    )
    k_idx = k

    # # original version of bounds in python
    m_k_h_lower = pi * (k_idx - 1/2) + np.finfo(float).eps
    m_k_h_upper = pi * k_idx - np.finfo(float).eps
    # x_0 =  (m_k_upper - m_k_lower) / 2
    
    # becca's version of bounds from MDOcean Matlab code
    m_k_h_lower = pi * (k_idx - 1/2) + (pi/180)* np.finfo(float).eps * (2**(np.floor(np.log(180*(k_idx- 1/2)) / np.log(2))) + 1)
    m_k_h_upper = pi * k_idx

    m_k_initial_guess = pi * (k_idx - 1/2) + np.finfo(float).eps
    result = root_scalar(m_k_h_err, x0=m_k_initial_guess, method="newton", bracket=[m_k_h_lower, m_k_h_upper])
    # result = minimize_scalar(
        # m_k_h_err, bounds=(m_k_h_lower, m_k_h_upper), method="bounded"
    # )

    m_k_val = result.root / h

    shouldnt_be_int = np.round(m0 * m_k_val / np.pi - 0.5, 4)
    # not_repeated = np.unique(m_k_val) == m_k_val
    assert np.all(shouldnt_be_int != np.floor(shouldnt_be_int))

        # m_k_mat[freq_idx, :] = m_k_vec
    return m_k_val

# create an array of m_k values for each k to avoid recomputation
m_k = (np.vectorize(m_k_entry, otypes = [float]))(list(range(NMK[-1])))

def m_k_newton(h):
    res = newton(lambda k: k * np.tanh(k * h) - m0**2 / 9.8, x0=1.0, tol=10 ** (-10))
    return res

#############################################
# vertical eigenvector coupling computation

def I_nm(n, m, i): # coupling integral for two i-type regions
    dj = max(d[i], d[i+1]) # integration bounds at -h and -d
    if n == 0 and m == 0:
        return h - dj
    lambda1 = lambda_ni(n, i)
    lambda2 = lambda_ni(m, i + 1)
    if n == 0 and m >= 1:
        if dj == d[i+1]:
            return 0
        else:
            return sqrt(2) * sin(lambda2 * (h - dj)) / lambda2
    if n >= 1 and m == 0:
        if dj == d[i]:
            return 0
        else:
            return sqrt(2) * sin(lambda1 * (h - dj)) / lambda1
    else:
        frac1 = sin((lambda1 + lambda2)*(h-dj))/(lambda1 + lambda2)
        if lambda1 == lambda2:
            frac2 = (h - dj)
        else:
            frac2 = sin((lambda1 - lambda2)*(h-dj))/(lambda1 - lambda2)
        return frac1 + frac2

def I_mk(m, k, i): # coupling integral for i and e-type regions
    dj = d[i]
    if m == 0 and k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            return (1/sqrt(N_k(0))) * sinh(m0 * (h - dj)) / m0
        else: # high m0h approximation
            return sqrt(2 * h / m0) * (exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
    if m == 0 and k >= 1:
        return (1/sqrt(N_k(k))) * sin(m_k[k] * (h - dj)) / m_k[k]
    if m >= 1 and k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            num = (-1)**m * sqrt(2) * (1/sqrt(N_k(0))) * m0 * sinh(m0 * (h - dj))
        else: # high m0h approximation
            num = (-1)**m * 2 * sqrt(h * m0 ** 3) *(exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
        denom = (m0**2 + lambda_ni(m, i) **2)
        return num/denom
    else:
        lambda1 = lambda_ni(m, i)
        if abs(m_k[k]) == lambda1:
            return sqrt(2/N_k(k)) * (h - dj)/2
        else:
            frac1 = sin((m_k[k] + lambda1)*(h-dj))/(m_k[k] + lambda1)
            frac2 = sin((m_k[k] - lambda1)*(h-dj))/(m_k[k] - lambda1)
            return sqrt(2/N_k(k)) * (frac1 + frac2)/2
            
#############################################
# b-vector computation

def b_potential_entry(n, i): # for two i-type regions
    #(integrate over shorter fluid, use shorter fluid eigenfunction)
    
    j = i + (d[i] <= d[i+1]) # index of shorter fluid
    constant = (heaving[i+1] / (h - d[i+1]) - heaving[i] / (h - d[i]))
    if n == 0:
        return constant * 1/2 * ((h - d[j])**3/3 - (h-d[j]) * a[i]**2/2)
    else:
        return sqrt(2) * (h - d[j]) * constant * ((-1) ** n)/(lambda_ni(n, j) ** 2)

def b_potential_end_entry(n, i): # between i and e-type regions
    constant = - heaving[i] / (h - d[i])
    if n == 0:
        return constant * 1/2 * ((h - d[i])**3/3 - (h-d[i]) * a[i]**2/2)
    else:
        return sqrt(2) * (h - d[i]) * constant * ((-1) ** n)/(lambda_ni(n, i) ** 2)

def b_velocity_entry(n, i): # for two i-type regions
    if n == 0:
        return (heaving[i+1] - heaving[i]) * (a[i]/2)
    if d[i] > d[i + 1]: #using i+1's vertical eigenvectors
        if heaving[i]:
            num = - sqrt(2) * a[i] * sin(lambda_ni(n, i+1) * (h-d[i]))
            denom = (2 * (h - d[i]) * lambda_ni(n, i+1))
            return num/denom
        else: return 0
    else: #using i's vertical eigenvectors
        if heaving[i+1]:
            num = sqrt(2) * a[i] * sin(lambda_ni(n, i) * (h-d[i+1]))
            denom = (2 * (h - d[i+1]) * lambda_ni(n, i))
            return num/denom
        else: return 0

def b_velocity_end_entry(k, i): # between i and e-type regions
    constant = - heaving[i] * a[i]/(2 * (h - d[i]))
    if k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            return constant * (1/sqrt(N_k(0))) * sinh(m0 * (h - d[i])) / m0
        else: # high m0h approximation
            return constant * sqrt(2 * h / m0) * (exp(- m0 * d[i]) - exp(m0 * d[i] - 2 * m0 * h))
    else:
        return constant * (1/sqrt(N_k(k))) * sin(m_k[k] * (h - d[i])) / m_k[k]

#############################################
# Phi particular and partial derivatives

def phi_p_i(d, r, z): 
    return (1 / (2* (h - d))) * ((z + h) ** 2 - (r**2) / 2)

def diff_r_phi_p_i(d, r, z): 
    return (- r / (2* (h - d)))

def diff_z_phi_p_i(d, r, z): 
    return ((z+h) / (h - d))

#############################################
# The "Bessel I" radial eigenfunction
def R_1n(n, r, i):
    if n == 0:
        return 0.5
    elif n >= 1:
        if r == scale[i]: # Saves bessel function eval
            return 1
        else:
            return besselie(0, lambda_ni(n, i) * r) / besselie(0, lambda_ni(n, i) * scale[i]) * exp(lambda_ni(n, i) * (r - scale[i]))
    else: 
        raise ValueError("Invalid value for n")

# Differentiate wrt r
def diff_R_1n(n, r, i):
    if n == 0:
        return 0
    else:
        top = lambda_ni(n, i) * besselie(1, lambda_ni(n, i) * r)
        bottom = besselie(0, lambda_ni(n, i) * scale[i])
        return top / bottom * exp(lambda_ni(n, i) * (r - scale[i]))

#############################################
# The "Bessel K" radial eigenfunction
def R_2n(n, r, i):
    if i == 0:
        raise ValueError("i cannot be 0")  # this shouldn't be called for i=0, innermost region.
    elif n == 0:
        return 0.5 * np.log(r / a[i])
    else:
        if r == scale[i]:  # Saves bessel function eval
            return 1
        else:
            return besselke(0, lambda_ni(n, i) * r) / besselke(0, lambda_ni(n, i) * scale[i]) * exp(lambda_ni(n, i) * (scale[i] - r))


# Differentiate wrt r
def diff_R_2n(n, r, i):
    if n == 0:
        return 1 / (2 * r)
    else:
        top = - lambda_ni(n, i) * besselke(1, lambda_ni(n, i) * r)
        bottom = besselke(0, lambda_ni(n, i) * scale[i])
        return top / bottom * exp(lambda_ni(n, i) * (scale[i] - r))


#############################################
# i-region vertical eigenfunctions
def Z_n_i(n, z, i):
    if n == 0:
        return 1
    else:
        return np.sqrt(2) * np.cos(lambda_ni(n, i) * (z + h))

def diff_Z_n_i(n, z, i):
    if n == 0:
        return 0
    else:
        lambda0 = lambda_ni(n, i)
        return - lambda0 * np.sqrt(2) * np.sin(lambda0 * (z + h))

#############################################
# Region e radial eigenfunction
def Lambda_k(k, r):
    if k == 0:
        if m0 == inf:
        # the true limit is not well-defined, but whatever value this returns will be multiplied by zero
            return 1
        else:
            if r == scale[-1]:  # Saves bessel function eval
                return 1
            else:
                return besselh(0, m0 * r) / besselh(0, m0 * scale[-1])
    else:
        if r == scale[-1]:  # Saves bessel function eval
            return 1
        else:
            return besselke(0, m_k[k] * r) / besselke(0, m_k[k] * scale[-1]) * exp(m_k[k] * (scale[-1] - r))

# Differentiate wrt r
def diff_Lambda_k(k, r):
    if k == 0:
        if m0 == inf:
        # the true limit is not well-defined, but this makes the assigned coefficient zero
            return 1
        else:
            numerator = -(m0 * besselh(1, m0 * r))
            denominator = besselh(0, m0 * scale[-1])
            return numerator / denominator
    else:
        numerator = -(m_k[k] * besselke(1, m_k[k] * r))
        denominator = besselke(0, m_k[k] * scale[-1])
        return numerator / denominator * exp(m_k[k] * (scale[-1] - r))


#############################################
# Equation 2.34 in analytical methods book, also eq 16 in Seah and Yeung 2006:
def N_k(k):
    if m0 == inf: return 1/2
    elif k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * m_k[k] * h) / (2 * m_k[k] * h))


#############################################
# e-region vertical eigenfunctions
def Z_k_e(k, z):
    if k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            return 1 / sqrt(N_k(k)) * cosh(m0 * (z + h))
        else: # high m0h approximation
            return sqrt(2 * m0 * h) * (exp(m0 * z) + exp(-m0 * (z + 2*h)))
    else:
        return 1 / sqrt(N_k(k)) * cos(m_k[k] * (z + h))

def diff_Z_k_e(k, z):
    if k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            return 1 / sqrt(N_k(k)) * m0 * sinh(m0 * (z + h))
        else: # high m0h approximation
            return m0 * sqrt(2 * h * m0) * (exp(m0 * z) - exp(-m0 * (z + 2*h)))
    else:
        return -1 / sqrt(N_k(k)) * m_k[k] * sin(m_k[k] * (z + h))

#############################################
# To calculate hydrocoefficients

#integrating R_1n * r
def int_R_1n(i, n):
    if n == 0:
        inner = (0 if i == 0 else a[i-1]) # central region has inner radius 0
        return a[i]**2/4 - inner**2/4
    else:
        lambda0 = lambda_ni(n, i)
        bottom = lambda0 * besselie(0, lambda0 * scale[i])
        if i == 0: inner_term = 0
        else: inner_term = (a[i-1] * besselie(1, lambda0 * a[i-1]) / bottom) * exp(lambda0 * (a[i-1] - scale[i]))
        outer_term = (a[i] * besselie(1, lambda0 * a[i]) / bottom) * exp(lambda0 * (a[i] - scale[i]))
        return outer_term - inner_term

#integrating R_2n * r
def int_R_2n(i, n):
    if i == 0:
        raise ValueError("i cannot be 0")
    lambda0 = lambda_ni(n, i)
    if n == 0:
        return (a[i-1]**2 * (2*np.log(a[i]/a[i-1]) + 1) - a[i]**2)/8
    else:
        outer_term = a[i] * besselke(1, lambda0 * a[i])
        inner_term = a[i-1] * besselke(1, lambda0 * a[i-1])
        bottom = - lambda0 * besselke(0, lambda0 * scale[i])
        return (outer_term / bottom) * exp(lambda0 * (scale[i] - a[i])) - (inner_term/bottom)* exp(lambda0 * (scale[i] - a[i-1]))

#integrating phi_p_i * d_phi_p_i/dz * r *d_r at z=d[i]
def int_phi_p_i(i):
    denom = 16 * (h - d[i])
    if i == 0:
        num = a[i]**2*(4*(h-d[i])**2-a[i]**2)
    else:
        num = (a[i]**2*(4*(h-d[i])**2-a[i]**2) - a[i-1]**2*(4*(h-d[i])**2-a[i-1]**2))
    return num/denom

# evaluate an interior region vertical eigenfunction at its top boundary
def z_n_d(n):
    if n ==0:
        return 1
    else:
        return sqrt(2)*(-1)**n

#############################################
def excitation_phase(x): # x-vector of unknown coefficients
    coeff = x[-NMK[-1]] # first coefficient of e-region expansion
    return -(pi/2) + np.angle(coeff) - np.angle(besselh(0, m0 * scale[-1]))
    
def excitation_force(damping):
    # Chau 2012 eq 98
    const = np.tanh(m0 * h) + m0 * h * (1 - (np.tanh(m0 * h))**2)
    return ( (2 * const * rho * (g ** 2) * damping) / (omega * m0) ) ** (1/2)