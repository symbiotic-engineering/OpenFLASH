#multi_equations.py
from openflash.multi_constants import g, rho
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
from functools import partial


M0_H_THRESH=14

def omega(m0,h,g):
    if m0 == inf:
        return inf
    else:
        return sqrt(m0 * np.tanh(m0 * h) * g)
    
def wavenumber(omega, h):
    m0_err = (lambda m0: (m0 * np.tanh(h * m0) - omega ** 2 / g))
    return (root_scalar(m0_err, x0 = 2, method="newton")).root

def scale(a: list):
    return [val for val in a if val is not None]

def lambda_ni(n, i, h, d):  # Cap avoids Bessel overflow
    return n * pi / (h - d[i])

#############################################
# some common computations

# creating a m_k function, used often in calculations
def m_k_entry(k, m0, h):
    # print(f"m_k_entry called with k: {k}, m0: {m0}, h: {h}")
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
def m_k(NMK, m0, h):
    # print(f"m_k called with NMK: {NMK}, m0: {m0}, h: {h}")
    func = np.vectorize(lambda k: m_k_entry(k, m0, h), otypes=[float])
    return func(range(NMK[-1]))

def m_k_newton(h, m0):
    res = newton(lambda k: k * np.tanh(k * h) - m0**2 / 9.8, x0=1.0, tol=10 ** (-10))
    return res

#############################################
# vertical eigenvector coupling computation

def I_nm(n, m, i, d, h): # coupling integral for two i-type regions
    dj = max(d[i], d[i+1]) # integration bounds at -h and -d
    if n == 0 and m == 0:
        return h - dj
    lambda1 = lambda_ni(n, i, h, d)
    lambda2 = lambda_ni(m, i + 1, h, d)
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

# REVISED I_mk to accept m_k_arr and N_k_arr
def I_mk(m, k, i, d, m0, h, m_k_arr, N_k_arr): # coupling integral for i and e-type regions
    # Use the pre-computed array
    local_m_k_k = m_k_arr[k] # Access directly from array
    
    dj = d[i]
    if m == 0 and k == 0:
        if m0 == inf: return 0
        elif m0 * h < M0_H_THRESH:
            return (1/sqrt(N_k_arr[0])) * sinh(m0 * (h - dj)) / m0 # Use N_k_arr[0]
        else: # high m0h approximation
            return sqrt(2 * h / m0) * (exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
    if m == 0 and k >= 1:
        return (1/sqrt(N_k_arr[k])) * sin(local_m_k_k * (h - dj)) / local_m_k_k # Use N_k_arr[k]
    if m >= 1 and k == 0:
        if m0 == inf: return 0
        elif m0 * h < M0_H_THRESH:
            num = (-1)**m * sqrt(2) * (1/sqrt(N_k_arr[0])) * m0 * sinh(m0 * (h - dj)) # Use N_k_arr[0]
        else: # high m0h approximation
            num = (-1)**m * 2 * sqrt(h * m0 ** 3) *(exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
        denom = (m0**2 + lambda_ni(m, i, h, d) **2)
        return num/denom
    else:
        lambda1 = lambda_ni(m, i, h, d)
        if abs(local_m_k_k) == lambda1:
            return sqrt(2/N_k_arr[k]) * (h - dj)/2
        else:
            frac1 = sin((local_m_k_k + lambda1)*(h-dj))/(local_m_k_k + lambda1)
            frac2 = sin((local_m_k_k - lambda1)*(h-dj))/(local_m_k_k - lambda1)
            return sqrt(2/N_k_arr[k]) * (frac1 + frac2)/2 # Use N_k_arr[k]

#############################################
# b-vector computation

def b_potential_entry(n, i, d, heaving, h, a): # for two i-type regions
    #(integrate over shorter fluid, use shorter fluid eigenfunction)    
    j = i + (d[i] <= d[i+1]) # index of shorter fluid
    constant = (heaving[i+1] / (h - d[i+1]) - heaving[i] / (h - d[i]))
    if n == 0:
        return constant * 1/2 * ((h - d[j])**3/3 - (h-d[j]) * a[i]**2/2)
    else:
        return sqrt(2) * (h - d[j]) * constant * ((-1) ** n)/(lambda_ni(n, j, h, d) ** 2)

def b_potential_end_entry(n, i, heaving, h, d, a): # between i and e-type regions
    constant = - heaving[i] / (h - d[i])
    if n == 0:
        return constant * 1/2 * ((h - d[i])**3/3 - (h-d[i]) * a[i]**2/2)
    else:
        return sqrt(2) * (h - d[i]) * constant * ((-1) ** n)/(lambda_ni(n, i, h, d) ** 2)

def b_velocity_entry(n, i, heaving, a, h, d): # for two i-type regions
    if n == 0:
        return (heaving[i+1] - heaving[i]) * (a[i]/2)
    if d[i] > d[i + 1]: #using i+1's vertical eigenvectors
        if heaving[i]:
            num = - sqrt(2) * a[i] * sin(lambda_ni(n, i+1, h, d) * (h-d[i]))
            denom = (2 * (h - d[i]) * lambda_ni(n, i+1, h, d))
            return num/denom
        else: return 0
    else: #using i's vertical eigenvectors
        if heaving[i+1]:
            num = sqrt(2) * a[i] * sin(lambda_ni(n, i, h, d) * (h-d[i+1]))
            denom = (2 * (h - d[i+1]) * lambda_ni(n, i, h, d))
            return num/denom
        else: return 0

# REVISED b_velocity_end_entry to accept m_k_arr and N_k_arr
# ADDED m_k_arr, N_k_arr
def b_velocity_end_entry(k, i, heaving, a, h, d, m0, NMK, m_k_arr, N_k_arr): # between i and e-type regions
    local_m_k_k = m_k_arr[k] # Access directly from array

    constant = - heaving[i] * a[i]/(2 * (h - d[i]))
    if k == 0:
        if m0 * h < M0_H_THRESH:
            return constant * (1/sqrt(N_k_arr[0])) * sinh(m0 * (h - d[i])) / m0 # Use N_k_arr[0]
        else: # high m0h approximation
            return constant * sqrt(2 * h / m0) * (exp(- m0 * d[i]) - exp(m0 * d[i] - 2 * m0 * h))
    else:
        return constant * (1/sqrt(N_k_arr[k])) * sin(local_m_k_k * (h - d[i])) / local_m_k_k # Use N_k_arr[k]

def b_velocity_end_entry_full(k, i, heaving, a, h, d, m0, NMK): # between i and e-type regions
    local_m_k = m_k(NMK, m0, h)
    constant = - heaving[i] * a[i]/(2 * (h - d[i]))
    if k == 0:
        if m0 * h < M0_H_THRESH:
            return constant * (1/sqrt(N_k_full(0, m0, h, NMK))) * sinh(m0 * (h - d[i])) / m0
        else: # high m0h approximation
            return constant * sqrt(2 * h / m0) * (exp(- m0 * d[i]) - exp(m0 * d[i] - 2 * m0 * h))
    else:
        return constant * (1/sqrt(N_k_full(k, m0, h, NMK))) * sin(local_m_k[k] * (h - d[i])) / local_m_k[k]

#############################################
# Phi particular and partial derivatives

def phi_p_i(d, r, z, h): 
    return (1 / (2* (h - d))) * ((z + h) ** 2 - (r**2) / 2)

def diff_r_phi_p_i(d, r, h): 
    return (- r / (2* (h - d)))

def diff_z_phi_p_i(d, z, h): 
    return ((z+h) / (h - d))

#############################################
# The "Bessel I" radial eigenfunction
def R_1n(n, r, i, h, d, a):
    if n == 0:
        return 0.5
    elif n >= 1:
        if r == scale(a)[i]: # Saves bessel function eval
            return 1
        else:
            return besselie(0, lambda_ni(n, i, h, d) * r) / besselie(0, lambda_ni(n, i, h, d) * scale(a)[i]) * exp(lambda_ni(n, i, h, d) * (r - scale(a)[i]))
    else: 
        raise ValueError("Invalid value for n")

# Differentiate wrt r
def diff_R_1n(n, r, i, h, d, a):
    if n == 0:
        return 0
    else:
        top = lambda_ni(n, i, h, d) * besselie(1, lambda_ni(n, i, h, d) * r)
        bottom = besselie(0, lambda_ni(n, i, h, d) * scale(a)[i])
        return top / bottom * exp(lambda_ni(n, i, h, d) * (r - scale(a)[i]))

#############################################
# The "Bessel K" radial eigenfunction
def R_2n(n, r, i, a, h, d):
    if i == 0:
        raise ValueError("i cannot be 0")  # this shouldn't be called for i=0, innermost region.
    elif n == 0:
        return 0.5 * np.log(r / a[i])
    else:
        if r == scale(a)[i]:  # Saves bessel function eval
            return 1
        else:
            return besselke(0, lambda_ni(n, i, h, d) * r) / besselke(0, lambda_ni(n, i, h, d) * scale(a)[i]) * exp(lambda_ni(n, i, h, d) * (scale(a)[i] - r))

# Differentiate wrt r
def diff_R_2n(n, r, i, h, d, a):
    if n == 0:
        return 1 / (2 * r)
    else:
        top = - lambda_ni(n, i, h, d) * besselke(1, lambda_ni(n, i, h, d) * r)
        bottom = besselke(0, lambda_ni(n, i, h, d) * scale(a)[i])
        return top / bottom * exp(lambda_ni(n, i, h, d) * (scale(a)[i] - r))

#############################################
# i-region vertical eigenfunctions
def Z_n_i(n, z, i, h, d):
    if n == 0:
        return 1
    else:
        return np.sqrt(2) * np.cos(lambda_ni(n, i, h, d) * (z + h))
    
def Z_n_i_vectorized(n, z_array, i, h, d):
    if n == 0:
        return np.ones_like(z_array) # Return array of ones if z_array is passed
    else:
        return np.sqrt(2) * np.cos(lambda_ni(n, i, h, d) * (z_array + h))

def diff_Z_n_i(n, z, i, h, d):
    if n == 0:
        return 0
    else:
        lambda0 = lambda_ni(n, i, h, d)
        return - lambda0 * np.sqrt(2) * np.sin(lambda0 * (z + h))

#############################################
# Region e radial eigenfunction
# REVISED Lambda_k to accept m_k_arr and N_k_arr
def Lambda_k(k, r, m0, a, m_k_arr): # ADDED m_k_arr, N_k_arr
    local_m_k_k = m_k_arr[k]
    if k == 0:
        if m0 == inf:
        # the true limit is not well-defined, but whatever value this returns will be multiplied by zero
            return 1
        else:
            if r == scale(a)[-1]:  # Saves bessel function eval
                return 1
            else:
                return besselh(0, m0 * r) / besselh(0, m0 * scale(a)[-1])
    else:
        if r == scale(a)[-1]:  # Saves bessel function eval
            return 1
        else:
            return besselke(0, local_m_k_k * r) / besselke(0, local_m_k_k * scale(a)[-1]) * exp(local_m_k_k * (scale(a)[-1] - r))
    
def Lambda_k_vectorized(k, r_array, m0, a, NMK, h, m_k_arr, N_k_arr): # Changed 'r' to 'r_array'
    local_scale = scale(a)
    local_m_k_k = m_k_arr[k]

    if k == 0:
        return besselh(0, m0 * r_array) / besselh(0, m0 * local_scale[-1])
    else:
        return besselk(0, local_m_k_k * r_array) / besselk(0, local_m_k_k * local_scale[-1])
    
def Lambda_k_full(k, r, m0, a, NMK, h):
    local_scale = scale(a)
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        return besselh(0, m0 * r) / besselh(0, m0 * local_scale[-1])
    else:
        return besselk(0, local_m_k[k] * r) / besselk(0, local_m_k[k] * local_scale[-1])

# Differentiate wrt r 
def diff_Lambda_k(k, r, m0, a, m_k_arr): 
    local_m_k_k = m_k_arr[k] # Access directly from array
    if k == 0:
        if m0 == inf:
        # the true limit is not well-defined, but this makes the assigned coefficient zero
            return 1
        else:
            numerator = -(m0 * besselh(1, m0 * r))
            denominator = besselh(0, m0 * scale(a)[-1])
            # print(f"DEBUG: diff_Lambda_k (k={k}, r={r}, m0={m0})")
            # print(f"  m0*r: {m0*r}")
            # print(f"  numerator: {numerator}")
            # print(f"  denominator: {denominator}")
            # print(f"  result: {numerator / denominator}") 
            return numerator / denominator
    else:
        numerator = -(local_m_k_k * besselke(1, local_m_k_k * r))
        denominator = besselke(0, local_m_k_k * scale(a)[-1])
        return numerator / denominator * exp(local_m_k_k * (scale(a)[-1] - r))

def diff_Lambda_k_full(k, r, m0, NMK, h, a):
    local_m_k = m_k(NMK, m0, h)
    local_scale = scale(a)
    if k == 0:
        numerator = -(m0 * besselh(1, m0 * r))
        denominator = besselh(0, m0 * local_scale[-1])
    else:
        numerator = -(local_m_k[k] * besselk(1, local_m_k[k] * r))
        denominator = besselk(0, local_m_k[k] * local_scale[-1])
    return numerator / denominator


#############################################
# Equation 2.34 in analytical methods book, also eq 16 in Seah and Yeung 2006:
# REVISED N_k to accept m_k_arr (as it previously called m_k itself)

def N_k_multi(k, m0, h, m_k_arr): # Added m_k_arr as optional argument
    # print("N_k_multi called with k:", k, "m0:", m0, "h:", h)
    if m0 == inf: return 1/2
    elif k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * m_k_arr[k] * h) / (2 * m_k_arr[k] * h))
    
def N_k_full(k, m0, h, NMK):
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    elif m0 == 0:
        return 1.0
    else:
        return 1 / 2 * (1 + sin(2 * local_m_k[k] * h) / (2 * local_m_k[k] * h))


#############################################
# e-region vertical eigenfunctions
def Z_k_e(k, z, m0, h, NMK, m_k_arr):
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        if m0 * h < M0_H_THRESH:
            return 1 / sqrt(N_k_multi(k, m0, h, m_k_arr)) * cosh(m0 * (z + h))
        else: # high m0h approximation
            return sqrt(2 * m0 * h) * (exp(m0 * z) + exp(-m0 * (z + 2*h)))
    else:
        return 1 / sqrt(N_k_multi(k, m0, h, m_k_arr)) * cos(local_m_k[k] * (z + h))
    
def Z_k_e_vectorized(k, z_array, m0, h, NMK, m_k_arr): # Changed 'z' to 'z_array'
    local_m_k = m_k(NMK, m0, h) 
    if k == 0:
        if m0 * h < M0_H_THRESH:
            return 1 / sqrt(N_k_multi(k, m0, h, m_k_arr)) * cosh(m0 * (z_array + h))
        else: # high m0h approximation
            return sqrt(2 * m0 * h) * (exp(m0 * z_array) + exp(-m0 * (z_array + 2*h)))
    else:
        return 1 / sqrt(N_k_multi(k, m0, h, m_k_arr)) * cos(local_m_k[k] * (z_array + h))

def diff_Z_k_e(k, z, m0, h, NMK, m_k_arr):
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        if m0 * h < M0_H_THRESH:
            return 1 / sqrt(N_k_multi(k, m0, h, m_k_arr)) * m0 * sinh(m0 * (z + h))
        else: # high m0h approximation
            return m0 * sqrt(2 * h * m0) * (exp(m0 * z) - exp(-m0 * (z + 2*h)))
    else:
        return -1 / sqrt(N_k_multi(k, m0, h, m_k_arr)) * local_m_k[k] * sin(local_m_k[k] * (z + h))

#############################################
# To calculate hydrocoefficients

#integrating R_1n * r
def int_R_1n(i, n, a, h, d):
    if n == 0:
        inner = (0 if i == 0 else a[i-1]) # central region has inner radius 0
        return a[i]**2/4 - inner**2/4
    else:
        lambda0 = lambda_ni(n, i, h, d)
        bottom = lambda0 * besselie(0, lambda0 * scale(a)[i])
        if i == 0: inner_term = 0
        else: inner_term = (a[i-1] * besselie(1, lambda0 * a[i-1]) / bottom) * exp(lambda0 * (a[i-1] - scale(a)[i]))
        outer_term = (a[i] * besselie(1, lambda0 * a[i]) / bottom) * exp(lambda0 * (a[i] - scale(a)[i]))
        return outer_term - inner_term

#integrating R_2n * r
def int_R_2n(i, n, a, h, d):
    if i == 0:
        raise ValueError("i cannot be 0")
    lambda0 = lambda_ni(n, i, h, d)
    if n == 0:
        return (a[i-1]**2 * (2*np.log(a[i]/a[i-1]) + 1) - a[i]**2)/8
    else:
        outer_term = a[i] * besselke(1, lambda0 * a[i])
        inner_term = a[i-1] * besselke(1, lambda0 * a[i-1])
        bottom = - lambda0 * besselke(0, lambda0 * scale(a)[i])
        return (outer_term / bottom) * exp(lambda0 * (scale(a)[i] - a[i])) - (inner_term/bottom)* exp(lambda0 * (scale[i] - a[i-1]))

#integrating phi_p_i * d_phi_p_i/dz * r *d_r at z=d[i]
def int_phi_p_i(i, h, d, a):
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
def excitation_phase(x, NMK, m0, a): # x-vector of unknown coefficients
    coeff = x[-NMK[-1]] # first coefficient of e-region expansion
    local_scale = scale(a)
    return -(pi/2) + np.angle(coeff) - np.angle(besselh(0, m0 * local_scale[-1]))

def excitation_force(damping, m0, h):
    # Chau 2012 eq 98
    const = np.tanh(m0 * h) + m0 * h * (1 - (np.tanh(m0 * h))**2)
    # print("damping:", damping)
    # print("omega:", omega(m0, h, g))
    # print("m0:", m0)
    # print("expression under sqrt:", (2 * const * rho * (g ** 2) * damping)/(omega(m0,h,g) * m0))
    return sqrt((2 * const * rho * (g ** 2) * damping)/(omega(m0,h,g) * m0)) ** (1/2)

def make_R_Z(a, h, d, sharp, spatial_res): # create coordinate array for graphing
    rmin = (2 * a[-1] / spatial_res) if sharp else 0.0
    r_vec = np.linspace(rmin, 2*a[-1], spatial_res)
    z_vec = np.linspace(0, -h, spatial_res)
    if sharp: # more precise near boundaries
        a_eps = 1.0e-4
        for i in range(len(a)):
            r_vec = np.append(r_vec, a[i]*(1-a_eps))
            r_vec = np.append(r_vec, a[i]*(1+a_eps))
        r_vec = np.unique(r_vec)
        for i in range(len(d)):
            z_vec = np.append(z_vec, -d[i])
        z_vec = np.unique(z_vec)
    return np.meshgrid(r_vec, z_vec)

def p_diagonal_block(left, radfunction, bd, h, d, a, NMK):
    region = bd if left else (bd + 1)
    sign = 1 if left else (-1)
    return sign * (h - d[region]) * np.diag(radfunction(list(range(NMK[region])), a[bd], region))

def p_dense_block(left, radfunction, bd, NMK, a, I_nm_vals):
    I_nm_array = I_nm_vals[0:NMK[bd],0:NMK[bd+1], bd]
    if left: # determine which is region to work in and which is adjacent
        region, adj = bd, bd + 1
        sign = 1
        I_nm_array = np.transpose(I_nm_array)
    else:
        region, adj = bd + 1, bd
        sign = -1
    radial_vector = radfunction(list(range(NMK[region])), a[bd], region)
    radial_array = np.outer((np.full((NMK[adj]), 1)), radial_vector)
    return sign * radial_array * I_nm_array

def p_dense_block_e(bd, I_mk_vals, NMK, a):
    # print(f"p_dense_block_e new called with bd: {bd}, NMK: {NMK}, a: {a}")
    I_mk_array = I_mk_vals
    # print(f"I_mk_array shape: {I_mk_array.shape}")
    radial_vector = (np.vectorize(Lambda_k, otypes = [complex]))(list(range(NMK[bd+1])), a[bd])
    # print(f"radial_vector shape: {radial_vector.shape}")
    radial_array = np.outer((np.full((NMK[bd]), 1)), radial_vector)
    # print(f"radial_array shape: {radial_array.shape}")
    # print(f"Returning: {(-1) * radial_array * I_mk_array}")
    return (-1) * radial_array * I_mk_array
            
# arguments: diagonal block on left (T/F), vectorized radial eigenfunction, boundary number
def v_diagonal_block(left, radfunction, bd, h, d, NMK, a):
    region = bd if left else (bd + 1)
    sign = (-1) if left else (1)
    return sign * (h - d[region]) * np.diag(radfunction(list(range(NMK[region])), a[bd], region))

# arguments: dense block on left (T/F), vectorized radial eigenfunction, boundary number
def v_dense_block(left, radfunction, bd, I_nm_vals, NMK, a):
    I_nm_array = I_nm_vals[0:NMK[bd],0:NMK[bd+1], bd]
    if left: # determine which is region to work in and which is adjacent
        region, adj = bd, bd + 1
        sign = -1
        I_nm_array = np.transpose(I_nm_array)
    else:
        region, adj = bd + 1, bd
        sign = 1
    radial_vector = radfunction(list(range(NMK[region])), a[bd], region)
    radial_array = np.outer((np.full((NMK[adj]), 1)), radial_vector)
    return sign * radial_array * I_nm_array

def v_diagonal_block_e(bd, h, NMK, a, m0, m_k_arr): # Added m0, m_k_arr to signature
    # Create the vectorized version of diff_Lambda_k, specifically for this block's needs
    # This partial application ensures diff_Lambda_k has access to necessary fixed parameters
    # The 'k' and 'r' for diff_Lambda_k are provided by np.vectorize in the call below.
    vectorized_diff_Lambda_k_func = np.vectorize(
        partial(diff_Lambda_k, m0=m0, a=a, m_k_arr=m_k_arr),
        otypes=[complex]
    )
    
    # Calculate the diagonal elements by applying the vectorized function
    # 'a[bd]' is the fixed 'r' value for this boundary (radius)
    diagonal_elements = vectorized_diff_Lambda_k_func(list(range(NMK[bd+1])), a[bd]) # NMK[bd+1] is M

    # Create the diagonal matrix and ensure complex dtype
    return h * np.diag(diagonal_elements).astype(complex)

def v_dense_block_e(radfunction, bd, I_mk_vals, NMK, a): # for region adjacent to e-type region
    I_km_array = np.transpose(I_mk_vals)
    radial_vector = radfunction(list(range(NMK[bd])), a[bd], bd)
    radial_array = np.outer((np.full((NMK[bd + 1]), 1)), radial_vector)
    if bd == 2:
        print(f"DEBUG: v_dense_block_e NEW (bd={bd}")
        print(f"  I_km_array shape: {I_km_array.shape}") # Transposed!
        print(f"  radial_vector shape: {radial_vector.shape}") # For the k-th element
    return (-1) * radial_array * I_km_array

def p_dense_block_e_entry(m, k, bd, I_mk_vals, NMK, a, m0, h, m_k_arr, N_k_arr):
    """
    Compute individual entry (m, k) of the p_dense_block_e matrix at boundary `bd`.

    Parameters:
        m: int – row index (0 <= m < NMK[bd])
        k: int – col index (0 <= k < NMK[bd+1])
        bd: int – boundary index
        I_mk_vals: ndarray – array of shape (NMK[bd], NMK[bd+1]) with precomputed I_mk values
        NMK: list[int] – number of harmonics for each region
        a: list[float] – cylinder radii for each region

    Returns:
        complex – the matrix entry value
    """
    # print(f"p_dense_block_e_entry called with m: {m}, k: {k}, bd: {bd}, NMK: {NMK}, a: {a}")
    # print(f"returning: {-1 * Lambda_k(k, a[bd], m0, a, NMK, h, m_k_arr, N_k_arr) * I_mk_vals[m, k]}")
    return -1 * Lambda_k(k, a[bd], m0, a, m_k_arr) * I_mk_vals[m, k]

def v_dense_block_e_entry(m, k, bd, I_mk_vals, a, h, d): # Added h,d,NMK
    """
    Compute individual entry (m, k) of the v_dense_block_e matrix at boundary `bd`.
    """
    
    # In the old code's v_dense_block_e:
    # radial_vector = radfunction(list(range(NMK[bd])), a[bd], bd)
    # radfunction would be diff_R_1n_func or diff_R_2n_func
    # For a given (m,k) entry, 'k' corresponds to the n in diff_R_1n/diff_R_2n.
    # The 'r' argument for diff_R_1n/diff_R_2n is a[bd].
    # The 'i' argument for diff_R_1n/diff_R_2n is bd.

    # Determine which radial function to call based on the original logic
    # This might depend on 'bd' or other conditions that determine R_1n vs R_2n use
    # For the i-e boundary (bd == boundary_count - 1), typically R_1n is used for the inner region.
    radial_term = diff_R_1n(k, a[bd], bd, h, d, a) # Assuming diff_R_1n is the correct one for this block
    
    # I_mk_vals is correctly defined as (NMK[prev_region], NMK[current_region])
    # The old code used I_km_array = np.transpose(I_mk_vals) and then indexed it as I_km_array[m, k] (local row, local col)
    # So, if I_mk_vals is (rows_of_prev_region, cols_of_current_region),
    # then I_km_array[m, k] corresponds to I_mk_vals_untransposed[k, m]
    imk_term = I_mk_vals[k, m] # k is the 'col' index of current block, m is the 'row' index of current block

    # The outer sign is (-1) from v_dense_block_e
    result = -1 * radial_term * imk_term
    
    # Debug prints (uncomment if needed for further checks)
    # if m == 19 and k == 18 and bd == 2:
    #     print(f"DEBUG: v_dense_block_e_entry NEW (m={m}, k={k}, bd={bd})")
    #     print(f"  radial_term (diff_R_1n result): {radial_term}")
    #     print(f"  I_mk_vals[{k},{m}]: {imk_term}")
    #     print(f"  Result: {result}")

    return result

def v_diagonal_block_e_entry(m, k, bd, m0, m_k_arr, a, h):
    """
    Compute individual (m,k) entry of the velocity diagonal block e at boundary bd.
    """
    
    # You need access to 'h' here. It's available in the outer scope
    # through the problem object, or pass it directly.
    # Since NMK and a are passed, h should be too if it's not a global constant.
    # Let's assume you'll retrieve h from the problem object or pass it.
    
    # If h is always domain_list[0].h, you can pass it from build_problem_cache
    # For now, let's explicitly pass h from build_problem_cache to this function
    # via the closure.
    
    radius = a[bd]
    
    # Call diff_Lambda_k, ensure it's the correct new version from multi_equations.py
    # (it is, from your code snippet)
    val = diff_Lambda_k(k, radius, m0, a, m_k_arr)
 
    result =  h * val 
    
    return result

def v_dense_block_e_entry_R2(m: int, k: int, bd: int, I_mk_vals: np.ndarray, a: list, h: float, d: list) -> complex:
    """
    Computes a single entry for the m0-dependent velocity block at the i-e 
    boundary, using the R_2n radial eigenfunctions.

    This is used for the second part of the dense coupling block at the final
    boundary when there are more than two regions.

    Args:
        m (int): The local row index within the block, corresponding to a mode
                 in the external region.
        k (int): The local column index within the block, corresponding to a mode
                 'n' in the adjacent internal region.
        bd (int): The boundary index, which should be the final boundary.
        I_mk_vals (np.ndarray): The m0-dependent coupling integral matrix, I_mk.
        a (list): A list of the cylinder radii.
        h (float): The total water depth.
        d (list): A list of the draft depths for each region.

    Returns:
        complex: The computed complex value for the matrix entry A[row, col].
    """
    # 1. Calculate the radial term using the derivative of the R_2n function.
    #    - 'k' is the mode 'n' for the internal region.
    #    - 'a[bd]' is the radius 'r' at the boundary.
    #    - 'bd' is the region index 'i'.
    radial_term = diff_R_2n(k, a[bd], bd, h, d, a)
    
    # 2. Get the corresponding coupling integral value.
    #    The original implementation used a transposed I_mk matrix. An entry
    #    at [m, k] in the final block corresponds to the entry at [k, m]
    #    in the non-transposed I_mk_vals matrix.
    imk_term = I_mk_vals[k, m]

    # 3. Apply the leading sign and return the result.
    #    The physics of the problem formulation gives this block a -1 sign.
    return -1 * radial_term * imk_term