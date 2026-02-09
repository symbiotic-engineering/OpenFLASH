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
from numpy import sqrt, cosh, cos, sinh, sin, pi, exp, inf, log
from scipy.optimize import newton, minimize_scalar, root_scalar
import scipy as sp
from functools import partial
from typing import Optional

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
    return [val for val in a if val not in (None, np.inf, float('inf'))]

def lambda_ni(n, i, h, d):  # Cap avoids Bessel overflow
    return n * pi / (h - d[i])

#############################################
# some common computations

# creating a m_k function, used often in calculations
def m_k_entry(k, m0, h):
    if k == 0: 
        print(">>> USING UPDATED m_k_entry WITH NEWTON SOLVER <<<")
        return m0
    elif m0 == inf:
        return ((k - 1/2) * pi)/h

    m_k_h_err = (lambda m_k_h: (m_k_h * np.tan(m_k_h) + m0 * h * np.tanh(m0 * h)))
    k_idx = k

    # Legacy code used standard float bounds logic which we mimic here by using
    # the newton method which was used in old_assembly.py.
    # While brentq is safer, newton reproduces the exact matrix values.
    
    m_k_h_lower = np.nextafter(pi * (k_idx - 1/2), np.inf)
    m_k_h_upper = np.nextafter(pi * k_idx, np.inf)
    m_k_initial_guess = pi * (k_idx - 1/2) + np.finfo(float).eps
    
    # Matching old_assembly.py: method="newton"
    result = root_scalar(m_k_h_err, x0=m_k_initial_guess, method="newton", bracket=[m_k_h_lower, m_k_h_upper])
    m_k_val = result.root / h
    return m_k_val

# create an array of m_k values for each k to avoid recomputation
def m_k(NMK, m0, h):
    func = np.vectorize(lambda k: m_k_entry(k, m0, h), otypes=[float])
    return func(range(NMK[-1]))

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
    constant = (float(heaving[i+1]) / (h - d[i+1]) - float(heaving[i]) / (h - d[i]))
    if n == 0:
        return constant * 1/2 * ((h - d[j])**3/3 - (h-d[j]) * a[i]**2/2)
    else:
        return sqrt(2) * (h - d[j]) * constant * ((-1) ** n)/(lambda_ni(n, j, h, d) ** 2)

def b_potential_end_entry(n, i, heaving, h, d, a): # between i and e-type regions
    constant = - float(heaving[i]) / (h - d[i])
    if n == 0:
        return constant * 1/2 * ((h - d[i])**3/3 - (h-d[i]) * a[i]**2/2)
    else:
        return sqrt(2) * (h - d[i]) * constant * ((-1) ** n)/(lambda_ni(n, i, h, d) ** 2)

def b_velocity_entry(n, i, heaving, a, h, d): # for two i-type regions
    if n == 0:
        return (float(heaving[i+1]) - float(heaving[i])) * (a[i]/2)
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

    constant = - float(heaving[i]) * a[i]/(2 * (h - d[i]))
    if k == 0:
        if m0 == inf:
            return 0.0
        elif m0 * h < M0_H_THRESH:
            return constant * (1/sqrt(N_k_arr[0])) * sinh(m0 * (h - d[i])) / m0 # Use N_k_arr[0]
        else: # high m0h approximation
            return constant * sqrt(2 * h / m0) * (exp(- m0 * d[i]) - exp(m0 * d[i] - 2 * m0 * h))
    else:
        return constant * (1/sqrt(N_k_arr[k])) * sin(local_m_k_k * (h - d[i])) / local_m_k_k # Use N_k_arr[k]
    
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
#############################################

def R_1n_vectorized(n, r, i, h, d, a):
    """
    Vectorized R_1n. 
    """
    n = np.asarray(n, dtype=float)
    r = np.asarray(r, dtype=float)

    cond_n_is_zero = (n == 0)
    
    # n=0 case
    outcome_for_n_zero = np.full_like(r, 0.5)
    
    # n>=1 cases
    lambda_val = lambda_ni(n, i, h, d)
    cond_r_at_boundary = (r == scale(a)[i])
    
    # Safety against n=0
    safe_lambda = np.where(cond_n_is_zero, 1.0, lambda_val)
    
    # Use direct division with errstate to match exact arithmetic order of old code
    with np.errstate(divide='ignore', invalid='ignore'):
        bessel_term = (besselie(0, safe_lambda * r) / besselie(0, safe_lambda * scale(a)[i])) * \
                      exp(safe_lambda * (r - scale(a)[i]))

    result_if_n_not_zero = np.where(cond_r_at_boundary, 1.0, bessel_term)
    
    return np.where(cond_n_is_zero, outcome_for_n_zero, result_if_n_not_zero)

def diff_R_1n_vectorized(n, r, i, h, d, a):
    """
    Vectorized derivative. 
    """
    n = np.asarray(n, dtype=float)
    r = np.asarray(r, dtype=float)
    
    condition_n_is_zero = (n == 0)
    
    value_if_true = np.zeros_like(r)
    
    lambda_val = lambda_ni(n, i, h, d)
    safe_lambda = np.where(condition_n_is_zero, 1.0, lambda_val)
    
    # Use standard division logic to match old_assembly.py arithmetic
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = safe_lambda * besselie(1, safe_lambda * r) 
        denominator = besselie(0, safe_lambda * scale(a)[i])
        # Direct division matches: top / bottom * exp(...)
        value_if_false = (numerator / denominator) * exp(safe_lambda * (r - scale(a)[i]))
    
    return np.where(condition_n_is_zero, value_if_true, value_if_false)

#############################################
# The "Bessel K" radial eigenfunction (Annular Regions)
#############################################
def R_2n_vectorized(n, r, i, a, h, d):
    """
    Vectorized version of the R_2n radial eigenfunction.
    """
    if i == 0:
        raise ValueError("R_2n function is not defined for the innermost region (i=0).")
    
    n = np.asarray(n, dtype=float)
    r = np.asarray(r, dtype=float)

    outer_r = scale(a)[i]

    cond_n_is_zero = (n == 0)
    cond_r_at_boundary = (r == outer_r)

    # Case 1: n = 0
    with np.errstate(divide='ignore', invalid='ignore'):
         outcome_for_n_zero = 0.5 * np.log(np.divide(r, outer_r))

    # Case 2: n > 0 and r is at the boundary
    outcome_for_r_boundary = 1.0

    # Case 3: n > 0
    lambda_val = lambda_ni(n, i, h, d)
    lambda_safe = np.where(cond_n_is_zero, 1.0, lambda_val)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = besselke(0, lambda_safe * outer_r)
        # Direct division order
        bessel_term = (besselke(0, lambda_safe * r) / denom) * exp(lambda_safe * (outer_r - r))

    result_if_n_not_zero = np.where(cond_r_at_boundary, outcome_for_r_boundary, bessel_term)

    return np.where(cond_n_is_zero, outcome_for_n_zero, result_if_n_not_zero)

def diff_R_2n_vectorized(n, r, i, h, d, a):
    n = np.asarray(n, dtype=float)
    r = np.asarray(r, dtype=float)
    
    value_if_true = np.divide(1.0, 2 * r, out=np.full_like(r, np.inf), where=(r != 0))
    
    lambda_val = lambda_ni(n, i, h, d)
    outer_r = scale(a)[i]
    lambda_safe = np.where(n == 0, 1.0, lambda_val)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = besselke(0, lambda_safe * outer_r)
        numerator = -lambda_safe * besselke(1, lambda_safe * r)
        # Match arithmetic: top / bottom * exp(...)
        value_if_false = (numerator / denom) * exp(lambda_safe * (outer_r - r))

    return np.where(n == 0, value_if_true, value_if_false)

#############################################
# i-region vertical eigenfunctions
def Z_n_i_vectorized(n, z, i, h, d):
    n = np.asarray(n, dtype=float)
    z = np.asarray(z, dtype=float)
    condition = (n == 0)
    lambda_val = lambda_ni(n, i, h, d)
    safe_lambda = np.where(condition, 0.0, lambda_val)
    value_if_false = np.sqrt(2) * np.cos(safe_lambda * (z + h))
    return np.where(condition, 1.0, value_if_false)

def diff_Z_n_i_vectorized(n, z, i, h, d):
    n = np.asarray(n, dtype=float)
    z = np.asarray(z, dtype=float)
    condition = (n == 0)
    value_if_true = 0.0
    lambda_val = lambda_ni(n, i, h, d)
    safe_lambda = np.where(condition, 0.0, lambda_val)
    value_if_false = -safe_lambda * np.sqrt(2) * np.sin(safe_lambda * (z + h))
    return np.where(condition, value_if_true, value_if_false)

#############################################
# Region e radial eigenfunction    
def Lambda_k_vectorized(k, r, m0, a, m_k_arr):
    k = np.asarray(k, dtype=float)
    r = np.asarray(r, dtype=float)

    cond_k_is_zero = (k == 0)
    cond_r_at_boundary = (r == scale(a)[-1])
    outcome_boundary = 1.0

    if m0 == inf:
        outcome_k_zero = np.ones_like(r, dtype=float) 
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            denom_k_zero = besselh(0, m0 * scale(a)[-1])
            numer_k_zero = besselh(0, m0 * r)
            outcome_k_zero = numer_k_zero / denom_k_zero

    k_int = k.astype(int)
    local_m_k_k = m_k_arr[k_int]
    safe_m_k = np.where(cond_k_is_zero, 1.0, local_m_k_k)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_k_nonzero = besselke(0, safe_m_k * scale(a)[-1])
        numer_k_nonzero = besselke(0, safe_m_k * r)
        outcome_k_nonzero = (numer_k_nonzero / denom_k_nonzero) * exp(safe_m_k * (scale(a)[-1] - r))

    result_if_not_boundary = np.where(cond_k_is_zero, outcome_k_zero, outcome_k_nonzero)

    return np.where(cond_r_at_boundary, outcome_boundary, result_if_not_boundary)

# Differentiate wrt r 
def diff_Lambda_k_vectorized(k, r, m0, a, m_k_arr):
    k = np.asarray(k, dtype=float)
    r = np.asarray(r, dtype=float)
    condition = (k == 0)

    if m0 == inf:
        outcome_k_zero = np.ones_like(r, dtype=float)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator_k_zero = -(m0 * besselh(1, m0 * r))
            denominator_k_zero = besselh(0, m0 * scale(a)[-1])
            outcome_k_zero = numerator_k_zero / denominator_k_zero

    k_int = k.astype(int)
    local_m_k_k = m_k_arr[k_int]
    safe_m_k = np.where(condition, 1.0, local_m_k_k)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator_k_nonzero = -(safe_m_k * besselke(1, safe_m_k * r))
        denominator_k_nonzero = besselke(0, safe_m_k * scale(a)[-1])
        outcome_k_nonzero = (numerator_k_nonzero / denominator_k_nonzero) * exp(safe_m_k * (scale(a)[-1] - r))

    return np.where(condition, outcome_k_zero, outcome_k_nonzero)

#############################################
# Equation 2.34 in analytical methods book, also eq 16 in Seah and Yeung 2006:

def N_k_multi(k, m0, h, m_k_arr): 
    if m0 == inf: return 1/2
    elif k == 0:
        if (2 * m0 * h) > 700: 
            return 1e308 
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * m_k_arr[k] * h) / (2 * m_k_arr[k] * h))

#############################################
# e-region vertical eigenfunctions
def Z_k_e_vectorized(k, z, m0, h, m_k_arr, N_k_arr):
    k = np.asarray(k, dtype=float)
    z = np.asarray(z, dtype=float)
    
    if m0 * h < M0_H_THRESH:
        outcome_k_zero = (1 / sqrt(N_k_arr[0])) * cosh(m0 * (z + h))
        k_int = k.astype(int)
        outcome_k_nonzero = (1 / sqrt(N_k_arr[k_int])) * cos(m_k_arr[k_int] * (z + h))
        return np.where(k == 0, outcome_k_zero, outcome_k_nonzero)
    else:
        outcome_k_zero = sqrt(2 * m0 * h) * (exp(m0 * z) + exp(-m0 * (z + 2 * h)))
        k_int = k.astype(int)
        outcome_k_nonzero = (1 / sqrt(N_k_arr[k_int])) * cos(m_k_arr[k_int] * (z + h))
        return np.where(k == 0, outcome_k_zero, outcome_k_nonzero)

def diff_Z_k_e_vectorized(k, z, m0, h, m_k_arr, N_k_arr):
    k = np.asarray(k, dtype=float)
    z = np.asarray(z, dtype=float)

    if m0 * h < M0_H_THRESH:
        outcome_k_zero = (1 / sqrt(N_k_arr[0])) * m0 * sinh(m0 * (z + h))
        k_int = k.astype(int)
        outcome_k_nonzero = -(1 / sqrt(N_k_arr[k_int])) * m_k_arr[k_int] * sin(m_k_arr[k_int] * (z + h))
        return np.where(k == 0, outcome_k_zero, outcome_k_nonzero)
    else:
        outcome_k_zero = m0 * sqrt(2 * h * m0) * (exp(m0 * z) - exp(-m0 * (z + 2 * h)))
        k_int = k.astype(int)
        outcome_k_nonzero = -(1 / sqrt(N_k_arr[k_int])) * m_k_arr[k_int] * sin(m_k_arr[k_int] * (z + h))
        return np.where(k == 0, outcome_k_zero, outcome_k_nonzero)

#############################################
# To calculate hydrocoefficients

#integrating R_1n * r
# Integration
def int_R_1n(i, n, a, h, d):
    if n == 0:
        if i == 0:
            return a[i]**2/4 
        else:
            outer_r = scale(a)[i]
            inner_r = a[i-1]
            cyl_term = (outer_r**2 - inner_r**2) / 2.0
            def log_indefinite_int(r):
                log_val = np.log(r/outer_r) if r > 0 else 0
                return 0.5 * ((r**2 / 2.0) * log_val - (r**2 / 4.0))
            val_outer = log_indefinite_int(outer_r)
            val_inner = log_indefinite_int(inner_r)
            return cyl_term + (val_outer - val_inner)
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
    
    outer_r = scale(a)[i]
    inner_r = a[i-1] 

    if n == 0:
        def indefinite(r):
            if r <= 0: return 0
            term_log = np.log(r / outer_r)
            return 0.5 * ((r**2 / 2) * term_log - (r**2 / 4))
        val_outer = indefinite(outer_r)
        val_inner = indefinite(inner_r)
        return val_outer - val_inner

    else:
        lambda0 = lambda_ni(n, i, h, d)
        term_outer = (outer_r * besselke(1, lambda0 * outer_r)) 
        term_inner = (inner_r * besselke(1, lambda0 * inner_r))
        term_inner *= np.exp(lambda0 * (outer_r - inner_r))
        denom = lambda0 * besselke(0, lambda0 * outer_r)
        return (term_inner - term_outer) / denom
    
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
def excitation_phase(x, NMK, m0, a): 
    coeff = x[-NMK[-1]] 
    local_scale = scale(a)
    return -(pi/2) + np.angle(coeff) - np.angle(besselh(0, m0 * local_scale[-1]))

def excitation_force(damping, m0, h):
    if m0 == inf:
        return 0.0
    const = np.tanh(m0 * h) + m0 * h * (1 - (np.tanh(m0 * h))**2)
    return sqrt((2 * const * rho * (g ** 2) * damping)/(omega(m0,h,g) * m0)) ** (1/2)

# --- AFTER ---
def make_R_Z(a, h, d, sharp, spatial_res, R_range: Optional[np.ndarray] = None, Z_range: Optional[np.ndarray] = None):
    
    if R_range is not None:
        r_vec = R_range
    else:
        rmin = (2 * a[-1] / spatial_res) if sharp else 0.0
        r_vec = np.linspace(rmin, 2*a[-1], spatial_res)

    if Z_range is not None:
        z_vec = Z_range
    else:
        z_vec = np.linspace(0, -h, spatial_res) 

    if sharp: 
        a_eps = 1.0e-4
        for i in range(len(a)):
            r_vec = np.append(r_vec, a[i]*(1-a_eps))
            r_vec = np.append(r_vec, a[i]*(1+a_eps))
        r_vec = np.unique(r_vec)
        for i in range(len(d)):
            z_vec = np.append(z_vec, -d[i])
        z_vec = np.unique(z_vec)
    
    return np.meshgrid(r_vec, z_vec, indexing='ij')

def p_dense_block_e_entry(m, k, bd, I_mk_vals, NMK, a, m0, h, m_k_arr, N_k_arr):
    radial_val = Lambda_k_vectorized(k, a[bd], m0, a, m_k_arr)
    return -1 * radial_val * I_mk_vals[m, k]

def v_dense_block_e_entry(m, k, bd, I_mk_vals, a, h, d):
    radial_term = diff_R_1n_vectorized(k, a[bd], bd, h, d, a)
    imk_term = I_mk_vals[k, m]
    return -1 * radial_term * imk_term

def v_diagonal_block_e_entry(m, k, bd, m0, m_k_arr, a, h):
    radius = a[bd]
    val = diff_Lambda_k_vectorized(k, radius, m0, a, m_k_arr)
    return h * val

def v_dense_block_e_entry_R2(m, k, bd, I_mk_vals, a, h, d):
    radial_term = diff_R_2n_vectorized(k, a[bd], bd, h, d, a)
    imk_term = I_mk_vals[k, m]
    return -1 * radial_term * imk_term

def p_diagonal_block(left, radfunction, bd, h, d, a, NMK):
    region = bd if left else (bd + 1)
    sign = 1 if left else (-1)
    radial_vals = radfunction(list(range(NMK[region])), a[bd], region)
    return sign * (h - d[region]) * np.diag(radial_vals)

def p_dense_block(left, radfunction, bd, NMK, a, I_nm_vals_bd):
    I_nm_array = I_nm_vals_bd
    if left: 
        region, adj = bd, bd + 1
        sign = 1
        I_nm_array = np.transpose(I_nm_array)
    else:
        region, adj = bd + 1, bd
        sign = -1
    
    indices = np.arange(NMK[region])
    radial_vector = radfunction(indices, a[bd], region)
    radial_array = np.outer(np.ones(NMK[adj]), radial_vector)
    
    return sign * radial_array * I_nm_array

def p_dense_block_e(bd, I_mk_vals, NMK, a, m0, m_k_arr):
    I_mk_array = I_mk_vals
    indices = np.arange(NMK[bd+1]) 
    radial_vector = Lambda_k_vectorized(indices, a[bd], m0, a, m_k_arr)
    radial_array = np.outer(np.ones(NMK[bd]), radial_vector)
    return (-1) * radial_array * I_mk_array

def v_diagonal_block(left, radfunction, bd, h, d, NMK, a):
    region = bd if left else (bd + 1)
    sign = (-1) if left else (1)
    indices = np.arange(NMK[region])
    radial_vals = radfunction(indices, a[bd], region)
    return sign * (h - d[region]) * np.diag(radial_vals)

def v_dense_block(left, radfunction, bd, NMK, a, I_nm_vals_bd):
    I_nm_array = I_nm_vals_bd
    if left: 
        region, adj = bd, bd + 1
        sign = -1
        I_nm_array = np.transpose(I_nm_array)
    else:
        region, adj = bd + 1, bd
        sign = 1
    indices = np.arange(NMK[region])
    radial_vector = radfunction(indices, a[bd], region)
    radial_array = np.outer(np.ones(NMK[adj]), radial_vector)
    return sign * radial_array * I_nm_array

def v_diagonal_block_e(bd, h, a, m0, m_k_arr, NMK):
    indices = np.arange(NMK[bd+1])
    vals = diff_Lambda_k_vectorized(indices, a[bd], m0, a, m_k_arr)
    return h * np.diag(vals)

def v_dense_block_e(radfunction, bd, I_mk_vals, NMK, a): 
    I_km_array = np.transpose(I_mk_vals)
    indices = np.arange(NMK[bd])
    radial_vector = radfunction(indices, a[bd], bd)
    radial_array = np.outer(np.ones(NMK[bd + 1]), radial_vector)
    return (-1) * radial_array * I_km_array