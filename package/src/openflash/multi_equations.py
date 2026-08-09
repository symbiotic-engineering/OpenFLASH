#multi_equations.py
from openflash.multi_constants import g, rho
import numpy as np
from scipy.special import hankel1 as besselh
from scipy.special import i0 as besseli0
from scipy.special import i1 as besseli1
from scipy.special import k0 as besselk0
from scipy.special import k1 as besselk1
from scipy.special import i0e as besseli0e
from scipy.special import i1e as besseli1e
from scipy.special import k0e as besselk0e
from scipy.special import k1e as besselk1e
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
    if omega == inf: return inf
    else:
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
    m_k_initial_guess =  (m_k_h_upper + m_k_h_lower) / 2

    result = root_scalar(m_k_h_err, x0=m_k_initial_guess, method="brentq", bracket=[m_k_h_lower, m_k_h_upper])
    m_k_val = result.root / h
    return m_k_val

# create an array of m_k values for each k to avoid recomputation
def m_k(NMK, m0, h):
    func = np.vectorize(lambda k: m_k_entry(k, m0, h), otypes=[float])
    return func(range(NMK[-1]))

#############################################
# vertical eigenvector coupling computation

def _prepare_mode_grids(row_mode, col_mode):
    row_arr = np.asarray(row_mode)
    col_arr = np.asarray(col_mode)
    scalar_input = row_arr.ndim == 0 and col_arr.ndim == 0

    if scalar_input:
        return row_arr.reshape(1, 1), col_arr.reshape(1, 1), True

    if row_arr.ndim <= 1 and col_arr.ndim <= 1:
        row_grid, col_grid = np.meshgrid(row_arr.reshape(-1), col_arr.reshape(-1), indexing="ij")
    else:
        row_grid, col_grid = np.broadcast_arrays(row_arr, col_arr)

    return row_grid, col_grid, False


def I_nm(n, m, i, d, h): # coupling integral for two i-type regions
    dj = max(d[i], d[i+1]) # integration bounds at -h and -d
    n_grid, m_grid, scalar_input = _prepare_mode_grids(n, m)
    delta = h - dj
    out = np.zeros(n_grid.shape, dtype=float)

    mask_00 = (n_grid == 0) & (m_grid == 0)
    out[mask_00] = delta

    mask_n0_mpos = (n_grid == 0) & (m_grid >= 1)
    if np.any(mask_n0_mpos) and dj != d[i+1]:
        lambda2 = lambda_ni(m_grid[mask_n0_mpos], i + 1, h, d)
        out[mask_n0_mpos] = sqrt(2) * sin(lambda2 * delta) / lambda2

    mask_npos_m0 = (n_grid >= 1) & (m_grid == 0)
    if np.any(mask_npos_m0) and dj != d[i]:
        lambda1 = lambda_ni(n_grid[mask_npos_m0], i, h, d)
        out[mask_npos_m0] = sqrt(2) * sin(lambda1 * delta) / lambda1

    mask_general = ~(mask_00 | mask_n0_mpos | mask_npos_m0)
    if np.any(mask_general):
        lambda1 = lambda_ni(n_grid[mask_general], i, h, d)
        lambda2 = lambda_ni(m_grid[mask_general], i + 1, h, d)
        frac1 = sin((lambda1 + lambda2) * delta) / (lambda1 + lambda2)
        frac2 = np.empty_like(frac1, dtype=float)
        eq_mask = np.isclose(lambda1, lambda2)
        frac2[eq_mask] = delta
        neq_mask = ~eq_mask
        if np.any(neq_mask):
            frac2[neq_mask] = sin((lambda1[neq_mask] - lambda2[neq_mask]) * delta) / (lambda1[neq_mask] - lambda2[neq_mask])
        out[mask_general] = frac1 + frac2

    return out.item() if scalar_input else out

# REVISED I_mk to accept m_k_arr and N_k_arr
def I_mk(m, k, i, d, m0, h, m_k_arr, N_k_arr): # coupling integral for i and e-type regions
    m_grid, k_grid, scalar_input = _prepare_mode_grids(m, k)
    m_grid = m_grid.astype(int, copy=False)
    k_grid = k_grid.astype(int, copy=False)
    dj = d[i]
    delta = h - dj
    out = np.zeros(m_grid.shape, dtype=float)

    mask_m0_k0 = (m_grid == 0) & (k_grid == 0)
    if np.any(mask_m0_k0) and m0 != inf:
        if m0 * h < M0_H_THRESH:
            out[mask_m0_k0] = (1 / sqrt(N_k_arr[0])) * sinh(m0 * delta) / m0
        else:
            out[mask_m0_k0] = sqrt(2 * h / m0) * (exp(-m0 * dj) - exp(m0 * dj - 2 * m0 * h))

    mask_m0_kpos = (m_grid == 0) & (k_grid >= 1)
    if np.any(mask_m0_kpos):
        k_local = k_grid[mask_m0_kpos]
        local_mk = m_k_arr[k_local]
        out[mask_m0_kpos] = (1 / sqrt(N_k_arr[k_local])) * sin(local_mk * delta) / local_mk

    mask_mpos_k0 = (m_grid >= 1) & (k_grid == 0)
    if np.any(mask_mpos_k0) and m0 != inf:
        m_local = m_grid[mask_mpos_k0]
        if m0 * h < M0_H_THRESH:
            num = ((-1) ** m_local) * sqrt(2) * (1 / sqrt(N_k_arr[0])) * m0 * sinh(m0 * delta)
        else:
            num = ((-1) ** m_local) * 2 * sqrt(h * m0 ** 3) * (exp(-m0 * dj) - exp(m0 * dj - 2 * m0 * h))
        denom = m0**2 + lambda_ni(m_local, i, h, d) ** 2
        out[mask_mpos_k0] = num / denom

    mask_general = ~(mask_m0_k0 | mask_m0_kpos | mask_mpos_k0)
    if np.any(mask_general):
        m_local = m_grid[mask_general]
        k_local = k_grid[mask_general]
        lambda1 = lambda_ni(m_local, i, h, d)
        local_mk = m_k_arr[k_local]
        norm = sqrt(2 / N_k_arr[k_local]) / 2
        eq_mask = np.isclose(np.abs(local_mk), lambda1)
        general_vals = np.empty_like(lambda1, dtype=float)
        general_vals[eq_mask] = norm[eq_mask] * delta
        neq_mask = ~eq_mask
        if np.any(neq_mask):
            frac1 = sin((local_mk[neq_mask] + lambda1[neq_mask]) * delta) / (local_mk[neq_mask] + lambda1[neq_mask])
            frac2 = sin((local_mk[neq_mask] - lambda1[neq_mask]) * delta) / (local_mk[neq_mask] - lambda1[neq_mask])
            general_vals[neq_mask] = norm[neq_mask] * (frac1 + frac2)
        out[mask_general] = general_vals

    return out.item() if scalar_input else out

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
        bessel_term = (besseli0e(safe_lambda * r) / besseli0e(safe_lambda * scale(a)[i])) * \
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
        numerator = safe_lambda * besseli1e(safe_lambda * r) 
        denominator = besseli0e(safe_lambda * scale(a)[i])
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
        denom = besselk0e(lambda_safe * outer_r)
        # Direct division order
        bessel_term = (besselk0e(lambda_safe * r) / denom) * exp(lambda_safe * (outer_r - r))

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
        denom = besselk0e(lambda_safe * outer_r)
        numerator = -lambda_safe * besselk1e(lambda_safe * r)
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
        denom_k_nonzero = besselk0e(safe_m_k * scale(a)[-1])
        numer_k_nonzero = besselk0e(safe_m_k * r)
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
        numerator_k_nonzero = -(safe_m_k * besselk1e(safe_m_k * r))
        denominator_k_nonzero = besselk0e(safe_m_k * scale(a)[-1])
        outcome_k_nonzero = (numerator_k_nonzero / denominator_k_nonzero) * exp(safe_m_k * (scale(a)[-1] - r))

    return np.where(condition, outcome_k_zero, outcome_k_nonzero)

#############################################
# Equation 2.34 in analytical methods book, also eq 16 in Seah and Yeung 2006:

def N_k_multi(k, m0, h, m_k_arr): 
    if m0 == inf: return 1/2
    elif k == 0:
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
        inner = (0 if i == 0 else a[i-1]) # central region has inner radius 0
        return a[i]**2/4 - inner**2/4
    else:
        lambda0 = lambda_ni(n, i, h, d)
        bottom = lambda0 * besseli0e(lambda0 * scale(a)[i])
        if i == 0: inner_term = 0
        else: inner_term = (a[i-1] * besseli1e(lambda0 * a[i-1]) / bottom) * exp(lambda0 * (a[i-1] - scale(a)[i]))
        outer_term = (a[i] * besseli1e(lambda0 * a[i]) / bottom) * exp(lambda0 * (a[i] - scale(a)[i]))
        return outer_term - inner_term
    
#integrating R_2n * r
def int_R_2n(i, n, a, h, d):
    if i == 0:
        raise ValueError("i cannot be 0")
    
    lambda0 = lambda_ni(n, i, h, d)

    if n == 0:
        return (a[i-1]**2 * (2*np.log(a[i]/a[i-1]) + 1) - a[i]**2)/8
    else:
        outer_term = a[i] * besselk1e(lambda0 * a[i])
        inner_term = a[i-1] * besselk1e(lambda0 * a[i-1])
        bottom = - lambda0 * besselk0e(lambda0 * scale(a)[i])
        return (outer_term / bottom) * exp(lambda0 * (scale(a)[i] - a[i])) - (inner_term/bottom)* exp(lambda0 * (scale(a)[i] - a[i-1]))
    
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
    if m0 == inf: return -(pi/2)
    coeff = x[-NMK[-1]] # first coefficient of e-region expansion
    exterior_scale = scale(a)[-1]
    return -(pi/2) + np.angle(coeff) - np.angle(besselh(0, m0 * exterior_scale))

def excitation_force(damping, m0, h):
    const = np.tanh(m0 * h) + m0 * h * (1 - (np.tanh(m0 * h))**2)
    return ( (2 * const * rho * (g ** 2) * damping) / (omega(m0,h,g) * m0) ) ** (1/2)

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

def p_diagonal_block(left, radfunction, bd, h, d, a, NMK):
    region = bd if left else (bd + 1)
    sign = 1 if left else (-1)
    radial_vals = radfunction(list(range(NMK[region])), a[bd], region)
    block = sign * (h - d[region]) * np.diag(radial_vals)
    
    return block

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
    
    block = sign * radial_array * I_nm_array
    
    return block

def p_dense_block_e(bd, I_mk_vals, NMK, a, m0, m_k_arr):
    I_mk_array = I_mk_vals
    indices = np.arange(NMK[bd+1]) 
    radial_vector = Lambda_k_vectorized(indices, a[bd], m0, a, m_k_arr)
    radial_array = np.outer(np.ones(NMK[bd]), radial_vector)
    block = (-1) * radial_array * I_mk_array
        
    return block

def v_diagonal_block(left, radfunction, bd, h, d, NMK, a):
    region = bd if left else (bd + 1)
    sign = (-1) if left else (1)
    indices = np.arange(NMK[region])
    radial_vals = radfunction(indices, a[bd], region)
    block = sign * (h - d[region]) * np.diag(radial_vals)
    
    return block

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
    block = sign * radial_array * I_nm_array
    
    return block

def v_diagonal_block_e(bd, h, a, m0, m_k_arr, NMK):
    indices = np.arange(NMK[bd+1])
    vals = diff_Lambda_k_vectorized(indices, a[bd], m0, a, m_k_arr)
    block = h * np.diag(vals)
    
    return block

def v_dense_block_e(radfunction, bd, I_mk_vals, NMK, a): 
    I_km_array = np.transpose(I_mk_vals)
    indices = np.arange(NMK[bd])
    radial_vector = radfunction(indices, a[bd], bd)
    radial_array = np.outer(np.ones(NMK[bd + 1]), radial_vector)
    block = (-1) * radial_array * I_km_array
    
    return block
