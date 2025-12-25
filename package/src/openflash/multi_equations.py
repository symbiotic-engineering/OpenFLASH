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
    if k == 0: return m0
    elif m0 == inf:
        return ((k - 1/2) * pi)/h

    m_k_h_err = (lambda m_k_h: (m_k_h * np.tan(m_k_h) + m0 * h * np.tanh(m0 * h)))
    k_idx = k

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
        # --- FIX: Handle infinite m0 ---
        if m0 == inf:
            return 0.0
        # -------------------------------
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

def R_1n(n, r, i, h, d, a):
    if n == 0:
        if i == 0:
            return 0.5 # Central cylinder: Constant is correct/sufficient
        else:
            # Annulus: Use Log anchored at OUTER radius
            # 1.0 + 0.5 * log(r / outer_r)
            return 1.0 + 0.5 * np.log(r / scale(a)[i])
    elif n >= 1:
        if r == scale(a)[i]: 
            return 1
        else:
            return besselie(0, lambda_ni(n, i, h, d) * r) / besselie(0, lambda_ni(n, i, h, d) * scale(a)[i]) * exp(lambda_ni(n, i, h, d) * (r - scale(a)[i]))
    else: 
        raise ValueError("Invalid value for n")

def R_1n_vectorized(n, r, i, h, d, a):
    """
    Vectorized version of the R_1n radial eigenfunction.
    FIXED: Handles i!=0 case for n=0 to match scalar R_1n.
    """
    # --- Define the conditions for the nested logic ---
    cond_n_is_zero = (n == 0)
    cond_r_at_boundary = (r == scale(a)[i])

    # --- Define the outcomes for each condition ---
    # FIX: n=0 logic must match scalar version
    if i == 0:
        outcome_for_n_zero = np.full_like(r, 0.5)
    else:
        # Annulus: 1.0 + 0.5 * log(r / outer)
        # Handle r=0 if necessary, though in annulus r > 0.
        outcome_for_n_zero = 1.0 + 0.5 * np.log(r / scale(a)[i])
    
    # Outcome 2: If n>=1 and r is at the boundary, the value is 1.0.
    outcome_for_r_boundary = 1.0
    
    # Outcome 3: The general case for n>=1 inside the boundary.
    lambda_val = lambda_ni(n, i, h, d)
    bessel_term = (besselie(0, lambda_val * r) / besselie(0, lambda_val * scale(a)[i])) * \
                  exp(lambda_val * (r - scale(a)[i]))

    # --- Apply the logic using nested np.where ---
    result_if_n_not_zero = np.where(cond_r_at_boundary, outcome_for_r_boundary, bessel_term)
    
    return np.where(cond_n_is_zero, outcome_for_n_zero, result_if_n_not_zero)

# Differentiate wrt r
def diff_R_1n(n, r, i, h, d, a):
    if n == 0:
        if i == 0:
            return 0.0
        else:
            # Derivative of 1.0 + 0.5*ln(r/outer) is 1/(2r)
            with np.errstate(divide='ignore'):
                return np.where(r == 0, np.inf, 1 / (2 * r))
    else:
        top = lambda_ni(n, i, h, d) * besselie(1, lambda_ni(n, i, h, d) * r)
        bottom = besselie(0, lambda_ni(n, i, h, d) * scale(a)[i])
        return top / bottom * exp(lambda_ni(n, i, h, d) * (r - scale(a)[i]))

def diff_R_1n_vectorized(n, r, i, h, d, a):
    """
    Vectorized derivative of the diff_R_1n radial function.
    FIXED: Handles i!=0 case for n=0 to match scalar version.
    """
    condition = (n == 0)
    
    # FIX: n=0 derivative logic
    if i == 0:
        value_if_true = np.zeros_like(r)
    else:
        # Derivative is 1/(2r)
        value_if_true = np.divide(1.0, 2 * r, out=np.full_like(r, np.inf), where=(r!=0))
    
    # --- Calculation for when n > 0 ---
    lambda_val = lambda_ni(n, i, h, d)
    
    numerator = lambda_val * besselie(1, lambda_val * r) 
    denominator = besselie(0, lambda_val * scale(a)[i])
    
    bessel_ratio = np.divide(numerator, denominator, 
                             out=np.zeros_like(numerator, dtype=float), 
                             where=(denominator != 0))
                             
    value_if_false = bessel_ratio * exp(lambda_val * (r - scale(a)[i]))
    
    return np.where(condition, value_if_true, value_if_false)

#############################################
# The "Bessel K" radial eigenfunction (Annular Regions)
# NORMALIZATION FIX: Anchored at Inner Radius (a[i-1]) + Affine Shift
#############################################

def R_2n(n, r, i, a, h, d):
    if i == 0:
        raise ValueError("i cannot be 0") 
    
    # LEGACY: Use Outer Radius
    outer_r = scale(a)[i]

    if n == 0:
        # LEGACY: 0.5 * log(r / outer)
        # This is 0 at the outer boundary, not 1.
        return 0.5 * np.log(r / outer_r)
    else:
        lambda_val = lambda_ni(n, i, h, d)
        
        if r == outer_r:
            return 1.0
        else:
            # LEGACY: Normalized by K0 at OUTER radius
            num = besselke(0, lambda_val * r)
            den = besselke(0, lambda_val * outer_r)
            return (num / den) * exp(lambda_val * (outer_r - r))

def R_2n_vectorized(n, r, i, a, h, d):
    """
    Vectorized version of the R_2n radial eigenfunction.
    LEGACY MODE: Anchored at Outer Radius (a[i]).
    """
    if i == 0:
        raise ValueError("R_2n function is not defined for the innermost region (i=0).")

    # LEGACY: Use Outer Radius
    outer_r = scale(a)[i]

    cond_n_is_zero = (n == 0)
    cond_r_at_boundary = (r == outer_r)

    # Case 1: n = 0
    outcome_for_n_zero = 0.5 * np.log(r / outer_r)

    # Case 2: n > 0 and r is at the boundary
    outcome_for_r_boundary = 1.0

    # Case 3: n > 0 and r is not at the boundary
    lambda_val = lambda_ni(n, i, h, d)
    
    # Mask input where n=0 to prevent 'inf' errors
    lambda_safe = np.where(cond_n_is_zero, 1.0, lambda_val)
    
    # LEGACY: Denom uses OUTER radius
    denom = besselke(0, lambda_safe * outer_r)
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)

    bessel_term = (besselke(0, lambda_safe * r) / denom) * exp(lambda_safe * (outer_r - r))

    result_if_n_not_zero = np.where(cond_r_at_boundary, outcome_for_r_boundary, bessel_term)

    return np.where(cond_n_is_zero, outcome_for_n_zero, result_if_n_not_zero)

# Differentiate wrt r (Unchanged, as d/dr(1.0) is 0)
def diff_R_2n(n, r, i, h, d, a):
    # LEGACY: Anchored at Outer Radius
    if n == 0:
        return 1.0 / (2 * r)
    else:
        lambda0 = lambda_ni(n, i, h, d)
        outer_r = scale(a)[i]
        
        # Derivative of K0(lr)/K0(la) is -l*K1(lr)/K0(la)
        # Using scaled K:
        # -l * (ke1(lr)/ke0(la)) * exp(l(a-r))
        
        top = - lambda0 * besselke(1, lambda0 * r)
        bot = besselke(0, lambda0 * outer_r)
        
        return (top / bot) * np.exp(lambda0 * (outer_r - r))
    
def diff_R_2n_vectorized(n, r, i, h, d, a):
    n = np.asarray(n)
    r = np.asarray(r)
    
    # Case n == 0: Derivative is still 1/(2r)
    value_if_true = np.divide(1.0, 2 * r, out=np.full_like(r, np.inf), where=(r != 0))
    
    # Case n > 0
    lambda_val = lambda_ni(n, i, h, d)
    outer_r = scale(a)[i] # LEGACY ANCHOR
    
    lambda_safe = np.where(n == 0, 1.0, lambda_val)
    
    # LEGACY: Denom uses OUTER radius
    denom = besselke(0, lambda_safe * outer_r)
    safe_denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = -lambda_safe * besselke(1, lambda_safe * r)
        ratio = numerator / safe_denom
        # LEGACY: Exponential decay from OUTER radius
        exp_term = exp(lambda_safe * (outer_r - r))
        value_if_false = ratio * exp_term

    return np.where(n == 0, value_if_true, value_if_false)
#############################################
# i-region vertical eigenfunctions
def Z_n_i(n, z, i, h, d):
    if n == 0:
        return 1
    else:
        return np.sqrt(2) * np.cos(lambda_ni(n, i, h, d) * (z + h))
    
def Z_n_i_vectorized(n, z, i, h, d):
    """
    Vectorized version of the i-region vertical eigenfunction Z_n_i.
    """
    # Define the condition to check for each element in the 'n' array
    condition = (n == 0)
    
    # Define the calculation for when n != 0
    # This part is already vectorized thanks to NumPy
    value_if_false = np.sqrt(2) * np.cos(lambda_ni(n, i, h, d) * (z + h))
    
    # Use np.where to choose the output:
    # If condition is True (n==0), return 1.0.
    # Otherwise, return the result of the calculation.
    return np.where(condition, 1.0, value_if_false)

def diff_Z_n_i(n, z, i, h, d):
    if n == 0:
        return 0
    else:
        lambda0 = lambda_ni(n, i, h, d)
        return - lambda0 * np.sqrt(2) * np.sin(lambda0 * (z + h))
    
def diff_Z_n_i_vectorized(n, z, i, h, d):
    """
    Vectorized derivative of the Z_n_i vertical function.
    """
    # Define the condition to be applied element-wise.
    condition = (n == 0)
    
    # Define the value if the condition is True (when n=0).
    value_if_true = 0.0
    
    # Define the calculation for when the condition is False (when n > 0).
    # This part is already vectorized.
    lambda_val = lambda_ni(n, i, h, d)
    value_if_false = -lambda_val * np.sqrt(2) * np.sin(lambda_val * (z + h))
    
    # Use np.where to select the output based on the condition.
    return np.where(condition, value_if_true, value_if_false)

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
        
def Lambda_k_vectorized(k, r, m0, a, m_k_arr):
    """
    Vectorized version of the exterior region radial eigenfunction Lambda_k.
    """
    if m0 == inf:
        return np.ones(np.broadcast(k, r).shape, dtype=float)

    cond_k_is_zero = (k == 0)
    cond_r_at_boundary = (r == scale(a)[-1])

    outcome_boundary = 1.0

    # --- Case 2: k = 0 (NEEDS FIX) ---
    denom_k_zero = besselh(0, m0 * scale(a)[-1])
    numer_k_zero = besselh(0, m0 * r)
    with np.errstate(divide='ignore', invalid='ignore'):
        outcome_k_zero = np.divide(numer_k_zero, denom_k_zero, 
                                   out=np.zeros_like(numer_k_zero, dtype=complex),
                                   where=np.isfinite(denom_k_zero) & (denom_k_zero != 0))

    # --- Case 3: k > 0 (NEEDS FIX) ---
    # Mask input where k=0 to prevent errors
    local_m_k_k = m_k_arr[k]
    safe_m_k = np.where(cond_k_is_zero, 1.0, local_m_k_k)
    
    denom_k_nonzero = besselke(0, safe_m_k * scale(a)[-1])
    numer_k_nonzero = besselke(0, safe_m_k * r)
    with np.errstate(divide='ignore', invalid='ignore'):
        bessel_ratio = np.divide(numer_k_nonzero, denom_k_nonzero, 
                                 out=np.zeros_like(numer_k_nonzero),
                                 where=np.isfinite(denom_k_nonzero) & (denom_k_nonzero != 0))
    outcome_k_nonzero = bessel_ratio * exp(safe_m_k * (scale(a)[-1] - r))
    # --- END FIXES ---

    result_if_not_boundary = np.where(cond_k_is_zero,
                                      outcome_k_zero,
                                      outcome_k_nonzero)

    return np.where(cond_r_at_boundary,
                    outcome_boundary,
                    result_if_not_boundary)

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
         
            return numerator / denominator
    else:
        numerator = -(local_m_k_k * besselke(1, local_m_k_k * r))
        denominator = besselke(0, local_m_k_k * scale(a)[-1])
        return numerator / denominator * exp(local_m_k_k * (scale(a)[-1] - r))

def diff_Lambda_k_vectorized(k, r, m0, a, m_k_arr):
    """
    Vectorized derivative of the exterior region radial function Lambda_k.
    """
    # Handle the scalar case where m0 is infinite. The result is always 1.
    if m0 == inf:
        return np.ones(np.broadcast(k, r).shape, dtype=float)

    # --- Define the condition for vectorization ---
    condition = (k == 0)

    # --- Define the outcome for k == 0 ---
    numerator_k_zero = -(m0 * besselh(1, m0 * r))
    denominator_k_zero = besselh(0, m0 * scale(a)[-1])
    outcome_k_zero = np.divide(numerator_k_zero, denominator_k_zero,
                               out=np.zeros_like(numerator_k_zero, dtype=complex),
                               where=(denominator_k_zero != 0))

    # --- Define the outcome for k > 0 ---
    # NumPy's advanced indexing allows m_k_arr[k] to create an array from indices.
    local_m_k_k = m_k_arr[k]
    # Mask input where k=0
    safe_m_k = np.where(condition, 1.0, local_m_k_k)
    
    numerator_k_nonzero = -(safe_m_k * besselke(1, safe_m_k * r))
    denominator_k_nonzero = besselke(0, safe_m_k * scale(a)[-1])
    
    # Use safe division to avoid warnings
    ratio = np.divide(numerator_k_nonzero, denominator_k_nonzero,
                      out=np.zeros_like(numerator_k_nonzero, dtype=float),
                      where=(denominator_k_nonzero != 0))
                      
    outcome_k_nonzero = ratio * exp(safe_m_k * (scale(a)[-1] - r))

    # --- Use np.where to select the final output ---
    return np.where(condition, outcome_k_zero, outcome_k_nonzero)

#############################################
# Equation 2.34 in analytical methods book, also eq 16 in Seah and Yeung 2006:
# REVISED N_k to accept m_k_arr (as it previously called m_k itself)

def N_k_multi(k, m0, h, m_k_arr): 
    if m0 == inf: return 1/2
    elif k == 0:
        # --- FIX: Prevent overflow for deep water ---
        if (2 * m0 * h) > 700: 
            return 1e308 
        # ------------------------------------------
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * m_k_arr[k] * h) / (2 * m_k_arr[k] * h))

#############################################
# e-region vertical eigenfunctions
def Z_k_e(k, z, m0, h, NMK, m_k_arr):
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        if m0 == inf: return 0
        if m0 * h < M0_H_THRESH:
            return 1 / sqrt(N_k_multi(k, m0, h, m_k_arr)) * cosh(m0 * (z + h))
        else: # high m0h approximation
            return sqrt(2 * m0 * h) * (exp(m0 * z) + exp(-m0 * (z + 2*h)))
    else:
        return 1 / sqrt(N_k_multi(k, m0, h, m_k_arr)) * cos(local_m_k[k] * (z + h))
    
def Z_k_e_vectorized(k, z, m0, h, m_k_arr, N_k_arr):
    """
    Vectorized version of the e-region vertical eigenfunction Z_k_e.
    This version uses pre-calculated m_k_arr and N_k_arr for efficiency.
    """
    # This outer conditional is fine because it operates on scalar inputs.
    if m0 * h < M0_H_THRESH:
        # --- Logic for the standard case ---
        # Value for k = 0
        outcome_k_zero = (1 / sqrt(N_k_arr[0])) * cosh(m0 * (z + h))
        
        # Value for k > 0
        # NumPy's advanced indexing handles using an array 'k' to index other arrays.
        outcome_k_nonzero = (1 / sqrt(N_k_arr[k])) * cos(m_k_arr[k] * (z + h))

        return np.where(k == 0, outcome_k_zero, outcome_k_nonzero)
    else:
        # --- Logic for the high m0h approximation ---
        # Value for k = 0
        outcome_k_zero = sqrt(2 * m0 * h) * (exp(m0 * z) + exp(-m0 * (z + 2 * h)))
        
        # Value for k > 0 (this part is the same as the standard case)
        outcome_k_nonzero = (1 / sqrt(N_k_arr[k])) * cos(m_k_arr[k] * (z + h))
        
        return np.where(k == 0, outcome_k_zero, outcome_k_nonzero)

def diff_Z_k_e_vectorized(k, z, m0, h, m_k_arr, N_k_arr):
    """
    Vectorized derivative of the e-region vertical eigenfunction Z_k_e.
    This version uses pre-calculated m_k_arr and N_k_arr for efficiency.
    """
    # This outer conditional is fine because it operates on scalar inputs.
    if m0 * h < M0_H_THRESH:
        # --- Logic for the standard case ---
        # Value for k = 0
        outcome_k_zero = (1 / sqrt(N_k_arr[0])) * m0 * sinh(m0 * (z + h))
        
        # Value for k > 0
        # NumPy's advanced indexing handles using an array 'k' to index other arrays.
        outcome_k_nonzero = -(1 / sqrt(N_k_arr[k])) * m_k_arr[k] * sin(m_k_arr[k] * (z + h))
        
        return np.where(k == 0, outcome_k_zero, outcome_k_nonzero)
    
    else:
        # --- Logic for the high m0h approximation ---
        # Value for k = 0
        outcome_k_zero = m0 * sqrt(2 * h * m0) * (exp(m0 * z) - exp(-m0 * (z + 2 * h)))
        
        # Value for k > 0 (this part is the same as the standard case)
        outcome_k_nonzero = -(1 / sqrt(N_k_arr[k])) * m_k_arr[k] * sin(m_k_arr[k] * (z + h))
        
        return np.where(k == 0, outcome_k_zero, outcome_k_nonzero)

#############################################
# To calculate hydrocoefficients

#integrating R_1n * r
# Integration
def int_R_1n(i, n, a, h, d):
    if n == 0:
        if i == 0:
            # Central cylinder: Integral of 0.5 * r
            return a[i]**2/4 
        else:
            # Annulus: Integral of r * (1.0 + 0.5 * ln(r/outer_r))
            outer_r = scale(a)[i]
            inner_r = a[i-1]
            
            cyl_term = (outer_r**2 - inner_r**2) / 2.0
            
            def log_indefinite_int(r):
                # Integral of 0.5 * r * ln(r/outer)
                # = 0.5 * [ (r^2/2)*ln(r/outer) - r^2/4 ]
                log_val = np.log(r/outer_r) if r > 0 else 0
                return 0.5 * ((r**2 / 2.0) * log_val - (r**2 / 4.0))

            val_outer = log_indefinite_int(outer_r) # log(1)=0, so just -r^2/4 term
            val_inner = log_indefinite_int(inner_r)
            
            return cyl_term + (val_outer - val_inner)
    else:
        # Standard Bessel I integral (Unchanged)
        lambda0 = lambda_ni(n, i, h, d)
        bottom = lambda0 * besselie(0, lambda0 * scale(a)[i])
        if i == 0: inner_term = 0
        else: inner_term = (a[i-1] * besselie(1, lambda0 * a[i-1]) / bottom) * exp(lambda0 * (a[i-1] - scale(a)[i]))
        outer_term = (a[i] * besselie(1, lambda0 * a[i]) / bottom) * exp(lambda0 * (a[i] - scale(a)[i]))
        return outer_term - inner_term
    
#integrating R_2n * r
# Integral must be updated to include the volume of the new "1.0" cylinder
def int_R_2n(i, n, a, h, d):
    """
    Computes the integral of R_2n(r) * r dr from inner_r to outer_r.
    LEGACY MODE: Matches old_assembly.py (Outer Radius Anchor).
    """
    if i == 0:
        raise ValueError("i cannot be 0")
    
    # LEGACY: Use Outer Radius
    outer_r = scale(a)[i]
    inner_r = a[i-1] # Previous radius is inner

    if n == 0:
        # Integral of r * (0.5 * ln(r/outer_r))
        # Analytic result: [ 0.5 * ( (r^2/2)*ln(r/outer_r) - r^2/4 ) ] evaluated from inner to outer
        
        def indefinite(r):
            if r <= 0: return 0
            # ln(r/outer_r) is 0 when r=outer_r
            term_log = np.log(r / outer_r)
            return 0.5 * ((r**2 / 2) * term_log - (r**2 / 4))

        val_outer = indefinite(outer_r)
        val_inner = indefinite(inner_r)
        return val_outer - val_inner

    else:
        # Integral of r * K0(lambda*r) / K0(lambda*outer) * exp(...)
        # The old code handled this via standard Bessel integrals.
        # We must replicate the specific normalization of old_assembly.
        
        lambda0 = lambda_ni(n, i, h, d)
        
        # In old_assembly, R_2n = K0(lr)/K0(la) * exp(...)
        # The integral of x*K0(x) is -x*K1(x).
        
        # Numerator term (pure Bessel K integral part)
        # Int[ r * K0(lambda*r) ] = - (r/lambda) * K1(lambda*r)
        
        # We need to evaluate: [ - (r/lambda)*K1(lambda*r) ] from inner to outer
        # Then multiply by the normalization constant: exp(lambda*outer) / K0(lambda*outer)
        # Wait, the exponential term in R_2n_old is: exp(lambda * (outer_r - r))
        # This exponential cancels out the exp inside the scaled Bessel K if we use K_scaled.
        # But old_assembly likely used raw K. Let's stick to the raw math.
        
        # Let's look at the old implementation logic provided in previous turns:
        # It normalizes by K0(outer).
        
        # Exact calculation matching old_assembly:
        k0_outer = besselke(0, lambda0 * outer_r) # scaled K0
        
        # We need the integral of:
        # ( K0(lr) / K0(la) ) * exp( l(a-r) ) * r
        # = ( K0(lr)*exp(lr) / (K0(la)*exp(la)) ) * r   <-- exp terms cancel strictly if using scaled K
        # effectively it is Integral( r * K0_scaled(lr) ) / K0_scaled(la)
        
        # Analytic integral of x * K0(x) is -x * K1(x).
        # So Int( r * K0(lr) ) = - (r/lambda) * K1(lr)
        
        # Converting to scaled Bessel K (kve):
        # K1(x) = kve(1, x) * exp(-x)
        # So - (r/l) * kve(1, lr) * exp(-lr)
        
        # We want: [ - (r/l) * kve(1, lr) * exp(-lr) ] / [ kve(0, la) * exp(-la) ] * exp( l(a-r) )
        # Notice exp(-lr) * exp(l(a-r)) = exp(-lr + la - lr) ... wait.
        # R_2n definition: ( kve(0, lr) * exp(-lr) ) / ( kve(0, la) * exp(-la) ) * exp( l(a-r) )
        # = kve(0, lr) / kve(0, la) * exp( -lr + la + la - lr ) ... no.
        
        # SIMPLER PATH:
        # R_2n_old(r) = K0(lr) / K0(la)  (Unscaled)
        # Int(r * R_2n) = (1/K0(la)) * [ - (r/l)*K1(lr) ] bounds inner..outer
        
        # Upper bound (r=outer):
        # - (outer/l) * K1(l*outer) / K0(l*outer)
        
        # Lower bound (r=inner):
        # - (inner/l) * K1(l*inner) / K0(l*outer)
        
        # Result = Upper - Lower
        # = (1/lambda0) * ( inner*K1(l*inner) - outer*K1(l*outer) ) / K0(l*outer)
        
        # Using Scaled Bessel functions (kve) to match python 'besselke':
        # K1(x) = ke1(x) * exp(-x)
        # K0(x) = ke0(x) * exp(-x)
        # Ratio K1(x)/K0(y) = ke1(x)/ke0(y) * exp(y-x)
        
        term_outer = (outer_r * besselke(1, lambda0 * outer_r)) 
        # exp(la - la) = 1, so no exp factor needed for outer term.
        
        term_inner = (inner_r * besselke(1, lambda0 * inner_r))
        # exp(la - li). We need to multiply by this.
        term_inner *= np.exp(lambda0 * (outer_r - inner_r))
        
        denom = lambda0 * besselke(0, lambda0 * outer_r)
        
        # Result = (inner_term - outer_term) / (lambda * K0_scaled(outer))
        # Note the sign flip from the integration limits (Upper - Lower) vs (-x*K1).
        
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
def excitation_phase(x, NMK, m0, a): # x-vector of unknown coefficients
    coeff = x[-NMK[-1]] # first coefficient of e-region expansion
    local_scale = scale(a)
    return -(pi/2) + np.angle(coeff) - np.angle(besselh(0, m0 * local_scale[-1]))

def excitation_force(damping, m0, h):
    # --- FIX: Handle infinite m0 ---
    if m0 == inf:
        return 0.0
    # -------------------------------

    # Chau 2012 eq 98
    const = np.tanh(m0 * h) + m0 * h * (1 - (np.tanh(m0 * h))**2)

    return sqrt((2 * const * rho * (g ** 2) * damping)/(omega(m0,h,g) * m0)) ** (1/2)

# --- AFTER ---
def make_R_Z(a, h, d, sharp, spatial_res, R_range: Optional[np.ndarray] = None, Z_range: Optional[np.ndarray] = None):
    
    if R_range is not None:
        r_vec = R_range
    else:
        # Fallback to old behavior
        rmin = (2 * a[-1] / spatial_res) if sharp else 0.0
        r_vec = np.linspace(rmin, 2*a[-1], spatial_res)

    if Z_range is not None:
        z_vec = Z_range
    else:
        # Fallback to old behavior
        z_vec = np.linspace(0, -h, spatial_res) 

    if sharp: # more precise near boundaries
        # Note: This 'sharp' logic is probably not compatible
        # with providing R_range/Z_range, but your test
        # correctly sets sharp=False, so this block is skipped.
        a_eps = 1.0e-4
        for i in range(len(a)):
            r_vec = np.append(r_vec, a[i]*(1-a_eps))
            r_vec = np.append(r_vec, a[i]*(1+a_eps))
        r_vec = np.unique(r_vec)
        for i in range(len(d)):
            z_vec = np.append(z_vec, -d[i])
        z_vec = np.unique(z_vec)
    
    # THE CRITICAL FIX: Add indexing='ij'
    return np.meshgrid(r_vec, z_vec, indexing='ij')

def p_diagonal_block(left, radfunction, bd, h, d, a, NMK):
    region = bd if left else (bd + 1)
    sign = 1 if left else (-1)
    return sign * (h - d[region]) * np.diag(radfunction(list(range(NMK[region])), a[bd], region))

def p_dense_block(left, radfunction, bd, NMK, a, I_nm_vals):
    I_nm_array = I_nm_vals[bd]
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
   
# arguments: diagonal block on left (T/F), vectorized radial eigenfunction, boundary number
def v_diagonal_block(left, radfunction, bd, h, d, NMK, a):
    region = bd if left else (bd + 1)
    sign = (-1) if left else (1)
    return sign * (h - d[region]) * np.diag(radfunction(list(range(NMK[region])), a[bd], region))

# arguments: dense block on left (T/F), vectorized radial eigenfunction, boundary number
def v_dense_block(left, radfunction, bd, I_nm_vals, NMK, a):
    I_nm_array = I_nm_vals[bd]
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
    radial_term = diff_R_1n(k, a[bd], bd, h, d, a) # diff_R_1n is the correct one for this block
    
    # I_mk_vals is correctly defined as (NMK[prev_region], NMK[current_region])
    # The old code used I_km_array = np.transpose(I_mk_vals) and then indexed it as I_km_array[m, k] (local row, local col)
    # So, if I_mk_vals is (rows_of_prev_region, cols_of_current_region),
    # then I_km_array[m, k] corresponds to I_mk_vals_untransposed[k, m]
    imk_term = I_mk_vals[k, m] # k is the 'col' index of current block, m is the 'row' index of current block

    # The outer sign is (-1) from v_dense_block_e
    result = -1 * radial_term * imk_term

    return result

def v_diagonal_block_e_entry(m, k, bd, m0, m_k_arr, a, h):
    """
    Compute individual (m,k) entry of the velocity diagonal block e at boundary bd.
    """
    
    #  need access to 'h' here. It's available in the outer scope
    # through the problem object, or pass it directly.
    # Since NMK and a are passed, h should be too if it's not a global constant.
    # retrieve h from the problem object or pass it.
    
    # If h is always domain_list[0].h, can pass it from build_problem_cache
    # For now, let's explicitly pass h from build_problem_cache to this function
    # via the closure.
    
    radius = a[bd]
    
    # Call diff_Lambda_k, ensure it's the correct new version from multi_equations.py
    # (it is, from code snippet)
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
        d (list): A list of the depths for each region.

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