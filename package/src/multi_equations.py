import numpy as np
from scipy.special import hankel1 as besselh
from scipy.special import iv as besseli
from scipy.special import kv as besselk
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from numpy import sqrt, cosh, cos, sinh, sin, pi, exp
from scipy.optimize import newton, minimize_scalar, root_scalar
import scipy as sp

def omega(m0,h,g):
    return sqrt(m0 * np.tanh(m0 * h) * g)

def scale(a):
    # Pad 'a' with a zero at the beginning to align for the average calculation
    # For a = [5, 10, 15], this would make padded_a = [0, 5, 10]
    padded_a_prev = np.concatenate(([0.0], a[:-1])) # Ensure 0.0 for float type consistency
    
    # Now, element-wise average: (previous_a + current_a) / 2
    result = (padded_a_prev + a) / 2
    return result

def lambda_ni(n, i, h, d): # factor used often in calculations
    return n * pi / (h - d[i])

#############################################
# some common computations

# creating a m_k function, used often in calculations
def m_k_entry(k, m0, h):
    # m_k_mat = np.zeros((len(m0_vec), 1))
    if k == 0: return m0

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
    vectorized_m_k_entry = np.vectorize(m_k_entry, otypes=[float])
    return vectorized_m_k_entry(list(range(NMK[-1])), m0, h)

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
def I_mk(m, k, i, d, m0, h, NMK, m_k_arr, N_k_arr): # coupling integral for i and e-type regions
    # Use the pre-computed array
    local_m_k_k = m_k_arr[k] # Access directly from array
    
    dj = d[i]
    if m == 0 and k == 0:
        if m0 * h < 14:
            return (1/sqrt(N_k_arr[0])) * sinh(m0 * (h - dj)) / m0 # Use N_k_arr[0]
        else: # high m0h approximation
            return sqrt(2 * h / m0) * (exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
    if m == 0 and k >= 1:
        return (1/sqrt(N_k_arr[k])) * sin(local_m_k_k * (h - dj)) / local_m_k_k # Use N_k_arr[k]
    if m >= 1 and k == 0:
        if m0 * h < 14:
            num = (-1)**m * sqrt(2) * (1/sqrt(N_k_arr[0])) * m0 * sinh(m0 * (h - dj)) # Use N_k_arr[0]
        else: # high m0h approximation
            num = (-1)**m * 2 * sqrt(h * m0 ** 3) *(exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
        denom = (m0**2 + lambda_ni(m, i, h, d) **2)
        return num/denom
    else:
        lambda1 = lambda_ni(m, i, h, d)
        if abs(local_m_k_k) == lambda1:
            return (h - dj)/2
        else:
            frac1 = sin((local_m_k_k + lambda1)*(h-dj))/(local_m_k_k + lambda1)
            frac2 = sin((local_m_k_k - lambda1)*(h-dj))/(local_m_k_k - lambda1)
            return sqrt(2)/2 * (1/sqrt(N_k_arr[k])) * (frac1 + frac2) # Use N_k_arr[k]
        
def I_mk_og(m, k, i, d, m0, h, NMK): # coupling integral for i and e-type regions
    local_m_k = m_k(NMK, m0, h)
    dj = d[i]
    if m == 0 and k == 0:
        if m0 * h < 14:
            return (1/sqrt(N_k_og(0, m0, h, NMK))) * sinh(m0 * (h - dj)) / m0
        else: # high m0h approximation
            return sqrt(2 * h / m0) * (exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
    if m == 0 and k >= 1:
        return (1/sqrt(N_k_og(k, m0, h, NMK))) * sin(local_m_k[k] * (h - dj)) / local_m_k[k]
    if m >= 1 and k == 0:
        if m0 * h < 14:
            num = (-1)**m * sqrt(2) * (1/sqrt(N_k_og(0, m0, h, NMK))) * m0 * sinh(m0 * (h - dj))
        else: # high m0h approximation
            num = (-1)**m * 2 * sqrt(h * m0 ** 3) *(exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
        denom = (m0**2 + lambda_ni(m, i, h, d) **2)
        return num/denom
    else:
        lambda1 = lambda_ni(m, i, h, d)
        if abs(local_m_k[k]) == lambda1:
            return (h - dj)/2
        else:
            frac1 = sin((local_m_k[k] + lambda1)*(h-dj))/(local_m_k[k] + lambda1)
            frac2 = sin((local_m_k[k] - lambda1)*(h-dj))/(local_m_k[k] - lambda1)
            return sqrt(2)/2 * (1/sqrt(N_k_og(k, m0, h, NMK))) * (frac1 + frac2)

#############################################
# b-vector computation

def b_potential_entry(n, i, d, heaving, h, a): # for two i-type regions
    #(integrate over shorter fluid, use shorter fluid eigenfunction)
    
    j = i + (d[i] < d[i+1]) # index of shorter fluid
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
        if m0 * h < 14:
            return constant * (1/sqrt(N_k_arr[0])) * sinh(m0 * (h - d[i])) / m0 # Use N_k_arr[0]
        else: # high m0h approximation
            return constant * sqrt(2 * h / m0) * (exp(- m0 * d[i]) - exp(m0 * d[i] - 2 * m0 * h))
    else:
        return constant * (1/sqrt(N_k_arr[k])) * sin(local_m_k_k * (h - d[i])) / local_m_k_k # Use N_k_arr[k]

def b_velocity_end_entry_og(k, i, heaving, a, h, d, m0, NMK): # between i and e-type regions
    local_m_k = m_k(NMK, m0, h)
    constant = - heaving[i] * a[i]/(2 * (h - d[i]))
    if k == 0:
        if m0 * h < 14:
            return constant * (1/sqrt(N_k_og(0, m0, h, NMK))) * sinh(m0 * (h - d[i])) / m0
        else: # high m0h approximation
            return constant * sqrt(2 * h / m0) * (exp(- m0 * d[i]) - exp(m0 * d[i] - 2 * m0 * h))
    else:
        return constant * (1/sqrt(N_k_og(k, m0, h, NMK))) * sin(local_m_k[k] * (h - d[i])) / local_m_k[k]

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

    local_scale = scale(a)
    if n == 0:
        return 0.5
    elif n >= 1:
        return besseli(0, lambda_ni(n, i, h, d) * r) / besseli(0, lambda_ni(n, i, h, d) * local_scale[i])
    else: 
        raise ValueError("Invalid value for n")
    
def R_1n_vectorized(n, r_array, i, h, d, a):
    local_scale = scale(a)
    if n == 0:
        return 0.5 * np.ones_like(r_array)
    elif n >= 1:
        return besseli(0, lambda_ni(n, i, h, d) * r_array) / besseli(0, lambda_ni(n, i, h, d) * local_scale[i])
    else:
        raise ValueError("Invalid value for n")

# Differentiate wrt r
def diff_R_1n(n, r, i, h, d, a):
    local_scale = scale(a)
    if n == 0:
        return 0
    else:
        top = lambda_ni(n, i, h, d) * besseli(1, lambda_ni(n, i, h, d) * r)
        bottom = besseli(0, lambda_ni(n, i, h, d) * local_scale[i])
        return top / bottom

#############################################
# The "Bessel K" radial eigenfunction
def R_2n(n, r, i, a, h, d): # this shouldn't be called for i=0, innermost.
    local_scale = scale(a)
    if i == 0:
        raise ValueError("i cannot be 0")  # this shouldn't be called for i=0, innermost region.
    elif n == 0:
        return 0.5 * np.log(r / a[i])
    else:
        return besselk(0, lambda_ni(n, i, h, d) * r) / besselk(0, lambda_ni(n, i, h, d) * local_scale[i])
    
def R_2n_vectorized(n, r_array, i, a, h, d): # Changed 'r' to 'r_array'
    local_scale = scale(a)
    if i == 0:
        raise ValueError("i cannot be 0")
    elif n == 0:
        # This needs to handle division by zero if r_array contains 0
        # ensure r_array doesn't contain 0 if R_2n is called for r=0
        return 0.5 * np.log(r_array / a[i])
    else:
        return besselk(0, lambda_ni(n, i, h, d) * r_array) / besselk(0, lambda_ni(n, i, h, d) * local_scale[i])

# Differentiate wrt r
def diff_R_2n(n, r, i, h, d, a):
    local_scale = scale(a)
    if n == 0:
        return 1 / (2 * r)
    else:
        top = - lambda_ni(n, i, h, d) * besselk(1, lambda_ni(n, i, h, d) * r)
        bottom = besselk(0, lambda_ni(n, i, h, d) * local_scale[i])
        return top / bottom


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
def Lambda_k(k, r, m0, a, NMK, h, m_k_arr, N_k_arr): # ADDED m_k_arr, N_k_arr
    local_scale = scale(a)
    local_m_k_k = m_k_arr[k] # Access directly from array
    
    if k == 0:
        return besselh(0, m0 * r) / besselh(0, m0 * local_scale[-1])
    else:
        return besselk(0, local_m_k_k * r) / besselk(0, local_m_k_k * local_scale[-1])
    
def Lambda_k_vectoized(k, r_array, m0, a, NMK, h, m_k_arr, N_k_arr): # Changed 'r' to 'r_array'
    local_scale = scale(a)
    local_m_k_k = m_k_arr[k]

    if k == 0:
        return besselh(0, m0 * r_array) / besselh(0, m0 * local_scale[-1])
    else:
        return besselk(0, local_m_k_k * r_array) / besselk(0, local_m_k_k * local_scale[-1])
    
def Lambda_k_og(k, r, m0, a, NMK, h):
    local_scale = scale(a)
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        return besselh(0, m0 * r) / besselh(0, m0 * local_scale[-1])
    else:
        return besselk(0, local_m_k[k] * r) / besselk(0, local_m_k[k] * local_scale[-1])

# Differentiate wrt r 
# REVISED diff_Lambda_k to accept m_k_arr and N_k_arr
def diff_Lambda_k(k, r, m0, NMK, h, a, m_k_arr, N_k_arr): # ADDED m_k_arr, N_k_arr
    local_m_k_k = m_k_arr[k] # Access directly from array
    local_scale = scale(a)
    if k == 0:
        numerator = -(m0 * besselh(1, m0 * r))
        denominator = besselh(0, m0 * local_scale[-1])
    else:
        numerator = -(local_m_k_k * besselk(1, local_m_k_k * r))
        denominator = besselk(0, local_m_k_k * local_scale[-1])
    return numerator / denominator

def diff_Lambda_k_og(k, r, m0, NMK, h, a):
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

def N_k_multi(k, m0, h, NMK, m_k_arr): # Added m_k_arr as optional argument
    if m_k_arr is not None:
        local_m_k_k = m_k_arr[k]
    else:
        # Fallback for compatibility or unoptimized calls (e.g., in _full_assemble_A_multi if you didn't pass array)
        local_m_k_k = m_k_entry(k, m0, h) # Call the single entry function

    if k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * local_m_k_k * h) / (2 * local_m_k_k * h))
    
def N_k_og(k, m0, h, NMK):
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * local_m_k[k] * h) / (2 * local_m_k[k] * h))


#############################################
# e-region vertical eigenfunctions
def Z_k_e(k, z, m0, h, NMK, m_k_arr):
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        if m0 * h < 14:
            return 1 / sqrt(N_k_multi(k, m0, h, NMK, m_k_arr)) * cosh(m0 * (z + h))
        else: # high m0h approximation
            return sqrt(2 * m0 * h) * (exp(m0 * z) + exp(-m0 * (z + 2*h)))
    else:
        return 1 / sqrt(N_k_multi(k, m0, h, NMK, m_k_arr)) * cos(local_m_k[k] * (z + h))
    
def Z_k_e_vectorized(k, z_array, m0, h, NMK, m_k_arr): # Changed 'z' to 'z_array'
    local_m_k = m_k(NMK, m0, h) 
    if k == 0:
        if m0 * h < 14:
            return 1 / sqrt(N_k_multi(k, m0, h, NMK, m_k_arr)) * cosh(m0 * (z_array + h))
        else: # high m0h approximation
            return sqrt(2 * m0 * h) * (exp(m0 * z_array) + exp(-m0 * (z_array + 2*h)))
    else:
        return 1 / sqrt(N_k_multi(k, m0, h, NMK, m_k_arr)) * cos(local_m_k[k] * (z_array + h))

def diff_Z_k_e(k, z, m0, h, NMK, m_k_arr):
    local_m_k = m_k(NMK, m0, h)
    if k == 0:
        if m0 * h < 14:
            return 1 / sqrt(N_k_multi(k, m0, h, NMK, m_k_arr)) * m0 * sinh(m0 * (z + h))
        else: # high m0h approximation
            return m0 * sqrt(2 * h * m0) * (exp(m0 * z) - exp(-m0 * (z + 2*h)))
    else:
        return -1 / sqrt(N_k_multi(k, m0, h, NMK, m_k_arr)) * local_m_k[k] * sin(local_m_k[k] * (z + h))

#############################################
# To calculate hydrocoefficients

#integrating R_1n * r
def int_R_1n(i, n, a, h, d):
    local_scale = scale(a)
    if n == 0:
        inner = (0 if i == 0 else a[i-1]) # central region has inner radius 0
        return a[i]**2/4 - inner**2/4
    else:
        lambda0 = lambda_ni(n, i, h, d)
        inner_term = (0 if i == 0 else a[i-1] * besseli(1, lambda0 * a[i-1])) # central region has inner radius 0
        top = a[i] * besseli(1, lambda0 * a[i]) - inner_term
        bottom = lambda0 * besseli(0, lambda0 * local_scale[i])
        return top / bottom

#integrating R_2n * r
def int_R_2n(i, n, a, h, d):
    local_scale = scale(a)
    if i == 0:
        raise ValueError("i cannot be 0")
    lambda0 = lambda_ni(n, i, h, d)
    if n == 0:
        return (a[i-1]**2 * (2*np.log(a[i]/a[i-1]) + 1) - a[i]**2)/8
    else:
        top = a[i] * besselk(1, lambda0 * a[i]) - a[i-1] * besselk(1, lambda0 * a[i-1])
        bottom = - lambda0 * besselk(0, lambda0 * local_scale[i])
        return top / bottom

#integrating phi_p_i * d_phi_p_i/dz * r *d_r at z=d[i]
def int_phi_p_i_no_coef(i, h, d, a):
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
def excitation_phase(coeff, m0, a): # first coefficient of e-region expansion
    local_scale_last = scale(a)[-1]
    return -(pi/2) + np.angle(coeff) - np.angle(besselh(0, m0 * local_scale_last))