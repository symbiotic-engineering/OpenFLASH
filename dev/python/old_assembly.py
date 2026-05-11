from functools import partial
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

LARGE_M0H = 14
def diff_z_phi_p_i_old(d, z, h): 
    return ((z+h) / (h - d))
def diff_r_phi_p_i_old(d, r, h): 
    return (- r / (2* (h - d)))
def diff_Z_n_i_old(n, z, i, h, d):
    if n == 0:
        return 0
    else:
        lambda0 = lambda_ni_old(n, i, h, d)
        return - lambda0 * np.sqrt(2) * np.sin(lambda0 * (z + h))
def diff_Z_k_e_old(k, z, NMK, m0, h):
    m_k = m_k_old(NMK, m0, h)
    N_k = N_k_old(k, m0, h, m_k)
    if k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            return 1 / sqrt(N_k(k)) * m0 * sinh(m0 * (z + h))
        else: # high m0h approximation
            return m0 * sqrt(2 * h * m0) * (exp(m0 * z) - exp(-m0 * (z + 2*h)))
    else:
        return -1 / sqrt(N_k(k)) * m_k[k] * sin(m_k[k] * (z + h))
def phi_p_i_old(d, r, z, h): 
    return (1 / (2* (h - d))) * ((z + h) ** 2 - (r**2) / 2)
def make_R_Z_old(a, h, d, sharp, spatial_res): # create coordinate array for graphing
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
def Z_k_e_old(k, z, m0, h, NMK):
    m_k = m_k_old(NMK, m0, h)
    if k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            return 1 / sqrt(N_k_old(k, m0, h, m_k)) * cosh(m0 * (z + h))
        else: # high m0h approximation
            return sqrt(2 * m0 * h) * (exp(m0 * z) + exp(-m0 * (z + 2*h)))
    else:
        return 1 / sqrt(N_k_old(k, m0, h, m_k)) * cos(m_k[k] * (z + h))
def R_2n_old(n, r, i, a, h, d):
    if i == 0:
        raise ValueError("i cannot be 0")  # this shouldn't be called for i=0, innermost region.
    elif n == 0:
        return 0.5 * np.log(r / a[i])
    else:
        if r == scale_old(a)[i]:  # Saves bessel function eval
            return 1
        else:
            return besselke(0, lambda_ni_old(n, i, h, d) * r) / besselke(0, lambda_ni_old(n, i, h, d) * scale_old(a)[i]) * exp(lambda_ni_old(n, i, h, d) * (scale_old(a)[i] - r))
def Z_n_i_old(n, z, i, h, d):
    if n == 0:
        return 1
    else:
        return np.sqrt(2) * np.cos(lambda_ni_old(n, i, h, d) * (z + h))
    
def R_1n_old(n, r, i, a, h, d):
    if n == 0:
        return 0.5
    elif n >= 1:
        if r == scale_old(a)[i]: # Saves bessel function eval
            return 1
        else:
            return besselie(0, lambda_ni_old(n, i, h, d) * r) / besselie(0, lambda_ni_old(n, i, h, d) * scale_old(a)[i]) * exp(lambda_ni_old(n, i, h, d) * (r - scale_old(a)[i]))
    else: 
        raise ValueError("Invalid value for n")

def Lambda_k_old_wrapped(m0, a, NMK, h):
    def func(k, r):
        return Lambda_k_old(k, r, m0, a, NMK, h)
    return np.vectorize(func, otypes=[complex])
    
def Lambda_k_old(k, r, m0, a, NMK, h):
    m_k = m_k_old(NMK, m0, h)
    if k == 0:
        if m0 == inf:
        # the true limit is not well-defined, but whatever value this returns will be multiplied by zero
            return 1
        else:
            if r == scale_old(a)[-1]:  # Saves bessel function eval
                return 1
            else:
                return besselh(0, m0 * r) / besselh(0, m0 * scale_old(a)[-1])
    else:
        if r == scale_old(a)[-1]:  # Saves bessel function eval
            return 1
        else:
            return besselke(0, m_k[k] * r) / besselke(0, m_k[k] * scale_old(a)[-1]) * exp(m_k[k] * (scale_old(a)[-1] - r))
        
def diff_Lambda_k_old(k, r, m0, a, NMK, h):
    m_k = m_k_old(NMK, m0, h)
    if k == 0:
        if m0 == inf:
        # the true limit is not well-defined, but this makes the assigned coefficient zero
            return 1
        else:
            numerator = -(m0 * besselh(1, m0 * r))
            denominator = besselh(0, m0 * scale_old(a)[-1])
            return numerator / denominator
    else:
        numerator = -(m_k[k] * besselke(1, m_k[k] * r))
        denominator = besselke(0, m_k[k] * scale_old(a)[-1])
        return numerator / denominator * exp(m_k[k] * (scale_old(a)[-1] - r))
def b_potential_entry_old(n,i, d, heaving, h, a):
    j = i + (d[i] <= d[i+1]) # index of shorter fluid
    constant = (heaving[i+1] / (h - d[i+1]) - heaving[i] / (h - d[i]))
    if n == 0:
        return constant * 1/2 * ((h - d[j])**3/3 - (h-d[j]) * a[i]**2/2)
    else:
        return sqrt(2) * (h - d[j]) * constant * ((-1) ** n)/(lambda_ni_old(n, j, h, d) ** 2)
def b_potential_end_entry_old(n,i, h, d, heaving, a):
    constant = - heaving[i] / (h - d[i])
    if n == 0:
        return constant * 1/2 * ((h - d[i])**3/3 - (h-d[i]) * a[i]**2/2)
    else:
        return sqrt(2) * (h - d[i]) * constant * ((-1) ** n)/(lambda_ni_old(n, i, h, d) ** 2)
def b_velocity_entry_old(n, i, heaving, a, d, h): # for two i-type regions
    if n == 0:
        return (heaving[i+1] - heaving[i]) * (a[i]/2)
    if d[i] > d[i + 1]: #using i+1's vertical eigenvectors
        if heaving[i]:
            num = - sqrt(2) * a[i] * sin(lambda_ni_old(n, i+1, h, d) * (h-d[i]))
            denom = (2 * (h - d[i]) * lambda_ni_old(n, i+1, h, d))
            return num/denom
        else: return 0
    else: #using i's vertical eigenvectors
        if heaving[i+1]:
            num = sqrt(2) * a[i] * sin(lambda_ni_old(n, i, h, d) * (h-d[i+1]))
            denom = (2 * (h - d[i+1]) * lambda_ni_old(n, i, h, d))
            return num/denom
        else: return 0
def b_velocity_end_entry_old(k, i, heaving, a, h, d, m0, NMK): # between i and e-type regions
    constant = - heaving[i] * a[i]/(2 * (h - d[i]))
    m_k = m_k_old(NMK, m0, h)
    if k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            return constant * (1/sqrt(N_k_old(0, m0, h, m_k))) * sinh(m0 * (h - d[i])) / m0
        else: # high m0h approximation
            return constant * sqrt(2 * h / m0) * (exp(- m0 * d[i]) - exp(m0 * d[i] - 2 * m0 * h))
    else:
        return constant * (1/sqrt(N_k_old(k, m0, h, m_k))) * sin(m_k[k] * (h - d[i])) / m_k[k]
def diff_R_2n_old(n, r, i, h, d, a):
    if n == 0:
        return 1 / (2 * r)
    else:
        top = - lambda_ni_old(n, i, h, d) * besselke(1, lambda_ni_old(n, i, h, d) * r)
        bottom = besselke(0, lambda_ni_old(n, i, h, d) * scale_old(a)[i])
        return top / bottom * exp(lambda_ni_old(n, i, h, d) * (scale_old(a)[i] - r))
def diff_R_1n_old(n, r, i, h, d, a):
    if n == 0:
        return 0
    else:
        top = lambda_ni_old(n, i, h, d) * besselie(1, lambda_ni_old(n, i, h, d) * r)
        bottom = besselie(0, lambda_ni_old(n, i, h, d) * scale_old(a)[i])
        return top / bottom * exp(lambda_ni_old(n, i, h, d) * (r - scale_old(a)[i]))
def scale_old(a):
    return a
def R_1n_old(n, r, i, h, d, a):
    if n == 0:
        return 0.5
    elif n >= 1:
        if r == scale_old(a)[i]: # Saves bessel function eval
            return 1
        else:
            return besselie(0, lambda_ni_old(n, i, h, d) * r) / besselie(0, lambda_ni_old(n, i, h, d) * scale_old(a)[i]) * exp(lambda_ni_old(n, i, h, d) * (r - scale_old(a)[i]))
    else: 
        raise ValueError("Invalid value for n")
def R_2n_old(n, r, i, a, h, d):
    if i == 0:
        raise ValueError("i cannot be 0")  # this shouldn't be called for i=0, innermost region.
    elif n == 0:
        return 0.5 * np.log(r / a[i])
    else:
        if r == scale_old(a)[i]:  # Saves bessel function eval
            return 1
        else:
            return besselke(0, lambda_ni_old(n, i, h, d) * r) / besselke(0, lambda_ni_old(n, i, h, d) * scale_old(a)[i]) * exp(lambda_ni_old(n, i, h, d) * (scale_old(a)[i] - r))
def I_nm_old(n, m, i, h, d):
    dj = max(d[i], d[i+1]) # integration bounds at -h and -d
    if n == 0 and m == 0:
        return h - dj
    lambda1 = lambda_ni_old(n, i, h, d)
    lambda2 = lambda_ni_old(m, i + 1, h, d)
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
    
def N_k_old(k, m0, h, m_k):
    if m0 == inf: return 1/2
    elif k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * m_k[k] * h) / (2 * m_k[k] * h))

def m_k_entry_old(k, m0, h):
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
def m_k_old(NMK, m0, h):
    func = np.vectorize(lambda k: m_k_entry_old(k, m0, h), otypes=[float])
    return func(range(NMK[-1]))

def lambda_ni_old(n, i, h, d):
    return n * pi / (h - d[i])

def I_mk_old(m,k,i, d, h, m0, NMK):
    m_k = m_k_old(NMK, m0, h)
    dj = d[i]
    if m == 0 and k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            return (1/sqrt(N_k_old(0, m0, h, m_k))) * sinh(m0 * (h - dj)) / m0
        else: # high m0h approximation
            return sqrt(2 * h / m0) * (exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
    if m == 0 and k >= 1:
        return (1/sqrt(N_k_old(k, m0, h, m_k))) * sin(m_k[k] * (h - dj)) / m_k[k]
    if m >= 1 and k == 0:
        if m0 == inf: return 0
        elif m0 * h < LARGE_M0H:
            num = (-1)**m * sqrt(2) * (1/sqrt(N_k_old(0, m0, h, m_k))) * m0 * sinh(m0 * (h - dj))
        else: # high m0h approximation
            num = (-1)**m * 2 * sqrt(h * m0 ** 3) *(exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
        denom = (m0**2 + lambda_ni_old(m, i, h, d) **2)
        return num/denom
    else:
        lambda1 = lambda_ni_old(m, i, h, d)
        if abs(m_k[k]) == lambda1:
            return sqrt(2/N_k_old(k, m0, h, m_k)) * (h - dj)/2
        else:
            frac1 = sin((m_k[k] + lambda1)*(h-dj))/(m_k[k] + lambda1)
            frac2 = sin((m_k[k] - lambda1)*(h-dj))/(m_k[k] - lambda1)
            return sqrt(2/N_k_old(k, m0, h, m_k)) * (frac1 + frac2)/2
def debug_block(block, name, bd):
    max_val = np.max(np.abs(block))
    if max_val > 1e6:  # suspiciously large
        print(f"WARNING: Large values detected in {name} at boundary {bd}, max abs={max_val}")
        print(block)
    else:
        print(f"{name} at boundary {bd} max abs={max_val}")
        
def assemble_old_A_and_b(h, d, a, NMK, heaving, m0):
    left = 0
    for radius in a:
        assert radius > left, "a entries should be increasing, and start greater than 0."
        left = radius

    for depth in d:
        assert depth >= 0, "d entries should be nonnegative."
        assert depth < h, "d entries should be less than h."

    for val in NMK:
        assert (type(val) == int and val > 0), "NMK entries should be positive integers."

    # CREATING THE A MATRIX
    size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
    boundary_count = len(NMK) - 1

    rows = [] # collection of rows of blocks in A matrix, to be concatenated later

    ## Define values/functions to help block creation
    #Coupling integral values
    I_nm_vals = np.zeros((max(NMK), max(NMK), boundary_count - 1), dtype = complex)
    for bd in range(boundary_count - 1):
        for n in range(NMK[bd]):
            for m in range(NMK[bd + 1]):
                I_nm_vals[n][m][bd] = I_nm_old(n, m, bd, h, d)
    I_mk_vals = np.zeros((NMK[boundary_count - 1], NMK[boundary_count]), dtype = complex)
    for m in range(NMK[boundary_count - 1]):
        for k in range(NMK[boundary_count]):
            I_mk_vals[m][k]= I_mk_old(m, k, boundary_count - 1, d, h, m0, NMK)

    ## Functions to create blocks of certain types
    # arguments: diagonal block on left (T/F), vectorized radial eigenfunction, boundary number
    def p_diagonal_block(left, radfunction, bd):
        region = bd if left else (bd + 1)
        sign = 1 if left else (-1)
        return sign * (h - d[region]) * np.diag(radfunction(list(range(NMK[region])), a[bd], region))
        
    # arguments: dense block on left (T/F), vectorized radial eigenfunction, boundary number
    def p_dense_block(left, radfunction, bd):
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

    def p_dense_block_e(bd):
        I_mk_array = I_mk_vals
        vectorized_func = Lambda_k_old_wrapped(m0, a, NMK, h)
        radial_vector = vectorized_func(list(range(NMK[bd+1])), a[bd])
        radial_array = np.outer((np.full((NMK[bd]), 1)), radial_vector)
        return (-1) * radial_array * I_mk_array

    #####
    # arguments: diagonal block on left (T/F), vectorized radial eigenfunction, boundary number
    def v_diagonal_block(left, radfunction, bd):
        region = bd if left else (bd + 1)
        sign = (-1) if left else (1)
        return sign * (h - d[region]) * np.diag(radfunction(list(range(NMK[region])), a[bd], region))

    # arguments: dense block on left (T/F), vectorized radial eigenfunction, boundary number
    def v_dense_block(left, radfunction, bd):
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
    
    diff_Lambda_k_func = np.vectorize(
        partial(diff_Lambda_k_old, m0=m0, a=a, NMK=NMK, h=h),
        otypes=[complex]
    )
    test_k = 0
    test_r = a[bd] # Assuming bd is 2, so a[2] = 10

    # Get direct output from diff_Lambda_k_old
    direct_dlk_old = diff_Lambda_k_old(test_k, test_r, m0, a, NMK, h)
    print(f"DEBUG_direct: diff_Lambda_k_old({test_k}, {test_r}, m0={m0}) -> {direct_dlk_old}")

    # Get output from the vectorized function
    vectorized_dlk = diff_Lambda_k_func([test_k], test_r) # Pass test_k in a list for vectorize
    print(f"DEBUG_vectorized: diff_Lambda_k_func([{test_k}], {test_r}) -> {vectorized_dlk}")

    # Check if they are close
    print(f"DEBUG_comparison: direct vs vectorized close? {np.isclose(direct_dlk_old, vectorized_dlk[0])}")
    def v_diagonal_block_e(bd):
        vectorized_diff_Lambda_k_func = np.vectorize(
            partial(diff_Lambda_k_old, m0=m0, a=a, NMK=NMK, h=h),
            otypes=[complex]
        )
        
        # Calculate the diagonal elements by applying the vectorized function
        # 'a[bd]' is the fixed 'r' value for this boundary (radius)
        diagonal_elements = vectorized_diff_Lambda_k_func(list(range(NMK[bd+1])), a[bd]) # NMK[bd+1] is M
        
        # Create the diagonal matrix and ensure complex dtype
        return h * np.diag(diagonal_elements).astype(complex)

    def v_dense_block_e(radfunction, bd): # for region adjacent to e-type region
        I_km_array = np.transpose(I_mk_vals)
        radial_vector = radfunction(list(range(NMK[bd])), a[bd], bd)
        radial_array = np.outer((np.full((NMK[bd + 1]), 1)), radial_vector)
        if bd == 2:
            print(f"DEBUG: v_dense_block_e OLD (bd={bd}")
            print(f"  I_km_array shape: {I_km_array.shape}") # Transposed!
            print(f"  radial_vector shape: {radial_vector.shape}") # For the k-th element
        return (-1) * radial_array * I_km_array
    R_1n_func = np.vectorize(partial(R_1n_old, h=h, d=d, a=a))
    R_2n_func = np.vectorize(partial(R_2n_old,  a=a, h=h, d=d))
    diff_R_1n_func = np.vectorize(partial(diff_R_1n_old, h=h, d=d, a=a), otypes=[complex])
    diff_R_2n_func = np.vectorize(partial(diff_R_2n_old, h=h, d=d, a=a), otypes=[complex])
    # Potential Blocks
    col = 0
    row_start = 0
    for bd in range(boundary_count):
        N = NMK[bd]
        M = NMK[bd + 1]
        if bd == (boundary_count - 1): # i-e boundary, inherently left diagonal
            row_height = N
            # print(f"[OLD] Adding potential block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + N + M}")
            left_block1 = p_diagonal_block(True, np.vectorize(R_1n_func), bd)
            print(f"At boundary {bd}, adding p_diagonal_block with shape {left_block1.shape}, max abs val: {np.max(np.abs(left_block1))}")
            print(f"Block values: {left_block1}")
            right_block = p_dense_block_e(bd)
            debug_block(right_block, "p_dense_block_e", bd)
            if bd == 0: # one cylinder
                rows.append(np.concatenate((left_block1,right_block), axis = 1))
            else:
                left_block2 = p_diagonal_block(True, np.vectorize(R_2n_func), bd)
                left_zeros = np.zeros((row_height, col), dtype=complex)
                rows.append(np.concatenate((left_zeros,left_block1,left_block2,right_block), axis = 1))
        elif bd == 0:
            left_diag = d[bd] > d[bd + 1] # which of the two regions gets diagonal entries
            if left_diag:
                row_height = N
                # print(f"[OLD] Adding potential block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + N + 2*M}")
                left_block = p_diagonal_block(True, np.vectorize(R_1n_func), 0)
                right_block1 = p_dense_block(False, np.vectorize(R_1n_func), 0)
                right_block2 = p_dense_block(False, np.vectorize(R_2n_func), 0)
            else:
                row_height = M
                # print(f"[OLD] Adding potential block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + N + 2*M}")
                left_block = p_dense_block(True, np.vectorize(R_1n_func), 0)
                right_block1 = p_diagonal_block(False, np.vectorize(R_1n_func), 0)
                right_block2 = p_diagonal_block(False, np.vectorize(R_2n_func), 0)
            right_zeros = np.zeros((row_height, size - (col + N + 2 * M)),dtype=complex)
            block_lst = [left_block, right_block1, right_block2, right_zeros]
            rows.append(np.concatenate(block_lst, axis = 1))
            col += N
        else: # i-i boundary
            left_diag = d[bd] > d[bd + 1] # which of the two regions gets diagonal entries
            if left_diag:
                row_height = N
                # print(f"[OLD] Adding potential block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + 2*N + 2*M}")
                left_block1 = p_diagonal_block(True, np.vectorize(R_1n_func), bd)
                print(f"At boundary {bd}, adding p_diagonal_block with shape {left_block1.shape}, max abs val: {np.max(np.abs(left_block1))}")
                print(f"Block values: {left_block1}")
                left_block2 = p_diagonal_block(True, np.vectorize(R_2n_func), bd)
                right_block1 = p_dense_block(False, np.vectorize(R_1n_func),  bd)
                right_block2 = p_dense_block(False, np.vectorize(R_2n_func),  bd)
            else:
                row_height = M
                # print(f"[OLD] Adding potential block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + 2*N + 2*M}")
                left_block1 = p_dense_block(True, np.vectorize(R_1n_func),  bd)
                print(f"At boundary {bd}, adding p_dense_block with shape {left_block1.shape}, max abs val: {np.max(np.abs(left_block1))}")
                print(f"Block values: {left_block1}")
                left_block2 = p_dense_block(True, np.vectorize(R_2n_func),  bd)
                right_block1 = p_diagonal_block(False, np.vectorize(R_1n_func),  bd)
                right_block2 = p_diagonal_block(False, np.vectorize(R_2n_func),  bd)
            left_zeros = np.zeros((row_height, col), dtype=complex)
            right_zeros = np.zeros((row_height, size - (col + 2 * N + 2 * M)),dtype=complex)
            block_lst = [left_zeros, left_block1, left_block2, right_block1, right_block2, right_zeros]
            rows.append(np.concatenate(block_lst, axis = 1))
            col += 2 * N
        row_start += row_height


    ###############################
    # Velocity Blocks
     # Compute row_start offset for velocity blocks safely
    row_start = 0
    for i in range(boundary_count):
        if i < boundary_count - 1:
            if d[i] > d[i + 1]:
                row_start += NMK[i]
            else:
                row_start += NMK[i + 1]
        else:
            # For last boundary, no d[i+1], just add NMK[i]
            row_start += NMK[i]
    col = 0
    for bd in range(boundary_count):
        N = NMK[bd]
        M = NMK[bd + 1]
        if bd == (boundary_count - 1): # i-e boundary, inherently left diagonal
            row_height = M
            print(f"[OLD] Adding velocity block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + N + M}")
            left_block1 = v_dense_block_e(np.vectorize(diff_R_1n_func, otypes=[complex]), bd)
            right_block = v_diagonal_block_e(bd)
            print(f"right_block[local_m=1, local_k=0]: {right_block[1, 0]}")
            debug_block(left_block1, "v_dense_block_e (left_block1)", bd)
            debug_block(right_block, "v_diagonal_block_e (right_block)", bd)
            if bd == 0: # one cylinder
                rows.append(np.concatenate((left_block1,right_block), axis = 1))
            else:
                left_block2 = v_dense_block_e(np.vectorize(diff_R_2n_func, otypes=[complex]), bd)
                left_zeros = np.zeros((row_height, col), dtype=complex)
                rows.append(np.concatenate((left_zeros,left_block1,left_block2,right_block), axis = 1))
        elif bd == 0:
            left_diag = d[bd] <= d[bd + 1] # taller fluid region gets diagonal entries
            if left_diag:
                row_height = N
                print(f"[OLD] Adding velocity block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + N + 2*M}")
                left_block = v_diagonal_block(True, np.vectorize(diff_R_1n_func, otypes=[complex]), 0)
                right_block1 = v_dense_block(False, np.vectorize(diff_R_1n_func, otypes=[complex]), 0)
                right_block2 = v_dense_block(False, np.vectorize(diff_R_2n_func, otypes=[complex]), 0)
            else:
                row_height = M
                print(f"[OLD] Adding velocity block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + N + 2*M}")
                left_block = v_dense_block(True, np.vectorize(diff_R_1n_func, otypes=[complex]), 0)
                right_block1 = v_diagonal_block(False, np.vectorize(diff_R_1n_func, otypes=[complex]), 0)
                right_block2 = v_diagonal_block(False, np.vectorize(diff_R_2n_func, otypes=[complex]), 0)
            right_zeros = np.zeros((row_height, size - (col + N + 2 * M)),dtype=complex)
            block_lst = [left_block, right_block1, right_block2, right_zeros]
            rows.append(np.concatenate(block_lst, axis = 1))
            col += N
        else: # i-i boundary
            left_diag = d[bd] <= d[bd + 1] # taller fluid region gets diagonal entries
            if left_diag:
                row_height = N
                print(f"[OLD] Adding velocity block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + 2*N + 2*M}")
                left_block1 = v_diagonal_block(True, np.vectorize(diff_R_1n_func, otypes=[complex]), bd)
                debug_block(left_block1, "v_diagonal_block (left_block1)", bd)
                left_block2 = v_diagonal_block(True, np.vectorize(diff_R_2n_func, otypes=[complex]), bd)
                debug_block(left_block2, "v_diagonal_block (left_block2)", bd)
                right_block1 = v_dense_block(False, np.vectorize(diff_R_1n_func, otypes=[complex]),  bd)
                debug_block(right_block1, "v_dense_block (right_block1)", bd)
                right_block2 = v_dense_block(False, np.vectorize(diff_R_2n_func, otypes=[complex]),  bd)
                debug_block(right_block2, "v_dense_block (right_block2)", bd)
            else:
                row_height = M
                print(f"[OLD] Adding velocity block at bd={bd}, rows {row_start}-{row_start + row_height}, cols {col}-{col + 2*N + 2*M}")
                left_block1 = v_dense_block(True, np.vectorize(diff_R_1n_func, otypes=[complex]),  bd)
                left_block2 = v_dense_block(True, np.vectorize(diff_R_2n_func, otypes=[complex]),  bd)
                right_block1 = v_diagonal_block(False, np.vectorize(diff_R_1n_func, otypes=[complex]),  bd)
                right_block2 = v_diagonal_block(False, np.vectorize(diff_R_2n_func, otypes=[complex]),  bd)
            left_zeros = np.zeros((row_height, col), dtype=complex)
            right_zeros = np.zeros((row_height, size - (col + 2* N + 2 * M)),dtype=complex)
            block_lst = [left_zeros, left_block1, left_block2, right_block1, right_block2, right_zeros]
            rows.append(np.concatenate(block_lst, axis = 1))
            col += 2 * N
        row_start += row_height

    ## Concatenate the rows of blocks into the square A matrix
    A = np.concatenate(rows, axis = 0)
    b = np.zeros(size, dtype=complex)

    index = 0

    # potential matching
    for boundary in range(boundary_count):
        if boundary == (boundary_count - 1): # i-e boundary
            for n in range(NMK[-2]):
                b[index] = b_potential_end_entry_old(n, boundary, h, d, heaving, a)
                index += 1
        else: # i-i boundary
            for n in range(NMK[boundary + (d[boundary] <= d[boundary + 1])]): # iterate over eigenfunctions for smaller h-d
                b[index] = b_potential_entry_old(n, boundary, d, heaving, h, a)
                index += 1

    # velocity matching
    for boundary in range(boundary_count):
        if boundary == (boundary_count - 1): # i-e boundary
            for n in range(NMK[-1]):
                b[index] = b_velocity_end_entry_old(n, boundary, heaving, a, h, d, m0, NMK)
                index += 1
        else: # i-i boundary
            for n in range(NMK[boundary + (d[boundary] > d[boundary + 1])]): # iterate over eigenfunctions for larger h-d
                b[index] = b_velocity_entry_old(n, boundary, heaving, a, d, h)
                index += 1
    return A, b