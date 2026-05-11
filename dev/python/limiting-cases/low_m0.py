import numpy as np
from scipy.special import hankel1 as besselh
import scipy.linalg as linalg
from numpy import log, pi, sqrt

g = 9.81

def low_particular_potential(h, d, r, n, heaving):
    if n >= len(d): # free surface region
        return 0
    if heaving[n]:
        return - (r**2)/(4 * (h - d[n]))
    else:
        return 0

def low_diff_particular_potential_total(a, r, n, heaving):
    if n >= len(a): # free surface region
        return 0
    if heaving[n]:
        return - r/2
    else:
        return 0

def low_b_potential_entry(h, d, a, n, heaving):
    return - low_particular_potential(h, d, a[n], n, heaving) + low_particular_potential(h, d, a[n], n + 1, heaving)

def low_b_velocity_entry(a, n, heaving):
    return - low_diff_particular_potential_total(a, a[n], n, heaving) + low_diff_particular_potential_total(a, a[n], n + 1, heaving)
    
# sets potential equation for nth boundary into the A matrix
def low_potential_row(a, A, n, m0, bds):
    if n == 0:
        if bds == 1: # single cylinder
            A[0][0] = 1
            A[0][1] = - besselh(0, m0 * a[0])
        else:
            A[0][0] = 1
            A[0][1] = - 1
            A[0][2] = - log(a[0])
    elif n == bds - 1:
        A[n][2*n - 1] = 1
        A[n][2*n] = log(a[n])
        A[n][2*n + 1] = - besselh(0, m0 * a[n])
    else:
        A[n][2*n - 1] = 1
        A[n][2*n] = log(a[n])
        A[n][2*n + 1] = - 1
        A[n][2*n + 2] = - log(a[n])

def low_velocity_row(h, d, a, A, n, m0, bds):
    if n == 0:
        if bds == 1: # single cylinder
            A[bds][0] = 0
            A[bds][1] = h * m0 * besselh(1, m0 * a[0])
        else:
            A[bds][0] = 0
            A[bds][1] = 0
            A[bds][2] = - 1/a[0] * (h - d[1])
    elif n == bds - 1:
        A[bds + n][2*n - 1] = 0
        A[bds + n][2*n] = 1/a[n] * (h - d[n])
        A[bds + n][2*n + 1] = h * m0 * besselh(1, m0 * a[n])
    else:
        A[bds + n][2*n - 1] = 0
        A[bds + n][2*n] = 1/a[n] * (h - d[n])
        A[bds + n][2*n + 1] = 0
        A[bds + n][2*n + 2] = - 1/a[n] * (h - d[n + 1])

def low_build_A(h, d, a, m0):
    bds = len(a)
    A = np.zeros((2 * bds, 2 * bds), dtype = complex)
    for n in range(bds):
        low_potential_row(a, A, n, m0, bds)
        low_velocity_row(h, d, a, A, n, m0, bds)
    return A

def low_build_B(h, d, a, heaving):
    f1 = lambda x : low_b_potential_entry(h, d, a, x, heaving)
    f2 = lambda x : low_b_velocity_entry(a, x, heaving)
    b1 = (np.vectorize(f1, otypes = [float]))(list(range(len(a))))
    b2 = (np.vectorize(f2, otypes = [float]))(list(range(len(a))))
    return np.concatenate([b1, b2])
    
# modifies A matrix for a particular m0, all other parameters the same.
def low_A_new_m0(h, a, A, m0):
    bds = len(a)
    A[bds-1][2*bds-1] = - besselh(0, m0 * a[-1]) 
    A[2*bds-1][2*bds-1] = h * m0 * besselh(1, m0 * a[-1])

# hydro coefficient calculation functions
def low_particular_potential_int_eval(h, d, a, region, boundary):
    return -(a[boundary]**4)/(16 * (h - d[region]))

def low_total_particular(h, d, a, heaving):
    accumulator = 0
    for region in range(len(a)):
        if heaving[region]:
            if region == 0:
                accumulator += low_particular_potential_int_eval(h, d, a, 0, 0)
            else:
                accumulator += low_particular_potential_int_eval(h, d, a, region, region) - low_particular_potential_int_eval(h, d, a, region, region - 1)
    return accumulator

#dphi/dz mandated to be 1 in the heaving regions, so just integrate potential * r

def low_ln_potential_int_eval(a, bd):
    return (a[bd]**2/2) * (log(a[bd]) - 1/2)

def low_const_potential_int_eval(a, bd):
    return (a[bd]**2/2)

def low_create_c_vector(a, heaving):
    c = []
    for region in range(len(a)):
        if region == 0:
            if heaving[0]:
                c.append(low_const_potential_int_eval(a, 0))
            else:
                c.append(0)
        else:
            if heaving[region]:
                c.append(low_const_potential_int_eval(a, region) - low_const_potential_int_eval(a, region - 1))
                c.append(low_ln_potential_int_eval(a, region) - low_ln_potential_int_eval(a, region - 1))
            else:
                c.append(0)
                c.append(0)
    return c

def low_get_hydro_coeffs(h, d, a, heaving, X, m0, rho = 1023):
    const = low_total_particular(h, d, a, heaving)
    c = low_create_c_vector(a, heaving)
    raw = np.dot(X, c) + const
    total = 2 * pi * rho * raw #* h**3 
    return np.real(total), (np.imag(total) * low_omega_from_m0(h, m0))

def low_to_nondim(coeff, a_norm, rho = 1023):
    return coeff/(pi * a_norm**3 * rho)

def low_omega_from_m0(h, m0): # at small m0 approximation
    return sqrt(m0**2 * h * g)

def low_get_max_heaving_radius(a, heaving):
    max_rad = a[0]
    for i in range(len(a) - 1, 0, -1):
        if heaving[i]:
            max_rad = a[i]
            break
    return max_rad

def low_get_nondim_hydros(h, d, a, heaving, m0s):
    a_norm = low_get_max_heaving_radius(a, heaving)
    ams, dps = low_get_hydros(h, d, a, heaving, m0s)
    nondim_am = [low_to_nondim(added_mass, a_norm) for added_mass in ams]
    nondim_dp = [low_to_nondim(damping, a_norm) for damping in dps]
    return nondim_am, nondim_dp

def low_get_hydros(h, d, a, heaving, m0s):
    A = low_build_A(h, d, a, m0s[0])
    b = low_build_B(h, d, a, heaving)
    solutions = []
    for m0 in m0s:
        low_A_new_m0(h, a, A, m0)
        solutions.append(linalg.solve(A, b))
    ams = []
    dps = []
    for i in range(len(m0s)):
        added_mass, damping = low_get_hydro_coeffs(h, d, a, heaving, solutions[i][:-1], m0s[i])
        ams.append(added_mass)
        dps.append(damping)
    return ams, dps