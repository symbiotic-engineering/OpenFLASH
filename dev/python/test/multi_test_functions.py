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

# This file needs to change or be scrapped, because code duplication is inefficient and error-prone.
# Potentially defer outside-of-notebook testing all to the package.

# This was previously a copy of the components of multi-meem and its equations used to
# calculate the appropriate values. For the sake of testing, we need one callable function instead
# of a jupyter notebook.

# Requires: h, d, a, heaving, m0, g, rho, NMK
class MultiEvaluator:
    def __init__(self, **constants):
        # Store constants as a dictionary
        self._constants = constants

        a = constants["a"]
        m0 = constants["m0"]
        NMK = constants["NMK"]
        h = constants["h"]
        g = constants["g"]
        
        # Add derived constants
        self._constants["scale"] = np.mean(a)
        self._constants["omega"] = np.sqrt(m0 * np.tanh(m0 * h) * g)
        self._constants["size"] = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
        self._constants["boundary_count"] = len(NMK) - 1

        self._inject_constants()

    def __getattr__(self, name):
        # Dynamically resolve constants by name
        if name in self._constants:
            return self._constants[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _inject_constants(self):
        # Dynamically add constants to the global scope of the methods
        global_vars = globals()
        for key, value in self._constants.items():
            global_vars[key] = value

    #############################################
    # Computational Functions here

    def A_matrix(self):
        # Initialize the A matrix with zeros
        A = np.zeros((size, size), dtype=complex)

        # Potential Matching
        col = 0
        row = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]
            if bd == (boundary_count - 1): # i-e boundary
                if bd == 0: # one cylinder
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + N + m] = - I_mk(n, m, bd) * Lambda_k(m, a[bd])
                else:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd)
                        A[row + n][col + N + n] = (h - d[bd]) * R_2n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = - I_mk(n, m, bd) * Lambda_k(m, a[bd])
                row += N
                    
            elif bd == 0:
                left_diag = d[bd] > d[bd + 1] # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + N + m] = - I_nm(n, m, bd) * R_1n(m, a[bd], bd + 1)
                            A[row + n][col + N + M + m] = - I_nm(n, m, bd) * R_2n(m, a[bd], bd + 1)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = I_nm(n, m, bd) * R_1n(n, a[bd], bd)
                        A[row + m][col + N + m] = - (h - d[bd + 1]) * R_1n(m, a[bd], bd + 1)
                        A[row + m][col + N + M + m] = - (h - d[bd + 1]) * R_2n(m, a[bd], bd + 1)
                    row += M
                col += N
            else: # i-i boundary
                left_diag = d[bd] > d[bd + 1] # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = (h - d[bd]) * R_1n(n, a[bd], bd)
                        A[row + n][col + N + n] = (h - d[bd]) * R_2n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = - I_nm(n, m, bd) * R_1n(m, a[bd], bd + 1)
                            A[row + n][col + 2*N + M + m] = - I_nm(n, m, bd) * R_2n(m, a[bd], bd + 1)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = I_nm(n, m, bd) * R_1n(n, a[bd], bd)
                            A[row + m][col + N + n] = I_nm(n, m, bd) * R_2n(n, a[bd], bd)
                        A[row + m][col + 2*N + m] = - (h - d[bd + 1]) * R_1n(m, a[bd], bd + 1)
                        A[row + m][col + 2*N + M + m] = - (h - d[bd + 1]) * R_2n(m, a[bd], bd + 1)
                    row += M
                col += 2 * N
        
        # Velocity Matching 
        col = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]
            if bd == (boundary_count - 1): # i-e boundary
                if bd == 0: # one cylinder
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - I_mk(n, m, bd) * diff_R_1n(n, a[bd], bd)
                        A[row + m][col + N + m] = h * diff_Lambda_k(m, a[bd])
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - I_mk(n, m, bd) * diff_R_1n(n, a[bd], bd)
                            A[row + m][col + N + n] = - I_mk(n, m, bd) * diff_R_2n(n, a[bd], bd)
                        A[row + m][col + 2*N + m] = h * diff_Lambda_k(m, a[bd])
                row += N
                    
            elif bd == 0:
                left_diag = d[bd] < d[bd + 1] # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = - (h - d[bd]) * diff_R_1n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + N + m] = I_nm(n, m, bd) * diff_R_1n(m, a[bd], bd + 1)
                            A[row + n][col + N + M + m] = I_nm(n, m, bd) * diff_R_2n(m, a[bd], bd + 1)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - I_nm(n, m, bd) * diff_R_1n(n, a[bd], bd)
                        A[row + m][col + N + m] = (h - d[bd + 1]) * diff_R_1n(m, a[bd], bd + 1)
                        A[row + m][col + N + M + m] = (h - d[bd + 1]) * diff_R_2n(m, a[bd], bd + 1)
                    row += M
                col += N
            else: # i-i boundary
                left_diag = d[bd] < d[bd + 1] # which of the two regions gets diagonal entries
                if left_diag:
                    for n in range(N):
                        A[row + n][col + n] = - (h - d[bd]) * diff_R_1n(n, a[bd], bd)
                        A[row + n][col + N + n] = - (h - d[bd]) * diff_R_2n(n, a[bd], bd)
                        for m in range(M):
                            A[row + n][col + 2*N + m] = I_nm(n, m, bd) * diff_R_1n(m, a[bd], bd + 1)
                            A[row + n][col + 2*N + M + m] = I_nm(n, m, bd) * diff_R_2n(m, a[bd], bd + 1)
                    row += N
                else:
                    for m in range(M):
                        for n in range(N):
                            A[row + m][col + n] = - I_nm(n, m, bd) * diff_R_1n(n, a[bd], bd)
                            A[row + m][col + N + n] = - I_nm(n, m, bd) * diff_R_2n(n, a[bd], bd)
                        A[row + m][col + 2*N + m] = (h - d[bd + 1]) * diff_R_1n(m, a[bd], bd + 1)
                        A[row + m][col + 2*N + M + m] = (h - d[bd + 1]) * diff_R_2n(m, a[bd], bd + 1)
                    row += M
                col += 2 * N
        return A

    def b_vector(self):
        b = np.zeros(size, dtype=complex)

        index = 0
        
        # potential matching
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1): # i-e boundary
                for n in range(NMK[-2]):
                    b[index] = b_potential_end_entry(n, boundary)
                    index += 1
            else: # i-i boundary
                for n in range(NMK[boundary + (d[boundary] < d[boundary + 1])]): # iterate over eigenfunctions for smaller h-d
                    b[index] = b_potential_entry(n, boundary)
                    index += 1
        
        # velocity matching
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1): # i-e boundary
                for n in range(NMK[-1]):
                    b[index] = b_velocity_end_entry(n, boundary)
                    index += 1
            else: # i-i boundary
                for n in range(NMK[boundary + (d[boundary] > d[boundary + 1])]): # iterate over eigenfunctions for larger h-d
                    b[index] = b_velocity_entry(n, boundary)
                    index += 1
        return b

    def c_vector(self):
        c = np.zeros((size - NMK[-1]), dtype=complex)
        col = 0
        for n in range(NMK[0]):
            c[n] = heaving[0] * int_R_1n(0, n)* z_n_d(n)
        col += NMK[0]
        for i in range(1, boundary_count):
            M = NMK[i]
            for m in range(M):
                c[col + m] = heaving[i] * int_R_1n(i, m)* z_n_d(m)
                c[col + M + m] = heaving[i] * int_R_2n(i, m)* z_n_d(m)
            col += 2 * M
        return c

    def Ab_coefficients(self, A, b):
        X = linalg.solve(A,b)
        Cs = []
        row = 0
        Cs.append(X[:NMK[0]])
        row += NMK[0]
        for i in range(1, boundary_count):
            Cs.append(X[row: row + NMK[i] * 2])
            row += NMK[i] * 2
        Cs.append(X[row:])
        self._constants["Cs"] = Cs
        self._inject_constants()
        return X
    
    def hydro_coefficients(self, X):
        hydro_p_terms = np.zeros(boundary_count, dtype=complex)
        for i in range(boundary_count):
            hydro_p_terms[i] = heaving[i] * int_phi_p_i_no_coef(i)

        max_rad = a[0]
        for i in range(boundary_count - 1, 0, -1):
            if heaving[i]:
                max_rad = a[i]
                break

        hydro_coef = 2 * pi * (np.dot(self.c_vector(), X[:-NMK[-1]]) + sum(hydro_p_terms))
        hydro_coef_real = hydro_coef.real * h **3 * rho
        hydro_coef_imag = hydro_coef.imag * omega * h**3 * rho
        hydro_coef_nondim = h**3/(max_rad**3 * pi)*hydro_coef

        return [hydro_coef_real, hydro_coef_imag, np.real(hydro_coef_nondim), np.imag(hydro_coef_nondim)]

    def potential_matrix(self):
        # Format Potential Matrix for Testing:
        r_vec = np.linspace(0.0, 2*a[-1], num=50)
        z_vec = np.linspace(0, -h, num=50) #h
        
        R, Z = np.meshgrid(r_vec, z_vec)
        
        regions = []
        regions.append((R <= a[0]) & (Z < -d[0]))
        for i in range(1, boundary_count):
            regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
        regions.append(R > a[-1])
        
        phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        
        for n in range(NMK[0]):
            temp_phiH = phi_h_n_inner_func(n, R[regions[0]], Z[regions[0]])
            phiH[regions[0]] = temp_phiH if n == 0 else phiH[regions[0]] + temp_phiH
        
        for i in range(1, boundary_count):
            for m in range(NMK[i]):
                temp_phiH = phi_h_m_i_func(i, m, R[regions[i]], Z[regions[i]])
                phiH[regions[i]] = temp_phiH if m == 0 else phiH[regions[i]] + temp_phiH
        
        for k in range(NMK[-1]):
            temp_phiH = phi_e_k_func(k, R[regions[-1]], Z[regions[-1]])
            phiH[regions[-1]] = temp_phiH if k == 0 else phiH[regions[-1]] + temp_phiH
        
        phi_p_i_vec = np.vectorize(phi_p_i)
        
        phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
        for i in range(1, boundary_count):
            phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
        phiP[regions[-1]] = 0
        
        phi = phiH + phiP
        
        nanregions = []
        nanregions.append((R <= a[0]) & (Z > -d[0]))
        for i in range(1, len(a)):
            nanregions.append((R > a[i-1]) & (R <= a[i]) & (Z > -d[i]))

        return R, Z, phi, nanregions
    
#############################################
# Functions to help potentials at points
def phi_h_n_inner_func(n, r, z):
    return (Cs[0][n] * R_1n(n, r, 0)) * Z_n_i(n, z, 0)

def phi_h_m_i_func(i, m, r, z):
    return (Cs[i][m] * R_1n(m, r, i) + Cs[i][NMK[i] + m] * R_2n(m, r, i)) * Z_n_i(m, z, i)

def phi_e_k_func(k, r, z):
    return Cs[-1][k] * Lambda_k(k, r) * Z_n_e(k, z)

#############################################
# Equations begin here

def wavenumber(omega):
    m0_err = (lambda m0: (m0 * np.tanh(h * m0) - omega ** 2 / g))
    return (root_scalar(m0_err, x0 = 2, method="newton")).root

#############################################
# some common computations

# Defining m_k function that will be used later on
def m_k(k):
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


def m_k_newton(h):
    res = newton(lambda k: k * np.tanh(k * h) - m0**2 / 9.8, x0=1.0, tol=10 ** (-10))
    return res

def lambda_ni(n, i):
    return n * pi / (h - d[i])

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
        if lambda1 == lambda2:
            return (h - dj) + sin(lambda1 * 2 * (h - dj)) / (2 * lambda1)
        else:
            frac1 = sin((lambda1 + lambda2)*(h-dj))/(lambda1 + lambda2)
            frac2 = sin((lambda1 - lambda2)*(h-dj))/(lambda1 - lambda2)
            return frac1 + frac2

def I_mk(m, k, i): # coupling integral for i and e-type regions
    dj = d[i]
    if m == 0 and k == 0:
        return (1/sqrt(N_k(0))) * sinh(m0 * (h - dj)) / m0
    if m == 0 and k >= 1:
        mk = m_k(k)
        return (1/sqrt(N_k(k))) * sin(mk * (h - dj)) / mk
    if m >= 1 and k == 0:
        num = sqrt(2) * (1/sqrt(N_k(0))) * m0 * (-1)**m * sinh(m0 * (h - dj))
        denom = (m0**2 + lambda_ni(m, i) **2)
        return num/denom
    else:
        lambda1 = lambda_ni(m, i)
        mk = m_k(k)
        if abs(mk) == lambda1:
            return (h - dj)/2
        else:
            frac1 = sin((mk + lambda1)*(h-dj))/(mk + lambda1)
            frac2 = sin((mk - lambda1)*(h-dj))/(mk - lambda1)
            return sqrt(2)/2 * (1/sqrt(N_k(k))) * (frac1 + frac2)

#############################################
# b-vector computation

def b_potential_entry(n, i): # for two i-type regions
    #(integrate over shorter fluid, use shorter fluid eigenfunction)
    
    j = i + (d[i] < d[i+1]) # index of shorter fluid
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
        return constant * (1/sqrt(N_k(0))) * sinh(m0 * (h - d[i])) / m0
    else:
        return constant * (1/sqrt(N_k(k))) * sin(m_k(k) * (h - d[i])) / m_k(k)


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
        return besseli(0, lambda_ni(n, i) * r) / besseli(0, lambda_ni(n, i) * scale)
    else: 
        raise ValueError("Invalid value for n")

# Differentiate wrt r
def diff_R_1n(n, r, i):
    if n == 0:
        return 0
    else:
        top = lambda_ni(n, i) * besseli(1, lambda_ni(n, i) * r)
        bottom = besseli(0, lambda_ni(n, i) * scale)
        return top / bottom

#############################################
# The "Bessel K" radial eigenfunction
def R_2n(n, r, i): # this shouldn't be called for i=0, innermost.
    if n == 0:
        return 0.5 * np.log(r / a[i])
    else:
        return besselk(0, lambda_ni(n, i) * r) / besselk(0, lambda_ni(n, i) * scale)


# Differentiate wrt r
def diff_R_2n(n, r, i):
    if n == 0:
        return 1 / (2 * r)
    else:
        top = - lambda_ni(n, i) * besselk(1, lambda_ni(n, i) * r)
        bottom = besselk(0, lambda_ni(n, i) * scale)
        return top / bottom


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
        return besselh(0, m0 * r) / besselh(0, m0 * scale)
    else:
        return besselk(0, m_k(k) * r) / besselk(0, m_k(k) * scale)

# Differentiate wrt r
def diff_Lambda_k(k, r):
    if k == 0:
        numerator = -(m0 * besselh(1, m0 * r))
        denominator = besselh(0, m0 * scale)
    else:
        mk = m_k(k)
        numerator = -(mk * besselk(1, mk * r))
        denominator = besselk(0, mk * scale)
    return numerator / denominator


#############################################
# Equation 2.34 in analytical methods book, also eq 16 in Seah and Yeung 2006:
def N_k(k):
    if k == 0:
        return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    else:
        return 1 / 2 * (1 + sin(2 * m_k(k) * h) / (2 * m_k(k) * h))


#############################################
# e-region vertical eigenfunctions
def Z_n_e(k, z):
    if k == 0:
        return 1 / sqrt(N_k(k)) * cosh(m0 * (z + h))
    else:
        return 1 / sqrt(N_k(k)) * cos(m_k(k) * (z + h))

def diff_Z_n_e(k, z):
    if k == 0:
        return 1 / sqrt(N_k(k)) * m0 * sinh(m0 * (z + h))
    else:
        mk = m_k(k)
        return -1 / sqrt(N_k(k)) * mk * sin(mk * (z + h))

#############################################
# To calculate hydrocoefficients

#integrating R_1n * r
def int_R_1n(i, n):
    lambda0 = lambda_ni(n, i)
    if i == 0:
        if n == 0:
            return a[i]**2/4
        else:
            top = a[i] * besseli(1, lambda0 * a[i])
            bottom = lambda0 * besseli(0, lambda0 * scale)
            return top/bottom
    else:
        if n == 0:
            return a[i]**2/4 - a[i-1]**2/4
        else:
            top = a[i] * besseli(1, lambda0 * a[i]) - a[i-1] * besseli(1, lambda0 * a[i-1])
            bottom = lambda0 * besseli(0, lambda0 * scale)
            return top / bottom

#integrating R_2n * r
def int_R_2n(i, n):
    lambda0 = lambda_ni(n, i)
    if n == 0:
        return (a[i-1]**2 * (2*np.log(a[i]) - 2*np.log(a[i-1]) + 1) - a[i]**2)/8
    else:
        top = a[i] * besselk(1, lambda0 * a[i]) - a[i-1] * besselk(1, lambda0 * a[i-1])
        bottom = - lambda0 * besselk(0, lambda0 * scale)
        return top / bottom

#integrating phi_p_i * d_phi_p_i/dz * r *d_r at z=d[i]
def int_phi_p_i_no_coef(i):
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