# For notebooks that don't deal with all the inner MEEM workings, and don't want to duplicate A-matrix, b-vector code, etc.
# Useful for just generating hydro coefficients, test cases.
# This file may not reflect recent edits to multi_equations, multi-MEEM.

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
import pandas as pd

g = 9.81

# generic helper function
def insert_submatrix(mat, submat, row, col):
        mat = mat.copy()  # Avoid modifying original A
        rows, cols = submat.shape
        mat[row:row+rows, col:col+cols] = submat
        return mat

class Problem:
    def __init__(self, h, d, a, heaving, NMK, m0, rho, scale = None):
        self.h = h
        self.d = d
        self.a = a
        self.heaving = heaving
        self.NMK = NMK
        self.m0 = m0
        self.rho = rho
        self.scale = a if scale is None else scale
        self.size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
        self.boundary_count = len(NMK) - 1
        self.m_k = self.m_k_array()

    def angular_freq(self, m0): # omega
        if m0 == inf:
            return inf
        else:
            return sqrt(m0 * np.tanh(m0 * self.h) * g)
        
    def wavenumber(self, omega): # m0
        m0_err = (lambda m0: (m0 * np.tanh(self.h * m0) - omega ** 2 / g))
        return (root_scalar(m0_err, x0 = 2, method="newton")).root
    
    def lambda_ni(self, n, i): # factor used often in calculations
        return n * pi / (self.h - self.d[i])
    
    #############################################
    def m_k_entry(self, k):
      m0 = self.m0
      h = self.h
      if k == 0: return m0
      elif m0 == inf:
          return ((k - 1/2) * pi)/h

      m_k_h_err = (lambda m_k_h: (m_k_h * np.tan(m_k_h) + m0 * h * np.tanh(m0 * h)))
      k_idx = k

      # becca's version of bounds from MDOcean Matlab code
      m_k_h_lower = pi * (k_idx - 1/2) + (pi/180)* np.finfo(float).eps * (2**(np.floor(np.log(180*(k_idx- 1/2)) / np.log(2))) + 1)
      m_k_h_upper = pi * k_idx

      m_k_initial_guess = pi * (k_idx - 1/2) + np.finfo(float).eps
      result = root_scalar(m_k_h_err, x0=m_k_initial_guess, method="newton", bracket=[m_k_h_lower, m_k_h_upper])

      m_k_val = result.root / h
      return m_k_val
    
    def m_k_array(self): # create an array of m_k values for each k to avoid recomputation
        m_k = (np.vectorize(self.m_k_entry, otypes = [float]))(list(range(self.NMK[-1])))
        return m_k
    
    def N_k(self, k):
        h, m0, m_k = self.h, self.m0, self.m_k
        if m0 == inf: return 1/2
        elif k == 0:
            return 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
        else:
            return 1 / 2 * (1 + sin(2 * m_k[k] * h) / (2 * m_k[k] * h))
    
    #############################################
    # vertical eigenvector coupling computation
    def I_nm(self, n, m, i): # coupling integral for two i-type regions
        h, d = self.h, self.d
        dj = max(d[i], d[i+1]) # integration bounds at -h and -d
        if n == 0 and m == 0:
            return h - dj
        lambda1 = self.lambda_ni(n, i)
        lambda2 = self.lambda_ni(m, i + 1)
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

    def I_mk(self, m, k, i): # coupling integral for i and e-type regions
        h, m0, m_k, N_k = self.h, self.m0, self.m_k, self.N_k
        dj = self.d[i]
        if m == 0 and k == 0:
            if m0 == inf: return 0
            elif m0 * h < 14:
                return (1/sqrt(N_k(0))) * sinh(m0 * (h - dj)) / m0
            else: # high m0h approximation
                return sqrt(2 * h / m0) * (exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
        if m == 0 and k >= 1:
            return (1/sqrt(N_k(k))) * sin(m_k[k] * (h - dj)) / m_k[k]
        if m >= 1 and k == 0:
            if m0 == inf: return 0
            elif m0 * h < 14:
                num = (-1)**m * sqrt(2) * (1/sqrt(N_k(0))) * m0 * sinh(m0 * (h - dj))
            else: # high m0h approximation
                num = (-1)**m * 2 * sqrt(h * m0 ** 3) *(exp(- m0 * dj) - exp(m0 * dj - 2 * m0 * h))
            denom = (m0**2 + self.lambda_ni(m, i) **2)
            return num/denom
        else:
            lambda1 = self.lambda_ni(m, i)
            if abs(m_k[k]) == lambda1:
                return sqrt(2/N_k(k)) * (h - dj)/2
            else:
                frac1 = sin((m_k[k] + lambda1)*(h-dj))/(m_k[k] + lambda1)
                frac2 = sin((m_k[k] - lambda1)*(h-dj))/(m_k[k] - lambda1)
                return sqrt(2/N_k(k)) * (frac1 + frac2)/2

    def I_nm_vals(self): # Computes all necessary I_nm
      NMK, boundary_count = self.NMK, self.boundary_count        
      I_nm_vals = np.zeros((max(NMK), max(NMK), boundary_count - 1), dtype = complex)
      for bd in range(boundary_count - 1):
          for n in range(NMK[bd]):
              for m in range(NMK[bd + 1]):
                  I_nm_vals[n][m][bd] = self.I_nm(n, m, bd)
      return I_nm_vals
    
    def I_mk_vals(self): # Computes all necessary I_mk
      NMK, boundary_count = self.NMK, self.boundary_count        
      I_mk_vals = np.zeros((NMK[boundary_count - 1], NMK[boundary_count]), dtype = complex)
      for m in range(NMK[boundary_count - 1]):
          for k in range(NMK[boundary_count]):
              I_mk_vals[m][k]= self.I_mk(m, k, boundary_count - 1)
      return I_mk_vals

    #############################################
    def b_vector(self):
        b = np.zeros(self.size, dtype=complex)
        index = 0
        d, boundary_count, NMK = self.d, self.boundary_count, self.NMK

        # potential matching
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1): # i-e boundary
                for n in range(NMK[-2]):
                    b[index] = self.b_potential_end_entry(n, boundary)
                    index += 1
            else: # i-i boundary
                for n in range(self.NMK[boundary + (d[boundary] <= d[boundary + 1])]): # iterate over eigenfunctions for smaller h-d
                    b[index] = self.b_potential_entry(n, boundary)
                    index += 1

        # velocity matching
        for boundary in range(boundary_count):
            if boundary == (boundary_count - 1): # i-e boundary
                for n in range(NMK[-1]):
                    b[index] = self.b_velocity_end_entry(n, boundary)
                    index += 1
            else: # i-i boundary
                for n in range(NMK[boundary + (d[boundary] > d[boundary + 1])]): # iterate over eigenfunctions for larger h-d
                    b[index] = self.b_velocity_entry(n, boundary)
                    index += 1
        return b
    
    def b_potential_entry(self, n, i): # for two i-type regions
        #(integrate over shorter fluid, use shorter fluid eigenfunction)
        h, d, a, heaving = self.h, self.d, self.a, self.heaving
        j = i + (d[i] <= d[i+1]) # index of shorter fluid
        constant = (heaving[i+1] / (h - d[i+1]) - heaving[i] / (h - d[i]))
        if n == 0:
            return constant * 1/2 * ((h - d[j])**3/3 - (h-d[j]) * a[i]**2/2)
        else:
            return sqrt(2) * (h - d[j]) * constant * ((-1) ** n)/(self.lambda_ni(n, j) ** 2)

    def b_potential_end_entry(self, n, i): # between i and e-type regions
        h, d, a, heaving = self.h, self.d, self.a, self.heaving
        constant = - heaving[i] / (h - d[i])
        if n == 0:
            return constant * 1/2 * ((h - d[i])**3/3 - (h-d[i]) * a[i]**2/2)
        else:
            return sqrt(2) * (h - d[i]) * constant * ((-1) ** n)/(self.lambda_ni(n, i) ** 2)

    def b_velocity_entry(self, n, i): # for two i-type regions
        h, d, a, heaving = self.h, self.d, self.a, self.heaving
        if n == 0:
            return (heaving[i+1] - heaving[i]) * (a[i]/2)
        if d[i] > d[i + 1]: #using i+1's vertical eigenvectors
            if heaving[i]:
                num = - sqrt(2) * a[i] * sin(self.lambda_ni(n, i+1) * (h-d[i]))
                denom = (2 * (h - d[i]) * self.lambda_ni(n, i+1))
                return num/denom
            else: return 0
        else: #using i's vertical eigenvectors
            if heaving[i+1]:
                num = sqrt(2) * a[i] * sin(self.lambda_ni(n, i) * (h-d[i+1]))
                denom = (2 * (h - d[i+1]) * self.lambda_ni(n, i))
                return num/denom
            else: return 0

    def b_velocity_end_entry(self, k, i): # between i and e-type regions
        h, d, a, heaving, m0, m_k = self.h, self.d, self.a, self.heaving, self.m0, self.m_k
        constant = - heaving[i] * a[i]/(2 * (h - d[i]))
        if k == 0:
            if m0 == inf: return 0
            elif m0 * h < 14:
                return constant * (1/sqrt(self.N_k(0))) * sinh(m0 * (h - d[i])) / m0
            else: # high m0h approximation
                return constant * sqrt(2 * h / m0) * (exp(- m0 * d[i]) - exp(m0 * d[i] - 2 * m0 * h))
        else:
            return constant * (1/sqrt(self.N_k(k))) * sin(m_k[k] * (h - d[i])) / m_k[k]
        
    #############################################
    # Eigenfunctions and derivatives, inner regions
    # The "Bessel I" radial eigenfunction
    def R_1n(self, n, r, i):
        scale, lambda_ni = self.scale, self.lambda_ni
        if n == 0:
            return 0.5
        elif n >= 1:
            if r == scale[i]:
                return 1
            else:
                return besselie(0, lambda_ni(n, i) * r) / besselie(0, lambda_ni(n, i) * scale[i]) * exp(lambda_ni(n, i) * (r - scale[i]))
        else: 
            raise ValueError("Invalid value for n")

    # Bessel I, differentiated wrt r
    def diff_R_1n(self, n, r, i):
        scale, lambda_ni = self.scale, self.lambda_ni
        if n == 0:
            return 0
        else:
            top = lambda_ni(n, i) * besselie(1, lambda_ni(n, i) * r)
            bottom = besselie(0, lambda_ni(n, i) * scale[i])
            return top / bottom * exp(lambda_ni(n, i) * (r - scale[i]))

    # The "Bessel K" radial eigenfunction
    def R_2n(self, n, r, i):
        scale, lambda_ni = self.scale, self.lambda_ni
        if i == 0:
            raise ValueError("i cannot be 0")  # this shouldn't be called for i=0, innermost region.
        elif n == 0:
            return 0.5 * np.log(r / self.a[i])
        else:
            if r == scale[i]:
                return 1
            else:
                return besselke(0, lambda_ni(n, i) * r) / besselke(0, lambda_ni(n, i) * scale[i]) * exp(lambda_ni(n, i) * (scale[i] - r))

    # Bessel K, differentiated wrt r
    def diff_R_2n(self, n, r, i):
        scale, lambda_ni = self.scale, self.lambda_ni
        if n == 0:
            return 1 / (2 * r)
        else:
            top = - lambda_ni(n, i) * besselke(1, lambda_ni(n, i) * r)
            bottom = besselke(0, lambda_ni(n, i) * scale[i])
            return top / bottom * exp(lambda_ni(n, i) * (scale[i] - r))

    # i-region vertical eigenfunction
    def Z_n_i(self, n, z, i):
        if n == 0:
            return 1
        else:
            return np.sqrt(2) * np.cos(self.lambda_ni(n, i) * (z + self.h))

    # i-region vertical eigenfunction, differentiated wrt z
    def diff_Z_n_i(self, n, z, i):
        if n == 0:
            return 0
        else:
            lambda0 = self.lambda_ni(n, i)
            return - lambda0 * np.sqrt(2) * np.sin(lambda0 * (z + self.h))

    #############################################
    # Eigenfunctions and derivatives, outer region
    # e-region radial eigenfunction
    def Lambda_k(self, k, r):
        m0, m_k, scale = self.m0, self.m_k, self.scale
        if k == 0:
            if m0 == inf:
            # the true limit is not well-defined, but whatever value this returns will be multiplied by zero
                return 1
            else:
                if r == scale[-1]:
                    return 1
                else:
                    return besselh(0, m0 * r) / besselh(0, m0 * scale[-1])
        else:
            if r == scale[-1]:
                return 1
            else:
                return besselke(0, m_k[k] * r) / besselke(0, m_k[k] * scale[-1]) * exp(m_k[k] * (scale[-1] - r))

    # e-region radial eigenfunction, differentiated wrt r
    def diff_Lambda_k(self, k, r):
        m0, m_k, scale = self.m0, self.m_k, self.scale
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

    # e-region vertical eigenfunction
    def Z_k_e(self, k, z):
        h, m0, m_k, N_k = self.h, self.m0, self.m_k, self.N_k
        if k == 0:
            if m0 == inf: return 0
            elif m0 * h < 14:
                return 1 / sqrt(N_k(k)) * cosh(m0 * (z + h))
            else: # high m0h approximation
                return sqrt(2 * m0 * h) * (exp(m0 * z) + exp(-m0 * (z + 2*h)))
        else:
            return 1 / sqrt(N_k(k)) * cos(m_k[k] * (z + h))

    # e-region vertical eigenfunction, differentiated wrt z
    def diff_Z_k_e(self, k, z):
        h, m0, m_k, N_k = self.h, self.m0, self.m_k, self.N_k
        if k == 0:
            if m0 == inf: return 0
            elif m0 * h < 14:
                return 1 / sqrt(N_k(k)) * m0 * sinh(m0 * (z + h))
            else: # high m0h approximation
                return m0 * sqrt(2 * h * m0) * (exp(m0 * z) - exp(-m0 * (z + 2*h)))
        else:
            return -1 / sqrt(N_k(k)) * m_k[k] * sin(m_k[k] * (z + h))
        
    #############################################
    # A matrix calculations
    def a_matrix(self):
        d, NMK, boundary_count, size = self.d, self.NMK, self.boundary_count, self.size
        # localize eigenfunctions
        R_1n, R_2n, diff_R_1n, diff_R_2n = self.R_1n, self.R_2n, self.diff_R_1n, self.diff_R_2n
        # localize block functions
        p_diagonal_block = self.p_diagonal_block
        p_dense_block, p_dense_block_e = self.p_dense_block, self.p_dense_block_e
        v_diagonal_block, v_diagonal_block_e = self.v_diagonal_block, self.v_diagonal_block_e
        v_dense_block, v_dense_block_e = self.v_dense_block, self.v_dense_block_e

        # compute the coupling integrals and store values
        I_nm_vals = self.I_nm_vals()
        I_mk_vals = self.I_mk_vals()

        rows = [] # collection of rows of blocks in A matrix, to be concatenated later

        # Potential Blocks
        col = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]
            if bd == (boundary_count - 1): # i-e boundary, inherently left diagonal
                row_height = N
                left_block1 = p_diagonal_block(True, np.vectorize(R_1n), bd)
                right_block = p_dense_block_e(bd, I_mk_vals)
                if bd == 0: # one cylinder
                    rows.append(np.concatenate((left_block1,right_block), axis = 1))
                else:
                    left_block2 = p_diagonal_block(True, np.vectorize(R_2n), bd)
                    left_zeros = np.zeros((row_height, col), dtype=complex)
                    rows.append(np.concatenate((left_zeros,left_block1,left_block2,right_block), axis = 1))
            elif bd == 0:
                left_diag = d[bd] > d[bd + 1] # which of the two regions gets diagonal entries
                if left_diag:
                    row_height = N
                    left_block = p_diagonal_block(True, np.vectorize(R_1n), 0)
                    right_block1 = p_dense_block(False, np.vectorize(R_1n), 0, I_nm_vals)
                    right_block2 = p_dense_block(False, np.vectorize(R_2n), 0, I_nm_vals)
                else:
                    row_height = M
                    left_block = p_dense_block(True, np.vectorize(R_1n), 0, I_nm_vals)
                    right_block1 = p_diagonal_block(False, np.vectorize(R_1n), 0)
                    right_block2 = p_diagonal_block(False, np.vectorize(R_2n), 0)
                right_zeros = np.zeros((row_height, size - (col + N + 2 * M)),dtype=complex)
                block_lst = [left_block, right_block1, right_block2, right_zeros]
                rows.append(np.concatenate(block_lst, axis = 1))
                col += N
            else: # i-i boundary
                left_diag = d[bd] > d[bd + 1] # which of the two regions gets diagonal entries
                if left_diag:
                    row_height = N
                    left_block1 = p_diagonal_block(True, np.vectorize(R_1n), bd)
                    left_block2 = p_diagonal_block(True, np.vectorize(R_2n), bd)
                    right_block1 = p_dense_block(False, np.vectorize(R_1n),  bd, I_nm_vals)
                    right_block2 = p_dense_block(False, np.vectorize(R_2n),  bd, I_nm_vals)
                else:
                    row_height = M
                    left_block1 = p_dense_block(True, np.vectorize(R_1n),  bd, I_nm_vals)
                    left_block2 = p_dense_block(True, np.vectorize(R_2n),  bd, I_nm_vals)
                    right_block1 = p_diagonal_block(False, np.vectorize(R_1n),  bd)
                    right_block2 = p_diagonal_block(False, np.vectorize(R_2n),  bd)
                left_zeros = np.zeros((row_height, col), dtype=complex)
                right_zeros = np.zeros((row_height, size - (col + 2 * N + 2 * M)),dtype=complex)
                block_lst = [left_zeros, left_block1, left_block2, right_block1, right_block2, right_zeros]
                rows.append(np.concatenate(block_lst, axis = 1))
                col += 2 * N

        # Velocity Blocks
        col = 0
        for bd in range(boundary_count):
            N = NMK[bd]
            M = NMK[bd + 1]
            if bd == (boundary_count - 1): # i-e boundary, inherently left diagonal
                row_height = M
                left_block1 = v_dense_block_e(np.vectorize(diff_R_1n, otypes=[complex]), bd, I_mk_vals)
                right_block = v_diagonal_block_e(bd)
                if bd == 0: # one cylinder
                    rows.append(np.concatenate((left_block1,right_block), axis = 1))
                else:
                    left_block2 = v_dense_block_e(np.vectorize(diff_R_2n, otypes=[complex]), bd, I_mk_vals)
                    left_zeros = np.zeros((row_height, col), dtype=complex)
                    rows.append(np.concatenate((left_zeros,left_block1,left_block2,right_block), axis = 1))
            elif bd == 0:
                left_diag = d[bd] <= d[bd + 1] # taller fluid region gets diagonal entries
                if left_diag:
                    row_height = N
                    left_block = v_diagonal_block(True, np.vectorize(diff_R_1n, otypes=[complex]), 0)
                    right_block1 = v_dense_block(False, np.vectorize(diff_R_1n, otypes=[complex]), 0, I_nm_vals)
                    right_block2 = v_dense_block(False, np.vectorize(diff_R_2n, otypes=[complex]), 0, I_nm_vals)
                else:
                    row_height = M
                    left_block = v_dense_block(True, np.vectorize(diff_R_1n, otypes=[complex]), 0, I_nm_vals)
                    right_block1 = v_diagonal_block(False, np.vectorize(diff_R_1n, otypes=[complex]), 0)
                    right_block2 = v_diagonal_block(False, np.vectorize(diff_R_2n, otypes=[complex]), 0)
                right_zeros = np.zeros((row_height, size - (col + N + 2 * M)),dtype=complex)
                block_lst = [left_block, right_block1, right_block2, right_zeros]
                rows.append(np.concatenate(block_lst, axis = 1))
                col += N
            else: # i-i boundary
                left_diag = d[bd] <= d[bd + 1] # taller fluid region gets diagonal entries
                if left_diag:
                    row_height = N
                    left_block1 = v_diagonal_block(True, np.vectorize(diff_R_1n, otypes=[complex]), bd)
                    left_block2 = v_diagonal_block(True, np.vectorize(diff_R_2n, otypes=[complex]), bd)
                    right_block1 = v_dense_block(False, np.vectorize(diff_R_1n, otypes=[complex]),  bd, I_nm_vals)
                    right_block2 = v_dense_block(False, np.vectorize(diff_R_2n, otypes=[complex]),  bd, I_nm_vals)
                else:
                    row_height = M
                    left_block1 = v_dense_block(True, np.vectorize(diff_R_1n, otypes=[complex]),  bd, I_nm_vals)
                    left_block2 = v_dense_block(True, np.vectorize(diff_R_2n, otypes=[complex]),  bd, I_nm_vals)
                    right_block1 = v_diagonal_block(False, np.vectorize(diff_R_1n, otypes=[complex]),  bd)
                    right_block2 = v_diagonal_block(False, np.vectorize(diff_R_2n, otypes=[complex]),  bd)
                left_zeros = np.zeros((row_height, col), dtype=complex)
                right_zeros = np.zeros((row_height, size - (col + 2* N + 2 * M)),dtype=complex)
                block_lst = [left_zeros, left_block1, left_block2, right_block1, right_block2, right_zeros]
                rows.append(np.concatenate(block_lst, axis = 1))
                col += 2 * N

        ## Concatenate the rows of blocks into the square A matrix
        return np.concatenate(rows, axis = 0)

    ### Blocks in the A matrix
    ## Potential blocks
    # arguments: diagonal block on left (T/F), vectorized radial eigenfunction, boundary number
    def p_diagonal_block(self, left, radfunction, bd):
        h, d, a, NMK = self.h, self.d, self.a, self.NMK
        region = bd if left else (bd + 1)
        sign = 1 if left else (-1)
        return sign * (h - d[region]) * np.diag(radfunction(list(range(NMK[region])), a[bd], region))
        
    # arguments: dense block on left (T/F), vectorized radial eigenfunction, boundary number
    def p_dense_block(self, left, radfunction, bd, I_nm_vals):
        a, NMK = self.a, self.NMK
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

    def p_dense_block_e(self, bd, I_mk_vals):
        a, NMK = self.a, self.NMK
        I_mk_array = I_mk_vals
        radial_vector = (np.vectorize(self.Lambda_k, otypes = [complex]))(list(range(NMK[bd+1])), a[bd])
        radial_array = np.outer((np.full((NMK[bd]), 1)), radial_vector)
        return (-1) * radial_array * I_mk_array

    ## Velocity blocks
    # arguments: diagonal block on left (T/F), vectorized radial eigenfunction, boundary number
    def v_diagonal_block(self, left, radfunction, bd):
        h, d, a, NMK = self.h, self.d, self.a, self.NMK
        region = bd if left else (bd + 1)
        sign = (-1) if left else (1)
        return sign * (h - d[region]) * np.diag(radfunction(list(range(NMK[region])), a[bd], region))

    # arguments: dense block on left (T/F), vectorized radial eigenfunction, boundary number
    def v_dense_block(self, left, radfunction, bd, I_nm_vals):
        a, NMK = self.a, self.NMK
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

    def v_diagonal_block_e(self, bd):
        h, a, NMK = self.h, self.a, self.NMK
        return h * np.diag((np.vectorize(self.diff_Lambda_k, otypes = [complex]))(list(range(NMK[bd+1])), a[bd]))

    def v_dense_block_e(self, radfunction, bd, I_mk_vals): # for region adjacent to e-type region
        I_km_array = np.transpose(I_mk_vals)
        a, NMK = self.a, self.NMK
        radial_vector = radfunction(list(range(NMK[bd])), a[bd], bd)
        radial_array = np.outer((np.full((NMK[bd + 1]), 1)), radial_vector)
        return (-1) * radial_array * I_km_array
    
    #############################################
    # Solver, returns a single vector with all unknown coefficients
    # This output is used for hydro coefficient computations
    def get_unknown_coeffs(self, a, b):
        return linalg.solve(a,b)
    
    # Input: Single vector of the coefficients
    # Output: List of lists, where each list contains the coefficients for that region.
    # This output is used for plotting.
    def reformat_coeffs(self, x):
        NMK, boundary_count = self.NMK, self.boundary_count
        cs = []
        row = 0
        cs.append(x[:NMK[0]])
        row += NMK[0]
        for i in range(1, boundary_count):
            cs.append(x[row: row + NMK[i] * 2])
            row += NMK[i] * 2
        cs.append(x[row:])
        return cs

    #############################################
    # Hydro coefficient computation
    def hydro_coeffs(self, x, convention):
        heaving, NMK, boundary_count = self.heaving, self.NMK, self.boundary_count
        # Build c-vector
        c = np.zeros((self.size - NMK[-1]), dtype=complex)
        col = 0
        for n in range(NMK[0]):
            c[n] = heaving[0] * self.int_R_1n(0, n)* self.z_n_d(n)
        col += NMK[0]
        for i in range(1, boundary_count):
            M = NMK[i]
            for m in range(M):
                c[col + m] = heaving[i] * self.int_R_1n(i, m)* self.z_n_d(m)
                c[col + M + m] = heaving[i] * self.int_R_2n(i, m)* self.z_n_d(m)
            col += 2 * M

        hydro_p_terms = np.zeros(boundary_count, dtype=complex)
        for i in range(boundary_count):
            hydro_p_terms[i] = heaving[i] * self.int_phi_p_i(i)

        hydro_coef = 2 * pi * (np.dot(c, x[:-NMK[-1]]) + sum(hydro_p_terms))

        if convention == "nondimensional":
            # find maximum heaving radius
            max_rad = self.a[0]
            for i in range(boundary_count - 1, 0, -1):
                if heaving[i]:
                    max_rad = self.a[i]
                    break
            hydro_coef_nondim = self.h**3/(max_rad**3 * pi)*hydro_coef
            added_mass = hydro_coef_nondim.real
            damping = hydro_coef_nondim.imag
        elif convention == "umerc":
            added_mass = hydro_coef.real * self.h**3 * self.rho
            if self.m0 == inf: damping = 0
            else: damping = hydro_coef.imag * self.angular_freq(self.m0) * self.h**3 * self.rho
        elif convention == "capytaine":
            added_mass = hydro_coef.real * self.rho
            if self.m0 == inf: damping = 0
            else: damping = hydro_coef.imag * self.angular_freq(self.m0) * self.rho
        else:
            raise ValueError("Allowed conventions are nondimensional, umerc, and capytaine.")
        return added_mass, damping

    # Some intermediate integrals
    # integrating R_1n * r in region i
    def int_R_1n(self, i, n):
        a, scale = self.a, self.scale
        if n == 0:
            inner = (0 if i == 0 else a[i-1]) # central region has inner radius 0
            return a[i]**2/4 - inner**2/4
        else:
            lambda0 = self.lambda_ni(n, i)
            bottom = lambda0 * besselie(0, lambda0 * scale[i])
            if i == 0: inner_term = 0
            else: inner_term = (a[i-1] * besselie(1, lambda0 * a[i-1]) / bottom) * exp(lambda0 * (a[i-1] - scale[i]))
            outer_term = (a[i] * besselie(1, lambda0 * a[i]) / bottom) * exp(lambda0 * (a[i] - scale[i]))
            return outer_term - inner_term

    # integrating R_2n * r in region i
    def int_R_2n(self, i, n):
        a, scale = self.a, self.scale
        if i == 0:
            raise ValueError("i cannot be 0")
        lambda0 = self.lambda_ni(n, i)
        if n == 0:
            return (a[i-1]**2 * (2*np.log(a[i]/a[i-1]) + 1) - a[i]**2)/8
        else:
            outer_term = a[i] * besselke(1, lambda0 * a[i])
            inner_term = a[i-1] * besselke(1, lambda0 * a[i-1])
            bottom = - lambda0 * besselke(0, lambda0 * scale[i])
            return (outer_term / bottom) * exp(lambda0 * (scale[i] - a[i])) - (inner_term/bottom)* exp(lambda0 * (scale[i] - a[i-1]))

    # integrating phi_p_i * d_phi_p_i/dz * r *d_r at z=d[i]
    # where phi_p_i is the particular solution in a heaving region i
    def int_phi_p_i(self, i):
        h, d, a = self.h, self.d, self.a
        denom = 16 * (h - d[i])
        if i == 0:
            num = a[i]**2*(4*(h-d[i])**2-a[i]**2)
        else:
            num = (a[i]**2*(4*(h-d[i])**2-a[i]**2) - a[i-1]**2*(4*(h-d[i])**2-a[i-1]**2))
        return num/denom

    # evaluate an interior region vertical eigenfunction at its top boundary
    def z_n_d(self, n):
        if n ==0:
            return 1
        else:
            return sqrt(2)*(-1)**n

    #############################################
    def excitation_phase(self, x):
        # from x, access the first coefficient of the e-region expansion
        return -(pi/2) + np.angle(x[-self.NMK[-1]]) - np.angle(besselh(0, self.m0 * self.scale[-1]))
    
    def excitation_force(self, damping):
        # Chau 2012 eq 98
        m0, h, omega = self.m0, self.h, self.angular_freq(self.m0)
        const = np.tanh(m0 * h) + m0 * h * (1 - (np.tanh(m0 * h))**2)
        return sqrt((2 * const * self.rho * (g ** 2) * damping)/(omega * m0)) ** (1/2)
    
    #############################################
    # Graphics functions
    
    def make_R_Z(self, sharp, spatial_res): # create coordinate array for graphing
        rmin = (2 * self.a[-1] / spatial_res) if sharp else 0.0
        r_vec = np.linspace(rmin, 2*self.a[-1], spatial_res)
        z_vec = np.linspace(0, -self.h, spatial_res)
        if sharp: # more precise near boundaries
            a_eps = 1.0e-4
            for i in range(len(self.a)):
                r_vec = np.append(r_vec, self.a[i]*(1-a_eps))
                r_vec = np.append(r_vec, self.a[i]*(1+a_eps))
            r_vec = np.unique(r_vec)
            for i in range(len(self.d)):
                z_vec = np.append(z_vec, -self.d[i])
            z_vec = np.unique(z_vec)
        return np.meshgrid(r_vec, z_vec)
    
    def plot_pv(self, field, R, Z, title):
        plt.figure(figsize=(8, 6))
        plt.contourf(R, Z, field, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Radial Distance (R)')
        plt.ylabel('Axial Distance (Z)')
        plt.show()

    def generate_potential_plot_array(self, cs):
        h, d, a, heaving, NMK = self.h, self.d, self.a, self.heaving, self.NMK
        R_1n, R_2n, Lambda_k= self.R_1n, self.R_2n, self.Lambda_k
        Z_n_i, Z_k_e = self.Z_n_i, self.Z_k_e

        def phi_h_n_inner_func(n, r, z):
            return (cs[0][n] * R_1n(n, r, 0)) * Z_n_i(n, z, 0)
        def phi_h_m_i_func(i, m, r, z):
            return (cs[i][m] * R_1n(m, r, i) + cs[i][NMK[i] + m] * R_2n(m, r, i)) * Z_n_i(m, z, i)
        def phi_e_k_func(k, r, z):
            return cs[-1][k] * Lambda_k(k, r) * Z_k_e(k, z)
        def phi_p_i(d, r, z): # particular solution
            return (1 / (2* (h - d))) * ((z + h) ** 2 - (r**2) / 2)

        phi_e_k_vec = np.vectorize(phi_e_k_func, otypes = [complex])
        phi_h_n_inner_vec = np.vectorize(phi_h_n_inner_func, otypes = [complex])
        phi_h_m_i_vec = np.vectorize(phi_h_m_i_func, otypes = [complex])
        phi_p_i_vec = np.vectorize(phi_p_i)

        R, Z = self.make_R_Z(True, 50)

        regions = []
        regions.append((R <= a[0]) & (Z < -d[0]))
        for i in range(1, self.boundary_count):
            regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
        regions.append(R > a[-1])

        phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

        for n in range(NMK[0]):
            temp_phiH = phi_h_n_inner_vec(n, R[regions[0]], Z[regions[0]])
            phiH[regions[0]] = temp_phiH if n == 0 else phiH[regions[0]] + temp_phiH

        for i in range(1, self.boundary_count):
            for m in range(NMK[i]):
                temp_phiH = phi_h_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
                phiH[regions[i]] = temp_phiH if m == 0 else phiH[regions[i]] + temp_phiH

        for k in range(NMK[-1]):
            temp_phiH = phi_e_k_vec(k, R[regions[-1]], Z[regions[-1]])
            phiH[regions[-1]] = temp_phiH if k == 0 else phiH[regions[-1]] + temp_phiH

        phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
        for i in range(1, self.boundary_count):
            phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
        phiP[regions[-1]] = 0

        phi = phiH + phiP
        return phi, phiH, phiP

    def generate_velocity_plot_array(self, cs):
        h, d, a, heaving, NMK = self.h, self.d, self.a, self.heaving, self.NMK
        R_1n, R_2n, Lambda_k= self.R_1n, self.R_2n, self.Lambda_k
        diff_R_1n, diff_R_2n, diff_Lambda_k= self.diff_R_1n, self.diff_R_2n, self.diff_Lambda_k
        Z_n_i, Z_k_e = self.Z_n_i, self.Z_k_e
        diff_Z_n_i, diff_Z_k_e = self.diff_Z_n_i, self.diff_Z_k_e

        def diff_r_phi_p_i(d, r, z): 
            return (- r / (2* (h - d)))
        def diff_z_phi_p_i(d, r, z): 
            return ((z+h) / (h - d))
        def v_r_inner_func(n, r, z):
            return (cs[0][n] * diff_R_1n(n, r, 0)) * Z_n_i(n, z, 0)
        def v_r_m_i_func(i, m, r, z):
            return (cs[i][m] * diff_R_1n(m, r, i) + cs[i][NMK[i] + m] * diff_R_2n(m, r, i)) * Z_n_i(m, z, i)
        def v_r_e_k_func(k, r, z):
            return cs[-1][k] * diff_Lambda_k(k, r) * Z_k_e(k, z)
        def v_z_inner_func(n, r, z):
            return (cs[0][n] * R_1n(n, r, 0)) * diff_Z_n_i(n, z, 0)
        def v_z_m_i_func(i, m, r, z):
            return (cs[i][m] * R_1n(m, r, i) + cs[i][NMK[i] + m] * R_2n(m, r, i)) * diff_Z_n_i(m, z, i)
        def v_z_e_k_func(k, r, z):
            return cs[-1][k] * Lambda_k(k, r) * diff_Z_k_e(k, z)

        vr_p_i_vec = np.vectorize(diff_r_phi_p_i)
        vz_p_i_vec = np.vectorize(diff_z_phi_p_i)
        v_r_inner_vec = np.vectorize(v_r_inner_func, otypes = [complex])
        v_r_m_i_vec = np.vectorize(v_r_m_i_func, otypes = [complex])
        v_r_e_k_vec = np.vectorize(v_r_e_k_func, otypes = [complex])
        v_z_inner_vec = np.vectorize(v_z_inner_func, otypes = [complex])
        v_z_m_i_vec = np.vectorize(v_z_m_i_func, otypes = [complex])
        v_z_e_k_vec = np.vectorize(v_z_e_k_func, otypes = [complex])

        R, Z = self.make_R_Z(True, 50)

        regions = []
        regions.append((R <= a[0]) & (Z < -d[0]))
        for i in range(1, self.boundary_count):
            regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
        regions.append(R > a[-1])

        vrH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        vrP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        vzH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        vzP = np.full_like(R, np.nan + np.nan*1j, dtype=complex)

        for n in range(NMK[0]):
            temp_vrH = v_r_inner_vec(n, R[regions[0]], Z[regions[0]])
            temp_vzH = v_z_inner_vec(n, R[regions[0]], Z[regions[0]])
            if n == 0:
                vrH[regions[0]] = temp_vrH
                vzH[regions[0]] = temp_vzH
            else:
                vrH[regions[0]] = vrH[regions[0]] + temp_vrH
                vzH[regions[0]] = vzH[regions[0]] + temp_vzH

        for i in range(1, self.boundary_count):
            for m in range(NMK[i]):
                temp_vrH = v_r_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
                temp_vzH = v_z_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
                if m == 0:
                    vrH[regions[i]] = temp_vrH
                    vzH[regions[i]] = temp_vzH
                else:
                    vrH[regions[i]] = vrH[regions[i]] + temp_vrH
                    vzH[regions[i]] = vzH[regions[i]] + temp_vzH

        for k in range(NMK[-1]):
            temp_vrH = v_r_e_k_vec(k, R[regions[-1]], Z[regions[-1]])
            temp_vzH = v_z_e_k_vec(k, R[regions[-1]], Z[regions[-1]])
            if k == 0:
                vrH[regions[-1]] = temp_vrH
                vzH[regions[-1]] = temp_vzH
            else:
                vrH[regions[-1]] = vrH[regions[-1]] + temp_vrH
                vzH[regions[-1]] = vzH[regions[-1]] + temp_vzH

        vrP[regions[0]] = heaving[0] * vr_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
        vzP[regions[0]] = heaving[0] * vz_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
        for i in range(1, self.boundary_count):
            vrP[regions[i]] = heaving[i] * vr_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
            vzP[regions[i]] = heaving[i] * vz_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
        vrP[regions[-1]] = 0
        vzP[regions[-1]] = 0

        vr = vrH + vrP
        vz = vzH + vzP

        return vr, vz
    
    def plot_potentials(self, cs):
        R, Z = self.make_R_Z(True, 50)
        phi, phiH, phiP = self.generate_potential_plot_array(cs)
        self.plot_pv(np.real(phiH), R, Z, 'Homogeneous Potential')
        self.plot_pv(np.imag(phiH), R, Z, 'Homogeneous Potential Imaginary')

        self.plot_pv(np.real(phiP), R, Z, 'Particular Potential')
        self.plot_pv(np.imag(phiP), R, Z, 'Particular Potential Imaginary')

        self.plot_pv(np.real(phi), R, Z, 'Potential (Real Part)')
        self.plot_pv(np.imag(phi), R, Z, 'Total Potential Imaginary')

    def plot_velocities(self, cs):
        R, Z = self.make_R_Z(True, 50)
        vr, vz = self.generate_velocity_plot_array(cs)
        
        self.plot_pv(np.real(vr), R, Z, 'Radial Velocity - Real')
        self.plot_pv(np.imag(vr), R, Z, 'Radial Velocity - Imaginary')
        self.plot_pv(np.real(vz), R, Z, 'Vertical Velocity - Real')
        self.plot_pv(np.imag(vz), R, Z, 'Vertical Velocity - Imaginary')

    #############################################
    # Format a 50 x 50 array of potentials for testing
    def config_potential_array(self, cs):
        boundary_count, NMK, heaving = self.boundary_count, self.NMK, self.heaving
        a, d, h, = self.a, self.d, self.h

        R_1n, R_2n, Lambda_k= self.R_1n, self.R_2n, self.Lambda_k
        Z_n_i, Z_k_e = self.Z_n_i, self.Z_k_e

        def phi_h_n_inner_func(n, r, z):
            return (cs[0][n] * R_1n(n, r, 0)) * Z_n_i(n, z, 0)
        def phi_h_m_i_func(i, m, r, z):
            return (cs[i][m] * R_1n(m, r, i) + cs[i][NMK[i] + m] * R_2n(m, r, i)) * Z_n_i(m, z, i)
        def phi_e_k_func(k, r, z):
            return cs[-1][k] * Lambda_k(k, r) * Z_k_e(k, z)
        def phi_p_i(d, r, z): # particular solution
            return (1 / (2* (h- d))) * ((z + h) ** 2 - (r**2) / 2)

        phi_e_k_vec = np.vectorize(phi_e_k_func, otypes = [complex])
        phi_h_n_inner_vec = np.vectorize(phi_h_n_inner_func, otypes = [complex])
        phi_h_m_i_vec = np.vectorize(phi_h_m_i_func, otypes = [complex])
        phi_p_i_vec = np.vectorize(phi_p_i)

        R, Z = self.make_R_Z(False, 50)

        regions = []
        regions.append((R <= a[0]) & (Z < -d[0]))
        for i in range(1, boundary_count):
            regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
        regions.append(R > a[-1])

        phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
        phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

        for n in range(NMK[0]):
            temp_phiH = phi_h_n_inner_vec(n, R[regions[0]], Z[regions[0]])
            phiH[regions[0]] = temp_phiH if n == 0 else phiH[regions[0]] + temp_phiH

        for i in range(1, boundary_count):
            for m in range(NMK[i]):
                temp_phiH = phi_h_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
                phiH[regions[i]] = temp_phiH if m == 0 else phiH[regions[i]] + temp_phiH

        for k in range(NMK[-1]):
            temp_phiH = phi_e_k_vec(k, R[regions[-1]], Z[regions[-1]])
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
    # Capabilities beyond the multi-constants/equations/MEEM format
    
    def change_m0(self, new_m0):
        self.m0 = new_m0
        self.m_k = self.m_k_array()

    # Given an A matrix for the same configuration/NMK but possibly different m0, return this problem's A matrix.
    # This reduces computation.
    def a_matrix_from_old(self, a_matrix):
        d = self.d
        I_mk_vals = self.I_mk_vals()
        # insert potential block
        row = 0
        for i in range(self.boundary_count - 1):
            row += self.NMK[i + (d[i] <= d[i + 1])]
        col = self.size - self.NMK[-1]
        submat = self.p_dense_block_e(-1, I_mk_vals)
        intermediate_matrix = insert_submatrix(a_matrix, submat, row, col)

        # insert velocity blocks
        row = self.size - self.NMK[-1]
        left_block1 = self.v_dense_block_e(np.vectorize(self.diff_R_1n, otypes=[complex]), -1, I_mk_vals)
        right_block = self.v_diagonal_block_e(-1)
        if self.boundary_count == 1:
            col = 0
            submat = np.concatenate((left_block1,right_block), axis = 1)
        else:
            col = self.size - (self.NMK[-1] + 2 * self.NMK[-2])
            left_block2 = self.v_dense_block_e(np.vectorize(self.diff_R_2n, otypes=[complex]), -1, I_mk_vals)
            submat = np.concatenate((left_block1,left_block2,right_block), axis = 1)
        final_matrix = insert_submatrix(intermediate_matrix, submat, row, col)
        return final_matrix

    # Given a b vector for the same configuration/NMK but possibly different m0, return this problem's b vector.
    def b_vector_from_old(self, b_vector):
        b_vector = b_vector.copy()
        index = len(b_vector) - self.NMK[-1]
        for k in range(self.NMK[-1]):
            b_vector[index] = self.b_velocity_end_entry(k, -1)
            index += 1
        return b_vector
        
        
        
