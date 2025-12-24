# test_multi_equations.py

import sys
import os
import numpy as np
import pytest
from numpy import pi, sqrt, cosh, cos, sinh, sin, exp, log
from scipy.special import iv as besseli
from scipy.special import kv as besselk
from scipy.special import kve as besselke
from scipy.special import ive as besselie
from scipy.special import hankel1 as besselh
from unittest.mock import Mock, patch
from scipy.optimize import root_scalar 

# Adjust the path to import from package's 'src' directory.
current_dir = os.path.dirname(__file__)
package_base_dir = os.path.join(current_dir, '..')
src_dir = os.path.join(package_base_dir, 'src')
sys.path.insert(0, os.path.abspath(src_dir))

# Import all functions from multi_equations.py
from openflash.multi_equations import (
    omega, scale, lambda_ni, m_k_entry, m_k, m_k_newton,
    I_nm, I_mk, b_potential_entry, b_potential_end_entry,
    b_velocity_entry, b_velocity_end_entry, 
    phi_p_i, diff_r_phi_p_i, diff_z_phi_p_i, R_1n, diff_R_1n, R_2n, diff_R_2n,
    Z_n_i, diff_Z_n_i, Lambda_k, diff_Lambda_k,
    N_k_multi, Z_k_e, int_R_1n, int_R_2n,
    z_n_d, excitation_phase,
    # New imports for added coverage
    Lambda_k_vectorized, diff_Lambda_k_vectorized, make_R_Z
)

# --- Fixtures for common parameters ---
@pytest.fixture
def coeff():
    return 1.0 + 1.0j 

@pytest.fixture
def h():
    return 100.0

@pytest.fixture
def g():
    return 9.81

@pytest.fixture
def m0():
    return 0.1 

@pytest.fixture
def d(): 
    return np.array([0.0, 20.0, 50.0])

@pytest.fixture
def a(): 
    return np.array([5.0, 10.0, 15.0]) 

@pytest.fixture
def heaving(): 
    return np.array([1.0, 0.5, 0.0]) 

@pytest.fixture
def NMK(): 
    return np.array([0, 5]) 

@pytest.fixture
def test_n():
    return 1 

@pytest.fixture
def test_k():
    return 1 

@pytest.fixture
def test_i(): 
    return 1 

@pytest.fixture
def test_r(a):
    return (a[0] + a[1]) / 2 

@pytest.fixture
def test_z(d, h):
    return (d[1] - h) / 2 

@pytest.fixture
def precomputed_m_k_arr(NMK, h):
    num_k = NMK[-1] 
    arr = np.linspace(0.01, 0.5, num_k) 
    arr[0] = 0.1 
    return arr

@pytest.fixture
def precomputed_N_k_arr(NMK):
    num_k = NMK[-1]
    return np.linspace(0.5, 2.0, num_k) 


# --- Test Cases ---

def test_omega(m0, h, g):
    expected = sqrt(m0 * np.tanh(m0 * h) * g)
    assert np.isclose(omega(m0, h, g), expected)

def test_lambda_ni(test_n, test_i, h, d):
    expected = test_n * pi / (h - d[test_i])
    assert np.isclose(lambda_ni(test_n, test_i, h, d), expected)

# --- m_k functions ---
def test_m_k_entry_k0(m0, h):
    assert np.isclose(m_k_entry(0, m0, h), m0)

def test_m_k_entry_k_positive(h):
    test_h = 10.0
    target_m_k_h = 2.0
    required_m0h_tanh_m0h = -(target_m_k_h * np.tan(target_m_k_h))

    def f_Y(Y):
        return Y * np.tanh(Y) - required_m0h_tanh_m0h

    y_result = root_scalar(f_Y, x0=4.5, method='newton') 
    calculated_m0h = y_result.root
    test_m0 = calculated_m0h / test_h

    k_val = 1
    expected_m_k_val = target_m_k_h / test_h
    
    result = m_k_entry(k_val, test_m0, test_h)
    assert np.isclose(result, expected_m_k_val, rtol=1e-5, atol=1e-8)
    
    shouldnt_be_int = np.round(test_m0 * result / np.pi - 0.5, 4)
    assert shouldnt_be_int != np.floor(shouldnt_be_int)

def test_m_k(NMK, m0, h):
    num_modes = NMK[-1] 
    result_array = m_k(NMK, m0, h)

    assert isinstance(result_array, np.ndarray)
    assert result_array.shape[0] == num_modes
    assert np.isclose(result_array[0], m0)

def test_m_k_newton(h, m0):
    expected_k = np.sqrt(m0**2 / 9.8 / h) 
    assert np.isclose(m_k_newton(h, m0), expected_k, rtol=2e-2, atol=1e-5) 

# --- Coupling Integrals ---
def test_I_nm_n0m0(h, d):
    i = 1 
    dj = max(d[i], d[i+1]) 
    expected = h - dj 
    assert np.isclose(I_nm(0, 0, i, d, h), expected)

def test_I_nm_n0m_positive(h, d):
    n = 0
    m = 1
    i = 0 
    dj = max(d[i], d[i+1]) 
    assert np.isclose(I_nm(n, m, i, d, h), 0)

    n = 0
    m = 1
    i = 1 
    custom_d = np.array([0.0, 50.0, 20.0]) 
    i = 1 
    dj = max(custom_d[i], custom_d[i+1]) 
    lambda2 = lambda_ni(m, i + 1, h, custom_d) 
    expected = sqrt(2) * sin(lambda2 * (h - dj)) / lambda2
    assert np.isclose(I_nm(n, m, i, custom_d, h), expected)

def test_I_nm_n_positive_m0(h, d):
    n = 1
    m = 0
    i = 0 
    dj = max(d[i], d[i+1]) 
    lambda1 = lambda_ni(n, i, h, d) 
    expected = sqrt(2) * sin(lambda1 * (h - dj)) / lambda1
    assert np.isclose(I_nm(n, m, i, d, h), expected)

def test_I_nm_n_positive_m_positive(h, d):
    n = 1
    m = 1
    i = 0
    dj = max(d[i], d[i+1]) 

    lambda1 = lambda_ni(n, i, h, d)
    lambda2 = lambda_ni(m, i + 1, h, d) 

    frac1 = sin((lambda1 + lambda2)*(h-dj))/(lambda1 + lambda2)
    frac2 = sin((lambda1 - lambda2)*(h-dj))/(lambda1 - lambda2) 
    expected = frac1 + frac2
    assert np.isclose(I_nm(n, m, i, d, h), expected)

    n_eq = 1
    m_eq = 1
    i_eq = 0
    d_eq = np.array([50.0, 50.0, 50.0]) 

    lambda1_internal = lambda_ni(n_eq, i_eq, h, d_eq)
    lambda2_internal = lambda_ni(m_eq, i_eq + 1, h, d_eq) 

    dj_eq = max(d_eq[i_eq], d_eq[i_eq+1]) 

    frac1_eq = sin((lambda1_internal + lambda2_internal)*(h-dj_eq))/(lambda1_internal + lambda2_internal)

    if np.isclose(lambda1_internal, lambda2_internal, atol=1e-12): 
        frac2_eq = (h - dj_eq)
    else:
        frac2_eq = sin((lambda1_internal - lambda2_internal)*(h-dj_eq))/(lambda1_internal - lambda2_internal)

    expected_eq = frac1_eq + frac2_eq

    assert np.isclose(I_nm(n_eq, m_eq, i_eq, d_eq, h), expected_eq)

# --- b-vector computation ---
def test_b_potential_entry_n0(test_i, d, heaving, h, a):
    j = test_i + (d[test_i] < d[test_i+1]) 
    constant = (heaving[test_i+1] / (h - d[test_i+1]) - heaving[test_i] / (h - d[test_i]))
    expected = constant * 0.5 * ((h - d[j])**3/3 - (h-d[j]) * a[test_i]**2/2)
    assert np.isclose(b_potential_entry(0, test_i, d, heaving, h, a), expected)

def test_b_potential_entry_n_positive(test_n, test_i, d, heaving, h, a):
    if test_n == 0: pytest.skip("Test for n>0")
    j = test_i + (d[test_i] < d[test_i+1])
    constant = (heaving[test_i+1] / (h - d[test_i+1]) - heaving[test_i] / (h - d[test_i]))
    expected = sqrt(2) * (h - d[j]) * constant * ((-1) ** test_n)/(lambda_ni(test_n, j, h, d) ** 2)
    assert np.isclose(b_potential_entry(test_n, test_i, d, heaving, h, a), expected)

def test_b_potential_end_entry_n0(test_i, heaving, h, d, a):
    constant = - heaving[test_i] / (h - d[test_i])
    expected = constant * 0.5 * ((h - d[test_i])**3/3 - (h-d[test_i]) * a[test_i]**2/2)
    assert np.isclose(b_potential_end_entry(0, test_i, heaving, h, d, a), expected)

def test_b_potential_end_entry_n_positive(test_n, test_i, heaving, h, d, a):
    if test_n == 0: pytest.skip("Test for n>0")
    constant = - heaving[test_i] / (h - d[test_i])
    expected = sqrt(2) * (h - d[test_i]) * constant * ((-1) ** test_n)/(lambda_ni(test_n, test_i, h, d) ** 2)
    assert np.isclose(b_potential_end_entry(test_n, test_i, heaving, h, d, a), expected)

def test_b_velocity_entry_n0(test_i, heaving, a, h, d):
    expected = (heaving[test_i+1] - heaving[test_i]) * (a[test_i]/2)
    assert np.isclose(b_velocity_entry(0, test_i, heaving, a, h, d), expected)

def test_b_velocity_entry_n_positive_di_greater(test_n, heaving, a, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    custom_d = np.array([0.0, 50.0, 20.0]) 
    i = 1
    num = - sqrt(2) * a[i] * sin(lambda_ni(test_n, i+1, h, custom_d) * (h-custom_d[i]))
    denom = (2 * (h - custom_d[i]) * lambda_ni(test_n, i+1, h, custom_d))
    expected = num/denom
    assert np.isclose(b_velocity_entry(test_n, i, heaving, a, h, custom_d), expected)

def test_b_velocity_entry_n_positive_di_smaller(test_n, heaving, a, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    i = 0 
    num = sqrt(2) * a[i] * sin(lambda_ni(test_n, i, h, d) * (h-d[i+1]))
    denom = (2 * (h - d[i+1]) * lambda_ni(test_n, i, h, d))
    expected = num/denom
    assert np.isclose(b_velocity_entry(test_n, i, heaving, a, h, d), expected)

def test_b_velocity_end_entry_k0_small_m0h(test_i, heaving, a, h, d, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    m0_local = 0.01 
    constant = - heaving[test_i] * a[test_i]/(2 * (h - d[test_i]))
    N_k_arr_local = precomputed_N_k_arr.copy()
    N_k_arr_local[0] = 0.5
    expected = constant * (1/sqrt(N_k_arr_local[0])) * sinh(m0_local * (h - d[test_i])) / m0_local
    assert np.isclose(b_velocity_end_entry(0, test_i, heaving, a, h, d, m0_local, NMK, precomputed_m_k_arr, N_k_arr_local), expected)

def test_b_velocity_end_entry_k0_large_m0h(test_i, heaving, a, h, d, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    m0_local = 0.2 
    constant = - heaving[test_i] * a[test_i]/(2 * (h - d[test_i]))
    N_k_arr_local = precomputed_N_k_arr.copy()
    N_k_arr_local[0] = 0.5
    expected = constant * sqrt(2 * h / m0_local) * (exp(- m0_local * d[test_i]) - exp(m0_local * d[test_i] - 2 * m0_local * h))
    assert np.isclose(b_velocity_end_entry(0, test_i, heaving, a, h, d, m0_local, NMK, precomputed_m_k_arr, N_k_arr_local), expected)

def test_b_velocity_end_entry_k_positive(test_k, test_i, heaving, a, h, d, m0, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    if test_k == 0: pytest.skip("Test for k>0")
    constant = - heaving[test_i] * a[test_i]/(2 * (h - d[test_i]))
    local_m_k_k = precomputed_m_k_arr[test_k]
    N_k_arr_k = precomputed_N_k_arr[test_k]
    expected = constant * (1/sqrt(N_k_arr_k)) * sin(local_m_k_k * (h - d[test_i])) / local_m_k_k
    assert np.isclose(b_velocity_end_entry(test_k, test_i, heaving, a, h, d, m0, NMK, precomputed_m_k_arr, precomputed_N_k_arr), expected)

# --- Phi particular and partial derivatives ---
def test_phi_p_i(d, test_r, test_z, h):
    d_val = d[1] 
    expected = (1 / (2 * (h - d_val))) * ((test_z + h)**2 - (test_r**2) / 2)
    assert np.isclose(phi_p_i(d_val, test_r, test_z, h), expected)

def test_diff_r_phi_p_i(d, test_r, h):
    d_val = d[1]
    expected = (- test_r / (2 * (h - d_val)))
    assert np.isclose(diff_r_phi_p_i(d_val, test_r, h), expected)

def test_diff_z_phi_p_i(d, test_z, h):
    d_val = d[1]
    expected = ((test_z + h) / (h - d_val))
    assert np.isclose(diff_z_phi_p_i(d_val, test_z, h), expected)

# --- Bessel I Radial Eigenfunction ---
def test_R_1n_n0(test_i, test_r, h, d, a):
    # Fixed expected value to match stable implementation: 1.0 + 0.5 * log(r/outer)
    if test_i == 0:
        expected = 0.5
    else:
        expected = 1.0 + 0.5 * np.log(test_r / a[test_i])
    assert np.isclose(R_1n(0, test_r, test_i, h, d, a), expected)

def test_R_1n_n_positive(test_n, test_i, test_r, h, d, a):
    if test_n == 0: pytest.skip("Test for n>0")
    local_scale = scale(a)
    # Fixed to use scaled Bessel I (besselie) and exp term
    lambda0 = lambda_ni(test_n, test_i, h, d)
    expected = besselie(0, lambda0 * test_r) / besselie(0, lambda0 * local_scale[test_i]) * exp(lambda0 * (test_r - local_scale[test_i]))
    assert np.isclose(R_1n(test_n, test_r, test_i, h, d, a), expected)

def test_R_1n_n_negative(test_i, test_r, h, d, a):
    with pytest.raises(ValueError, match="Invalid value for n"):
        R_1n(-1, test_r, test_i, h, d, a)

def test_diff_R_1n_n0(test_i, test_r, h, d, a):
    # Fixed expected derivative for log term: 1/(2r)
    if test_i == 0:
        expected = 0.0
    else:
        expected = 1 / (2 * test_r)
    assert np.isclose(diff_R_1n(0, test_r, test_i, h, d, a), expected)

def test_diff_R_1n_n_positive(test_n, test_i, test_r, h, d, a):
    if test_n == 0: pytest.skip("Test for n>0")
    local_scale = scale(a)
    # Fixed to match scaled derivative
    lambda0 = lambda_ni(test_n, test_i, h, d)
    top = lambda0 * besselie(1, lambda0 * test_r)
    bottom = besselie(0, lambda0 * local_scale[test_i])
    expected = top / bottom * exp(lambda0 * (test_r - local_scale[test_i]))
    assert np.isclose(diff_R_1n(test_n, test_r, test_i, h, d, a), expected)

# --- Bessel K Radial Eigenfunction ---
def test_R_2n_i0_raises_error(test_r, h, d, a):
    with pytest.raises(ValueError, match="i cannot be 0"):
        R_2n(1, test_r, 0, a, h, d)

def test_R_2n_n0(a, test_r, h, d):
    # Fixed expected value: 0.5 * log(r/outer) anchored at a[i]
    i = 1
    outer_r = a[i]
    expected = 0.5 * np.log(test_r / outer_r) 
    assert np.isclose(R_2n(0, test_r, i, a, h, d), expected)

def test_R_2n_n_positive(test_n, a, test_r, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    i = 1
    # Fixed expected value: Scaled K0 using besselke and exp decay from OUTER radius (Legacy Mode)
    lambda0 = lambda_ni(test_n, i, h, d)
    outer_r = a[i]
    expected = (besselke(0, lambda0 * test_r) / besselke(0, lambda0 * outer_r)) * exp(lambda0 * (outer_r - test_r))
    assert np.isclose(R_2n(test_n, test_r, i, a, h, d), expected)

def test_diff_R_2n_n0(test_r, h, d, a):
    i = 1
    expected = 1 / (2 * test_r)
    assert np.isclose(diff_R_2n(0, test_r, i, h, d, a), expected)

def test_diff_R_2n_n_positive(test_n, test_r, h, d, a):
    if test_n == 0: pytest.skip("Test for n>0")
    i = 1
    # Fixed expected value: Scaled derivative anchored at OUTER radius (Legacy Mode)
    lambda0 = lambda_ni(test_n, i, h, d)
    outer_r = a[i]
    top = -lambda0 * besselke(1, lambda0 * test_r)
    bottom = besselke(0, lambda0 * outer_r)
    expected = (top / bottom) * exp(lambda0 * (outer_r - test_r))
    assert np.isclose(diff_R_2n(test_n, test_r, i, h, d, a), expected)

# --- i-region vertical eigenfunctions ---
def test_Z_n_i_n0(test_z, test_i, h, d):
    assert np.isclose(Z_n_i(0, test_z, test_i, h, d), 1)

def test_Z_n_i_n_positive(test_n, test_z, test_i, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    expected = sqrt(2) * np.cos(lambda_ni(test_n, test_i, h, d) * (test_z + h))
    assert np.isclose(Z_n_i(test_n, test_z, test_i, h, d), expected)

def test_diff_Z_n_i_n0(test_z, test_i, h, d):
    assert np.isclose(diff_Z_n_i(0, test_z, test_i, h, d), 0)

def test_diff_Z_n_i_n_positive(test_n, test_z, test_i, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    lambda0 = lambda_ni(test_n, test_i, h, d)
    expected = - lambda0 * sqrt(2) * np.sin(lambda0 * (test_z + h))
    assert np.isclose(diff_Z_n_i(test_n, test_z, test_i, h, d), expected)

# --- e-region vertical eigenfunctions ---
def test_Z_k_e_k0_large_m0h(test_z, m0, h, NMK, precomputed_m_k_arr):
    m0_local = 0.2
    expected = sqrt(2 * m0_local * h) * (exp(m0_local * test_z) + exp(-m0_local * (test_z + 2*h)))
    assert np.isclose(Z_k_e(0, test_z, m0_local, h, NMK, precomputed_m_k_arr), expected)

# --- To calculate hydrocoefficients ---
def test_int_R_1n_n0_i0(a, h, d):
    # i=0 (innermost region), inner radius 0
    expected = a[0]**2/4 - 0**2/4
    assert np.isclose(int_R_1n(0, 0, a, h, d), expected)

def test_int_R_1n_n0_i_positive(a, h, d):
    # Fixed expected value: Integral of r * (1.0 + 0.5 * log(r/outer))
    i = 1
    outer_r = a[i]
    inner_r = a[i-1]
    
    # 1. Cylinder term (integral of r*1.0)
    cyl_term = (outer_r**2 - inner_r**2) / 2.0
    
    # 2. Log term helper
    def log_indefinite_int(r):
        log_val = np.log(r/outer_r) if r > 0 else 0
        return 0.5 * ((r**2 / 2.0) * log_val - (r**2 / 4.0))

    val_outer = log_indefinite_int(outer_r) 
    val_inner = log_indefinite_int(inner_r)
    
    expected = cyl_term + (val_outer - val_inner)
    assert np.isclose(int_R_1n(1, 0, a, h, d), expected)

def test_int_R_1n_n_positive(test_n, test_i, a, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    local_scale = scale(a)
    lambda0 = lambda_ni(test_n, test_i, h, d)
    # Fixed to use scaled Bessel I
    bottom = lambda0 * besselie(0, lambda0 * local_scale[test_i])
    if test_i == 0: 
        inner_term = 0
    else: 
        # Scaled inner term
        inner_term = (a[test_i-1] * besselie(1, lambda0 * a[test_i-1]) / bottom) * exp(lambda0 * (a[test_i-1] - local_scale[test_i]))
    
    # Scaled outer term (exp(0) = 1)
    outer_term = (a[test_i] * besselie(1, lambda0 * a[test_i]) / bottom) * 1.0
    
    expected = outer_term - inner_term
    assert np.isclose(int_R_1n(test_n, test_i, a, h, d), expected) 

def test_int_R_2n_i0_raises_error(test_n, a, h, d):
    with pytest.raises(ValueError, match="i cannot be 0"):
        int_R_2n(0, test_n, a, h, d)

def test_int_R_2n_n0(a, h, d):
    # Fixed expected value: Integral of r * (0.5 * log(r/outer)) anchored at outer
    i = 1 
    outer_r = a[i]
    inner_r = a[i-1]

    # Note: No 'cyl_term' here because R_2n(n=0) is just the log term.
    
    def log_indefinite_int(r):
        if r <= 0: return 0
        log_term = np.log(r/outer_r)
        return 0.5 * ((r**2 / 2.0) * log_term - (r**2 / 4.0))

    val_outer = log_indefinite_int(outer_r)
    val_inner = log_indefinite_int(inner_r) 
    
    expected = val_outer - val_inner
    assert np.isclose(int_R_2n(i, 0, a, h, d), expected)

def test_int_R_2n_n_positive(test_n, a, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    i = 1 
    # Fixed expected value: Scaled Bessel K integral, anchored at Outer (Legacy Mode)
    lambda0 = lambda_ni(test_n, i, h, d)
    outer_r = a[i]
    inner_r = a[i-1]
    
    # Normalized by lambda * K0(lambda * outer_r)
    # Denominator matches 'denom = lambda0 * besselke(0, lambda0 * outer_r)' in multi_equations.py
    denom = lambda0 * besselke(0, lambda0 * outer_r)
    
    term_outer = outer_r * besselke(1, lambda0 * outer_r)
    # No exp shift needed for outer term because exp(l(a-a)) = 1
    
    term_inner = inner_r * besselke(1, lambda0 * inner_r)
    # Apply exponential shift exp(l(a-r)) -> exp(l(a-inner))
    term_inner *= np.exp(lambda0 * (outer_r - inner_r))
    
    # Result is (inner - outer) / denom, which handles the sign flip from integration
    expected = (term_inner - term_outer) / denom
    assert np.isclose(int_R_2n(i, test_n, a, h, d), expected) 

def test_z_n_d_n0():
    assert np.isclose(z_n_d(0), 1)

def test_z_n_d_n_positive(test_n):
    if test_n == 0: pytest.skip("Test for n>0")
    expected = sqrt(2) * (-1)**test_n
    assert np.isclose(z_n_d(test_n), expected)

# ==============================================================================
# NEW TEST CASES FOR COVERAGE
# ==============================================================================

def test_Lambda_k_k0(a, m0, precomputed_m_k_arr):
    """Test Lambda_k for k=0 (Hankel case)."""
    r = a[-1] + 2.0
    k = 0
    # Expected: besselh(0, m0 * r) / besselh(0, m0 * a[-1])
    expected = besselh(0, m0 * r) / besselh(0, m0 * scale(a)[-1])
    assert np.isclose(Lambda_k(k, r, m0, a, precomputed_m_k_arr), expected)

def test_Lambda_k_k_positive(a, m0, precomputed_m_k_arr):
    """Test Lambda_k for k>0 (Bessel K case)."""
    r = a[-1] + 2.0
    k = 1
    local_mk = precomputed_m_k_arr[k]
    # Expected: K0(mk*r)/K0(mk*a) * exp(mk*(a-r))
    term = besselke(0, local_mk * r) / besselke(0, local_mk * scale(a)[-1])
    expected = term * np.exp(local_mk * (scale(a)[-1] - r))
    assert np.isclose(Lambda_k(k, r, m0, a, precomputed_m_k_arr), expected)

def test_Lambda_k_vectorized_m0_inf(a, precomputed_m_k_arr):
    """Test Lambda_k_vectorized returns ones when m0 is infinite."""
    k_vals = np.array([0, 1])
    r_vals = np.array([10.0, 10.0])
    res = Lambda_k_vectorized(k_vals, r_vals, np.inf, a, precomputed_m_k_arr)
    assert np.allclose(res, 1.0)

def test_diff_Lambda_k_vectorized_m0_inf(a, precomputed_m_k_arr):
    """Test diff_Lambda_k_vectorized returns ones when m0 is infinite."""
    k_vals = np.array([0, 1])
    r_vals = np.array([10.0, 10.0])
    res = diff_Lambda_k_vectorized(k_vals, r_vals, np.inf, a, precomputed_m_k_arr)
    assert np.allclose(res, 1.0)

def test_Z_k_e_k0_small_m0h(test_z, h, NMK, precomputed_m_k_arr):
    """Test Z_k_e for k=0 and small m0 (low frequency approximation)."""
    m0_small = 0.01 # m0*h = 1.0 < 14
    k = 0
    # Expected: 1/sqrt(N_k) * cosh(m0*(z+h))
    Nk = N_k_multi(k, m0_small, h, precomputed_m_k_arr)
    expected = (1 / np.sqrt(Nk)) * np.cosh(m0_small * (test_z + h))
    assert np.isclose(Z_k_e(k, test_z, m0_small, h, NMK, precomputed_m_k_arr), expected)

def test_Z_k_e_k_positive(test_z, m0, h, NMK, precomputed_m_k_arr):
    """Test Z_k_e for k>0."""
    k = 1
    # Z_k_e recalculates m_k internally for the cosine argument
    local_m_k_fresh = m_k(NMK, m0, h)
    
    Nk = N_k_multi(k, m0, h, precomputed_m_k_arr)
    expected = (1 / np.sqrt(Nk)) * np.cos(local_m_k_fresh[k] * (test_z + h))
    
    assert np.isclose(Z_k_e(k, test_z, m0, h, NMK, precomputed_m_k_arr), expected)

def test_make_R_Z_sharp(a, h, d):
    """Test make_R_Z with sharp=True to ensure refinement points are added."""
    spatial_res = 10
    R, Z = make_R_Z(a, h, d, sharp=True, spatial_res=spatial_res)
    
    # Check for epsilon points around a[i]
    r_unique = np.unique(R)
    a_eps = 1.0e-4
    for r_val in a:
        # Check for presence of r * (1 - eps) and r * (1 + eps)
        assert np.any(np.isclose(r_unique, r_val * (1 - a_eps)))
        assert np.any(np.isclose(r_unique, r_val * (1 + a_eps)))
        
    # Check for -d[i] points in Z
    z_unique = np.unique(Z)
    for d_val in d:
        assert np.any(np.isclose(z_unique, -d_val))