# test_multi_equations.py

import sys
import os
import numpy as np
import pytest
from numpy import pi, sqrt, cosh, cos, sinh, sin, exp
from scipy.special import iv as besseli
from scipy.special import kv as besselk
from scipy.special import hankel1 as besselh
from unittest.mock import Mock, patch
from scipy.optimize import root_scalar # For m_k_entry testing

# Adjust the path to import from package's 'src' directory.
current_dir = os.path.dirname(__file__)
package_base_dir = os.path.join(current_dir, '..')
src_dir = os.path.join(package_base_dir, 'src')
sys.path.insert(0, os.path.abspath(src_dir))

# Import all functions from multi_equations.py
from openflash.multi_equations import (
    omega, scale, lambda_ni, m_k_entry, m_k, m_k_newton,
    I_nm, I_mk, I_mk_full, b_potential_entry, b_potential_end_entry,
    b_velocity_entry, b_velocity_end_entry, b_velocity_end_entry_full,
    phi_p_i, diff_r_phi_p_i, diff_z_phi_p_i, R_1n, diff_R_1n, R_2n, diff_R_2n,
    Z_n_i, diff_Z_n_i, Lambda_k, Lambda_k_full, diff_Lambda_k, diff_Lambda_k_full,
    N_k_multi, N_k_full, Z_k_e, diff_Z_k_e, int_R_1n, int_R_2n,
    int_phi_p_i_no_coef, z_n_d, excitation_phase
)

# --- Fixtures for common parameters ---
@pytest.fixture
def coeff():
    return 1.0 + 1.0j # Arbitrary complex coefficient 

@pytest.fixture
def h():
    return 100.0

@pytest.fixture
def g():
    return 9.81

@pytest.fixture
def m0():
    return 0.1 # Example m0 value

@pytest.fixture
def d(): # Array of depths for multiple regions
    return np.array([0.0, 20.0, 50.0]) # d[0] is not used, d[1] for region 0-1, d[2] for region 1-2 etc.

@pytest.fixture
def a(): # Array of radii for multiple regions
    return np.array([5.0, 10.0, 15.0]) # a[0] for innermost, a[1] for middle, a[2] for outermost internal cylinder

@pytest.fixture
def heaving(): # Example heaving amplitudes
    return np.array([1.0, 0.5, 0.0]) # heaving[0] for inner, heaving[1] for middle, heaving[2] for outer.

@pytest.fixture
def NMK(): # Number of modes for k (e.g., [0, 5] means 6 modes 0 to 5)
    return np.array([0, 5]) # [start_k, end_k] for iteration. `range(NMK[-1])` makes it 0 to N-1

@pytest.fixture
def test_n():
    return 1 # Example n value for functions

@pytest.fixture
def test_k():
    return 1 # Example k value for functions

@pytest.fixture
def test_i(): # Example region index
    return 1 # Corresponds to region between a[0] and a[1]

@pytest.fixture
def test_r(a):
    return (a[0] + a[1]) / 2 # Example r value (between two cylinders)

@pytest.fixture
def test_z(d, h):
    return (d[1] - h) / 2 # Example z value within a fluid column


@pytest.fixture
def precomputed_m_k_arr(NMK, h):
    # Create a mock m_k_arr for testing functions that take it as input
    # These values are arbitrary but consistent for testing purposes
    num_k = NMK[-1] # From 0 to N-1
    arr = np.linspace(0.01, 0.5, num_k) # Dummy values for m_k_arr
    # Ensure m_k_arr[0] corresponds to m0 (k=0 case) if used.
    arr[0] = 0.1 # Example m0
    return arr

@pytest.fixture
def precomputed_N_k_arr(NMK):
    # Create a mock N_k_arr for testing functions that take it as input
    num_k = NMK[-1]
    return np.linspace(0.5, 2.0, num_k) # Dummy values for N_k_arr


# --- Test Cases ---

def test_omega(m0, h, g):
    expected = sqrt(m0 * np.tanh(m0 * h) * g)
    assert np.isclose(omega(m0, h, g), expected)

def test_scale(a):
    expected_result = np.array([a[0]/2, (a[0]+a[1])/2, (a[1]+a[2])/2]) # for a = [5, 10, 15]
    assert np.allclose(scale(a), expected_result)

def test_lambda_ni(test_n, test_i, h, d):
    expected = test_n * pi / (h - d[test_i])
    assert np.isclose(lambda_ni(test_n, test_i, h, d), expected)

# --- m_k functions ---
def test_m_k_entry_k0(m0, h):
    assert np.isclose(m_k_entry(0, m0, h), m0)

def test_m_k_entry_k_positive(h):
    # Test m_k_entry with a specific setup where we can verify the root
    # Target m_k_h = 2.0. Then m0*h*tanh(m0*h) = -(2.0 * tan(2.0)) approx 4.474
    # Solve Y*tanh(Y) = 4.474 for Y=m0*h. Y approx 4.475
    # So if h=100, m0 = 4.475/100 = 0.04475
    test_h = 10.0
    target_m_k_h = 2.0
    required_m0h_tanh_m0h = -(target_m_k_h * np.tan(target_m_k_h))

    def f_Y(Y):
        return Y * np.tanh(Y) - required_m0h_tanh_m0h

    # Find Y (m0*h)
    y_result = root_scalar(f_Y, x0=4.5, method='newton') # initial guess needs to be reasonable
    calculated_m0h = y_result.root
    test_m0 = calculated_m0h / test_h

    # Test for k=1 (where k is just used for the initial guess for root_scalar)
    k_val = 1
    # The actual numerical root will be what m_k_entry should return
    expected_m_k_val = target_m_k_h / test_h
    
    # We patch root_scalar for stability, but we also ensure the internal function logic is sound.
    # The actual call to root_scalar will use the lambda function from m_k_entry.
    # We will let the actual root_scalar run and verify the outcome.
    result = m_k_entry(k_val, test_m0, test_h)
    assert np.isclose(result, expected_m_k_val, rtol=1e-5, atol=1e-8)
    
    # Verify the assertion within m_k_entry doesn't fail for these values.
    # This is implicitly tested if the function runs without error.
    # A more explicit check:
    shouldnt_be_int = np.round(test_m0 * result / np.pi - 0.5, 4)
    assert shouldnt_be_int != np.floor(shouldnt_be_int)


def test_m_k(NMK, m0, h):
    # Test the vectorized m_k function
    num_modes = NMK[-1] # Number of modes (e.g., 5 for [0,5])
    result_array = m_k(NMK, m0, h)

    assert isinstance(result_array, np.ndarray)
    assert result_array.shape[0] == num_modes

    # Basic check for k=0 mode
    assert np.isclose(result_array[0], m0)

    # For k > 0, check that m_k_entry was likely called correctly
    # (Cannot check individual values easily without re-running root_scalar for each)
    # This test mainly ensures the vectorization and array creation.
    # For a full integration test, this would pass.
    # For unit testing m_k_entry, we have test_m_k_entry_k_positive.

def test_m_k_newton(h, m0):
    # Equation: k * tanh(k * h) - m0**2 / 9.8 = 0
    # Let m0=0.1, h=100. Then m0^2/9.8 = 0.01/9.8 = 0.00102
    # So k * tanh(k * 100) = 0.00102
    # For small k, tanh(k*h) ~ k*h
    # So k * (k*h) = 0.00102 => k^2 * h = 0.00102
    # k^2 = 0.00102 / 100 = 1.02e-5
    # k = sqrt(1.02e-5) approx 0.00319
    expected_k = np.sqrt(m0**2 / 9.8 / h) # approximation for small k*h
    assert np.isclose(m_k_newton(h, m0), expected_k, rtol=2e-2, atol=1e-5) # Looser rtol, e.g., 2% relative tolerance



# --- Coupling Integrals ---
def test_I_nm_n0m0(h, d):
    i = 1 # region 1 (between d[1] and d[2])
    dj = max(d[i], d[i+1]) # d[1]=20, d[2]=50 -> dj=50
    expected = h - dj # 100 - 50 = 50
    assert np.isclose(I_nm(0, 0, i, d, h), expected)

def test_I_nm_n0m_positive(h, d):
    n = 0
    m = 1
    i = 0 # region 0 (from 0 to d[1])
    dj = max(d[i], d[i+1]) # d[0]=0, d[1]=20 -> dj=20
    # d[i] is the *upper* boundary, d[i+1] is the *lower* boundary.
    # "integration bounds at -h and -d".
    # And "dj = max(d[i], d[i+1])". This suggests d[i] and d[i+1] are depths from surface.
    # Region 'i' is between a[i-1] and a[i], and extends from -h to -d[i].
    # Then I_nm is for "two i-type regions".
    # If d[i] is shallower than d[i+1], then d[i+1] is deeper, so dj = d[i+1].
    # (h-dj) means (h - deeper_depth_boundary).
    # If dj == d[i+1], it means d[i+1] is the deeper one.
    # The condition "if dj == d[i+1]: return 0" seems to be for this case.
    # But if dj == d[i] (meaning d[i] is deeper than d[i+1]), it also returns 0 for n>=1 and m=0.
    # Let's take i=0: d[0]=0, d[1]=20. dj = max(0, 20) = 20.
    # For n=0, m=1, dj=d[i+1] is true (20==20), so return 0.
    assert np.isclose(I_nm(n, m, i, d, h), 0)

    # Case where it's not 0
    n = 0
    m = 1
    i = 1 # Region between d[1]=20 and d[2]=50
    # dj is the *lower* integration limit (closer to h).
    # The original MATLAB code or documentation is needed for precise interpretation of `dj`.
    # Let's pick a case where the `if dj == d[i+1]: return 0` is false.
    # This implies d[i] > d[i+1] (meaning d[i] is deeper), and `dj = d[i]`.
    # In `d = [0, 20, 50]`, d[0] < d[1] < d[2]. This always makes `dj = d[i+1]`.
    # To test the `else` branch, Let's make a custom d.
    custom_d = np.array([0.0, 50.0, 20.0]) # d[1]=50, d[2]=20.
    i = 1 # region between 50 and 20
    dj = max(custom_d[i], custom_d[i+1]) # max(50, 20) = 50
    # For n=0, m>=1: if dj == d[i+1] (50 == 20) is FALSE. This branch will execute.
    lambda2 = lambda_ni(m, i + 1, h, custom_d) # m=1, i+1=2, d=custom_d, h=100
    expected = sqrt(2) * sin(lambda2 * (h - dj)) / lambda2
    assert np.isclose(I_nm(n, m, i, custom_d, h), expected)

def test_I_nm_n_positive_m0(h, d):
    n = 1
    m = 0
    i = 0 # region 0 (from 0 to d[1]=20)
    dj = max(d[i], d[i+1]) # max(0, 20) = 20
    # For n>=1, m=0: if dj == d[i] (20 == 0) is FALSE. This branch will execute.
    lambda1 = lambda_ni(n, i, h, d) # n=1, i=0, d=d, h=100
    expected = sqrt(2) * sin(lambda1 * (h - dj)) / lambda1
    assert np.isclose(I_nm(n, m, i, d, h), expected)

def test_I_nm_n_positive_m_positive(h, d):
    n = 1
    m = 1
    i = 0
    dj = max(d[i], d[i+1]) # max(0, 20) = 20

    # These lambda values match how I_nm calculates them
    lambda1 = lambda_ni(n, i, h, d)
    lambda2 = lambda_ni(m, i + 1, h, d) # This is crucial: I_nm uses i+1 for m's lambda

    # Calculate expected based on these lambda values.
    # Since lambda1 (based on d[0],d[1]) is likely not equal to lambda2 (based on d[1],d[2]),
    # this branch will be taken.
    frac1 = sin((lambda1 + lambda2)*(h-dj))/(lambda1 + lambda2)
    # Original test setup: d = array([0., 20., 50.])
    # lambda_ni(1, 0, 100, [0,20,50]) will be different from lambda_ni(1, 1, 100, [0,20,50])
    frac2 = sin((lambda1 - lambda2)*(h-dj))/(lambda1 - lambda2) # lambda1 != lambda2
    expected = frac1 + frac2
    assert np.isclose(I_nm(n, m, i, d, h), expected)

    # Test lambda1 == lambda2 case
    n_eq = 1
    m_eq = 1
    i_eq = 0
    # Make lambda1 == lambda2 for the _purpose of the test's expected_eq_
    # This means we need to ensure I_nm's internal lambdas are equal.
    # To do this, we need d[i] and d[i+1] to result in the same lambda_ni value.
    # Or, simpler, set d_eq such that lambda_ni(n_eq, i_eq, h, d_eq) and
    # lambda_ni(m_eq, i_eq + 1, h, d_eq) become numerically equal.
    # If lambda_ni depends on `d[idx]` and `d[idx+1]`, then for lambda_ni(1,0) == lambda_ni(1,1),
    # we need d_eq[0], d_eq[1] to lead to the same lambda as d_eq[1], d_eq[2].
    # This can happen if all depths are the same, e.g., d_eq = np.array([20.0, 20.0, 20.0])
    # Let's try to make the *internal* lambdas in I_nm equal.
    d_eq = np.array([50.0, 50.0, 50.0]) # Example: all depths same, so lambda_ni(1,0) == lambda_ni(1,1)

    # These are the lambdas I_nm will calculate internally for the call below:
    lambda1_internal = lambda_ni(n_eq, i_eq, h, d_eq)
    lambda2_internal = lambda_ni(m_eq, i_eq + 1, h, d_eq) # i_eq + 1 is key here!

    # Now calculate expected_eq based on these *internal* lambdas
    dj_eq = max(d_eq[i_eq], d_eq[i_eq+1]) # max(50, 50) = 50

    frac1_eq = sin((lambda1_internal + lambda2_internal)*(h-dj_eq))/(lambda1_internal + lambda2_internal)

    # Use np.isclose for the condition for the limit
    if np.isclose(lambda1_internal, lambda2_internal, atol=1e-12): # Use a small tolerance for floating point comparison
        frac2_eq = (h - dj_eq)
    else:
        frac2_eq = sin((lambda1_internal - lambda2_internal)*(h-dj_eq))/(lambda1_internal - lambda2_internal)

    expected_eq = frac1_eq + frac2_eq

    assert np.isclose(I_nm(n_eq, m_eq, i_eq, d_eq, h), expected_eq)


def test_I_mk_k0m0_small_m0h(test_i, d, m0, h, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    i = test_i # For I_mk, d[i] is the boundary depth
    dj = d[i] # dj = d[1] = 20.0
    m0_local = 0.01 # Small m0 for m0*h < 14 (0.01 * 100 = 1)
    NMK_local = NMK # Keep NMK for parameter compatibility
    N_k_arr_local = precomputed_N_k_arr.copy()
    N_k_arr_local[0] = 0.5 # A realistic N_k(0) for testing

    expected = (1/sqrt(N_k_arr_local[0])) * sinh(m0_local * (h - dj)) / m0_local
    assert np.isclose(I_mk(0, 0, i, d, m0_local, h, NMK_local, precomputed_m_k_arr, N_k_arr_local), expected)

def test_I_mk_k0m0_large_m0h(test_i, d, h, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    i = test_i
    dj = d[i]
    m0_local = 0.2 # Large m0 for m0*h >= 14 (0.2 * 100 = 20)
    NMK_local = NMK
    N_k_arr_local = precomputed_N_k_arr.copy()
    N_k_arr_local[0] = 0.5 # A realistic N_k(0) for testing

    expected = sqrt(2 * h / m0_local) * (exp(- m0_local * dj) - exp(m0_local * dj - 2 * m0_local * h))
    assert np.isclose(I_mk(0, 0, i, d, m0_local, h, NMK_local, precomputed_m_k_arr, N_k_arr_local), expected)

def test_I_mk_k_positive_m0(test_k, test_i, d, m0, h, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    if test_k == 0: pytest.skip("Test for k>0")
    i = test_i
    dj = d[i]
    
    local_m_k_k = precomputed_m_k_arr[test_k]
    N_k_arr_k = precomputed_N_k_arr[test_k]

    expected = (1/sqrt(N_k_arr_k)) * sin(local_m_k_k * (h - dj)) / local_m_k_k
    assert np.isclose(I_mk(0, test_k, i, d, m0, h, NMK, precomputed_m_k_arr, precomputed_N_k_arr), expected)

def test_I_mk_k0_m_positive_small_m0h(test_n, test_i, d, m0, h, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    if test_n == 0: pytest.skip("Test for m>0")
    i = test_i
    dj = d[i]
    m_val = test_n # Using test_n for m

    m0_local = 0.01 # Small m0
    N_k_arr_local = precomputed_N_k_arr.copy()
    N_k_arr_local[0] = 0.5

    num = (-1)**m_val * sqrt(2) * (1/sqrt(N_k_arr_local[0])) * m0_local * sinh(m0_local * (h - dj))
    denom = (m0_local**2 + lambda_ni(m_val, i, h, d) **2)
    expected = num/denom
    assert np.isclose(I_mk(m_val, 0, i, d, m0_local, h, NMK, precomputed_m_k_arr, N_k_arr_local), expected)

def test_I_mk_k0_m_positive_large_m0h(test_n, test_i, d, m0, h, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    if test_n == 0: pytest.skip("Test for m>0")
    i = test_i
    dj = d[i]
    m_val = test_n # Using test_n for m

    m0_local = 0.2 # Large m0
    N_k_arr_local = precomputed_N_k_arr.copy()
    N_k_arr_local[0] = 0.5

    num = (-1)**m_val * 2 * sqrt(h * m0_local ** 3) *(exp(- m0_local * dj) - exp(m0_local * dj - 2 * m0_local * h))
    denom = (m0_local**2 + lambda_ni(m_val, i, h, d) **2)
    expected = num/denom
    assert np.isclose(I_mk(m_val, 0, i, d, m0_local, h, NMK, precomputed_m_k_arr, N_k_arr_local), expected)

def test_I_mk_k_positive_m_positive(test_n, test_k, test_i, d, m0, h, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    if test_n == 0 or test_k == 0: pytest.skip("Test for m>0 and k>0")
    i = test_i
    dj = d[i]
    m_val = test_n

    local_m_k_k = precomputed_m_k_arr[test_k]
    N_k_arr_k = precomputed_N_k_arr[test_k]
    lambda1 = lambda_ni(m_val, i, h, d)

    if np.isclose(abs(local_m_k_k), lambda1): # Check for the equality branch
        expected = (h - dj)/2
    else:
        frac1 = sin((local_m_k_k + lambda1)*(h-dj))/(local_m_k_k + lambda1)
        frac2 = sin((local_m_k_k - lambda1)*(h-dj))/(local_m_k_k - lambda1)
        expected = sqrt(2)/2 * (1/sqrt(N_k_arr_k)) * (frac1 + frac2)
    assert np.isclose(I_mk(m_val, test_k, i, d, m0, h, NMK, precomputed_m_k_arr, precomputed_N_k_arr), expected)

# Testing _full versions implicitly relies on m_k and N_k_full being correct.
# We skip these for brevity since the main ones are tested with precomputed arrays.

# --- b-vector computation ---
def test_b_potential_entry_n0(test_i, d, heaving, h, a):
    j = test_i + (d[test_i] < d[test_i+1]) # Example for i=1, d[1]=20, d[2]=50. j=2
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
    # Custom d to make d[i] > d[i+1]
    custom_d = np.array([0.0, 50.0, 20.0]) # d[1]=50, d[2]=20
    i = 1
    num = - sqrt(2) * a[i] * sin(lambda_ni(test_n, i+1, h, custom_d) * (h-custom_d[i]))
    denom = (2 * (h - custom_d[i]) * lambda_ni(test_n, i+1, h, custom_d))
    expected = num/denom
    assert np.isclose(b_velocity_entry(test_n, i, heaving, a, h, custom_d), expected)

def test_b_velocity_entry_n_positive_di_smaller(test_n, heaving, a, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    i = 0 # d[0]=0, d[1]=20. d[i] < d[i+1]
    num = sqrt(2) * a[i] * sin(lambda_ni(test_n, i, h, d) * (h-d[i+1]))
    denom = (2 * (h - d[i+1]) * lambda_ni(test_n, i, h, d))
    expected = num/denom
    assert np.isclose(b_velocity_entry(test_n, i, heaving, a, h, d), expected)

def test_b_velocity_end_entry_k0_small_m0h(test_i, heaving, a, h, d, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    m0_local = 0.01 # For m0*h < 14
    constant = - heaving[test_i] * a[test_i]/(2 * (h - d[test_i]))
    N_k_arr_local = precomputed_N_k_arr.copy()
    N_k_arr_local[0] = 0.5
    expected = constant * (1/sqrt(N_k_arr_local[0])) * sinh(m0_local * (h - d[test_i])) / m0_local
    assert np.isclose(b_velocity_end_entry(0, test_i, heaving, a, h, d, m0_local, NMK, precomputed_m_k_arr, N_k_arr_local), expected)

def test_b_velocity_end_entry_k0_large_m0h(test_i, heaving, a, h, d, NMK, precomputed_m_k_arr, precomputed_N_k_arr):
    m0_local = 0.2 # For m0*h >= 14
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
    d_val = d[1] # Using d[1] for the depth
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
    assert np.isclose(R_1n(0, test_r, test_i, h, d, a), 0.5)

def test_R_1n_n_positive(test_n, test_i, test_r, h, d, a):
    if test_n == 0: pytest.skip("Test for n>0")
    local_scale = scale(a)
    expected = besseli(0, lambda_ni(test_n, test_i, h, d) * test_r) / besseli(0, lambda_ni(test_n, test_i, h, d) * local_scale[test_i])
    assert np.isclose(R_1n(test_n, test_r, test_i, h, d, a), expected)

def test_R_1n_n_negative(test_i, test_r, h, d, a):
    with pytest.raises(ValueError, match="Invalid value for n"):
        R_1n(-1, test_r, test_i, h, d, a)

def test_diff_R_1n_n0(test_i, test_r, h, d, a):
    assert np.isclose(diff_R_1n(0, test_r, test_i, h, d, a), 0)

def test_diff_R_1n_n_positive(test_n, test_i, test_r, h, d, a):
    if test_n == 0: pytest.skip("Test for n>0")
    local_scale = scale(a)
    top = lambda_ni(test_n, test_i, h, d) * besseli(1, lambda_ni(test_n, test_i, h, d) * test_r)
    bottom = besseli(0, lambda_ni(test_n, test_i, h, d) * local_scale[test_i])
    expected = top / bottom
    assert np.isclose(diff_R_1n(test_n, test_r, test_i, h, d, a), expected)

# --- Bessel K Radial Eigenfunction ---
def test_R_2n_i0_raises_error(test_r, h, d, a):
    with pytest.raises(ValueError, match="i cannot be 0"):
        R_2n(1, test_r, 0, a, h, d)

def test_R_2n_n0(a, test_r, h, d):
    # Using i=1 for R_2n
    i = 1
    expected = 0.5 * np.log(test_r / a[i]) # Note: R_2n uses a[i] as inner radius
    assert np.isclose(R_2n(0, test_r, i, a, h, d), expected)

def test_R_2n_n_positive(test_n, a, test_r, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    i = 1
    local_scale = scale(a)
    expected = besselk(0, lambda_ni(test_n, i, h, d) * test_r) / besselk(0, lambda_ni(test_n, i, h, d) * local_scale[i])
    assert np.isclose(R_2n(test_n, test_r, i, a, h, d), expected)

def test_diff_R_2n_n0(test_r, h, d, a):
    i = 1
    expected = 1 / (2 * test_r)
    assert np.isclose(diff_R_2n(0, test_r, i, h, d, a), expected)

def test_diff_R_2n_n_positive(test_n, test_r, h, d, a):
    if test_n == 0: pytest.skip("Test for n>0")
    i = 1
    local_scale = scale(a)
    top = - lambda_ni(test_n, i, h, d) * besselk(1, lambda_ni(test_n, i, h, d) * test_r)
    bottom = besselk(0, lambda_ni(test_n, i, h, d) * local_scale[i])
    expected = top / bottom
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

# --- Region e radial eigenfunction ---
def test_Lambda_k_k0(test_r, m0, a, NMK, h, precomputed_m_k_arr, precomputed_N_k_arr):
    local_scale = scale(a)
    expected = besselh(0, m0 * test_r) / besselh(0, m0 * local_scale[-1])
    assert np.isclose(Lambda_k(0, test_r, m0, a, NMK, h, precomputed_m_k_arr, precomputed_N_k_arr), expected)

def test_Lambda_k_k_positive(test_k, test_r, m0, a, NMK, h, precomputed_m_k_arr, precomputed_N_k_arr):
    if test_k == 0: pytest.skip("Test for k>0")
    local_scale = scale(a)
    local_m_k_k = precomputed_m_k_arr[test_k]
    expected = besselk(0, local_m_k_k * test_r) / besselk(0, local_m_k_k * local_scale[-1])
    assert np.isclose(Lambda_k(test_k, test_r, m0, a, NMK, h, precomputed_m_k_arr, precomputed_N_k_arr), expected)

def test_diff_Lambda_k_k0(test_r, m0, NMK, h, a, precomputed_m_k_arr, precomputed_N_k_arr):
    local_scale = scale(a)
    numerator = -(m0 * besselh(1, m0 * test_r))
    denominator = besselh(0, m0 * local_scale[-1])
    expected = numerator / denominator
    assert np.isclose(diff_Lambda_k(0, test_r, m0, NMK, h, a, precomputed_m_k_arr, precomputed_N_k_arr), expected)

def test_diff_Lambda_k_k_positive(test_k, test_r, m0, NMK, h, a, precomputed_m_k_arr, precomputed_N_k_arr):
    if test_k == 0: pytest.skip("Test for k>0")
    local_m_k_k = precomputed_m_k_arr[test_k]
    local_scale = scale(a)
    numerator = -(local_m_k_k * besselk(1, local_m_k_k * test_r))
    denominator = besselk(0, local_m_k_k * local_scale[-1])
    expected = numerator / denominator
    assert np.isclose(diff_Lambda_k(test_k, test_r, m0, NMK, h, a, precomputed_m_k_arr, precomputed_N_k_arr), expected)

# --- N_k_multi ---
def test_N_k_multi_k0(m0, h, NMK):
    # Test N_k_multi without passing m_k_arr
    expected = 1 / 2 * (1 + sinh(2 * m0 * h) / (2 * m0 * h))
    assert np.isclose(N_k_multi(0, m0, h, NMK, None), expected) # Test case where m_k_arr is None

def test_N_k_multi_k_positive(test_k, m0, h, NMK, precomputed_m_k_arr):
    if test_k == 0: pytest.skip("Test for k>0")
    # Test N_k_multi with precomputed_m_k_arr
    local_m_k_k = precomputed_m_k_arr[test_k]
    expected = 1 / 2 * (1 + sin(2 * local_m_k_k * h) / (2 * local_m_k_k * h))
    assert np.isclose(N_k_multi(test_k, m0, h, NMK, precomputed_m_k_arr), expected)

# --- e-region vertical eigenfunctions ---
def test_Z_k_e_k0_small_m0h(test_z, m0, h, NMK, precomputed_m_k_arr):
    m0_local = 0.01
    # Patch N_k_multi to return a known value for this test
    with patch('multi_equations.N_k_multi', return_value=0.5): # Mock N_k_multi for simplicity
        expected = 1 / sqrt(0.5) * cosh(m0_local * (test_z + h))
        assert np.isclose(Z_k_e(0, test_z, m0_local, h, NMK, precomputed_m_k_arr), expected)

def test_Z_k_e_k0_large_m0h(test_z, m0, h, NMK, precomputed_m_k_arr):
    m0_local = 0.2
    expected = sqrt(2 * m0_local * h) * (exp(m0_local * test_z) + exp(-m0_local * (test_z + 2*h)))
    assert np.isclose(Z_k_e(0, test_z, m0_local, h, NMK, precomputed_m_k_arr), expected)

def test_Z_k_e_k_positive(test_k, test_z, m0, h, NMK, precomputed_m_k_arr):
    if test_k == 0: pytest.skip("Test for k>0")
    local_m_k_k_from_precomputed = precomputed_m_k_arr[test_k] # This is 0.1325

    # Patch N_k_multi AND m_k
    with patch('multi_equations.N_k_multi', return_value=0.8), \
         patch('multi_equations.m_k', return_value=precomputed_m_k_arr) as mock_m_k: # Mock m_k to return the whole array
        expected = 1 / sqrt(0.8) * cos(local_m_k_k_from_precomputed * (test_z + h))
        print(f"DEBUG TEST Z_k_e: Expected calculated in test: {expected}") 
        assert np.isclose(Z_k_e(test_k, test_z, m0, h, NMK, precomputed_m_k_arr), expected)

def test_diff_Z_k_e_k0_small_m0h(test_z, m0, h, NMK, precomputed_m_k_arr):
    m0_local = 0.01
    with patch('multi_equations.N_k_multi', return_value=0.5):
        expected = 1 / sqrt(0.5) * m0_local * sinh(m0_local * (test_z + h))
        assert np.isclose(diff_Z_k_e(0, test_z, m0_local, h, NMK, precomputed_m_k_arr), expected)

def test_diff_Z_k_e_k0_large_m0h(test_z, m0, h, NMK, precomputed_m_k_arr):
    m0_local = 0.2
    expected = m0_local * sqrt(2 * h * m0_local) * (exp(m0_local * test_z) - exp(-m0_local * (test_z + 2*h)))
    assert np.isclose(diff_Z_k_e(0, test_z, m0_local, h, NMK, precomputed_m_k_arr), expected)

def test_diff_Z_k_e_k_positive(test_k, test_z, m0, h, NMK, precomputed_m_k_arr):
    if test_k == 0: pytest.skip("Test for k>0")
    local_m_k_k_from_precomputed = precomputed_m_k_arr[test_k] # This is 0.1325

    # Patch N_k_multi AND m_k
    with patch('multi_equations.N_k_multi', return_value=0.8), \
         patch('multi_equations.m_k', return_value=precomputed_m_k_arr) as mock_m_k: # Mock m_k to return the whole array
        expected = -1 / sqrt(0.8) * local_m_k_k_from_precomputed * sin(local_m_k_k_from_precomputed * (test_z + h))
        print(f"DEBUG TEST diff_Z_k_e: Expected calculated in test: {expected}")
        assert np.isclose(diff_Z_k_e(test_k, test_z, m0, h, NMK, precomputed_m_k_arr), expected)

# --- To calculate hydrocoefficients ---
def test_int_R_1n_n0_i0(a, h, d):
    # i=0 (innermost region), inner radius 0
    expected = a[0]**2/4 - 0**2/4
    assert np.isclose(int_R_1n(0, 0, a, h, d), expected)

def test_int_R_1n_n0_i_positive(a, h, d):
    # i=1 (middle region), inner radius a[0]
    expected = a[1]**2/4 - a[0]**2/4
    assert np.isclose(int_R_1n(1, 0, a, h, d), expected)

def test_int_R_1n_n_positive(test_n, test_i, a, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    local_scale = scale(a)
    lambda0 = lambda_ni(test_n, test_i, h, d)
    inner_term = (0 if test_i == 0 else a[test_i-1] * besseli(1, lambda0 * a[test_i-1]))
    top = a[test_i] * besseli(1, lambda0 * a[test_i]) - inner_term
    bottom = lambda0 * besseli(0, lambda0 * local_scale[test_i])
    expected = top / bottom
    assert np.isclose(int_R_1n(test_n, test_i, a, h, d), expected) # Switched args order for test_n, test_i

def test_int_R_2n_i0_raises_error(test_n, a, h, d):
    with pytest.raises(ValueError, match="i cannot be 0"):
        int_R_2n(0, test_n, a, h, d)

def test_int_R_2n_n0(a, h, d):
    i = 1 # Test for i=1
    expected = (a[i-1]**2 * (2*np.log(a[i]/a[i-1]) + 1) - a[i]**2)/8
    assert np.isclose(int_R_2n(i, 0, a, h, d), expected)

def test_int_R_2n_n_positive(test_n, a, h, d):
    if test_n == 0: pytest.skip("Test for n>0")
    i = 1 # Test for i=1
    local_scale = scale(a)
    lambda0 = lambda_ni(test_n, i, h, d)
    top = a[i] * besselk(1, lambda0 * a[i]) - a[i-1] * besselk(1, lambda0 * a[i-1])
    bottom = - lambda0 * besselk(0, lambda0 * local_scale[i])
    expected = top / bottom
    assert np.isclose(int_R_2n(i, test_n, a, h, d), expected) # Switched args order for test_n, i

def test_int_phi_p_i_no_coef_i0(a, h, d):
    i = 0
    denom = 16 * (h - d[i])
    num = a[i]**2 * (4 * (h - d[i])**2 - a[i]**2)
    expected = num/denom
    assert np.isclose(int_phi_p_i_no_coef(i, h, d, a), expected)

def test_int_phi_p_i_no_coef_i_positive(a, h, d):
    i = 1
    denom = 16 * (h - d[i])
    num = (a[i]**2 * (4 * (h - d[i])**2 - a[i]**2) - a[i-1]**2 * (4 * (h - d[i])**2 - a[i-1]**2))
    expected = num/denom
    assert np.isclose(int_phi_p_i_no_coef(i, h, d, a), expected)

def test_z_n_d_n0():
    assert np.isclose(z_n_d(0), 1)

def test_z_n_d_n_positive(test_n):
    if test_n == 0: pytest.skip("Test for n>0")
    expected = sqrt(2) * (-1)**test_n
    assert np.isclose(z_n_d(test_n), expected)

def test_excitation_phase(coeff, m0, a): # <-- 'a' is present
    local_scale_last = scale(a)[-1] # 'a' is the actual array
    expected = -(pi/2) + np.angle(coeff) - np.angle(besselh(0, m0 * local_scale_last))
    assert np.isclose(excitation_phase(coeff, m0, a), expected) # excitation_phase needs 'a' too