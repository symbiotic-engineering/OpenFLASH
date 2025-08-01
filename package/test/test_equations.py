import numpy as np
import pytest
import sys
import os


# --- Path Setup ---
# Adjust the path to import from package's 'src' directory.
current_dir = os.path.dirname(__file__)
package_base_dir = os.path.join(current_dir, '..')
src_dir = os.path.join(package_base_dir, 'src')
sys.path.insert(0, os.path.abspath(src_dir))

from equations import (
    m_k, lambda_n1, lambda_n2, R_1n_1, R_2n_2,
    N_k, Z_n_e, phi_p_i1, diff_phi_p_i1_dz,
)

def test_lambda_n1():
    h = 1.0
    d1 = 0.2
    n = 1
    expected = n * np.pi / (h - d1)
    assert np.isclose(lambda_n1(n, h, d1), expected)

def test_m_k_returns_real_number():
    # Just test that it returns a real number and doesn't crash
    val = m_k(k=1, m0=1.0, h=1.0)
    assert np.isreal(val)
    assert val > 0

def test_R_1n_1_n0():
    result = R_1n_1(n=0, r=1.0, a2=2.0, h=1.0, d1=0.5)
    assert result == 0.5

def test_R_2n_2_n0_log_behavior():
    r = 1.5
    a2 = 1.0
    val = R_2n_2(n=0, r=r, a2=a2, h=1.0, d2=0.5)
    expected = 0.5 * np.log(r / a2)
    assert np.isclose(val, expected)

def test_N_k_zero():
    m0 = 1.0
    h = 1.0
    result = N_k(0, m0, h)
    expected = 1 / 2 * (1 + np.sinh(2 * m0 * h) / (2 * m0 * h))
    assert np.isclose(result, expected)

def test_phi_p_i1():
    r = 0.5
    z = -0.5
    h = 1.0
    d1 = 0.2
    val = phi_p_i1(r, z, h, d1)
    expected = (1 / (2 * (h - d1))) * ((z + h) ** 2 - (r ** 2) / 2)
    assert np.isclose(val, expected)

def test_diff_phi_p_i1_dz():
    z = -0.5
    h = 1.0
    d1 = 0.2
    val = diff_phi_p_i1_dz(z, h, d1)
    expected = (h + z) / (h - d1)
    assert np.isclose(val, expected)
