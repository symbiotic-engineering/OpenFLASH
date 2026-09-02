# test_problem_cache.py

import sys
import os
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock # MagicMock is useful for mocking callable attributes

# --- Path Setup ---
# Adjust the path to import from package's 'src' directory.
current_dir = os.path.dirname(__file__)
package_base_dir = os.path.join(current_dir, '..')
src_dir = os.path.join(package_base_dir, 'src')
sys.path.insert(0, os.path.abspath(src_dir))

# Import the class to be tested
from openflash.problem_cache import ProblemCache
# No need to import MEEMProblem or multi_equations directly here for these tests,
# as we'll mock the 'problem' object and the functions.

# --- Fixtures ---

@pytest.fixture
def mock_problem():
    """Provides a mock MEEMProblem instance for the ProblemCache."""
    mock = Mock()
    # Add any attributes or methods that ProblemCache might try to access on 'problem'
    # For now, ProblemCache only stores it, but if other methods access problem.frequencies, etc.
    return mock

@pytest.fixture
def problem_cache(mock_problem):
    """Provides an initialized ProblemCache instance for each test."""
    return ProblemCache(problem=mock_problem)

@pytest.fixture
def sample_A_template():
    """Provides a sample NumPy array for A_template."""
    return np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=complex)

@pytest.fixture
def sample_b_template():
    """Provides a sample NumPy array for b_template."""
    return np.array([5+5j, 6+6j], dtype=complex)

@pytest.fixture
def mock_calc_func():
    """Provides a simple mock callable for m0-dependent calculation functions."""
    return MagicMock(return_value=100.0 + 50.0j) # Example return value for the function

@pytest.fixture
def mock_m_k_entry_func():
    """Provides a mock for the m_k_entry function."""
    return MagicMock(name='m_k_entry_func')

@pytest.fixture
def mock_N_k_func():
    """Provides a mock for the N_k function."""
    return MagicMock(name='N_k_func')

# --- Test Cases ---

def test_initialization(problem_cache, mock_problem):
    """Test that ProblemCache initializes correctly."""
    assert problem_cache.problem is mock_problem
    assert problem_cache.A_template is None
    assert problem_cache.b_template is None
    assert problem_cache.m0_dependent_A_indices == []
    assert problem_cache.m0_dependent_b_indices == []
    assert problem_cache.m_k_entry_func is None
    assert problem_cache.N_k_func is None

def test_set_and_get_A_template(problem_cache, sample_A_template):
    """Test setting and retrieving the A_template."""
    problem_cache._set_A_template(sample_A_template)
    retrieved_A = problem_cache._get_A_template()

    # Check if the template was set
    np.testing.assert_array_equal(retrieved_A, sample_A_template)
    # Check that a copy is returned (to prevent external modification of the stored template)
    assert retrieved_A is not sample_A_template

def test_get_A_template_not_set(problem_cache):
    """Test ValueError is raised if A_template is accessed before being set."""
    with pytest.raises(ValueError, match="A_template has not been set."):
        problem_cache._get_A_template()

def test_set_and_get_b_template(problem_cache, sample_b_template):
    """Test setting and retrieving the b_template."""
    problem_cache._set_b_template(sample_b_template)
    retrieved_b = problem_cache._get_b_template()

    # Check if the template was set
    np.testing.assert_array_equal(retrieved_b, sample_b_template)
    # Check that a copy is returned
    assert retrieved_b is not sample_b_template

def test_get_b_template_not_set(problem_cache):
    """Test ValueError is raised if b_template is accessed before being set."""
    with pytest.raises(ValueError, match="b_template has not been set."):
        problem_cache._get_b_template()

def test_add_m0_dependent_A_entry(problem_cache, mock_calc_func):
    """Test adding m0-dependent A matrix entries."""
    problem_cache._add_m0_dependent_A_entry(0, 0, mock_calc_func)
    assert problem_cache.m0_dependent_A_indices == [(0, 0, mock_calc_func)]

    another_mock_func = MagicMock(return_value=200.0)
    problem_cache._add_m0_dependent_A_entry(1, 2, another_mock_func)
    assert problem_cache.m0_dependent_A_indices == [
        (0, 0, mock_calc_func),
        (1, 2, another_mock_func)
    ]

def test_add_m0_dependent_b_entry(problem_cache, mock_calc_func):
    """Test adding m0-dependent b vector entries."""
    problem_cache._add_m0_dependent_b_entry(0, mock_calc_func)
    assert problem_cache.m0_dependent_b_indices == [(0, mock_calc_func)]

    another_mock_func = MagicMock(return_value=300.0)
    problem_cache._add_m0_dependent_b_entry(5, another_mock_func)
    assert problem_cache.m0_dependent_b_indices == [
        (0, mock_calc_func),
        (5, another_mock_func)
    ]

def test_set_m_k_and_N_k_funcs(problem_cache, mock_m_k_entry_func, mock_N_k_func):
    """Test setting the m_k_entry_func and N_k_func references."""
    problem_cache._set_m_k_and_N_k_funcs(mock_m_k_entry_func, mock_N_k_func)

    assert problem_cache.m_k_entry_func is mock_m_k_entry_func
    assert problem_cache.N_k_func is mock_N_k_func

# --- New Test Cases Added ---

def test_get_integration_constants_not_set(problem_cache):
    """Test ValueError is raised if integration constants are accessed before being set."""
    # Ensure attributes are initialized to None to simulate uninitialized state correctly.
    problem_cache.int_R1_vals = None
    
    with pytest.raises(ValueError, match="Integration constants have not been set."):
        problem_cache._get_integration_constants()

def test_set_and_get_integration_constants(problem_cache):
    """Test setting and retrieving integration constants."""
    mock_R1 = {'key': 1}
    mock_R2 = {'key': 2}
    mock_phi = np.array([1.0, 2.0])
    
    problem_cache._set_integration_constants(mock_R1, mock_R2, mock_phi)
    
    r1, r2, phi = problem_cache._get_integration_constants()
    
    assert r1 == mock_R1
    assert r2 == mock_R2
    np.testing.assert_array_equal(phi, mock_phi)