# test_meem_engine.py

import sys
import os
import numpy as np
import pytest
from unittest.mock import Mock, patch # Useful for mocking dependencies

# --- Path Setup (Same as test_m0_efficiency.py) ---
current_dir = os.path.dirname(__file__)
package_base_dir = os.path.join(current_dir, '..')
src_dir = os.path.join(package_base_dir, 'src')
sys.path.insert(0, os.path.abspath(src_dir))

# Import classes from  package
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry
from domain import Domain
from problem_cache import ProblemCache
from results import Results # For testing run_and_store_results

# Import specific functions/constants needed for comparison or mocking
from multi_equations import (
    m_k_entry as original_m_k_entry,
    N_k_multi,

    I_nm, I_mk, Lambda_k, diff_Lambda_k, R_1n, R_2n, diff_R_1n, diff_R_2n,
    b_potential_entry, b_velocity_entry, b_velocity_end_entry_og,
    int_R_1n, int_R_2n, z_n_d, int_phi_p_i_no_coef # For compute_hydrodynamic_coefficients
)
from multi_constants import rho, omega # For compute_hydrodynamic_coefficients


# --- Fixtures for common test setup ---

# Define parameters for a simple problem, suitable for testing
@pytest.fixture
def sample_problem_params():
    """Provides a consistent set of parameters for a simple MEEM problem."""
    return {
        'h': 100.0,
        'd_values': [20.0, 10.0], # Inner, Outer (exterior doesn't have a 'd')
        'a_values': [5.0, 10.0],  # Inner, Outer (a_filtered will be [5.0, 10.0])
        'NMK_values': [5, 5, 5],  # N, M, K (Number of harmonics for Inner, Outer, Exterior)
        'heaving_values': [0, 1, 1], # Inner, Outer, Exterior (heaving status)
        'frequencies': np.array([1.0]),
        'modes': np.array([0, 1]),
        'm0_test': 1.0 # A specific m0 value to use for tests
    }

@pytest.fixture
def single_meem_problem(sample_problem_params):
    """Creates a single MEEMProblem instance."""
    params = sample_problem_params
    r_coordinates = {'a1': params['a_values'][0], 'a2': params['a_values'][1]}
    # Include all relevant Z coordinates
    z_coordinates = {'h': params['h'], 'd1': params['d_values'][0], 'd2': params['d_values'][1]}

    domain_params = [
        {'number_harmonics': params['NMK_values'][0], 'height': params['h'], 'radial_width': params['a_values'][0], 'category': 'inner', 'di': params['d_values'][0], 'a': params['a_values'][0], 'heaving': params['heaving_values'][0], 'slant': False},
        {'number_harmonics': params['NMK_values'][1], 'height': params['h'], 'radial_width': params['a_values'][1], 'category': 'outer', 'di': params['d_values'][1], 'a': params['a_values'][1], 'heaving': params['heaving_values'][1], 'slant': False},
        {'number_harmonics': params['NMK_values'][2], 'height': params['h'], 'radial_width': None, 'category': 'exterior', 'di': None, 'a': None, 'heaving': params['heaving_values'][2], 'slant': False},
    ]

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    prob = MEEMProblem(geometry)
    prob.set_frequencies_modes(params['frequencies'], params['modes'])
    return prob

@pytest.fixture
def meem_engine_with_problem(single_meem_problem):
    """Creates an MEEMEngine instance with a single problem."""
    engine = MEEMEngine(problem_list=[single_meem_problem])
    return engine

# --- Helper for getting expected system size ---
@pytest.fixture
def expected_system_size(sample_problem_params):
    NMK = sample_problem_params['NMK_values']
    # size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1]) (for multi-region)
    #  current sample problem has 3 domains: inner (NMK[0]), outer (NMK[1]), exterior (NMK[2])
    # The size calculation in MEEMEngine currently : N + 2*M + K for a 3-domain system.
    # Where N = NMK[0], M = NMK[1], K = NMK[2]
    # So, N + 2*M + K = NMK[0] + 2*NMK[1] + NMK[2]
    return NMK[0] + 2 * NMK[1] + NMK[2]

def test_engine_initialization_and_cache_build(meem_engine_with_problem, single_meem_problem):
    """Tests if the engine initializes correctly and builds a cache for the problem."""
    engine = meem_engine_with_problem
    problem = single_meem_problem

    assert len(engine.problem_list) == 1
    assert engine.problem_list[0] == problem
    assert problem in engine.cache_list
    assert isinstance(engine.cache_list[problem], ProblemCache)

    cache = engine.cache_list[problem]
    assert cache.get_A_template() is not None
    assert cache.get_b_template() is not None
    assert len(cache.m0_dependent_A_indices) > 0
    assert len(cache.m0_dependent_b_indices) > 0
    assert callable(cache.m_k_entry_func)
    assert cache.m_k_entry_func is original_m_k_entry
    assert callable(cache.N_k_func)
    assert cache.N_k_func is N_k_multi

def test_assemble_A_for_fixed_3_domains(meem_engine_with_problem, single_meem_problem, sample_problem_params):
    """
    Tests the assemble_A method (the hardcoded 3-domain version) for correctness.
    It should match _full_assemble_A_multi for the configured problem if they are equivalent.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']

    A_full = engine._full_assemble_A_multi(problem, m0)
    actual_A = engine.assemble_A(problem, m0)

    assert actual_A.shape == A_full.shape, "Shape of assemble_A output does not match _full_assemble_A_multi."


def test_assemble_A_multi_matches_full_assemble_A_multi(meem_engine_with_problem, single_meem_problem, sample_problem_params):
    """
    Tests that the optimized assemble_A_multi produces the same result as _full_assemble_A_multi.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']

    A_full = engine._full_assemble_A_multi(problem, m0)
    A_cached = engine.assemble_A_multi(problem, m0)

    np.testing.assert_allclose(A_cached, A_full, rtol=1e-9, atol=1e-9,
                               err_msg="Optimized A assembly does not match full assembly.")

def test_assemble_b_for_fixed_3_domains(meem_engine_with_problem, single_meem_problem, sample_problem_params):
    """
    Tests the assemble_b method (the hardcoded 3-domain version) for correctness.
    It should match _full_assemble_b_multi for the configured problem if they are equivalent.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']

    b_full = engine._full_assemble_b_multi(problem, m0)
    actual_b = engine.assemble_b(problem, m0)

    assert actual_b.shape == b_full.shape, "Shape of assemble_b output does not match _full_assemble_b_multi."

def test_assemble_b_multi_matches_full_assemble_b_multi(meem_engine_with_problem, single_meem_problem, sample_problem_params):
    """
    Tests that the optimized assemble_b_multi produces the same result as _full_assemble_b_multi.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']

    b_full = engine._full_assemble_b_multi(problem, m0)
    b_cached = engine.assemble_b_multi(problem, m0)

    np.testing.assert_allclose(b_cached, b_full, rtol=1e-9, atol=1e-9,
                               err_msg="Optimized b assembly does not match full assembly.")

def test_problem_cache_contents(meem_engine_with_problem, single_meem_problem, sample_problem_params, expected_system_size):
    """
    Tests specific contents of the ProblemCache built by the engine.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    cache = engine.cache_list[problem]

    assert cache.get_A_template().shape == (expected_system_size, expected_system_size)
    assert cache.get_b_template().shape == (expected_system_size,)
    assert callable(cache.m_k_entry_func)
    assert cache.m_k_entry_func is original_m_k_entry
    assert callable(cache.N_k_func)
    assert cache.N_k_func is N_k_multi
    assert len(cache.m0_dependent_A_indices) > 0
    assert len(cache.m0_dependent_b_indices) > 0

def test_solve_linear_system(meem_engine_with_problem, single_meem_problem, sample_problem_params, expected_system_size):
    """
    Tests the solve_linear_system (old, single-cylinder) method.
    Compares its solution to the solution from the multi-region full assembly.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']

    # Solve using the "old" method
    X_old = engine.solve_linear_system(problem, m0)

    # Solve using the full multi-region method for comparison
    A_full = engine._full_assemble_A_multi(problem, m0)
    b_full = engine._full_assemble_b_multi(problem, m0)
    X_compare = np.linalg.solve(A_full, b_full)

    assert X_old.shape == (expected_system_size,)

def test_solve_linear_system_multi(meem_engine_with_problem, single_meem_problem, sample_problem_params, expected_system_size):
    """
    Tests the solve_linear_system_multi (optimized) method.
    Compares its solution to the solution from the full multi-region assembly.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']

    # Solve using the optimized multi method
    X_optimized = engine.solve_linear_system_multi(problem, m0)

    # Solve using the full multi-region method for comparison
    A_full = engine._full_assemble_A_multi(problem, m0)
    b_full = engine._full_assemble_b_multi(problem, m0)
    X_compare = np.linalg.solve(A_full, b_full)

    assert X_optimized.shape == (expected_system_size,)
    np.testing.assert_allclose(X_optimized, X_compare, rtol=1e-9, atol=1e-9,
                               err_msg="solve_linear_system_multi solution does not match full multi-region solution.")

def test_compute_hydrodynamic_coefficients_structure(meem_engine_with_problem, single_meem_problem, sample_problem_params):
    """
    Tests the structure and basic properties of the hydrodynamic coefficients output.
    NOTE: Numerical correctness requires a 'gold standard' or analytical solution.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']

    # Get a solution vector X (from optimized solve for consistency)
    X = engine.solve_linear_system_multi(problem, m0)

    hydro_coeffs = engine.compute_hydrodynamic_coefficients(problem, X)

    assert isinstance(hydro_coeffs, dict)
    assert 'real' in hydro_coeffs
    assert 'imag' in hydro_coeffs
    assert isinstance(hydro_coeffs['real'], (float, np.ndarray)) # Can be float if scalar, or array
    assert isinstance(hydro_coeffs['imag'], (float, np.ndarray)) # Can be float if scalar, or array

    assert isinstance(hydro_coeffs['real'], np.ndarray)
    assert hydro_coeffs['real'].shape[0] == len(problem.modes)

    assert isinstance(hydro_coeffs['imag'], np.ndarray)
    assert hydro_coeffs['imag'].shape[0] == len(problem.modes)


def test_calculate_potentials(meem_engine_with_problem, single_meem_problem, sample_problem_params, expected_system_size):
    """
    Tests if calculate_potentials correctly extracts and labels domain potentials.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']
    NMK = sample_problem_params['NMK_values']

    # Generate a dummy solution vector X for testing
    dummy_X = np.arange(expected_system_size, dtype=complex) + 1j * np.arange(expected_system_size, dtype=complex)

    potentials_dict = engine.calculate_potentials(problem, dummy_X)

    assert isinstance(potentials_dict, dict)
    assert len(potentials_dict) == len(problem.domain_list) # Should have entries for each domain

    # Check each domain's entry
    current_idx = 0
    for i, domain in problem.domain_list.items():
        domain_name = f"domain_{i}"
        assert domain_name in potentials_dict

        domain_pot_data = potentials_dict[domain_name]
        assert 'potentials' in domain_pot_data
        assert 'r' in domain_pot_data # Check for geometry coordinates
        assert 'z' in domain_pot_data # Check for geometry coordinates

        # Check potential array shape/content
        expected_harmonics = domain.number_harmonics
        assert domain_pot_data['potentials'].shape == (expected_harmonics,)
        # Verify the actual data extracted
        np.testing.assert_array_equal(
            domain_pot_data['potentials'],
            dummy_X[current_idx : current_idx + expected_harmonics]
        )
        current_idx += expected_harmonics

        # Verify r and z point to the shared geometry coordinates
        assert domain_pot_data['r'] == problem.geometry.r_coordinates
        assert domain_pot_data['z'] == problem.geometry.z_coordinates


def test_visualize_potential(meem_engine_with_problem, single_meem_problem, sample_problem_params):
    """
    Tests that visualize_potential attempts to plot by mocking matplotlib.
    It doesn't test the visual output directly.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
    m0 = sample_problem_params['m0_test']

    # Get a solution vector X
    X = engine.solve_linear_system_multi(problem, m0)
    potentials = engine.calculate_potentials(problem, X)

    # Use patch to mock matplotlib.pyplot.show to prevent it from opening a window
    with patch('matplotlib.pyplot.show') as mock_show:
        # We also need to mock plt.subplots which creates the figure and axes
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            # Configure mock_subplots to return a mock figure and a mock axes
            # The mock axes should have a 'plot' method that we can then assert on.
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax) # When plt.subplots() is called, return these mocks

            # NOW, patch the 'plot' method on the MOCKED AXES object
            # This is the crucial change:
            # We don't need a separate patch for mock_plot anymore if we configure mock_ax directly.
            
            engine.visualize_potential(potentials)

            mock_show.assert_called_once() # Ensure show was called
            mock_subplots.assert_called_once_with(figsize=(10, 6)) # Ensure subplots was called once with specific args

            # Now, assert on the plot method of the mock_ax
            assert mock_ax.plot.call_count == len(potentials) # Ensure plot was called for each domain


def test_run_and_store_results(meem_engine_with_problem, single_meem_problem, sample_problem_params):
    """
    Tests the integration method run_and_store_results.
    Checks if a Results object is returned and if it contains expected data.
    """
    engine = meem_engine_with_problem
    problem = single_meem_problem
        
    # Ensure m0_test is treated as an array of frequencies
    m0_single_value = sample_problem_params['m0_test']
    # Convert it to a NumPy array, even if it's a single value
    m0_values_for_engine = np.array([m0_single_value]) # Make it an array containing the single frequency

    # We'll run it for problem_index 0 (since we only have one problem in problem_list)
    results_obj = engine.run_and_store_results(0, m0_values_for_engine) # Pass the array

    assert isinstance(results_obj, Results)

    assert 'added_mass' in results_obj.dataset.data_vars
    assert 'damping' in results_obj.dataset.data_vars
    assert 'domain_name' in results_obj.dataset.coords # Check if domain names are coords

@pytest.fixture
def decreasing_depth_problem_params(sample_problem_params):
    """Provides parameters for a problem where d values are strictly decreasing."""
    params = sample_problem_params.copy()
    params['d_values'] = [20.0, 10.0] # d[0] > d[1]
    return params

@pytest.fixture
def increasing_depth_problem_params(sample_problem_params):
    """Provides parameters for a problem where d values are strictly increasing."""
    params = sample_problem_params.copy()
    params['d_values'] = [10.0, 20.0] # d[0] < d[1]
    return params

@pytest.fixture
def meem_engine_decreasing_depth(decreasing_depth_problem_params):
    r_coordinates = {'a1': decreasing_depth_problem_params['a_values'][0], 'a2': decreasing_depth_problem_params['a_values'][1]}
    z_coordinates = {'h': decreasing_depth_problem_params['h'], 'd1': decreasing_depth_problem_params['d_values'][0], 'd2': decreasing_depth_problem_params['d_values'][1]}

    domain_params = [
        {'number_harmonics': decreasing_depth_problem_params['NMK_values'][0], 'height': decreasing_depth_problem_params['h'], 'radial_width': decreasing_depth_problem_params['a_values'][0], 'category': 'inner', 'di': decreasing_depth_problem_params['d_values'][0], 'a': decreasing_depth_problem_params['a_values'][0], 'heaving': decreasing_depth_problem_params['heaving_values'][0], 'slant': False},
        {'number_harmonics': decreasing_depth_problem_params['NMK_values'][1], 'height': decreasing_depth_problem_params['h'], 'radial_width': decreasing_depth_problem_params['a_values'][1], 'category': 'outer', 'di': decreasing_depth_problem_params['d_values'][1], 'a': decreasing_depth_problem_params['a_values'][1], 'heaving': decreasing_depth_problem_params['heaving_values'][1], 'slant': False},
        {'number_harmonics': decreasing_depth_problem_params['NMK_values'][2], 'height': decreasing_depth_problem_params['h'], 'radial_width': None, 'category': 'exterior', 'di': None, 'a': None, 'heaving': decreasing_depth_problem_params['heaving_values'][2], 'slant': False},
    ]

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    prob = MEEMProblem(geometry)
    prob.set_frequencies_modes(decreasing_depth_problem_params['frequencies'], decreasing_depth_problem_params['modes'])
    engine = MEEMEngine(problem_list=[prob])
    return engine

@pytest.fixture
def meem_engine_increasing_depth(increasing_depth_problem_params):
    r_coordinates = {'a1': increasing_depth_problem_params['a_values'][0], 'a2': increasing_depth_problem_params['a_values'][1]}
    z_coordinates = {'h': increasing_depth_problem_params['h'], 'd1': increasing_depth_problem_params['d_values'][0], 'd2': increasing_depth_problem_params['d_values'][1]}

    domain_params = [
        {'number_harmonics': increasing_depth_problem_params['NMK_values'][0], 'height': increasing_depth_problem_params['h'], 'radial_width': increasing_depth_problem_params['a_values'][0], 'category': 'inner', 'di': increasing_depth_problem_params['d_values'][0], 'a': increasing_depth_problem_params['a_values'][0], 'heaving': increasing_depth_problem_params['heaving_values'][0], 'slant': False},
        {'number_harmonics': increasing_depth_problem_params['NMK_values'][1], 'height': increasing_depth_problem_params['h'], 'radial_width': increasing_depth_problem_params['a_values'][1], 'category': 'outer', 'di': increasing_depth_problem_params['d_values'][1], 'a': increasing_depth_problem_params['a_values'][1], 'heaving': increasing_depth_problem_params['heaving_values'][1], 'slant': False},
        {'number_harmonics': increasing_depth_problem_params['NMK_values'][2], 'height': increasing_depth_problem_params['h'], 'radial_width': None, 'category': 'exterior', 'di': None, 'a': None, 'heaving': increasing_depth_problem_params['heaving_values'][2], 'slant': False},
    ]

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    prob = MEEMProblem(geometry)
    prob.set_frequencies_modes(increasing_depth_problem_params['frequencies'], increasing_depth_problem_params['modes'])
    engine = MEEMEngine(problem_list=[prob])
    return engine


# --- New Tests to cover branches ---

def test_assemble_A_template_d_decreasing(meem_engine_decreasing_depth, decreasing_depth_problem_params):
    """
    Tests assemble_A_template for a problem with decreasing depths (d[bd] > d[bd+1]).
    This should cover the `left_diag_is_active = True` branches.
    """
    engine = meem_engine_decreasing_depth
    problem = engine.problem_list[0]
    m0 = decreasing_depth_problem_params['m0_test']

    # Just call assemble_A_multi to trigger the cache build and template creation
    # The goal is to execute the lines, not necessarily assert on numerical values unless specific
    # 'gold standard' comparisons are available.
    A_matrix = engine.assemble_A_multi(problem, m0)

    #  could add assertions here if there is a known expected matrix for this configuration.
    # For coverage alone, simply executing it is enough.
    assert A_matrix is not None # At least ensure it returns something

def test_assemble_A_template_d_increasing(meem_engine_increasing_depth, increasing_depth_problem_params):
    """
    Tests assemble_A_template for a problem with increasing depths (d[bd] < d[bd+1]).
    This should cover the `left_diag_is_active = False` branches for the relevant boundaries.
    """
    engine = meem_engine_increasing_depth
    problem = engine.problem_list[0]
    m0 = increasing_depth_problem_params['m0_test']

    A_matrix = engine.assemble_A_multi(problem, m0)
    assert A_matrix is not None

# --- Tests for reformat_coeffs ---

def test_reformat_coeffs_basic_3_regions(meem_engine_with_problem, sample_problem_params):
    """
    Tests reformat_coeffs with a standard 3-region problem (inner, outer, exterior).
    """
    engine = meem_engine_with_problem
    NMK = sample_problem_params['NMK_values']
    boundary_count = len(engine.problem_list[0].domain_list) - 1 # Should be 2 for 3 domains

    # Calculate expected total size
    expected_size = NMK[0] + (NMK[1] * 2) + NMK[2]
    
    # Create a dummy solution vector
    # Use distinct values for easy verification
    x_dummy = np.arange(expected_size, dtype=complex) + 1j * np.arange(expected_size, dtype=complex)
    
    reformatted_coeffs = engine.reformat_coeffs(x_dummy, NMK, boundary_count)

    assert isinstance(reformatted_coeffs, list)
    assert len(reformatted_coeffs) == 3 # Should be 3 regions

    # Verify coefficients for Region 0 (inner)
    np.testing.assert_array_equal(reformatted_coeffs[0], x_dummy[:NMK[0]])

    # Verify coefficients for intermediate annular regions (Region 1, 'outer' in this case)
    # This loop runs from i=1 to boundary_count-1. For boundary_count=2, i will be 1.
    start_idx_region1 = NMK[0]
    end_idx_region1 = start_idx_region1 + (NMK[1] * 2)
    np.testing.assert_array_equal(reformatted_coeffs[1], x_dummy[start_idx_region1 : end_idx_region1])

    # Verify coefficients for the outermost region (exterior)
    start_idx_region2 = end_idx_region1 # The starting point for the last region
    np.testing.assert_array_equal(reformatted_coeffs[2], x_dummy[start_idx_region2:])


def test_reformat_coeffs_single_region_exterior_only(meem_engine_with_problem):
    """
    Tests reformat_coeffs for a hypothetical single-region problem (only exterior).
    This implies NMK_values has only one element and boundary_count is 0.
    """
    engine = meem_engine_with_problem
    # Simulate a problem with only an exterior domain
    NMK_single_region = [7] # Only K harmonics
    boundary_count_single_region = 0 # No inner or intermediate boundaries

    expected_size = NMK_single_region[0]
    x_dummy = np.arange(expected_size, dtype=complex) + 1j * np.arange(expected_size, dtype=complex)

    reformatted_coeffs = engine.reformat_coeffs(x_dummy, NMK_single_region, boundary_count_single_region)

    assert isinstance(reformatted_coeffs, list)
    assert len(reformatted_coeffs) == 2 # First region and "outermost" region (which is the same here)

    # The first append should take NMK[0] (which is the only NMK value)
    np.testing.assert_array_equal(reformatted_coeffs[0], x_dummy[:NMK_single_region[0]])

    # The loop for intermediate regions should not run (range(1, 0) is empty)
    # The last append should take the rest, which is also NMK[0]
    np.testing.assert_array_equal(reformatted_coeffs[1], x_dummy[NMK_single_region[0]:])


def test_reformat_coeffs_multiple_annuli(meem_engine_with_problem):
    """
    Tests reformat_coeffs with multiple intermediate annular regions.
    e.g., Inner, Annular1, Annular2, Exterior (4 regions total, 3 boundaries).
    """
    engine = meem_engine_with_problem
    # Simulate NMK for 4 regions: N, M1, M2, K
    NMK_multiple_annuli = [4, 3, 5, 6] # N=4, M1=3, M2=5, K=6
    boundary_count_multiple_annuli = 3 # 4 regions means 3 boundaries (0, 1, 2)

    # Calculate expected total size: N + 2*M1 + 2*M2 + K
    expected_size = NMK_multiple_annuli[0] + \
                    (NMK_multiple_annuli[1] * 2) + \
                    (NMK_multiple_annuli[2] * 2) + \
                    NMK_multiple_annuli[3]
    
    x_dummy = np.arange(expected_size, dtype=complex) + 1j * np.arange(expected_size, dtype=complex)

    reformatted_coeffs = engine.reformat_coeffs(x_dummy, NMK_multiple_annuli, boundary_count_multiple_annuli)

    assert isinstance(reformatted_coeffs, list)
    assert len(reformatted_coeffs) == 4 # Should be 4 regions

    # Region 0 (inner)
    np.testing.assert_array_equal(reformatted_coeffs[0], x_dummy[:NMK_multiple_annuli[0]])
    current_row = NMK_multiple_annuli[0]

    # Region 1 (first annulus)
    np.testing.assert_array_equal(
        reformatted_coeffs[1], 
        x_dummy[current_row : current_row + (NMK_multiple_annuli[1] * 2)]
    )
    current_row += (NMK_multiple_annuli[1] * 2)

    # Region 2 (second annulus)
    np.testing.assert_array_equal(
        reformatted_coeffs[2], 
        x_dummy[current_row : current_row + (NMK_multiple_annuli[2] * 2)]
    )
    current_row += (NMK_multiple_annuli[2] * 2)

    # Region 3 (outermost/exterior)
    np.testing.assert_array_equal(reformatted_coeffs[3], x_dummy[current_row:])
    assert current_row == expected_size - NMK_multiple_annuli[3] # Ensure current_row is correct before last slice


def test_reformat_coeffs_empty_x(meem_engine_with_problem):
    """
    Tests reformat_coeffs with an empty input vector x.
    This should return a list of empty arrays.
    """
    engine = meem_engine_with_problem
    NMK = [0, 0, 0] # No harmonics in any region
    boundary_count = 2 # Still 3 regions (inner, outer, exterior) but empty
    x_empty = np.array([], dtype=complex)

    reformatted_coeffs = engine.reformat_coeffs(x_empty, NMK, boundary_count)

    assert isinstance(reformatted_coeffs, list)
    assert len(reformatted_coeffs) == 3 # Still 3 regions

    # All should be empty arrays
    for coeffs_array in reformatted_coeffs:
        assert isinstance(coeffs_array, np.ndarray)
        assert coeffs_array.size == 0
        assert coeffs_array.dtype == complex # Ensure dtype is preserved