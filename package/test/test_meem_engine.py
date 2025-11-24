import pytest
import numpy as np
import sys
import os
import xarray as xr

# --- Path Setup ---
# This ensures pytest can find package source files
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import Package Modules ---
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.problem_cache import ProblemCache
from openflash.results import Results
from openflash.multi_equations import omega
from openflash.multi_constants import g
from openflash.body import SteppedBody
from openflash.geometry import ConcentricBodyGroup
from openflash.basic_region_geometry import BasicRegionGeometry

# ==============================================================================
# Pytest Fixture: Reusable Test Problem
# ==============================================================================
@pytest.fixture(scope="module")
def sample_problem():
    """
    Creates a standard, reusable MEEMProblem instance for all tests in this module.
    `scope="module"` means this function runs only once per test session.
    """
    # Define a simple but complete 2-cylinder problem
    NMK = [10, 10, 10]
    h = 100.0
    a = np.array([5.0, 10.0])
    d = np.array([20.0, 10.0])
    
    # FIX: Change heaving to only have one body heaving to pass the new assertion
    heaving = np.array([1, 0]) # [True, False] -> Only the first body heaves
    
    # 1. Define the physical bodies
    bodies = []
    for i in range(len(a)):
        body = SteppedBody(
            a=np.array([a[i]]),
            d=np.array([d[i]]),
            slant_angle=np.array([0.0]), # Assuming zero slant for test
            heaving=bool(heaving[i])
        )
        bodies.append(body)

    # 2. Create the body arrangement
    # This call now checks the assertion (heaving_count <= 1)
    arrangement = ConcentricBodyGroup(bodies)

    # 3. Instantiate the concrete geometry class
    geometry = BasicRegionGeometry(arrangement, h, NMK)
    
    # 4. Create the problem
    problem = MEEMProblem(geometry)
    
    # --- Set frequencies and modes for the problem ---
    m0 = 1.0 
    local_omega = omega(m0, h, g)
    problem_frequencies = np.array([local_omega])
    
    # Modes correspond to heaving bodies. Only body 0 is heaving.
    # The `problem.modes` property will now correctly return array([0])
    problem.set_frequencies(problem_frequencies)
    
    return problem

# ==============================================================================
# Test Suite for MEEMEngine
# ==============================================================================

def test_engine_initialization(sample_problem):
    """
    Tests if the MEEMEngine initializes correctly and creates a problem cache.
    """
    engine = MEEMEngine(problem_list=[sample_problem])
    
    assert len(engine.problem_list) == 1
    assert sample_problem in engine.cache_list
    assert engine.cache_list[sample_problem] is not None
    print("✅ Engine initialization test passed.")

def test_matrix_assembly(sample_problem):
    """
    Tests the assembly of the A matrix and b vector.
    """
    engine = MEEMEngine(problem_list=[sample_problem])
    m0 = 1.0
    
    # Ensure cache is populated for m0-dependent parts
    engine._ensure_m_k_and_N_k_arrays(sample_problem, m0)
    
    A = engine.assemble_A_multi(sample_problem, m0)
    b = engine.assemble_b_multi(sample_problem, m0)
    
    expected_size = 10 + 2 * 10 + 10 # NMK[0] + 2*NMK[1] + NMK[2]
    
    assert isinstance(A, np.ndarray)
    assert A.shape == (expected_size, expected_size)
    assert np.iscomplexobj(A)
    
    assert isinstance(b, np.ndarray)
    assert b.shape == (expected_size,)
    assert np.iscomplexobj(b)
    print("✅ Matrix assembly test passed.")

def test_solve_linear_system(sample_problem):
    """
    Tests if the linear system solver runs and returns the correct shape.
    """
    engine = MEEMEngine(problem_list=[sample_problem])
    m0 = 1.0
    
    X = engine.solve_linear_system_multi(sample_problem, m0)
    
    expected_size = 10 + 2 * 10 + 10
    
    assert isinstance(X, np.ndarray)
    assert X.shape == (expected_size,)
    assert np.iscomplexobj(X)
    print("✅ Linear system solver test passed.")
    
def test_compute_hydrodynamic_coefficients(sample_problem):
    """
    Tests the calculation of hydrodynamic coefficients.
    """
    engine = MEEMEngine(problem_list=[sample_problem])
    m0 = 1.0
    X = engine.solve_linear_system_multi(sample_problem, m0)
    
    coeffs = engine.compute_hydrodynamic_coefficients(problem=sample_problem, X=X, m0=m0)
    
    assert isinstance(coeffs, list), "Expected list of dictionaries"
    # NOTE: compute_hydrodynamic_coefficients iterates over all *possible* modes/bodies,
    # which is 2 bodies in this fixture.
    assert len(coeffs) == 2, "Expected coefficients for 2 bodies"
    
    for c in coeffs:
        assert isinstance(c, dict), "Each entry in the result should be a dictionary"
        assert "real" in c, "Missing 'real' in coefficient dictionary"
        assert "imag" in c, "Missing 'imag' in coefficient dictionary"
        assert "nondim_real" in c, "Missing 'nondim_real'"
        assert "nondim_imag" in c, "Missing 'nondim_imag'"
        assert "excitation_phase" in c, "Missing 'excitation_phase'"
        assert "excitation_force" in c, "Missing 'excitation_force'"
    print("✅ Hydrodynamic coefficients test passed.")

def test_calculate_potentials_and_velocities(sample_problem):
    """
    Tests that potential and velocity calculations run and return correct data structures.
    """
    engine = MEEMEngine(problem_list=[sample_problem])
    m0 = 1.0
    X = engine.solve_linear_system_multi(sample_problem, m0)
    
    # Test potentials
    potentials = engine.calculate_potentials(sample_problem, X, m0, spatial_res=10, sharp=False)
    assert isinstance(potentials, dict)
    assert "phi" in potentials and potentials["phi"].shape == (10, 10)
    assert "R" in potentials and "Z" in potentials
    
    # Test velocities
    velocities = engine.calculate_velocities(sample_problem, X, m0, spatial_res=10, sharp=False)
    assert isinstance(velocities, dict)
    assert "vr" in velocities and velocities["vr"].shape == (10, 10)
    assert "vz" in velocities and velocities["vz"].shape == (10, 10)
    print("✅ Potential and velocity calculation tests passed.")
    
def test_ensure_m_k_and_N_k_arrays(sample_problem):
    """
    Tests that _ensure_m_k_and_N_k_arrays correctly populates the cache
    and is idempotent (does not re-calculate if values already exist).
    """
    engine = MEEMEngine(problem_list=[sample_problem])
    m0 = 1.0
    cache = engine.cache_list[sample_problem]

    # 1. Assert initial state is empty
    assert cache.m_k_arr is None
    assert cache.N_k_arr is None

    # 2. Act: Call the method for the first time
    engine._ensure_m_k_and_N_k_arrays(sample_problem, m0)

    # 3. Assert that cache is now populated
    assert isinstance(cache.m_k_arr, np.ndarray)
    assert isinstance(cache.N_k_arr, np.ndarray)
    
    # Check shape based on the fixture's NMK = [10, 10, 10]
    expected_len = 10 
    assert cache.m_k_arr.shape == (expected_len,)
    assert cache.N_k_arr.shape == (expected_len,)
    
    # Store the object IDs of the created arrays
    id_m_k_before = id(cache.m_k_arr)
    id_N_k_before = id(cache.N_k_arr)

    # 4. Act: Call the method a second time
    engine._ensure_m_k_and_N_k_arrays(sample_problem, m0)
    
    # 5. Assert that the arrays were not re-calculated (idempotency check)
    assert id(cache.m_k_arr) == id_m_k_before
    assert id(cache.N_k_arr) == id_N_k_before
    
    print("✅ Cache population and idempotency test passed.")
    
def test_build_problem_cache(sample_problem):
    """
    Tests that the build_problem_cache method correctly populates the cache
    with templates and m0-dependent calculation functions.
    """
    engine = MEEMEngine(problem_list=[sample_problem])
    cache = engine.cache_list[sample_problem]

    # 1. Check that the cache object was created and populated
    assert isinstance(cache, ProblemCache)
    assert cache.A_template is not None
    assert cache.b_template is not None
    
    # 2. Verify the shapes of the templates
    NMK = [10, 10, 10]
    expected_size = NMK[0] + 2 * NMK[1] + NMK[2]
    assert cache.A_template.shape == (expected_size, expected_size)
    assert cache.b_template.shape == (expected_size,)

    # 3. Verify that m0-independent parts have been pre-computed
    # The A_template should not be all zeros; some blocks are m0-independent.
    assert np.any(cache.A_template != 0)
    # The b_template should also have some pre-computed values.
    assert np.any(cache.b_template != 0)

    # 4. Verify that the lists for m0-dependent parts are populated
    # For a 2-cylinder problem (3 domains), there are m0-dependent blocks.
    assert len(cache.m0_dependent_A_indices) > 0
    assert len(cache.m0_dependent_b_indices) > 0

    # 5. Check a specific m0-dependent entry to ensure it's a callable
    # The third element of the tuple should be the calculation function.
    assert callable(cache.m0_dependent_A_indices[0][2])
    assert callable(cache.m0_dependent_b_indices[0][1])
    
    print("✅ Problem cache build test passed.")
    
def test_reformat_coeffs():
    """
    Tests the reformat_coeffs method to ensure it correctly splits the
    solution vector `x` into arrays for each physical region.
    """
    # 1. Arrange: Set up a mock problem
    # We only need an engine instance to call the method
    engine = MEEMEngine(problem_list=[]) 
    NMK = [3, 4, 5]  # Inner (3), Intermediate (4), Exterior (5)
    boundary_count = len(NMK) - 1
    
    # Calculate the total size of the mock solution vector
    # Inner region has NMK[0] coeffs
    # Intermediate region has 2 * NMK[1] coeffs
    # Exterior region has NMK[2] coeffs
    size = NMK[0] + 2 * NMK[1] + NMK[2]  # 3 + 2*4 + 5 = 16
    x = np.arange(size) # Create a predictable vector: [0, 1, ..., 15]

    # 2. Act: Call the function to be tested
    reformatted_cs = engine.reformat_coeffs(x, NMK, boundary_count)

    # 3. Assert: Check the results
    # Check that the output is a list with the correct number of regions
    assert isinstance(reformatted_cs, list)
    assert len(reformatted_cs) == len(NMK)

    # Check the shape of each region's coefficient array
    assert reformatted_cs[0].shape == (NMK[0],)      # Inner region
    assert reformatted_cs[1].shape == (2 * NMK[1],)  # Intermediate region
    assert reformatted_cs[2].shape == (NMK[2],)      # Exterior region

    # Check the content of each array to ensure the slicing was correct
    np.testing.assert_array_equal(reformatted_cs[0], np.arange(0, 3))
    np.testing.assert_array_equal(reformatted_cs[1], np.arange(3, 3 + 8))
    np.testing.assert_array_equal(reformatted_cs[2], np.arange(11, 16))
    
    print("✅ Coefficient reformatting test passed.")
    
# test_meem_engine.py

def test_run_and_store_results(sample_problem):
    """
    Tests the full computation loop over a set of frequencies, ensuring
    results are correctly stored in a Results object.
    """
    # 1. Arrange: Define a set of test frequencies
    # Using omega function to get valid frequencies based on m0 values
    test_m0s = [0.5, 1.0, 1.5]
    test_frequencies = np.array([omega(m0, sample_problem.geometry.h, g) for m0 in test_m0s])
    
    # FIX: Use the new set_frequencies method
    sample_problem.set_frequencies(test_frequencies)

    # Infer modes from the sample_problem fixture's geometry
    # The fixture is now constrained to 1 heaving body (mode 0)
    num_modes = len(sample_problem.modes)
    num_freqs = len(test_frequencies)
    
    # Check that only one mode is active
    assert num_modes == 1 

    engine = MEEMEngine(problem_list=[sample_problem])

    # 2. Act: Run the main computation method
    results = engine.run_and_store_results(problem_index=0)

    # 3. Assert: Check the structure of the Results object
    assert isinstance(results, Results)
    assert len(results.frequencies) == num_freqs
    assert len(results.modes) == num_modes
    assert np.array_equal(results.modes, sample_problem.modes)

    # Check the shape of the stored hydrodynamic coefficients
    # Should be (num_freqs, 2 total bodies, 1 active mode) 
    ds = results.get_results()
    
    # NOTE: The size of the hydrodynamic matrix is (Total Bodies x Active Modes)
    # The compute function iterates over all bodies (2) for force (j) and active modes (1) for motion (i).
    expected_shape = (num_freqs, 2, num_modes) 
    
    assert ds['added_mass'].shape == expected_shape
    assert ds['damping'].shape == expected_shape

    # Check that the data is not all NaN (i.e., computation ran)
    assert not np.isnan(ds['added_mass'].values).all()
    assert not np.isnan(ds['damping'].values).all()