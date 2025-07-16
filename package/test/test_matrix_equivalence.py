import pytest
import numpy as np

# Import your classes and functions
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry
from openflash.problem_cache import ProblemCache
from openflash.multi_constants import g, rho
from openflash.multi_equations import (
    omega, scale, lambda_ni, m_k_entry, m_k, N_k_multi, N_k_og,
    I_nm, I_mk, I_mk_og,
    b_potential_entry, b_potential_end_entry, b_velocity_entry, b_velocity_end_entry, b_velocity_end_entry_og,
    R_1n, diff_R_1n, R_2n, diff_R_2n, Lambda_k, diff_Lambda_k, Lambda_k_og, diff_Lambda_k_og
)


# Define a fixture for a common problem setup
@pytest.fixture
def sample_problem():
    r_coords_for_geometry = {'a1': 0.5, 'a2': 1.0}
    z_coordinates = {'h': 1.001} # Note: h in multi_equations.py expects float, not a dict here.
                                # Make sure geometry.domain_params[0]['height'] is used for h in calculations.
    domain_params = [
        {'number_harmonics': 5, 'height': 1.0, 'radial_width': 0.5, 'category': 'inner', 'di': 0.5, 'a': 0.5, 'heaving': 1.0},
        {'number_harmonics': 8, 'height': 1.0, 'radial_width': 1.0, 'category': 'outer', 'di': 0.25, 'a': 1.0, 'heaving': 0.0}
    ]

    geometry = Geometry(r_coords_for_geometry, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)

    # Set frequencies and modes, crucial for problem setup even if not directly used in A/b assembly
    boundary_count = len(domain_params) - 1
    # Example frequency and modes for the problem object (can be adjusted)
    problem_frequencies = np.array([omega(0.1, domain_params[0]['height'], g), omega(1.0, domain_params[0]['height'], g)])
    problem_modes = np.arange(boundary_count + 1) # Example modes: 0, 1
    problem.set_frequencies_modes(problem_frequencies, problem_modes)

    return problem

# Test for A matrix equivalence
def test_assemble_A_equivalence(sample_problem):
    problem = sample_problem
    engine = MEEMEngine(problem_list=[problem]) # Initialize engine to build cache

    # Choose a specific m0 for comparison
    # This m0 should correspond to one of the problem_frequencies or be a standalone value for testing
    test_m0 = 0.5 # Example m0 value

    # Calculate A matrix using the ORIGINAL (full re-calculation) method
    A_original = engine._full_assemble_A_multi(problem, test_m0)

    # Calculate A matrix using the NEW (optimized, cache-based) method
    A_new = engine.assemble_A_multi(problem, test_m0)

    print("\n--- A Matrix Comparison ---")
    print("A_original shape:", A_original.shape)
    print("A_new shape:", A_new.shape)
    # print("A_original:\n", A_original)
    # print("A_new:\n", A_new)

    # Assert that the shapes are the same
    assert A_original.shape == A_new.shape, "A matrix shapes do not match!"

    # Assert that the matrices are element-wise close
    # Use a small absolute tolerance (atol) and relative tolerance (rtol)
    # Floating point comparisons require tolerance
    tolerance_rtol = 1e-9 # Relative tolerance
    tolerance_atol = 1e-12 # Absolute tolerance (for values close to zero)

    assert np.allclose(A_original, A_new, rtol=tolerance_rtol, atol=tolerance_atol), \
        f"A matrices differ significantly.\nMax absolute difference: {np.max(np.abs(A_original - A_new))}\nMax relative difference: {np.max(np.abs((A_original - A_new) / A_original[A_original != 0]))}"
    print("A matrices are equivalent within specified tolerances.")


# Test for b vector equivalence
def test_assemble_b_equivalence(sample_problem):
    problem = sample_problem
    engine = MEEMEngine(problem_list=[problem]) # Initialize engine to build cache

    # Choose a specific m0 for comparison (consistent with A matrix test)
    test_m0 = 0.5

    # Calculate b vector using the ORIGINAL (full re-calculation) method
    b_original = engine._full_assemble_b_multi(problem, test_m0)

    # Calculate b vector using the NEW (optimized, cache-based) method
    b_new = engine.assemble_b_multi(problem, test_m0)

    print("\n--- b Vector Comparison ---")
    print("b_original shape:", b_original.shape)
    print("b_new shape:", b_new.shape)
    # print("b_original:\n", b_original)
    # print("b_new:\n", b_new)

    # Assert that the shapes are the same
    assert b_original.shape == b_new.shape, "b vector shapes do not match!"

    # Assert that the vectors are element-wise close
    tolerance_rtol = 1e-9
    tolerance_atol = 1e-12

    assert np.allclose(b_original, b_new, rtol=tolerance_rtol, atol=tolerance_atol), \
        f"b vectors differ significantly.\nMax absolute difference: {np.max(np.abs(b_original - b_new))}\nMax relative difference: {np.max(np.abs((b_original - b_new) / b_original[b_original != 0]))}"
    print("b vectors are equivalent within specified tolerances.")