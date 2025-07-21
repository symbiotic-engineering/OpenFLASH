import pytest
import numpy as np
# Assuming these are the correct imports based on your file structure
from openflash.multi_equations import * 
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.domain import Domain # <-- IMPORT DOMAIN CLASS HERE
from openflash.geometry import Geometry
from typing import Dict, Any

# Adjust these tolerances as needed
ATOL = 1e-9
RTOL = 1e-7

# --- Helper function to run the original calculation logic ---
# (This part is unchanged from your provided code)
def calculate_potentials_original_logic(problem: MEEMProblem, solution_vector: np.ndarray, m0: float, m_k_arr, N_k_arr) -> Dict[str, Any]:
    """
    Replicates the potential calculation from the original, un-refactored code.
    This is the ground truth for our tests.
    """
    domain_list = problem.domain_list
    domain_keys = list(domain_list.keys())
    boundary_count = len(domain_keys) - 1

    NMK = [domain_list[idx].number_harmonics for idx in domain_keys]
    h = domain_list[0].h
    d = [domain_list[idx].di for idx in domain_keys]
    a = [domain_list[idx].a for idx in domain_keys if domain_list[idx].a is not None]
    heaving = [domain_list[idx].heaving for idx in domain_keys]
    spatial_res = 50 
    sharp = True 

    # Original reformat_coeffs logic embedded here for clarity and independence
    Cs = []
    row = 0
    Cs.append(solution_vector[:NMK[0]])
    row += NMK[0]
    for i in range(1, boundary_count):
        Cs.append(solution_vector[row: row + NMK[i] * 2])
        row += NMK[i] * 2
    Cs.append(solution_vector[row:])

    def phi_h_n_inner_func(n, r, z):
        return (Cs[0][n] * R_1n(n, r, 0, h, d, a)) * Z_n_i(n, z, 0, h, d)

    def phi_h_m_i_func(i, m, r, z):
        return (Cs[i][m] * R_1n(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n(m, r, i, a, h, d)) * Z_n_i(m, z, i, h, d)

    def phi_e_k_func(k, r, z, m_k_arr, N_k_arr):
        return Cs[-1][k] * Lambda_k(k, r, m0, a, NMK, h, m_k_arr, N_k_arr) * Z_k_e(k, z, m0, h, NMK, m_k_arr)

    phi_h_n_inner_vec = np.vectorize(phi_h_n_inner_func, otypes=[complex])
    phi_h_m_i_vec = np.vectorize(phi_h_m_i_func, otypes=[complex])
    phi_e_k_vec = np.vectorize(
        lambda k, r, z: phi_e_k_func(k, r, z, m_k_arr, N_k_arr)
    )
    phi_p_i_vec = np.vectorize(phi_p_i, otypes=[complex])


    R, Z = make_R_Z(a, h, d, sharp, spatial_res)
    
    regions = []
    regions.append((R <= a[0]) & (Z < -d[0]))
    for i in range(1, boundary_count):
        regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
    regions.append(R > a[-1])

    shape = R.shape
    phiH = np.full(shape, np.nan + 1j*np.nan, dtype=complex)
    phiP = np.full(shape, np.nan + 1j*np.nan, dtype=complex)

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

    phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]], h)

    for i in range(1, boundary_count):
        phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]], h)
    phiP[regions[-1]] = 0.0

    phi = phiH + phiP

    return {
        "R": R,
        "Z": Z,
        "phiH": phiH,
        "phiP": phiP,
        "phi": phi
    }

# --- CORRECTED FIXTURE FOR TEST SETUP ---
@pytest.fixture
def sample_problem():
    # 1. Define the parameters for the mock Geometry object
    r_coords = {'a1': 2.0, 'a2': 5.0}
    z_coords = {'h': 10.0}
    
    # Corrected structure to match Geometry.make_domain_list() expectations
    domain_params = [
        {'number_harmonics': 3, 'height': 10.0, 'radial_width': 2.0, 'top_BC': None, 'bottom_BC': None, 'category': 'inner', 'di': 5.0, 'a': 2.0, 'heaving': 1, 'index': 0},
        {'number_harmonics': 4, 'height': 10.0, 'radial_width': 3.0, 'top_BC': None, 'bottom_BC': None, 'category': 'outer', 'di': 8.0, 'a': 5.0, 'heaving': 0, 'index': 1},
        {'number_harmonics': 5, 'height': 10.0, 'radial_width': None, 'top_BC': None, 'bottom_BC': None, 'category': 'exterior', 'di': 0.0, 'a': None, 'heaving': 1, 'index': 2},
    ]

    # 2. Create the Geometry object first. This will internally create the domain_list.
    mock_geometry = Geometry(r_coordinates=r_coords, z_coordinates=z_coords, domain_params=domain_params)

    # 3. Create the MEEMProblem instance using the mock_geometry.
    problem = MEEMProblem(geometry=mock_geometry)
    
    # 4. Set frequencies and modes, as required by the test.
    problem.set_frequencies_modes(np.array([0.5, 1.0, 2.5]), np.array([0]))
    
    return problem

# --- CORRECTED PYTEST TEST CASE ---
@pytest.mark.parametrize("m0", [0.5])
def test_calculate_potentials_equivalence(sample_problem, m0):
    """
    Tests if the new calculate_potentials method produces the same results as
    the original, un-refactored logic.
    """
    problem = sample_problem
    
    # 1. Create an instance of the engine
    engine = MEEMEngine([problem])
    
    # 2. Get the necessary data (A, b, X, m_k_arr, N_k_arr)
    A = engine._full_assemble_A_multi(problem, m0)
    b = engine._full_assemble_b_multi(problem, m0)

    X = np.linalg.solve(A, b)

    # 3. Build the cache and get m_k_arr and N_k_arr from it.
    engine._build_problem_cache(problem)
    cache = engine.cache_list[problem]
    
    engine._ensure_m_k_and_N_k_arrays(problem, m0)
    m_k_arr = cache.m_k_arr
    N_k_arr = cache.N_k_arr

    # 4. Calculate potentials using the ORIGINAL logic
    potentials_original = calculate_potentials_original_logic(problem, X, m0, m_k_arr, N_k_arr)

    # 5. Calculate potentials using the NEW (refactored) method
    potentials_new = engine.calculate_potentials(
        problem, X, m0, m_k_arr, N_k_arr, spatial_res=50, sharp=True
    )

    # 6. Compare the results
    print(f"\n--- Potentials Comparison for m0 = {m0} ---")
    
    assert potentials_original.keys() == potentials_new.keys(), "Potential dictionary keys do not match!"

    for key in potentials_original.keys():
        print(f"Comparing array '{key}'...")
        nan_original = np.isnan(potentials_original[key])
        nan_new = np.isnan(potentials_new[key])
        assert np.all(nan_original == nan_new), f"NaN locations differ in array '{key}' for m0={m0}."
        
        non_nan_mask = ~nan_original
        np.testing.assert_allclose(
            potentials_new[key][non_nan_mask], potentials_original[key][non_nan_mask],
            rtol=RTOL, atol=ATOL,
            err_msg=f"Array '{key}' differs significantly for m0={m0}."
        )
        print(f"Array '{key}' matches.")

    print(f"All potentials for m0={m0} match the original logic.")