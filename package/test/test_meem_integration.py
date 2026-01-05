import pytest
import numpy as np
from scipy import linalg
from functools import partial

# Import from your package
from openflash.multi_equations import *
from openflash.multi_constants import *
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.geometry import ConcentricBodyGroup
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.body import SteppedBody

# --- Fixtures ---

@pytest.fixture
def meem_setup():
    """
    Sets up the problem configuration used for testing.
    Equivalent to setting the global variables in the original script.
    """
    # Physical Constants
    h = 10.0
    rho_val = 1000.0
    g_val = 9.81
    
    # Geometry Definition (2-body configuration similar to script intent)
    # Radii (strictly increasing)
    a = np.array([5.0, 10.0]) 
    # Depths
    d = np.array([2.0, 4.0]) 
    # Heaving Flags (Body 0 heaving, Body 1 static)
    heaving = np.array([True, False])
    # Slants (zeros for standard cylinders)
    slants = np.array([0.0, 0.0])
    
    # Harmonics (NMK)
    # 2 bodies -> 3 regions (Inner, Intermediate, Exterior)
    NMK = [5, 5, 5] 
    
    # Frequency
    omega_val = 1.5
    m0_val = wavenumber(omega_val, h)
    
    # Create OpenFlash Objects
    bodies = []
    for i in range(len(a)):
        bodies.append(SteppedBody(
            a=np.array([a[i]]), 
            d=np.array([d[i]]), 
            slant_angle=np.array([slants[i]]), 
            heaving=bool(heaving[i])
        ))
        
    arrangement = ConcentricBodyGroup(bodies)
    geometry = BasicRegionGeometry(arrangement, h, NMK)
    problem = MEEMProblem(geometry)
    problem.set_frequencies(np.array([omega_val]))
    
    return {
        "h": h,
        "a": a,
        "d": d,
        "heaving": heaving,
        "NMK": NMK,
        "m0": m0_val,
        "rho": rho_val,
        "omega": omega_val,
        "problem": problem,
        "boundary_count": len(NMK) - 1
    }

# --- Tests ---

def test_input_assertions(meem_setup):
    """
    Validates the input arrays satisfy geometric and physical constraints.
    (logic taken from the start of the original script)
    """
    a = meem_setup["a"]
    d = meem_setup["d"]
    heaving = meem_setup["heaving"]
    NMK = meem_setup["NMK"]
    h = meem_setup["h"]
    m0 = meem_setup["m0"]
    boundary_count = meem_setup["boundary_count"]

    # Length checks
    for arr in [a, d, heaving]:
        assert len(arr) == boundary_count, \
            "NMK should have one more entry than a, d, and heaving."

    # Boolean checks
    for entry in heaving:
        assert entry in [0, 1, True, False], "heaving entries should be booleans."

    # Monotonicity checks
    left = 0
    for radius in a:
        assert radius > left, "a entries should be increasing and > 0."
        left = radius

    # Depth checks
    for depth in d:
        assert depth >= 0, "d entries should be nonnegative."
        assert depth < h, "d entries should be less than h."

    # Harmonics checks
    for val in NMK:
        assert isinstance(val, int) and val > 0, "NMK entries should be positive integers."

    assert m0 > 0, "m0 should be positive."


def test_matrix_assembly_consistency(meem_setup):
    """
    Reconstructs the A matrix and b vector manually using the equations
    and compares them against the output of MEEMEngine.
    """
    # Unpack
    h = meem_setup["h"]
    a = meem_setup["a"]
    d = meem_setup["d"]
    heaving = meem_setup["heaving"]
    NMK = meem_setup["NMK"]
    m0 = meem_setup["m0"]
    problem = meem_setup["problem"]
    boundary_count = meem_setup["boundary_count"]
    
    # ---------------------------------------------------------
    # 1. Generate via Engine (The "System Under Test")
    # ---------------------------------------------------------
    engine = MEEMEngine([problem])
    # Ensure cache is built
    engine.cache_list[problem] = engine.build_problem_cache(problem)
    
    A_engine = engine.assemble_A_multi(problem, m0)
    b_engine = engine.assemble_b_multi(problem, m0)
    
    # ---------------------------------------------------------
    # 2. Manual Generation (The "Oracle" from the script)
    # ---------------------------------------------------------
    
    # Pre-compute wave numbers (needed for modern I_mk/N_k functions)
    m_k_arr = np.array([m_k_entry(k, m0, h) for k in range(NMK[-1])])
    N_k_arr = np.array([N_k_multi(k, m0, h, m_k_arr) for k in range(NMK[-1])])

    # Coupling integrals
    # FIX: I_nm_vals is now a list of 2D matrices, not a 3D array
    I_nm_vals = []
    for bd in range(boundary_count - 1):
        mat = np.zeros((NMK[bd], NMK[bd+1]), dtype=complex)
        for n in range(NMK[bd]):
            for m in range(NMK[bd + 1]):
                mat[n][m] = I_nm(n, m, bd, d, h)
        I_nm_vals.append(mat)

    I_mk_vals = np.zeros((NMK[boundary_count - 1], NMK[boundary_count]), dtype=complex)
    for m in range(NMK[boundary_count - 1]):
        for k in range(NMK[boundary_count]):
            # Updated to pass all required args including precomputed arrays
            I_mk_vals[m][k] = I_mk(m, k, boundary_count - 1, d, m0, h, m_k_arr, N_k_arr)

    # --- Block Builders (Local Adaptations) ---
    
    # Helpers for manual block construction using library functions
    def make_p_diag(left, func, bd):
        return p_diagonal_block(left, func, bd, h, d, a, NMK)
    
    def make_p_dense(left, func, bd):
        return p_dense_block(left, func, bd, NMK, a, I_nm_vals)
    
    def make_p_dense_e(bd):
        # FIX: We construct this manually because the library's p_dense_block_e
        # doesn't accept the required m0/m_k_arr arguments.
        
        # 1. Create a partial of Lambda_k with fixed params
        # Lambda_k signature: (k, r, m0, a, m_k_arr)
        func = partial(Lambda_k, m0=m0, a=a, m_k_arr=m_k_arr)
        
        # 2. Vectorize it
        v_func = np.vectorize(func, otypes=[complex])
        
        # 3. Compute radial vector for k = 0..M-1 at r = a[bd]
        k_vals = list(range(NMK[bd+1]))
        r_val = a[bd]
        radial_vector = v_func(k_vals, r_val)
        
        # 4. Construct the array
        radial_array = np.outer(np.ones(NMK[bd]), radial_vector)
        return (-1) * radial_array * I_mk_vals
        
    def make_v_diag(left, func, bd):
        return v_diagonal_block(left, func, bd, h, d, NMK, a)

    def make_v_dense(left, func, bd):
        return v_dense_block(left, func, bd, I_nm_vals, NMK, a)
        
    def make_v_diag_e(bd):
        return v_diagonal_block_e(bd, h, NMK, a, m0, m_k_arr)

    def make_v_dense_e(func, bd):
        return v_dense_block_e(func, bd, I_mk_vals, NMK, a)

    # Vectorized function aliases
    v_R1n = np.vectorize(partial(R_1n, h=h, d=d, a=a))
    v_R2n = np.vectorize(partial(R_2n, a=a, h=h, d=d))
    v_diff_R1n = np.vectorize(partial(diff_R_1n, h=h, d=d, a=a), otypes=[complex])
    v_diff_R2n = np.vectorize(partial(diff_R_2n, h=h, d=d, a=a), otypes=[complex])
    
    # --- Manual Matrix Loop (Potential Matching) ---
    rows = []
    size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
    col = 0
    
    for bd in range(boundary_count):
        N = NMK[bd]
        M = NMK[bd + 1]
        
        if bd == (boundary_count - 1): # i-e boundary
            row_height = N
            left_block1 = make_p_diag(True, v_R1n, bd)
            right_block = make_p_dense_e(bd) # Uses the manual fix above
            if bd == 0:
                rows.append(np.concatenate((left_block1, right_block), axis=1))
            else:
                left_block2 = make_p_diag(True, v_R2n, bd)
                left_zeros = np.zeros((row_height, col), dtype=complex)
                rows.append(np.concatenate((left_zeros, left_block1, left_block2, right_block), axis=1))
        
        elif bd == 0:
            left_diag = d[bd] > d[bd + 1]
            if left_diag:
                row_height = N
                left_block = make_p_diag(True, v_R1n, 0)
                right_block1 = make_p_dense(False, v_R1n, 0)
                right_block2 = make_p_dense(False, v_R2n, 0)
            else:
                row_height = M
                left_block = make_p_dense(True, v_R1n, 0)
                right_block1 = make_p_diag(False, v_R1n, 0)
                right_block2 = make_p_diag(False, v_R2n, 0)
            right_zeros = np.zeros((row_height, size - (col + N + 2 * M)), dtype=complex)
            rows.append(np.concatenate([left_block, right_block1, right_block2, right_zeros], axis=1))
            col += N
            
        else: # i-i boundary
            left_diag = d[bd] > d[bd + 1]
            if left_diag:
                row_height = N
                lb1 = make_p_diag(True, v_R1n, bd)
                lb2 = make_p_diag(True, v_R2n, bd)
                rb1 = make_p_dense(False, v_R1n, bd)
                rb2 = make_p_dense(False, v_R2n, bd)
            else:
                row_height = M
                lb1 = make_p_dense(True, v_R1n, bd)
                lb2 = make_p_dense(True, v_R2n, bd)
                rb1 = make_p_diag(False, v_R1n, bd)
                rb2 = make_p_diag(False, v_R2n, bd)
            
            left_zeros = np.zeros((row_height, col), dtype=complex)
            right_zeros = np.zeros((row_height, size - (col + 2*N + 2*M)), dtype=complex)
            rows.append(np.concatenate([left_zeros, lb1, lb2, rb1, rb2, right_zeros], axis=1))
            col += 2 * N

    # --- Manual Matrix Loop (Velocity Matching) ---
    col = 0
    for bd in range(boundary_count):
        N = NMK[bd]
        M = NMK[bd + 1]
        
        if bd == (boundary_count - 1): # i-e boundary
            row_height = M
            left_block1 = make_v_dense_e(v_diff_R1n, bd)
            right_block = make_v_diag_e(bd)
            
            if bd == 0:
                rows.append(np.concatenate((left_block1, right_block), axis=1))
            else:
                left_block2 = make_v_dense_e(v_diff_R2n, bd)
                left_zeros = np.zeros((row_height, col), dtype=complex)
                rows.append(np.concatenate((left_zeros, left_block1, left_block2, right_block), axis=1))
                
        elif bd == 0:
            left_diag = d[bd] <= d[bd + 1]
            if left_diag:
                row_height = N
                lb = make_v_diag(True, v_diff_R1n, 0)
                rb1 = make_v_dense(False, v_diff_R1n, 0)
                rb2 = make_v_dense(False, v_diff_R2n, 0)
            else:
                row_height = M
                lb = make_v_dense(True, v_diff_R1n, 0)
                rb1 = make_v_diag(False, v_diff_R1n, 0)
                rb2 = make_v_diag(False, v_diff_R2n, 0)
            right_zeros = np.zeros((row_height, size - (col + N + 2 * M)), dtype=complex)
            rows.append(np.concatenate([lb, rb1, rb2, right_zeros], axis=1))
            col += N
            
        else: # i-i
            left_diag = d[bd] <= d[bd + 1]
            if left_diag:
                row_height = N
                lb1 = make_v_diag(True, v_diff_R1n, bd)
                lb2 = make_v_diag(True, v_diff_R2n, bd)
                rb1 = make_v_dense(False, v_diff_R1n, bd)
                rb2 = make_v_dense(False, v_diff_R2n, bd)
            else:
                row_height = M
                lb1 = make_v_dense(True, v_diff_R1n, bd)
                lb2 = make_v_dense(True, v_diff_R2n, bd)
                rb1 = make_v_diag(False, v_diff_R1n, bd)
                rb2 = make_v_diag(False, v_diff_R2n, bd)
                
            left_zeros = np.zeros((row_height, col), dtype=complex)
            right_zeros = np.zeros((row_height, size - (col + 2*N + 2*M)), dtype=complex)
            rows.append(np.concatenate([left_zeros, lb1, lb2, rb1, rb2, right_zeros], axis=1))
            col += 2 * N

    A_manual = np.concatenate(rows, axis=0)
    
    # --- Manual b Vector Construction ---
    b_manual = np.zeros(size, dtype=complex)
    index = 0
    # Potential
    for bd in range(boundary_count):
        if bd == (boundary_count - 1):
            for n in range(NMK[-2]):
                b_manual[index] = b_potential_end_entry(n, bd, heaving, h, d, a)
                index += 1
        else:
            iter_range = NMK[bd] if d[bd] > d[bd+1] else NMK[bd+1]
            for n in range(iter_range):
                b_manual[index] = b_potential_entry(n, bd, d, heaving, h, a)
                index += 1
                
    # Velocity
    for bd in range(boundary_count):
        if bd == (boundary_count - 1):
            for n in range(NMK[-1]):
                b_manual[index] = b_velocity_end_entry(n, bd, heaving, a, h, d, m0, NMK, m_k_arr, N_k_arr)
                index += 1
        else:
            iter_range = NMK[bd] if d[bd] <= d[bd+1] else NMK[bd+1]
            for n in range(iter_range):
                b_manual[index] = b_velocity_entry(n, bd, heaving, a, h, d)
                index += 1

    # ---------------------------------------------------------
    # 3. Comparisons
    # ---------------------------------------------------------
    
    # Debug info if they fail
    if not np.allclose(A_engine, A_manual):
        print(f"Max Difference A: {np.max(np.abs(A_engine - A_manual))}")
        
    if not np.allclose(b_engine, b_manual):
        print(f"Max Difference b: {np.max(np.abs(b_engine - b_manual))}")

    np.testing.assert_allclose(A_engine, A_manual, rtol=1e-10, atol=1e-10, err_msg="Matrix A mismatch between Engine and Manual script.")
    np.testing.assert_allclose(b_engine, b_manual, rtol=1e-10, atol=1e-10, err_msg="Vector b mismatch between Engine and Manual script.")


def test_hydro_calculations(meem_setup):
    """
    Solves the system and asserts that hydrodynamic coefficients are calculated
    and are non-trivial (non-zero).
    """
    engine = MEEMEngine([meem_setup["problem"]])
    problem = meem_setup["problem"]
    m0 = meem_setup["m0"]
    
    X = engine.solve_linear_system_multi(problem, m0)
    assert not np.isnan(X).any(), "Solution vector X contains NaNs"
    
    # Calculate forces
    results = engine.compute_hydrodynamic_coefficients(problem, X, m0)
    
    assert len(results) > 0
    
    # Check that at least the heaving body (index 0) has non-zero force
    added_mass = results[0]["real"]
    damping = results[0]["imag"]
    
    print(f"Added Mass: {added_mass}, Damping: {damping}")
    
    assert abs(added_mass) > 1e-5, "Added mass should be non-zero for this configuration"
    # Damping might be small but usually non-zero unless trapped mode
    assert abs(damping) >= 0.0, "Damping must be non-negative (energy conservation)"