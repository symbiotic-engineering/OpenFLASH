# package/test/test_config3_vs_old.py
import numpy as np
import pytest
import os
import sys
import pandas as pd
from scipy.interpolate import griddata

# Ensure imports work for both package and local dev
from openflash.multi_equations import omega
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.multi_constants import g

# Add hydro/python for old code import
current_dir = os.path.dirname(__file__)
old_code_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'dev', 'python'))
if old_code_dir not in sys.path:
    sys.path.insert(0, old_code_dir)

# Import assemble AND the math functions needed for potential calculation
from old_assembly import (
    assemble_old_A_and_b, 
    R_1n_old, R_2n_old, Z_n_i_old, Lambda_k_old, Z_k_e_old, 
    make_R_Z_old, phi_p_i_old
)

# Define path to Benchmark Data
BENCHMARK_DATA_PATH = os.path.join(old_code_dir, "test", "data")

# --- HELPER FUNCTIONS ---

def load_capytaine_data(config_name):
    """Loads the "golden" potential field data for a specific config."""
    real_path = os.path.join(BENCHMARK_DATA_PATH, f"{config_name}-real.csv")
    imag_path = os.path.join(BENCHMARK_DATA_PATH, f"{config_name}-imag.csv")
    
    if not os.path.exists(real_path):
        pytest.skip(f"Benchmark file not found: {real_path}")
    if not os.path.exists(imag_path):
        pytest.skip(f"Benchmark file not found: {imag_path}")

    real_data = np.loadtxt(real_path, delimiter=",")
    imag_data = np.loadtxt(imag_path, delimiter=",")
    return real_data + 1j * imag_data

def calculate_potentials_old(X, NMK, a, h, d, m0, heaving):
    """Encapsulates the original, non-vectorized potential calculation logic."""
    print("\n--- Running OLD Potential Calculation ---")
    
    # Split Cs
    Cs = []
    row = 0
    Cs.append(X[:NMK[0]])
    row += NMK[0]
    boundary_count = len(NMK) - 1
    for i in range(1, boundary_count):
        Cs.append(X[row: row + NMK[i] * 2])
        row += NMK[i] * 2
    Cs.append(X[row:])

    def phi_h_n_inner_func_old(n, r, z):
        return (Cs[0][n] * R_1n_old(n, r, 0, h, d, a)) * Z_n_i_old(n, z, 0, h, d)

    def phi_h_m_i_func_old(i, m, r, z):
        # FIX APPLIED: Argument order (h, d, a)
        term1 = Cs[i][m] * R_1n_old(m, r, i, h, d, a)
        term2 = Cs[i][NMK[i] + m] * R_2n_old(m, r, i, a, h, d)
        return (term1 + term2) * Z_n_i_old(m, z, i, h, d)

    def phi_e_k_func_old(k, r, z):
        return Cs[-1][k] * Lambda_k_old(k, r, m0, a, NMK, h) * Z_k_e_old(k, z, m0, h, NMK)

    phi_e_k_vec = np.vectorize(phi_e_k_func_old, otypes = [complex])
    phi_h_n_inner_vec = np.vectorize(phi_h_n_inner_func_old, otypes = [complex])
    phi_h_m_i_vec = np.vectorize(phi_h_m_i_func_old, otypes = [complex])

    R, Z = make_R_Z_old(a, h, d, True, 50)
    regions = []
    regions.append((R <= a[0]) & (Z < -d[0]))
    for i in range(1, boundary_count):
        regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
    regions.append(R > a[-1])
    
    phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

    # Homogeneous
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

    # Particular
    def phi_p_i_wrapper(d_scalar, r, z):
        return phi_p_i_old(d_scalar, r, z, h)
    phi_p_i_vec = np.vectorize(phi_p_i_wrapper)

    phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
    for i in range(1, boundary_count):
        phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
    phiP[regions[-1]] = 0

    return {"R": R, "Z": Z, "phi": phiH + phiP}

# --- FIXTURES (Shared Data) ---

@pytest.fixture
def config3_params():
    """Returns the dictionary of parameters for Config 3."""
    h = 1.9
    m0 = 1.0
    return {
        "h": h,
        "m0": m0,
        "omega": omega(m0, h, g),
        "a": np.array([0.3, 0.5, 1.0, 1.2, 1.6]),
        "d": np.array([0.5, 0.7, 0.8, 0.2, 0.5]),
        "NMK": [40, 40, 40, 40, 40, 40],
        "body_map": [0, 1, 2, 3, 4]
    }

@pytest.fixture
def old_full_assembly(config3_params):
    """Runs the OLD assembly ONCE for the full configuration to use as ground truth."""
    p = config3_params
    heaving_full = np.array([1, 1, 1, 1, 1])
    print("\n[Fixture] Running Old Assembly (Full)...")
    A_old, b_old = assemble_old_A_and_b(p['h'], p['d'], p['a'], p['NMK'], heaving_full, p['m0'])
    return A_old, b_old, heaving_full

# --- THE 3 TESTS ---

def test_1_atomic_consistency(config3_params):
    """
    TEST 1: Checks if New Code matches Old Code for every individual body.
    """
    p = config3_params
    
    print("\n=== TEST 1: ATOMIC CONSISTENCY (Old vs New) ===")
    
    for body_idx in range(5):
        # 1. Run Old (Atomic)
        heaving_old = np.zeros(5, dtype=int)
        heaving_old[body_idx] = 1
        A_old, b_old = assemble_old_A_and_b(p['h'], p['d'], p['a'], p['NMK'], heaving_old, p['m0'])
        
        # 2. Run New (Atomic)
        heaving_new = [False] * 5
        heaving_new[body_idx] = True
        
        geometry = BasicRegionGeometry.from_vectors(
            a=p['a'], d=p['d'], h=p['h'], NMK=p['NMK'], 
            body_map=p['body_map'], heaving_map=heaving_new
        )
        problem = MEEMProblem(geometry)
        problem.set_frequencies(np.array([p['omega']]))
        engine = MEEMEngine([problem])
        
        # Build cache explicitly to ensure stability
        if problem not in engine.cache_list:
            engine.cache_list[problem] = engine.build_problem_cache(problem)
            
        A_new = engine.assemble_A_multi(problem, p['m0'])
        b_new = engine.assemble_b_multi(problem, p['m0'])
        
        # 3. Assert
        # We use a try-except block here purely to print a nice message, 
        # but we let the assertion error bubble up so pytest marks it as FAILED.
        try:
            np.testing.assert_allclose(A_new, A_old, rtol=1e-10, atol=1e-10, err_msg=f"Body {body_idx} Matrix A")
            np.testing.assert_allclose(b_new, b_old, rtol=1e-10, atol=1e-10, err_msg=f"Body {body_idx} Vector b")
            print(f"  ✅ Body {body_idx}: Match confirmed.")
        except AssertionError as e:
            print(f"  ❌ Body {body_idx}: FAILED.")
            raise e

def test_2_superposition_consistency(config3_params, old_full_assembly):
    """
    TEST 2: Checks if Sum(New Atomic Results) == Old Full Result.
    """
    p = config3_params
    _, b_old_full, _ = old_full_assembly # Get ground truth from fixture
    
    print("\n=== TEST 2: SUPERPOSITION CHECK ===")
    
    b_new_accumulated = None
    
    for body_idx in range(5):
        heaving_new = [False] * 5
        heaving_new[body_idx] = True
        
        geometry = BasicRegionGeometry.from_vectors(
            a=p['a'], d=p['d'], h=p['h'], NMK=p['NMK'], 
            body_map=p['body_map'], heaving_map=heaving_new
        )
        problem = MEEMProblem(geometry)
        problem.set_frequencies(np.array([p['omega']]))
        engine = MEEMEngine([problem])
        
        # Build cache
        if problem not in engine.cache_list:
            engine.cache_list[problem] = engine.build_problem_cache(problem)

        b_new = engine.assemble_b_multi(problem, p['m0'])
        
        if b_new_accumulated is None:
            b_new_accumulated = np.zeros_like(b_new)
        b_new_accumulated += b_new

    # Compare
    diff = np.max(np.abs(b_old_full - b_new_accumulated))
    print(f"  Max Difference (Old Full vs Sum New): {diff:.6e}")
    
    np.testing.assert_allclose(b_new_accumulated, b_old_full, rtol=1e-10, atol=1e-10)
    print("  ✅ Superposition confirmed.")

def test_3_old_code_vs_capytaine(config3_params, old_full_assembly):
    """
    TEST 3: Checks if Old Code output matches Capytaine Benchmark.
    """
    p = config3_params
    A_old, b_old, heaving_full = old_full_assembly
    
    print("\n=== TEST 3: OLD CODE vs CAPYTAINE ===")
    
    # 1. Solve Old System
    print("  -> Solving Linear System (Old Code)...")
    X_old = np.linalg.solve(A_old, b_old)
    
    # 2. Calculate Potentials
    res = calculate_potentials_old(X_old, p['NMK'], p['a'], p['h'], p['d'], p['m0'], heaving_full)
    phi_old = res['phi']
    R_old, Z_old = res['R'], res['Z']
    
    # 3. Load Capytaine
    try:
        phi_cap_raw = load_capytaine_data("config3")
    except Exception as e:
        pytest.skip(f"Could not load Capytaine data: {e}")
        
    # 4. Interpolate Capytaine to Old Grid
    # Hardcoded Capytaine grid based on 'test_capytaine_potential.py' config
    R_cap_grid, Z_cap_grid = np.meshgrid(
        np.linspace(0.0, 2 * 1.6, num=50), 
        np.linspace(0, -1.9, num=50), 
        indexing='ij'
    )
    
    # Unit Conversion: Capytaine (Diffraction) -> Velocity Potential
    # Real = Imag * (-1/w), Imag = Real * (1/w)
    cap_real_conv = phi_cap_raw.imag * (-1.0 / p['omega'])
    cap_imag_conv = phi_cap_raw.real * (1.0 / p['omega'])
    
    points_cap = np.array([R_cap_grid.flatten(), Z_cap_grid.flatten()]).T
    phi_cap_interp_real = griddata(points_cap, cap_real_conv.flatten(), (R_old, Z_old), method='linear')
    phi_cap_interp_imag = griddata(points_cap, cap_imag_conv.flatten(), (R_old, Z_old), method='linear')
    
    # 5. Compare
    valid_mask = ~np.isnan(phi_old) & ~np.isnan(phi_cap_interp_real)
    
    if np.sum(valid_mask) == 0:
        pytest.fail("No valid overlapping points found.")

    diff_real = np.abs(phi_old.real[valid_mask] - phi_cap_interp_real[valid_mask])
    diff_imag = np.abs(phi_old.imag[valid_mask] - phi_cap_interp_imag[valid_mask])
    
    print(f"  [Real Part] Max Diff: {np.max(diff_real):.6e}")
    print(f"  [Imag Part] Max Diff: {np.max(diff_imag):.6e}")
    
    # Relaxed tolerance for interpolation/mesh differences
    tol = 0.2
    if np.max(diff_real) > tol or np.max(diff_imag) > tol:
        pytest.fail(f"Old code diverges from Capytaine > {tol}")
        
    print("  ✅ Old Code matches Capytaine.")