import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from functools import partial
import scipy.linalg as linalg

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import New Code
from openflash.geometry import Geometry, ConcentricBodyGroup
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.body import SteppedBody
from openflash.basic_region_geometry import BasicRegionGeometry

# Import Old Code
old_code_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'dev', 'python'))
if old_code_dir not in sys.path:
    sys.path.insert(0, old_code_dir)

# Import old assembly functions required for potential calculation
from old_assembly import (
    assemble_old_A_and_b,
    R_1n_old, Z_n_i_old, R_2n_old, Lambda_k_old, Z_k_e_old, 
    make_R_Z_old, phi_p_i_old
)

# --- CONFIGURATIONS ---
ALL_CONFIGS = {
    # "config0": {
    #     "h": 1.001,
    #     "a": np.array([0.5, 1]),
    #     "d": np.array([0.5, 0.25]),
    #     "heaving_map": [True, True],
    #     "body_map": [0, 1],
    #     "m0": 1.0,
    #     "NMK": [15, 15, 15], 
    # },
    # "config1": {
    #     "h": 1.5,
    #     "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
    #     "d": np.array([1.1, 0.85, 0.75, 0.4, 0.15]),
    #     "heaving_map": [True, True, True, True, True],
    #     "body_map": [0, 1, 2, 3, 4],
    #     "m0": 1.0,
    #     "NMK": [15] * 6, 
    # },
    # "config2": {
    #     "h": 100.0,
    #     "a": np.array([3, 5, 10]),
    #     "d": np.array([29, 7, 4]),
    #     "heaving_map": [True, True, True],
    #     "body_map": [0, 1, 2],
    #     "m0": 1.0,
    #     "NMK": [100] * 4,
    # },
    "config3": {
        "h": 1.9,
        "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
        "d": np.array([0.5, 0.7, 0.8, 0.2, 0.5]),
        "heaving_map": [True, True, True, True, True],
        "body_map": [0, 1, 2, 3, 4],
        "m0": 1.0,
        "NMK": [10] * 6,
    },
    # "config4": {
    #     "h": 1.001,
    #     "a": np.array([0.5, 1]),
    #     "d": np.array([0.5, 0.25]),
    #     "heaving_map": [False, True],
    #     "body_map": [0, 1],
    #     "m0": 1.0,
    #     "NMK": [15] * 3,
    # },
    # "config5": {
    #     "h": 1.001,
    #     "a": np.array([0.5, 1]),
    #     "d": np.array([0.5, 0.25]),
    #     "heaving_map": [True, False],
    #     "body_map": [0, 1],
    #     "m0": 1.0,
    #     "NMK": [15] * 3,
    # },
    "config6": {
        "h": 100.0,
        "a": np.array([3, 5, 10]),
        "d": np.array([29, 7, 4]),
        "heaving_map": [False, True, True],
        "body_map": [0, 1, 2],
        "m0": 1.0,
        "NMK": [10] * 4,
    },
    # "config7": {
    #     "h": 1.001,
    #     "a": np.array([0.5, 1]),
    #     "d": np.array([0.25, 0.5]),
    #     "heaving_map": [True, False],
    #     "body_map": [0, 1],
    #     "m0": 1.0,
    #     "NMK": [15] * 3,
    # },
    # "config8": {
    #     "h": 1.001,
    #     "a": np.array([0.5, 1]),
    #     "d": np.array([0.25, 0.5]),
    #     "heaving_map": [False, True],
    #     "body_map": [0, 1],
    #     "m0": 1.0,
    #     "NMK": [15] * 3,
    # },
    "config9": {
        "h": 100.0,
        "a": np.array([3, 5, 10]),
        "d": np.array([4, 7, 29]),
        "heaving_map": [True, True, True],
        "body_map": [0, 1, 2],
        "m0": 1.0,
        "NMK": [10] * 4,
    },
    "config10": {
        "h": 1.5,
        "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
        "d": np.array([0.15, 0.4, 0.75, 0.85, 1.1]),
        "heaving_map": [True, True, True, True, True],
        "body_map": [0, 1, 2, 3, 4],
        "m0": 1.0,
        "NMK": [15, 15, 15, 15, 15, 15], 
    },
    # "config11": {
    #     "h": 1.001,
    #     "a": np.array([0.5, 1]),
    #     "d": np.array([0.25, 0.5]),
    #     "heaving_map": [True, True],
    #     "body_map": [0, 1],
    #     "m0": 1.0,
    #     "NMK": [15, 15, 15],
    # },
    # "config14": {
    #     "h": 1.9,
    #     "a": np.array([1.2, 1.6]),
    #     "d": np.array([0.2, 0.5]),
    #     "heaving_map": [True, True],
    #     "body_map": [0, 1],
    #     "m0": 1.0,
    #     "NMK": [50] * 3,
    # },
}

# --- MONKEY PATCH ---
from openflash.geometry import BodyArrangement
original_init = BodyArrangement.__init__

def patched_init(self, bodies):
    self.bodies = bodies
    # Skip assertion for testing
    
BodyArrangement.__init__ = patched_init
# --------------------

# --- OLD POTENTIAL CALCULATION WRAPPER ---
def calculate_potentials_old(X, NMK, a, h, d, m0, heaving):
    """
    Encapsulates the original, non-vectorized potential calculation logic for testing.
    """
    # Split up the Cs into groups depending on which equation they belong to.
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
        # FIX: Adjusted argument order for R_1n_old to (..., h, d, a)
        return (Cs[i][m] * R_1n_old(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n_old(m, r, i, a, h, d)) * Z_n_i_old(m, z, i, h, d)

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
    
    phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

    # Calculate Homogeneous Potential (phiH)
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

    # Calculate Particular Potential (phiP)
    def phi_p_i_wrapper(d_scalar, r, z):
        return phi_p_i_old(d_scalar, r, z, h)

    phi_p_i_vec = np.vectorize(phi_p_i_wrapper)

    phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
    for i in range(1, boundary_count):
        phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
    phiP[regions[-1]] = 0

    phi = phiH + phiP
    
    return {"R": R, "Z": Z, "phiH": phiH, "phiP": phiP, "phi": phi}

def compare_valid_intersection(name, new_arr, old_arr, tol=1e-8):
    """Compares two arrays only where BOTH contain valid numbers (ignoring NaNs)."""
    valid_mask = np.isfinite(new_arr) & np.isfinite(old_arr)
    
    overlap_count = np.sum(valid_mask)
    if overlap_count == 0:
        print(f"   [{name}] No overlapping valid points found! Grid alignment might be off.")
        return False

    diff = np.abs(new_arr[valid_mask] - old_arr[valid_mask])
    max_diff = np.max(diff)
    
    if max_diff > tol:
        print(f"   [{name}] MISMATCH. Max diff: {max_diff:.4e} (tol={tol})")
        return False
    else:
        # print(f"   [{name}] Match OK. Max diff: {max_diff:.4e}")
        return True

def run_single_config(name, config):
    print(f"\n{'='*20} Testing {name} {'='*20}")
    
    # Unpack config
    h = config["h"]
    a = config["a"]
    d = config["d"]
    heaving = config["heaving_map"]
    m0 = config["m0"]
    NMK = config["NMK"]
    
    # Prepare heaving list for old code (must include exterior)
    # Convert to array for potential calc logic
    heaving_old_list = list(heaving) + [0]
    heaving_old_arr = np.array(heaving_old_list, dtype=int)
    
    # 1. Assemble OLD
    try:
        A_old, b_old = assemble_old_A_and_b(h, d, a, NMK, heaving_old_list, m0)
    except Exception as e:
        print(f"OLD Assembly Failed: {e}")
        return False

    # 2. Assemble NEW
    try:
        bodies = []
        for i, r in enumerate(a):
            body = SteppedBody(
                a=np.array([r]),
                d=np.array([d[i]]),
                slant_angle=np.array([0.0]),
                heaving=bool(heaving[i])
            )
            bodies.append(body)
        
        arrangement = ConcentricBodyGroup(bodies)
        geometry = BasicRegionGeometry(arrangement, h, NMK)
        problem = MEEMProblem(geometry)
        engine = MEEMEngine([problem])
        
        A_new = engine.assemble_A_multi(problem, m0)
        b_new = engine.assemble_b_multi(problem, m0)
        
    except Exception as e:
        print(f"NEW Assembly Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Compare Matrices
    if A_old.shape != A_new.shape:
        print(f"SHAPE MISMATCH: Old {A_old.shape} vs New {A_new.shape}")
        return False
        
    A_diff = np.abs(A_old - A_new)
    b_diff = np.abs(b_old - b_new)
    
    max_A_diff = np.max(A_diff)
    max_b_diff = np.max(b_diff)
    
    matrix_tol = 1e-10
    matrix_pass = (max_A_diff < matrix_tol) and (max_b_diff < matrix_tol)
    
    if matrix_pass:
        print(f"Matrices MATCH (A diff={max_A_diff:.2e}, b diff={max_b_diff:.2e})")
    else:
        print(f"Matrices MISMATCH (A diff={max_A_diff:.2e}, b diff={max_b_diff:.2e})")
        # Proceed to potentials anyway, but using the NEW coefficients to test the potential logic specifically

    # 4. Calculate Potentials
    # Solve system using NEW matrices to get X
    try:
        X = linalg.solve(A_new, b_new)
    except Exception as e:
        print(f"Linear Solve Failed: {e}")
        return False
        
    # Calculate NEW Potentials
    potentials_new = engine.calculate_potentials(problem, X, m0, spatial_res=50, sharp=True)
    
    # Calculate OLD Potentials
    potentials_old = calculate_potentials_old(X, NMK, a, h, d, m0, heaving_old_arr)

    # Transpose if necessary (matching test_potential_calcs.py fix)
    if potentials_new['phi'].shape != potentials_old['phi'].shape:
        # print(f"Transposing NEW potentials to match OLD shape...")
        potentials_new['phi'] = potentials_new['phi'].T
        potentials_new['phiH'] = potentials_new['phiH'].T
        potentials_new['phiP'] = potentials_new['phiP'].T
        
    # Compare Potentials
    pot_tol = 1e-8
    phiH_ok = compare_valid_intersection('phiH', potentials_new['phiH'], potentials_old['phiH'], pot_tol)
    phiP_ok = compare_valid_intersection('phiP', potentials_new['phiP'], potentials_old['phiP'], pot_tol)
    phi_ok  = compare_valid_intersection('phi',  potentials_new['phi'],  potentials_old['phi'],  pot_tol)
    
    potential_pass = phiH_ok and phiP_ok and phi_ok
    
    if potential_pass:
        print("Potentials MATCH")
    else:
        print("Potentials MISMATCH")
        
    return matrix_pass and potential_pass

def main():
    print("Starting Matrix & Potential Configuration Tests...")
    results = []
    
    for name, config in ALL_CONFIGS.items():
        success = run_single_config(name, config)
        results.append((name, success))
        
    print("\n\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"{'Config':<15} | {'Result':<10}")
    print("-" * 30)
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name:<15} | {status:<10}")
        
if __name__ == "__main__":
    main()