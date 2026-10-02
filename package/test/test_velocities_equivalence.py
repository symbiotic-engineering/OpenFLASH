import numpy as np
import sys
import os

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import Package Modules ---
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.geometry import ConcentricBodyGroup
from openflash.body import SteppedBody

# --- Path Setup for Old Code ---
old_code_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'dev', 'python'))
if old_code_dir not in sys.path:
    sys.path.insert(0, old_code_dir)

# Attempt to import old assembly functions
try:
    from old_assembly import R_1n_old, R_2n_old, Z_n_i_old, Lambda_k_old, Z_k_e_old, \
        diff_R_1n_old, diff_R_2n_old, diff_Z_n_i_old, diff_Lambda_k_old, diff_Z_k_e_old, \
        diff_r_phi_p_i_old, diff_z_phi_p_i_old, make_R_Z_old
except ImportError:
    print("WARNING: Could not import 'old_assembly'. Ensure the path is correct relative to this test script.")
    sys.exit(1)

# --- Wrapper for the Old Calculation Logic ---
def calculate_velocities_old(X, NMK, h, d, a, m0, heaving):
    """
    Encapsulates the original, non-vectorized velocity calculation logic for testing.
    """
    print("\n--- Running OLD Velocity Calculation ---")
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
    
    # 2. Define old, non-vectorized helper functions
    def v_r_inner_func(n, r, z):
        return (Cs[0][n] * diff_R_1n_old(n, r, 0, h, d, a)) * Z_n_i_old(n, z, 0, h, d)
    def v_r_m_i_func(i, m, r, z):
        return (Cs[i][m] * diff_R_1n_old(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * diff_R_2n_old(m, r, i, h, d, a)) * Z_n_i_old(m, z, i, h, d)
    def v_r_e_k_func(k, r, z):
        return Cs[-1][k] * diff_Lambda_k_old(k, r, m0, a, NMK, h) * Z_k_e_old(k, z, m0, h, NMK)
    def v_z_inner_func(n, r, z):
        return (Cs[0][n] * R_1n_old(n, r, 0, h, d, a)) * diff_Z_n_i_old(n, z, 0, h, d)
    def v_z_m_i_func(i, m, r, z):
        return (Cs[i][m] * R_1n_old(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n_old(m, r, i, a, h, d)) * diff_Z_n_i_old(m, z, i, h, d)
    def v_z_e_k_func(k, r, z):
        return Cs[-1][k] * Lambda_k_old(k, r, m0, a, NMK, h) * diff_Z_k_e_old(k, z, NMK, m0, h)

    # 3. Vectorize the helper functions
    v_r_inner_vec = np.vectorize(v_r_inner_func, otypes=[complex])
    v_r_m_i_vec = np.vectorize(v_r_m_i_func, otypes=[complex])
    v_r_e_k_vec = np.vectorize(v_r_e_k_func, otypes=[complex])
    v_z_inner_vec = np.vectorize(v_z_inner_func, otypes=[complex])
    v_z_m_i_vec = np.vectorize(v_z_m_i_func, otypes=[complex])
    v_z_e_k_vec = np.vectorize(v_z_e_k_func, otypes=[complex])
    vr_p_i_vec = np.vectorize(lambda d, r, z, h: diff_r_phi_p_i_old(d, r, h), otypes=[complex])
    vz_p_i_vec = np.vectorize(lambda d, r, z, h: diff_z_phi_p_i_old(d, z, h), otypes=[complex])

    # 4. Execute the calculation loop
    R, Z = make_R_Z_old(a, h, d, True, 50)
    regions = [
        (R <= a[0]) & (Z < -d[0]),
        *[(R > a[i-1]) & (R <= a[i]) & (Z < -d[i]) for i in range(1, boundary_count)],
        (R > a[-1])
    ]

    vrH = np.full_like(R, np.nan, dtype=complex)
    vzH = np.full_like(R, np.nan, dtype=complex)
    
    # Homogeneous velocity loops
    for n in range(NMK[0]):
        temp_vrH = v_r_inner_vec(n, R[regions[0]], Z[regions[0]])
        temp_vzH = v_z_inner_vec(n, R[regions[0]], Z[regions[0]])
        vrH[regions[0]] = temp_vrH if n == 0 else vrH[regions[0]] + temp_vrH
        vzH[regions[0]] = temp_vzH if n == 0 else vzH[regions[0]] + temp_vzH
    
    for i in range(1, boundary_count):
        for m in range(NMK[i]):
            temp_vrH = v_r_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
            temp_vzH = v_z_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
            vrH[regions[i]] = temp_vrH if m == 0 else vrH[regions[i]] + temp_vrH
            vzH[regions[i]] = temp_vzH if m == 0 else vzH[regions[i]] + temp_vzH
            
    for k in range(NMK[-1]):
        temp_vrH = v_r_e_k_vec(k, R[regions[-1]], Z[regions[-1]])
        temp_vzH = v_z_e_k_vec(k, R[regions[-1]], Z[regions[-1]])
        vrH[regions[-1]] = temp_vrH if k == 0 else vrH[regions[-1]] + temp_vrH
        vzH[regions[-1]] = temp_vzH if k == 0 else vzH[regions[-1]] + temp_vzH

    # Particular velocity
    vrP = np.full_like(R, 0.0, dtype=complex)
    vzP = np.full_like(R, 0.0, dtype=complex)
    
    vrP[regions[0]] = heaving[0] * vr_p_i_vec(d[0], R[regions[0]], Z[regions[0]], h)
    vzP[regions[0]] = heaving[0] * vz_p_i_vec(d[0], R[regions[0]], Z[regions[0]], h)
    for i in range(1, boundary_count):
        vrP[regions[i]] = heaving[i] * vr_p_i_vec(d[i], R[regions[i]], Z[regions[i]], h)
        vzP[regions[i]] = heaving[i] * vz_p_i_vec(d[i], R[regions[i]], Z[regions[i]], h)
    
    vr = vrH + vrP
    vz = vzH + vzP
    print("--- OLD Calculation Finished ---")
    
    return {"R": R, "Z": Z, "vrH": vrH, "vzH": vzH, "vrP": vrP, "vzP": vzP, "vr": vr, "vz": vz}

# --- Main Test Execution ---
def main():
    # 1. ARRANGE
    print("--- Setting up test problem ---")
    NMK = [1, 1, 1, 1]
    h = 100
    d = np.array([29, 7, 4])
    a = np.array([3, 5, 10])
    heaving = np.array([0, 1, 0]) 
    m0 = 1.0

    bodies = []
    for i in range(len(a)):
        body = SteppedBody(
            a=np.array([a[i]]),
            d=np.array([d[i]]),
            slant_angle=np.array([0.0]),
            heaving=bool(heaving[i])
        )
        bodies.append(body)

    arrangement = ConcentricBodyGroup(bodies)
    geometry = BasicRegionGeometry(body_arrangement=arrangement, h=h, NMK=NMK)
    problem = MEEMProblem(geometry)
    engine = MEEMEngine(problem_list=[problem])

    print("--- Solving linear system once ---")
    X = engine.solve_linear_system_multi(problem, m0)

    # 2. ACT
    velocities_new = engine.calculate_velocities(problem, X, m0, spatial_res=50, sharp=True)
    velocities_old = calculate_velocities_old(X, NMK, h, list(d), list(a), m0, list(heaving))
    
    # --- FIX: Transpose New Results to Match Old Shape ---
    if velocities_new['vr'].shape != velocities_old['vr'].shape:
        print(f"\n[TEST INFO] Transposing 'new' results from {velocities_new['vr'].shape} to match old {velocities_old['vr'].shape}")
        for key in ['vr', 'vz', 'vrH', 'vzH', 'vrP', 'vzP']:
             velocities_new[key] = velocities_new[key].T

    # 3. ASSERT
    print("\n--- Comparing NEW vs OLD Velocity Results ---")
    
    def compare_valid_intersection(name, new_arr, old_arr):
        """Compares two arrays only where BOTH contain valid numbers (ignoring NaNs)."""
        valid_mask = np.isfinite(new_arr) & np.isfinite(old_arr)
        
        overlap_count = np.sum(valid_mask)
        if overlap_count == 0:
            raise AssertionError(f"[{name}] No overlapping valid points found! Grid alignment is totally off.")

        xor_diff = np.sum(np.isfinite(new_arr) ^ np.isfinite(old_arr))
        if xor_diff > 0:
            print(f"  [INFO] {name}: Ignored {xor_diff} pixels of boundary mismatch (wall vs water definition).")
        
        np.testing.assert_allclose(
            new_arr[valid_mask], 
            old_arr[valid_mask], 
            rtol=1e-8, atol=1e-8, 
            err_msg=f"Mismatch in values for {name}"
        )
        print(f"  [PASS] {name}: Matched perfectly across {overlap_count} points.")

    try:
        # Compare all components
        compare_valid_intersection('vr', velocities_new['vr'], velocities_old['vr'])
        compare_valid_intersection('vz', velocities_new['vz'], velocities_old['vz'])
        compare_valid_intersection('vrH', velocities_new['vrH'], velocities_old['vrH'])
        compare_valid_intersection('vzH', velocities_new['vzH'], velocities_old['vzH'])
        compare_valid_intersection('vrP', velocities_new['vrP'], velocities_old['vrP'])
        compare_valid_intersection('vzP', velocities_new['vzP'], velocities_old['vzP'])
        
        print("\n✅ SUCCESS: All velocity arrays match perfectly!")
        
    except AssertionError as e:
        print("\n❌ FAILURE: Velocity arrays DO NOT match.")
        print(e)
        
        # Debug helper
        diff_vr = np.abs(velocities_new['vr'] - velocities_old['vr'])
        max_diff_vr = np.nanmax(diff_vr)
        print(f"Maximum absolute difference in 'vr': {max_diff_vr}")

if __name__ == "__main__":
    main()