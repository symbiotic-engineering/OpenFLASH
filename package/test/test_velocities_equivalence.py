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
from openflash.geometry import Geometry
from openflash.domain import Domain

# --- Path Setup ---
current_dir = os.path.dirname(__file__)

# Add hydro/python for old code
old_code_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'hydro', 'python'))
if old_code_dir not in sys.path:
    sys.path.insert(0, old_code_dir)

from old_assembly import R_1n_old, R_2n_old, Z_n_i_old, Lambda_k_old, Z_k_e_old, diff_R_1n_old, diff_R_2n_old, diff_Z_n_i_old, diff_Lambda_k_old, diff_Z_k_e_old, diff_r_phi_p_i_old, diff_z_phi_p_i_old, make_R_Z_old

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

    # 3. Vectorize the helper functions (the slow, old way)
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
    # 1. ARRANGE: Set up the common problem parameters
    print("--- Setting up test problem ---")
    NMK = [1, 1, 1, 1]
    h = 100
    d = [29, 7, 4]
    a = [3, 5, 10]
    heaving = [0, 1, 1]
    m0 = 1.0

    domain_params = Domain.build_domain_params(NMK, a, d, heaving, h)
    geometry = Geometry(
        Domain.build_r_coordinates_dict(a),
        Domain.build_z_coordinates_dict(h),
        domain_params
    )
    problem = MEEMProblem(geometry)
    engine = MEEMEngine(problem_list=[problem])

    print("--- Solving linear system once ---")
    X = engine.solve_linear_system_multi(problem, m0)

    # 2. ACT: Run both the new and old calculation methods
    velocities_new = engine.calculate_velocities(problem, X, m0, spatial_res=50, sharp=True)
    velocities_old = calculate_velocities_old(X, NMK, h, d, a, m0, heaving)
    
    # 3. ASSERT: Compare the results
    print("\n--- Comparing NEW vs OLD Velocity Results ---")
    try:
        # Compare total velocities
        np.testing.assert_allclose(velocities_new['vr'], velocities_old['vr'], rtol=1e-8, atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(velocities_new['vz'], velocities_old['vz'], rtol=1e-8, atol=1e-8, equal_nan=True)
        
        # Optionally, compare components for more detailed debugging
        np.testing.assert_allclose(velocities_new['vrH'], velocities_old['vrH'], rtol=1e-8, atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(velocities_new['vzH'], velocities_old['vzH'], rtol=1e-8, atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(velocities_new['vrP'], velocities_old['vrP'], rtol=1e-8, atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(velocities_new['vzP'], velocities_old['vzP'], rtol=1e-8, atol=1e-8, equal_nan=True)
        
        print("\n✅ SUCCESS: All velocity arrays match perfectly!")
        
    except AssertionError as e:
        print("\n❌ FAILURE: Velocity arrays DO NOT match.")
        print("Error details:")
        print(e)
        
        diff_vr = np.abs(velocities_new['vr'] - velocities_old['vr'])
        max_diff_vr = np.nanmax(diff_vr)
        print(f"Maximum absolute difference in 'vr': {max_diff_vr}")
        
        diff_vz = np.abs(velocities_new['vz'] - velocities_old['vz'])
        max_diff_vz = np.nanmax(diff_vz)
        print(f"Maximum absolute difference in 'vz': {max_diff_vz}")

if __name__ == "__main__":
    main()