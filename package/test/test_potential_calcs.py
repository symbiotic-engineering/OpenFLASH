import numpy as np
import sys
import os

# --- Path Setup ---
# This ensures the script can find package files
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

# --- Path Setup ---
current_dir = os.path.dirname(__file__)

# Add hydro/python for old code
old_code_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'dev', 'python'))
if old_code_dir not in sys.path:
    sys.path.insert(0, old_code_dir)

from old_assembly import R_1n_old, Z_n_i_old, R_2n_old, Lambda_k_old, Z_k_e_old, make_R_Z_old, phi_p_i_old

# --- Wrapper for the Old Calculation Logic ---
def calculate_potentials_old(X, NMK, a, h, d, m0, heaving):
    """
    Encapsulates the original, non-vectorized potential calculation logic for testing.
    """
    print("\n--- Running OLD Potential Calculation ---")
    print("--- [DEBUG OLD] Entering function with parameters: ---")
    print(f"  - X shape: {X.shape}, NMK: {NMK}")
    print(f"  - a: {a}, h: {h}, d: {d}, m0: {m0}")
    
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
    print("\n--- [DEBUG OLD] Coefficients (Cs) reformatted: ---")
    for i, C_group in enumerate(Cs):
        print(f"  - Group {i} shape: {C_group.shape}, Max abs value: {np.max(np.abs(C_group)):.4f}")

    def phi_h_n_inner_func_old(n, r, z):
        return (Cs[0][n] * R_1n_old(n, r, 0, h, d, a)) * Z_n_i_old(n, z, 0, h, d)

    def phi_h_m_i_func_old(i, m, r, z):
        return (Cs[i][m] * R_1n_old(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n_old(m, r, i, a, h, d)) * Z_n_i_old(m, z, i, h, d)

    def phi_e_k_func_old(k, r, z):
        return Cs[-1][k] * Lambda_k_old(k, r, m0, a, NMK, h) * Z_k_e_old(k, z, m0, h, NMK)

    phi_e_k_vec = np.vectorize(phi_e_k_func_old, otypes = [complex])
    phi_h_n_inner_vec = np.vectorize(phi_h_n_inner_func_old, otypes = [complex])
    phi_h_m_i_vec = np.vectorize(phi_h_m_i_func_old, otypes = [complex])

    R, Z = make_R_Z_old(a, h, d, True, 50)
    print(f"\n--- [DEBUG OLD] Meshgrid created: R shape={R.shape}, Z shape={Z.shape} ---")


    regions = []
    regions.append((R <= a[0]) & (Z < -d[0]))
    for i in range(1, boundary_count):
        regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
    regions.append(R > a[-1])
    
    print("--- [DEBUG OLD] Region mask point counts: ---")
    for i, region_mask in enumerate(regions):
        print(f"  - Region {i}: {np.sum(region_mask)} points")

    phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

    print("\n--- [DEBUG OLD] Calculating Homogeneous Potential (phiH)... ---")

    for n in range(NMK[0]):
        temp_phiH = phi_h_n_inner_vec(n, R[regions[0]], Z[regions[0]])
        phiH[regions[0]] = temp_phiH if n == 0 else phiH[regions[0]] + temp_phiH
    print(f"  - Done with Region 0. Max abs phiH so far: {np.nanmax(np.abs(phiH[regions[0]])):.4f}")


    for i in range(1, boundary_count):
        for m in range(NMK[i]):
            temp_phiH = phi_h_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
            phiH[regions[i]] = temp_phiH if m == 0 else phiH[regions[i]] + temp_phiH
    print(f"  - Done with Region {i}. Max abs phiH so far: {np.nanmax(np.abs(phiH[regions[i]])):.4f}")


    for k in range(NMK[-1]):
        temp_phiH = phi_e_k_vec(k, R[regions[-1]], Z[regions[-1]])
        phiH[regions[-1]] = temp_phiH if k == 0 else phiH[regions[-1]] + temp_phiH
    print(f"  - Done with Exterior Region. Max abs phiH so far: {np.nanmax(np.abs(phiH[regions[-1]])):.4f}")

    print("\n--- [DEBUG OLD] Calculating Particular Potential (phiP)... ---")
    def phi_p_i_wrapper(d_scalar, r, z):
        return phi_p_i_old(d_scalar, r, z, h)

    phi_p_i_vec = np.vectorize(phi_p_i_wrapper)

    phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
    for i in range(1, boundary_count):
        phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
    phiP[regions[-1]] = 0
    print(f"  - phiP calculated. Max abs value: {np.nanmax(np.abs(phiP)):.4f}")


    phi = phiH + phiP
    print("\n--- [DEBUG OLD] Final potential array summaries: ---")
    print(f"  - phiH shape: {phiH.shape}, max abs: {np.nanmax(np.abs(phiH)):.4f}")
    print(f"  - phiP shape: {phiP.shape}, max abs: {np.nanmax(np.abs(phiP)):.4f}")
    print(f"  - phi shape: {phi.shape}, max abs: {np.nanmax(np.abs(phi)):.4f}")


    print("--- OLD Calculation Finished ---")
    
    return {"R": R, "Z": Z, "phiH": phiH, "phiP": phiP, "phi": phi}


# --- Main Test Execution ---
def main():
    # 1. ARRANGE: Set up the common problem parameters
    print("--- Setting up test problem ---")
    NMK = [1, 1, 1, 1]
    h = 100
    d = np.array([29, 7, 4])
    a = np.array([3, 5, 10])
    heaving = np.array([0, 1, 1])
    m0 = 1.0

    # --- CORRECTED Geometry and Problem Setup ---
    # 1. Define the physical bodies
    bodies = []
    for i in range(len(a)):
        body = SteppedBody(
            a=np.array([a[i]]),
            d=np.array([d[i]]),
            slant_angle=np.array([0.0]),  # Assuming zero slant for test
            heaving=bool(heaving[i])
        )
        bodies.append(body)

    # 2. Create the body arrangement
    arrangement = ConcentricBodyGroup(bodies)

    # 3. Instantiate the CONCRETE geometry class
    geometry = BasicRegionGeometry(
        body_arrangement=arrangement,
        h=h,
        NMK=NMK
    )
    
    # 4. Create the problem object
    problem = MEEMProblem(geometry)
    engine = MEEMEngine(problem_list=[problem])

    # Solve the system once to get the common solution vector X
    print("--- Solving linear system once ---")
    X = engine.solve_linear_system_multi(problem, m0)

    # 2. ACT: Run both the new and old calculation methods
    potentials_new = engine.calculate_potentials(problem, X, m0, spatial_res=50, sharp=True)
    potentials_old = calculate_potentials_old(X, NMK, a, h, d, m0, heaving)
    
    # 3. ASSERT: Compare the results
    print("\n--- Comparing NEW vs OLD Results ---")
    try:
        # Compare phi, phiH, and phiP, ignoring NaNs used for masking
        np.testing.assert_allclose(potentials_new['phi'], potentials_old['phi'], rtol=1e-8, atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(potentials_new['phiH'], potentials_old['phiH'], rtol=1e-8, atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(potentials_new['phiP'], potentials_old['phiP'], rtol=1e-8, atol=1e-8, equal_nan=True)
        
        print("\n✅ SUCCESS: All potential arrays match perfectly!")
        
    except AssertionError as e:
        print("\n❌ FAILURE: Potential arrays DO NOT match.")
        print("Error details:")
        print(e)
        
        # Optional: Calculate and print the maximum difference for debugging
        diff = np.abs(potentials_new['phi'] - potentials_old['phi'])
        max_diff = np.nanmax(diff)
        print(f"Maximum absolute difference in 'phi': {max_diff}")

if __name__ == "__main__":
    main()