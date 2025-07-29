import logging
import sys
import os
import numpy as np
import pandas as pd # Can be removed if not used
from scipy import linalg # Can be removed if not used (only solve is used, not linalg directly)
from scipy.integrate import quad # Can be removed if not used
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import core modules ---
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry
from openflash.multi_equations import *
from openflash.multi_constants import g
from openflash.domain import Domain

# Set print options for better visibility in console
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

def main(): # Renamed from test_main
    """
    Main function to set up a multi-region MEEM problem, solve it,
    and visualize the potential and velocity fields.
    """
    # --- Problem Definition Parameters ---
    # get an error when NMK[idx] > 89
    NMK = [100, 100, 100, 100] # Number of terms in approximation of each region (including e).
    boundary_count = len(NMK) - 1
    h = 100
    d = [29, 7, 4]
    a = [3, 5, 10]
    heaving = [0, 1, 1]
    # 0/false if not heaving, 1/true if yes heaving
    # All computations assume at least 2 regions.

    # --- Geometry Setup ---
    domain_params = Domain.build_domain_params(NMK, a, d, heaving, h)
    
    a_recovered = [p['a'] for p in domain_params[:-1]]
    d_recovered = [p['di'] for p in domain_params[:-1]]
    heaving_recovered = [p['heaving'] for p in domain_params[:-1]]
    h_recovered = domain_params[0]['height']

    assert a_recovered == a
    assert d_recovered == d
    assert heaving_recovered == heaving
    assert h_recovered == h

    # Create Geometry object
    r_coordinates = Domain.build_r_coordinates_dict(a)
    z_coordinates = Domain.build_z_coordinates_dict(h)

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)
    m0 = 1
        
    # --- MEEM Engine Operations ---
    engine = MEEMEngine(problem_list=[problem])

    problem_cache = engine.cache_list[problem] # Access the existing cache

    if problem_cache is None:
        logging.error("Problem cache is None! Cannot proceed with computations.")
        raise RuntimeError("Problem cache is None! Cannot proceed with computations.")
    else:
        logging.debug(f"problem_cache.m_k_arr shape: {problem_cache.m_k_arr.shape if problem_cache.m_k_arr is not None else 'None'}")
        logging.debug(f"problem_cache.N_k_arr shape: {problem_cache.N_k_arr.shape if problem_cache.N_k_arr is not None else 'None'}")

    m_k_arr = problem_cache.m_k_arr
    N_k_arr = problem_cache.N_k_arr

    # Debugging: Check the values of m_k_arr and N_k_arr before passing to plotting functions
    print(f"DEBUG: m_k_arr in main() before plotting functions: {m_k_arr.shape if m_k_arr is not None else 'None'}")
    print(f"DEBUG: N_k_arr in main() before plotting functions: {N_k_arr.shape if N_k_arr is not None else 'None'}")

    # Solve the linear system A x = b
    X = engine.solve_linear_system_multi(problem, m0)
    print(f"System solved. Solution vector X shape: {X.shape}")

    # Compute and print hydrodynamic coefficients
    hydro_coefficients = engine.compute_hydrodynamic_coefficients(problem, X, m0)
    print(type(hydro_coefficients["real"]), type(hydro_coefficients["imag"]))

    real = hydro_coefficients["real"]
    imag = hydro_coefficients["imag"]
    nondim_real = hydro_coefficients["nondim_real"]
    nondim_imag = hydro_coefficients["nondim_imag"]
    excitation_phase = hydro_coefficients["excitation_phase"]
    excitation_force = hydro_coefficients["excitation_force"]

    print("\n--- Hydrodynamic Coefficient Breakdown ---")
    print(f"real (added mass): {real:.10f}")
    print(f"imag (damping): {imag:.10f}")
    print(f"real/(h^3): {real / h**3:.10f}")
    print(f"imag/(h^3): {imag / h**3:.10f}")
    print(f"nondimensional, real: {nondim_real:.10f}")
    print(f"nondimensional, imag (no omega factor): {nondim_imag:.10f}")
    print(f"Excitation Phase: {excitation_phase:.10f} radians")
    print(f"Excitation Force: {excitation_force:.10f} N")
    # --- Reformat coefficients using the dedicated MEEMEngine method ---
    reformat_boundary_count = len(NMK) - 1
    Cs = engine.reformat_coeffs(X, NMK, reformat_boundary_count)
    print(f"\nCoefficients reformatted into {len(Cs)} regions.")
    for i, c_region in enumerate(Cs):
        print(f"  Region {i} (NMK={NMK[i]}): {c_region.shape} coefficients")

    # --- Potential and Velocity Field Calculation ---
    
    # --- Use MEEMEngine to calculate potentials ---
    potentials = engine.calculate_potentials(problem, X, m0, spatial_res=50, sharp=True)

    # Unpack
    R = potentials["R"]
    Z = potentials["Z"]
    phiH = potentials["phiH"]
    phiP = potentials["phiP"]
    phi = potentials["phi"]

    # --- Plot using built-in visualizer ---
    engine.visualize_potential(np.real(phiH), R, Z, "Homogeneous Potential (Real)")
    engine.visualize_potential(np.imag(phiH), R, Z, "Homogeneous Potential (Imag)")
    engine.visualize_potential(np.real(phiP), R, Z, "Particular Potential (Real)")
    engine.visualize_potential(np.imag(phiP), R, Z, "Particular Potential (Imag)")
    engine.visualize_potential(np.real(phi), R, Z, "Total Potential (Real)")
    engine.visualize_potential(np.imag(phi), R, Z, "Total Potential (Imag)")

    # --- Velocity Field Calculation ---
    # Define velocity component functions using reformatted Cs
    def v_r_inner_func(n, r, z):
        return (Cs[0][n] * diff_R_1n(n, r, 0, h, d, a)) * Z_n_i(n, z, 0, h, d)

    def v_r_m_i_func(i, m, r, z):
        return (Cs[i][m] * diff_R_1n(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * diff_R_2n(m, r, i, h, d, a)) * Z_n_i(m, z, i, h, d)

    def v_r_e_k_func(k, r, z, m_k_arr, N_k_arr):
        # *** FIX: Pass m_k_arr and N_k_arr to diff_Lambda_k and Z_n_e ***
        return Cs[-1][k] * diff_Lambda_k(k, r, m0, NMK, h, a, m_k_arr, N_k_arr) * \
               Z_k_e(k, z, m0, h, NMK, m_k_arr)

    def v_z_inner_func(n, r, z):
        return (Cs[0][n] * R_1n(n, r, 0, h, d, a)) * diff_Z_n_i(n, z, 0, h, d)

    def v_z_m_i_func(i, m, r, z):
        return (Cs[i][m] * R_1n(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n(m, r, i, a, h, d)) * diff_Z_n_i(m, z, i, h, d)

    def v_z_e_k_func(k, r, z, m_k_arr, N_k_arr):
        # *** FIX: Pass m_k_arr and N_k_arr to Lambda_k and diff_Z_k_e ***
        return Cs[-1][k] * Lambda_k(k, r, m0, a, NMK, h, m_k_arr, N_k_arr) * \
               diff_Z_k_e(k, z, m0, h, NMK, m_k_arr)

    # Initialize velocity arrays
    vr = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    vrH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    vrP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

    vz = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    vzH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    vzP = np.full_like(R, np.nan + np.nan*1j, dtype=complex)

    # Calculate homogeneous velocities (vrH, vzH)
    for n in range(NMK[0]):
        temp_vrH = v_r_inner_func(n, R[regions[0]], Z[regions[0]])
        temp_vzH = v_z_inner_func(n, R[regions[0]], Z[regions[0]])
        if n == 0:
            vrH[regions[0]] = temp_vrH
            vzH[regions[0]] = temp_vzH
        else:
            vrH[regions[0]] = vrH[regions[0]] + temp_vrH
            vzH[regions[0]] = vzH[regions[0]] + temp_vzH

    for i in range(1, boundary_count):
        for m in range(NMK[i]):
            temp_vrH = v_r_m_i_func(i, m, R[regions[i]], Z[regions[i]])
            temp_vzH = v_z_m_i_func(i, m, R[regions[i]], Z[regions[i]])
            if m == 0:
                vrH[regions[i]] = temp_vrH
                vzH[regions[i]] = temp_vzH
            else:
                vrH[regions[i]] = vrH[regions[i]] + temp_vrH
                vzH[regions[i]] = vzH[regions[i]] + temp_vzH

    for k in range(NMK[-1]):
        temp_vrH = v_r_e_k_func(k, R[regions[-1]], Z[regions[-1]], m_k_arr, N_k_arr)
        temp_vzH = v_z_e_k_func(k, R[regions[-1]], Z[regions[-1]], m_k_arr, N_k_arr)
        if k == 0:
            vrH[regions[-1]] = temp_vrH
            vzH[regions[-1]] = temp_vzH
        else:
            vrH[regions[-1]] += temp_vrH
            vzH[regions[-1]] += temp_vzH

    # Calculate Particular Velocities (vrP, vzP)
    vr_p_i_vec = np.vectorize(diff_r_phi_p_i)
    vz_p_i_vec = np.vectorize(diff_z_phi_p_i)

    vrP[regions[0]] = heaving[0] * vr_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
    vzP[regions[0]] = heaving[0] * vz_p_i_vec(d[0], R[regions[0]], Z[regions[0]])
    for i in range(1, boundary_count):
        vrP[regions[i]] = heaving[i] * vr_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
        vzP[regions[i]] = heaving[i] * vz_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
    vrP[regions[-1]] = 0
    vzP[regions[-1]] = 0

    vr = vrH + vrP
    vz = vzH + vzP

    print("\nPlotting Velocities...")
    plot_field(vr, R, Z, 'Radial Velocity')
    plot_field(vz, R, Z, 'Vertical Velocity')

    print("\nScript finished. Close plot windows to exit.")

if __name__ == "__main__":
    main()