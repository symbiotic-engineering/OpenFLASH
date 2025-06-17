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
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry
# Rely on global constants and functions from multi_constants and multi_equations
from multi_constants import h, d, a, heaving, m0, rho, omega
from multi_equations import (
    Z_n_i, R_1n, R_2n, Lambda_k, phi_p_i, diff_r_phi_p_i, diff_z_phi_p_i,
    diff_R_1n, diff_R_2n, diff_Lambda_k, diff_Z_n_i, diff_Z_k_e
)
from equations import (
    Z_n_e # You imported this from equations, confirming it's Z_n_e, not Z_k_e.
)

# Set print options for better visibility in console
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

def main(): # Renamed from test_main
    """
    Main function to set up a multi-region MEEM problem, solve it,
    and visualize the potential and velocity fields.
    """
    # --- Problem Definition Parameters ---
    NMK = [30, 30, 30] # Number of harmonics for inner, outer, exterior regions
    boundary_count = len(NMK) - 1
    
    # --- Geometry Setup ---
    domain_params = []
    for idx in range(len(NMK)):
        params = {
            'number_harmonics': NMK[idx],
            'height': h - d[idx] if idx < len(d) else h,
            'radial_width': a[idx] if idx < len(a) else a[-1]*1.5,
            'top_BC': None,
            'bottom_BC': None,
            'category': 'multi',  # Adjust category as needed
            'di': d[idx] if idx < len(d) else 0,
            'a': a[idx] if idx < len(a) else a[-1]*1.5,
            'heaving': heaving[idx] if idx < len(heaving) else False,
            'slant': [0, 0, 1]  # Set True if the region is slanted
        }
        domain_params.append(params)

    # Create Geometry object
    r_coordinates = {'a1': a[0], 'a2': a[1], 'a3': a[2]}
    z_coordinates = {'h': h}

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)

    # --- MEEM Engine Operations ---
    engine = MEEMEngine(problem_list=[problem])


    # Assemble A matrix and b vector
    A = engine.assemble_A_multi(problem, m0)
    b = engine.assemble_b_multi(problem, m0)


    problem_cache = engine.cache_list[problem] # Access the existing cache

    if problem_cache is None:
        print("ERROR: problem_cache is None!")
    else:
        print(f"DEBUG: problem_cache.m_k_arr is {problem_cache.m_k_arr.shape if problem_cache.m_k_arr is not None else 'None'}")
        print(f"DEBUG: problem_cache.N_k_arr is {problem_cache.N_k_arr.shape if problem_cache.N_k_arr is not None else 'None'}")

    m_k_arr = problem_cache.m_k_arr
    N_k_arr = problem_cache.N_k_arr

    # Debugging: Check the values of m_k_arr and N_k_arr before passing to plotting functions
    print(f"DEBUG: m_k_arr in main() before plotting functions: {m_k_arr.shape if m_k_arr is not None else 'None'}")
    print(f"DEBUG: N_k_arr in main() before plotting functions: {N_k_arr.shape if N_k_arr is not None else 'None'}")

    # Solve the linear system A x = b
    X = engine.solve_linear_system_multi(problem, m0)
    print(f"System solved. Solution vector X shape: {X.shape}")

    # Compute and print hydrodynamic coefficients
    hydro_coefficients = engine.compute_hydrodynamic_coefficients(problem, X)
    print("\nHydrodynamic Coefficients:")
    print(f"Real part (Added Mass): {hydro_coefficients['real']}")
    print(f"Imaginary part (Damping): {hydro_coefficients['imag']}")

    # --- Reformat coefficients using the dedicated MEEMEngine method ---
    reformat_boundary_count = len(NMK) - 1
    Cs = engine.reformat_coeffs(X, NMK, reformat_boundary_count)
    print(f"\nCoefficients reformatted into {len(Cs)} regions.")
    for i, c_region in enumerate(Cs):
        print(f"  Region {i} (NMK={NMK[i]}): {c_region.shape} coefficients")

    # --- Potential and Velocity Field Calculation ---

    # Define potential calculation functions, now passing m_k_arr and N_k_arr
    def phi_h_n_inner_func(n, r, z):
        # Assuming R_1n and Z_n_i don't need m_k_arr/N_k_arr from multi_equations
        return (Cs[0][n] * R_1n(n, r, 0, h, d, a)) * Z_n_i(n, z, 0, h, d)

    def phi_h_m_i_func(i_region_idx, m, r, z):
        # Assuming R_1n, R_2n, Z_n_i don't need m_k_arr/N_k_arr from multi_equations
        return (Cs[i_region_idx][m] * R_1n(m, r, i_region_idx, h, d, a) +
                Cs[i_region_idx][NMK[i_region_idx] + m] * R_2n(m, r, i_region_idx, a, h, d)) * \
                Z_n_i(m, z, i_region_idx, h, d)

    def phi_e_k_func(k, r, z, m_k_arr, N_k_arr):
        return Cs[-1][k] * Lambda_k(k, r, m0, a, NMK, h, m_k_arr, N_k_arr) * \
               Z_n_e(k, z, m0, h)
    
    spatial_res = 50
    r_vec = np.linspace(2 * a[-1] / spatial_res, 2*a[-1], spatial_res)
    z_vec = np.linspace(-h, 0, spatial_res)

    #add values at the radii
    a_eps = 1.0e-4
    for i in range(len(a)):
        r_vec = np.append(r_vec, a[i]*(1-a_eps))
        r_vec = np.append(r_vec, a[i]*(1+a_eps))
    r_vec = np.unique(r_vec)

    for i in range(len(d)):
        z_vec = np.append(z_vec, -d[i])
    z_vec = np.unique(z_vec)

    R, Z = np.meshgrid(r_vec, z_vec)

    regions = []
    regions.append((R <= a[0]) & (Z < -d[0])) # Region 0
    for i in range(1, boundary_count):
        regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i])) # Middle regions
    regions.append(R > a[-1]) # Last (exterior) region, but without Z condition

    phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex)
    phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex)
    phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex)

    # Calculate homogeneous potentials (phiH)
    for n in range(NMK[0]):
        temp_phiH = phi_h_n_inner_func(n, R[regions[0]], Z[regions[0]])
        phiH[regions[0]] = temp_phiH if n == 0 else phiH[regions[0]] + temp_phiH

    for i in range(1, boundary_count):
        for m in range(NMK[i]):
            temp_phiH = phi_h_m_i_func(i, m, R[regions[i]], Z[regions[i]])
            phiH[regions[i]] = temp_phiH if m == 0 else phiH[regions[i]] + temp_phiH

    for k in range(NMK[-1]):
        temp_phiH = phi_e_k_func(k, R[regions[-1]], Z[regions[-1]], m_k_arr, N_k_arr)
        phiH[regions[-1]] = temp_phiH if k == 0 else phiH[regions[-1]] + temp_phiH

    # Calculate Particular Potentials (phiP)
    phi_p_i_vec = np.vectorize(phi_p_i)
    
    phiP[regions[0]] = heaving[0] * phi_p_i_vec(d[0], R[regions[0]], Z[regions[0]], h)
    for i in range(1, boundary_count):
        phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]], h)
    phiP[regions[-1]] = 0

    phi = phiH + phiP

    # --- Plotting Function ---
    def plot_field(field, R, Z, title): # Removed field_type as it's not used
        plt.figure(figsize=(10, 8))
        
        plt.subplot(1, 2, 1)
        c = plt.contourf(R, Z, np.real(field), levels=50, cmap='viridis')
        plt.colorbar(c)
        for r_val in a:
            plt.axvline(r_val, color='grey', linestyle='--', linewidth=0.8)
        for z_val in d:
            plt.axhline(-z_val, color='grey', linestyle='--', linewidth=0.8)
        plt.title(f'{title} - Real Part')
        plt.xlabel('Radial Distance (R)')
        plt.ylabel('Axial Distance (Z)')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.subplot(1, 2, 2)
        c = plt.contourf(R, Z, np.imag(field), levels=50, cmap='viridis')
        plt.colorbar(c)
        for r_val in a:
            plt.axvline(r_val, color='grey', linestyle='--', linewidth=0.8)
        for z_val in d:
            plt.axhline(-z_val, color='grey', linestyle='--', linewidth=0.8)
        plt.title(f'{title} - Imaginary Part')
        plt.xlabel('Radial Distance (R)')
        plt.ylabel('Axial Distance (Z)')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()

    print("\nPlotting Potentials...")
    plot_field(phiH, R, Z, 'Homogeneous Potential')
    plot_field(phiP, R, Z, 'Particular Potential')
    plot_field(phi, R, Z, 'Total Potential')

    # --- Velocity Field Calculation ---
    # Define velocity component functions using reformatted Cs
    def v_r_inner_func(n, r, z):
        return (Cs[0][n] * diff_R_1n(n, r, 0, h, d, a)) * Z_n_i(n, z, 0, h, d)

    def v_r_m_i_func(i, m, r, z):
        return (Cs[i][m] * diff_R_1n(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * diff_R_2n(m, r, i, h, d, a)) * Z_n_i(m, z, i, h, d)

    def v_r_e_k_func(k, r, z, m_k_arr, N_k_arr):
        # *** FIX: Pass m_k_arr and N_k_arr to diff_Lambda_k and Z_n_e ***
        return Cs[-1][k] * diff_Lambda_k(k, r, m0, NMK, h, a, m_k_arr, N_k_arr) * \
               Z_n_e(k, z, m0, h)

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