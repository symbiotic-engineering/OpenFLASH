import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'package/src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import core modules ---
try:
    from meem_engine import MEEMEngine
    from meem_problem import MEEMProblem
    from geometry import Geometry
    # Rely on global constants and functions from multi_constants and multi_equations
    # Note: For Streamlit, these values will primarily come from user inputs,
    # but the functions themselves are still needed.
    from multi_equations import (
        Z_n_i, R_1n, R_2n, Lambda_k, phi_p_i, diff_r_phi_p_i, diff_z_phi_p_i,
        diff_R_1n, diff_R_2n, diff_Lambda_k, diff_Z_n_i, diff_Z_k_e
    )
    from equations import Z_n_e 
except ImportError as e:
    st.error(f"Error importing core modules. Make sure 'src' directory is correctly configured and contains all necessary files. Error: {e}")
    st.stop()

# Set print options for better visibility in console (less relevant for Streamlit, but kept for consistency)
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

def plot_field(field, R, Z, title, a_list, d_list, plot_imaginary=True):
    """
    Plots the real and optionally imaginary parts of a field using Matplotlib.
    Returns the matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2 if plot_imaginary else 1, figsize=(16 if plot_imaginary else 8, 7))

    if not plot_imaginary:
        axes = [axes] # Make it a list for consistent indexing

    # Helper to determine levels dynamically
    def get_levels(data):
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if np.isclose(min_val, max_val):
            return [min_val, min_val + 1e-9] if not np.isclose(min_val, 0) else [-1e-9, 1e-9]
        return 50

    # Plot Real Part
    plot_data_real = np.real(field)
    if not np.all(np.isnan(plot_data_real)):
        c_real = axes[0].contourf(R, Z, plot_data_real, levels=get_levels(plot_data_real), cmap='viridis')
        fig.colorbar(c_real, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, "No real data to plot", horizontalalignment='center',
                     verticalalignment='center', transform=axes[0].transAxes)
    
    for r_val in a_list:
        axes[0].axvline(r_val, color='red', linestyle='--', linewidth=1, alpha=0.7)
    for z_val in d_list:
        axes[0].axhline(-z_val, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_title(f'{title} - Real Part')
    axes[0].set_xlabel('Radial Distance (R)')
    axes[0].set_ylabel('Axial Distance (Z)')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Plot Imaginary Part if requested
    if plot_imaginary:
        plot_data_imag = np.imag(field)
        if not np.all(np.isnan(plot_data_imag)):
            c_imag = axes[1].contourf(R, Z, plot_data_imag, levels=get_levels(plot_data_imag), cmap='plasma')
            fig.colorbar(c_imag, ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, "No imaginary data to plot", horizontalalignment='center',
                         verticalalignment='center', transform=axes[1].transAxes)
        
        for r_val in a_list:
            axes[1].axvline(r_val, color='red', linestyle='--', linewidth=1, alpha=0.7)
        for z_val in d_list:
            axes[1].axhline(-z_val, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
        axes[1].set_title(f'{title} - Imaginary Part')
        axes[1].set_xlabel('Radial Distance (R)')
        axes[1].set_ylabel('Axial Distance (Z)')
        axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    return fig

def main():
    st.title("MEEM Multi-Cylinder Simulation")
    st.sidebar.header("Configuration Parameters")

    # --- User Inputs for Parameters ---
    h = st.sidebar.slider("Water Depth (h)", 0.5, 5.0, 1.001, step=0.001)
    
    d_input = st.sidebar.text_input(" (d) - comma-separated for each cylinder", "0.5,0.25")
    a_input = st.sidebar.text_input("Cylinder Radii (a) - comma-separated for each cylinder", "0.5,1.0")
    
    heaving_input = st.sidebar.text_input("Heaving States (1=True, 0=False) - comma-separated for each domain (inner, middle, exterior)", "1,0,0")
    slant_input = st.sidebar.text_input("Slanted States (1=True, 0=False) - comma-separated for each domain", "0,0,0")
    
    NMK_input = st.sidebar.text_input("Number of Harmonics (NMK) - comma-separated for each domain (inner, middle, exterior)", "30,30,30")
    
    m0 = st.sidebar.number_input("Wave Number (m0)", value=1.0)
    omega = st.sidebar.number_input("Angular Frequency (omega)", value=2.0)

    spatial_res = st.sidebar.slider("Spatial Resolution", min_value=20, max_value=150, value=75, step=5)

    st.sidebar.subheader("Plot Options")
    show_homogeneous = st.sidebar.checkbox("Show Homogeneous Potential Plots", value=True)
    show_particular = st.sidebar.checkbox("Show Particular Potential Plots", value=True)
    show_total = st.sidebar.checkbox("Show Total Potential Plots", value=True)
    show_radial_vel = st.sidebar.checkbox("Show Radial Velocity Plots", value=True)
    show_vertical_vel = st.sidebar.checkbox("Show Vertical Velocity Plots", value=True)

    # --- Parse and Validate Inputs ---
    try:
        d_list = list(map(float, d_input.split(',')))
        a_list = list(map(float, a_input.split(',')))
        heaving_list = list(map(int, heaving_input.split(',')))
        slant_list = list(map(int, slant_input.split(','))) # Use int for 0/1 values
        NMK = list(map(int, NMK_input.split(',')))
    except ValueError:
        st.error("Invalid input format. Please use comma-separated numbers.")
        return

    boundary_count = len(a_list) # Number of radial boundaries (cylinders)

    if len(NMK) != (boundary_count + 1):
        st.error(f"Input Error: Number of harmonics (NMK: {len(NMK)}) must be 1 greater than number of cylinder radii (a: {boundary_count}). Please adjust inputs.")
        return
    
    if len(d_list) != boundary_count:
        st.warning(f"Warning: Number of (d: {len(d_list)}) does not match number of cylinders ({boundary_count}). Adapting d_list.")
        if len(d_list) < boundary_count:
            d_list.extend([d_list[-1] if d_list else 0.0] * (boundary_count - len(d_list)))
        else:
            d_list = d_list[:boundary_count]
    
    if len(heaving_list) != len(NMK):
        st.warning(f"Warning: Number of heaving states ({len(heaving_list)}) does not match number of domains ({len(NMK)}). Adapting heaving_list.")
        if len(heaving_list) < len(NMK):
            heaving_list.extend([0] * (len(NMK) - len(heaving_list))) # Default to not heaving
        else:
            heaving_list = heaving_list[:len(NMK)]

    if len(slant_list) != len(NMK):
        st.warning(f"Warning: Number of slanted states ({len(slant_list)}) does not match number of domains ({len(NMK)}). Adapting slant_list.")
        if len(slant_list) < len(NMK):
            slant_list.extend([0] * (len(NMK) - len(slant_list))) # Default to not slanted (0)
        else:
            slant_list = slant_list[:len(NMK)]

    # --- Construct Domain Parameters for Geometry ---
    domain_params = []
    # Inner domain (Region 0)
    params0 = {
        'number_harmonics': NMK[0],
        'height': h,
        'radial_width': a_list[0],
        'di': d_list[0],
        'a': a_list[0],
        'heaving': heaving_list[0],
        'slant': slant_list[0], # Will be 0 or 1
        'category': 'multi',
        'top_BC': None, 'bottom_BC': None
    }
    domain_params.append(params0)

    # Intermediate domains
    for idx in range(1, boundary_count):
        params_mid = {
            'number_harmonics': NMK[idx],
            'height': h,
            'radial_width': a_list[idx],
            'di': d_list[idx],
            'a': a_list[idx],
            'heaving': heaving_list[idx],
            'slant': slant_list[idx], # Will be 0 or 1
            'category': 'multi',
            'top_BC': None, 'bottom_BC': None
        }
        domain_params.append(params_mid)

    # Exterior domain (Last region)
    params_ext = {
        'number_harmonics': NMK[-1],
        'height': h,
        'radial_width': None, # Extends to infinity
        'di': 0,
        'a': a_list[-1], # Inner radius for the exterior domain
        'heaving': heaving_list[-1],
        'slant': slant_list[-1], # Will be 0 or 1
        'category': 'multi',
        'top_BC': None, 'bottom_BC': None
    }
    domain_params.append(params_ext)
    
    with st.expander("View Generated Domain Parameters"):
        st.json(domain_params)

    # Create Geometry object
    # r_coordinates in geometry.py expects a dict like {'a1': val, 'a2': val}
    # Let's create the r_coordinates dict in the format Geometry expects for 'a' values.
    r_coords_for_geometry = {'a': a_list} # Updated based on the provided geometry.py's `r_coordinates` usage
    z_coordinates = {'h': h}

    geometry = Geometry(r_coords_for_geometry, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)

    problem_frequencies = np.array([omega])  
    problem_modes = np.arange(boundary_count) # Modes 0, 1, ... (up to boundary_count-1)

    problem.set_frequencies_modes(problem_frequencies, problem_modes)
    st.write(f"Problem configured with {len(problem.frequencies)} frequency(ies) and {len(problem.modes)} mode(s).")

    # --- MEEM Engine Operations ---
    engine = MEEMEngine(problem_list=[problem])

    # Solve the linear system
    X = engine.solve_linear_system_multi(problem, m0)
    with st.expander("Show Solver Details"):
        st.write(f"The system of equations was successfully solved. The solution vector has a shape of: `{X.shape}`.")

    # Compute and print hydrodynamic coefficients
    hydro_coefficients = engine.compute_hydrodynamic_coefficients(problem, X)
    st.write("Hydrodynamic Coefficients:")
    if hydro_coefficients.get('real') is not None and hydro_coefficients.get('imag') is not None:
        if np.isscalar(hydro_coefficients['real']): # Unlikely if num_modes > 1
            st.metric(label="Added Mass (Real Part)", value=f"{hydro_coefficients['real']:.4f}")
            st.metric(label="Damping (Imaginary Part)", value=f"{hydro_coefficients['imag']:.4f}")
        elif len(hydro_coefficients['real']) > 0: # Check if array is not empty
            st.write("The calculated hydrodynamic coefficients are:")
            df_coeffs = pd.DataFrame({ 
                "Mode Index": problem_modes, # Use actual mode indices/names
                "Added Mass (Real)": hydro_coefficients['real'],
                "Damping (Imaginary)": hydro_coefficients['imag']
            })
            st.dataframe(df_coeffs)
        else:
            st.warning("Hydrodynamic coefficients arrays are empty. Check calculation logic.")
    else:
        st.warning("Hydrodynamic coefficients could not be calculated. `real` or `imag` keys are missing or None.")

    # --- Reformat coefficients using the dedicated MEEMEngine method ---
    reformat_boundary_count = len(a_list) # This is the number of interfaces/cylinders
    Cs = engine.reformat_coeffs(X, NMK, reformat_boundary_count)
    st.write(f"Coefficients reformatted into {len(Cs)} regions.")
    for i, c_region in enumerate(Cs):
        st.write(f"  Region {i} (NMK={NMK[i]}): {c_region.shape} coefficients")

    # Access precomputed arrays from cache
    problem_cache = engine.cache_list[problem]
    m_k_arr = None
    N_k_arr = None

    if problem_cache and hasattr(problem_cache, 'm_k_arr') and problem_cache.m_k_arr is not None:
        m_k_arr = problem_cache.m_k_arr
        st.write(f"m_k_arr shape from cache: {m_k_arr.shape}")
    else:
        st.warning("m_k_arr not found in cache or is None. This might affect calculations for exterior domain.")

    if problem_cache and hasattr(problem_cache, 'N_k_arr') and problem_cache.N_k_arr is not None:
        N_k_arr = problem_cache.N_k_arr
        st.write(f"N_k_arr shape from cache: {N_k_arr.shape}")
    else:
        st.warning("N_k_arr not found in cache or is None. This might affect calculations for exterior domain.")


    # --- Potential and Velocity Field Calculation ---

    # Define potential calculation functions, now passing m_k_arr and N_k_arr
    def phi_h_n_inner_func(n, r, z_val):
        return (Cs[0][n] * R_1n(n, r, 0, h, d_list, a_list)) * Z_n_i(n, z_val, 0, h, d_list)

    def phi_h_m_i_func(i_region_idx, m, r, z_val):
        return (Cs[i_region_idx][m] * R_1n(m, r, i_region_idx, h, d_list, a_list) +
                Cs[i_region_idx][NMK[i_region_idx] + m] * R_2n(m, r, i_region_idx, a_list, h, d_list)) * \
                Z_n_i(m, z_val, i_region_idx, h, d_list)

    def phi_e_k_func(k, r, z_val):
        # Ensure Lambda_k and Z_n_e correctly use m_k_arr, N_k_arr (if needed) and NMK
        if m_k_arr is None or N_k_arr is None:
            st.error("Cannot calculate exterior potential: m_k_arr or N_k_arr is missing.")
            return np.full_like(r, np.nan + np.nan*1j)
        return Cs[-1][k] * Lambda_k(k, r, m0, a_list, NMK, h, m_k_arr, N_k_arr) * \
               Z_n_e(k, z_val, m0, h)
    
    # --- Grid Generation ---
    r_vec = np.linspace(1e-6, 2 * a_list[-1], spatial_res)
    z_vec = np.linspace(-h, 0, spatial_res)

    # Add points at cylinder radii for better resolution
    a_eps = 1.0e-4
    for radius_val in a_list:
        r_vec = np.append(r_vec, radius_val * (1 - a_eps))
        r_vec = np.append(r_vec, radius_val * (1 + a_eps))
    r_vec = np.unique(r_vec)
    r_vec.sort()

    # Add points
    for draft_val in d_list:
        if -draft_val not in z_vec:
            z_vec = np.append(z_vec, -draft_val)
    z_vec = np.unique(z_vec)
    z_vec.sort()

    R, Z = np.meshgrid(r_vec, z_vec)

    # --- Define Spatial Regions ---
    regions = []
    regions.append((R <= a_list[0]) & (Z < -d_list[0])) # Region 0
    for i in range(1, boundary_count):
        regions.append((R > a_list[i-1]) & (R <= a_list[i]) & (Z < -d_list[i])) # Middle regions
    regions.append(R > a_list[-1]) # Last (exterior) region, but without Z condition

    # Initialize potential arrays
    phi = np.full_like(R, np.nan + np.nan*1j, dtype=complex)
    phiH = np.full_like(R, np.nan + np.nan*1j, dtype=complex)
    phiP = np.full_like(R, np.nan + np.nan*1j, dtype=complex)

    # --- Calculate Homogeneous Potentials (phiH) ---
    st.subheader("Calculating Potentials...")
    progress_text = st.empty()

    for n in range(NMK[0]):
        temp_phiH = phi_h_n_inner_func(n, R[regions[0]], Z[regions[0]])

        phiH[regions[0]] = temp_phiH if n == 0 else phiH[regions[0]] + temp_phiH

    for i in range(1, boundary_count):
        for m in range(NMK[i]):
            temp_phiH = phi_h_m_i_func(i, m, R[regions[i]], Z[regions[i]])
            phiH[regions[i]] = temp_phiH if m == 0 else phiH[regions[i]] + temp_phiH

    for k in range(NMK[-1]):
        temp_phiH = phi_e_k_func(k, R[regions[-1]], Z[regions[-1]])
        phiH[regions[-1]] = temp_phiH if k == 0 else phiH[regions[-1]] + temp_phiH
    
    progress_text.text("Homogeneous potential calculation complete.")

    # --- Calculate Particular Potentials (phiP) ---
    phi_p_i_vec = np.vectorize(phi_p_i)

    phiP[regions[0]] = heaving_list[0] * phi_p_i_vec(d_list[0], R[regions[0]], Z[regions[0]], h)
    for i in range(1, boundary_count):
        phiP[regions[i]] = heaving_list[i] * phi_p_i_vec(d_list[i], R[regions[i]], Z[regions[i]], h)
    phiP[regions[-1]] = 0

    # Combine homogeneous and particular solutions for total potential
    phi = phiH + phiP

    # --- Plotting Potentials ---
    st.subheader("Potential Fields")
    if show_homogeneous:
        st.write("#### Homogeneous Potential")
        fig_homo = plot_field(phiH, R, Z, 'Homogeneous Potential', a_list, d_list)
        st.pyplot(fig_homo)

    if show_particular:
        st.write("#### Particular Potential")
        fig_part = plot_field(phiP, R, Z, 'Particular Potential', a_list, d_list)
        st.pyplot(fig_part)

    if show_total:
        st.write("#### Total Potential")
        fig_total = plot_field(phi, R, Z, 'Total Potential', a_list, d_list)
        st.pyplot(fig_total)

    # --- Velocity Field Calculation ---
    st.subheader("Calculating Velocities...")
    progress_text_vel = st.empty()
    
    # Define velocity component functions using reformatted Cs
    def v_r_inner_func(n, r, z_val):
        return (Cs[0][n] * diff_R_1n(n, r, 0, h, d_list, a_list)) * Z_n_i(n, z_val, 0, h, d_list)

    def v_r_m_i_func(i, m, r, z_val):
        return (Cs[i][m] * diff_R_1n(m, r, i, h, d_list, a_list) + Cs[i][NMK[i] + m] * diff_R_2n(m, r, i, h, d_list, a_list)) * Z_n_i(m, z_val, i, h, d_list)

    def v_r_e_k_func(k, r, z_val):
        if m_k_arr is None or N_k_arr is None:
            st.error("Cannot calculate exterior radial velocity: m_k_arr or N_k_arr is missing.")
            return np.full_like(r, np.nan + np.nan*1j)
        return Cs[-1][k] * diff_Lambda_k(k, r, m0, NMK, h, a_list, m_k_arr, N_k_arr) * \
               Z_n_e(k, z_val, m0, h)

    def v_z_inner_func(n, r, z_val):
        return (Cs[0][n] * R_1n(n, r, 0, h, d_list, a_list)) * diff_Z_n_i(n, z_val, 0, h, d_list)

    def v_z_m_i_func(i, m, r, z_val):
        return (Cs[i][m] * R_1n(m, r, i, h, d_list, a_list) + Cs[i][NMK[i] + m] * R_2n(m, r, i, a_list, h, d_list)) * diff_Z_n_i(m, z_val, i, h, d_list)

    def v_z_e_k_func(k, r, z_val):
        if m_k_arr is None or N_k_arr is None:
            st.error("Cannot calculate exterior vertical velocity: m_k_arr or N_k_arr is missing.")
            return np.full_like(r, np.nan + np.nan*1j)
        return Cs[-1][k] * Lambda_k(k, r, m0, a_list, NMK, h, m_k_arr, N_k_arr) * \
               diff_Z_k_e(k, z_val, m0, h, NMK, m_k_arr)

    # Initialize velocity arrays
    vr = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    vrH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    vrP = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 

    vz = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    vzH = np.full_like(R, np.nan + np.nan*1j, dtype=complex) 
    vzP = np.full_like(R, np.nan + np.nan*1j, dtype=complex)

    calculated_so_far_vel = 0
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
        temp_vrH = v_r_e_k_func(k, R[regions[-1]], Z[regions[-1]])
        temp_vzH = v_z_e_k_func(k, R[regions[-1]], Z[regions[-1]])
        if k == 0:
            vrH[regions[-1]] = temp_vrH
            vzH[regions[-1]] = temp_vzH
        else:
            vrH[regions[-1]] += temp_vrH
            vzH[regions[-1]] += temp_vzH

    progress_text_vel.text("Homogeneous velocity calculation complete.")

    # Calculate Particular Velocities (vrP, vzP)
    vr_p_i_vec = np.vectorize(diff_r_phi_p_i)
    vz_p_i_vec = np.vectorize(diff_z_phi_p_i)

    vrP[regions[0]] = heaving_list[0] * vr_p_i_vec(d_list[0], R[regions[0]], Z[regions[0]])
    vzP[regions[0]] = heaving_list[0] * vz_p_i_vec(d_list[0], R[regions[0]], Z[regions[0]])
    for i in range(1, boundary_count):
        vrP[regions[i]] = heaving_list[i] * vr_p_i_vec(d_list[i], R[regions[i]], Z[regions[i]])
        vzP[regions[i]] = heaving_list[i] * vz_p_i_vec(d_list[i], R[regions[i]], Z[regions[i]])
    vrP[regions[-1]] = 0
    vzP[regions[-1]] = 0

    vr = vrH + vrP
    vz = vzH + vzP

    # --- Plotting Velocities ---
    st.subheader("Velocity Fields")
    if show_radial_vel:
        st.write("#### Radial Velocity")
        fig_vr = plot_field(vr, R, Z, 'Radial Velocity', a_list, d_list)
        st.pyplot(fig_vr)

    if show_vertical_vel:
        st.write("#### Vertical Velocity")
        fig_vz = plot_field(vz, R, Z, 'Vertical Velocity', a_list, d_list)
        st.pyplot(fig_vz)

    st.success("Simulation complete!")

if __name__ == "__main__":
    main()