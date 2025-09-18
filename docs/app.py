import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openflash

# --- Import core MEEM modules ---
try:
    from openflash import *
    from openflash.multi_equations import m_k_newton # Needed to convert omega to m0
except ImportError as e:
    st.error(f"Error importing core modules from openflash. Error: {e}")
    st.stop()

# Set print options for better visibility in console
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

def main():
    st.title("MEEM Multi-Cylinder Simulation")
    st.sidebar.header("Configuration Parameters")

    # --- User Inputs for Parameters ---
    h = st.sidebar.slider("Water Depth (h)", 0.5, 5.0, 1.001, step=0.001)
    
    d_input = st.sidebar.text_input("Drafts (d) - comma-separated", "0.5,0.25")
    a_input = st.sidebar.text_input("Cylinder Radii (a) - comma-separated", "0.5,1.0")
    
    heaving_input = st.sidebar.text_input("Heaving States (1/0) - one per domain", "1,0")
    slant_input = st.sidebar.text_input("Slanted States (1/0) - one per domain", "0,0,0")
    
    NMK_input = st.sidebar.text_input("Number of Harmonics (NMK) - one per domain", "30,30,30")
    
    # --- UI for Single Point Test ---
    st.sidebar.subheader("Single Frequency Test")
    omega_single = st.sidebar.number_input("Angular Frequency (omega)", value=2.0)
    spatial_res = st.sidebar.slider("Plot Spatial Resolution", min_value=20, max_value=150, value=75, step=5)

    # --- UI for Frequency Sweep ---
    st.sidebar.subheader("Frequency Sweep for Coefficients")
    omega_start = st.sidebar.number_input("Start Omega", value=0.1)
    omega_end = st.sidebar.number_input("End Omega", value=4.0)
    omega_steps = st.sidebar.slider("Number of Steps", min_value=10, max_value=200, value=50)


    # --- Parse and Validate Inputs ---
    try:
        d_list = list(map(float, d_input.split(',')))
        a_list = list(map(float, a_input.split(',')))
        heaving_list = list(map(int, heaving_input.split(',')))
        slant_list = list(map(int, slant_input.split(',')))
        NMK = list(map(int, NMK_input.split(',')))
    except ValueError:
        st.error("Invalid input format. Please use comma-separated numbers.")
        st.stop()

    boundary_count = len(NMK) - 1

    # --- Geometry Setup (runs once) ---
    try:
        domain_params = Domain.build_domain_params(NMK, a_list, d_list, heaving_list, h, slant_list)
        r_coords_for_geometry = Domain.build_r_coordinates_dict(a_list)
        z_coordinates = Domain.build_z_coordinates_dict(h)
        geometry = Geometry(r_coords_for_geometry, z_coordinates, domain_params)
        problem = MEEMProblem(geometry)
        problem_modes = np.arange(boundary_count)
        problem.set_frequencies_modes(np.array([]), problem_modes) # Freqs will be set later
    except Exception as e:
        st.error(f"Error during geometry setup: {e}")
        st.stop()
    
    # --- Main Action Buttons ---
    st.header("Run Analysis")
    col1, col2 = st.columns(2)

    if col1.button("Run Single Test & Plot Potentials"):
        st.info(f"Running simulation for single omega = {omega_single:.2f}")
        # --- Convert single omega to m0 ---
        m0_single = m_k_newton(h, omega_single)
        
        # --- MEEM Engine Operations ---
        engine = MEEMEngine(problem_list=[problem])
        X = engine.solve_linear_system_multi(problem, m0_single)
        
        # Display Hydrodynamic Coefficients for the single run
        st.subheader("Hydrodynamic Coefficients (Single Run)")
        hydro_coefficients = engine.compute_hydrodynamic_coefficients(problem, X, m0_single)
        if hydro_coefficients:
            df_coeffs = pd.DataFrame(hydro_coefficients)
            st.dataframe(df_coeffs[['mode', 'real', 'imag', 'nondim_real', 'nondim_imag']])
        else:
            st.warning("Could not calculate hydrodynamic coefficients.")

        # --- Visualize Potentials ---
        st.subheader("Potential Field Plots")
        potentials = engine.calculate_potentials(problem, X, m0_single, spatial_res, sharp=True)
        R, Z, phi = potentials["R"], potentials["Z"], potentials["phi"]
        
        fig1, _ = engine.visualize_potential(np.real(phi), R, Z, "Total Potential (Real)")
        st.pyplot(fig1)

        fig2, _ = engine.visualize_potential(np.imag(phi), R, Z, "Total Potential (Imag)")
        st.pyplot(fig2)
        
        st.success("Single frequency test complete.")

    if col2.button("Run Frequency Sweep & Plot Coefficients"):
        # --- Frequency Sweep Logic ---
        omegas_to_run = np.linspace(omega_start, omega_end, omega_steps)
        results_list = []
        
        engine = MEEMEngine(problem_list=[problem])
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner(f"Running simulation for {omega_steps} frequencies..."):
            for i, omega_val in enumerate(omegas_to_run):
                status_text.text(f"Calculating for omega = {omega_val:.3f}...")
                try:
                    m0_val = m_k_newton(h, omega_val)
                    X = engine.solve_linear_system_multi(problem, m0_val)
                    coeffs = engine.compute_hydrodynamic_coefficients(problem, X, m0_val)
                    
                    # Store results for each mode
                    for c in coeffs:
                        results_list.append({
                            'omega': omega_val,
                            'mode': c['mode'],
                            'added_mass': c['real'],
                            'damping': c['imag']
                        })
                except Exception as e:
                    st.warning(f"Could not solve for omega={omega_val:.3f}. Error: {e}")
                
                progress_bar.progress((i + 1) / omega_steps)

        status_text.text("Calculation complete. Generating plots...")
        progress_bar.empty()

        if not results_list:
            st.error("No data was generated. Cannot create plots.")
            st.stop()
            
        # Convert results to DataFrame for easy plotting
        df_results = pd.DataFrame(results_list)

        # --- Plotting Hydrodynamic Coefficients vs. Frequency ---
        st.subheader("Hydrodynamic Coefficients vs. Frequency")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot Added Mass
        for mode, group in df_results.groupby('mode'):
            ax1.plot(group['omega'], group['added_mass'], label=f'Mode {mode}')
        ax1.set_title('Added Mass vs. Frequency')
        ax1.set_ylabel('Added Mass (Real Part)')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot Damping
        for mode, group in df_results.groupby('mode'):
            ax2.plot(group['omega'], group['damping'], label=f'Mode {mode}')
        ax2.set_title('Damping vs. Frequency')
        ax2.set_ylabel('Damping (Imaginary Part)')
        ax2.set_xlabel('Angular Frequency (omega)')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.expander("View Raw Data"):
            st.dataframe(df_results)


if __name__ == "__main__":
    # Wrap main execution in a try-catch to handle potential errors gracefully
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")