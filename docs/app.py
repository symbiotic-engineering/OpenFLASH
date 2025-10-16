import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Import core MEEM modules ---
try:
    from openflash import *
    from openflash.multi_equations import m_k_newton, wavenumber # Needed to convert omega to m0
except ImportError as e:
    st.error(f"Error importing core modules from openflash. Error: {e}")
    st.stop()

# Set print options for better visibility in console
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

def main():
    st.title("OpenFLASH: MEEM Multi-Cylinder Simulation")
    st.sidebar.header("Configuration Parameters")

    # --- User Inputs for Parameters ---
    h = st.sidebar.slider("Water Depth (h)", 0.5, 5.0, 1.001, step=0.001)
    
    d_input = st.sidebar.text_input("Body Step Depths (d) - comma-separated", "0.5,0.25")
    a_input = st.sidebar.text_input("Body Radii (a) - comma-separated", "0.5,1.0")
    heaving_input = st.sidebar.text_input("Heaving Bodies (1=True/0=False) - one per body", "1,0")
    NMK_input = st.sidebar.text_input("Harmonics (NMK) - one per domain", "30,30,30")
    
    # --- UI for Single Point Test ---
    st.sidebar.subheader("Single Frequency Test")
    omega_single = st.sidebar.number_input("Angular Frequency (omega)", value=2.0, format="%.3f")
    spatial_res = st.sidebar.slider("Plot Spatial Resolution", min_value=20, max_value=150, value=75, step=5)

    # --- UI for Frequency Sweep ---
    st.sidebar.subheader("Frequency Sweep for Coefficients")
    omega_start = st.sidebar.number_input("Start Omega", value=0.1, format="%.3f")
    omega_end = st.sidebar.number_input("End Omega", value=4.0, format="%.3f")
    omega_steps = st.sidebar.slider("Number of Steps", min_value=10, max_value=200, value=50)


    # --- Parse and Validate Inputs ---
    try:
        d_list = np.array(list(map(float, d_input.split(','))))
        a_list = np.array(list(map(float, a_input.split(','))))
        heaving_list = np.array(list(map(bool, heaving_input.split(','))))
        NMK = list(map(int, NMK_input.split(',')))
        
        # Validation
        if not (len(d_list) == len(a_list) == len(heaving_list)):
            st.error("The number of depths, radii, and heaving flags must be the same (one for each body).")
            st.stop()
        if len(NMK) != len(a_list) + 1:
            st.error("The number of NMK values must be one greater than the number of bodies (one for each domain).")
            st.stop()
    except ValueError:
        st.error("Invalid input format. Please use comma-separated numbers.")
        st.stop()

    # --- Modern, Object-Oriented Geometry and Problem Setup ---
    try:
        # 1. Create Body objects
        bodies = [
            SteppedBody(a=np.array([a_val]), d=np.array([d_val]), slant_angle=np.array([0.0]), heaving=h_flag)
            for a_val, d_val, h_flag in zip(a_list, d_list, heaving_list)
        ]
        # 2. Create Arrangement and Geometry
        arrangement = ConcentricBodyGroup(bodies)
        geometry = BasicRegionGeometry(arrangement, h=h, NMK=NMK)
        # 3. Create the Problem
        problem = MEEMProblem(geometry)
        
    except Exception as e:
        st.error(f"Error during geometry setup: {e}")
        st.stop()
    
    # --- Main Action Buttons ---
    st.header("Run Analysis")
    col1, col2 = st.columns(2)

    if col1.button("Run Single Test & Plot Potentials"):
        st.info(f"Running simulation for single omega = {omega_single:.2f}")
        # --- Convert single omega to m0 ---
        m0_single = wavenumber(omega_single, h)
        problem_modes = np.where(heaving_list)[0]
        problem.set_frequencies_modes(np.array([omega_single]), problem_modes)
        
        # --- MEEM Engine Operations ---
        engine = MEEMEngine(problem_list=[problem])
        X = engine.solve_linear_system_multi(problem, m0_single)
        
        # Display Hydrodynamic Coefficients for the single run
        st.subheader("Hydrodynamic Coefficients (Single Run)")
        hydro_coefficients = engine.compute_hydrodynamic_coefficients(problem, X, m0_single)
        if hydro_coefficients:
            df_coeffs = pd.DataFrame(hydro_coefficients)
            st.dataframe(df_coeffs[['mode_i', 'mode_j', 'real', 'imag']])
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
        st.info(f"Running frequency sweep for {omega_steps} steps...")
        ## --- FIX: Use the efficient `run_and_store_results` method ---
        omegas_to_run = np.linspace(omega_start, omega_end, omega_steps)
        problem_modes = np.where(heaving_list)[0]
        problem.set_frequencies_modes(omegas_to_run, problem_modes)
        
        engine = MEEMEngine(problem_list=[problem])
        with st.spinner("Running simulation..."):
            # This single call runs the entire sweep
            results_obj = engine.run_and_store_results(problem_index=0)
        st.success("Frequency sweep complete.")
        # Extract the dataset and convert to a DataFrame for plotting
        dataset = results_obj.get_results()
        df_results = dataset[['added_mass', 'damping']].to_dataframe().reset_index()

        # --- Plotting Hydrodynamic Coefficients vs. Frequency ---
        st.subheader("Hydrodynamic Coefficients vs. Frequency")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot Added Mass
        for (mode_i, mode_j), group in df_results.groupby(['mode_i', 'mode_j']):
            ax1.plot(group['frequencies'], group['added_mass'], label=f'A({mode_i},{mode_j})')
        ax1.set_title('Added Mass vs. Frequency')
        ax1.set_ylabel('Added Mass')
        ax1.legend()
        ax1.grid(True, linestyle='--')

        # Plot Damping
        for (mode_i, mode_j), group in df_results.groupby(['mode_i', 'mode_j']):
            ax2.plot(group['frequencies'], group['damping'], label=f'B({mode_i},{mode_j})')
        ax2.set_title('Damping vs. Frequency')
        ax2.set_ylabel('Damping')
        ax2.set_xlabel('Angular Frequency (omega)')
        ax2.legend()
        ax2.grid(True, linestyle='--')
        
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