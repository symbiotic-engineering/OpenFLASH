import streamlit as st
import numpy as np
import pandas as pd
import openflash

# --- Import core MEEM modules ---
try:
    from openflash import *
except ImportError as e:
    st.error(f"Error importing core modules from openflash. Error: {e}")
    st.stop()

# Set print options for better visibility in console (less relevant for Streamlit, but kept for consistency)
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

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
        slant_list = list(map(int, slant_input.split(',')))  # Use int for 0/1 values
        NMK = list(map(int, NMK_input.split(',')))
    except ValueError:
        st.error("Invalid input format. Please use comma-separated numbers.")
        return

    boundary_count = len(NMK) - 1  # Number of radial boundaries (cylinders)

    # --- Validate Length Consistency ---
    if len(a_list) != boundary_count:
        st.error(f"Expected {boundary_count} entries in cylinder radii (a), but got {len(a_list)}.")
        return

    if len(d_list) != boundary_count:
        st.error(f"Expected {boundary_count} entries in draft values (d), but got {len(d_list)}.")
        return

    if len(heaving_list) != len(NMK):
        st.error(f"Expected {len(NMK)} heaving states (1/0), one per domain, but got {len(heaving_list)}.")
        return

    if len(slant_list) != len(NMK):
        st.error(f"Expected {len(NMK)} slant states (1/0), one per domain, but got {len(slant_list)}.")
        return

    # --- Validate Value Ranges and Rules ---
    if not all(entry in [0, 1] for entry in heaving_list):
        st.error("Heaving entries must be either 0 (False) or 1 (True).")
        return

    if not all(entry in [0, 1] for entry in slant_list):
        st.error("Slant entries must be either 0 (False) or 1 (True).")
        return

    if not all(isinstance(val, int) and val > 0 for val in NMK):
        st.error("All NMK values must be positive integers.")
        return

    if m0 <= 0:
        st.error("Wave number (m0) must be positive.")
        return

    if not all(depth >= 0 for depth in d_list):
        st.error("All draft values (d) must be nonnegative.")
        return

    if not all(depth < h for depth in d_list):
        st.error("All draft values (d) must be less than the water depth (h).")
        return

    if not all(a_list[i] > a_list[i - 1] for i in range(1, len(a_list))):
        st.error("Cylinder radii (a) must be strictly increasing.")
        return

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
    reformat_boundary_count = len(a_list) # This is the number of regions
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

   
    # Modal Potentials
    st.subheader("Modal Potential Magnitudes")
    potentials = engine.calculate_potentials(problem, X, m0, m_k_arr, N_k_arr, spatial_res, sharp=True)
    fig = engine.visualize_potential(potentials)
    st.pyplot(fig)

    # --- Velocity Field Calculation ---
    st.subheader("Calculating Velocities...")

    st.success(f"Simulation complete with {len(domain_params)} domains configured.")


if __name__ == "__main__":
    main()