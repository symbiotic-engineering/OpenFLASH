import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the necessary classes from  MEEM package
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry, Domain
from results import Results

def main():
    """
    Demonstrates the usage of the MEEMEngine to solve a problem for multiple frequencies,
    performing batch processing of hydrodynamic coefficients and potentials,
    visualizing coefficients, and storing all in a NetCDF file.
    """
    print("--- Starting MEEMEngine Batch Processing, Plotting, and Data Saving Example ---")

    # --- 1. Define Problem Parameters ---
    h_val = 1.001
    d_vals = [0.5, 0.25]
    a_vals = [0.5, 1]
    heaving_vals = [1, 1]
    NMK_vals = [50, 50, 50]
    rho_val = 1023.0

    analysis_frequencies = np.linspace(0.1, 3.0, 50)
    analysis_modes = np.array([1]) 

    print(f"\nProblem Parameters:")
    print(f"  Water Depth (h): {h_val}")
    print(f"  (d): {d_vals}")
    print(f"  Radii (a): {a_vals}")
    print(f"  Harmonics (NMK): {NMK_vals}")
    print(f"  Fluid Density (rho): {rho_val}")
    print(f"  Frequencies to analyze: {analysis_frequencies[0]:.2f} to {analysis_frequencies[-1]:.2f} rad/s ({len(analysis_frequencies)} points)")

    # --- 2. Create Geometry and Domain Objects ---
    domain_params_list = [
        {'number_harmonics': NMK_vals[0], 'height': h_val, 'radial_width': a_vals[0],
         'top_BC': None, 'bottom_BC': None, 'category': 'inner',
         'a': a_vals[0], 'di': d_vals[0], 'heaving': heaving_vals[0]},
        {'number_harmonics': NMK_vals[1], 'height': h_val, 'radial_width': a_vals[1] - a_vals[0],
         'top_BC': None, 'bottom_BC': None, 'category': 'outer',
         'a': a_vals[1], 'di': d_vals[1], 'heaving': heaving_vals[1]},
        {'number_harmonics': NMK_vals[2], 'height': h_val, 'radial_width': None,
         'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
    ]

    r_coords = {'a1': a_vals[0], 'a2': a_vals[1]}
    z_coords = {'h': h_val}

    geometry = Geometry(r_coordinates=r_coords, z_coordinates=z_coords, domain_params=domain_params_list)
    print("\nGeometry object created.")

    # --- 3. Create a MEEMProblem Instance and set multiple frequencies ---
    meem_problem = MEEMProblem(geometry=geometry)
    meem_problem.set_frequencies_modes(analysis_frequencies, analysis_modes)
    print("MEEMProblem instance created and configured with multiple frequencies.")

    # --- 4. Instantiate the MEEMEngine ---
    engine = MEEMEngine(problem_list=[meem_problem])
    print("MEEMEngine instance created. Problem cache built.")

    # --- 5. Batch Process for Each Frequency ---
    collected_added_mass = []
    collected_damping = []
    all_potentials_batch_data = [] # <--- NEW: List to store potential data for all frequencies/modes

    print("\n--- Starting Batch Processing for Multiple Frequencies ---")
    for i, current_m0_val in enumerate(meem_problem.frequencies):
        #  only one mode for now (mode_idx = 0 as analysis_modes is [1])
        mode_idx = 0

        if (i + 1) % 10 == 0 or i == 0 or i == len(meem_problem.frequencies) - 1:
            print(f"Processing frequency {i+1}/{len(meem_problem.frequencies)}: m0 = {current_m0_val:.4f} rad/s")

        A = engine.assemble_A_multi(meem_problem, current_m0_val)
        b = engine.assemble_b_multi(meem_problem, current_m0_val)

        try:
            x_coeffs = np.linalg.solve(A, b)
            # Compute Hydrodynamic Coefficients
            hydro_coeffs = engine.compute_hydrodynamic_coefficients(meem_problem, x_coeffs)
            collected_added_mass.append(hydro_coeffs.get('real', np.nan))
            collected_damping.append(hydro_coeffs.get('imag', np.nan))

            # <--- Calculate Potentials for the current frequency/mode --->
            current_potentials_data = engine.calculate_potentials(meem_problem, x_coeffs)
            
            # Store potentials with their corresponding frequency/mode indices
            formatted_potentials_for_batch = {}
            for domain_name, domain_data in current_potentials_data.items():
                formatted_potentials_for_batch[domain_name] = {
                    'potentials': domain_data['potentials'],
                    'r_coords_dict': domain_data['r'], # Pass the dictionary for r coords
                    'z_coords_dict': domain_data['z']  # Pass the dictionary for z coords
                }

            all_potentials_batch_data.append({
                'frequency_idx': i,
                'mode_idx': mode_idx,
                'data': formatted_potentials_for_batch
            })

        except np.linalg.LinAlgError as e:
            print(f"  ERROR: Could not solve for m0={current_m0_val:.4f}: {e}. Recording NaN for coefficients and skipping potentials.")
            collected_added_mass.append(np.nan)
            collected_damping.append(np.nan)
            continue

    print("\n--- Batch Processing Complete ---")

    # --- 6. Prepare data for Results object and Visualization ---
    frequencies_for_results = np.array(meem_problem.frequencies)
    modes_for_results = np.array(meem_problem.modes)

    num_frequencies = len(frequencies_for_results)
    num_modes = len(modes_for_results)

    added_mass_matrix = np.array(collected_added_mass).reshape(num_frequencies, num_modes)
    damping_matrix = np.array(collected_damping).reshape(num_frequencies, num_modes)

    # --- 7. Instantiate Results and Store Data ---
    results_obj = Results(geometry, frequencies_for_results, modes_for_results)

    # Store hydrodynamic coefficients
    results_obj.store_hydrodynamic_coefficients(
        frequencies=frequencies_for_results,
        modes=modes_for_results,
        added_mass_matrix=added_mass_matrix,
        damping_matrix=damping_matrix
    )
    
    # <--- Store ALL collected Potentials --->
    results_obj.store_all_potentials(all_potentials_batch_data)

    # --- 8. Export Results to NetCDF ---
    output_netcdf_file = "meem_hydro_results_full.nc" 
    results_obj.export_to_netcdf(output_netcdf_file)
    print(f"\nResults successfully exported to {output_netcdf_file}")

    # --- 9. Present Summary of Results (Table - using data from Results object for consistency) ---
    print("\nSummary of Hydrodynamic Coefficients vs. Frequency (from Results object):")
    print(f"{'Frequency (rad/s)':<20} | {'Added Mass':<20} | {'Damping':<20}")
    print("-" * 65)
    
    freqs_display = results_obj.dataset['frequencies'].values
    am_display = results_obj.dataset['added_mass'].values.squeeze()
    damp_display = results_obj.dataset['damping'].values.squeeze()

    for i in range(len(freqs_display)):
        freq_str = f"{freqs_display[i]:.4f}"
        am_str = f"{am_display[i]:.6f}" if not np.isnan(am_display[i]) else "NaN"
        damp_str = f"{damp_display[i]:.6f}" if not np.isnan(damp_display[i]) else "NaN"
        print(f"{freq_str:<20} | {am_str:<20} | {damp_str:<20}")

    # --- 10. Visualize Hydrodynamic Coefficients (Plots) ---
    print("\n--- Plotting Hydrodynamic Coefficients ---")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(frequencies_for_results, added_mass_matrix.squeeze(), 'b-o', markersize=4, label='Added Mass')
    ax1.set_title('Hydrodynamic Coefficients for Heaving Cylinders')
    ax1.set_ylabel('Added Mass ($A_{33}$)')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    ax2.plot(frequencies_for_results, damping_matrix.squeeze(), 'r-o', markersize=4, label='Damping')
    ax2.set_xlabel(r'Angular Frequency ($\omega$, rad/s)')
    ax2.set_ylabel('Damping ($B_{33}$)')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("\n--- MEEMEngine Example Finished ---")

if __name__ == "__main__":
    main()