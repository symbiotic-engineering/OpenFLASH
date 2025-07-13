import numpy as np
import sys
import os

# --- Path Setup ---
# Adjust the Python path to allow importing modules from the 'src' directory.
#  this example file is in 'semi-analytical-hydro/package/test/'
# and the MEEM package source is in 'semi-analytical-hydro/package/src/'.
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src')) # Navigates from 'test' up to 'package', then into 'src'
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the necessary classes from  MEEM package
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry, Domain # Domain is used internally by Geometry with domain_params

def main():
    """
    Demonstrates the usage of the MEEMProblem class.
    It sets up geometry and assigns frequencies/modes.
    """
    print("--- Starting MEEMProblem Example ---")

    # --- 1. Define Problem Parameters (Using 'config1' parameters for consistency) ---
    h_val = 1.001  # Water depth
    d_vals = [0.5, 0.25]  # for inner and outer bodies
    a_vals = [0.5, 1]     # Radii for inner and outer bodies
    heaving_vals = [1, 1] # Heaving flags for inner and outer bodies (1 for heaving, 0 for fixed)
    NMK_vals = [50, 50, 50] # Number of harmonics for inner, outer, and exterior domains

    # Frequencies and modes relevant to the MEEMProblem itself
    analysis_frequencies = np.array([1.0, 1.5, 2.0]) # Example: analyze at 3 different frequencies
    analysis_modes = np.array([1]) 

    print(f"\nProblem Parameters for Setup:")
    print(f"  Water Depth (h): {h_val}")
    print(f"  (d): {d_vals}")
    print(f"  Radii (a): {a_vals}")
    print(f"  Harmonics (NMK): {NMK_vals}")
    print(f"  Analysis Frequencies: {analysis_frequencies}")
    print(f"  Analysis Modes: {analysis_modes}")

    # --- 2. Prepare Domain Parameters for Geometry ---
    # The `Geometry` class will create `Domain` objects internally based on this list.
    domain_params_list = [
        # Inner domain (index 0)
        {'number_harmonics': NMK_vals[0], 'height': h_val, 'radial_width': a_vals[0],
         'top_BC': None, 'bottom_BC': None, 'category': 'inner',
         'a': a_vals[0], 'di': d_vals[0], 'heaving': heaving_vals[0]},
        # Outer domain (index 1) - Annular region between a[0] and a[1]
        {'number_harmonics': NMK_vals[1], 'height': h_val, 'radial_width': a_vals[1] - a_vals[0],
         'top_BC': None, 'bottom_BC': None, 'category': 'outer',
         'a': a_vals[1], 'di': d_vals[1], 'heaving': heaving_vals[1]},
        # Exterior domain (index 2) - From a[1] to infinity
        {'number_harmonics': NMK_vals[2], 'height': h_val, 'radial_width': None, # radial_width not applicable for exterior
         'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
    ]

    # Radial and Z coordinates information for Geometry
    r_coords = {'a1': a_vals[0], 'a2': a_vals[1]}
    z_coords = {'h': h_val}

    # --- 3. Create a Geometry Object ---
    geometry = Geometry(r_coordinates=r_coords, z_coordinates=z_coords, domain_params=domain_params_list)
    print("\nGeometry object created and domains are defined within it.")

    # --- 4. Instantiate MEEMProblem ---
    # The MEEMProblem object takes the configured Geometry.
    meem_problem = MEEMProblem(geometry=geometry)
    print("\nMEEMProblem instance created.")

    # --- 5. Set Frequencies and Modes for the Problem ---
    meem_problem.set_frequencies_modes(analysis_frequencies, analysis_modes)
    print("Frequencies and modes set for the MEEMProblem.")

    # --- 6. Access and Verify Problem Attributes ---
    print(f"\nVerifying MEEMProblem Attributes:")
    print(f"  Geometry object reference: {meem_problem.geometry}")
    print(f"  Number of domains in problem: {len(meem_problem.domain_list)}")
    print(f"  Problem Frequencies: {meem_problem.frequencies}")
    print(f"  Problem Modes: {meem_problem.modes}")

    # Example: Accessing details of the first domain
    first_domain_key = list(meem_problem.domain_list.keys())[0]
    first_domain = meem_problem.domain_list[first_domain_key]
    print(f"  First Domain (Category: {first_domain.category}) Details:")
    print(f"    Number of Harmonics: {first_domain.number_harmonics}")
    print(f"    Height (h): {first_domain.h}")
    print(f"    (di): {first_domain.di}")
    print(f"    Radius (a): {first_domain.a}")


    print("\n--- MEEMProblem Example Finished ---")

if __name__ == "__main__":
    main()