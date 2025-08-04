import numpy as np
import sys
import os

# --- Path Setup ---
# This ensures the script can find your package files.
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import Your Package Classes ---
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry
from openflash.domain import Domain

# ==============================================================================
#  Purpose of the MEEMProblem Class
# ==============================================================================
#
# The `MEEMProblem` class is a data container that bundles everything needed
# to define a complete hydrodynamic analysis. It holds two key pieces of information:
#
# 1.  **The Physical System**: This is stored as a `Geometry` object, which
#     describes the physical layout of cylinders, water depth, etc.
# 2.  **The Computation Targets**: These are the specific frequencies and
#     modes of motion (e.g., heave, surge) you want to analyze for that geometry.
#
# Essentially, a `MEEMProblem` object is a complete "job description" that you
# can hand off to the `MEEMEngine` to be solved.

def main():
    """
    An example script demonstrating how to create and configure a MEEMProblem.
    """
    print("--- MEEMProblem Class Example ---")

    # ==========================================================================
    # STEP 1: Create a Geometry Object (Prerequisite)
    # ==========================================================================
    #
    # A MEEMProblem cannot exist without a Geometry. First, we must define
    # the physical layout of the system.
    
    print("\n[1] Defining the physical geometry...")
    NMK = [10, 10, 10]
    h = 100.0
    a = [5.0, 10.0]
    d = [20.0, 10.0]
    heaving = [1, 0]

    domain_params = Domain.build_domain_params(NMK, a, d, heaving, h)
    r_coords = Domain.build_r_coordinates_dict(a)
    z_coords = Domain.build_z_coordinates_dict(h)
    
    geometry = Geometry(r_coords, z_coords, domain_params)
    print("    ...Geometry object created.")

    # ==========================================================================
    # STEP 2: Create and Configure the MEEMProblem
    # ==========================================================================
    
    # --- Part A: Initialization ---
    # A MEEMProblem is initialized with the geometry from Step 1.
    
    print("\n[2] Initializing the MEEMProblem...")
    problem = MEEMProblem(geometry)
    
    print("    Initial state of the problem:")
    print(f"      Frequencies: {problem.frequencies} (empty by default)")
    print(f"      Modes: {problem.modes} (empty by default)")

    # --- Part B: Setting Computation Targets ---
    # Now, we use the `set_frequencies_modes` method to tell the problem
    # what we want the MEEMEngine to compute.
    
    print("\n[3] Setting frequencies and modes of motion...")
    
    # Define the frequencies and modes to be analyzed
    frequencies_to_run = np.linspace(0.1, 2.0, 5)
    modes_to_run = np.array([1, 2, 3]) # e.g., representing Heave, Surge, Pitch

    # Set them on the problem object
    problem.set_frequencies_modes(frequencies=frequencies_to_run, modes=modes_to_run)
    
    print("    State after configuration:")
    print(f"      Frequencies: {problem.frequencies}")
    print(f"      Modes: {problem.modes}")

    # ==========================================================================
    # STEP 3: Using the MEEMProblem
    # ==========================================================================
    #
    # The fully configured `problem` object is now ready to be passed to the
    # MEEMEngine to perform calculations, such as in `engine.run_and_store_results()`.

    print("\n[4] The MEEMProblem object is now fully configured and ready for the solver.")
    print("\n--- Example Finished ---")


if __name__ == "__main__":
    main()