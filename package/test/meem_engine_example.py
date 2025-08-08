import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- Path Setup ---
# This ensures the script can find your package source files.
# Adjust the path if your project structure is different.
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import Your Package Classes ---
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry
from openflash.domain import Domain
from openflash.multi_equations import omega
from openflash.multi_constants import g

def main():
    """
    An example script demonstrating the end-to-end workflow for using the MEEMEngine.
    """
    print("--- MEEMEngine Full Workflow Example ---")

    # ==========================================================================
    # STEP 1: Define the Physical System
    # ==========================================================================
    # Describe the system with simple lists: radii, drafts, etc.
    # This example defines a system with two cylinders.

    NMK = [30, 30, 30]  # Harmonics for inner, outer, and exterior domains
    h = 100.0           # Total water depth
    a = [5.0, 10.0]     # Radii of the two cylinders
    d = [20.0, 10.0]    # Drafts of the two cylinders
    heaving = [1, 0]    # The first cylinder is heaving, the second is fixed

    # ==========================================================================
    # STEP 2: Create the Problem Definition
    # ==========================================================================
    # Use the helper methods to structure the data and create the problem object.

    print("\n[1] Setting up the geometry and problem definition...")
    
    # Use helper methods to create the required data structures
    domain_params = Domain.build_domain_params(NMK, a, d, heaving, h)
    r_coords = Domain.build_r_coordinates_dict(a)
    z_coords = Domain.build_z_coordinates_dict(h)
    
    # Create the Geometry and MEEMProblem objects
    geometry = Geometry(r_coords, z_coords, domain_params)
    problem = MEEMProblem(geometry)
    
    # Define the frequencies (as m0 values) and modes to solve for
    m0 = 1.0
    local_omega = omega(m0,h,g)
    problem_frequencies = np.array([local_omega])
    boundary_count = len(NMK) -1
    problem_modes = np.arange(boundary_count) # Modes 0, 1, ... up to boundary_count-1 (heaving modes)
    problem.set_frequencies_modes(problem_frequencies, problem_modes)

    # ==========================================================================
    # STEP 3: Instantiate and Run the MEEMEngine
    # ==========================================================================
    # The engine is the main workhorse. It takes a list of problems to manage.

    print("[2] Initializing the MEEMEngine...")
    engine = MEEMEngine(problem_list=[problem])

    # The `run_and_store_results` method iterates through all frequencies,
    # solves the system, and stores the hydrodynamic coefficients.
    print("[3] Running solver for all frequencies...")
    results = engine.run_and_store_results(problem_index=0)
    print("    ...Done.")

    # You can display the stored hydrodynamic coefficients
    print("\n--- Hydrodynamic Coefficients (Added Mass) ---")
    print(results.dataset['added_mass'].to_pandas())


    # ==========================================================================
    # STEP 4: Calculate and Visualize Detailed Fields
    # ==========================================================================
    # After solving, you can calculate the full potential or velocity fields
    # for any single frequency.

    # Let's pick a single frequency to visualize
    m0_for_viz = 1.0
    
    print(f"\n[4] Solving for a single frequency (m0={m0_for_viz}) to get the solution vector X...")
    X = engine.solve_linear_system_multi(problem, m0=m0_for_viz)

    print("[5] Calculating detailed potential and velocity fields...")
    potentials = engine.calculate_potentials(problem, X, m0=m0_for_viz, spatial_res=50, sharp=True)
    velocities = engine.calculate_velocities(problem, X, m0=m0_for_viz, spatial_res=50, sharp=True)
    print("    ...Done.")
    
    # ==========================================================================
    # STEP 5: Visualize the Results
    # ==========================================================================
    # Use the `visualize_potential` method to plot the results.

    print("[6] Generating plots...")

    # Extract the meshgrid and fields from the results dictionaries
    R = potentials["R"]
    Z = potentials["Z"]
    total_potential = potentials["phi"]
    radial_velocity = velocities["vr"]

    # Visualize the real part of the total potential
    engine.visualize_potential(
        field=np.real(total_potential),
        R=R,
        Z=Z,
        title=f"Real Part of Total Potential (phi) at m0={m0_for_viz}"
    )

    # Visualize the magnitude of the radial velocity
    engine.visualize_potential(
        field=np.abs(radial_velocity),
        R=R,
        Z=Z,
        title=f"Magnitude of Radial Velocity (vr) at m0={m0_for_viz}"
    )
    
    plt.show()
    
    print("\n--- Example Finished ---")


if __name__ == "__main__":
    main()