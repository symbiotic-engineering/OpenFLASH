import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import updated package modules ---
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.geometry import ConcentricBodyGroup
from openflash.body import SteppedBody
from openflash.multi_equations import omega
from openflash.multi_constants import g

# --- FIX: Use sys.maxsize for integer "infinity" to satisfy Pylance ---
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision=8, suppress=True)


def main():
    """
    Main function to set up a multi-region MEEM problem, solve it using superposition,
    and visualize the total potential and velocity fields.
    """
    # -------------------------
    # Problem Definition Parameters
    # -------------------------
    NMK = [100, 100, 100, 100]  # Number of harmonics per domain
    h = 100                      # Total water depth
    a = [3, 5, 10]               # Body radii
    d = [29, 7, 4]               # Step depths
    
    # This configuration demands multiple bodies heaving.
    # We must solve this via superposition.
    target_heaving_flags = [False, True, True] 
    
    m0 = 1.0
    
    # -------------------------
    # Superposition Loop
    # -------------------------
    
    # Determine which bodies are active
    active_indices = [i for i, x in enumerate(target_heaving_flags) if x]
    
    # Storage for the total accumulated potential
    phi_total_accumulated = None
    R_grid = None
    Z_grid = None
    
    # We will iterate through each active body, solve, and add to total
    print(f"Target Configuration: {target_heaving_flags}")
    print(f"Active bodies indices: {active_indices}")
    print("-" * 40)

    # Need an engine instance for visualization tools later
    engine_for_plotting = None

    for active_idx in active_indices:
        print(f"\n--- Solving for Single Heaving Body: Index {active_idx} ---")
        
        # 1. Create a "Single Body Heaving" map for this iteration
        current_heaving_flags = [False] * len(target_heaving_flags)
        current_heaving_flags[active_idx] = True
        
        # 2. Re-create Bodies with this specific heaving configuration
        bodies = []
        for i in range(len(a)):
            bodies.append(SteppedBody(
                a=np.array([a[i]]),
                d=np.array([d[i]]),
                slant_angle=np.array([0.0]),
                heaving=current_heaving_flags[i]
            ))

        # 3. Create Geometry & Problem for this iteration
        arrangement = ConcentricBodyGroup(bodies) # Now passes assertion (only 1 True)
        geometry = BasicRegionGeometry(arrangement, h=h, NMK=NMK)
        problem = MEEMProblem(geometry)
        
        # Set Frequency
        problem_frequencies = np.array([omega(m0, h, g)])
        problem.set_frequencies(problem_frequencies)

        # 4. Create Engine & Solve
        engine = MEEMEngine(problem_list=[problem])
        engine_for_plotting = engine # Save reference for plotting later
        
        # Solve linear system
        X = engine.solve_linear_system_multi(problem, m0)
        print(f"  System solved. X shape: {X.shape}")

        # Compute Hydrodynamics (Optional check)
        hydro_coeffs = engine.compute_hydrodynamic_coefficients(problem, X, m0)
        # Just printing the coeff for the active body
        print(f"  Hydro Coeffs (Mode {active_idx}): {hydro_coeffs[active_idx]['real']:.4e} + {hydro_coeffs[active_idx]['imag']:.4e}j")

        # 5. Calculate Potentials
        potentials = engine.calculate_potentials(problem, X, m0, spatial_res=50, sharp=True)
        
        # 6. Accumulate Results (Superposition)
        if phi_total_accumulated is None:
            # First pass: initialize grids and total array
            R_grid = potentials["R"]
            Z_grid = potentials["Z"]
            phi_total_accumulated = potentials["phi"] # Complex array
        else:
            # Subsequent passes: Add to total
            phi_total_accumulated += potentials["phi"]

    # -------------------------
    # Visualization (Total Field)
    # -------------------------
    print("\n" + "="*40)
    print("Visualizing Total Superimposed Field")
    print("="*40)
    
    # --- FIX: Type Guard ---
    # Ensure variables are populated before using them
    if phi_total_accumulated is None or engine_for_plotting is None:
        print("No active bodies were simulated. Nothing to visualize.")
        return

    # Now Pylance knows phi_total_accumulated is not None
    engine_for_plotting.visualize_potential(np.real(phi_total_accumulated), R_grid, Z_grid, "Total Potential (Real) - Superimposed")
    engine_for_plotting.visualize_potential(np.imag(phi_total_accumulated), R_grid, Z_grid, "Total Potential (Imag) - Superimposed")

    plt.show()
    print("Script finished. Close plots to exit.")


if __name__ == "__main__":
    main()