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

# Set print options for better visibility
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)


def main():
    """
    Main function to set up a multi-region MEEM problem, solve it,
    and visualize the potential and velocity fields.
    """
    # -------------------------
    # Problem Definition Parameters
    # -------------------------
    NMK = [100, 100, 100, 100]  # Number of harmonics per domain
    h = 100                      # Total water depth
    a = [3, 5, 10]               # Body radii
    d = [29, 7, 4]               # Step depths
    heaving_flags = [False, True, True]  # Heaving flags per body

    # -------------------------
    # Create SteppedBody objects
    # -------------------------
    bodies = []
    for i in range(len(a)):
        bodies.append(SteppedBody(
            a=np.array([a[i]]),
            d=np.array([d[i]]),
            slant_angle=np.array([0.0]),  # Flat tops
            heaving=heaving_flags[i]
        ))

    # -------------------------
    # Create Geometry
    # -------------------------
    arrangement = ConcentricBodyGroup(bodies)
    geometry = BasicRegionGeometry(arrangement, h=h, NMK=NMK)

    # -------------------------
    # Create MEEMProblem and MEEMEngine
    # -------------------------
    problem = MEEMProblem(geometry)

    # --- FIX: Define the frequencies and modes to be solved ---
    m0 = 1.0  # The non-dimensional frequency parameter
    
    # Calculate the physical frequency from m0
    problem_frequencies = np.array([omega(m0, h, g)])
    
    # Identify which bodies are heaving to define the modes
    # heaving_flags is [False, True, True], so bodies 1 and 2 are heaving.
    
    # Set them in the problem object
    problem.set_frequencies(problem_frequencies)
    # --- End of FIX ---

    engine = MEEMEngine(problem_list=[problem])

    # The m0 value is now taken from the problem's frequency list, but we can pass it
    # directly to the engine methods as you were doing.

    # -------------------------
    # Solve the linear system
    # -------------------------
    # This single call now handles everything:
    # 1. Calls _ensure_m_k_and_N_k_arrays
    # 2. Calls assemble_A_multi
    # 3. Calls assemble_b_multi
    # 4. Solves the system
    X = engine.solve_linear_system_multi(problem, m0)
    print(f"System solved. Solution vector shape: {X.shape}")

    # -------------------------
    # Compute hydrodynamic coefficients
    # -------------------------
    hydro_coeffs = engine.compute_hydrodynamic_coefficients(problem, X, m0)
    coeff0 = hydro_coeffs[0]
    print(f"\nHydrodynamic coefficients (body 0): {coeff0}")

    # -------------------------
    # Reformat coefficients per region
    # -------------------------
    Cs = engine.reformat_coeffs(X, NMK, len(NMK) - 1)
    for i, c_region in enumerate(Cs):
        print(f"Region {i} (NMK={NMK[i]}): {c_region.shape} coefficients")

    # -------------------------
    # Calculate potentials
    # -------------------------
    potentials = engine.calculate_potentials(problem, X, m0, spatial_res=50, sharp=True)
    R, Z = potentials["R"], potentials["Z"]
    phiH, phiP, phi = potentials["phiH"], potentials["phiP"], potentials["phi"]

    # -------------------------
    # Visualize potentials
    # -------------------------
    engine.visualize_potential(np.real(phiH), R, Z, "Homogeneous Potential (Real)")
    engine.visualize_potential(np.imag(phiH), R, Z, "Homogeneous Potential (Imag)")
    engine.visualize_potential(np.real(phiP), R, Z, "Particular Potential (Real)")
    engine.visualize_potential(np.imag(phiP), R, Z, "Particular Potential (Imag)")
    engine.visualize_potential(np.real(phi), R, Z, "Total Potential (Real)")
    engine.visualize_potential(np.imag(phi), R, Z, "Total Potential (Imag)")

    plt.show()
    print("Script finished. Close plots to exit.")


if __name__ == "__main__":
    main()
