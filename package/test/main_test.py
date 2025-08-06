import logging
import sys
import os
import numpy as np
import pandas as pd # Can be removed if not used
from scipy import linalg # Can be removed if not used (only solve is used, not linalg directly)
from scipy.integrate import quad # Can be removed if not used
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import core modules ---
from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry
from openflash.multi_equations import *
from openflash.multi_constants import g
from openflash.domain import Domain

# Set print options for better visibility in console
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=8, suppress=True)

def main(): # Renamed from test_main
    """
    Main function to set up a multi-region MEEM problem, solve it,
    and visualize the potential and velocity fields.
    """
    # --- Problem Definition Parameters ---
    # get an error when NMK[idx] > 89
    NMK = [100, 100, 100, 100] # Number of terms in approximation of each region (including e).
    boundary_count = len(NMK) - 1
    h = 100
    d = [29, 7, 4]
    a = [3, 5, 10]
    heaving = [0, 1, 1]
    # 0/false if not heaving, 1/true if yes heaving
    # All computations assume at least 2 regions.

    # --- Geometry Setup ---
    domain_params = Domain.build_domain_params(NMK, a, d, heaving, h)
    
    a_recovered = [p['a'] for p in domain_params[:-1]]
    d_recovered = [p['di'] for p in domain_params[:-1]]
    heaving_recovered = [p['heaving'] for p in domain_params[:-1]]
    h_recovered = domain_params[0]['height']

    assert a_recovered == a
    assert d_recovered == d
    assert heaving_recovered == heaving
    assert h_recovered == h

    # Create Geometry object
    r_coordinates = Domain.build_r_coordinates_dict(a)
    z_coordinates = Domain.build_z_coordinates_dict(h)

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)
    m0 = 1
        
    # --- MEEM Engine Operations ---
    engine = MEEMEngine(problem_list=[problem])

    # Solve the linear system A x = b
    X = engine.solve_linear_system_multi(problem, m0)
    print(f"System solved. Solution vector X shape: {X.shape}")

    # Compute and print hydrodynamic coefficients
    hydro_coefficients = engine.compute_hydrodynamic_coefficients(problem, X, m0)
    print(type(hydro_coefficients["real"]), type(hydro_coefficients["imag"]))

    real = hydro_coefficients["real"]
    imag = hydro_coefficients["imag"]
    nondim_real = hydro_coefficients["nondim_real"]
    nondim_imag = hydro_coefficients["nondim_imag"]
    excitation_phase = hydro_coefficients["excitation_phase"]
    excitation_force = hydro_coefficients["excitation_force"]

    print("\n--- Hydrodynamic Coefficient Breakdown ---")
    print(f"real (added mass): {real:.10f}")
    print(f"imag (damping): {imag:.10f}")
    print(f"real/(h^3): {real / h**3:.10f}")
    print(f"imag/(h^3): {imag / h**3:.10f}")
    print(f"nondimensional, real: {nondim_real:.10f}")
    print(f"nondimensional, imag (no omega factor): {nondim_imag:.10f}")
    print(f"Excitation Phase: {excitation_phase:.10f} radians")
    print(f"Excitation Force: {excitation_force:.10f} N")
    # --- Reformat coefficients using the dedicated MEEMEngine method ---
    reformat_boundary_count = len(NMK) - 1
    Cs = engine.reformat_coeffs(X, NMK, reformat_boundary_count)
    print(f"\nCoefficients reformatted into {len(Cs)} regions.")
    for i, c_region in enumerate(Cs):
        print(f"  Region {i} (NMK={NMK[i]}): {c_region.shape} coefficients")

    # --- Potential and Velocity Field Calculation ---
    
    # --- Use MEEMEngine to calculate potentials ---
    potentials = engine.calculate_potentials(problem, X, m0, spatial_res=50, sharp=True)

    # Unpack
    R = potentials["R"]
    Z = potentials["Z"]
    phiH = potentials["phiH"]
    phiP = potentials["phiP"]
    phi = potentials["phi"]

    # --- Plot using built-in visualizer ---
    engine.visualize_potential(np.real(phiH), R, Z, "Homogeneous Potential (Real)")
    engine.visualize_potential(np.imag(phiH), R, Z, "Homogeneous Potential (Imag)")
    engine.visualize_potential(np.real(phiP), R, Z, "Particular Potential (Real)")
    engine.visualize_potential(np.imag(phiP), R, Z, "Particular Potential (Imag)")
    engine.visualize_potential(np.real(phi), R, Z, "Total Potential (Real)")
    engine.visualize_potential(np.imag(phi), R, Z, "Total Potential (Imag)")\
        
    plt.show()

    print("\nScript finished. Close plot windows to exit.")

if __name__ == "__main__":
    main()