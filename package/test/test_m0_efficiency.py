import sys
import os
import numpy as np
import time
import pytest

# Adjust the path to import from package's 'src' directory.
#
# The test file is at: /semi-analytical-hydro/package/test/test_m0_efficiency.py
# Modules are at:  /semi-analytical-hydro/package/src/
#
# So, we need to add '/semi-analytical-hydro/package/src/' to sys.path

# This gets the directory of the current test file: .../package/test/
current_dir = os.path.dirname(__file__)

# This goes up one level to: .../package/
package_base_dir = os.path.join(current_dir, '..')

# This points to the 'src' directory: .../package/src/
src_dir = os.path.join(package_base_dir, 'src')

# Add 'src_dir' to the Python path
sys.path.insert(0, os.path.abspath(src_dir))

# Now, import directly from the modules in src directory
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.geometry import Geometry
from openflash.domain import Domain


# Define parameters for the problem, mimicking the notebook's setup (semi-analytical-hydro/hydro/python/test/m0_mod_confirmation.ipynb)
# These parameters will be used to create Domain and Geometry objects
h = 100
d_values = [29, 7, 4] # Corresponds to di for inner, outer, and then a dummy for exterior if needed
a_values = [3, 5, 10] # Corresponds to a for inner, outer, and then a dummy for exterior if needed
heaving_values = [0, 1, 1] # 0/false if not heaving, 1/true if yes heaving
# NMK_values = [100, 100, 100, 100] # Number of harmonics for each domain (inner, outer, exterior)    
NMK_values = [10, 10, 10, 10] # Try smaller values first
rho = 1023 # Water density

# Frequencies and modes are needed for MEEMProblem
frequencies = np.array([1.0])
modes = np.array([0, 1])

# m0s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
m0s = [0.1, 0.2, 0.3] # Reduce for quick testing


# Helper function to set up a MEEMProblem and MEEMEngine
def setup_problem_and_engine(m0_val):
    # Define r_coordinates and z_coordinates for Geometry
    r_coordinates = {'a1': a_values[0], 'a2': a_values[1]}
    z_coordinates = {'h': h, 'd1': d_values[0], 'd2': d_values[1]}

    # Define domain_params
    # The 'di' and 'a' for the exterior domain are None as per Domain._get_di and _get_a
    domain_params = [
        {'number_harmonics': NMK_values[0], 'height': h, 'radial_width': a_values[0], 'category': 'inner', 'di': d_values[0], 'a': a_values[0], 'heaving': heaving_values[0], 'slant': False},
        {'number_harmonics': NMK_values[1], 'height': h, 'radial_width': a_values[1], 'category': 'outer', 'di': d_values[1], 'a': a_values[1], 'heaving': heaving_values[1], 'slant': False}, # Changed to 'outer' from 'intermediate' as per original description
        {'number_harmonics': NMK_values[2], 'height': h, 'radial_width': None, 'category': 'exterior', 'di': None, 'a': None, 'heaving': heaving_values[2], 'slant': False},
    ]
    # We are using NMK_values[0], NMK_values[1], NMK_values[2] for the three domains.
    # Added 'slant': False for completeness as it's a parameter in Domain init.

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    prob = MEEMProblem(geometry)
    prob.set_frequencies_modes(frequencies, modes) # Set dummy frequencies/modes

    engine = MEEMEngine(problem_list=[prob])
    return prob, engine

# Scenario 1: Modify (re-assemble A/b using existing problem/engine)
def scenario_modify_and_solve(prob, engine, m0):
    # In MEEMEngine, assemble_A_multi and assemble_b_multi rebuild the matrices/vectors
    # for the given m0, so this is equivalent to "modifying" them.
    print(f"Modifying problem for m0={m0} (using cached template and incremental update)...")
    # The actual optimization happens inside engine.assemble_A_multi and engine.assemble_b_multi
    A = engine.assemble_A_multi(prob, m0)
    b = engine.assemble_b_multi(prob, m0)
    print(f"Solving linear system for m0={m0}...")
    X = np.linalg.solve(A, b)
    print(f"Computing hydrodynamic coefficients for m0={m0}...")
    hydro_coeffs = engine.compute_hydrodynamic_coefficients(prob, X)
    print(f"Hydrodynamic coefficients computed for m0={m0}.")
    return hydro_coeffs

# Scenario 2: Recreate (creates new problem/engine instances for each m0)
def scenario_create_and_solve(m0):
    print(f"Recreating problem for m0={m0} (full new setup each time)...")
    prob, engine = setup_problem_and_engine(m0)
    A = engine._full_assemble_A_multi(prob, m0) # <--- Using the full, unoptimized assembly
    b = engine._full_assemble_b_multi(prob, m0) # <--- Using the full, unoptimized assembly
    
    print(f"Solving linear system for m0={m0}...")
    X = np.linalg.solve(A, b)
    print(f"Computing hydrodynamic coefficients for m0={m0}...")
    hydro_coeffs = engine.compute_hydrodynamic_coefficients(prob, X)
    print(f"Hydrodynamic coefficients computed for m0={m0}.")
    return hydro_coeffs


# Test function
def test_m0_performance_comparison():
    am_modified = []
    dp_modified = [] 
    am_recreate = []
    dp_recreate = []

    # --- Initial setup for 'modify' scenario ---
    start_single = time.perf_counter()
    # Initialize engine with the first problem. The engine will build its cache here.
    initial_prob, initial_engine = setup_problem_and_engine(m0s[0])
    end_single = time.perf_counter()
    tsingle_init = end_single - start_single # Time to initialize a single problem/engine and build cache

    # --- Run 'modify' scenario (optimized) ---
    start_mod = time.perf_counter()
    for m0 in m0s:
        # MEEMEngine.assemble_A_multi and assemble_b_multi will use the pre-built cache
        hydro_coeffs = scenario_modify_and_solve(initial_prob, initial_engine, m0)
        am_modified.append(hydro_coeffs['real'])
        dp_modified.append(hydro_coeffs['imag'])
    end_mod = time.perf_counter()
    tmod = end_mod - start_mod

    # --- Run 'recreate' scenario (unoptimized, full re-creation each time) ---
    start_create = time.perf_counter()
    for m0 in m0s:
        # This will call setup_problem_and_engine which creates a new MEEMEngine
        # and then scenario_create_and_solve will explicitly use the _full_assemble methods.
        hydro_coeffs = scenario_create_and_solve(m0)
        am_recreate.append(hydro_coeffs['real'])
        dp_recreate.append(hydro_coeffs['imag'])
    end_create = time.perf_counter()
    tcreate = end_create - start_create

    print(f"\nInitial setup time for one problem (includes cache build): {tsingle_init:.6f} s")
    print(f"Total time for 'modify' (incremental update): {tmod:.6f} s")
    print(f"Total time for 'recreate' (full problem setup per m0): {tcreate:.6f} s")

    # Assert that the results are numerically close
    for i in range(len(am_modified)):
        np.testing.assert_allclose(am_modified[i], am_recreate[i], rtol=1e-5, atol=1e-8,
                                   err_msg=f"Real part mismatch for m0={m0s[i]}")
        np.testing.assert_allclose(dp_modified[i], dp_recreate[i], rtol=1e-5, atol=1e-8,
                                   err_msg=f"Imaginary part mismatch for m0={m0s[i]}")

    print("\nResults are numerically identical for both approaches.")
    # Now this assertion should pass because 'modify' is truly faster
    assert tmod < tcreate, "Expected 'modify' scenario to be faster than 'recreate' scenario."

    # Breakdown of steps for the 'modify' approach (for insight, not part of test logic)
    print("\nBreakdown of steps (for a single m0 using the 'modify' approach):")
    _prob, _engine = setup_problem_and_engine(m0s[0]) # Re-initialize for clean timing of breakdown
    
    start = time.perf_counter()
    _A = _engine.assemble_A_multi(_prob, m0s[0]) # This now uses the cache
    end = time.perf_counter()
    print(f"Assemble A-matrix (cached update): {end - start:.6f} s")

    start = time.perf_counter()
    _b = _engine.assemble_b_multi(_prob, m0s[0]) # This now uses the cache
    end = time.perf_counter()
    print(f"Assemble b-vector (cached update): {end - start:.6f} s")

    start = time.perf_counter()
    _X = np.linalg.solve(_A, _b)
    end = time.perf_counter()
    print(f"Matrix Solve: {end - start:.6f} s")

    start = time.perf_counter()
    _hydro_coeffs = _engine.compute_hydrodynamic_coefficients(_prob, _X)
    end = time.perf_counter()
    print(f"Hydro Coeff Computation: {end - start:.6f} s")

# To run this test:
# Run `pytest`
#    or cd into package/test directory and run `python -m pytest test_m0_efficiency.py` or `python -m pytest -s test_m0_efficiency.py` to see print statements.