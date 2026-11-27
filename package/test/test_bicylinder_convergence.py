import pytest
import numpy as np

# Import the classes from your package
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine

# Assuming 'g' is defined in multi_constants
# If not, you can just use g = 9.81
try:
    from openflash.multi_constants import g
except ImportError:
    g = 9.81

pi = np.pi

def run_bicylinder_sim(nmk_list: list[int], h: float = 1.001, m0: float = 1.0) -> tuple[float, float]:
    """
    Helper function to run a single simulation with the new package.
    
    This function replicates the setup from the "math script"
    (bicylinder, m0=1.0) and returns the *total non-dimensional*
    added mass and damping.
    """
    
    # 1. Define geometry params from "math script"
    d_vec = np.array([0.5, 0.25])
    a_vec = np.array([0.25, 0.5])
    
    # --- FIX: Initialize with NO heaving bodies to pass geometry assertion ---
    # The MEEMEngine will calculate the full interaction matrix regardless of initial state,
    # provided we tell it which modes to iterate over.
    heaving_map = [False, False] 
    
    # Body 0 = radius 0.25, Body 1 = radius 0.5
    body_map = [0, 1] 

    # 2. Create Geometry
    # The NMK list is passed in for the convergence test
    geometry = BasicRegionGeometry.from_vectors(
        a=a_vec,
        d=d_vec,
        h=h,
        NMK=nmk_list,
        body_map=body_map,
        heaving_map=heaving_map
    )

    # 3. Create Problem
    problem = MEEMProblem(geometry)
    
    # --- FIX: Explicitly define the modes we want to solve for ---
    # Since we set heaving=False above, problem.modes would default to empty.
    # We force it to solve for Body 0 and Body 1.
    # Note: You might need to adjust MEEMProblem to allow setting 'modes' if it's a property.
    # If MEEMProblem.modes is a property that reads from geometry, we might need a workaround.
    # Looking at previous code, problem.modes inferred from geometry. 
    # Workaround: MEEMEngine.run_and_store_results uses problem.modes.
    # If we cannot set problem.modes directly, we rely on the fact that for calculation of coefficients,
    # we usually want the full matrix. 
    
    # Let's inspect MEEMProblem. If it doesn't allow setting modes, we mock it 
    # or subclass it for the test. 
    # Assuming standard behavior, let's try to set it if possible, or monkeypatch.
    # Ideally, MEEMProblem should allow defining "active degrees of freedom".
    
    # HACK for Test: Override the property on the instance if Python allows, 
    # or rely on the engine to calculate all if modes is empty? No, engine loops over modes.
    
    # OPTION B: Modify the geometry *after* creation? No, ConcentricBodyGroup is strict.
    
    # OPTION C: We modify the test to rely on the fact that `run_and_store_results`
    # likely iterates over `problem.modes`. 
    # If `MEEMProblem` derives modes from `geometry.body_arrangement.heaving`, 
    # and `ConcentricBodyGroup` asserts sum(heaving) <= 1... 
    # Then we can ONLY simulate one body heaving at a time in the "Problem definition".
    
    # BUT `run_and_store_results` is supposed to handle the N-body problem.
    # It does this by creating *temporary* problems with single heaving bodies.
    # It just needs to know *which* indices to do this for.
    
    # If we cannot set `problem.modes`, we can manually execute the engine loop here
    # instead of calling `run_and_store_results`.
    
    # --- MANUAL EXECUTION STRATEGY ---
    # This bypasses the dependency on `problem.modes` being pre-set to [0,1]
    
    # Convert m0 to omega
    omega_val = np.sqrt(g * m0 * np.tanh(m0 * h))
    problem.set_frequencies(np.array([omega_val]))
    
    engine = MEEMEngine(problem_list=[problem])
    
    # Initialize total accumulators
    total_am_dim = 0.0
    total_dp_dim = 0.0
    
    # Loop over both bodies (0 and 1)
    for mode_idx in [0, 1]:
        # 1. Create a temp geometry where ONLY mode_idx is heaving
        # We can reuse the helper from BasicRegionGeometry to make a new one quickly
        temp_heaving = [False, False]
        temp_heaving[mode_idx] = True
        
        temp_geom = BasicRegionGeometry.from_vectors(
            a=a_vec, d=d_vec, h=h, NMK=nmk_list,
            body_map=body_map, heaving_map=temp_heaving
        )
        temp_prob = MEEMProblem(temp_geom)
        temp_prob.set_frequencies(np.array([omega_val]))
        
        temp_engine = MEEMEngine(problem_list=[temp_prob])
        
        # 2. Solve for potential X_i
        X_i = temp_engine.solve_linear_system_multi(temp_prob, m0)
        
        # 3. Calculate forces on ALL bodies due to motion of mode_idx
        # This returns the column of coefficients [ (F_0i), (F_1i) ]
        # We pass modes_to_calculate=[0, 1] to get forces on both bodies
        coeffs = temp_engine.compute_hydrodynamic_coefficients(
            temp_prob, X_i, m0, modes_to_calculate=np.array([0, 1])
        )
        
        # 4. Sum up the dimensional values
        # For 2 bodies moving in unison, Force_total = Sum(F_ji) for all j, i
        for c in coeffs:
            total_am_dim += c['real']
            total_dp_dim += c['imag']

    # 7. Non-dimensionalize
    rho = 1023.0       # From "math script"
    a_outer = a_vec[-1]  # Non-dim length (0.5)
    
    # A' = A / (rho * pi * a^3)
    factor_am = 1.0 / (rho * pi * (a_outer**3))
    # B' = B / (rho * pi * a^3 * omega_val)
    factor_dp = 1.0 / (rho * pi * (a_outer**3) * omega_val)

    total_am_nondim = total_am_dim * factor_am
    total_dp_nondim = total_dp_dim * factor_dp

    return total_am_nondim, total_dp_nondim

# --- Pytest Fixture and Test ---

@pytest.fixture(scope="module")
def converged_solution():
    """
    Calculates the "golden" solution using a high number of terms.
    This runs only ONCE per test session.
    """
    # Based on the "math script", N=30 is the max
    nmk_converged = [30, 30, 30]
    print("\nCalculating converged solution (NMK=30)...")
    am_conv, dp_conv = run_bicylinder_sim(nmk_converged)
    print(f"Converged AM (nondim): {am_conv}")
    print(f"Converged DP (nondim): {dp_conv}")
    return am_conv, dp_conv


def test_bicylinder_convergence(converged_solution):
    """
    Tests if the solution with fewer terms (N=20) is close
    to the converged solution (N=30).
    
    This replicates the *intent* of the math team's script
    which checks for: abs(diff) < 0.001
    """
    # Get the pre-calculated converged solution
    am_converged, dp_converged = converged_solution
    
    # Run with fewer terms, e.g., N=20
    nmk_test = [20, 20, 20]
    print("Calculating test solution (NMK=20)...")
    am_test, dp_test = run_bicylinder_sim(nmk_test)
    print(f"Test AM (nondim): {am_test}")
    print(f"Test DP (nondim): {dp_test}")

    # The "math script" checks for 0.1% difference (1e-3)
    # We use np.isclose which is safer and checks relative tolerance
    convergence_threshold = 1e-3 
    
    assert np.isclose(am_test, am_converged, rtol=convergence_threshold)
    assert np.isclose(dp_test, dp_converged, rtol=convergence_threshold)