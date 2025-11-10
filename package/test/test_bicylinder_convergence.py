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
    
    # Heaving = [1, 1] means both bodies are heaving
    heaving_map = [True, True]
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
    
    # 4. Set frequency
    # Convert m0 (wavenumber k) to omega (angular frequency)
    # omega^2 = g * k * tanh(k*h)
    omega = np.sqrt(g * m0 * np.tanh(m0 * h))
    problem.set_frequencies(np.array([omega]))

    # 5. Create Engine and Run
    # We create a new engine for each problem to get a clean cache
    engine = MEEMEngine(problem_list=[problem])
    results = engine.run_and_store_results(problem_index=0)

    # 6. Extract and Process Results
    ds = results.get_results()
    
    # Get the 2x2 matrices for the first (and only) frequency
    # .values[0] selects the data for the first frequency
    added_mass_matrix = ds['added_mass'].values[0]
    damping_matrix = ds['damping'].values[0]

    # The "math script" calculates one AM and one DP value.
    # For a 2-body system heaving in phase, the total force
    # is related to the sum of all elements in the matrices.
    total_am_dim = np.sum(added_mass_matrix)
    total_dp_dim = np.sum(damping_matrix)

    # 7. Non-dimensionalize
    rho = 1023.0       # From "math script"
    a_outer = a_vec[-1]  # Non-dim length (0.5)
    
    # A' = A / (rho * pi * a^3)
    factor_am = 1.0 / (rho * pi * (a_outer**3))
    # B' = B / (rho * pi * a^3 * omega)
    factor_dp = 1.0 / (rho * pi * (a_outer**3) * omega)

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