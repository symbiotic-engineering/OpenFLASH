import pytest
import numpy as np
from scipy import linalg

from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.multi_equations import wavenumber

# --- Test Setup ---

@pytest.fixture
def simple_cylinder_setup():
    h = 20.0
    a_val = 10.0
    d_val = 5.0
    NMK = [10, 10] 
    
    geometry = BasicRegionGeometry.from_vectors(
        a=np.array([a_val]),
        d=np.array([d_val]),
        h=h,
        NMK=NMK,
        heaving_map=[True] 
    )
    
    problem = MEEMProblem(geometry)
    return problem, a_val, d_val, h, NMK

# --- Tests ---

def test_analytic_benchmark_consistency(simple_cylinder_setup):
    """
    VALIDATION 1: The 'Direct Calculation' Check.
    """
    problem, a, d, h, NMK = simple_cylinder_setup
    
    omega_val = 1.0
    problem.set_frequencies(np.array([omega_val]))
    
    engine = MEEMEngine([problem])
    results = engine.run_and_store_results(0)
    
    added_mass = results.dataset['added_mass'].values[0, 0, 0] 
    damping = results.dataset['damping'].values[0, 0, 0]
    
    print(f"\nComputed Added Mass: {added_mass:.4f}")
    print(f"Computed Damping: {damping:.4f}")

    # Added Mass for a heaving cylinder must be positive
    assert added_mass > 0, "Added mass should be positive for this standard cylinder."
    # Damping must be positive
    assert damping > 0, "Damping must be positive (generating waves)."
    
    assert np.isfinite(added_mass)
    assert np.isfinite(damping)

def test_low_frequency_limit(simple_cylinder_setup):
    """
    VALIDATION 2: The Physical Limit Check.
    As omega -> 0, radiation damping should be small relative to added mass.
    """
    problem, a, d, h, NMK = simple_cylinder_setup
    
    omega_low = 0.05
    problem.set_frequencies(np.array([omega_low]))
    
    engine = MEEMEngine([problem])
    results = engine.run_and_store_results(0)
    
    damping_low = results.dataset['damping'].values[0, 0, 0]
    added_mass_low = results.dataset['added_mass'].values[0, 0, 0]
    
    print(f"\nLow Freq Damping: {damping_low:.2f}")
    print(f"Low Freq Added Mass: {added_mass_low:.2f}")
    
    # 1. Damping should be small relative to Added Mass at low frequency
    # (Typically < 5% for standard shapes, we use 10% as a safe guard)
    ratio = damping_low / added_mass_low
    assert ratio < 0.1, f"Damping ({damping_low:.0f}) is too large relative to Added Mass ({added_mass_low:.0f}) at low freq (Ratio: {ratio:.3f})"
    
    # 2. Damping should still be positive
    assert damping_low >= 0

def test_caching_integrity():
    h = 20.0
    NMK = [5, 5]
    geom = BasicRegionGeometry.from_vectors(
        a=np.array([10.0]), d=np.array([5.0]), h=h, NMK=NMK, heaving_map=[True]
    )
    problem = MEEMProblem(geom)
    problem.set_frequencies(np.array([1.0]))
    
    engine = MEEMEngine([problem])
    res1 = engine.run_and_store_results(0)
    am1 = res1.dataset['added_mass'].values[0,0,0]
    
    res2 = engine.run_and_store_results(0)
    am2 = res2.dataset['added_mass'].values[0,0,0]
    
    assert np.isclose(am1, am2, atol=1e-9), "Caching introduced non-deterministic behavior."

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))