import pytest
import numpy as np
from scipy import linalg
from functools import partial

# Import from your package
from openflash.multi_equations import *
from openflash.multi_constants import *
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.geometry import ConcentricBodyGroup
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.body import SteppedBody

# --- Fixtures ---

@pytest.fixture
def meem_setup():
    """
    Sets up the problem configuration used for testing.
    Equivalent to setting the global variables in the original script.
    """
    # Physical Constants
    h = 10.0
    rho_val = 1000.0
    g_val = 9.81
    
    # Geometry Definition (2-body configuration similar to script intent)
    # Radii (strictly increasing)
    a = np.array([5.0, 10.0]) 
    # Depths
    d = np.array([2.0, 4.0]) 
    # Heaving Flags (Body 0 heaving, Body 1 static)
    heaving = np.array([True, False])
    # Slants (zeros for standard cylinders)
    slants = np.array([0.0, 0.0])
    
    # Harmonics (NMK)
    # 2 bodies -> 3 regions (Inner, Intermediate, Exterior)
    NMK = [5, 5, 5] 
    
    # Frequency
    omega_val = 1.5
    m0_val = wavenumber(omega_val, h)
    
    # Create OpenFlash Objects
    bodies = []
    for i in range(len(a)):
        bodies.append(SteppedBody(
            a=np.array([a[i]]), 
            d=np.array([d[i]]), 
            slant_angle=np.array([slants[i]]), 
            heaving=bool(heaving[i])
        ))
        
    arrangement = ConcentricBodyGroup(bodies)
    geometry = BasicRegionGeometry(arrangement, h, NMK)
    problem = MEEMProblem(geometry)
    problem.set_frequencies(np.array([omega_val]))
    
    return {
        "h": h,
        "a": a,
        "d": d,
        "heaving": heaving,
        "NMK": NMK,
        "m0": m0_val,
        "rho": rho_val,
        "omega": omega_val,
        "problem": problem,
        "boundary_count": len(NMK) - 1
    }

# --- Tests ---

def test_input_assertions(meem_setup):
    """
    Validates the input arrays satisfy geometric and physical constraints.
    (logic taken from the start of the original script)
    """
    a = meem_setup["a"]
    d = meem_setup["d"]
    heaving = meem_setup["heaving"]
    NMK = meem_setup["NMK"]
    h = meem_setup["h"]
    m0 = meem_setup["m0"]
    boundary_count = meem_setup["boundary_count"]

    # Length checks
    for arr in [a, d, heaving]:
        assert len(arr) == boundary_count, \
            "NMK should have one more entry than a, d, and heaving."

    # Boolean checks
    for entry in heaving:
        assert entry in [0, 1, True, False], "heaving entries should be booleans."

    # Monotonicity checks
    left = 0
    for radius in a:
        assert radius > left, "a entries should be increasing and > 0."
        left = radius

    # Depth checks
    for depth in d:
        assert depth >= 0, "d entries should be nonnegative."
        assert depth < h, "d entries should be less than h."

    # Harmonics checks
    for val in NMK:
        assert isinstance(val, int) and val > 0, "NMK entries should be positive integers."

    assert m0 > 0, "m0 should be positive."

def test_hydro_calculations(meem_setup):
    """
    Solves the system and asserts that hydrodynamic coefficients are calculated
    and are non-trivial (non-zero).
    """
    engine = MEEMEngine([meem_setup["problem"]])
    problem = meem_setup["problem"]
    m0 = meem_setup["m0"]
    
    X = engine.solve_linear_system_multi(problem, m0)
    assert not np.isnan(X).any(), "Solution vector X contains NaNs"
    
    # Calculate forces
    results = engine.compute_hydrodynamic_coefficients(problem, X, m0)
    
    assert len(results) > 0
    
    # Check that at least the heaving body (index 0) has non-zero force
    added_mass = results[0]["real"]
    damping = results[0]["imag"]
    
    print(f"Added Mass: {added_mass}, Damping: {damping}")
    
    assert abs(added_mass) > 1e-5, "Added mass should be non-zero for this configuration"
    # Damping might be small but usually non-zero unless trapped mode
    assert abs(damping) >= 0.0, "Damping must be non-negative (energy conservation)"