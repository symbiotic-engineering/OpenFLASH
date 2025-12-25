# package/test/test_high_frequency_convergence.py
import pytest
import numpy as np
from openflash_utils import run_openflash_case

# Define Configurations
CONFIGS = {
    "config0": {
        "h": 1.001, "d": [0.5, 0.25], "a": [0.5, 1], "heaving": [1, 1]
    },
    "config1": {
        "h": 1.5, "d": [1.1, 0.85, 0.75, 0.4, 0.15], "a": [0.3, 0.5, 1, 1.2, 1.6], 
        "heaving": [1, 1, 1, 1, 1]
    },
    "config2": {
        "h": 100, "d": [29, 7, 4], "a": [3, 5, 10], "heaving": [1, 1, 1]
    },
    "config3": {
        "h": 1.9, "d": [0.5, 0.7, 0.8, 0.2, 0.5], "a": [0.3, 0.5, 1, 1.2, 1.6], 
        "heaving": [1, 1, 1, 1, 1]
    },
    "config4": {
        "h": 1.001, "d": [0.5, 0.25], "a": [0.5, 1], "heaving": [0, 1]
    },
    "config5": {
        "h": 1.001, "d": [0.5, 0.25], "a": [0.5, 1], "heaving": [1, 0]
    },
    "config6": {
        "h": 100, "d": [29, 7, 4], "a": [3, 5, 10], "heaving": [0, 1, 1]
    }
}

RHO = 1023
# High m0 list to test convergence (we just test the endpoint for assertion)
M0_MAX = 1e6 
TOLERANCE = 1e-2 # 1% or 0.01 nondimensional units

@pytest.mark.parametrize("name, cfg", CONFIGS.items())
def test_high_frequency_limit(name, cfg):
    """
    Verifies that the finite high-frequency result (m0=1e6) matches 
    the infinite frequency result (m0=inf).
    """
    print(f"\nRunning {name}...")
    
    # NMK setup: 100 per region (len(heaving) + 1)
    # Note: len(heaving) = num_segments. Num regions = num_segments + 1 (exterior)
    num_regions = len(cfg['heaving']) + 1
    NMK = [100] * num_regions

    # 1. Solve for m0 = inf
    inf_am, inf_dp, _ = run_openflash_case(
        cfg['h'], cfg['d'], cfg['a'], cfg['heaving'], NMK, np.inf, RHO
    )
    
    # 2. Solve for m0 = 1e6
    fin_am, fin_dp, _ = run_openflash_case(
        cfg['h'], cfg['d'], cfg['a'], cfg['heaving'], NMK, M0_MAX, RHO
    )

    # 3. Assertions
    
    # Added Mass Convergence
    # Check absolute difference or relative difference
    am_diff = abs(fin_am - inf_am)
    print(f"Added Mass: Inf={inf_am:.5f}, 1e6={fin_am:.5f}, Diff={am_diff:.5e}")
    
    assert am_diff < TOLERANCE, \
        f"Added Mass mismatch for {name}: {inf_am} vs {fin_am}"

    # Damping Convergence
    # Damping at infinite frequency should be 0.
    # OpenFLASH usually returns 0 for inf_dp correctly.
    # Finite high freq damping should also be very small.
    print(f"Damping: Inf={inf_dp:.5f}, 1e6={fin_dp:.5f}")
    
    assert abs(inf_dp) < 1e-9, f"Infinite Damping should be 0, got {inf_dp}"
    assert abs(fin_dp) < TOLERANCE, f"High-freq Damping should be near 0, got {fin_dp}"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])