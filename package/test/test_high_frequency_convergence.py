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
M0_MAX = 1e6  
TOLERANCE = 0.01

@pytest.mark.parametrize("name, cfg", CONFIGS.items())
def test_high_frequency_limit(name, cfg):
    """
    Verifies that the finite high-frequency result matches 
    the infinite frequency result (m0=inf).
    """
    print(f"\nRunning {name}...")
    
    num_regions = len(cfg['heaving']) + 1
    NMK = [100] * num_regions 
    # -----------------------------------------------------------

    # 1. Solve for m0 = inf
    inf_am, inf_dp, _ = run_openflash_case(
        cfg['h'], cfg['d'], cfg['a'], cfg['heaving'], NMK, np.inf, RHO
    )
    
    # 2. Solve for m0 = M0_MAX
    fin_am, fin_dp, _ = run_openflash_case(
        cfg['h'], cfg['d'], cfg['a'], cfg['heaving'], NMK, M0_MAX, RHO
    )

    # 3. Assertions
    
    # Added Mass Convergence
    am_diff = abs(fin_am - inf_am)
    
    # Calculate relative error if values are large, absolute if small
    if abs(inf_am) > 1.0:
        rel_error = am_diff / abs(inf_am)
        print(f"Added Mass: Inf={inf_am:.5f}, HighFreq={fin_am:.5f}, RelDiff={rel_error:.2%}")
        assert rel_error < TOLERANCE, f"Relative error {rel_error:.2%} exceeds {TOLERANCE}"
    else:
        print(f"Added Mass: Inf={inf_am:.5f}, HighFreq={fin_am:.5f}, AbsDiff={am_diff:.5e}")
        assert am_diff < TOLERANCE, f"Absolute mismatch {am_diff:.5e} > {TOLERANCE}"

    # Damping Convergence (Should approach 0)
    print(f"Damping: Inf={inf_dp:.5e}, HighFreq={fin_dp:.5e}")
    
    # Infinite frequency damping is theoretically 0
    assert abs(inf_dp) < 1e-9, f"Infinite Damping should be 0, got {inf_dp}"
    # High frequency damping should be small
    assert abs(fin_dp) < 0.5, f"High-freq Damping should be small, got {fin_dp}"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])