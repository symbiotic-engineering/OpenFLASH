# test_openflash_convergence.py
import pytest
import numpy as np
from openflash_utils import run_openflash_case

# --- Define Configurations ---
CONFIGS = {
    "some_bicylinder": {
        "h": 1.001, "d": [0.5, 0.25], "a": [0.25, 0.5], "heaving": [1, 1],
        "m0": 1.0, "rho": 1023
    },
    "mini_tricylinder": {
        "h": 2.001, "d": [1.0, 0.5, 0.25], "a": [0.25, 0.5, 1.0], "heaving": [1, 1, 1],
        "m0": 1.0, "rho": 1023
    }
}

@pytest.mark.parametrize("config_name, cfg", CONFIGS.items())
def test_convergence_openflash(config_name, cfg):
    """
    Replicates the loop over NMK terms to check for convergence.
    Passes if the relative difference drops below 0.1% (0.001) within 30 terms.
    """
    print(f"\nTesting convergence for: {config_name}")
    
    history_real = []
    history_imag = []
    
    converged_real = False
    converged_imag = False
    
    # Loop from 2 to 30 terms, similar to your script
    for n_term in range(2, 31):
        # Create NMK list: [n, n, ..., n] (one for each body + 1 for exterior)
        num_regions = len(cfg['a']) + 1
        NMK = [n_term] * num_regions
        
        am, dp, duration = run_openflash_case(
            cfg['h'], cfg['d'], cfg['a'], cfg['heaving'], NMK, cfg['m0'], cfg['rho']
        )
        
        history_real.append(am)
        history_imag.append(dp)
        
        # Check convergence (starting from the second iteration)
        if len(history_real) > 1:
            prev_real = history_real[-2]
            prev_imag = history_imag[-2]
            
            diff_real = abs((am - prev_real) / prev_real) if prev_real != 0 else 0
            diff_imag = abs((dp - prev_imag) / prev_imag) if prev_imag != 0 else 0
            
            # Print status for visibility (use -s flag with pytest to see this)
            print(f"Terms: {n_term}, AM: {am:.6f} ({diff_real:.2%}), DP: {dp:.6f} ({diff_imag:.2%}), Time: {duration:.4f}s")
            
            if diff_real < 0.001:
                converged_real = True
            if diff_imag < 0.001:
                converged_imag = True
            
            # If both converged, we can stop the test early and declare success
            if converged_real and converged_imag:
                print(f"Converged at N={n_term}")
                return

    # If loop finishes without returning, assertion fails
    assert converged_real, "Added Mass did not converge within 30 terms."
    assert converged_imag, "Damping did not converge within 30 terms."

def test_sanity_check_values():
    """
    A quick test to ensure values aren't NaN or Infinite.
    """
    cfg = CONFIGS["some_bicylinder"]
    NMK = [5, 5, 5] # Low count for speed
    
    am, dp, _ = run_openflash_case(
        cfg['h'], cfg['d'], cfg['a'], cfg['heaving'], NMK, cfg['m0'], cfg['rho']
    )
    
    assert np.isfinite(am)
    assert np.isfinite(dp)
    assert dp >= 0 # Damping should be positive