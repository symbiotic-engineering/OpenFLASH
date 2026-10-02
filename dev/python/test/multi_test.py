import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
python_dir = os.path.abspath(os.path.join(current_dir, '..'))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

# Import Legacy Class
from multi_condensed import Problem

# --- Data Path Helper ---
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data') 

# --- Configurations (Identical to New Package) ---
ALL_CONFIGS = {
    "config1": { # (Note: labeled config1 in your legacy data, config0 in new package)
        "h": 1.001,
        "a": [0.5, 1],
        "d": [0.5, 0.25],
        "heaving_map": [True, True],
        "m0": 1.0,
        "NMK": [15, 15, 15], 
    },
    "config3": {
        "h": 1.9,
        "a": [0.3, 0.5, 1, 1.2, 1.6],
        "d": [0.5, 0.7, 0.8, 0.2, 0.5],
        "heaving_map": [True, True, True, True, True],
        "m0": 1.0,
        "NMK": [15] * 6,
    },
    "config9": {
        "h": 100.0,
        "a": [3, 5, 10],
        "d": [4, 7, 29],
        "heaving_map": [True, True, True],
        "m0": 1.0,
        "NMK": [10] * 4, 
    },
    "config10": {
        "h": 1.5,
        "a": [0.3, 0.5, 1, 1.2, 1.6],
        "d": [0.15, 0.4, 0.75, 0.85, 1.1],
        "heaving_map": [True, True, True, True, True],
        "m0": 1.0,
        "NMK": [6] * 6,
    },
}

def interpret_capytaine_file(filename, omega):
    # Safe path handling
    file_path_imag = os.path.join(DATA_DIR, f"{filename}-imag.csv")
    file_path_real = os.path.join(DATA_DIR, f"{filename}-real.csv")
    
    if not os.path.exists(file_path_real):
        print(f"  [WARN] Benchmark file not found: {filename}")
        return None, None

    df_real = pd.read_csv(file_path_imag, header=None) # Note: Cross-swapped in original logic?
    real_array = (df_real.to_numpy()) * (-1/omega)
        
    df_imag = pd.read_csv(file_path_real, header=None)
    imag_array = (df_imag.to_numpy()) * (1/omega)

    return(real_array, imag_array)

def potential_comparison(filename, arr, rtol, atol, omega, nan_mask):
    real_bench, imag_bench = interpret_capytaine_file(filename, omega)
    
    if real_bench is None:
        return 0, 0, 0 # Skip if no file

    # Compare Real
    is_close_r = np.isclose(real_bench, np.real(arr), rtol=rtol, atol=atol)
    # Compare Imag
    is_close_i = np.isclose(imag_bench, np.imag(arr), rtol=rtol, atol=atol)

    # Mask out NaNs (bodies)
    for i in range(len(nan_mask)):
        is_close_r[nan_mask[i]] = True # Count masked areas as "passing" to ignore them
        is_close_i[nan_mask[i]] = True

    # Count Failures (False values)
    # Total points - Sum of Trues = Number of Fails
    total_pts = is_close_r.size
    failures_r = total_pts - np.sum(is_close_r)
    failures_i = total_pts - np.sum(is_close_i)

    return failures_r, failures_i, total_pts

def run_legacy_test(config_name, p):
    print(f"\n--- Testing Legacy Code: {config_name} ---")
    
    # 1. Map Config to Legacy Problem Signature
    # Legacy uses [1, 0] for heaving, not [True, False]
    heaving_ints = [1 if h else 0 for h in p['heaving_map']]
    
    try:
        # Initialize Legacy Problem
        prob = Problem(
            h=p['h'], 
            d=p['d'], 
            a=p['a'], 
            heaving=heaving_ints, 
            NMK=p['NMK'], 
            m0=p['m0'], 
            rho=1023
        )
        
        # 2. Solve
        a0 = prob.a_matrix()
        b0 = prob.b_vector()
        x = prob.get_unknown_coeffs(a0, b0)
        
        # 3. Get Potentials
        R, Z, phi, nanregions = prob.config_potential_array(prob.reformat_coeffs(x))
        omega_val = prob.angular_freq(1) # Assuming mode 1 matches config
        
        # 4. Compare
        # Using tolerance 0.1 (10%) and 0.2 abs, matching your new test suite
        fail_r, fail_i, total = potential_comparison(
            config_name, phi, 
            rtol=0.1, atol=0.2, 
            omega=omega_val, nan_mask=nanregions
        )
        
        if total > 0:
            print(f"  Failures (Real): {fail_r} / {total} ({fail_r/total:.2%})")
            print(f"  Failures (Imag): {fail_i} / {total} ({fail_i/total:.2%})")
            
            if (fail_r / total) > 0.2:
                print("  [RESULT] SIGNIFICANT MISMATCH (>20%)")
            else:
                print("  [RESULT] MATCHES (within tolerance)")
        else:
            print("  [SKIP] No benchmark data found.")

    except Exception as e:
        print(f"  [CRASH] Legacy code failed: {e}")

if __name__ == "__main__":
    for name, config in ALL_CONFIGS.items():
        run_legacy_test(name, config)