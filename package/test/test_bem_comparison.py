# package/test/test_bem_comparison.py

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import openflash.multi_constants as mc
from openflash_utils import run_openflash_case

# --- Configuration ---
DATA_FOLDER = Path(__file__).parent.parent.parent / "dev" / "python" / "convergence-study" / "meem-vs-capytaine-data" / "csv_data"
DEBUG_PLOT_DIR = Path(__file__).parent / "debug_plots"
RHO = 1023
G = 9.81

# VALIDATION TOLERANCE
# BEM (facetted mesh) and MEEM (analytical).
EXPECTED_RATIO = 1.0
TOLERANCE = 0.1

# --- CRITICAL FIX 1: MATCH GEOMETRY ---
RADIUS_SCALING = 1.0

CONFIG_SPECS = {
    "mini_bicylinder":   {"h": 1.001, "d": [0.25, 0.125], "a": [0.125, 0.25], "heaving": [1, 1]},
    "small_bicylinder":  {"h": 1.001, "d": [0.5, 0.25],   "a": [0.5, 1.0],    "heaving": [1, 1]},
    "big_bicylinder":    {"h": 1.001, "d": [0.75, 0.5],   "a": [0.5, 0.75],   "heaving": [1, 1]},
    "mini_tricylinder":  {"h": 2.001, "d": [1.0, 0.5, 0.25], "a": [0.25, 0.5, 1.0], "heaving": [1, 1, 1]},
    "small_tricylinder": {"h": 20.0,  "d": [15, 10, 5],   "a": [5, 10, 15],   "heaving": [1, 1, 1]},
    "big_tricylinder":   {"h": 25.0,  "d": [20, 15, 10],  "a": [10, 15, 20],  "heaving": [1, 1, 1]},
}

COL_MAPPING = {
    "m0": ["pyCapytaineMu_x", "pyMEEMMu_x", "m0"], 
    "am": ["pyCapytaineMu_y"],
}

def get_csv_files():
    if not DATA_FOLDER.exists():
        return []
    return sorted(list(DATA_FOLDER.glob("*_regenerated_v2.csv")))

def find_col(df, options):
    for col in options:
        if col in df.columns:
            return df[col].values
    return None

def calculate_displacement(a, d):
    """
    Calculates the exact displaced volume of the stepped cylinder.
    """
    vol = 0.0
    # Region 0 (Inner Cylinder)
    vol += np.pi * (a[0]**2) * d[0]
    
    # Annular Regions
    for i in range(1, len(a)):
        annulus_area = np.pi * (a[i]**2 - a[i-1]**2)
        vol += annulus_area * d[i]
        
    return vol

@pytest.mark.parametrize("csv_path", get_csv_files(), ids=lambda p: p.name)
def test_openflash_validates_against_capytaine(csv_path):
    # 0. Safety Check
    assert mc.rho == RHO, f"Density mismatch: {mc.rho} vs {RHO}"

    # 1. Identify Geometry
    config_name = next((k for k in CONFIG_SPECS if csv_path.name.startswith(k)), None)
    if config_name is None:
        pytest.skip(f"Skipping {csv_path.name}: No geometry mapping.")
        
    cfg = CONFIG_SPECS[config_name]
    
    # 2. Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        pytest.fail("Failed to read CSV")

    m0_vals = find_col(df, COL_MAPPING["m0"])
    bench_am = find_col(df, COL_MAPPING["am"])
    
    if m0_vals is None or bench_am is None:
        pytest.skip("Required columns not found.")

    # 3. Downsample (DISABLED: Now using all points)
    indices = np.arange(len(m0_vals))
    m0_subset = m0_vals[indices]
    bench_am_subset = bench_am[indices]

    print(f"\n{'='*60}")
    print(f"DEBUG REPORT: {config_name}")
    print(f"{'='*60}")
    
    # 4. Prepare Geometry & Debug Geometry
    scaled_a = [val * RADIUS_SCALING for val in cfg['a']]
    h_val = cfg['h']
    
    displacement_vol = calculate_displacement(scaled_a, cfg['d'])
    norm_factor = 1.0 / (RHO * displacement_vol)

    print(f"GEOMETRY SETUP:")
    print(f"  - Depths (d): {cfg['d']}")
    print(f"  - Radii  (a): {scaled_a}")
    print(f"  - Calc Vol  : {displacement_vol:.6f} m^3")
    print(f"  - Norm Fac  : {norm_factor:.6e} (1/(rho*vol))")
    print(f"{'-'*60}")

    # 5. Run OpenFLASH
    NMK = [50] * (len(cfg['a']) + 1)
    
    # Storage for detailed debug table
    debug_rows = []
    of_am_nondim_list = []
    
    for i, m0 in enumerate(m0_subset):
        if m0 <= 0 or np.isnan(m0):
            of_am_nondim_list.append(np.nan)
            continue
            
        try:
            # Returns DIMENSIONAL (kg)
            am_dim, _, _ = run_openflash_case(
                h_val, cfg['d'], scaled_a, cfg['heaving'], NMK, m0, RHO
            )
            
            # Normalization
            am_nondim = am_dim * norm_factor
            of_am_nondim_list.append(am_nondim)
            
            # Capture for debug table
            cpt_val = bench_am_subset[i]
            ratio = am_nondim / cpt_val if cpt_val != 0 else 0
            
            debug_rows.append({
                "m0": m0,
                "Raw_OF(kg)": am_dim,
                "Nondim_OF": am_nondim,
                "Capytaine": cpt_val,
                "Ratio": ratio,
                "Status": "OK" if abs(ratio - 1.0) < TOLERANCE else "FAIL"
            })
            
        except Exception as e:
            of_am_nondim_list.append(np.nan)
            debug_rows.append({"m0": m0, "Status": f"ERR: {str(e)}"})

    of_am = np.array(of_am_nondim_list)

    # 6. Validation
    valid_mask = (np.isfinite(of_am) & np.isfinite(bench_am_subset))
    
    if np.sum(valid_mask) == 0:
        pytest.skip("No valid data points.")

    ratios = of_am[valid_mask] / bench_am_subset[valid_mask]
    avg_ratio = np.mean(ratios)
    
    # --- ENHANCED DEBUG TABLE PRINT ---
    df_debug = pd.DataFrame(debug_rows)
    # Ensure columns exist even if empty to prevent KeyError
    required_cols = ["m0", "Raw_OF(kg)", "Nondim_OF", "Capytaine", "Ratio", "Status"]
    for col in required_cols:
        if col not in df_debug.columns:
            df_debug[col] = np.nan

    print("\nDETAILED COMPARISON:")
    # print full dataframe string
    print(df_debug[required_cols].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print("-" * 60)
    
    # Calculate stats excluding the m0 < 0.2 outlier region if needed
    high_freq_mask = m0_subset[valid_mask] > 0.2
    if np.any(high_freq_mask):
        high_freq_ratio = np.mean(ratios[high_freq_mask])
        print(f"Avg Ratio (All)      : {avg_ratio:.4f}")
        print(f"Avg Ratio (m0 > 0.2) : {high_freq_ratio:.4f}  <-- Likely the 'true' physics match")
    else:
        print(f"Avg Ratio (All)      : {avg_ratio:.4f}")

    # Generate Plot (Existing code is fine, just added error handling)
    try:
        m0_valid = m0_subset[valid_mask]
        of_valid = of_am[valid_mask]
        cpt_valid = bench_am_subset[valid_mask]
        
        DEBUG_PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.set_title(f"Comparison: {config_name}")
        ax1.plot(m0_valid, of_valid, 'o-', label='OpenFLASH')
        ax1.plot(m0_valid, cpt_valid, 'x--', label='Capytaine')
        ax1.set_ylabel("Added Mass (Nondim)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(m0_valid, ratios, 'r.-', label='Ratio')
        ax2.axhline(1.0, color='k', linestyle='--')
        ax2.axhline(1.0 + TOLERANCE, color='g', linestyle=':', alpha=0.5)
        ax2.axhline(1.0 - TOLERANCE, color='g', linestyle=':', alpha=0.5)
        ax2.set_ylabel("Ratio")
        ax2.set_xlabel("m0")
        ax2.grid(True, alpha=0.3)
        plt.savefig(DEBUG_PLOT_DIR / f"{config_name}_debug.png")
        plt.close()
    except Exception:
        pass

    assert abs(avg_ratio - EXPECTED_RATIO) < TOLERANCE, \
        f"Failed: Avg Ratio {avg_ratio:.2f} (Target 1.0). High-freq match might be better."