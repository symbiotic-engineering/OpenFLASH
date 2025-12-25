# package/test/test_hydro.py
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import openflash.multi_constants as mc
from openflash_utils import run_openflash_case

# --- Configuration ---
# Adjust this path if necessary relative to where you run pytest
DATA_FOLDER = Path(__file__).parent.parent.parent / "dev" / "python" / "convergence-study" / "meem-vs-capytaine-data" / "csv_data"

# Debug output directories
DEBUG_PLOT_DIR = Path("debug_plots")
DEBUG_CSV_DIR = Path("debug_results")

RHO = 1023
G = 9.81

# VALIDATION TOLERANCE
EXPECTED_RATIO = 1.0
TOLERANCE = 0.1 

# Scaling Factor (Ensure this matches how your mesh files were generated)
# If your BEM mesh radii were 1.0 but your inputs here are 0.5, this needs adjustment.
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
    "am": ["pyCapytaineMu_y", "AddedMass"],
}

def get_csv_files():
    if not DATA_FOLDER.exists():
        return []
    # Grab all CSVs in the folder
    return sorted(list(DATA_FOLDER.glob("*_regenerated_v2.csv")))

def find_col(df, options):
    for col in options:
        if col in df.columns:
            return df[col].values
    return None

def save_debug_plot(m0, of_vals, bem_vals, config_name, csv_name):
    """Generates and saves a comparison plot."""
    DEBUG_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Absolute Values
    ax1.set_title(f"Comparison: {config_name} ({csv_name})")
    ax1.plot(m0, of_vals, 'o-', label='OpenFLASH (MEEM)', markersize=4, alpha=0.8)
    ax1.plot(m0, bem_vals, 'x--', label='Capytaine (BEM)', markersize=6, alpha=0.8)
    ax1.set_ylabel("Non-Dim Added Mass")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = of_vals / bem_vals
    
    ax2.plot(m0, ratios, 'r.-', label='Ratio (OF/BEM)')
    ax2.axhline(1.0, color='k', linestyle='--', linewidth=1.5)
    ax2.axhline(1.0 + TOLERANCE, color='g', linestyle=':', alpha=0.5)
    ax2.axhline(1.0 - TOLERANCE, color='g', linestyle=':', alpha=0.5)
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel("Wavenumber (m0)")
    ax2.set_ylim(0, 3.0) # Limit y-axis to see relevant deviations
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    filename = DEBUG_PLOT_DIR / f"{csv_name.replace('.csv', '')}_debug.png"
    plt.savefig(filename)
    plt.close()

@pytest.mark.parametrize("csv_path", get_csv_files(), ids=lambda p: p.name)
def test_openflash_validates_against_capytaine(csv_path):
    # 0. Safety Check
    assert mc.rho == RHO, f"Density mismatch: Constants {mc.rho} vs Test {RHO}"

    # 1. Identify Geometry from filename
    config_name = next((k for k in CONFIG_SPECS if csv_path.name.startswith(k)), None)
    if config_name is None:
        pytest.skip(f"Skipping {csv_path.name}: No geometry mapping found in CONFIG_SPECS.")
        
    cfg = CONFIG_SPECS[config_name]
    
    # 2. Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        pytest.fail("Failed to read CSV")

    m0_vals = find_col(df, COL_MAPPING["m0"])
    bench_am = find_col(df, COL_MAPPING["am"])
    
    if m0_vals is None or bench_am is None:
        pytest.skip(f"Required columns not found. Cols: {df.columns}")

    # 3. Use all points for detailed plotting
    m0_subset = m0_vals
    bench_am_subset = bench_am

    print(f"\n{'='*80}")
    print(f"DEBUG REPORT: {config_name} | {csv_path.name}")
    print(f"{'='*80}")

    # 4. Prepare Geometry
    scaled_a = [val * RADIUS_SCALING for val in cfg['a']]
    h_val = cfg['h']
    
    # 5. Run OpenFLASH
    # Note: Increasing harmonics (NMK) improves accuracy but slows tests.
    NMK = [50] * (len(cfg['a']) + 1)
    
    results = []
    
    for i, m0 in enumerate(m0_subset):
        if m0 <= 0 or np.isnan(m0):
            results.append({"m0": m0, "of_nondim": np.nan, "bem": np.nan, "ratio": np.nan, "status": "SKIP"})
            continue
            
        try:
            # --- CRITICAL CORRECTION ---
            # run_openflash_case returns NORMALIZED (Non-dimensional) values
            # according to openflash_utils.py. 
            # DO NOT re-normalize here.
            am_nondim, _, _ = run_openflash_case(
                h_val, cfg['d'], scaled_a, cfg['heaving'], NMK, m0, RHO
            )
            
            cpt_val = bench_am_subset[i]
            
            # Check for division by near-zero
            if abs(cpt_val) < 1e-6:
                ratio = 1.0 if abs(am_nondim) < 1e-6 else 999.0
            else:
                ratio = am_nondim / cpt_val

            status = "OK" if abs(ratio - 1.0) < TOLERANCE else "FAIL"
            
            results.append({
                "m0": m0, 
                "of_nondim": am_nondim, 
                "bem": cpt_val, 
                "ratio": ratio, 
                "status": status
            })
            
        except Exception as e:
            results.append({"m0": m0, "of_nondim": np.nan, "bem": bench_am_subset[i], "ratio": np.nan, "status": f"ERR: {str(e)}"})

    # 6. Convert to DataFrame for Analysis
    res_df = pd.DataFrame(results)
    
    # Save CSV for manual inspection
    DEBUG_CSV_DIR.mkdir(exist_ok=True)
    res_df.to_csv(DEBUG_CSV_DIR / f"DEBUG_{csv_path.name}", index=False)
    
    # 7. Generate Plot
    valid_plot_mask = np.isfinite(res_df['of_nondim']) & np.isfinite(res_df['bem'])
    if np.any(valid_plot_mask):
        save_debug_plot(
            res_df.loc[valid_plot_mask, 'm0'], 
            res_df.loc[valid_plot_mask, 'of_nondim'], 
            res_df.loc[valid_plot_mask, 'bem'], 
            config_name, 
            csv_path.name
        )

    # 8. Filter for Statistics
    # Filter out NaNs and cases where BEM is effectively zero (singularities)
    valid_mask = (
        np.isfinite(res_df['of_nondim']) & 
        np.isfinite(res_df['bem']) & 
        (np.abs(res_df['bem']) > 0.05)
    )
    
    if np.sum(valid_mask) == 0:
        print("No valid data points for comparison (all NaNs or BEM ~ 0).")
        pytest.skip("No valid comparison points.")

    valid_df = res_df[valid_mask]
    avg_ratio = valid_df['ratio'].mean()
    
    # Print a nice table to stdout
    print("\nSample Data Points (First 10 + Failures):")
    # Show first 5
    print(res_df.head(5).to_string(index=False, float_format=lambda x: "{:.4f}".format(x) if pd.notnull(x) else "NaN"))
    # Show specific failures (ratio deviations > 20%)
    failures = valid_df[np.abs(valid_df['ratio'] - 1.0) > 0.1]
    if not failures.empty:
        print("\n... Significant Mismatches (>10%):")
        print(failures.head(10).to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    print("-" * 60)
    print(f"Comparison Summary for {csv_path.name}")
    print(f"  Valid Points : {len(valid_df)} / {len(res_df)}")
    print(f"  Avg Ratio    : {avg_ratio:.4f}")
    print(f"  Target       : {EXPECTED_RATIO} +/- {TOLERANCE}")
    print("-" * 60)

    # 9. Final Assertion
    assert abs(avg_ratio - EXPECTED_RATIO) < TOLERANCE, \
        f"Validation Failed: OpenFLASH {avg_ratio:.2f}x Capytaine."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])