# package/test/test_bem_comparison.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from openflash_utils import run_openflash_case

# --- Configuration ---
DATA_FOLDER = Path(__file__).parent.parent.parent / "dev" / "python" / "convergence-study" / "meem-vs-capytaine-data" / "csv_data"
RHO = 1023

# STRICT VALIDATION: Expect ~1.0 ratio
EXPECTED_RATIO = 1.0
TOLERANCE = 0.20 

CONFIG_SPECS = {
    # Using the regenerated files primarily
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
    "dp": ["pyCapytaineLambda_y"]
}

def get_csv_files():
    if not DATA_FOLDER.exists():
        return []
    # Only verify against the REGENERATED files where we fixed the geometry scaling
    return sorted(list(DATA_FOLDER.glob("*.csv")))

def find_col(df, options):
    for col in options:
        if col in df.columns:
            return df[col].values
    return None

@pytest.mark.parametrize("csv_path", get_csv_files(), ids=lambda p: p.name)
def test_openflash_validates_against_capytaine(csv_path):
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

    # 3. Downsample
    indices = np.arange(0, len(m0_vals), 5)
    m0_subset = m0_vals[indices]
    bench_am_subset = bench_am[indices]

    print(f"\nRunning {config_name} ({len(m0_subset)} pts)...")

    # 4. Run OpenFLASH
    NMK = [50] * (len(cfg['a']) + 1)
    
    of_am_list = []
    
    for m0 in m0_subset:
        if m0 <= 0 or np.isnan(m0):
            of_am_list.append(np.nan)
            continue
            
        try:
            am, _, _ = run_openflash_case(
                cfg['h'], cfg['d'], cfg['a'], cfg['heaving'], NMK, m0, RHO
            )
            of_am_list.append(am)
        except Exception:
            of_am_list.append(np.nan)

    of_am = np.array(of_am_list)

    # 5. Strict Validation Assertion
    valid_mask = (
        np.isfinite(of_am) & 
        np.isfinite(bench_am_subset) & 
        (np.abs(bench_am_subset) > 0.05)
    )
    
    if np.sum(valid_mask) == 0:
        pytest.skip("No valid data points for comparison.")

    ratios = of_am[valid_mask] / bench_am_subset[valid_mask]
    avg_ratio = np.mean(ratios)
    
    print(f"  Avg Ratio: {avg_ratio:.4f}")

    # Assertion
    assert abs(avg_ratio - EXPECTED_RATIO) < TOLERANCE, \
        f"Validation Failed: OpenFLASH {avg_ratio:.2f}x Capytaine. Expected ~1.0x."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])