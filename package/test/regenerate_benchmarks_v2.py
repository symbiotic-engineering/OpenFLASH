import numpy as np
import pandas as pd
import sys
from pathlib import Path

# --- 1. SETUP IMPORTS ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
SLANTS_DIR = PROJECT_ROOT / "dev" / "python" / "slants"

if str(SLANTS_DIR) not in sys.path:
    sys.path.append(str(SLANTS_DIR))

try:
    from capytaine_generator import CapytaineSlantSolver
except ImportError as e:
    raise ImportError(f"Could not import CapytaineSlantSolver from {SLANTS_DIR}. Check the path.") from e

# --- 2. CONFIGURATION ---
RHO = 1023
G = 9.81
DATA_FOLDER = PROJECT_ROOT / "dev" / "python" / "convergence-study" / "meem-vs-capytaine-data" / "csv_data"

CONFIGS = {
    "mini_bicylinder":   {"h": 1.001, "d": [0.25, 0.125], "a": [0.125, 0.25], "heaving": [1, 1]},
    "small_bicylinder":  {"h": 1.001, "d": [0.5, 0.25],   "a": [0.5, 1.0],    "heaving": [1, 1]},
    "big_bicylinder":    {"h": 1.001, "d": [0.75, 0.5],   "a": [0.5, 0.75],   "heaving": [1, 1]},
    "mini_tricylinder":  {"h": 2.001, "d": [1.0, 0.5, 0.25], "a": [0.25, 0.5, 1.0], "heaving": [1, 1, 1]},
    "small_tricylinder": {"h": 20.0,  "d": [15, 10, 5],   "a": [5, 10, 15],   "heaving": [1, 1, 1]},
    "big_tricylinder":   {"h": 25.0,  "d": [20, 15, 10],  "a": [10, 15, 20],  "heaving": [1, 1, 1]},
}

# --- 3. HELPER FUNCTIONS ---
def calculate_displacement(a, d):
    """
    Calculates the exact displaced volume of the stepped cylinder.
    Assumes 'd' represents the depth of the bottom of each region from the surface
    and 'a' represents the outer radius of each region.
    
    Structure is an inverted stepped cone (widest at top/outermost).
    Regions: 
    0: r in [0, a[0]], depth d[0]
    i: r in [a[i-1], a[i]], depth d[i]
    """
    vol = 0.0
    # Region 0 (Inner Cylinder)
    vol += np.pi * (a[0]**2) * d[0]
    
    # Annular Regions
    for i in range(1, len(a)):
        annulus_area = np.pi * (a[i]**2 - a[i-1]**2)
        vol += annulus_area * d[i]
        
    return vol

def extract_scalar(data):
    """
    Recursively extracts a single scalar value from a potentially nested structure
    (float, numpy array, xarray, or dictionary).
    """
    if isinstance(data, (float, int, np.floating, np.integer)):
        return float(data)
    if isinstance(data, dict):
        if len(data) == 0: return 0.0
        first_val = next(iter(data.values()))
        return extract_scalar(first_val)
    if hasattr(data, "values") and not callable(data.values):
        return extract_scalar(data.values)
    if isinstance(data, np.ndarray):
        if data.size == 1: return float(data.item())
        else: return float(data.flatten()[0])
    try:
        return float(data)
    except Exception:
        raise TypeError(f"Cannot extract scalar from type {type(data)}: {data}")

# --- 4. EXECUTION SCRIPT ---
def regenerate():
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    m0_values = np.linspace(0.1, 6.0, 15)
    
    solver = CapytaineSlantSolver(mesh=False, panel_count=False, hydros=False, times=False, phase=False)

    for name, cfg in CONFIGS.items():
        print(f"Regenerating {name}...")
        
        am_list = []
        dp_list = []
        
        a = cfg['a']
        d = cfg['d']
        heaving = [bool(x) for x in cfg['heaving']]
        h = cfg['h']
        t_densities = [30] * len(a)
        face_units = 30
        
        # --- Pre-calculate Volume for Normalization ---
        # Note: 'a' here matches 'scaled_a' in the test if Radius Scaling is 1.0
        disp_vol = calculate_displacement(a, d)

        for m0 in m0_values:
            result, _, _, _, _ = solver.construct_and_solve(
                a=a, d_in=d, d_out=d, heaving=heaving, 
                t_densities=t_densities, face_units=face_units, 
                h=h, m0=m0, rho=RHO, reps=1
            )
            
            # Robust Extraction
            am_dim = extract_scalar(result.added_mass)
            dp_dim = extract_scalar(result.radiation_damping)
            
            # Non-dimensionalize using Displacement Volume
            # Standard: Coeff = Dimensional / (rho * Volume)
            omega = np.sqrt(m0 * np.tanh(m0 * h) * G)
            
            am_nondim = am_dim / (RHO * disp_vol)
            dp_nondim = dp_dim / (omega * RHO * disp_vol)
            
            am_list.append(am_nondim)
            dp_list.append(dp_nondim)
            
        df = pd.DataFrame({
            "m0": m0_values,
            "pyCapytaineMu_x": m0_values,
            "pyCapytaineMu_y": am_list,
            "pyCapytaineLambda_y": dp_list
        })
        
        output_path = DATA_FOLDER / f"{name}_regenerated_v2.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    regenerate()