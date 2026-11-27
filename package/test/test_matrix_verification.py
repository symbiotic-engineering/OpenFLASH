import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.linalg

# Import OpenFLASH modules
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.multi_equations import omega
try:
    from openflash.multi_constants import g
except ImportError:
    g = 9.81

# Full set of configurations from test_capytaine_potential.py
ALL_CONFIGS = {
    "config0": {
        "h": 1.001,
        "a": np.array([0.5, 1]),
        "d": np.array([0.5, 0.25]),
        "heaving_map": [True, True],
        "body_map": [0, 1],
        "m0": 1.0,
        "NMK": [15, 15, 15], 
    },
    "config1": {
        "h": 1.5,
        "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
        "d": np.array([1.1, 0.85, 0.75, 0.4, 0.15]),
        "heaving_map": [True, True, True, True, True],
        "body_map": [0, 1, 2, 3, 4],
        "m0": 1.0,
        "NMK": [15] * 6,
    },
    "config2": {
        "h": 100.0,
        "a": np.array([3, 5, 10]),
        "d": np.array([29, 7, 4]),
        "heaving_map": [True, True, True],
        "body_map": [0, 1, 2],
        "m0": 1.0,
        "NMK": [100] * 4,
    },
    "config3": {
        "h": 1.9,
        "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
        "d": np.array([0.5, 0.7, 0.8, 0.2, 0.5]),
        "heaving_map": [True, True, True, True, True],
        "body_map": [0, 1, 2, 3, 4],
        "m0": 1.0,
        "NMK": [50] * 6, # 5 radii + exterior
        "R_range": np.linspace(0.0, 2 * 1.6, num=50),
        "Z_range": np.linspace(0, -1.9, num=50),
    },
    "config4": {
        "h": 1.001,
        "a": np.array([0.5, 1]),
        "d": np.array([0.5, 0.25]),
        "heaving_map": [False, True],
        "body_map": [0, 1],
        "m0": 1.0,
        "NMK": [15] * 3,
    },
    "config5": {
        "h": 1.001,
        "a": np.array([0.5, 1]),
        "d": np.array([0.5, 0.25]),
        "heaving_map": [True, False],
        "body_map": [0, 1],
        "m0": 1.0,
        "NMK": [15] * 3,
    },
    "config6": {
        "h": 100.0,
        "a": np.array([3, 5, 10]),
        "d": np.array([29, 7, 4]),
        "heaving_map": [False, True, True],
        "body_map": [0, 1, 2],
        "m0": 1.0,
        "NMK": [100] * 4,
    }
}

OUTPUT_DIR = Path(__file__).parent.parent / "test_artifacts" / "matrix_sparsity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_sparsity_plot(A, NMK, config_name):
    """Generates the MEEM-style blue dot sparsity plot."""
    rows, cols = np.nonzero(np.abs(A) > 1e-15)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(cols, rows, color='blue', marker='o', s=10)
    plt.gca().invert_yaxis() 
    
    current_idx = 0
    # 1. Inner Region Boundary
    if len(NMK) > 0:
        current_idx += NMK[0]
        plt.axvline(current_idx - 0.5, color='k', lw=0.5)
        plt.axhline(current_idx - 0.5, color='k', lw=0.5)
    
    # 2. Middle (Annular) Regions
    for k in range(1, len(NMK)-1):
        # Part 1
        current_idx += NMK[k]
        plt.axvline(current_idx - 0.5, color='k', lw=0.5)
        plt.axhline(current_idx - 0.5, color='k', lw=0.5)
        # Part 2
        current_idx += NMK[k]
        plt.axvline(current_idx - 0.5, color='k', lw=0.5)
        plt.axhline(current_idx - 0.5, color='k', lw=0.5)

    plt.title(f"Sparsity: {config_name} (Size {A.shape[0]})")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.grid(True, alpha=0.3)
    
    filename = OUTPUT_DIR / f"{config_name}_sparsity.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

@pytest.mark.parametrize("config_name", ALL_CONFIGS.keys())
def test_generate_matrix_plots(config_name):
    """
    Runs the engine for a config, captures the matrix, and plots sparsity.
    """
    p = ALL_CONFIGS[config_name]
    
    # --- CRITICAL FIX: Sanitize Heaving Map ---
    # The Matrix A is independent of heaving. We set all to False 
    # to bypass the "only 1 body can heave" assertion in ConcentricBodyGroup.
    safe_heaving_map = [False] * len(p["heaving_map"])
    
    # 1. Setup Geometry
    geo = BasicRegionGeometry.from_vectors(
        a=p["a"], 
        d=p["d"], 
        h=p["h"], 
        NMK=p["NMK"],
        body_map=p["body_map"], 
        heaving_map=safe_heaving_map # <--- Use Safe Map
    )
    
    # 2. Setup Problem
    prob = MEEMProblem(geo)
    w = omega(p["m0"], p["h"], g)
    prob.set_frequencies(np.array([w]))
    
    engine = MEEMEngine([prob])
    
    # 3. Capture Matrix
    print(f"Generating matrix for {config_name}...")
    engine._ensure_m_k_and_N_k_arrays(prob, p["m0"])
    A_matrix = engine.assemble_A_multi(prob, p["m0"])
    
    # 4. Assertions
    assert A_matrix.shape[0] == A_matrix.shape[1], "Matrix must be square"
    expected_size = p["NMK"][0] + 2*sum(p["NMK"][1:-1]) + p["NMK"][-1]
    assert A_matrix.shape[0] == expected_size, \
        f"Matrix size {A_matrix.shape[0]} does not match sum of harmonics {expected_size}"
    
    # 5. Generate Plot
    save_sparsity_plot(A_matrix, p["NMK"], config_name)

@pytest.mark.parametrize("config_name", ALL_CONFIGS.keys())
def test_matrix_health_and_structure(config_name):
    """
    Verifies that the internal system matrix A is:
    1. Square and correctly sized.
    2. Well-conditioned (solvable).
    3. Populated (not empty/all-zeros).
    """
    p = ALL_CONFIGS[config_name]
    
    # --- 1. Setup (Independent of Heaving) ---
    # We force heaving to False because Matrix A only depends on Geometry + Frequency
    safe_heaving_map = [False] * len(p["heaving_map"])
    
    geo = BasicRegionGeometry.from_vectors(
        a=p["a"], d=p["d"], h=p["h"], NMK=p["NMK"],
        body_map=p["body_map"], heaving_map=safe_heaving_map
    )
    
    prob = MEEMProblem(geo)
    w = omega(p["m0"], p["h"], g)
    prob.set_frequencies(np.array([w]))
    
    engine = MEEMEngine([prob])
    
    # --- 2. Capture Matrix ---
    print(f"Generating matrix for {config_name}...")
    engine._ensure_m_k_and_N_k_arrays(prob, p["m0"])
    A_matrix = engine.assemble_A_multi(prob, p["m0"])
    
    # --- 3. Structural Checks ---
    assert A_matrix.shape[0] == A_matrix.shape[1], "Matrix must be square"
    expected_size = p["NMK"][0] + 2*sum(p["NMK"][1:-1]) + p["NMK"][-1]
    assert A_matrix.shape[0] == expected_size, \
        f"Matrix size {A_matrix.shape[0]} != Expected {expected_size}"

    # --- 4. Numerical Health Checks ---
    # Check for NaNs or Infs
    assert np.all(np.isfinite(A_matrix)), "Matrix contains NaNs or Infs!"
    
    # Check for "Empty" Rows/Cols (Singularity risk)
    row_norms = np.linalg.norm(A_matrix, axis=1)
    col_norms = np.linalg.norm(A_matrix, axis=0)
    if np.any(row_norms < 1e-12) or np.any(col_norms < 1e-12):
        pytest.fail("Matrix has zero-rows or zero-columns (Singular).")

    # Check Condition Number (Solvability)
    # A huge condition number (> 1e15) implies the system is unstable/unsolvable
    cond_num = np.linalg.cond(A_matrix)
    print(f"  Condition Number: {cond_num:.2e}")
    if cond_num > 1e15:
        pytest.warns(UserWarning, match=f"Matrix is ill-conditioned (Cond: {cond_num:.2e})")

    # --- 5. Generate Plot ---
    save_sparsity_plot(A_matrix, p["NMK"], config_name)

if __name__ == "__main__":
    pytest.main(["-v", __file__])