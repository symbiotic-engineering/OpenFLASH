import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import warnings # To suppress plotting warnings
from typing import Optional

# Import your package's classes
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.multi_equations import omega as openflash_omega

# Constants from the Capytaine script
try:
    from openflash.multi_constants import g
except ImportError:
    g = 9.81

# --- Test Configuration ---

# 1. Define path to the "golden" benchmark data
BENCHMARK_DATA_PATH = Path(__file__).parent.parent.parent / "dev" / "python" / "test" / "data"

# 2. Define path for saving debug plots on failure
DEBUG_PLOT_PATH = Path(__file__).parent.parent / "test_artifacts"

# 3. Define all configuration parameters (Hydro Coeffs Removed)
ALL_CONFIGS = {
    "config0": {
        "h": 1.001,
        "a": np.array([0.5, 1]),
        "d": np.array([0.5, 0.25]),
        "heaving_map": [True, True],
        "body_map": [0, 1],
        "m0": 1.0,
        "NMK": [40, 40, 40], # 2 radii + exterior
        "R_range": np.linspace(0.0, 2 * 1, num=50),
        "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config1": {
        "h": 1.5,
        "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
        "d": np.array([1.1, 0.85, 0.75, 0.4, 0.15]),
        "heaving_map": [True, True, True, True, True],
        "body_map": [0, 1, 2, 3, 4],
        "m0": 1.0,
        "NMK": [40] * 6, # 5 radii + exterior
        "R_range": np.linspace(0.0, 2 * 1.6, num=50),
        "Z_range": np.linspace(0, -1.5, num=50),
    },
    "config2": {
        "h": 100.0,
        "a": np.array([3, 5, 10]),
        "d": np.array([29, 7, 4]),
        "heaving_map": [True, True, True],
        "body_map": [0, 1, 2],
        "m0": 1.0,
        "NMK": [40] * 4, # 3 radii + exterior
        "R_range": np.linspace(0.0, 2 * 10, num=50),
        "Z_range": np.linspace(0, -100, num=50),
    },
    "config3": {
        "h": 1.9,
        "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
        "d": np.array([0.5, 0.7, 0.8, 0.2, 0.5]),
        "heaving_map": [True, True, True, True, True],
        "body_map": [0, 1, 2, 3, 4],
        "m0": 1.0,
        "NMK": [40] * 6, # 5 radii + exterior
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
        "NMK": [40] * 3, # 2 radii + exterior
        "R_range": np.linspace(0.0, 2 * 1, num=50),
        "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config5": {
        "h": 1.001,
        "a": np.array([0.5, 1]),
        "d": np.array([0.5, 0.25]),
        "heaving_map": [True, False],
        "body_map": [0, 1],
        "m0": 1.0,
        "NMK": [40] * 3, # 2 radii + exterior
        "R_range": np.linspace(0.0, 2 * 1, num=50),
        "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config6": {
        "h": 100.0,
        "a": np.array([3, 5, 10]),
        "d": np.array([29, 7, 4]),
        "heaving_map": [False, True, True],
        "body_map": [0, 1, 2],
        "m0": 1.0,
        "NMK": [40] * 4, # 3 radii + exterior
        "R_range": np.linspace(0.0, 2 * 10, num=50),
        "Z_range": np.linspace(0, -100, num=50),
    }
}

# 4. Define comparison tolerance
RELATIVE_TOLERANCE = 1.5e-1 

# --- End Configuration ---

# --- Helper Functions ---

def load_capytaine_data(config_name):
    """
    Loads the "golden" potential field data for a specific config.
    """
    real_path = BENCHMARK_DATA_PATH / f"{config_name}-real.csv"
    imag_path = BENCHMARK_DATA_PATH / f"{config_name}-imag.csv"
    
    if not real_path.exists():
        pytest.skip(f"Benchmark file not found: {real_path}")
    if not imag_path.exists():
        pytest.skip(f"Benchmark file not found: {imag_path}")

    try:
        real_data = np.loadtxt(real_path, delimiter=",")
        imag_data = np.loadtxt(imag_path, delimiter=",")
        potential_field = real_data + 1j * imag_data
        return potential_field
    
    except Exception as e:
        pytest.fail(f"Failed to load benchmark data for {config_name}: {e}")


# --- MODIFIED FUNCTION ---
def run_openflash_sim(config_name, R_range: Optional[np.ndarray] = None, Z_range: Optional[np.ndarray] = None):
    """
    Runs the openflash simulation for a specific config to get the potential field.
    
    Returns:
        dict: A dictionary containing the results 'R', 'Z', 'phi'
        float: The calculated angular frequency 'omega'
    """
    if config_name not in ALL_CONFIGS:
        pytest.fail(f"Unknown config_name: {config_name}")
        
    p = ALL_CONFIGS[config_name]
    
    # 1. Create Geometry
    geometry = BasicRegionGeometry.from_vectors(
        a=p["a"],
        d=p["d"],
        h=p["h"],
        NMK=p["NMK"],
        body_map=p["body_map"],
        heaving_map=p["heaving_map"]
    )

    # 2. Create Problem
    problem = MEEMProblem(geometry)
    
    # 3. Set Frequency
    omega = openflash_omega(p["m0"], p["h"], g)
    problem.set_frequencies(np.array([omega]))

    # 4. Create Engine
    engine = MEEMEngine(problem_list=[problem])
    
    # --- Hydro Coeff calculation removed ---

    # 5. Calculate Potential Field
    # We need to solve for the *specific* problem defined in the config
    # (which might have multiple bodies heaving at once for the potential field)
    solution_vector = engine.solve_linear_system_multi(problem, p["m0"])
    
    potentials_dict = engine.calculate_potentials(
        problem, 
        solution_vector, 
        p["m0"], 
        spatial_res=50, 
        sharp=False, 
        R_range=R_range,
        Z_range=Z_range
    )
    
    # Return the grid/potential AND the frequency
    results_dict = {
        "R": potentials_dict["R"],
        "Z": potentials_dict["Z"],
        "phi": potentials_dict["phi"],
    }
    
    return results_dict, omega


def save_debug_plots(R_grid, Z_grid, openflash_data, capytaine_data_converted, nan_mask, config_name, plot_type):
    """
    Saves a 4-panel comparison plot to the debug artifact folder.
    """
    # Create the output directory if it doesn't exist
    DEBUG_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = DEBUG_PLOT_PATH / f"{config_name}_{plot_type}_comparison.png"

    # --- Prepare data for plotting ---
    
    # Calculate differences
    diff = openflash_data - capytaine_data_converted
    
    # Calculate percent difference, avoiding division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) # Ignore "divide by zero"
        percent_diff = 100 * (diff / capytaine_data_converted)
    
    # Apply the 'nan' mask from the body to all plots
    openflash_data[nan_mask] = np.nan
    capytaine_data_converted[nan_mask] = np.nan
    diff[nan_mask] = np.nan
    percent_diff[nan_mask] = np.nan
    
    # Find common min/max for the main plots
    v_min = np.nanmin([openflash_data, capytaine_data_converted])
    v_max = np.nanmax([openflash_data, capytaine_data_converted])
    
    # Find min/max for the difference plots
    diff_vmax = np.nanmax(np.abs(diff))
    perc_vmax = np.nanmin([500, np.nanmax(np.abs(percent_diff))]) # Cap at 500%

    # --- Create the Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Debug Comparison for: {config_name} ({plot_type.upper()})", fontsize=16)

    # Plot 1: Openflash
    ax1 = axes[0, 0]
    im1 = ax1.pcolormesh(R_grid, Z_grid, openflash_data, vmin=v_min, vmax=v_max, cmap='viridis')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("Openflash (ACTUAL)")

    # Plot 2: Capytaine (CONVERTED)
    ax2 = axes[0, 1]
    im2 = ax2.pcolormesh(R_grid, Z_grid, capytaine_data_converted, vmin=v_min, vmax=v_max, cmap='viridis')
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Capytaine (CONVERTED TO OPENFLASH UNITS)")

    # Plot 3: Absolute Difference
    ax3 = axes[1, 0]
    im3 = ax3.pcolormesh(R_grid, Z_grid, np.abs(diff), vmin=0, vmax=diff_vmax, cmap='Reds')
    fig.colorbar(im3, ax=ax3)
    ax3.set_title("Absolute Difference |Actual - Desired|")

    # Plot 4: Percent Difference
    ax4 = axes[1, 1]
    im4 = ax4.pcolormesh(R_grid, Z_grid, np.abs(percent_diff), vmin=0, vmax=perc_vmax, cmap='Reds')
    fig.colorbar(im4, ax=ax4)
    ax4.set_title("Percent Difference |Diff / Desired| (Capped at 500%)")
    
    for ax in axes.flat:
        ax.set_xlabel("R (radius)")
        ax.set_ylabel("Z (depth)")
    
    # Fix for Pylance TypeError
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_file)
    plt.close(fig)
    print(f"\n[Debug plot saved to: {output_file}]")


def save_debug_csvs(R_grid, Z_grid, openflash_real, capytaine_real, openflash_imag, capytaine_imag, nan_mask, config_name):
    """
    Saves a detailed CSV of all interpolated grid data for debugging.
    """
    DEBUG_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = DEBUG_PLOT_PATH / f"{config_name}_debug_data.csv"
    
    # Create a structured DataFrame
    data = {
        "R": R_grid.ravel(),
        "Z": Z_grid.ravel(),
        "is_body_nan": nan_mask.ravel(),
        "openflash_real": openflash_real.ravel(),
        "capytaine_real_converted": capytaine_real.ravel(),
        "openflash_imag": openflash_imag.ravel(),
        "capytaine_imag_converted": capytaine_imag.ravel(),
    }
    df = pd.DataFrame(data)
    
    # Calculate difference columns
    df["diff_real"] = df["openflash_real"] - df["capytaine_real_converted"]
    df["diff_imag"] = df["openflash_imag"] - df["capytaine_imag_converted"]
    
    # Calculate relative difference, handling division by zero
    df["rel_diff_real"] = np.where(
        df["capytaine_real_converted"] != 0,
        100 * np.abs(df["diff_real"] / df["capytaine_real_converted"]),
        np.inf
    )
    df["rel_diff_imag"] = np.where(
        df["capytaine_imag_converted"] != 0,
        100 * np.abs(df["diff_imag"] / df["capytaine_imag_converted"]),
        np.inf
    )
    
    # Filter out the points inside the body (where capytaine is NaN)
    # This lets you focus on the valid fluid domain
    df_fluid_only = df[~df["is_body_nan"]].copy()

    # Save the filtered data
    df_fluid_only.to_csv(output_file, index=False, float_format="%.6e")
    print(f"[Debug CSV saved to: {output_file}]")


# --- Test Function (Hydro Coeff Test Removed) ---

@pytest.mark.parametrize("config_name", ALL_CONFIGS.keys())
def test_potential_field_vs_capytaine(config_name):
    """
    Compares the openflash-calculated potential field against the
    Capytaine-generated benchmark data FOR A GIVEN CONFIG.
    
    On failure, this test will save debug plots to the 'test_artifacts' folder.
    """
    
    # 1. Load data for this config
    phi_capytaine_raw = load_capytaine_data(config_name)
    # --- MODIFIED ORDER ---
    # Get params for this config FIRST
    p = ALL_CONFIGS[config_name] 
    # Pass the ranges to the sim function
    openflash_results, omega = run_openflash_sim(config_name, R_range=p["R_range"], Z_range=p["Z_range"])
    
    # 2. Get the openflash grid and total potential
    R_openflash = openflash_results["R"]
    Z_openflash = openflash_results["Z"]
    phi_openflash = openflash_results["phi"]

    # 3. Define the Capytaine grid (where we want to interpolate to)
    R_cap_grid, Z_cap_grid = np.meshgrid(
        p["R_range"],
        p["Z_range"],
        indexing='ij'
    )

    # 4. No interpolation needed! 
    # Because we passed R_range and Z_range to the engine,
    # the grids are identical. We can compare the arrays directly.
    
    phi_openflash_interp_real = phi_openflash.real
    phi_openflash_interp_imag = phi_openflash.imag

    # --- (NEW) 5a. Check for unexpected NaNs from Openflash interpolation ---
    # This explicitly checks your question: "are there values in openflash that return nan when they shouldnt?"
    capytaine_body_mask = np.isnan(phi_capytaine_raw.real)
    openflash_nan_real = np.isnan(phi_openflash_interp_real)
    openflash_nan_imag = np.isnan(phi_openflash_interp_imag)

    # Check real part: Points where Capytaine is NOT NaN, but Openflash IS NaN
    bad_nans_real_mask = ~capytaine_body_mask & openflash_nan_real
    num_bad_nans_real = np.sum(bad_nans_real_mask)
    if num_bad_nans_real > 0:
        pytest.fail(f"[{config_name}] Openflash (Real) produced {num_bad_nans_real} NaNs in the valid fluid domain where Capytaine had data.")

    # Check imag part: Points where Capytaine is NOT NaN, but Openflash IS NaN
    bad_nans_imag_mask = ~capytaine_body_mask & openflash_nan_imag
    num_bad_nans_imag = np.sum(bad_nans_imag_mask)
    if num_bad_nans_imag > 0:
        pytest.fail(f"[{config_name}] Openflash (Imag) produced {num_bad_nans_imag} NaNs in the valid fluid domain where Capytaine had data.")
    # --- END NEW 5a ---

    # 5b. Mask out the 'nan' values (inside the body) from the Capytaine data
    # (This is the original Step 5)
    nan_mask = np.isnan(phi_capytaine_raw.real)
    valid_mask = ~nan_mask & ~np.isnan(phi_openflash_interp_real)
    
    if np.sum(valid_mask) < 0.5 * nan_mask.size:
        pytest.fail(f"[{config_name}] Interpolation failed: >50% of grid points are invalid.")

    # 6. Convert Capytaine data to Openflash units BEFORE comparing
    capytaine_real_converted = phi_capytaine_raw.imag * (-1.0 / omega)
    capytaine_imag_converted = phi_capytaine_raw.real * (1.0 / omega)

    # Get the valid (non-body) points for comparison
    openflash_real_valid = phi_openflash_interp_real[valid_mask]
    capytaine_real_valid = capytaine_real_converted[valid_mask]
    
    openflash_imag_valid = phi_openflash_interp_imag[valid_mask]
    capytaine_imag_valid = capytaine_imag_converted[valid_mask]
    
    # --- ENHANCED DEBUGGING ---
    # [This section remains the same]
    # ... (calculating max_rel_diff, printing debug info, etc.) ...
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        real_abs_diff = np.abs(openflash_real_valid - capytaine_real_valid)
        real_rel_diff = np.abs(real_abs_diff / capytaine_real_valid)
        real_rel_diff[np.isinf(real_rel_diff)] = 0.0 
        real_rel_diff = np.nan_to_num(real_rel_diff)
    max_rel_diff_idx = np.argmax(real_rel_diff)
    max_rel_diff = real_rel_diff[max_rel_diff_idx]
    valid_R_coords = R_cap_grid[valid_mask]
    valid_Z_coords = Z_cap_grid[valid_mask]
    worst_R = valid_R_coords[max_rel_diff_idx]
    worst_Z = valid_Z_coords[max_rel_diff_idx]
    print(f"\n--- DEBUG INFO FOR: {config_name} ---")
    print(f"Comparing Total Potential (phi) vs. CONVERTED Capytaine Potential")
    print(f"Omega = {omega:.4f} rad/s")
    print(f"Max abs(openflash.real):       {np.max(np.abs(openflash_real_valid)):.6e}")
    print(f"Max abs(Capytaine_CONVERTED.real): {np.max(np.abs(capytaine_real_valid)):.6e}")
    max_capytaine_real = np.max(np.abs(capytaine_real_valid))
    if max_capytaine_real > 1e-9:
        print(f"Scaling Factor (openflash / Capytaine_CONVERTED): {np.max(np.abs(openflash_real_valid)) / max_capytaine_real:.2f}")
    print(f"WORST RELATIVE ERROR (REAL): {max_rel_diff:.2%} at (R={worst_R:.2f}, Z={worst_Z:.2f})")
    print(f"  -> Openflash val: {openflash_real_valid[max_rel_diff_idx]:.4e}")
    print(f"  -> Capytaine val: {capytaine_real_valid[max_rel_diff_idx]:.4e}")
    # --- END DEBUGGING ---
    

    # --- (NEW) ALWAYS SAVE ARTIFACTS ---
    # These functions are now called every time the test runs.
    
    # 1. Save REAL part plot
    save_debug_plots(
        R_cap_grid, Z_cap_grid, 
        phi_openflash_interp_real, capytaine_real_converted, 
        nan_mask, config_name, "real"
    )
    
    # 2. Save IMAGINARY part plot
    save_debug_plots(
        R_cap_grid, Z_cap_grid, 
        phi_openflash_interp_imag, capytaine_imag_converted, 
        nan_mask, config_name, "imag"
    )

    # 3. Save CSV data
    save_debug_csvs(
        R_cap_grid, Z_cap_grid,
        phi_openflash_interp_real, capytaine_real_converted,
        phi_openflash_interp_imag, capytaine_imag_converted,
        nan_mask, config_name
    )
    # --- END NEW SECTION ---

    
    # --- Test Real Part ---
    # The try/except block now *only* handles the assertion
    try:
        np.testing.assert_allclose(
            openflash_real_valid,
            capytaine_real_valid,
            rtol=RELATIVE_TOLERANCE,
            atol=1e-2,
            err_msg=f"[{config_name}] Real part of potential field does not match Capytaine benchmark."
        )
    except AssertionError as e:
        pytest.fail(str(e)) # Re-raise the assertion to fail the test
    
    # --- Test Imaginary Part ---
    # The try/except block now *only* handles the assertion
    try:
        np.testing.assert_allclose(
            openflash_imag_valid,
            capytaine_imag_valid,
            rtol=RELATIVE_TOLERANCE,
            atol=1e-2,
            err_msg=f"[{config_name}] Imaginary part of potential field does not match Capytaine benchmark."
        )
    except AssertionError as e:
        pytest.fail(str(e)) # Re-raise the assertion to fail the test