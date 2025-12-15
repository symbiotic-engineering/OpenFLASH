import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import warnings # To suppress plotting warnings
from typing import Optional, List, Dict, Any, Tuple

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
        "NMK": [15, 15, 15], # 2 radii + exterior
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
        "NMK": [15] * 6, # 5 radii + exterior
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
    "config4": {
        "h": 1.001,
        "a": np.array([0.5, 1]),
        "d": np.array([0.5, 0.25]),
        "heaving_map": [False, True],
        "body_map": [0, 1],
        "m0": 1.0,
        "NMK": [15] * 3, # 2 radii + exterior
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
        "NMK": [15] * 3, # 2 radii + exterior
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


def run_openflash_sim(config_name, R_range: Optional[np.ndarray] = None, Z_range: Optional[np.ndarray] = None, heaving_map_override: Optional[List[bool]] = None) -> Tuple[Dict[str, Any], float]:
    """
    Runs the openflash simulation for a specific config to get the potential field.
    
    Args:
        config_name (str): Name of the configuration.
        R_range (np.ndarray): Array of R coordinates.
        Z_range (np.ndarray): Array of Z coordinates.
        heaving_map_override (list, optional): Force a specific heaving map (to satisfy single-body assertion).
    
    Returns:
        dict: A dictionary containing the results 'R', 'Z', 'phi'
        float: The calculated angular frequency 'omega'
    """
    if config_name not in ALL_CONFIGS:
        pytest.fail(f"Unknown config_name: {config_name}")
        
    p = ALL_CONFIGS[config_name]
    
    # Use the override map if provided, otherwise use the config default
    active_heaving_map = heaving_map_override if heaving_map_override is not None else p["heaving_map"]
    
    # 1. Create Geometry
    geometry = BasicRegionGeometry.from_vectors(
        a=p["a"],
        d=p["d"],
        h=p["h"],
        NMK=p["NMK"],
        body_map=p["body_map"],
        heaving_map=active_heaving_map
    )

    # 2. Create Problem
    problem = MEEMProblem(geometry)
    
    # 3. Set Frequency
    omega = openflash_omega(p["m0"], p["h"], g)
    problem.set_frequencies(np.array([omega]))

    # 4. Create Engine
    engine = MEEMEngine(problem_list=[problem])
    
    # 5. Calculate Potential Field
    # We need to solve for the *specific* problem defined in the config
    solution_vector = engine.solve_linear_system_multi(problem, p["m0"])

    
    potentials_dict = engine.calculate_potentials(
        problem, 
        solution_vector, 
        p["m0"], 
        spatial_res=50, 
        sharp=False,
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


def save_1d_cuts(R_grid, Z_grid, openflash_data, capytaine_data, config_name, component_name):
    """
    Saves 1D line cuts (slices) through the domain to visualize 
    profile differences in detail.
    """
    # Create directory
    DEBUG_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Vertical Cut (Z-axis) ---
    # Find index closest to R = 0.5 * max_radius (approx middle of fluid domain radially)
    # We want a slice that passes through the fluid, avoiding the body if possible
    # For now, simplistic approach: use the middle index of the R dimension
    r_idx = R_grid.shape[0] // 2
    r_val = R_grid[r_idx, 0]
    
    z_line = Z_grid[r_idx, :]
    of_line_z = openflash_data[r_idx, :]
    cap_line_z = capytaine_data[r_idx, :]
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_line, of_line_z, 'b-', label='OpenFLASH', linewidth=2)
    plt.plot(z_line, cap_line_z, 'r--', label='Capytaine', linewidth=2)
    plt.title(f"Vertical Slice (Z-axis) at R={r_val:.2f} [{component_name}]")
    plt.xlabel("Z (Depth)")
    plt.ylabel("Potential")
    plt.legend()
    plt.grid(True)
    plt.savefig(DEBUG_PLOT_PATH / f"{config_name}_{component_name}_cut_vertical.png")
    plt.close()

    # --- 2. Radial Cut (R-axis) ---
    # Cut at Z = -h/2 (Mid-depth)
    z_target = np.min(Z_grid) / 2.0
    z_idx = np.argmin(np.abs(Z_grid[0, :] - z_target))
    z_val = Z_grid[0, z_idx]
    
    r_line = R_grid[:, z_idx]
    of_line_r = openflash_data[:, z_idx]
    cap_line_r = capytaine_data[:, z_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(r_line, of_line_r, 'b-', label='OpenFLASH', linewidth=2)
    plt.plot(r_line, cap_line_r, 'r--', label='Capytaine', linewidth=2)
    plt.title(f"Radial Slice (R-axis) at Z={z_val:.2f} [{component_name}]")
    plt.xlabel("R (Radius)")
    plt.ylabel("Potential")
    plt.legend()
    plt.grid(True)
    plt.savefig(DEBUG_PLOT_PATH / f"{config_name}_{component_name}_cut_radial.png")
    plt.close()


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


def check_phase_rotation(openflash_complex, capytaine_complex):
    """
    Checks if a global phase rotation (e.g. -1, j, -j, conjugate)
    would minimize the error. This helps detect convention mismatches.
    """
    # Create flattened valid arrays (ignoring NaNs)
    valid = ~np.isnan(openflash_complex) & ~np.isnan(capytaine_complex)
    of = openflash_complex[valid]
    cap = capytaine_complex[valid]
    
    if of.size == 0:
        return "No Valid Data"

    transformations = {
        "None": of,
        "Negated (-1)": -of,
        "Conjugate (*)": np.conj(of),
        "Rotated 90 (j)": 1j * of,
        "Rotated -90 (-j)": -1j * of,
        "Conjugate & Negated": -np.conj(of)
    }
    
    print("\n--- PHASE / CONVENTION DIAGNOSTIC ---")
    best_name = "None"
    best_error = np.inf
    
    for name, transformed_of in transformations.items():
        # Calculate Relative L2 Norm Error
        diff_norm = np.linalg.norm(transformed_of - cap)
        ref_norm = np.linalg.norm(cap)
        rel_error = diff_norm / ref_norm if ref_norm > 0 else np.inf
        
        print(f"  Transform '{name}': Rel Error = {rel_error:.4%}")
        if rel_error < best_error:
            best_error = rel_error
            best_name = name
            
    print(f"  [DIAGNOSTIC] Best match is: '{best_name}'")
    return best_name


# --- Test Function (Hydro Coeff Test Removed) ---

@pytest.mark.parametrize("config_name", ALL_CONFIGS.keys())
def test_potential_field_vs_capytaine(config_name):
    """
    Compares the openflash-calculated potential field against the
    Capytaine-generated benchmark data FOR A GIVEN CONFIG.
    """
    
    # 1. Load data for this config
    phi_capytaine_raw = load_capytaine_data(config_name)
    p = ALL_CONFIGS[config_name]
    
    print(f"\n\n=== TESTING CONFIG: {config_name} ===")

    # --- IMPLEMENT SUPERPOSITION ---
    original_heaving_map = p["heaving_map"]
    heaving_indices = [i for i, is_heaving in enumerate(original_heaving_map) if is_heaving]
    
    phi_total: Optional[np.ndarray] = None
    omega_final: Optional[float] = None
    results_template: Optional[Dict[str, Any]] = None 

    if not heaving_indices:
        # Case: No heaving bodies
        res, omega_final = run_openflash_sim(config_name, R_range=p["R_range"], Z_range=p["Z_range"])
        phi_total = res["phi"]
        results_template = res
    else:
        # Loop through each heaving body and sum potentials
        for idx in heaving_indices:
            # Create a compliant heaving map (only one body True)
            single_heaving_map = [False] * len(original_heaving_map)
            single_heaving_map[idx] = True
            
            print(f"\n  [SUPERPOSITION DEBUG] Body {idx} Active:")
            print(f"    Map Override: {single_heaving_map}")
            
            # Run simulation
            res, omega = run_openflash_sim(
                config_name, 
                R_range=p["R_range"], 
                Z_range=p["Z_range"],
                heaving_map_override=single_heaving_map
            )
            print(f"Body {idx} OpenFlash max: {np.max(np.abs(res['phi']))}")
            print(f"Body {idx} Capytaine max: {np.max(np.abs(phi_capytaine_raw[..., idx]))}")  # If available
            
            # Check magnitude
            max_phi = np.nanmax(np.abs(res["phi"]))
            print(f"    Max |phi|: {max_phi:.6e}")
            
            if max_phi < 1e-10:
                print(f"    🚨 ALERT: Body {idx} produced ZERO potential! This is likely the bug.")
            
            # Save intermediate plot
            if np.any(res["phi"]):
                debug_dir = DEBUG_PLOT_PATH / "contributions"
                debug_dir.mkdir(parents=True, exist_ok=True)
                plt.figure()
                plt.pcolormesh(res["R"], res["Z"], res["phi"].real, cmap='viridis')
                plt.colorbar(label="Real(phi)")
                plt.title(f"{config_name} - Body {idx} Contribution")
                plt.savefig(debug_dir / f"{config_name}_body_{idx}_real.png")
                plt.close()

            if phi_total is None:
                phi_total = np.zeros_like(res["phi"], dtype=complex)
                omega_final = omega
                results_template = res
            
            phi_total += res["phi"]

    if results_template is None or phi_total is None or omega_final is None:
        pytest.fail(f"[{config_name}] Simulation failed to produce results.")
    
    # 2. Get the openflash grid and total potential
    R_openflash = results_template["R"]
    Z_openflash = results_template["Z"]
    phi_openflash = phi_total
    print("phi_total before sum:", np.nanmax(np.abs(phi_total)))
    print("contribution max:", np.nanmax(np.abs(res['phi'])))

    # 3. Define the Capytaine grid
    R_cap_grid, Z_cap_grid = np.meshgrid(p["R_range"], p["Z_range"], indexing='ij')
    print("OpenFlash phi shape:", phi_openflash.shape)
    print("Capytaine grid shape:", R_cap_grid.shape)

    # 4. Compare (No Interpolation Needed)
    phi_openflash_interp_real = phi_openflash.real
    phi_openflash_interp_imag = phi_openflash.imag
    omega = omega_final

    # 5. Validation & Conversion
    capytaine_body_mask = np.isnan(phi_capytaine_raw.real)
    openflash_nan_real = np.isnan(phi_openflash_interp_real)
    
    # Check for mismatched NaNs (Solver failing to mask body)
    bad_nans_real_mask = ~capytaine_body_mask & openflash_nan_real
    if np.sum(bad_nans_real_mask) > 0:
        pytest.fail(f"[{config_name}] Openflash produced NaNs in valid fluid domain.")

    nan_mask = np.isnan(phi_capytaine_raw.real)
    valid_mask = ~nan_mask
    
    if np.sum(valid_mask) < 0.5 * nan_mask.size:
        pytest.fail(f"[{config_name}] Interpolation failed: >50% of grid points are invalid.")

    # Convert Capytaine to Velocity Potential
    capytaine_real_converted = phi_capytaine_raw.imag * (-1.0 / omega)
    capytaine_imag_converted = phi_capytaine_raw.real * (1.0 / omega)

    openflash_real_valid = phi_openflash_interp_real[valid_mask]
    capytaine_real_valid = capytaine_real_converted[valid_mask]
    openflash_imag_valid = phi_openflash_interp_imag[valid_mask]
    capytaine_imag_valid = capytaine_imag_converted[valid_mask]

    # --- DIAGNOSTICS ---
    print(f"\n  [FINAL COMPARISON] {config_name}")
    print(f"    Omega: {omega:.4f}")
    print(f"    Max Abs OpenFlash (Real): {np.max(np.abs(openflash_real_valid)):.6e}")
    print(f"    Max Abs Capytaine (Real): {np.max(np.abs(capytaine_real_valid)):.6e}")
    
    # Save Final Plots
    save_debug_plots(R_cap_grid, Z_cap_grid, phi_openflash_interp_real, capytaine_real_converted, nan_mask, config_name, "real")
    save_debug_plots(R_cap_grid, Z_cap_grid, phi_openflash_interp_imag, capytaine_imag_converted, nan_mask, config_name, "imag")
    save_1d_cuts(R_cap_grid, Z_cap_grid, phi_openflash_interp_real, capytaine_real_converted, config_name, "real")
    save_1d_cuts(R_cap_grid, Z_cap_grid, phi_openflash_interp_imag, capytaine_imag_converted, config_name, "imag")

    # 6. Assertions
    try:
        np.testing.assert_allclose(
            openflash_real_valid, capytaine_real_valid,
            rtol=RELATIVE_TOLERANCE, atol=1e-2,
            err_msg=f"[{config_name}] Real part mismatch"
        )
    except AssertionError as e:
        pytest.fail(str(e))
    
    try:
        np.testing.assert_allclose(
            openflash_imag_valid, capytaine_imag_valid,
            rtol=RELATIVE_TOLERANCE, atol=1e-2,
            err_msg=f"[{config_name}] Imaginary part mismatch"
        )
    except AssertionError as e:
        pytest.fail(str(e))