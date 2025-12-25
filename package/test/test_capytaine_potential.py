# package/test/test_capytaine_potential.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from scipy import integrate
import matplotlib.pyplot as plt
import warnings # To suppress plotting warnings
from typing import Optional, List, Dict, Any, Tuple

# Import your package's classes
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.multi_equations import *

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
        "NMK": [100] * 4, # 3 radii + exterior
        "R_range": np.linspace(0.0, 2 * 10, num=50),
        "Z_range": np.linspace(0, -100, num=50),
    },
    # "config3": {
    #     "h": 1.9,
    #     "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
    #     "d": np.array([0.5, 0.7, 0.8, 0.2, 0.5]),
    #     "heaving_map": [True, True, True, True, True],
    #     "body_map": [0, 1, 2, 3, 4],
    #     "m0": 1.0,
    #     "NMK": [50] * 6, # 5 radii + exterior
    #     "R_range": np.linspace(0.0, 2 * 1.6, num=50),
    #     "Z_range": np.linspace(0, -1.9, num=50),
    # },
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
        "NMK": [100] * 4, # 3 radii + exterior
        "R_range": np.linspace(0.0, 2 * 10, num=50),
        "Z_range": np.linspace(0, -100, num=50),
    }
}

# 4. Define comparison tolerance
RELATIVE_TOLERANCE = 0.1

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


def run_openflash_sim(config_name, R_range: Optional[np.ndarray] = None, Z_range: Optional[np.ndarray] = None, heaving_map_override: Optional[List[bool]] = None, verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Runs the openflash simulation for a specific config to get the potential field.
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
    openflash_omega = omega(p["m0"], p["h"], g)
    problem.set_frequencies(np.array([openflash_omega]))

    # 4. Create Engine
    engine = MEEMEngine(problem_list=[problem])
    
    # --- CONTROLLED PRINTING ---
    if verbose:
        diagnose_geometry_depths(geometry)
        # Loop through ALL valid regions instead of hardcoding '3'
        print(f"\n  [DEBUG] Running Orthogonality Checks for {len(geometry.domain_list)} regions...")
        for r_idx in geometry.domain_list.keys():
            debug_vertical_orthogonality(engine, problem, region_idx=r_idx, m0=p["m0"])
        print(f"    [Diagnosing Linear System for {config_name}]")
    # ---------------------------
    
    solution_vector = engine.solve_linear_system_multi(problem, p["m0"])
    
    # --- CONTROLLED PRINTING ---
    if verbose:
        diagnose_continuity(engine, problem, solution_vector, config_name, p["m0"])
        
        # --- NEW: Check Potential Continuity ---
        # Your previous logs showed Flux was perfect, but Potential was wrong.
        # This function checks if Phi is continuous (Pressure matching).
        diagnose_potential_continuity(engine, problem, solution_vector, config_name, p["m0"])
        audit_potential_matrix_continuity(engine, problem, solution_vector, config_name, p["m0"])
        
    # ---------------------------
    
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
        # Return engine/problem for further deep debugging
        "_engine": engine,
        "_problem": problem,
        "_solution": solution_vector,
        "_m0": p["m0"]
    }
    
    return results_dict, openflash_omega


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
    df_fluid_only = df[~df["is_body_nan"]].copy()

    # Save the filtered data
    df_fluid_only.to_csv(output_file, index=False, float_format="%.6e")
    print(f"[Debug CSV saved to: {output_file}]")

def audit_potential_matrix_continuity(engine, problem, solution_vector, config_name, m0):
    """
    6️⃣ ONE-LINE KILLER CHECK
    Reconstructs the P-matrix blocks locally and verifies:
       P_L @ C_L + P_R @ C_R == b  (Summation, because P_R block carries the -1 sign)
    """
    print(f"\n  [MATRIX BALANCE AUDIT - 6️⃣ KILLER CHECK] {config_name}")
    
    # --- FIX 1: Retrieve Properties from Domain List (Same as Engine) ---
    domains = problem.geometry.domain_list
    domain_keys = list(domains.keys()) 
    
    h = domains[0].h
    d = [domains[idx].di for idx in domain_keys] 
    a = [domains[idx].a for idx in domain_keys]
    heaving = [domains[idx].heaving for idx in domain_keys]
    
    # Get precomputed integrals
    engine._ensure_m_k_and_N_k_arrays(problem, m0)
    cache = engine.cache_list[problem]
    I_nm_vals = cache.I_nm_vals
    
    # Helper to reformat coefficients
    NMK = [domains[i].number_harmonics for i in range(len(domains))]
    all_coeffs = engine.reformat_coeffs(solution_vector, NMK, len(domains)-1)
    
    # Radial Functions
    from functools import partial
    from openflash.multi_equations import R_1n, R_2n, p_diagonal_block, p_dense_block, b_potential_entry
    
    R_1n_func = np.vectorize(partial(R_1n, h=h, d=d, a=a))
    R_2n_func = np.vectorize(partial(R_2n, a=a, h=h, d=d))

    boundary_count = len(domains) - 1
    
    for bd in range(boundary_count):
        # --- FIX 2: Skip Exterior Boundary immediately ---
        # The exterior boundary uses I_mk (m0-dependent) and different functions.
        # p_dense_block will crash with IndexError if used here.
        if bd == boundary_count - 1:
            print(f"    Boundary {bd}: Skipping (Exterior Boundary handled by different functions)")
            continue
        # -------------------------------------------------

        # 1. Identify Shorter Region
        project_on_left = (d[bd] > d[bd+1])
        
        # 2. Reconstruct P_L and P_R matrices
        if project_on_left:
            # Left (Diagonal) part
            P_L_R1 = p_diagonal_block(True, R_1n_func, bd, h, d, a, NMK)
            P_L_R2 = p_diagonal_block(True, R_2n_func, bd, h, d, a, NMK) if bd > 0 else None
            
            # Right (Dense) part
            P_R_R1 = p_dense_block(False, R_1n_func, bd, NMK, a, I_nm_vals)
            P_R_R2 = p_dense_block(False, R_2n_func, bd, NMK, a, I_nm_vals)
            
            c_L = all_coeffs[bd]
            c_R = all_coeffs[bd+1]
            
            if bd > 0:
                N = NMK[bd]
                term_L = P_L_R1 @ c_L[:N] + P_L_R2 @ c_L[N:]
            else:
                term_L = P_L_R1 @ c_L
            
            N_R = NMK[bd+1]
            term_R = P_R_R1 @ c_R[:N_R] + P_R_R2 @ c_R[N_R:]
            
            LHS = term_L + term_R 

        else: # Right is Shorter (Project on Right)
            
            # Left (Dense) part
            P_L_R1 = p_dense_block(True, R_1n_func, bd, NMK, a, I_nm_vals)
            P_L_R2 = p_dense_block(True, R_2n_func, bd, NMK, a, I_nm_vals) if bd > 0 else None
            
            # Right (Diagonal) part
            P_R_R1 = p_diagonal_block(False, R_1n_func, bd, h, d, a, NMK)
            P_R_R2 = p_diagonal_block(False, R_2n_func, bd, h, d, a, NMK)
            
            c_L = all_coeffs[bd]
            c_R = all_coeffs[bd+1]
            
            if bd > 0:
                N = NMK[bd]
                term_L = P_L_R1 @ c_L[:N] + P_L_R2 @ c_L[N:]
            else:
                term_L = P_L_R1 @ c_L
                
            N_R = NMK[bd+1]
            term_R = P_R_R1 @ c_R[:N_R] + P_R_R2 @ c_R[N_R:]
            
            LHS = term_L + term_R

        # 3. Reconstruct b vector (Forcing)
        num_rows = NMK[bd] if project_on_left else NMK[bd+1]
        b_vec = np.zeros(num_rows, dtype=complex)
        for n in range(num_rows):
            b_vec[n] = b_potential_entry(n, bd, heaving, a, h, d)
            
        # 4. THE KILLER CHECK
        diff = LHS - b_vec
        max_err = np.max(np.abs(diff))
        
        print(f"    Boundary {bd} (Project on {'LEFT' if project_on_left else 'RIGHT'}):")
        print(f"      Matrix Imbalance: {max_err:.4e}")
        
        try:
            np.testing.assert_allclose(
                LHS,
                b_vec,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"P-matrix mismatch at Boundary {bd}"
            )
            print("      ✅ PASS")
        except AssertionError as e:
            print("      ❌ FAIL")
            print(f"      Term L (Sample): {term_L[:3]}")
            print(f"      Term R (Sample): {term_R[:3]}")
            print(f"      b_vec  (Sample): {b_vec[:3]}")
            
def check_phase_rotation(openflash_complex, capytaine_complex):
    """
    Checks if a global phase rotation (e.g. -1, j, -j, conjugate)
    would minimize the error.
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

def debug_vertical_orthogonality(engine, problem, region_idx, m0):
    """
    Verifies vertical orthogonality for a specific region.
    Integrates Z_m * Z_n from -h to -d.
    """
    print(f"\n  [ORTHOGONALITY CHECK] Region {region_idx}")
    
    # 1. Setup Geometry Parameters
    geo = problem.geometry
    h = geo.h
    domains = geo.domain_list
    d_vals = geo.body_arrangement.d
    
    # Handle Exterior vs Interior
    is_exterior = (region_idx == len(d_vals))
    
    # Initialize the function variable
    get_Z = None
    d_local = 0.0

    if is_exterior:
        d_local = 0.0 # Seabed
        print(f"    Type: Exterior (d={d_local}, h={h})")
        
        # Ensure cache is ready for Exterior functions
        engine._ensure_m_k_and_N_k_arrays(problem, m0)
        cache = engine.cache_list[problem]
        m_k_arr, N_k_arr = cache.m_k_arr, cache.N_k_arr
        NMK = domains[region_idx].number_harmonics
        
        # Define specific exterior function
        def _get_z_exterior(mode, z):
            # Use vectorized version with precomputed arrays for stability
            # Wrap scalars in arrays for the vectorized function, then take [0]
            return Z_k_e_vectorized(np.array([mode]), np.array([z]), m0, h, m_k_arr, N_k_arr)[0]
            
        get_Z = _get_z_exterior

    else:
        d_local = d_vals[region_idx]
        print(f"    Type: Interior (d={d_local}, h={h})")
        
        # Define specific interior function
        def _get_z_interior(mode, z):
            return Z_n_i(mode, z, region_idx, h, d_vals)
            
        get_Z = _get_z_interior

    # 2. Integration Bounds
    z_min = -h
    z_max = -d_local
    
    # 3. Compute Inner Products
    print("    Inner Products <Zm, Zn> (Expect Diag ~ h-d, Off-Diag ~ 0):")
    
    for m in range(3):
        row_str = ""
        for n in range(3):
            # Define integrand using the assigned get_Z
            # Capture m and n by default value to avoid late binding issues in lambdas
            integrand = lambda z, m=m, n=n: get_Z(m, z) * get_Z(n, z)
            
            # Numerical Integration (Quad is precise enough)
            val_real, _ = integrate.quad(lambda z: np.real(integrand(z)), z_min, z_max)
            val_imag, _ = integrate.quad(lambda z: np.imag(integrand(z)), z_min, z_max)
            val = val_real + 1j*val_imag
            
            # Formatting
            if abs(val) < 1e-9: val = 0.0
            
            row_str += f"{val.real: .4e}  "
            
        print(f"    m={m}: [ {row_str}]")

    print(f"    (Theoretical H-d = {h - d_local:.4f})")
    
def diagnose_geometry_depths(geometry):
    """
     audits the depth consistency between the global Geometry object and individual Domain objects.
     Mismatches here are the #1 cause of 'Flux Passes, Potential Fails' errors.
    """
    print("\n  [DEPTH CONSISTENCY CHECK]")
    print(f"    Global Geometry h: {geometry.h}")
    
    # Handle list vs dict structure
    domains = geometry.fluid_domains
    if isinstance(domains, dict):
        domains = list(domains.values())
        
    for i, domain in enumerate(domains):
        # In OpenFLASH:
        # domain.h  -> Global Water Depth (d_upper)
        # domain.di -> Draft / Bottom Depth (d_lower)
        # Local Depth = domain.h - domain.di
        
        local_depth = domain.h - domain.di
        
        match_status = "✅" if abs(domain.h - geometry.h) < 1e-9 else "❌ MISMATCH"
        
        print(f"    Region {i} (Index {domain.index}):")
        print(f"      Global h (Geometry) : {geometry.h:.4f}")
        print(f"      Global h (Domain)   : {domain.h:.4f}  {match_status}")
        print(f"      Draft d (Domain)    : {domain.di:.4f}")
        print(f"      Local Depth (h - d) : {local_depth:.4f}")
        
        if abs(domain.h - geometry.h) > 1e-9:
             print(f"      🚨 CRITICAL ERROR: Domain {i} thinks global depth is {domain.h}, but Geometry says {geometry.h}!")

def diagnose_continuity(engine, problem, solution_vector, config_name, m0):
    """
    Checks if Mass Flux is conserved across boundaries.
    """
    print(f"\n  [CONTINUITY DIAGNOSTIC - FLUX] {config_name}")
    
    geo = problem.geometry
    domains = geo.domain_list
    a = geo.body_arrangement.a
    d = geo.body_arrangement.d
    h = geo.h
    boundary_count = len(domains) - 1
    
    for bd in range(boundary_count):
        radius = a[bd]
        
        d_left = d[bd]
        d_right = d[bd+1] if bd < len(d)-1 else d[bd]
        
        max_draft = max(d_left, d_right) 
        common_depth = h - max_draft
        
        z_common = np.linspace(0, -max_draft, 100)
        eps = 1e-4 
        
        # --- LEFT SIDE ---
        vel_left = engine.calculate_velocities(problem, solution_vector, m0, 10, False, 
                                               R_range=np.array([radius - eps]), Z_range=z_common)
        vr_left = vel_left['vr'].flatten()
        
        # --- RIGHT SIDE ---
        vel_right = engine.calculate_velocities(problem, solution_vector, m0, 10, False, 
                                                R_range=np.array([radius + eps]), Z_range=z_common)
        vr_right = vel_right['vr'].flatten()

        valid = ~np.isnan(vr_left) & ~np.isnan(vr_right)
        
        if not np.any(valid):
            print(f"    Boundary {bd}: NO VALID POINTS")
            continue
            
        flux_diff = integrate.simpson(np.abs(vr_left[valid] - vr_right[valid]), x=z_common[valid])
        total_flux = integrate.simpson(np.abs(vr_left[valid]), x=z_common[valid])
        
        mse = np.mean(np.abs(vr_left[valid] - vr_right[valid])**2)
        rms_diff = np.sqrt(mse)

        if total_flux > 1e-6:
            rel_err = flux_diff / total_flux
        else:
            rel_err = flux_diff
            
        status = "✅ PASS" if rel_err < 0.05 else "❌ FAIL"
            
        print(f"    Boundary {bd} (R={radius:.2f}): {status}")
        print(f"      Common Height: {max_draft:.2f}")
        print(f"      RMS Vel Diff : {rms_diff:.4e}")
        print(f"      Rel Flux Err : {rel_err*100:.2f}%")

        if d_left != d_right:
            step_top = -min(d_left, d_right)
            step_bot = -max(d_left, d_right)
            z_step = np.linspace(step_top - 0.01, step_bot + 0.01, 50)
            
            if d_left < d_right: 
                check_side = f"Left (Reg {bd})"
                vel_step = engine.calculate_velocities(problem, solution_vector, m0, 10, False, 
                                               R_range=np.array([radius - eps]), Z_range=z_step)
                target_vr = 0.0 
            else:
                check_side = f"Right (Reg {bd+1})"
                vel_step = engine.calculate_velocities(problem, solution_vector, m0, 10, False, 
                                               R_range=np.array([radius + eps]), Z_range=z_step)
                target_vr = 0.0

            vr_step = vel_step['vr'].flatten()
            valid_step = ~np.isnan(vr_step)
            
            if np.any(valid_step):
                leak_flux = integrate.simpson(np.abs(vr_step[valid_step] - target_vr), x=z_step[valid_step])
                step_height = abs(step_top - step_bot)
                avg_leak_vel = leak_flux / step_height if step_height > 0 else 0
                step_status = "✅" if avg_leak_vel < 0.1 else "⚠️ LEAK" 
                print(f"      Step Check ({check_side}): {step_status}")
                print(f"      Avg Leak Vel : {avg_leak_vel:.4e}")
                print(f"      Step Height  : {step_height:.4f}")

def debug_phi_reconstruction(coeffs, r, z, region_idx, h, d, a, label):
    """
    Reconstructs Phi manually using the basis functions to check for normalization/sign errors.
    """
    print(f"\n  [{label}] Phi Reconstruction @ (r={r:.3f}, z={z:.3f})")
    
    # --- FIX: Handle Exterior Region ---
    if region_idx >= len(a):
        print(f"    Region {region_idx} is EXTERIOR. Manual reconstruction skipped (requires Lambda_k).")
        # Return 0.0 so the diff calculation doesn't crash
        return 0.0
    # -----------------------------------

    phi_accum = 0.0
    
    # Check if this is an Inner region (i=0) or Annulus
    is_inner = (region_idx == 0)
    
    if is_inner:
        # Inner Region: R_1n * Z_n
        for n, c in enumerate(coeffs[:6]): # Check first 6 modes
            psi_z = Z_n_i(n, z, region_idx, h, d)
            psi_r = R_1n(n, r, region_idx, h, d, a)
            contrib = c * psi_r * psi_z
            phi_accum += contrib
            print(f"    n={n}: c={c:.3e}, Psi_Z={psi_z:.3f}, Psi_R={psi_r:.3f}, Contrib={contrib:.3e}")
    else:
        # Annulus: Has R1 and R2 parts. 
        # Coeffs are usually [C_0_R1, C_1_R1..., C_0_R2, C_1_R2...]
        # We assume coefficients are split evenly in the list passed to us
        n_modes = len(coeffs) // 2
        
        # R1 Terms
        print("    -- R1 Terms --")
        for n in range(min(n_modes, 3)):
            c = coeffs[n]
            psi_z = Z_n_i(n, z, region_idx, h, d)
            psi_r = R_1n(n, r, region_idx, h, d, a)
            contrib = c * psi_r * psi_z
            phi_accum += contrib
            print(f"    n={n}: c={c:.3e}, Psi_Z={psi_z:.3f}, Psi_R={psi_r:.3f}, Contrib={contrib:.3e}")

        # R2 Terms
        print("    -- R2 Terms --")
        for n in range(min(n_modes, 3)):
            idx = n_modes + n
            c = coeffs[idx]
            psi_z = Z_n_i(n, z, region_idx, h, d)
            psi_r = R_2n(n, r, region_idx, a, h, d)
            contrib = c * psi_r * psi_z
            phi_accum += contrib
            print(f"    n={n}: c={c:.3e}, Psi_Z={psi_z:.3f}, Psi_R={psi_r:.3f}, Contrib={contrib:.3e}")

    print(f"  = Total Phi Reconstructed: {phi_accum:.4e}")
    return phi_accum

def diagnose_potential_continuity(engine, problem, solution_vector, config_name, m0):
    print(f"\n  [CONTINUITY DIAGNOSTIC - POTENTIAL] {config_name}")
    geo = problem.geometry
    domains = geo.domain_list
    a = geo.body_arrangement.a
    d = geo.body_arrangement.d
    h = geo.h
    boundary_count = len(domains) - 1
    
    NMK_list = [domains[i].number_harmonics for i in range(len(domains))]
    all_coeffs = engine.reformat_coeffs(solution_vector, NMK_list, len(domains)-1)

    for bd in range(boundary_count):
        radius = a[bd]
        d_left = d[bd]; d_right = d[bd+1] if bd < len(d)-1 else d[bd]
        max_draft = max(d_left, d_right)
        
        eps = 1e-4
        z_common = np.linspace(-0.01, -max_draft + 0.01, 50)
        
        pot_left = engine.calculate_potentials(problem, solution_vector, m0, 10, False, R_range=np.array([radius - eps]), Z_range=z_common)
        pot_right = engine.calculate_potentials(problem, solution_vector, m0, 10, False, R_range=np.array([radius + eps]), Z_range=z_common)
        
        # --- FIX: Check 'phi' (Total), NOT 'phiH' (Homogeneous) ---
        # phiH is allowed to jump if bodies are heaving. Total phi must be continuous.
        phi_L_arr = pot_left['phi'].flatten()
        phi_R_arr = pot_right['phi'].flatten()
        
        valid = ~np.isnan(phi_L_arr) & ~np.isnan(phi_R_arr)
        if not np.any(valid): continue
            
        diff = np.abs(phi_L_arr[valid] - phi_R_arr[valid])
        max_diff = np.max(diff)
        rel_diff = max_diff / np.mean(np.abs(phi_L_arr[valid])) if np.mean(np.abs(phi_L_arr[valid])) > 1e-9 else max_diff
        
        status = "✅ PASS" if rel_diff < 0.05 else "❌ FAIL"
        print(f"    Boundary {bd} (R={radius:.2f}): {status}")
        print(f"      Max Abs Phi Diff: {max_diff:.4e}")
        print(f"      Rel Phi Err     : {rel_diff*100:.2f}%")
        
        if status == "❌ FAIL":
            print("      >>> TRIGGERING RECONSTRUCTION AUDIT <<<")
            z_audit = -max_draft / 2.0 
            
            c_left = all_coeffs[bd]
            phi_recon_L = debug_phi_reconstruction(c_left, radius, z_audit, bd, h, d, a, "LEFT")
            
            c_right = all_coeffs[bd+1]
            phi_recon_R = debug_phi_reconstruction(c_right, radius, z_audit, bd+1, h, d, a, "RIGHT")
            
            print(f"      Δφ (Recon, H-only) = {phi_recon_L - phi_recon_R:.4e}")
            print("      (Note: H-only recon should NOT match if Heaving is present. Use Total Phi check above.)")

def diagnose_series_at_point(engine, problem, solution_vector, r_target, z_target, m0, config_name):
    """
    Deconstructs the potential at a specific (R, Z) point into its mode components.
    Helps identify if specific modes are blowing up.
    """
    print(f"\n  [SERIES SUMMATION DIAGNOSTIC] at R={r_target:.4f}, Z={z_target:.4f}")
    
    # 1. Identify Region
    geo = problem.geometry
    a = geo.body_arrangement.a
    d = geo.body_arrangement.d
    h = geo.h
    
    region_idx = -1
    if r_target <= a[0]:
        region_idx = 0
    elif r_target > a[-1]:
        region_idx = len(a)
    else:
        for i in range(1, len(a)):
            if r_target > a[i-1] and r_target <= a[i]:
                region_idx = i
                break
                
    if region_idx == -1:
        print("    Could not determine region.")
        return
    
    print(f"    [REGION LOCATOR] r={r_target:.4f} -> Region {region_idx} selected")
    
    # Determine theoretical bounds for this region
    r_min = 0.0 if region_idx == 0 else a[region_idx-1]
    r_max = a[region_idx] if region_idx < len(a) else float('inf')
    
    print(f"    Region Bounds: [{r_min:.4f}, {r_max:.4f}]")
    
    # Check for "Silent Killer" boundary proximity
    # If r is 1.200000001 and boundary is 1.2, you might snap to the wrong side 
    # if floating point epsilon isn't handled.
    if abs(r_target - r_max) < 1e-5:
        print(f"      🚨 WARNING: Point is extremely close ({abs(r_target - r_max):.2e}) to UPPER boundary ({r_max}).")
        print(f"      Check if Logic (r <= a) vs (r < a) matches Formulation.")
    if abs(r_target - r_min) < 1e-5:
        print(f"      🚨 WARNING: Point is extremely close ({abs(r_target - r_min):.2e}) to LOWER boundary ({r_min}).")

    print(f"    Point is in Region {region_idx}")
    
    # 2. Get Coefficients for this region
    NMK = [dom.number_harmonics for dom in geo.domain_list.values()]
    Cs = engine.reformat_coeffs(solution_vector, NMK, len(NMK)-1)[region_idx]
    
    # 3. Calculate Terms
    from openflash.multi_equations import R_1n, R_2n, Z_n_i, Lambda_k, Z_k_e, scale, m_k, N_k_multi
    
    # Need access to cached m_k/N_k for exterior
    cache = engine.cache_list[problem]
    engine._ensure_m_k_and_N_k_arrays(problem, m0)
    m_k_arr = cache.m_k_arr
    N_k_arr = cache.N_k_arr
    
    terms = []
    
    if region_idx == len(a): # Exterior
        for k in range(len(Cs)):
            rad_val = Lambda_k(k, r_target, m0, a, m_k_arr)
            vert_val = Z_k_e(k, z_target, m0, h, NMK, m_k_arr) # Uses cache implicitly via m_k func? No, need manual passing in this raw check
            # Actually Z_k_e in multi_equations calls m_k(NMK...), which recomputes. That's fine for debug.
            term_val = Cs[k] * rad_val * vert_val
            terms.append((k, term_val))
    elif region_idx == 0: # Inner
        for n in range(len(Cs)):
            rad_val = R_1n(n, r_target, 0, h, d, a)
            vert_val = Z_n_i(n, z_target, 0, h, d)
            term_val = Cs[n] * rad_val * vert_val
            terms.append((n, term_val))
    else: # Annulus
        num_modes = NMK[region_idx]
        # R1 terms
        for n in range(num_modes):
            rad_val = R_1n(n, r_target, region_idx, h, d, a)
            vert_val = Z_n_i(n, z_target, region_idx, h, d)
            term_val = Cs[n] * rad_val * vert_val
            terms.append((f"R1_{n}", term_val))
            
        # R2 terms
        for n in range(num_modes):
            rad_val = R_2n(n, r_target, region_idx, a, h, d)
            vert_val = Z_n_i(n, z_target, region_idx, h, d)
            term_val = Cs[num_modes + n] * rad_val * vert_val
            terms.append((f"R2_{n}", term_val))
            
    # 4. Print Largest Terms
    terms.sort(key=lambda x: np.abs(x[1]), reverse=True)
    
    print("    Top 5 Contributing Terms:")
    for name, val in terms[:5]:
        print(f"      Mode {name}: {val:.4e} (Abs: {np.abs(val):.4e})")
        
    print("    Term Summation Check:")
    total_sum = sum(t[1] for t in terms)
    print(f"      Total Sum: {total_sum:.4f}")

                
@pytest.mark.parametrize("config_name", ALL_CONFIGS.keys())
def test_potential_field_vs_capytaine(config_name):
    """
    Compares the openflash-calculated potential field against the
    Capytaine-generated benchmark data FOR A GIVEN CONFIG.
    """
    
    # 1. Load data for this config
    phi_capytaine_raw = load_capytaine_data(config_name)
    p = ALL_CONFIGS[config_name]
    
    print(f"\n\n{'='*40}")
    print(f"=== TESTING CONFIG: {config_name} ===")
    print(f"{'='*40}")
    print(f"  Parameters:")
    print(f"    h (depth): {p['h']}")
    print(f"    m0 (wavenum): {p['m0']}")
    print(f"    a (radii): {p['a']}")
    print(f"    d (drafts): {p['d']}")

    # --- IMPLEMENT SUPERPOSITION ---
    original_heaving_map = p["heaving_map"]
    heaving_indices = [i for i, is_heaving in enumerate(original_heaving_map) if is_heaving]
    
    phi_total: Optional[np.ndarray] = None
    omega_final: Optional[float] = None
    results_template: Optional[Dict[str, Any]] = None 
    
    # KEEP TRACK OF ENGINE FOR DEBUGGING
    last_engine = None
    last_problem = None
    last_solution = None

    if not heaving_indices:
        # Case: No heaving bodies (Pass verbose=True)
        res, omega_final = run_openflash_sim(config_name, R_range=p["R_range"], Z_range=p["Z_range"], verbose=True)
        phi_total = res["phi"]
        results_template = res
        last_engine = res["_engine"]
        last_problem = res["_problem"]
        last_solution = res["_solution"]
    else:
        print(f"\n  [SUPERPOSITION START] Combining {len(heaving_indices)} active bodies...")
        for i, idx in enumerate(heaving_indices): # Use enumerate to track index
            
            # Create a compliant heaving map
            single_heaving_map = [False] * len(original_heaving_map)
            single_heaving_map[idx] = True
            
            # --- FIX: Only be verbose on the FIRST body ---
            is_first_run = (i == 0)
            
            res, omega = run_openflash_sim(
                config_name, 
                R_range=p["R_range"], 
                Z_range=p["Z_range"],
                heaving_map_override=single_heaving_map,
                verbose=is_first_run # Only print diagnostics once!
            )
            
            # --- DEBUG: Enhanced Body Stats ---
            of_mag = np.abs(res['phi'])
            cap_slice_mag = np.abs(phi_capytaine_raw[..., idx]) if phi_capytaine_raw.ndim > 2 else np.nan
            
            print(f"    > Body {idx} Active ({single_heaving_map}):")
            print(f"      OpenFlash Max |phi|: {np.nanmax(of_mag):.6e}")
            print(f"      OpenFlash Mean |phi|: {np.nanmean(of_mag):.6e}")
            if not np.isnan(cap_slice_mag).all():
                 print(f"      Capytaine Max |phi|: {np.nanmax(cap_slice_mag):.6e}")
            
            if np.nanmax(of_mag) < 1e-10:
                print(f"      🚨 ALERT: Body {idx} produced ZERO potential! This is likely the bug.")
            
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
                # Just capture the last one for debugging structure
                last_engine = res["_engine"]
                last_problem = res["_problem"]
                last_solution = res["_solution"]
            
            phi_total += res["phi"]

    if results_template is None or phi_total is None or omega_final is None:
        pytest.fail(f"[{config_name}] Simulation failed to produce results.")
    
    # 2. Get the openflash grid and total potential
    R_openflash = results_template["R"]
    Z_openflash = results_template["Z"]
    phi_openflash = phi_total
    omega = omega_final

    # 3. Define the Capytaine grid
    R_cap_grid, Z_cap_grid = np.meshgrid(p["R_range"], p["Z_range"], indexing='ij')

    # 4. Compare (No Interpolation Needed)
    phi_openflash_interp_real = phi_openflash.real
    phi_openflash_interp_imag = phi_openflash.imag
    
    print("\n  [GEOMETRY TRANSITION ANALYSIS]")
    print(f"    Global Water Depth (h): {p['h']}")
        
    current_r = 0.0
    d_vals = p['d']
        
    for i, (r_outer, d_val) in enumerate(zip(p['a'], d_vals)):
        fluid_depth = p['h'] - d_val
            
        # Determine Transition Type from Previous Domain
        if i == 0:
            trans_str = "Center Start"
        else:
            prev_depth = p['h'] - d_vals[i-1]
            if fluid_depth > prev_depth:
                trans_str = "EXPANSION (Step Down / Deeper Fluid)"
            elif fluid_depth < prev_depth:
                trans_str = "CONTRACTION (Step Up / Shallower Fluid)"
            else:
                trans_str = "FLAT"
        # Detect "Pocket" (Deeper than left AND right neighbors)
        is_pocket = False
        if i > 0 and i < len(d_vals) - 1:
            prev_depth = p['h'] - d_vals[i-1]
            next_depth = p['h'] - d_vals[i+1]
            if fluid_depth > prev_depth and fluid_depth > next_depth:
                is_pocket = True
            
        pocket_alert = " <--- 🚨 POCKET REGION (Likely Failure Point)" if is_pocket else ""
            
        print(f"    Domain {i}: R=[{current_r:.2f}, {r_outer:.2f}] | Depth={fluid_depth:.4f} | {trans_str}{pocket_alert}")
        current_r = r_outer

    # Exterior Domain Info
    print(f"    Domain {len(d_vals)}: R=[{current_r:.2f}, inf] | Depth={p['h']:.4f} | EXPANSION (Exit to Open Ocean)")

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

    # --- DEBUG: Conversion Factors ---
    print(f"\n  [CONVERSION DEBUG]")
    print(f"    Omega (w): {omega:.6f}")
    print(f"    Scaling Factor (1/w): {1.0/omega:.6f}")
    print(f"    Capytaine Raw Real [Min, Max]: [{np.nanmin(phi_capytaine_raw.real):.4e}, {np.nanmax(phi_capytaine_raw.real):.4e}]")
    print(f"    Capytaine Raw Imag [Min, Max]: [{np.nanmin(phi_capytaine_raw.imag):.4e}, {np.nanmax(phi_capytaine_raw.imag):.4e}]")

    # Convert Capytaine to Velocity Potential
    # Note: Ensure this conversion matches your theoretical definition 
    # (Capytaine usually outputs diffraction potential, check if incident is included)
    capytaine_real_converted = phi_capytaine_raw.imag * (-1.0 / omega)
    capytaine_imag_converted = phi_capytaine_raw.real * (1.0 / omega)
    
    # Reconstruct complex form for phase comparison
    capytaine_complex_converted = capytaine_real_converted + 1j * capytaine_imag_converted
    
    # --- NEW DEBUGGING BLOCK START ---
    
    # A. Check for Global Phase Rotation (Critical for Config 3)
    # This helps identify if we are off by -1, j, or conjugate
    best_transform = check_phase_rotation(phi_openflash, capytaine_complex_converted)

    # B. Magnitude vs Phase Diagnostics
    # Comparison of Real/Imag is fragile. Magnitude is robust.
    mag_of = np.abs(phi_openflash)
    mag_cap = np.abs(capytaine_complex_converted)
    
    # Calculate Magnitude Error only on valid points
    mag_diff = np.abs(mag_of - mag_cap)
    mag_diff[nan_mask] = np.nan
    max_mag_diff = np.nanmax(mag_diff)
    
    print(f"\n  [MAGNITUDE CHECK] Max | |phi_of| - |phi_cap| |: {max_mag_diff:.6e}")
    if max_mag_diff < 0.05 and np.max(np.abs(phi_openflash.real - capytaine_real_converted)) > 0.1:
        print("  >>> STRONG HINT: Magnitude is correct, but Phase is wrong. Check time convention (e^-iwt vs e^iwt).")

    # C. Pinpoint Location of Maximum Error (Critical for Config 2 & 6)
    # Find indices of max error in Real part
    diff_grid_real = np.abs(phi_openflash.real - capytaine_real_converted)
    diff_grid_real[nan_mask] = -1.0 # Ignore body points
    
    # Get top 3 errors
    flat_indices = np.argsort(diff_grid_real.ravel())[-3:][::-1]
    
    print(f"\n  [LOCATOR] Top 3 Real Part Errors:")
    first_locator = True
    for flat_idx in flat_indices:
        idx_2d = np.unravel_index(flat_idx, diff_grid_real.shape)
        r_err = R_cap_grid[idx_2d]
        z_err = Z_cap_grid[idx_2d]
        val_of = phi_openflash.real[idx_2d]
        val_cap = capytaine_real_converted[idx_2d]
        diff_val = diff_grid_real[idx_2d]
        
        print(f"    @ (R={r_err:.3f}, Z={z_err:.3f}) -> Diff: {diff_val:.4f} (OF: {val_of:.4f} vs CAP: {val_cap:.4f})")
        
        # Check proximity to corners
        for body_idx, (a_val, d_val) in enumerate(zip(p['a'], p['d'])):
            dist = np.sqrt((r_err - a_val)**2 + (z_err - (-d_val))**2)
            if dist < 0.2:
                print(f"      -> NEAR CORNER of Body {body_idx} (a={a_val}, d={d_val}) dist={dist:.4f}")
        
        # --- NEW: Run Point Diagnostics on the worst point ---
        if first_locator and last_engine:
             diagnose_series_at_point(last_engine, last_problem, last_solution, r_err, z_err, p["m0"], config_name)
             first_locator = False

    # Save CSV for detailed inspection if things fail
    save_debug_csvs(R_cap_grid, Z_cap_grid, 
                   phi_openflash.real, capytaine_real_converted, 
                   phi_openflash.imag, capytaine_imag_converted, 
                   nan_mask, config_name)

    # --- NEW DEBUGGING BLOCK END ---

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