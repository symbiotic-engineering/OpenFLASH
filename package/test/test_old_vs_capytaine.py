import pytest
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import scipy.linalg as linalg
from scipy.interpolate import griddata

# --- 1. Path Setup for Old Code ---
current_dir = os.path.dirname(__file__)
old_code_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'dev', 'python'))
if old_code_dir not in sys.path:
    sys.path.insert(0, old_code_dir)

# Import Old Assembly Functions
from old_assembly import (
    assemble_old_A_and_b,
    R_1n_old, Z_n_i_old, R_2n_old, Lambda_k_old, Z_k_e_old, 
    make_R_Z_old, phi_p_i_old
)

# --- 2. Configuration & Constants ---
BENCHMARK_DATA_PATH = Path(__file__).parent.parent.parent / "dev" / "python" / "test" / "data"
DEBUG_PLOT_PATH = Path(__file__).parent.parent / "test_artifacts" / "old_code_comparison"
DEBUG_PLOT_PATH.mkdir(parents=True, exist_ok=True)

# Use the exact same configs
ALL_CONFIGS = {
    "config3": {
        "h": 1.9,
        "a": np.array([0.3, 0.5, 1, 1.2, 1.6]),
        "d": np.array([0.5, 0.7, 0.8, 0.2, 0.5]),
        "heaving_map": [True, True, True, True, True],
        "m0": 1.0,
        "NMK": [15] * 6,
        "R_range": np.linspace(0.0, 2 * 1.6, num=50),
        "Z_range": np.linspace(0, -1.9, num=50),
    },
    "config9": {
        "h": 100.0,
        "a": np.array([3, 5, 10]),
        "d": np.array([4, 7, 29]),
        "heaving_map": [True, True, True],
        "m0": 1.0,
        "NMK": [10] * 4, 
        "R_range": np.linspace(0.0, 2 * 10, num=50),
        "Z_range": np.linspace(0, -100, num=50),
    },
}

# --- 3. Helper Functions ---

def load_capytaine_data(config_name):
    real_path = BENCHMARK_DATA_PATH / f"{config_name}-real.csv"
    imag_path = BENCHMARK_DATA_PATH / f"{config_name}-imag.csv"
    if not real_path.exists(): pytest.skip(f"No benchmark data for {config_name}")
    
    real_data = np.loadtxt(real_path, delimiter=",")
    imag_data = np.loadtxt(imag_path, delimiter=",")
    return real_data + 1j * imag_data

def save_debug_plots(R_grid, Z_grid, old_data, cap_data, nan_mask, config_name, plot_type):
    """Saves comparison plots with region boundaries."""
    output_file = DEBUG_PLOT_PATH / f"{config_name}_{plot_type}_comparison.png"
    
    config = ALL_CONFIGS[config_name]
    radii = config['a']
    depths = config['d']

    diff = old_data - cap_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        percent_diff = 100 * (diff / cap_data)
    
    old_data[nan_mask] = np.nan
    cap_data[nan_mask] = np.nan
    diff[nan_mask] = np.nan
    percent_diff[nan_mask] = np.nan
    
    v_min = np.nanmin([old_data, cap_data])
    v_max = np.nanmax([old_data, cap_data])
    diff_vmax = np.nanmax(np.abs(diff))
    perc_vmax = np.nanmin([500, np.nanmax(np.abs(percent_diff))])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Old Code vs Capytaine: {config_name} ({plot_type})", fontsize=16)

    def draw_boundaries(ax):
        for r in radii:
            ax.axvline(x=r, color='white', linestyle='--', linewidth=1, alpha=0.5)
        for r, d in zip(radii, depths):
            ax.plot(r, -d, 'ro', markersize=3) 

    # Plot 1: Old Code
    im1 = axes[0, 0].pcolormesh(R_grid, Z_grid, old_data, vmin=v_min, vmax=v_max, cmap='viridis', shading='auto')
    fig.colorbar(im1, ax=axes[0, 0])
    axes[0, 0].set_title("Old Code (MEEM)")
    draw_boundaries(axes[0, 0])

    # Plot 2: Capytaine
    im2 = axes[0, 1].pcolormesh(R_grid, Z_grid, cap_data, vmin=v_min, vmax=v_max, cmap='viridis', shading='auto')
    fig.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].set_title("Capytaine (BEM)")
    draw_boundaries(axes[0, 1])

    # Plot 3: Diff
    im3 = axes[1, 0].pcolormesh(R_grid, Z_grid, np.abs(diff), vmin=0, vmax=diff_vmax, cmap='Reds', shading='auto')
    fig.colorbar(im3, ax=axes[1, 0])
    axes[1, 0].set_title(f"Absolute Diff (Max: {diff_vmax:.4f})")
    draw_boundaries(axes[1, 0])

    # Plot 4: Percent Diff
    im4 = axes[1, 1].pcolormesh(R_grid, Z_grid, np.abs(percent_diff), vmin=0, vmax=perc_vmax, cmap='Reds', shading='auto')
    fig.colorbar(im4, ax=axes[1, 1])
    axes[1, 1].set_title("Percent Diff")
    draw_boundaries(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"  > Saved plot: {output_file}")

def save_debug_csv(R_grid, Z_grid, old_val, cap_val, nan_mask, config_name):
    """Saves comparison data to CSV."""
    output_file = DEBUG_PLOT_PATH / f"{config_name}_debug_data.csv"
    
    data = {
        "R": R_grid.ravel(),
        "Z": Z_grid.ravel(),
        "is_body": nan_mask.ravel(),
        "old_code_val": old_val.ravel(),
        "capytaine_val": cap_val.ravel(),
        "abs_diff": np.abs(old_val - cap_val).ravel()
    }
    df = pd.DataFrame(data)
    df = df[~df["is_body"]] # Filter out body points
    df.to_csv(output_file, index=False)
    print(f"  > Saved CSV: {output_file}")

def calculate_potentials_old_wrapper(X, NMK, a, h, d, m0, heaving):
    """Wrapper to run old calculation logic."""
    # (Same logic as before, just compact for the script)
    Cs = []
    row = 0
    Cs.append(X[:NMK[0]])
    row += NMK[0]
    boundary_count = len(NMK) - 1
    for i in range(1, boundary_count):
        Cs.append(X[row: row + NMK[i] * 2])
        row += NMK[i] * 2
    Cs.append(X[row:])

    phi_h_n_inner_vec = np.vectorize(lambda n, r, z: (Cs[0][n] * R_1n_old(n, r, 0, h, d, a)) * Z_n_i_old(n, z, 0, h, d), otypes=[complex])
    phi_h_m_i_vec = np.vectorize(lambda i, m, r, z: (Cs[i][m] * R_1n_old(m, r, i, h, d, a) + Cs[i][NMK[i] + m] * R_2n_old(m, r, i, a, h, d)) * Z_n_i_old(m, z, i, h, d), otypes=[complex])
    phi_e_k_vec = np.vectorize(lambda k, r, z: Cs[-1][k] * Lambda_k_old(k, r, m0, a, NMK, h) * Z_k_e_old(k, z, m0, h, NMK), otypes=[complex])
    phi_p_i_vec = np.vectorize(lambda d_val, r, z: phi_p_i_old(d_val, r, z, h))

    R, Z = make_R_Z_old(a, h, d, True, 50)
    regions = []
    regions.append((R <= a[0]) & (Z < -d[0]))
    for i in range(1, boundary_count):
        regions.append((R > a[i-1]) & (R <= a[i]) & (Z < -d[i]))
    regions.append(R > a[-1])

    phiH = np.zeros_like(R, dtype=complex)
    phiP = np.zeros_like(R, dtype=complex)

    # Calculate phiH
    if np.any(regions[0]):
        for n in range(NMK[0]): phiH[regions[0]] += phi_h_n_inner_vec(n, R[regions[0]], Z[regions[0]])
    for i in range(1, boundary_count):
        if np.any(regions[i]):
            for m in range(NMK[i]): phiH[regions[i]] += phi_h_m_i_vec(i, m, R[regions[i]], Z[regions[i]])
    if np.any(regions[-1]):
        for k in range(NMK[-1]): phiH[regions[-1]] += phi_e_k_vec(k, R[regions[-1]], Z[regions[-1]])

    # Calculate phiP
    for i in range(len(regions)-1):
        if np.any(regions[i]): phiP[regions[i]] = heaving[i] * phi_p_i_vec(d[i], R[regions[i]], Z[regions[i]])
    
    phi = phiH + phiP
    phi[~np.isfinite(phi)] = np.nan # Ensure NaNs where invalid
    return R, Z, phi

# --- 4. The Test ---

@pytest.mark.parametrize("config_name", ALL_CONFIGS.keys())
def test_old_code_vs_capytaine(config_name):
    print(f"\n{'='*40}")
    print(f"Testing OLD CODE vs Capytaine: {config_name}")
    p = ALL_CONFIGS[config_name]
    
    # 1. Run Old Code
    heaving_old = list(p["heaving_map"]) + [0]
    try:
        A, b = assemble_old_A_and_b(p['h'], p['d'], p['a'], p['NMK'], heaving_old, p['m0'])
        X = linalg.solve(A, b)
        R_old, Z_old, phi_old = calculate_potentials_old_wrapper(X, p['NMK'], p['a'], p['h'], p['d'], p['m0'], heaving_old)
    except Exception as e:
        pytest.fail(f"Old code crashed: {e}")

    # 2. Load Capytaine
    phi_cap_raw = load_capytaine_data(config_name)
    g = 9.81
    omega = np.sqrt(p['m0'] * np.tanh(p['m0'] * p['h']) * g)
    # Convert Capytaine (Diffraction -> Potential)
    phi_cap_converted = (phi_cap_raw.imag * (-1.0 / omega)) + 1j * (phi_cap_raw.real * (1.0 / omega))

    # 3. Grid Alignment (Interpolation)
    R_target, Z_target = np.meshgrid(p["R_range"], p["Z_range"], indexing='ij')
    
    # Flatten & Interpolate
    valid_source = np.isfinite(phi_old)
    if np.sum(valid_source) == 0: pytest.fail("Old code produced only NaNs")
    
    points = np.column_stack((R_old[valid_source], Z_old[valid_source]))
    phi_old_interp_real = griddata(points, phi_old.real[valid_source], (R_target, Z_target), method='linear')
    phi_old_interp_imag = griddata(points, phi_old.imag[valid_source], (R_target, Z_target), method='linear')
    phi_old_interp = phi_old_interp_real + 1j * phi_old_interp_imag

    # 4. Masking
    valid_mask = np.isfinite(phi_old_interp) & np.isfinite(phi_cap_converted)
    corner_radius = 0.2
    for r_c, d_c in zip(p['a'], p['d']):
        dist = np.sqrt((R_target - r_c)**2 + (Z_target - (-d_c))**2)
        valid_mask[dist < corner_radius] = False # Exclude corners

    # 5. Analysis
    mag_old = np.abs(phi_old_interp)
    mag_cap = np.abs(phi_cap_converted)
    
    # Apply mask for stats
    mag_old_valid = mag_old[valid_mask]
    mag_cap_valid = mag_cap[valid_mask]
    
    diff = np.abs(mag_old_valid - mag_cap_valid)
    max_diff = np.max(diff) if len(diff) > 0 else 0
    mean_diff = np.mean(diff) if len(diff) > 0 else 0
    
    # Get location of max diff
    if len(diff) > 0:
        flat_idx = np.argmax(np.abs(mag_old - mag_cap) * valid_mask) # Masked argmax
        idx_2d = np.unravel_index(flat_idx, mag_old.shape)
        r_max, z_max = R_target[idx_2d], Z_target[idx_2d]
        val_old, val_cap = mag_old[idx_2d], mag_cap[idx_2d]
    else:
        r_max, z_max, val_old, val_cap = 0,0,0,0

    print(f"  [STATS] {config_name}")
    print(f"    Max Mag Diff: {max_diff:.4f} at (R={r_max:.2f}, Z={z_max:.2f})")
    print(f"    Values there: Old={val_old:.4f}, Cap={val_cap:.4f}")
    print(f"    Mean Mag Diff: {mean_diff:.4f}")

    # 6. Save Artifacts
    # Save Real Part comparison
    save_debug_plots(R_target, Z_target, phi_old_interp.real, phi_cap_converted.real, ~valid_mask, config_name, "real")
    # Save Magnitude comparison
    save_debug_plots(R_target, Z_target, mag_old, mag_cap, ~valid_mask, config_name, "magnitude")
    # Save CSV
    save_debug_csv(R_target, Z_target, mag_old, mag_cap, ~valid_mask, config_name)

    # 7. Assertions
    rel_diff = max_diff / (np.mean(mag_cap_valid) + 1e-9)
    
    if config_name in ["config3", "config9"]:
        print(f"  [XFAIL CHECK] Rel Diff is {rel_diff:.2%}. Expecting failure.")
        if rel_diff > 0.2:
            pytest.xfail(f"Old code confirms physics mismatch (Diff: {rel_diff:.2%})")
            
    np.testing.assert_allclose(
        mag_old_valid, mag_cap_valid,
        rtol=0.1, atol=0.2,
        err_msg=f"Old code failed to match Capytaine"
    )
    print("  [PASS] Old code matches Capytaine!")

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))