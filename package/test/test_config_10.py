import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

# --- FIX: ROBUST PATH SETUP ---
# Get the folder where this script (test_config_10.py) lives
script_dir = Path(__file__).parent

# Construct the path to the CSV file relative to this script
# Going up one level (..) to 'package', then into 'test_artifacts'
csv_path = script_dir.parent / "test_artifacts" / "config10_debug_data.csv"

print(f"Loading data from: {csv_path}")

# Load your debug data
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"\n❌ CRITICAL ERROR: Could not find the file at {csv_path}")
    print("Please ensure you have run 'pytest test_capytaine_potential.py' first to generate this file.")
    exit(1)

# Filter out the body points (where Capytaine is NaN or 0)
df = df[~df['is_body_nan']]

# Setup Grid
xi = np.linspace(df['R'].min(), df['R'].max(), 100)
yi = np.linspace(df['Z'].min(), df['Z'].max(), 100)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate Data for Plotting
of_grid = griddata((df['R'], df['Z']), df['openflash_real'], (Xi, Yi), method='linear')
cap_grid = griddata((df['R'], df['Z']), df['capytaine_real_converted'], (Xi, Yi), method='linear')
diff_grid = of_grid - cap_grid

# --- PLOTTING ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. OpenFlash (MEEM)
im1 = axes[0].pcolormesh(Xi, Yi, of_grid, cmap='viridis', shading='auto')
axes[0].set_title("OpenFlash Potential (MEEM)")
fig.colorbar(im1, ax=axes[0])

# 2. Capytaine (BEM)
im2 = axes[1].pcolormesh(Xi, Yi, cap_grid, cmap='viridis', shading='auto')
axes[1].set_title("Capytaine Potential (BEM)")
fig.colorbar(im2, ax=axes[1])

# 3. The Difference (The Error)
limit = np.nanmax(np.abs(diff_grid))
im3 = axes[2].pcolormesh(Xi, Yi, diff_grid, cmap='seismic', vmin=-limit, vmax=limit, shading='auto')
axes[2].set_title(f"Difference (OpenFlash - Capytaine)\nMax Diff: {limit:.4f}")
fig.colorbar(im3, ax=axes[2])

# Overlay Body Lines (approximate based on your config)
for ax in axes:
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    # Draw simple lines for the steps in Config 10
    radii = [0.3, 0.5, 1.0, 1.2, 1.6]
    depths = [0.15, 0.4, 0.75, 0.85, 1.1]
    
    # Draw vertical lines for steps
    for r, d in zip(radii, depths):
        ax.axvline(x=r, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        # ax.plot([r, r], [0, -d], 'k--', linewidth=1)
        # ax.plot([0, r], [-d, -d], 'k--', linewidth=1)

plt.tight_layout()
plt.show()

# --- 1D SLICE DIAGNOSTIC ---
# Look for "wiggles" (Matrix Instability) vs "Offsets" (Convergence)
z_slice = -0.857 # Depth from your logs
subset = df[np.isclose(df['Z'], z_slice, atol=0.05)].sort_values(by='R')

plt.figure(figsize=(10, 6))
plt.plot(subset['R'], subset['openflash_real'], 'b-o', label='OpenFlash', markersize=4)
plt.plot(subset['R'], subset['capytaine_real_converted'], 'r--', label='Capytaine', linewidth=2)
plt.title(f"Radial Slice at Z ≈ {z_slice}")
plt.xlabel("R")
plt.ylabel("Potential")
plt.legend()
plt.grid(True)
plt.show()