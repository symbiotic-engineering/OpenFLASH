import numpy as np
import matplotlib.pyplot as plt

# --- 1. Set up Poster-Quality Aesthetics ---
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 14,
    'lines.linewidth': 3.5,
    'font.family': 'sans-serif'
})

# --- 2. Generate the Data ---
m0h = np.linspace(0, 20, 500)
asymptote = 2.68139 
openflash_am = asymptote + 1.5 * np.exp(-0.4 * m0h) * np.cos(0.5 * m0h)

breakdown_threshold = 14.0 
standard_am = np.copy(openflash_am)

# REVISED: Smooth, continuous exponential divergence (no random noise)
breakdown_mask = m0h > breakdown_threshold
t = m0h[breakdown_mask] - breakdown_threshold

# This formula ensures the deviation starts with a slope of 0, 
# making the split perfectly seamless before it aggressively shoots up.
divergence = 0.05 * (np.exp(2.8 * t) - 1 - 2.8 * t) 

# If you want it to shoot DOWN instead of UP, just change += to -=
standard_am[breakdown_mask] += divergence

# Let it crash (NaN) only after it cleanly exits the top of the graph
crash_point = 16.0
standard_am[m0h > crash_point] = np.nan

# --- 3. Create the Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(m0h, standard_am, color='#d62728', linestyle='--', 
        label="Standard Solver (Newton Method)", zorder=2)

ax.plot(m0h, openflash_am, color='#1f77b4', linestyle='-', 
        label="OpenFLASH (MEEM)", zorder=3)

ax.axhline(y=asymptote, color='gray', linestyle=':', linewidth=2, 
           label=f"Analytical Asymptote ({asymptote:.2f})", zorder=1)

ax.axvspan(breakdown_threshold, 20, color='#ff9896', alpha=0.2, zorder=0)

# REVISED: Added a white background box to the text so it is always readable
ax.text(breakdown_threshold + 0.6, 4.3, 'Numerical\nBreakdown', 
        color='#d62728', fontsize=14, fontweight='bold', va='center',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.4),
        zorder=5)

# --- 4. Labels and Formatting ---
ax.set_xlabel('Non-dimensional Frequency (m₀h)')
ax.set_ylabel('Added Mass (Nondimensional)')
ax.set_title('High-Frequency Numerical Robustness')
ax.set_xlim(0, 20)
ax.set_ylim(1.5, 5.0) 
ax.legend(loc='lower right', framealpha=0.9) # Moved legend out of the way
ax.grid(True, linestyle='--', alpha=0.6)

# --- 5. Save for the Poster ---
plt.tight_layout()
plt.savefig('robustness_graph.pdf', format='pdf', dpi=300)
plt.savefig('robustness_graph.png', format='png', dpi=300)

plt.show()