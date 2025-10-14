import numpy as np
import os
import sys

# Add src directory to path
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from openflash.basic_region_geometry import SimpleGeometry

# Example: 3 inner domains + exterior
a_list = [1.0, 2.0, 3.5]       # outer radii
d_list = [1.0, 1.0, 1.0]       # depths
NMK = [2, 2, 2, 2]             # harmonics
h = 5.0                         # total water depth

# --- Automatic adjacency only ---
geom_auto = SimpleGeometry(a=a_list, d=d_list, h=h, NMK=NMK)
print("Automatic adjacency only:")
geom_auto.show_adjacency()

# --- Manual adjacency override ---
manual_adj = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 1],  # skip i2 <-> i3, connect i2 <-> e instead
    [0, 0, 0, 1],
    [0, 1, 1, 0]
], dtype=bool)

geom_manual = SimpleGeometry(a=a_list, d=d_list, h=h, NMK=NMK, manual_adjacency=manual_adj)
print("\nAutomatic + Manual adjacency visualization:")
geom_manual.show_adjacency()
