import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

import xarray as xr
import numpy as np
from geometry import Geometry
from results import Results

# Example Geometry Setup
r_coordinates = {
    'a1': 2.0,
    'a2': 4.0,
    'a3': 6.0,
}
z_coordinates = {
    'h': 5.0
}
domain_params = [
    {
        'number_harmonics': 3,
        'height': 5.0,
        'radial_width': 2.0,
        'top_BC': 'Free Surface',
        'bottom_BC': 'Sea Floor',
        'category': 'inner',
        'di': 1.0,
        'a': 2.0,
        'heaving': 1.0,
    },
    {
        'number_harmonics': 4,
        'height': 5.0,
        'radial_width': 2.0,
        'top_BC': 'Free Surface',
        'bottom_BC': 'Sea Floor',
        'category': 'outer',
        'di': 2.0,
        'a': 4.0,
        'heaving': 1.0,
    },
    {
        'number_harmonics': 5,
        'height': 5.0,
        'radial_width': 2.0,
        'top_BC': 'Free Surface',
        'bottom_BC': 'Sea Floor',
        'category': 'outer',
        'di': 3.0,
        'a': 6.0,
        'heaving': 1.0,
    },
    {
        'number_harmonics': 6,
        'height': 5.0,
        'radial_width': np.inf,
        'top_BC': 'Free Surface',
        'bottom_BC': 'Sea Floor',
        'category': 'exterior',
        'di': 5.0,
        'a': None,
        'heaving': 1.0,
    },
]

geometry = Geometry(r_coordinates, z_coordinates, domain_params)

# Example Frequencies and Modes
frequencies = np.array([1.0, 2.0])
modes = np.array([1, 2])

# Initialize Results object
results = Results(geometry, frequencies, modes)

# Example Eigenfunction Data
# Ensure radial_data and vertical_data shapes match (frequencies, modes, spatial_coords)
# spatial_coords for r is len(r_coordinates), for z is len(z_coordinates)
radial_data = np.random.rand(len(frequencies), len(modes), len(r_coordinates))
vertical_data = np.random.rand(len(frequencies), len(modes), len(z_coordinates))

# Store Eigenfunction Results for domain 0
results.store_results(0, radial_data, vertical_data)

# Example Potential Data (corrected format for store_all_potentials)
example_potentials_for_one_freq_mode = {
    'domain_0': {
        'potentials': (np.random.rand(3) + 1j * np.random.rand(3)).astype(complex), # Ensure complex
        'r_coords_dict': {'r_h1': 0.1, 'r_h2': 0.2, 'r_h3': 0.3}, # <--- CHANGED KEY
        'z_coords_dict': {'z_h1': 0.1, 'z_h2': 0.2, 'z_h3': 0.3}  # <--- CHANGED KEY
    },
    'domain_1': {
        'potentials': (np.random.rand(4) + 1j * np.random.rand(4)).astype(complex), # Ensure complex
        'r_coords_dict': {'r_h1': 0.4, 'r_h2': 0.5, 'r_h3': 0.6, 'r_h4': 0.7}, # <--- CHANGED KEY
        'z_coords_dict': {'z_h1': 0.4, 'z_h2': 0.5, 'z_h3': 0.6, 'z_h4': 0.7}  # <--- CHANGED KEY
    },
    'domain_2': {
        'potentials': (np.random.rand(5) + 1j * np.random.rand(5)).astype(complex), # Ensure complex
        'r_coords_dict': {'r_h1': 0.8, 'r_h2': 0.9, 'r_h3': 1.0, 'r_h4': 1.1, 'r_h5': 1.2}, # <--- CHANGED KEY
        'z_coords_dict': {'z_h1': 0.8, 'z_h2': 0.9, 'z_h3': 1.0, 'z_h4': 1.1, 'z_h5': 1.2}  # <--- CHANGED KEY
    },
    'domain_3': {
        'potentials': (np.random.rand(6) + 1j * np.random.rand(6)).astype(complex), # Ensure complex
        'r_coords_dict': {'r_h1': 1.3, 'r_h2': 1.4, 'r_h3': 1.5, 'r_h4': 1.6, 'r_h5': 1.7, 'r_h6': 1.8}, # <--- CHANGED KEY
        'z_coords_dict': {'z_h1': 1.3, 'z_h2': 1.4, 'z_h3': 1.5, 'z_h4': 1.6, 'z_h5': 1.7, 'z_h6': 1.8}  # <--- CHANGED KEY
    }
}

# The `store_all_potentials` method expects a LIST of dictionaries,
# where each dictionary has 'frequency_idx', 'mode_idx', and 'data'.
# So, we need to wrap the 'example_potentials_for_one_freq_mode' like this:
all_potentials_batch = [
    {
        'frequency_idx': 0, # Corresponds to frequencies[0]
        'mode_idx': 0,      # Corresponds to modes[0]
        'data': example_potentials_for_one_freq_mode
    },
    {
        'frequency_idx': 1, # Corresponds to frequencies[1]
        'mode_idx': 1,      # Corresponds to modes[1]
        'data': example_potentials_for_one_freq_mode # Using same data for simplicity, but it would vary
    }
]

# Store Potential Results
results.store_all_potentials(all_potentials_batch)

# Export Results to NetCDF
output_folder = "output/netcdf"
os.makedirs(output_folder, exist_ok=True)  # Create the folder(s) if they don't exist
results.export_to_netcdf(os.path.join(output_folder, "example_results.nc"))

# Display Results
print(results.display_results())

# Access Results
dataset = results.get_results()
print("\nAccessing the xarray dataset:")
print(dataset)