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
radial_data = np.random.rand(len(frequencies), len(modes), len(r_coordinates))
vertical_data = np.random.rand(len(frequencies), len(modes), len(z_coordinates))

# Store Eigenfunction Results
results.store_results(0, radial_data, vertical_data)

# Example Potential Data
potentials = {
    'domain_0': {
        'potentials': np.random.rand(3),
        'r': r_coordinates,
        'z': z_coordinates
    },
    'domain_1': {
        'potentials': np.random.rand(4),
        'r': r_coordinates,
        'z': z_coordinates
    },
    'domain_2': {
        'potentials': np.random.rand(5),
        'r': r_coordinates,
        'z': z_coordinates
    },
    'domain_3': {
        'potentials': np.random.rand(6),
        'r': r_coordinates,
        'z': z_coordinates
    }
}

# Store Potential Results
results.store_potentials(potentials)

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