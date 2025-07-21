import numpy as np
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)
from openflash.geometry import Geometry  # Your actual Geometry class
from openflash.domain import Domain     # Required since Geometry builds Domain objects

# Define radial and vertical coordinates
r_coordinates = {
    'a1': 1.0,   # Inner radius
    'a2': 2.0,   # Middle radius
    'a3': 3.0    # Outer radius
}
z_coordinates = {
    'h': 5.0     # Height of all regions
}

# Define parameters for 3 regions: inner, middle, and exterior
domain_params = [
    {
        'number_harmonics': 10,
        'height': 5.0,
        'radial_width': 1.0,
        'top_BC': None,
        'bottom_BC': None,
        'category': 'inner',
        'di': 2.0,
        'a': 1.0,
        'heaving': True,
        'slant': False
    },
    {
        'number_harmonics': 15,
        'height': 5.0,
        'radial_width': 1.0,
        'top_BC': None,
        'bottom_BC': None,
        'category': 'outer',
        'di': 1.0,
        'a': 2.0,
        'heaving': False,
        'slant': False
    },
    {
        'number_harmonics': 20,
        'height': 5.0,
        'radial_width': None,  # Exterior region has infinite extent
        'top_BC': None,
        'bottom_BC': None,
        'category': 'exterior',
        'a': 3.0,
        'heaving': False,
        'slant': False
    }
]

# Create the Geometry object
geometry = Geometry(r_coordinates, z_coordinates, domain_params)

# Access domains
print("\n--- Domain Summary ---")
for idx, domain in geometry.domain_list.items():
    print(f"Domain {idx}:")
    print(f"  Category: {domain.category}")
    print(f"  Harmonics: {domain.number_harmonics}")
    print(f"  Radial width: {domain.radial_width}")
    print(f"  Heaving: {domain.heaving}")
    print(f"  r_coords: {domain.r_coords}")
    print(f"  z_coords: {domain.z_coords}")
    print()

# Display adjacency matrix
print("--- Adjacency Matrix ---")
print(geometry.adjacency_matrix)
