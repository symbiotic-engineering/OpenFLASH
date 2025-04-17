import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from geometry import Geometry
from domain import Domain 
import numpy as np

# Sample data (replace with your actual data)
r_coordinates = {
    'a1': 2.0,  # Radius of the inner domain
    'a2': 4.0   # Radius of the outer domain
}

z_coordinates = {
    'h': 5.0    # Height of the geometry
}

domain_params = [
    {
        'number_harmonics': 5,
        'height': z_coordinates['h'],
        'radial_width': r_coordinates['a1'],
        'top_BC': 'Free Surface',
        'bottom_BC': 'Sea Floor',
        'category': 'inner',
        'di': 1.0,
        'a': r_coordinates['a1'],
        'heaving': 1.0
    },
    {
        'number_harmonics': 7,
        'height': z_coordinates['h'],
        'radial_width': r_coordinates['a2'] - r_coordinates['a1'],
        'top_BC': 'Free Surface',
        'bottom_BC': 'Sea Floor',
        'category': 'outer',
        'di': 2.0,
        'a': r_coordinates['a2'],
        'heaving': 1.0
    },
    {
        'number_harmonics': 9,
        'height': z_coordinates['h'],
        'radial_width': np.inf,
        'top_BC': 'Free Surface',
        'bottom_BC': 'Sea Floor',
        'category': 'exterior',
        'di': None,
        'a': None,
        'heaving': 1.0
    }
]

# Create the Geometry object
geometry = Geometry(r_coordinates, z_coordinates, domain_params)

# Access the domain list
domain_list = geometry.domain_list

# Print information about the first domain
print("Domain 0:")
print(f"  Number of Harmonics: {domain_list[0].number_harmonics}")
print(f"  Height: {domain_list[0].height}")
print(f"  Radial Width: {domain_list[0].radial_width}")
print(f"  Category: {domain_list[0].category}")
print(f"  di: {domain_list[0].di}")
print(f"  a: {domain_list[0].a}")