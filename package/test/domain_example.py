import numpy as np
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from openflash.geometry import Geometry
from openflash.domain import Domain

# Step 1: Define radial and vertical coordinates
r_coordinates = {'a1': 1.0, 'a2': 2.0}
z_coordinates = {'h': 5.0}

# Step 2: Define domain parameters for one region
domain_params = [
    {
        'number_harmonics': 10,
        'height': 5.0,
        'radial_width': 1.0,
        'top_BC': None,
        'bottom_BC': None,
        'category': 'inner',  # can be 'inner', 'outer', or 'exterior'
        'di': 2.0,
        'a': 1.0,
        'heaving': True,
        'slant': False
    }
]

# Step 3: Create a Geometry instance
geometry = Geometry(r_coordinates, z_coordinates, domain_params)

# Step 4: Retrieve a Domain object from the geometry
domain = geometry.domain_list[0]

# Step 5: Display domain attributes
print("Domain index:", domain.index)
print("Number of harmonics:", domain.number_harmonics)
print("Height:", domain.height)
print("Radial width:", domain.radial_width)
print("r_coords:", domain.r_coords)
print("z_coords:", domain.z_coords)
print("Heaving?", domain.heaving)
print("Slanted domain?", domain.slant)
