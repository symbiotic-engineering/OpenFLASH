import numpy as np
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from openflash.geometry import Geometry

def main():
    # -------------------------
    # Step 1. User provides raw inputs
    # -------------------------
    NMK = [5, 7, 9]        # number of harmonics in each domain (inner, outer, exterior)
    a = [2.0, 4.0]         # cylinder radii (must be strictly increasing)
    d = [3.0, 2.0]         
    heaving = [1, 0]       # 1 = heaving, 0 = fixed
    h = 10.0               # total water depth
    slant = [0, 0]         # optional slant flags

    # -------------------------
    # Step 2. Build domain parameters (blueprints)
    # -------------------------
    domain_params = Geometry.build_domain_params(NMK, a, d, heaving, h, slant)

    # -------------------------
    # Step 3. Build coordinate dictionaries
    # -------------------------
    r_coords = Geometry.build_r_coordinates(a)
    z_coords = Geometry.build_z_coordinates(h)

    # -------------------------
    # Step 4. Construct Geometry object (which creates Domain objects)
    # -------------------------
    geometry = Geometry(r_coordinates=r_coords, z_coordinates=z_coords, domain_params=domain_params)

    # -------------------------
    # Step 5. Access the ready-to-use Domain objects
    # -------------------------
    for idx, domain in geometry.domain_list.items():
        print(f"Domain {idx}: category={domain.category}, a={domain.a}, di={domain.di}, top_BC={domain.top_BC}")

    # -------------------------
    # Step 6. Check adjacency matrix
    # -------------------------
    print("Adjacency matrix:")
    print(geometry.adjacency_matrix)



if __name__ == "__main__":
    main()