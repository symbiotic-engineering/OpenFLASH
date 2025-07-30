import numpy as np
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from openflash.domain import Domain
from openflash.geometry import Geometry

def main():
    """
    An example script demonstrating the creation and use of the Geometry class.
    """
    print("--- MEEM Geometry Class Example ---")

    # The primary role of the Geometry class is to act as a container
    # that correctly assembles and manages a collection of individual Domain
    # objects based on the overall physical properties of the system.

    # ==========================================================================
    # STEP 1: Define the High-Level Physical Parameters
    # ==========================================================================
    # We describe the entire system using simple Python lists and variables.

    NMK = [50, 50, 50]  # Harmonics for inner, outer, and exterior domains
    a = [5.0, 10.0]     # Radii of the two cylinders
    d = [20.0, 10.0]    # Depths of the two cylinders
    heaving = [1, 0]    # First domain is heaving, second is fixed
    h = 100.0           # Total water depth

    print("\n## Step 1: Defined System-Wide Physical Properties ##")
    print(f"Cylinder Radii (a): {a}")
    print(f"Cylinder depths (d): {d}")
    print(f"Total Water Depth (h): {h}")

    # ==========================================================================
    # STEP 2: Use Helper Methods to Build Configuration Data
    # ==========================================================================
    # We use the static methods on the Domain class to convert our simple lists
    # into the structured format required by the Geometry class constructor.

    domain_params = Domain.build_domain_params(NMK, a, d, heaving, h)
    r_coordinates = Domain.build_r_coordinates_dict(a)
    z_coordinates = Domain.build_z_coordinates_dict(h)

    print("\n## Step 2: Converted Properties into Structured Format ##")
    print("Helper methods from the Domain class were used to create the required inputs.")


    # ==========================================================================
    # STEP 3: Create the Geometry Object
    # ==========================================================================
    # We now instantiate the Geometry class. It will automatically create and
    # configure all the necessary Domain objects based on the parameters provided.

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)

    print("\n## Step 3: Instantiated the `Geometry` Object ##")
    print("The Geometry object has now created and stored all individual domains.")

    # ==========================================================================
    # STEP 4: Access and Inspect Geometry Properties
    # ==========================================================================
    # The Geometry object provides useful properties that describe the whole system.

    print("\n## Step 4: Inspecting `Geometry` and its `Domain` Objects ##")

    # You can access the list of Domain objects it created
    num_domains = len(geometry.domain_list)
    print(f"\nNumber of domains created: {num_domains}")

    # You can inspect the properties of a specific domain
    print("\nProperties of the first domain (index 0):")
    first_domain = geometry.domain_list[0]
    print(f"  - Category: {first_domain.category}")
    print(f"  - Radius (a): {first_domain.a}")
    print(f"  - Depth (di): {first_domain.di}")

    # The Geometry object can also provide system-wide information
    print("\nSystem-wide properties from Geometry:")
    print(f"  - Adjacency Matrix:\n{geometry.adjacency_matrix}")

    print("\n--- Example Finished ---")


if __name__ == "__main__":
    main()