import numpy as np
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from openflash.domain import Domain
from openflash.geometry import Geometry

# ==============================================================================
#  Purpose of the Domain and Geometry Classes
# ==============================================================================
#
# The `Domain` class represents a SINGLE region. Each Domain has its
# own properties, like its depth (`di`), radius (`a`), and the number of
# harmonics (`number_harmonics`).
#
# The `Geometry` class acts as a CONTAINER that manages a collection of these
# `Domain` objects. You don't create individual Domains yourself. Instead, you
# define the overall properties of your entire system and pass them to the
# `Geometry` class, which then automatically creates and configures all the
# necessary `Domain` objects for you.
#
# This example walks you through the standard workflow.
#

def main():
    """
    An example script demonstrating how to define a physical system and
    create the corresponding Geometry and Domain objects.
    """
    print("--- MEEM Domain and Geometry Class Example ---")

    # ==========================================================================
    # STEP 1: Define the High-Level Physical Parameters of the System
    # ==========================================================================
    #
    # We describe the system with simple lists. 
    # NMK has 4 elements, while the lists for cylinder-specific
    # properties (a, d, heaving) have 3 elements. 

    # Number of harmonics (terms) for each region's approximation.
    # Index 0: inner, 1: outer, 2: outer, 3: exterior
    NMK = [50, 50, 50, 50]

    # Radii, Must be increasing.
    a = [3.0, 5.0, 10.0]

    #  (depths) of the cylinders.
    d = [29.0, 7.0, 4.0]

    # Heaving status for each cylinder (1 for heaving, 0 for not).
    heaving = [0, 1, 1]

    # Total water depth.
    h = 100.0
    
    print("\n## Step 1: Defined Basic Physical Properties ##")
    print(f"Number of Harmonics (NMK): {NMK}")
    print(f"Cylinder Radii (a): {a}")
    print(f"Cylinder Drafts (d): {d}")
    print(f"Total Water Depth (h): {h}")


    # ==========================================================================
    # STEP 2: Use Helper Methods to Build Configuration Data
    # ==========================================================================
    #
    # The `Geometry` class requires its inputs in a specific structured format
    # (lists of dictionaries, etc.). The static methods on the `Domain` class
    # are helpers that convert our simple lists from Step 1 into this format.

    # Create the list of parameter dictionaries for each domain.
    domain_params = Domain.build_domain_params(NMK, a, d, heaving, h)

    # Create the dictionary of radial coordinates.
    r_coordinates = Domain.build_r_coordinates_dict(a)

    # Create the dictionary of vertical coordinates.
    z_coordinates = Domain.build_z_coordinates_dict(h)

    print("\n## Step 2: Converted Properties into Structured Format ##")
    print(f"\nCreated `r_coordinates` dictionary:\n{r_coordinates}")
    print(f"\nCreated `z_coordinates` dictionary:\n{z_coordinates}")
    print(f"\nCreated `domain_params` list (showing first element):")
    import json
    print(json.dumps(domain_params[0], indent=2))


    # ==========================================================================
    # STEP 3: Create the Geometry Object
    # ==========================================================================
    #
    # Now, we pass the structured data into the `Geometry` constructor.
    # It will automatically create and manage all the `Domain` objects internally.

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    
    print("\n## Step 3: Created the Main `Geometry` Object ##")
    print("Geometry object successfully instantiated.")


    # ==========================================================================
    # STEP 4: Access and Inspect the Individual Domain Objects
    # ==========================================================================
    #
    # The `Geometry` object holds a list of all the `Domain` instances it created.
    # We can now access this list to inspect the properties of each individual region.

    print("\n## Step 4: Inspecting the `Domain` Objects inside `Geometry` ##")
    
    # The `domain_list` is a dictionary mapping an index to a Domain object.
    for index, domain in geometry.domain_list.items():
        print(f"\n--- Domain Index: {index} ---")
        print(f"  Category: {domain.category}")
        print(f"  Number of Harmonics: {domain.number_harmonics}")
        print(f"  Radius (a): {domain.a}")
        print(f"  Depth (di): {domain.di}")
        print(f"  Is Heaving: {bool(domain.heaving)}")
        print(f"  Total Water Depth (h): {domain.h}")

    print("\n--- Example Finished ---")


if __name__ == "__main__":
    main()