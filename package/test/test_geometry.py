import unittest
import numpy as np

# Adjust import paths (same as before)
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from geometry import Geometry
from domain import Domain  # Import Domain for assertions
from multi_constants import h, d, a, heaving, m0

class TestGeometry(unittest.TestCase):

    def test_geometry_initialization(self):
        """Tests the initialization of the Geometry class."""

        # Test case 1: Basic initialization
        r_coordinates1 = {'a1': 0.5, 'a2': 1.0}
        z_coordinates1 = {'h': 1.001}
        domain_params1 = [
            {'number_harmonics': 5, 'height': 1.0, 'radial_width': 0.5, 'category': 'inner', 'di': 0.5, 'a': 0.5},
            {'number_harmonics': 8, 'height': 1.0, 'radial_width': 1.0, 'category': 'outer', 'di': 0.25, 'a': 1.0},
        ]
        geometry1 = Geometry(r_coordinates1, z_coordinates1, domain_params1)

        self.assertEqual(len(geometry1.domain_list), 2)  # Check number of domains
        self.assertIsInstance(geometry1.domain_list[0], Domain)  # Check domain type
        self.assertEqual(geometry1.domain_list[0].number_harmonics, 5)
        self.assertEqual(geometry1.domain_list[0].a, 0.5)
        self.assertEqual(geometry1.domain_list[0].di, 0.5)
        self.assertEqual(geometry1.domain_list[0].scale, 0.75) #Scale should be the mean of the list
        self.assertEqual(geometry1.domain_list[1].number_harmonics, 8)
        self.assertEqual(geometry1.domain_list[1].a, 1.0)
        self.assertEqual(geometry1.domain_list[1].di, 0.25)
        self.assertEqual(geometry1.domain_list[1].scale, 0.75)  #Scale should be the mean of the list

        # Test case 2: Using default values
        r_coordinates2 = {}  # Empty
        z_coordinates2 = {'h':h}  # Empty
        domain_params2 = [
            {'number_harmonics': 3, 'height': 2.0, 'radial_width': 0.75, 'category': 'inner', 'di': d[0], 'a': a[0], 'heaving': 1},
            {'number_harmonics': 6, 'height': 1.5, 'radial_width': 0.5, 'category': 'outer', 'di': d[1], 'a': a[1], 'heaving': 1},
        ]
        geometry2 = Geometry(r_coordinates2, z_coordinates2, domain_params2)
        self.assertEqual(len(geometry2.domain_list), 2)
        self.assertEqual(geometry2.domain_list[0].h, h)  # Default h
        self.assertEqual(geometry2.domain_list[0].di, d[0])  # Default d[0]
        self.assertEqual(geometry2.domain_list[0].a, a[0])  # Default a[0]
        self.assertEqual(geometry2.domain_list[1].h, h)  # Default h
        self.assertEqual(geometry2.domain_list[1].di, d[1])  # Default d[1]
        self.assertEqual(geometry2.domain_list[1].a, a[1])  # Default a[1]

        # Test case 3: Empty domain_params
        r_coordinates3 = {'a': 1.0}
        z_coordinates3 = {'h': 1.0}
        domain_params3 = []  # Empty list
        geometry3 = Geometry(r_coordinates3, z_coordinates3, domain_params3)
        self.assertEqual(len(geometry3.domain_list), 0)  # No domains should be created.

        # Test case 4: When 'a' is a list in domain_params
        r_coordinates4 = {'a1': 0.5, 'a2': 1.0}
        z_coordinates4 = {'h': 1.001}
        domain_params4 = [
            {'number_harmonics': 5, 'height': 1.0, 'radial_width': 0.5, 'category': 'inner', 'di': 0.5, 'a': [0.25, 0.5]},
            {'number_harmonics': 8, 'height': 1.0, 'radial_width': 1.0, 'category': 'outer', 'di': 0.25, 'a': [0.75, 1.0]},
        ]
        geometry4 = Geometry(r_coordinates4, z_coordinates4, domain_params4)

        self.assertEqual(len(geometry4.domain_list), 2)  # Check number of domains
        self.assertIsInstance(geometry4.domain_list[0], Domain)  # Check domain type
        self.assertEqual(geometry4.domain_list[0].number_harmonics, 5)
        self.assertEqual(geometry4.domain_list[0].a, [0.25, 0.5])
        self.assertEqual(geometry4.domain_list[0].di, 0.5)
        self.assertEqual(geometry4.domain_list[1].number_harmonics, 8)
        self.assertEqual(geometry4.domain_list[1].a, [0.75, 1.0])
        self.assertEqual(geometry4.domain_list[1].di, 0.25)

if __name__ == '__main__':
    unittest.main()