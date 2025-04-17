import unittest
import numpy as np

import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from domain import Domain
from multi_constants import h, d, a, m0, heaving  # Import necessary constants

class MockGeometry:
    def __init__(self, r_coordinates, z_coordinates):
        self.r_coordinates = r_coordinates
        self.z_coordinates = z_coordinates

class TestDomain(unittest.TestCase):

    def test_domain_initialization(self):
        """Tests the initialization of the Domain class."""

        # Mock Geometry object
        geometry = MockGeometry({'a1': 0.5, 'a2': 1.0, 'a3': 1.5}, {'h': 1.0})

        # Test case 1: Basic initialization (inner domain)
        params1 = {'h': 1.0, 'di': 0.5, 'm0': 2.0, 'heaving': 0} 
        domain1 = Domain(5, 1.0, 1.0, None, None, 'inner', params1, 0, geometry)
        self.assertEqual(domain1.number_harmonics, 5)
        self.assertEqual(domain1.height, 1.0)
        self.assertEqual(domain1.radial_width, 1.0)
        self.assertEqual(domain1.category, 'inner')
        self.assertEqual(domain1.h, 1.0)
        self.assertEqual(domain1.di, 0.5)
        self.assertEqual(domain1.a, 0.5)
        self.assertEqual(domain1.m0, 2.0)
        self.assertEqual(domain1.heaving, 0)
        self.assertFalse(domain1.slant)
        self.assertEqual(domain1.r_coords, 0)
        self.assertEqual(domain1.z_coords, [0, 1.0])

        # Test case 2: Outer domain
        params2 = {'h': 1.0, 'di': 0.25, 'a': 1.0, 'm0': 2.0, 'heaving': 1}
        domain2 = Domain(8, 1.0, 1.0, None, None, 'outer', params2, 1, geometry)
        self.assertEqual(domain2.di, 0.25)
        self.assertEqual(domain2.a, 1.0)
        self.assertEqual(domain2.heaving, 1)
        self.assertEqual(domain2.r_coords, [0.5, 1.0])
        self.assertEqual(domain2.z_coords, [0, 1.0])

        # Test case 3: Exterior domain
        params3 = {'h': 1.0, 'm0': 2.0, 'heaving': 0}
        domain3 = Domain(10, 1.0, 1.0, None, None, 'exterior', params3, 2, geometry)
        self.assertIsNone(domain3.di)
        self.assertIsNone(domain3.a)
        self.assertEqual(domain3.heaving, 0)
        self.assertEqual(domain3.r_coords, np.inf)
        self.assertEqual(domain3.z_coords, [0, 1.0])

        # Test case 4: Multi-domain parameters
        params4 = {'h': 1.5, 'di': 0.75, 'a': 1.25, 'm0': 2.5, 'scale': 0.5, 'heaving': 1, 'slant': True}
        domain4 = Domain(8, 1.5, 0.75, None, None, 'multi', params4, 2, geometry)
        self.assertEqual(domain4.h, 1.5)
        self.assertEqual(domain4.di, 0.75)
        self.assertEqual(domain4.a, 1.25)
        self.assertEqual(domain4.m0, 2.5)
        self.assertEqual(domain4.scale, 0.5)
        self.assertEqual(domain4.heaving, 1)
        self.assertTrue(domain4.slant)
        self.assertEqual(domain4.r_coords, 0.5)
        self.assertEqual(domain4.z_coords, [0, 1.5])

        # Test case 5: a as a list of values
        params5 = {'h': 1.0, 'di': 0.5, 'a': [0.5, 1.0, 1.0], 'm0': 2.0, 'heaving':0} # a is now a list
        domain5 = Domain(5, 1.0, 1.0, None, None, 'inner', params5, 0, geometry)
        self.assertEqual(domain5.a, [0.5, 1.0, 1.0]) # a should now be a list
        self.assertEqual(domain5.scale, np.mean([0.5, 1.0, 1.5])) # scale should be the mean of the geometry r_coordinates values

if __name__ == '__main__':
    unittest.main()