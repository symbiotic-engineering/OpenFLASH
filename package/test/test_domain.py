import unittest
import numpy as np

import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from domain import Domain
from multi_constants import h, d, a, m0, heaving  # Import necessary constants

class TestDomain(unittest.TestCase):

    def test_domain_initialization(self):
        """Tests the initialization of the Domain class."""

        # Test case 1: Basic initialization (a is a single value)
        params1 = {'h': 1.0, 'di': 0.5, 'a': 1.0, 'm0': 2.0}
        domain1 = Domain(5, 1.0, 1.0, None, None, 'inner', params1, 0)
        self.assertEqual(domain1.number_harmonics, 5)
        self.assertEqual(domain1.height, 1.0)
        self.assertEqual(domain1.radial_width, 1.0)
        self.assertEqual(domain1.category, 'inner')
        self.assertEqual(domain1.h, 1.0)
        self.assertEqual(domain1.di, 0.5)
        self.assertEqual(domain1.a, 1.0)
        self.assertEqual(domain1.m0, 2.0)
        self.assertFalse(domain1.slant)


        # Test case 2: Using default values for parameters (a is from multi_constants)
        domain2 = Domain(10, 2.0, 0.5, None, None, 'outer', {}, 1)  # Empty params
        self.assertEqual(domain2.number_harmonics, 10)
        self.assertEqual(domain2.height, 2.0)
        self.assertEqual(domain2.radial_width, 0.5)
        self.assertEqual(domain2.category, 'outer')
        self.assertEqual(domain2.h, h)
        self.assertEqual(domain2.di, d[1] if 1 < len(d) else 0.0)
        self.assertEqual(domain2.a, a[1] if 1 < len(a) else a[-1]) # Use a[1]
        self.assertEqual(domain2.m0, m0)
        self.assertEqual(domain2.scale, np.mean(a))  # mean of all a's (is this supposed to be correct?)
        self.assertEqual(domain2.heaving, heaving[1] if 1 < len(heaving) else 0)
        self.assertFalse(domain2.slant)

        # Test case 3: Multi-domain parameters
        params3 = {'h': 1.5, 'di': 0.75, 'a': 1.25, 'm0': 2.5, 'scale': 0.5, 'heaving': 1, 'slant': True}
        domain3 = Domain(8, 1.5, 0.75, None, None, 'multi', params3, 2)
        self.assertEqual(domain3.h, 1.5)
        self.assertEqual(domain3.di, 0.75)
        self.assertEqual(domain3.a, 1.25)
        self.assertEqual(domain3.m0, 2.5)
        self.assertEqual(domain3.scale, 0.5)
        self.assertEqual(domain3.heaving, 1)
        self.assertTrue(domain3.slant)

        # Test case 4: Index out of range for d, a, heaving
        params4 = {}
        domain4 = Domain(5, 1.0, 1.0, None, None, 'inner', params4, 10)
        self.assertEqual(domain4.di, 0.0)
        self.assertEqual(domain4.a, a[-1]) # use a[-1]
        self.assertEqual(domain4.heaving, 0)
        # if the index is out of range then should we return that ?


        # Test case 5: a as a list of values
        params5 = {'h': 1.0, 'di': 0.5, 'a': [0.5, 1.0], 'm0': 2.0} # a is now a list
        domain5 = Domain(5, 1.0, 1.0, None, None, 'inner', params5, 0)
        self.assertEqual(domain5.a, [0.5, 1.0]) # a should now be a list
        self.assertEqual(domain5.scale, np.mean([0.5, 1.0])) # scale should be the mean of the list

if __name__ == '__main__':
    unittest.main()