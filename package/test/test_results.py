import unittest
from unittest.mock import MagicMock, patch
import xarray as xr
import numpy as np

# Adjust import paths (as before)
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from results import Results
from geometry import Geometry
from domain import Domain


class TestResults(unittest.TestCase):

    def setUp(self):
        """Set up a basic Results object for testing."""
        self.frequencies = np.array([1.0, 2.0, 3.0])
        self.modes = np.array(['mode1', 'mode2'])

        # Mock Geometry and Domain objects
        mock_geometry = MagicMock(spec=Geometry)
        mock_domain1 = MagicMock(spec=Domain)
        mock_domain1.r_coordinates = {'r1': np.array([0.1, 0.2])}  # Example r_coordinates
        mock_domain1.z_coordinates = {'z1': np.array([0.0, 0.1, 0.2])} # Example z_coordinates
        mock_domain2 = MagicMock(spec=Domain)
        mock_domain2.r_coordinates = {'r2': np.array([0.3, 0.4, 0.5])} # Example r_coordinates
        mock_domain2.z_coordinates = {'z2': np.array([0.0, 0.1])} # Example z_coordinates
        mock_geometry.domain_list = {0: mock_domain1, 1: mock_domain2}

        self.results = Results(mock_geometry, self.frequencies, self.modes)

    def test_results_initialization(self):
        self.assertIsInstance(self.results, Results)
        np.testing.assert_array_equal(self.results.geometry, self.results.geometry)
        np.testing.assert_array_equal(self.results.frequencies, self.frequencies)
        np.testing.assert_array_equal(self.results.modes, self.modes)
        self.assertIsNone(self.results.dataset)

  


if __name__ == '__main__':
    unittest.main()