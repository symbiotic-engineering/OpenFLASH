import unittest
from unittest.mock import MagicMock, patch
import xarray as xr
import numpy as np
import tempfile  # Import tempfile

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
        mock_domain1.r_coordinates = {'r1': 0.1, 'r2': 0.2}
        mock_domain1.z_coordinates = {'z1': 0.0, 'z2': 0.1, 'z3': 0.2}
        mock_domain2 = MagicMock(spec=Domain)
        mock_domain2.r_coordinates = {'r2': 0.3, 'r3': 0.4, 'r4': 0.5}
        mock_domain2.z_coordinates = {'z2': 0.0, 'z3': 0.1}
        mock_geometry.domain_list = {0: mock_domain1, 1: mock_domain2}

        mock_geometry.r_coordinates = {'r1': 0.1, 'r2': 0.2}
        mock_geometry.z_coordinates = {'z1': 0.0, 'z2': 0.1, 'z3': 0.2}

        self.results = Results(mock_geometry, self.frequencies, self.modes)

    def test_results_initialization(self):
        self.assertIsInstance(self.results, Results)
        self.assertTrue(self.results.geometry is self.results.geometry)

        np.testing.assert_array_equal(self.results.frequencies, self.frequencies)
        np.testing.assert_array_equal(self.results.modes, self.modes)


    def test_store_results(self):
        """Test storing eigenfunction results."""
        radial_data = np.random.rand(len(self.frequencies), len(self.modes), 2)
        vertical_data = np.random.rand(len(self.frequencies), len(self.modes), 3)

        self.results.store_results(0, radial_data, vertical_data)

        self.assertIn('radial_eigenfunctions', self.results.dataset)
        self.assertIn('vertical_eigenfunctions', self.results.dataset)
        radial_da = self.results.dataset['radial_eigenfunctions']
        vertical_da = self.results.dataset['vertical_eigenfunctions']

        self.assertIsInstance(radial_da, xr.DataArray)
        self.assertIsInstance(vertical_da, xr.DataArray)

        self.assertEqual(radial_da.dims, ('frequencies', 'modes', 'r'))
        self.assertEqual(vertical_da.dims, ('frequencies', 'modes', 'z'))

        np.testing.assert_array_equal(radial_da.coords['frequencies'], self.frequencies)
        np.testing.assert_array_equal(radial_da.coords['modes'], self.modes)

    def test_store_potentials(self):
        """Test storing potential results."""
        potentials = {
            'domain1': {'potentials': np.array([1,2,3]),
                        'r': {'r1': 0.1, 'r2': 0.2, 'r3': 0.3},
                        'z': {'z1': 0.0, 'z2': 0.1}},
            'domain2': {'potentials': np.array([4,5]),
                        'r': {'r1': 0.4, 'r2': 0.5},
                        'z': {'z1': 0.2, 'z2': 0.3, 'z3': 0.4}}
        }

        self.results.store_potentials(potentials)

        self.assertIn('domain_potentials', self.results.dataset)
        self.assertIn('domain_r', self.results.dataset.coords)
        self.assertIn('domain_z', self.results.dataset.coords)

        domain_potentials_da = self.results.dataset['domain_potentials']
        domain_r_coords = self.results.dataset.coords['domain_r']
        domain_z_coords = self.results.dataset.coords['domain_z']

        self.assertIsInstance(domain_potentials_da, xr.DataArray)
        self.assertIsInstance(domain_r_coords, xr.DataArray)
        self.assertIsInstance(domain_z_coords, xr.DataArray)

        self.assertEqual(domain_potentials_da.dims, ('domain', 'harmonics'))
        self.assertEqual(domain_r_coords.dims, ('domain', 'harmonics'))
        self.assertEqual(domain_z_coords.dims, ('domain', 'harmonics'))

        # Add more specific assertions to check the contents of the stored data
        # For example, check if the potential values and coordinates are stored correctly.
        np.testing.assert_array_equal(domain_potentials_da.values[0, :3], potentials['domain1']['potentials'])
        np.testing.assert_array_equal(domain_potentials_da.values[1, :2], potentials['domain2']['potentials'])


    def test_export_to_netcdf(self):
        """Test exporting results to a NetCDF file."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=True) as temp_file:
            file_path = temp_file.name
            self.results.export_to_netcdf(file_path)
            self.assertTrue(os.path.exists(file_path))

            # Optionally, you can try to read the file back in and check its contents
            try:
                xr.open_dataset(file_path)
            except Exception as e:
                self.fail(f"Failed to open the NetCDF file: {e}")

    def test_get_results(self):
        """Test getting the stored results."""
        dataset = self.results.get_results()
        self.assertTrue(dataset is self.results.dataset)

    def test_display_results(self):
        """Test displaying the stored results."""
        # Store some results first
        radial_data = np.random.rand(len(self.frequencies), len(self.modes), 2)
        vertical_data = np.random.rand(len(self.frequencies), len(self.modes), 3)
        self.results.store_results(0, radial_data, vertical_data)

        display_str = self.results.display_results()
        self.assertIsInstance(display_str, str)
        self.assertTrue(len(display_str) > 0)  # Check if the string is not empty

if __name__ == '__main__':
    unittest.main()