import unittest
import numpy as np
from unittest.mock import patch, MagicMock  # For mocking dependencies

import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from domain import Domain
from geometry import Geometry
from coupling import A_nm, A_mk
import equations
import multi_equations
from results import Results

class TestMEEMEngine(unittest.TestCase):

    def setUp(self):
        """Set up a basic MEEMProblem for testing."""
        r_coordinates = {'a1': 0.5, 'a2': 1.0}
        z_coordinates = {'h': 1.001}
        domain_params = [
            {'number_harmonics': 3, 'height': 1.0, 'radial_width': 0.5, 'category': 'inner', 'di': 0.5, 'a': 0.5},
            {'number_harmonics': 4, 'height': 1.0, 'radial_width': 1.0, 'category': 'outer', 'di': 0.25, 'a': 1.0},
            {'number_harmonics': 5, 'height': 1.0, 'radial_width': 1.5, 'category': 'exterior'}
        ]
        geometry = Geometry(r_coordinates, z_coordinates, domain_params)
        self.problem = MEEMProblem(geometry)
        self.engine = MEEMEngine([self.problem])  # Create an engine instance


    def test_assemble_A(self):
        A = self.engine.assemble_A(self.problem)
        self.assertTrue(np.all(np.isfinite(A)))  # No NaNs or Infs


    def test_assemble_A_multi(self):
        A = self.engine.assemble_A_multi(self.problem)
        self.assertTrue(np.all(np.isfinite(A)))  # No NaNs or Infs


    def test_assemble_b(self):
        b = self.engine.assemble_b(self.problem)
        self.assertTrue(np.all(np.isfinite(b)))  # No NaNs or Infs

    def test_assemble_b_multi(self):
        b = self.engine.assemble_b_multi(self.problem)
        self.assertTrue(np.all(np.isfinite(b)))  # No NaNs or Infs


    @patch('scipy.linalg.solve')  # Mock the linear solver
    def test_solve_linear_system(self, mock_solve):
        mock_solve.return_value = np.ones(12, dtype=complex)  # Mock a solution
        X = self.engine.solve_linear_system(self.problem)
        self.assertEqual(X.shape, (12,))
        mock_solve.assert_called_once()  # Check that solve was called

    @patch('scipy.linalg.solve')
    def test_solve_linear_system_multi(self, mock_solve):
        mock_solve.return_value = np.ones(12, dtype=complex)
        X = self.engine.solve_linear_system_multi(self.problem)
        self.assertEqual(X.shape, (12,))
        mock_solve.assert_called_once()

    def test_compute_hydrodynamic_coefficients(self):
        # Mock a solution vector for testing
        mock_solution = np.ones(12, dtype=complex)
        hydro_coeffs = self.engine.compute_hydrodynamic_coefficients(self.problem, mock_solution)
        # Basic check to ensure the output is not None and has some length
        self.assertIsNotNone(hydro_coeffs)

    def test_calculate_potentials(self):
        # Mock a solution vector for testing
        mock_solution = np.ones(12, dtype=complex)
        potentials = self.engine.calculate_potentials(self.problem, mock_solution)
        # Check if the potentials dictionary is not empty and has the correct keys
        self.assertTrue(potentials)
        self.assertIn('domain_0', potentials)
        self.assertIn('domain_1', potentials)
        self.assertIn('domain_2', potentials)
        # Check if each domain has 'potentials', 'r', and 'z' keys
        for domain_data in potentials.values():
            self.assertIn('potentials', domain_data)
            self.assertIn('r', domain_data)
            self.assertIn('z', domain_data)


    @patch('matplotlib.pyplot.show')  # Mock show to avoid displaying plots
    def test_visualize_potential(self, mock_show):
        potentials = {
            'inner': np.arange(3),
            'outer': np.arange(4),
            'exterior': np.arange(5)
        }
        self.engine.visualize_potential(potentials)
        mock_show.assert_called_once()

    def test_run_and_store_results(self):
        # Mock frequencies and modes for the problem
        self.problem.frequencies = np.array([1.0, 2.0])
        self.problem.modes = np.array(['heave', 'surge'])
        # Run the computation and store results
        results = self.engine.run_and_store_results(0)
        # Check if the results object is created and has the expected attributes
        self.assertIsInstance(results, Results)
        self.assertIsNotNone(results.dataset)
        self.assertIn('hydrodynamic_coefficients', results.dataset)
        self.assertIn('domain_potentials', results.dataset)



if __name__ == '__main__':
    unittest.main()