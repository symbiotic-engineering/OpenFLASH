import unittest
from unittest.mock import MagicMock  # For mocking dependencies

import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)


from meem_problem import MEEMProblem
from geometry import Geometry
from domain import Domain


class TestMEEMProblem(unittest.TestCase):

    def test_meem_problem_initialization(self):
        """Tests the initialization of the MEEMProblem class."""

        # Mock the Geometry class and its domain_list
        mock_geometry = MagicMock(spec=Geometry)  # Mock the Geometry object
        mock_domain1 = MagicMock(spec=Domain)
        mock_domain2 = MagicMock(spec=Domain)
        mock_domain_list = {0: mock_domain1, 1: mock_domain2} #Mock domain list
        mock_geometry.domain_list = mock_domain_list

        # Create a MEEMProblem instance
        problem = MEEMProblem(mock_geometry)

        # Assertions
        self.assertIsInstance(problem, MEEMProblem)  # Check instance type
        self.assertEqual(problem.domain_list, mock_domain_list)  # Check domain_list
        self.assertEqual(problem.geometry, mock_geometry) #Check geometry object

        # Test case 2: Empty domain list
        mock_geometry2 = MagicMock(spec=Geometry)  # Mock the Geometry object
        mock_domain_list2 = {} #Mock domain list
        mock_geometry2.domain_list = mock_domain_list2

        problem2 = MEEMProblem(mock_geometry2)

        self.assertEqual(problem2.domain_list, mock_domain_list2)  # Check domain_list
        self.assertEqual(problem2.geometry, mock_geometry2) #Check geometry object

if __name__ == '__main__':
    unittest.main()