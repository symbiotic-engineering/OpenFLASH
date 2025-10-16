# test_meem_problem.py

import sys
import os
import numpy as np
import pytest
from unittest.mock import Mock # For mocking dependencies if needed

# --- Path Setup ---
# Adjust the path to import from package's 'src' directory.
current_dir = os.path.dirname(__file__)
package_base_dir = os.path.join(current_dir, '..')
src_dir = os.path.join(package_base_dir, 'src')
sys.path.insert(0, os.path.abspath(src_dir))

# Import the class to be tested and its dependencies
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry
from openflash.domain import Domain 
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.geometry import ConcentricBodyGroup
from openflash.body import SteppedBody

# --- Fixtures ---

@pytest.fixture
def sample_geometry_params():
    """Provides parameters to create a simple Geometry object."""
    return {
        'h': 100.0,
        'd_values': [20.0, 10.0],
        'a_values': [5.0, 10.0],
        'NMK_values': [5, 5, 5],
        'heaving_values': [False, True], # One flag per body
        'r_coordinates': {'a1': 5.0, 'a2': 10.0},
        'z_coordinates': {'h': 100.0, 'd1': 20.0, 'd2': 10.0}
    }

@pytest.fixture
def mock_geometry(sample_geometry_params):
    """
    Creates a mock Geometry object with essential attributes for MEEMProblem.
    This avoids needing a full, functional Geometry for MEEMProblem tests.
    """
    params = sample_geometry_params
    mock_geo = Mock(spec=Geometry) # Create a mock that looks like a Geometry instance

    # Populate the mock with attributes MEEMProblem expects
    # MEEMProblem expects geometry.domain_list
    mock_geo.domain_list = {
        0: Mock(spec=Domain, number_harmonics=params['NMK_values'][0], h=params['h'], di=params['d_values'][0], a=params['a_values'][0]),
        1: Mock(spec=Domain, number_harmonics=params['NMK_values'][1], h=params['h'], di=params['d_values'][1], a=params['a_values'][1]),
        2: Mock(spec=Domain, number_harmonics=params['NMK_values'][2], h=params['h'], di=None, a=None) # Exterior domain often has None for a, di
    }
    mock_geo.r_coordinates = params['r_coordinates']
    mock_geo.z_coordinates = params['z_coordinates']

    return mock_geo

@pytest.fixture
def real_geometry(sample_geometry_params):
    """Creates a real BasicRegionGeometry object for testing."""
    params = sample_geometry_params
    a_vals = np.array(params['a_values'])
    d_vals = np.array(params['d_values'])
    heaving_vals = np.array(params['heaving_values'])
    
    # 1. Define the physical bodies
    bodies = []
    for i in range(len(a_vals)):
        body = SteppedBody(
            a=np.array([a_vals[i]]),
            d=np.array([d_vals[i]]),
            slant_angle=np.array([0.0]), # Assuming zero slant
            heaving=heaving_vals[i]
        )
        bodies.append(body)
        
    # 2. Create the body arrangement
    arrangement = ConcentricBodyGroup(bodies)
    
    # 3. Instantiate the CONCRETE geometry class
    return BasicRegionGeometry(
        body_arrangement=arrangement,
        h=params['h'],
        NMK=params['NMK_values']
    )


# --- Test Cases ---

def test_meem_problem_initialization(mock_geometry):
    """
    Tests that MEEMProblem initializes correctly with a Geometry object
    and sets default empty arrays for frequencies and modes.
    """
    problem = MEEMProblem(geometry=mock_geometry)

    # Check if geometry and domain_list are assigned from the provided geometry
    assert problem.geometry is mock_geometry
    assert problem.domain_list is mock_geometry.domain_list

    # Check initial state of frequencies and modes
    assert isinstance(problem.frequencies, np.ndarray)
    assert problem.frequencies.size == 0
    assert isinstance(problem.modes, np.ndarray)
    assert problem.modes.size == 0

def test_meem_problem_set_frequencies_modes(mock_geometry):
    """
    Tests that set_frequencies_modes correctly updates the frequencies and modes.
    """
    problem = MEEMProblem(geometry=mock_geometry)

    test_frequencies = np.array([0.5, 1.0, 1.5])
    test_modes = np.array([0, 1, 2, 3])

    problem.set_frequencies_modes(test_frequencies, test_modes)

    # Use np.testing.assert_array_equal for NumPy array comparison
    np.testing.assert_array_equal(problem.frequencies, test_frequencies)
    np.testing.assert_array_equal(problem.modes, test_modes)

    # Test with different arrays
    another_frequencies = np.array([2.0])
    another_modes = np.array([5])
    problem.set_frequencies_modes(another_frequencies, another_modes)
    np.testing.assert_array_equal(problem.frequencies, another_frequencies)
    np.testing.assert_array_equal(problem.modes, another_modes)

def test_meem_problem_with_real_geometry(real_geometry):
    """
    Tests initialization with a real Geometry object, ensuring attributes are correctly linked.
    This acts as an integration check for Geometry dependency.
    """
    problem = MEEMProblem(geometry=real_geometry)

    assert problem.geometry is real_geometry
    assert problem.domain_list is real_geometry.domain_list
    assert len(problem.domain_list) == 3 # Based on the real_geometry fixture setup

    # Check default empty arrays
    assert isinstance(problem.frequencies, np.ndarray)
    assert problem.frequencies.size == 0
    assert isinstance(problem.modes, np.ndarray)
    assert problem.modes.size == 0

    # Set frequencies and modes again to ensure it works with real geometry
    test_frequencies = np.array([10.0])
    test_modes = np.array([10])
    problem.set_frequencies_modes(test_frequencies, test_modes)
    np.testing.assert_array_equal(problem.frequencies, test_frequencies)
    np.testing.assert_array_equal(problem.modes, test_modes)

#  can add more specific tests if MEEMProblem gains more logic in the future,
# such as validation for frequencies/modes, or interactions with the geometry's data.