import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, MagicMock # Import Mock and MagicMock

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import Package Modules ---
from openflash.meem_problem import MEEMProblem
from openflash.geometry import Geometry, ConcentricBodyGroup # Import necessary geometry classes
from openflash.body import SteppedBody # Import SteppedBody

# ==============================================================================
# Pytest Fixtures
# ==============================================================================
@pytest.fixture
def mock_geometry():
    """
    Creates a mock Geometry object suitable for testing MEEMProblem
    independent of concrete geometry implementations.
    Now includes nested mocks for body_arrangement and bodies.
    """
    # Create mock bodies with a 'heaving' attribute
    mock_body1 = Mock(spec=SteppedBody)
    mock_body1.heaving = True # Example: first body heaves
    mock_body2 = Mock(spec=SteppedBody)
    mock_body2.heaving = False # Example: second body doesn't

    # Create a mock for BodyArrangement containing the mock bodies
    mock_arrangement = Mock(spec=ConcentricBodyGroup)
    mock_arrangement.bodies = [mock_body1, mock_body2]

    # Create the main mock Geometry
    mock_geom = Mock(spec=Geometry)
    mock_geom.domain_list = {0: 'domain0', 1: 'domain1'} # Example domain list
    mock_geom.body_arrangement = mock_arrangement # Attach the nested mock

    return mock_geom

@pytest.fixture
def real_geometry():
    """
    Creates a simple but real BasicRegionGeometry instance for integration testing.
    Uses a 2-body setup where only the SECOND body heaves initially.
    """
    NMK = [5, 5, 5] # Reduced complexity for faster tests
    h = 10.0
    a = np.array([1.0, 2.0])
    d = np.array([2.0, 1.0])
    
    # FIX: Set only the second body (index 1) to heave for this test fixture
    heaving = np.array([0, 1]) # [False, True]

    bodies = []
    for i in range(len(a)):
        body = SteppedBody(
            a=np.array([a[i]]),
            d=np.array([d[i]]),
            slant_angle=np.array([0.0]),
            heaving=bool(heaving[i])
        )
        bodies.append(body)

    arrangement = ConcentricBodyGroup(bodies)
    # Import BasicRegionGeometry here if not already imported globally
    from openflash.basic_region_geometry import BasicRegionGeometry
    geometry = BasicRegionGeometry(arrangement, h, NMK)
    return geometry

# ==============================================================================
# Test Suite for MEEMProblem
# ==============================================================================

def test_meem_problem_initialization(mock_geometry):
    """
    Tests that MEEMProblem initializes correctly with a Geometry object
    and sets default empty arrays for frequencies. Checks inferred modes.
    """
    problem = MEEMProblem(geometry=mock_geometry)

    # Check if geometry and domain_list are assigned
    assert problem.geometry is mock_geometry
    assert problem.domain_list is mock_geometry.domain_list

    # Check initial state of frequencies
    assert isinstance(problem.frequencies, np.ndarray)
    assert problem.frequencies.size == 0
    
    # Check inferred modes based on the mock setup (body 0 heaves)
    assert isinstance(problem.modes, np.ndarray)
    np.testing.assert_array_equal(problem.modes, np.array([0])) # Mock has body 0 heaving
    print("✅ Initialization test passed.")

def test_meem_problem_set_frequencies(mock_geometry): # Renamed test function
    """
    Tests that set_frequencies correctly updates the frequencies.
    Modes are now inferred and not set directly.
    """
    problem = MEEMProblem(geometry=mock_geometry)

    test_frequencies = np.array([0.5, 1.0, 1.5])
    
    # Act: Set only frequencies
    problem.set_frequencies(test_frequencies)

    # Assert frequencies are set
    np.testing.assert_array_equal(problem.frequencies, test_frequencies)
    
    # Assert modes are still correctly inferred (unchanged by set_frequencies)
    np.testing.assert_array_equal(problem.modes, np.array([0])) # Mock still has body 0 heaving
    print("✅ Set frequencies test passed.")

def test_meem_problem_with_real_geometry(real_geometry):
    """
    Tests initialization with a real Geometry object, ensuring attributes
    are correctly linked and modes are inferred correctly.
    """
    problem = MEEMProblem(geometry=real_geometry)

    assert problem.geometry is real_geometry
    assert problem.domain_list is real_geometry.domain_list
    assert len(problem.domain_list) == 3 # Based on the real_geometry fixture setup

    # Check default empty array for frequencies
    assert isinstance(problem.frequencies, np.ndarray)
    assert problem.frequencies.size == 0
    
    # Check that modes are correctly inferred from the real_geometry fixture
    # The fixture sets the second body (index 1) to heave.
    assert isinstance(problem.modes, np.ndarray)
    np.testing.assert_array_equal(problem.modes, np.array([1])) # Expecting mode 1
    assert problem.modes.size == 1 # Check the size explicitly
    print("✅ Real geometry integration test passed.")