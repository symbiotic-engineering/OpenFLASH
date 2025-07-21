import pytest
import numpy as np
from openflash.domain import Domain

# Mock Geometry class for testing
class MockGeometry:
    def __init__(self, r_coords, z_coords):
        self.r_coordinates = r_coords
        self.z_coordinates = z_coords

@pytest.fixture
def mock_geometry():
    return MockGeometry(
        r_coords={'a1': 1.0, 'a2': 2.0},
        z_coords={'h': 10.0}
    )

def test_domain_inner(mock_geometry):
    params = {
        'di': 3.0,
        'a': 1.0,
        'h': 10.0,
        'heaving': True,
        'slant': False
    }

    domain = Domain(
        number_harmonics=5,
        height=10.0,
        radial_width=1.0,
        top_BC=None,
        bottom_BC=None,
        category='inner',
        params=params,
        index=0,
        geometry=mock_geometry
    )

    assert domain.number_harmonics == 5
    assert domain.di == 3.0
    assert domain.a == 1.0
    assert domain.h == 10.0
    assert domain.r_coords == 0
    assert domain.z_coords == [0, 10.0]
    assert domain.heaving is True
    assert domain.slant is False
    assert isinstance(domain.scale, float)

def test_domain_outer(mock_geometry):
    params = {
        'di': 3.0,
        'a': 1.5,
        'heaving': False,
    }

    domain = Domain(
        number_harmonics=4,
        height=10.0,
        radial_width=1.0,
        top_BC='Free Surface',
        bottom_BC='Sea Floor',
        category='outer',
        params=params,
        index=1,
        geometry=mock_geometry
    )

    assert domain.r_coords == [1.0, 2.0]
    assert domain.z_coords == [0, 10.0]
    assert domain.top_BC == 'Free Surface'
    assert domain.bottom_BC == 'Sea Floor'
    assert domain.heaving is False

def test_domain_exterior(mock_geometry):
    params = {
        'heaving': True
    }

    domain = Domain(
        number_harmonics=3,
        height=10.0,
        radial_width=None,
        top_BC=None,
        bottom_BC=None,
        category='exterior',
        params=params,
        index=2,
        geometry=mock_geometry
    )

    assert domain.di is None
    assert domain.a is None
    assert domain.r_coords == np.inf
    assert domain.z_coords == [0, 10.0]
