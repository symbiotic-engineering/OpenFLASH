import pytest
import numpy as np
from openflash.geometry import Geometry

# Mock Domain class if needed (avoid circular import)
from openflash.domain import Domain

def sample_domain_params():
    return [
        {
            'number_harmonics': 10,
            'height': 5.0,
            'radial_width': 1.0,
            'di': 2.0,
            'a': 1.0,
            'heaving': True,
            'slant': False,
            'category': 'inner',
            'top_BC': None,
            'bottom_BC': None
        },
        {
            'number_harmonics': 8,
            'height': 5.0,
            'radial_width': 1.0,
            'di': 2.5,
            'a': 1.0,  # Same 'a' as domain 0 → should be adjacent
            'heaving': False,
            'slant': False,
            'category': 'outer',
            'top_BC': None,
            'bottom_BC': None
        },
        {
            'number_harmonics': 6,
            'height': 5.0,
            'radial_width': None,
            'heaving': False,
            'slant': False,
            'category': 'exterior',
            'top_BC': None,
            'bottom_BC': None
        }
    ]

def test_geometry_initialization():
    r_coords = {'a1': 1.0, 'a2': 2.0}
    z_coords = {'h': 5.0}
    domain_params = sample_domain_params()

    geometry = Geometry(r_coords, z_coords, domain_params)

    assert isinstance(geometry.domain_list, dict)
    assert len(geometry.domain_list) == 3
    assert isinstance(geometry.domain_list[0], Domain)
    assert geometry.domain_list[0].heaving is True
    assert geometry.domain_list[2].category == 'exterior'

def test_missing_z_coordinate_raises():
    r_coords = {'a1': 1.0, 'a2': 2.0}
    z_coords = {}  # Missing 'h'
    domain_params = sample_domain_params()

    with pytest.raises(ValueError, match="z_coordinates must contain key 'h'"):
        Geometry(r_coords, z_coords, domain_params)

def test_missing_di_in_non_exterior_domain():
    bad_params = sample_domain_params()
    del bad_params[0]['di']  # Remove required key

    r_coords = {'a1': 1.0}
    z_coords = {'h': 5.0}

    with pytest.raises(ValueError, match=r"domain_params\[0\] missing required 'di'"):
        Geometry(r_coords, z_coords, bad_params)

def test_adjacency_matrix_behavior():
    r_coords = {'a1': 1.0, 'a2': 2.0}
    z_coords = {'h': 5.0}
    domain_params = sample_domain_params()

    geometry = Geometry(r_coords, z_coords, domain_params)
    adj = geometry.adjacency_matrix

    assert isinstance(adj, np.ndarray)
    assert adj.shape == (3, 3)
    assert adj[0, 1]
    assert adj[1, 0]
    assert not adj[0, 2] # exterior domain has a=None

def test_adjacency_matrix_is_symmetric():
    r_coords = {'a1': 1.0}
    z_coords = {'h': 5.0}
    domain_params = sample_domain_params()

    geometry = Geometry(r_coords, z_coords, domain_params)
    adj = geometry.adjacency_matrix

    assert np.allclose(adj, adj.T)
