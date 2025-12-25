import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, MagicMock
import os
import sys

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import Package Modules ---
from openflash.results import Results
from openflash.geometry import Geometry, ConcentricBodyGroup
from openflash.body import SteppedBody, CoordinateBody
from openflash.meem_problem import MEEMProblem

# ==============================================================================
# Mock Geometry Fixture
# ==============================================================================
@pytest.fixture
def mock_geometry():
    """
    Creates a mock Geometry object suitable for testing Results.
    Includes nested mocks for body_arrangement and bodies with heaving flags.
    """
    # --- FIX: Ensure bodies have coordinate attributes for store_results ---
    # We use specs for both SteppedBody (default) and add CoordinateBody attrs
    # so the checks in store_results pass.
    
    mock_body1 = Mock(spec=CoordinateBody) 
    mock_body1.heaving = False
    mock_body1.r_coords = np.array([0.5]) # 1 coordinate
    mock_body1.z_coords = np.array([-0.5])

    mock_body2 = Mock(spec=CoordinateBody)
    mock_body2.heaving = True
    mock_body2.r_coords = np.array([1.0]) # 1 coordinate
    mock_body2.z_coords = np.array([-1.0])

    mock_body3 = Mock(spec=CoordinateBody)
    mock_body3.heaving = True
    # If we want total coords to be 2 (as in the failing test),
    # body3 can have empty coords or we adjust the test data size.
    # Let's make total coords = 2 by giving body3 empty arrays.
    mock_body3.r_coords = np.array([])
    mock_body3.z_coords = np.array([])

    mock_arrangement = Mock(spec=ConcentricBodyGroup)
    mock_arrangement.bodies = [mock_body1, mock_body2, mock_body3]

    mock_geom = Mock(spec=Geometry)
    mock_geom.domain_list = {
        0: Mock(category='inner', index=0),
        1: Mock(category='outer', index=1)
    }
    mock_geom.body_arrangement = mock_arrangement
    
    return mock_geom

# ==============================================================================
# Sample Data Fixtures
# ==============================================================================
@pytest.fixture
def sample_frequencies():
    """Provides a sample array of frequencies."""
    return np.array([0.5, 1.0, 1.5])

# ==============================================================================
# Mock Problem Fixture
# ==============================================================================
@pytest.fixture
def mock_problem(mock_geometry, sample_frequencies):
    """ Creates a mock MEEMProblem containing mock geometry and frequencies. """
    problem = Mock(spec=MEEMProblem)
    problem.geometry = mock_geometry
    problem.frequencies = sample_frequencies
    # We don't need to mock problem.modes because Results infers it directly
    # from problem.geometry
    return problem

# ==============================================================================
# Results Instance Fixture
# ==============================================================================
@pytest.fixture
def results_instance(mock_problem): 
    """
    Creates a Results instance using a mock MEEMProblem object.
    """
    return Results(mock_problem)

# ==============================================================================
# Test Suite for Results Class
# ==============================================================================

def test_results_initialization(results_instance, mock_problem, sample_frequencies): 
    """
    Tests that the Results class initializes correctly, infers modes,
    and creates an xarray dataset with the right coordinates.
    """
    assert results_instance.geometry is mock_problem.geometry 
    np.testing.assert_array_equal(results_instance.frequencies, sample_frequencies)

    # Check inferred modes based on mock_geometry (bodies 1 and 2 heave)
    expected_modes = np.array([1, 2])
    assert isinstance(results_instance.modes, np.ndarray)
    np.testing.assert_array_equal(results_instance.modes, expected_modes)

    # Check the initialized dataset
    assert isinstance(results_instance.dataset, xr.Dataset)
    assert 'frequency' in results_instance.dataset.coords
    assert 'mode_i' in results_instance.dataset.coords
    assert 'mode_j' in results_instance.dataset.coords
    np.testing.assert_array_equal(results_instance.dataset.coords['frequency'], sample_frequencies)
    np.testing.assert_array_equal(results_instance.dataset.coords['mode_i'], expected_modes)
    np.testing.assert_array_equal(results_instance.dataset.coords['mode_j'], expected_modes)
    print("✅ Initialization and mode inference test passed.")

def test_store_hydrodynamic_coefficients(results_instance, sample_frequencies):
    """
    Tests storing hydrodynamic coefficients. Assumes 2 modes from fixture.
    """
    num_freqs = len(sample_frequencies)
    num_modes = len(results_instance.modes)
    assert num_modes == 2

    added_mass = np.random.rand(num_freqs, num_modes, num_modes)
    damping = np.random.rand(num_freqs, num_modes, num_modes)

    results_instance.store_hydrodynamic_coefficients(
        frequencies=sample_frequencies,
        added_mass_matrix=added_mass,
        damping_matrix=damping
    )

    assert 'added_mass' in results_instance.dataset
    assert 'damping' in results_instance.dataset
    assert results_instance.dataset['added_mass'].shape == (num_freqs, num_modes, num_modes)
    assert results_instance.dataset['damping'].shape == (num_freqs, num_modes, num_modes)
    np.testing.assert_array_equal(results_instance.dataset['added_mass'].values, added_mass)
    np.testing.assert_array_equal(results_instance.dataset['damping'].values, damping)
    print("✅ Storing hydrodynamic coefficients test passed.")


def test_store_results_eigenfunctions(results_instance, sample_frequencies):
    """
    Tests storing eigenfunction data. Assumes 2 modes from fixture.
    """
    num_freqs = len(sample_frequencies)
    num_modes = len(results_instance.modes)
    
    # Matches the mock geometry bodies (body1 has 1 coord, body2 has 1, body3 has 0)
    # Total coords = 1 + 1 + 0 = 2
    num_r = 2 
    num_z = 2 

    radial_data = np.random.rand(num_freqs, num_modes, num_r)
    vertical_data = np.random.rand(num_freqs, num_modes, num_z)
    domain_index = 0
    domain_name = f"radial_eigenfunctions_{results_instance.geometry.domain_list[domain_index].category}"

    results_instance.store_results(domain_index, radial_data, vertical_data)

    assert domain_name in results_instance.dataset
    assert results_instance.dataset[domain_name].shape == (num_freqs, num_modes, num_r)
    print("✅ Storing eigenfunctions test passed.")


def test_store_all_potentials(results_instance, sample_frequencies):
    """
    Tests storing batched potential coefficient data. Assumes 2 modes.
    """
    num_freqs = len(sample_frequencies)
    num_modes = len(results_instance.modes)
    domain_names = ['inner', 'outer']
    max_harmonics = 5

    batch_data = []
    for f_idx in range(num_freqs):
        for m_idx in range(num_modes):
            mode_data = {}
            for d_idx, d_name in enumerate(domain_names):
                num_harmonics = max_harmonics - d_idx
                mode_data[d_name] = {
                    "potentials": np.random.rand(num_harmonics) + 1j * np.random.rand(num_harmonics),
                    "r_coords_dict": {f"r{k}": k * 0.1 for k in range(num_harmonics)},
                    "z_coords_dict": {f"z{k}": -k * 0.1 for k in range(num_harmonics)}
                }
            batch_data.append({
                "frequency_idx": f_idx,
                "mode_idx": m_idx,
                "data": mode_data
            })

    results_instance.store_all_potentials(batch_data)

    assert 'potentials_real' in results_instance.dataset
    assert 'potentials_imag' in results_instance.dataset
    expected_shape = (num_freqs, num_modes, len(domain_names), max_harmonics)
    assert results_instance.dataset['potentials_real'].shape == expected_shape
    assert results_instance.dataset['potentials_imag'].shape == expected_shape
    assert 'potential_r_coords' in results_instance.dataset
    assert 'potential_z_coords' in results_instance.dataset
    print("✅ Storing all potentials test passed.")


def test_export_to_netcdf(results_instance, tmp_path):
    """
    Tests exporting the dataset to a NetCDF file.
    Includes storing some data first.
    """
    num_freqs = len(results_instance.frequencies)
    num_modes = len(results_instance.modes)
    added_mass = np.random.rand(num_freqs, num_modes, num_modes)
    damping = np.random.rand(num_freqs, num_modes, num_modes)
    results_instance.store_hydrodynamic_coefficients(
        results_instance.frequencies, added_mass, damping
    )
    # Adding complex test data to verify splitting
    results_instance.dataset['complex_test'] = (('frequency', 'mode_i'), np.random.rand(num_freqs, num_modes) + 1j*np.random.rand(num_freqs, num_modes))

    file_path = tmp_path / "test_results.nc"
    results_instance.export_to_netcdf(file_path)

    assert file_path.exists()

    loaded_ds = xr.open_dataset(file_path, engine="h5netcdf")
    assert 'added_mass' in loaded_ds
    assert 'damping' in loaded_ds
    assert 'complex_test_real' in loaded_ds
    assert 'complex_test_imag' in loaded_ds
    assert loaded_ds['added_mass'].shape == (num_freqs, num_modes, num_modes)
    loaded_ds.close()
    print("✅ Export to NetCDF test passed.")

def test_get_results(results_instance):
    """
    Tests that get_results returns the underlying xarray Dataset.
    """
    ds = results_instance.get_results()
    assert isinstance(ds, xr.Dataset)
    assert ds is results_instance.dataset
    print("✅ Get results test passed.")

def test_display_results(results_instance):
    """
    Tests that display_results returns a string representation of the dataset.
    """
    num_freqs = len(results_instance.frequencies)
    num_modes = len(results_instance.modes)
    added_mass = np.random.rand(num_freqs, num_modes, num_modes)
    results_instance.store_hydrodynamic_coefficients(
        results_instance.frequencies, added_mass, np.zeros_like(added_mass)
    )

    output_string = results_instance.display_results()
    assert isinstance(output_string, str)
    assert 'xarray.Dataset' in output_string
    assert 'added_mass' in output_string
    print("✅ Display results test passed.")

# ==============================================================================
# NEW: Error Handling and Edge Case Tests
# ==============================================================================

def test_store_results_domain_not_found(results_instance):
    """
    Coverage for: raise ValueError(f"Domain index {domain_index} not found.")
    """
    with pytest.raises(ValueError, match="Domain index 999 not found"):
        results_instance.store_results(999, np.array([]), np.array([]))

def test_store_results_no_coordinate_body(mock_problem, sample_frequencies):
    """
    Coverage for: 
    r_coords = np.array([])
    z_coords = np.array([])
    """
    # Replace bodies with just SteppedBody (not CoordinateBody)
    # FIX: Ensure 'heaving' attribute exists on the mock BEFORE passing it to Results,
    # as Results.__init__ accesses body.heaving immediately.
    stepped_body_mock = Mock(spec=SteppedBody)
    stepped_body_mock.heaving = False # Set heaving to False (or True) to allow access
    
    mock_problem.geometry.body_arrangement.bodies = [stepped_body_mock] 
    
    results = Results(mock_problem)
    
    num_freqs = len(sample_frequencies)
    
    # Expected shape: (freqs, modes, 0)
    # Modes list will be empty if heaving=False for all bodies.
    num_modes = len(results.modes)
    radial_data = np.zeros((num_freqs, num_modes, 0))
    vertical_data = np.zeros((num_freqs, num_modes, 0))
    
    # Should run without error
    results.store_results(0, radial_data, vertical_data)

def test_store_results_shape_mismatch(results_instance):
    """
    Coverage for:
    raise ValueError(f"radial_data shape ...")
    raise ValueError(f"vertical_data shape ...")
    """
    # Default fixture has 3 bodies, 2 coordinate bodies, total 2 coordinates.
    # Expected spatial dim is 2.
    num_freqs = len(results_instance.frequencies)
    num_modes = len(results_instance.modes)
    expected_spatial = 2
    
    # Create mismatched data
    bad_data = np.zeros((num_freqs, num_modes, expected_spatial + 1))
    good_data = np.zeros((num_freqs, num_modes, expected_spatial))
    
    with pytest.raises(ValueError, match="radial_data shape"):
        results_instance.store_results(0, bad_data, good_data)
        
    with pytest.raises(ValueError, match="vertical_data shape"):
        results_instance.store_results(0, good_data, bad_data)

def test_store_single_potential_field(results_instance):
    """
    Coverage for:
    raise ValueError("potential_data must contain 'R', 'Z', and 'phi' keys.")
    and the successful execution block.
    """
    # Test Exception
    with pytest.raises(ValueError, match="must contain 'R', 'Z', and 'phi'"):
        results_instance.store_single_potential_field({'R': []})
        
    # Test Success
    data = {
        'R': np.zeros((5, 5)),
        'Z': np.zeros((5, 5)),
        'phi': np.zeros((5, 5), dtype=complex)
    }
    results_instance.store_single_potential_field(data, frequency_idx=0, mode_idx=0)
    assert 'potential_phi_real_0_0' in results_instance.dataset

def test_store_all_potentials_edge_cases(results_instance, capsys):
    """
    Coverage for:
    print("No potentials data to store.")
    print("No domain data found in potentials batch.")
    domain_potentials = domain_potentials.astype(complex)
    """
    # 1. Empty Input
    results_instance.store_all_potentials([])
    captured = capsys.readouterr()
    assert "No potentials data to store" in captured.out
    
    # 2. Batch with no domain data
    results_instance.store_all_potentials([{'data': {}}])
    captured = capsys.readouterr()
    assert "No domain data found" in captured.out
    
    # 3. Real input triggering astype(complex)
    batch = [{
        'frequency_idx': 0, 'mode_idx': 0,
        'data': {
            'domain_0': {
                'potentials': np.array([1.0, 2.0]), # Floats, not complex
                'r_coords_dict': {'r0': 0},
                'z_coords_dict': {'z0': 0}
            }
        }
    }]
    results_instance.store_all_potentials(batch)
    # Verification: Check that the stored data is complex/contains NaNs where appropriate
    assert 'potentials_real' in results_instance.dataset

def test_store_hydrodynamic_coefficients_shape_mismatch(results_instance, sample_frequencies):
    """
    Coverage for:
    raise ValueError(f"Matrices must have shape ...")
    """
    bad_matrix = np.zeros((1, 1, 1)) # Wrong shape
    with pytest.raises(ValueError, match="Matrices must have shape"):
        results_instance.store_hydrodynamic_coefficients(sample_frequencies, bad_matrix, bad_matrix)

def test_display_results_none(results_instance):
    """
    Coverage for:
    else: return "No results stored."
    """
    results_instance.dataset = None
    assert results_instance.display_results() == "No results stored."