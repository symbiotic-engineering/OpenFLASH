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
from openflash.geometry import Geometry, ConcentricBodyGroup # Import necessary geometry classes
from openflash.body import SteppedBody # Import SteppedBody

# ==============================================================================
# Mock Geometry Fixture
# ==============================================================================
@pytest.fixture
def mock_geometry():
    """
    Creates a mock Geometry object suitable for testing Results.
    Includes nested mocks for body_arrangement and bodies with heaving flags.
    """
    # Create mock bodies with a 'heaving' attribute
    mock_body1 = Mock(spec=SteppedBody)
    mock_body1.heaving = False # Example: first body doesn't heave
    mock_body2 = Mock(spec=SteppedBody)
    mock_body2.heaving = True # Example: second body heaves
    mock_body3 = Mock(spec=SteppedBody)
    mock_body3.heaving = True # Example: third body heaves

    # Create a mock for BodyArrangement containing the mock bodies
    mock_arrangement = Mock(spec=ConcentricBodyGroup)
    mock_arrangement.bodies = [mock_body1, mock_body2, mock_body3]

    # Create the main mock Geometry
    mock_geom = Mock(spec=Geometry)
    mock_geom.domain_list = {
        0: Mock(category='inner', index=0),
        1: Mock(category='outer', index=1)
    } # Example domain list with Mock domains
    mock_geom.body_arrangement = mock_arrangement # Attach the nested mock
    
    # Mock necessary attributes for store_results_eigenfunctions
    mock_geom.r_coordinates = {'r1': 0.5, 'r2': 1.0}
    mock_geom.z_coordinates = {'z1': -0.5, 'z2': -1.0}


    return mock_geom

# ==============================================================================
# Sample Data Fixtures
# ==============================================================================
@pytest.fixture
def sample_frequencies():
    """Provides a sample array of frequencies."""
    return np.array([0.5, 1.0, 1.5])

# Sample_modes fixture is no longer needed directly by results_instance

# ==============================================================================
# Results Instance Fixture
# ==============================================================================
@pytest.fixture
def results_instance(mock_geometry, sample_frequencies):
    """
    Creates a Results instance using mock geometry and sample frequencies.
    Modes are now inferred by the Results constructor itself.
    """
    # FIX: Call constructor without modes argument
    return Results(mock_geometry, sample_frequencies)

# ==============================================================================
# Test Suite for Results Class
# ==============================================================================

def test_results_initialization(results_instance, mock_geometry, sample_frequencies):
    """
    Tests that the Results class initializes correctly, infers modes,
    and creates an xarray dataset with the right coordinates.
    """
    assert results_instance.geometry is mock_geometry
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
    num_modes = len(results_instance.modes) # Get inferred modes count
    assert num_modes == 2 # Verify mock setup assumption

    # Create sample 3D data (freqs x modes x modes)
    added_mass = np.random.rand(num_freqs, num_modes, num_modes)
    damping = np.random.rand(num_freqs, num_modes, num_modes)

    # FIX: Call method without modes argument
    results_instance.store_hydrodynamic_coefficients(
        frequencies=sample_frequencies,
        added_mass_matrix=added_mass,
        damping_matrix=damping
    )

    # Assert data is stored correctly in the dataset
    assert 'added_mass' in results_instance.dataset
    assert 'damping' in results_instance.dataset
    assert results_instance.dataset['added_mass'].shape == (num_freqs, num_modes, num_modes)
    assert results_instance.dataset['damping'].shape == (num_freqs, num_modes, num_modes)
    np.testing.assert_array_equal(results_instance.dataset['added_mass'].values, added_mass)
    np.testing.assert_array_equal(results_instance.dataset['damping'].values, damping)
    print("✅ Storing hydrodynamic coefficients test passed.")


def test_store_results_eigenfunctions(results_instance, sample_frequencies):
    """
    Tests storing eigenfunction data. Note: The current signature of
    store_results seems incorrect based on typical use cases (mixing modes/space).
    This test follows the current signature but might need revision if the
    method's purpose changes. Assumes 2 modes from fixture.
    """
    num_freqs = len(sample_frequencies)
    num_modes = len(results_instance.modes)
    num_r = 2 # From mock_geometry setup
    num_z = 2 # From mock_geometry setup

    # Create data matching the expected (freqs, modes, spatial) shape
    radial_data = np.random.rand(num_freqs, num_modes, num_r)
    vertical_data = np.random.rand(num_freqs, num_modes, num_z)
    domain_index = 0
    domain_name = f"radial_eigenfunctions_{results_instance.geometry.domain_list[domain_index].category}"
    
    # Mock the geometry attributes needed by the method
    results_instance.geometry.r_coordinates = {'r1': 0.1, 'r2': 0.2}
    results_instance.geometry.z_coordinates = {'z1': -0.1, 'z2': -0.2}


    results_instance.store_results(domain_index, radial_data, vertical_data)

    assert domain_name in results_instance.dataset
    assert results_instance.dataset[domain_name].shape == (num_freqs, num_modes, num_r)
    # Add similar checks for vertical data if needed
    print("✅ Storing eigenfunctions test passed (based on current signature).")


def test_store_all_potentials(results_instance, sample_frequencies):
    """
    Tests storing batched potential coefficient data. Assumes 2 modes.
    """
    num_freqs = len(sample_frequencies)
    num_modes = len(results_instance.modes)
    domain_names = ['inner', 'outer']
    max_harmonics = 5

    # Create realistic batch data structure
    batch_data = []
    for f_idx in range(num_freqs):
        for m_idx in range(num_modes):
            mode_data = {}
            for d_idx, d_name in enumerate(domain_names):
                 # Vary harmonics length for realism
                num_harmonics = max_harmonics - d_idx
                mode_data[d_name] = {
                    "potentials": np.random.rand(num_harmonics) + 1j * np.random.rand(num_harmonics),
                    "r_coords_dict": {f"r{k}": k * 0.1 for k in range(num_harmonics)},
                    "z_coords_dict": {f"z{k}": -k * 0.1 for k in range(num_harmonics)}
                }
            batch_data.append({
                "frequency_idx": f_idx,
                "mode_idx": m_idx, # Use actual mode index for mapping
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
    # Store some sample data first
    num_freqs = len(results_instance.frequencies)
    num_modes = len(results_instance.modes)
    added_mass = np.random.rand(num_freqs, num_modes, num_modes)
    damping = np.random.rand(num_freqs, num_modes, num_modes)
    results_instance.store_hydrodynamic_coefficients(
        results_instance.frequencies, added_mass, damping
    )
    
    # Add complex data to test splitting
    results_instance.dataset['complex_test'] = (('frequency', 'mode_i'), np.random.rand(num_freqs, num_modes) + 1j*np.random.rand(num_freqs, num_modes))


    file_path = tmp_path / "test_results.nc"
    results_instance.export_to_netcdf(file_path)

    assert file_path.exists()

    # Try loading it back to verify format
    loaded_ds = xr.open_dataset(file_path, engine="h5netcdf")
    assert 'added_mass' in loaded_ds
    assert 'damping' in loaded_ds
    assert 'complex_test_real' in loaded_ds # Check if complex data was split
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
    assert ds is results_instance.dataset # Should be the same object
    print("✅ Get results test passed.")

def test_display_results(results_instance):
    """
    Tests that display_results returns a string representation of the dataset.
    """
    # Store some data to make the display meaningful
    num_freqs = len(results_instance.frequencies)
    num_modes = len(results_instance.modes)
    added_mass = np.random.rand(num_freqs, num_modes, num_modes)
    results_instance.store_hydrodynamic_coefficients(
        results_instance.frequencies, added_mass, np.zeros_like(added_mass)
    )

    output_string = results_instance.display_results()
    assert isinstance(output_string, str)
    assert 'xarray.Dataset' in output_string # Basic check for xarray repr
    assert 'added_mass' in output_string
    print("✅ Display results test passed.")