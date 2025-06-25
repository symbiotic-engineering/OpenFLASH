# test_results.py
import pytest
import xarray as xr
import numpy as np
import os
import sys

# --- Path Setup ---
# Adjust the path to import from package's 'src' directory.
current_dir = os.path.dirname(__file__)
package_base_dir = os.path.join(current_dir, '..')
src_dir = os.path.join(package_base_dir, 'src')
sys.path.insert(0, os.path.abspath(src_dir))

# Import the Results class from your results.py
from results import Results

# --- Mock Geometry Class ---
# Since Results depends on Geometry, we'll create a simple mock for testing purposes.
class MockDomain:
    def __init__(self, domain_id, category, number_harmonics, di=0.0, a=0.0, heaving=1.0):
        self.id = domain_id
        self.category = category
        self.number_harmonics = number_harmonics
        self.di = di
        self.a = a
        self.heaving = heaving

class MockGeometry:
    def __init__(self):
        self.r_coordinates = {0: 1.0, 1: 2.0, 2: 3.0}
        self.z_coordinates = {0: -1.0, 1: -2.0, 2: -3.0}
        # Example domains for potentials
        self.domain_list = {
            0: MockDomain(0, 'inner_domain', 5),  # 5 harmonics
            1: MockDomain(1, 'outer_domain', 3),  # 3 harmonics
            2: MockDomain(2, 'interior_domain', 7) # 7 harmonics
        }

# --- Fixtures for Tests ---

@pytest.fixture
def mock_geometry():
    return MockGeometry()

@pytest.fixture
def sample_frequencies():
    return np.array([0.5, 1.0, 1.5])

@pytest.fixture
def sample_modes():
    return np.array([1, 2]) # e.g., heaving, pitching

@pytest.fixture
def results_instance(mock_geometry, sample_frequencies, sample_modes):
    return Results(mock_geometry, sample_frequencies, sample_modes)

# --- Test Functions ---

def test_results_initialization(results_instance, sample_frequencies, sample_modes):
    """Test if Results object is initialized correctly."""
    assert isinstance(results_instance, Results)
    assert isinstance(results_instance.dataset, xr.Dataset)
    assert 'frequencies' in results_instance.dataset.coords
    assert 'modes' in results_instance.dataset.coords
    np.testing.assert_array_equal(results_instance.dataset.coords['frequencies'].values, sample_frequencies)
    np.testing.assert_array_equal(results_instance.dataset.coords['modes'].values, sample_modes)

def test_store_hydrodynamic_coefficients(results_instance, sample_frequencies, sample_modes):
    """Test storing added mass and damping coefficients."""
    num_freq = len(sample_frequencies)
    num_modes = len(sample_modes)

    added_mass_data = np.random.rand(num_freq, num_modes) * 100
    damping_data = np.random.rand(num_freq, num_modes) * 50

    results_instance.store_hydrodynamic_coefficients(
        sample_frequencies, sample_modes, added_mass_data, damping_data
    )

    assert 'added_mass' in results_instance.dataset.data_vars
    assert 'damping' in results_instance.dataset.data_vars

    added_mass_da = results_instance.dataset['added_mass']
    damping_da = results_instance.dataset['damping']

    assert added_mass_da.dims == ('frequencies', 'modes')
    assert damping_da.dims == ('frequencies', 'modes')

    np.testing.assert_array_almost_equal(added_mass_da.values, added_mass_data)
    np.testing.assert_array_almost_equal(damping_da.values, damping_data)

    # Test ValueError for incorrect shape
    with pytest.raises(ValueError, match="matrices must have shape"):
        results_instance.store_hydrodynamic_coefficients(
            sample_frequencies, sample_modes, np.random.rand(num_freq, 1), damping_data
        )

def test_store_results_eigenfunctions(results_instance, mock_geometry, sample_frequencies, sample_modes):
    """Test storing eigenfunctions for a specific domain."""
    num_freq = len(sample_frequencies)
    num_modes = len(sample_modes)
    num_r = len(mock_geometry.r_coordinates)
    num_z = len(mock_geometry.z_coordinates)

    # Test domain 0 (inner_domain)
    domain_idx = 0
    domain_name = mock_geometry.domain_list[domain_idx].category

    radial_data = (np.random.rand(num_freq, num_modes, num_r) + 1j * np.random.rand(num_freq, num_modes, num_r))
    vertical_data = (np.random.rand(num_freq, num_modes, num_z) + 1j * np.random.rand(num_freq, num_modes, num_z))

    results_instance.store_results(domain_idx, radial_data, vertical_data)

    assert f'radial_eigenfunctions_{domain_name}' in results_instance.dataset.data_vars
    assert f'vertical_eigenfunctions_{domain_name}' in results_instance.dataset.data_vars

    rad_ef_da = results_instance.dataset[f'radial_eigenfunctions_{domain_name}']
    vert_ef_da = results_instance.dataset[f'vertical_eigenfunctions_{domain_name}']

    assert rad_ef_da.dims == ('frequencies', 'modes', 'r')
    assert vert_ef_da.dims == ('frequencies', 'modes', 'z')
    assert rad_ef_da.dtype == complex
    assert vert_ef_da.dtype == complex

    np.testing.assert_array_almost_equal(rad_ef_da.values, radial_data)
    np.testing.assert_array_almost_equal(vert_ef_da.values, vertical_data)

    # Test ValueError for incorrect shape
    with pytest.raises(ValueError, match="radial_data shape"):
        results_instance.store_results(domain_idx, np.random.rand(1,1,1), vertical_data)
    
    # Test ValueError for domain not found
    with pytest.raises(ValueError, match="Domain index 999 not found"):
        results_instance.store_results(999, radial_data, vertical_data)


def test_store_all_potentials(results_instance, mock_geometry, sample_frequencies, sample_modes):
    """Test storing batched potentials correctly."""
    num_freq = len(sample_frequencies)
    num_modes = len(sample_modes)

    # Pre-generate static r and z coordinates for each domain and its harmonics
    # These should be consistent across frequencies and modes
    static_domain_coords = {}
    for domain_id, domain in mock_geometry.domain_list.items():
        num_harmonics = domain.number_harmonics
        static_domain_coords[domain.category] = {
            'r_coords_dict': {f'r_h{k}': np.random.rand() * 10 for k in range(num_harmonics)},
            'z_coords_dict': {f'z_h{k}': np.random.rand() * -10 for k in range(num_harmonics)}
        }

    all_potentials_batch = []
    # Simulate data for each frequency and mode
    for f_idx in range(num_freq):
        for m_idx in range(num_modes):
            data_for_freq_mode = {}
            for domain_id, domain in mock_geometry.domain_list.items():
                num_harmonics = domain.number_harmonics
                # Generate dummy complex potentials (these can vary per freq/mode)
                potentials = np.random.rand(num_harmonics) + 1j * np.random.rand(num_harmonics)
                
                # Use the STATIC r and z coordinates
                r_coords_dict = static_domain_coords[domain.category]['r_coords_dict']
                z_coords_dict = static_domain_coords[domain.category]['z_coords_dict']
                
                data_for_freq_mode[domain.category] = {
                    'potentials': potentials,
                    'r_coords_dict': r_coords_dict,
                    'z_coords_dict': z_coords_dict,
                }
            all_potentials_batch.append({
                'frequency_idx': f_idx,
                'mode_idx': m_idx,
                'data': data_for_freq_mode
            })
    
    results_instance.store_all_potentials(all_potentials_batch)

    assert 'potential_r_coords' in results_instance.dataset.data_vars
    assert 'potential_z_coords' in results_instance.dataset.data_vars

    real_da = results_instance.dataset['potentials_real']
    imag_da = results_instance.dataset['potentials_imag']
    potentials_da = real_da + 1j * imag_da
    r_coords_da = results_instance.dataset['potential_r_coords']
    z_coords_da = results_instance.dataset['potential_z_coords']

    # Check dimensions
    expected_domain_names = sorted([d.category for d in mock_geometry.domain_list.values()])
    max_harmonics_in_geom = max(d.number_harmonics for d in mock_geometry.domain_list.values())

    assert potentials_da.dims == ('frequencies', 'modes', 'domain_name', 'harmonics')
    assert r_coords_da.dims == ('domain_name', 'harmonics')
    assert z_coords_da.dims == ('domain_name', 'harmonics')

    # Check coordinates values
    np.testing.assert_array_equal(potentials_da.coords['frequencies'].values, sample_frequencies)
    np.testing.assert_array_equal(potentials_da.coords['modes'].values, sample_modes)
    np.testing.assert_array_equal(potentials_da.coords['domain_name'].values, expected_domain_names)
    np.testing.assert_array_equal(potentials_da.coords['harmonics'].values, np.arange(max_harmonics_in_geom))

    assert real_da.dtype == float
    assert imag_da.dtype == float
    assert r_coords_da.dtype == float
    assert z_coords_da.dtype == float

    # Verify data integrity
    for item in all_potentials_batch:
        f_idx = item['frequency_idx']
        m_idx = item['mode_idx']
        for domain_name, data in item['data'].items():

            original_potentials = data['potentials']
            
            # Retrieve potentials
            retrieved_potentials = potentials_da.sel(
                frequencies=sample_frequencies[f_idx],
                modes=sample_modes[m_idx],
                domain_name=domain_name
            ).values[:len(original_potentials)] # Slice to actual harmonics stored

            np.testing.assert_array_almost_equal(retrieved_potentials, original_potentials)

            # Retrieve r and z coordinates using the xarray domain name directly
            # These should be consistent across freq/modes, so we only need to check them once per domain
            original_r_coords = np.array(list(data['r_coords_dict'].values()))
            original_z_coords = np.array(list(data['z_coords_dict'].values()))

            # Retrieved coordinates should be sliced based on the actual number of harmonics for that domain
            retrieved_r_coords_full_array = r_coords_da.sel(domain_name=domain_name).values
            retrieved_z_coords_full_array = z_coords_da.sel(domain_name=domain_name).values

            retrieved_r_coords = retrieved_r_coords_full_array[:len(original_r_coords)]
            retrieved_z_coords = retrieved_z_coords_full_array[:len(original_z_coords)]

            np.testing.assert_array_almost_equal(retrieved_r_coords, original_r_coords)
            np.testing.assert_array_almost_equal(retrieved_z_coords, original_z_coords)

    # Test with no potentials data (should print a message and not add vars)
    empty_results = Results(mock_geometry, sample_frequencies, sample_modes)
    empty_results.store_all_potentials([])
    assert 'potentials' not in empty_results.dataset.data_vars

    # Test with potentials data that has no domain data (e.g., [{'frequency_idx': 0, 'mode_idx': 0, 'data': {}}])
    empty_domain_data_results = Results(mock_geometry, sample_frequencies, sample_modes)
    empty_domain_data_results.store_all_potentials([{'frequency_idx': 0, 'mode_idx': 0, 'data': {}}])
    assert 'potentials' not in empty_domain_data_results.dataset.data_vars


def test_export_to_netcdf(results_instance, sample_frequencies, sample_modes, mock_geometry):
    """Test exporting the dataset to a NetCDF file."""
    # Populate with some data first
    num_freq = len(sample_frequencies)
    num_modes = len(sample_modes)
    
    # Hydro coeffs
    added_mass_data = np.random.rand(num_freq, num_modes)
    damping_data = np.random.rand(num_freq, num_modes)
    results_instance.store_hydrodynamic_coefficients(sample_frequencies, sample_modes, added_mass_data, damping_data)

    # Eigenfunctions
    domain_idx_ef = 0
    domain_name_ef = mock_geometry.domain_list[domain_idx_ef].category
    num_r = len(mock_geometry.r_coordinates)
    num_z = len(mock_geometry.z_coordinates)
    radial_data_ef = (np.random.rand(num_freq, num_modes, num_r) + 1j * np.random.rand(num_freq, num_modes, num_r))
    vertical_data_ef = (np.random.rand(num_freq, num_modes, num_z) + 1j * np.random.rand(num_freq, num_modes, num_z))
    results_instance.store_results(domain_idx_ef, radial_data_ef, vertical_data_ef)

    # Potentials
    all_potentials_batch = []
    f_idx = 0 # Just one frequency/mode for simplicity in export test
    m_idx = 0
    data_for_freq_mode = {}
    domain_id_pot = 0
    domain_pot = mock_geometry.domain_list[domain_id_pot]
    num_harmonics_pot = domain_pot.number_harmonics
    potentials_val = np.random.rand(num_harmonics_pot) + 1j * np.random.rand(num_harmonics_pot)
    r_coords_dict_pot = {f'r_h{k}': np.random.rand() * 10 for k in range(num_harmonics_pot)}
    z_coords_dict_pot = {f'z_h{k}': np.random.rand() * -10 for k in range(num_harmonics_pot)}
    data_for_freq_mode[domain_pot.category] = {
        'potentials': potentials_val,
        'r_coords_dict': r_coords_dict_pot,
        'z_coords_dict': z_coords_dict_pot,
    }
    all_potentials_batch.append({
        'frequency_idx': f_idx,
        'mode_idx': m_idx,
        'data': data_for_freq_mode
    })
    results_instance.store_all_potentials(all_potentials_batch)


    file_path = "test_results.nc"
    if os.path.exists(file_path):
        os.remove(file_path)

    results_instance.export_to_netcdf(file_path)

    assert os.path.exists(file_path)

    # Verify content after reloading
    loaded_dataset = xr.open_dataset(file_path)

    assert 'added_mass' in loaded_dataset.data_vars
    assert 'damping' in loaded_dataset.data_vars
    assert f'radial_eigenfunctions_{domain_name_ef}' in loaded_dataset.data_vars
    assert f'vertical_eigenfunctions_{domain_name_ef}' in loaded_dataset.data_vars
    assert 'potential_r_coords' in loaded_dataset.data_vars
    assert 'potential_z_coords' in loaded_dataset.data_vars
    
    # Assert that the data is numerically correct (this covers the main concern)
    # The dtype assertion can be more specific
    assert loaded_dataset[f'radial_eigenfunctions_{domain_name_ef}'].dtype.kind == 'V' or loaded_dataset[f'radial_eigenfunctions_{domain_name_ef}'].dtype == complex

    # Verify data integrity (e.g., potentials) - this is the most important part
    loaded_real = loaded_dataset['potentials_real'].sel(
        frequencies=sample_frequencies[f_idx],
        modes=sample_modes[m_idx],
        domain_name=domain_pot.category
    ).values[:len(potentials_val)]

    loaded_imag = loaded_dataset['potentials_imag'].sel(
        frequencies=sample_frequencies[f_idx],
        modes=sample_modes[m_idx],
        domain_name=domain_pot.category
    ).values[:len(potentials_val)]

    loaded_potentials = loaded_real + 1j * loaded_imag

    if loaded_potentials.dtype.names is not None and 'r' in loaded_potentials.dtype.names and 'i' in loaded_potentials.dtype.names:
        loaded_potentials_combined = loaded_potentials['r'] + 1j * loaded_potentials['i']
    else:
        loaded_potentials_combined = loaded_potentials

    np.testing.assert_array_almost_equal(loaded_potentials_combined, potentials_val)

    # Do the same for eigenfunctions
    loaded_radial_ef = loaded_dataset[f'radial_eigenfunctions_{domain_name_ef}'].sel(
        frequencies=sample_frequencies[0],
        modes=sample_modes[0],
        r=mock_geometry.r_coordinates[0] # Select a specific r-coordinate for check
    ).values

    # Cleanup
    os.remove(file_path)

def test_get_results(results_instance):
    """Test get_results method."""
    dataset = results_instance.get_results()
    assert isinstance(dataset, xr.Dataset)
    # Further checks could involve specific data variables if populated

def test_display_results(results_instance):
    """Test display_results method."""
    display_str = results_instance.display_results()
    assert isinstance(display_str, str)
    assert "xarray.Dataset" in display_str

    empty_results = Results(MockGeometry(), np.array([]), np.array([]))
    empty_display_str = empty_results.display_results()
    assert "No results stored." not in empty_display_str # Initial dataset is empty, but not None
    assert "Dimensions" in empty_display_str # It will show empty dimensions

    # If dataset was explicitly set to None, then 'No results stored.' would appear
    results_instance.dataset = None
    assert results_instance.display_results() == "No results stored."