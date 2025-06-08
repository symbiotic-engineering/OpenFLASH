import xarray as xr
import numpy as np
from geometry import Geometry

class Results:
    """
    Class to store results in an xarray format similar to Capytaine's conventions.
    Provides methods to store, access, and export results to a .nc file.
    """

    def __init__(self, geometry: Geometry, frequencies: np.ndarray, modes: np.ndarray):
        """
        Initializes the Results class.

        :param geometry: Geometry object that contains the domain and body information.
        :param frequencies: Array of frequency values.
        :param modes: Array of mode shapes or identifiers.
        """
        self.geometry = geometry
        self.frequencies = frequencies
        self.modes = modes
        self.dataset = xr.Dataset()  # xarray Dataset to store the results

    def store_results(self, domain_index: int, radial_data: np.ndarray, vertical_data: np.ndarray):
        """Store results."""
        domain = self.geometry.domain_list.get(domain_index)
        if domain is None:
            raise ValueError(f"Domain index {domain_index} not found.")
        

        r_coords = np.array(list(self.geometry.r_coordinates.values()))
        z_coords = np.array(list(self.geometry.z_coordinates.values()))

        # Use xr.DataArray to explicitly define dimensions
        radial_da = xr.DataArray(
            radial_data,
            dims=['frequencies', 'modes', 'r'],  # Explicit dimensions!
            coords={'frequencies': self.frequencies, 'modes': self.modes, 'r': r_coords}
        )

        vertical_da = xr.DataArray(
            vertical_data,
            dims=['frequencies', 'modes', 'z'],  # Explicit dimensions!
            coords={'frequencies': self.frequencies, 'modes': self.modes, 'z': z_coords}
        )

        if 'radial_eigenfunctions' not in self.dataset: #check for existence of the DataArray
            self.dataset['radial_eigenfunctions'] = radial_da
            self.dataset['vertical_eigenfunctions'] = vertical_da
        else:
            self.dataset['radial_eigenfunctions'] = xr.concat([self.dataset['radial_eigenfunctions'], radial_da], dim='r')
            self.dataset['vertical_eigenfunctions'] = xr.concat([self.dataset['vertical_eigenfunctions'], vertical_da], dim='z')

    def store_potentials(self, potentials: dict):
        """
        Store potentials in the dataset.

        :param potentials: Dictionary containing potential values and their coordinates.
                        Example format: {'domain_name': {'potentials': ..., 'r': ..., 'z': ...}}
        """
        if self.dataset is None:
            raise ValueError("Dataset not initialized. Store eigenfunctions first.")

        domain_names = list(potentials.keys())
        num_domains = len(domain_names)
        max_harmonics = max(len(data['potentials']) for data in potentials.values())

        r_coords_array = np.full((num_domains, max_harmonics), np.nan)  # Use NaN for padding
        z_coords_array = np.full((num_domains, max_harmonics), np.nan)  # Use NaN for padding
        potentials_array = np.full((num_domains, max_harmonics), np.nan, dtype=complex)

        for i, domain_name in enumerate(domain_names):
            data = potentials[domain_name]
            domain_potentials = data['potentials']
            r_coords = data['r']
            z_coords = data['z']

            r_coords_values = np.array(list(r_coords.values()))
            z_coords_values = np.array(list(z_coords.values()))

            # Debugging: Inspect coordinate data
            print(f"Domain: {domain_name}, r_coords length: {len(r_coords)}, z_coords length: {len(z_coords)}, potential length: {len(domain_potentials)}")
            print(f"Domain: {domain_name}, r_coords keys: {list(r_coords.keys())}, z_coords keys: {list(z_coords.keys())}")

            r_coords_array[i, :len(r_coords_values)] = r_coords_values
            z_coords_array[i, :len(z_coords_values)] = z_coords_values
            potentials_array[i, :len(domain_potentials)] = domain_potentials

        self.dataset.coords['domain_r'] = (['domain', 'harmonics'], r_coords_array)
        self.dataset.coords['domain_z'] = (['domain', 'harmonics'], z_coords_array)
        self.dataset.coords['domain'] = domain_names
        self.dataset['domain_potentials'] = (['domain', 'harmonics'], potentials_array)

    def export_to_netcdf(self, file_path: str):
        """
        Export the results to a NetCDF (.nc) file.

        :param file_path: Path where the .nc file will be saved.
        """
        if self.dataset is not None:
            self.dataset.to_netcdf(file_path)
        else:
            print("No results to export!")

    def get_results(self):
        """
        Get the stored results as an xarray Dataset.

        :return: xarray.Dataset containing the results.
        """
        return self.dataset

    def display_results(self):
        """
        Display the stored results in a readable format.

        :return: String representation of the results.
        """
        if self.dataset is not None:
            return str(self.dataset)
        else:
            return "No results stored."
