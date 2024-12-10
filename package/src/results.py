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
        self.dataset = None  # xarray Dataset to store the results

    def store_results(self, domain_index: int, radial_data: np.ndarray, vertical_data: np.ndarray):
        """
        Store radial and vertical eigenfunction results for a specific domain.

        :param domain_index: Index of the domain. Must correspond to a key in domain_list.
        :param radial_data: Array of radial eigenfunction values.
        :param vertical_data: Array of vertical eigenfunction values.
        """
        # Retrieve the domain object using the index
        domain = self.geometry.domain_list.get(domain_index)
        if domain is None:
            raise ValueError(f"Domain index {domain_index} not found in domain_list.")

        # Create a dictionary of coordinates and data variables for xarray
        coords = {
            'frequencies': self.frequencies,
            'modes': self.modes,
            'r': domain.r_coordinates,  # Coordinates specific to the Domain object
            'z': domain.z_coordinates   # Coordinates specific to the Domain object
        }

        data_vars = {
            'radial_eigenfunctions': (['frequencies', 'modes', 'r'], radial_data),
            'vertical_eigenfunctions': (['frequencies', 'modes', 'z'], vertical_data),
        }

        # Initialize or update the dataset
        if self.dataset is None:
            self.dataset = xr.Dataset(data_vars=data_vars, coords=coords)
        else:
            for var, values in data_vars.items():
                if var in self.dataset:
                    self.dataset[var] = xr.concat(
                        [self.dataset[var], xr.DataArray(values, dims=list(coords.keys()))],
                        dim='r'
                    )
                else:
                    self.dataset[var] = xr.DataArray(values, dims=list(coords.keys()))

    def store_potentials(self, potentials: dict):
        """
        Store potentials in the dataset.

        :param potentials: Dictionary containing potential values and their coordinates.
                        Example format: {'domain_name': {'potentials': ..., 'r': ..., 'z': ...}}
        """
        if self.dataset is None:
            raise ValueError("Dataset not initialized. Store eigenfunctions first.")

        for domain_name, data in potentials.items():
            # Extract potential data and coordinates
            domain_potentials = data['potentials']
            r_coords = data['r']
            z_coords = data['z']

            # Add to dataset with domain-specific dimensions
            self.dataset[f"{domain_name}_potentials"] = xr.DataArray(
                domain_potentials,
                dims=['frequencies', 'modes', 'harmonics'],
                coords={'frequencies': self.frequencies, 'modes': self.modes, 'harmonics': np.arange(len(domain_potentials))}
            )

            # Optionally, store coordinates if not already present
            if f"{domain_name}_r" not in self.dataset.coords:
                self.dataset.coords[f"{domain_name}_r"] = ('harmonics', r_coords)
            if f"{domain_name}_z" not in self.dataset.coords:
                self.dataset.coords[f"{domain_name}_z"] = ('harmonics', z_coords)

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
