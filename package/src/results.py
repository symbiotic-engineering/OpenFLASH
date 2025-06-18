import xarray as xr
import numpy as np
from geometry import Geometry # Assuming Geometry class is imported correctly

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
        self.dataset = xr.Dataset(coords={
            'frequencies': frequencies,
            'modes': modes
        }) # Initialize with frequencies and modes as coordinates

    def store_results(self, domain_index: int, radial_data: np.ndarray, vertical_data: np.ndarray):
        """
        Store results (e.g., eigenfunctions) for a specific domain.
        This method expects radial_data and vertical_data to already be dimensioned
        as (frequencies, modes, spatial_coords).
        """
        domain = self.geometry.domain_list.get(domain_index)
        if domain is None:
            raise ValueError(f"Domain index {domain_index} not found.")

        # NOTE: The original store_results with xr.concat on 'r'/'z' implies
        # it's meant for adding *more* spatial coordinates over time for a given domain.
        # For batch processing across frequencies/modes, you'd typically want to store
        # the full (frequencies, modes, spatial_coords) array once.
        # For now, let's assume radial_data and vertical_data are pre-batched.

        r_coords = np.array(list(self.geometry.r_coordinates.values()))
        z_coords = np.array(list(self.geometry.z_coordinates.values()))

        # Ensure that passed data matches expected dimensions (frequencies, modes, spatial)
        if radial_data.shape != (len(self.frequencies), len(self.modes), len(r_coords)):
            raise ValueError(f"radial_data shape {radial_data.shape} does not match expected "
                             f"({len(self.frequencies)}, {len(self.modes)}, {len(r_coords)})")
        if vertical_data.shape != (len(self.frequencies), len(self.modes), len(z_coords)):
            raise ValueError(f"vertical_data shape {vertical_data.shape} does not match expected "
                             f"({len(self.frequencies)}, {len(self.modes)}, {len(z_coords)})")

        # Assuming r_coords and z_coords are fixed for all frequencies/modes for this domain
        # If 'r' and 'z' coords vary per domain, this would need more complexity.
        domain_name = domain.category # or a more specific identifier

        self.dataset[f'radial_eigenfunctions_{domain_name}'] = xr.DataArray(
            radial_data,
            dims=['frequencies', 'modes', 'r'],
            coords={
                'frequencies': self.frequencies,
                'modes': self.modes,
                'r': r_coords # These should be constant for the geometry
            }
        )
        self.dataset[f'vertical_eigenfunctions_{domain_name}'] = xr.DataArray(
            vertical_data,
            dims=['frequencies', 'modes', 'z'],
            coords={
                'frequencies': self.frequencies,
                'modes': self.modes,
                'z': z_coords # These should be constant for the geometry
            }
        )
        print(f"Eigenfunctions for domain {domain_index} stored in dataset.")


    # --- METHOD TO STORE BATCHED POTENTIALS ---
    def store_all_potentials(self, all_potentials_batch: list[dict]):
        """
        Store potentials for all frequencies and modes in a structured xarray DataArray.

        :param all_potentials_batch: A list where each element corresponds to a frequency-mode calculation.
                                    Each element is a dictionary:
                                    {'frequency_idx': int, 'mode_idx': int,
                                    'data': {'domain_name': {'potentials': np.ndarray, 'r_coords_dict': dict, 'z_coords_dict': dict}}}
        """
        if not all_potentials_batch:
            print("No potentials data to store.")
            return

        # Determine unique domain names and max harmonics across all batches
        domain_names = sorted(list(set(domain_name for item in all_potentials_batch for domain_name in item['data'].keys())))
        if not domain_names:
            print("No domain data found in potentials batch.")
            return

        # Find the maximum number of harmonics for any potential in any domain/frequency
        max_harmonics = 0
        for item in all_potentials_batch:
            for domain_name in item['data'].keys():
                max_harmonics = max(max_harmonics, len(item['data'][domain_name]['potentials']))

        # Initialize a 4D array: (frequencies, modes, domains, harmonics)
        # IMPORTANT: Use a complex NaN for the fill value to prevent casting warnings
        potentials_array = np.full(
            (len(self.frequencies), len(self.modes), len(domain_names), max_harmonics),
            np.nan + 1j * np.nan, # Ensure complex NaNs for fill value
            dtype=complex # Explicitly define complex dtype
        )

        # Create coordinate arrays for r and z (assuming they are consistent across frequencies/modes
        # but can vary by domain and harmonic within a domain)
        r_coord_values = np.full((len(domain_names), max_harmonics), np.nan, dtype=float) # These should be real floats
        z_coord_values = np.full((len(domain_names), max_harmonics), np.nan, dtype=float) # These should be real floats


        for item in all_potentials_batch:
            freq_idx = item['frequency_idx']
            mode_idx = item['mode_idx']

            for domain_name, data in item['data'].items():
                domain_idx = domain_names.index(domain_name)
                domain_potentials = data['potentials']
                domain_r_coords = np.concatenate([np.atleast_1d(v) for _, v in sorted(data['r_coords_dict'].items())])
                domain_z_coords = np.concatenate([np.atleast_1d(v) for _, v in sorted(data['z_coords_dict'].items())])

                
                # Ensure domain_potentials are complex before assigning
                if domain_potentials.dtype != complex:
                    domain_potentials = domain_potentials.astype(complex)

                potentials_array[freq_idx, mode_idx, domain_idx, :len(domain_potentials)] = domain_potentials
                r_coord_values[domain_idx, :len(domain_r_coords)] = domain_r_coords
                z_coord_values[domain_idx, :len(domain_z_coords)] = domain_z_coords
        
        # Add new coordinates to the dataset if they don't exist
        if 'harmonics' not in self.dataset.coords:
            self.dataset.coords['harmonics'] = np.arange(max_harmonics)
        if 'domain_name' not in self.dataset.coords:
            self.dataset.coords['domain_name'] = domain_names

        self.dataset['potentials_real'] = xr.DataArray(
            potentials_array.real,
            dims=['frequencies', 'modes', 'domain_name', 'harmonics'],
            coords={
                'frequencies': self.frequencies,
                'modes': self.modes,
                'domain_name': domain_names,
                'harmonics': np.arange(max_harmonics)
            }
        )

        self.dataset['potentials_imag'] = xr.DataArray(
            potentials_array.imag,
            dims=['frequencies', 'modes', 'domain_name', 'harmonics'],
            coords={
                'frequencies': self.frequencies,
                'modes': self.modes,
                'domain_name': domain_names,
                'harmonics': np.arange(max_harmonics)
            }
        )

        
        # Store r and z coordinates specific to each domain/harmonic
        # No complex numbers here, so no specific encoding needed beyond default
        self.dataset['potential_r_coords'] = xr.DataArray(
            r_coord_values,
            dims=['domain_name', 'harmonics'],
            coords={'domain_name': domain_names, 'harmonics': np.arange(max_harmonics)}
        )
        self.dataset['potential_z_coords'] = xr.DataArray(
            z_coord_values,
            dims=['domain_name', 'harmonics'],
            coords={'domain_name': domain_names, 'harmonics': np.arange(max_harmonics)}
        )

        print("Potentials stored in xarray dataset (batched across frequencies/modes).")


    def store_hydrodynamic_coefficients(self, frequencies: np.ndarray, modes: np.ndarray,
                                        added_mass_matrix: np.ndarray, damping_matrix: np.ndarray):
        """
        Store hydrodynamic coefficients (added mass and damping).

        :param frequencies: Array of frequency values.
        :param modes: Array of mode identifiers (e.g., [1] for heaving).
        :param added_mass_matrix: 2D array (frequencies x modes) of added mass coefficients.
        :param damping_matrix: 2D array (frequencies x modes) of damping coefficients.
        """
        # Ensure dimensions match for clarity and correctness
        if added_mass_matrix.shape != (len(frequencies), len(modes)) or \
           damping_matrix.shape != (len(frequencies), len(modes)):
            raise ValueError("Added mass and damping matrices must have shape (num_frequencies, num_modes).")

        self.dataset['added_mass'] = xr.DataArray(
            added_mass_matrix,
            dims=['frequencies', 'modes'],
            coords={'frequencies': frequencies, 'modes': modes}
        )
        self.dataset['damping'] = xr.DataArray(
            damping_matrix,
            dims=['frequencies', 'modes'],
            coords={'frequencies': frequencies, 'modes': modes}
        )
        print("Hydrodynamic coefficients stored in xarray dataset.")

    def export_to_netcdf(self, file_path: str):
        """
        Export the results to a NetCDF (.nc) file.

        :param file_path: Path where the .nc file will be saved.
        """
        if self.dataset is not None:
            # Ensure no incompatible complex variable exists
            if 'potentials' in self.dataset.data_vars:
                self.dataset = self.dataset.drop_vars('potentials')

            # Export only the real/imag split (standard-compliant NetCDF)
            self.dataset.to_netcdf(file_path, engine='h5netcdf')
            print(f"Results successfully exported to {file_path} (h5netcdf format).")
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