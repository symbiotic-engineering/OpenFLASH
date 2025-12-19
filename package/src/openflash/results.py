import xarray as xr
import numpy as np
from openflash.geometry import Geometry
from openflash.meem_problem import MEEMProblem
from .body import CoordinateBody  # <-- ADD THIS IMPORT

class Results:
    """
    Class to store results in an xarray format similar to Capytaine's conventions.
    Provides methods to store, access, and export results to a .nc file.
    """

    def __init__(self, problem: MEEMProblem):
        """
        Initializes the Results class from a MEEMProblem object.
        """
        self.geometry = problem.geometry
        self.frequencies = problem.frequencies

        heaving_bodies = [
            i for i, body in enumerate(self.geometry.body_arrangement.bodies)
            if body.heaving
        ]
        self.modes = np.array(heaving_bodies)

        self.dataset = xr.Dataset(coords={
            'frequency': self.frequencies,
            'mode_i': self.modes,
            'mode_j': self.modes
        })

    def store_results(self, domain_index: int, radial_data: np.ndarray, vertical_data: np.ndarray):
        """
        Store results (e.g., eigenfunctions) for a specific domain.
        This method expects radial_data and vertical_data to already be dimensioned
        as (frequencies, modes, spatial_coords).
        """
        domain = self.geometry.domain_list.get(domain_index)
        if domain is None:
            raise ValueError(f"Domain index {domain_index} not found.")

        # --- THIS IS THE FIX ---
        # You are correct, r_coords and z_coords are on the Body objects.
        # We need to iterate through the bodies in the geometry and collect them.
        all_r_coords = []
        all_z_coords = []
        for body in self.geometry.body_arrangement.bodies:
            if isinstance(body, CoordinateBody):
                all_r_coords.append(body.r_coords)
                all_z_coords.append(body.z_coords)

        # This method is designed for CoordinateBody. If none are found
        # (e.g., if only SteppedBody is used), r_coords will be empty.
        if not all_r_coords:
             r_coords = np.array([])
             z_coords = np.array([])
        else:
             # Concatenate all coordinates from all bodies
             r_coords = np.concatenate(all_r_coords)
             z_coords = np.concatenate(all_z_coords)
        # --- END FIX ---

        if radial_data.shape != (len(self.frequencies), len(self.modes), len(r_coords)):
            raise ValueError(f"radial_data shape {radial_data.shape} does not match expected "
                             f"({len(self.frequencies)}, {len(self.modes)}, {len(r_coords)})")
        if vertical_data.shape != (len(self.frequencies), len(self.modes), len(z_coords)):
            raise ValueError(f"vertical_data shape {vertical_data.shape} does not match expected "
                             f"({len(self.frequencies)}, {len(self.modes)}, {len(z_coords)})")

        domain_name = domain.category if domain.category else f"domain_{domain_index}"

        self.dataset[f'radial_eigenfunctions_{domain_name}'] = xr.DataArray(
            radial_data,
            dims=['frequency', 'modes', 'r'],
            coords={
                'frequency': self.frequencies,
                'modes': self.modes,
                'r': r_coords # These should be constant for the geometry
            }
        )
        self.dataset[f'vertical_eigenfunctions_{domain_name}'] = xr.DataArray(
            vertical_data,
            dims=['frequency', 'modes', 'z'],
            coords={
                'frequency': self.frequencies,
                'modes': self.modes,
                'z': z_coords # These should be constant for the geometry
            }
        )
        print(f"Eigenfunctions for domain {domain_index} stored in dataset.")
    
    def store_single_potential_field(self, potential_data: dict, frequency_idx: int = 0, mode_idx: int = 0):
        """
        Stores a single, fully computed potential field (R, Z, phi) in the dataset.

        :param potential_data: The dictionary returned by `calculate_potentials`.
        :param frequency_idx: The index of the frequency for this data.
        :param mode_idx: The index of the mode for this data.
        """
        if not all(k in potential_data for k in ["R", "Z", "phi"]):
            raise ValueError("potential_data must contain 'R', 'Z', and 'phi' keys.")

        # Store each component of the potential field
        # We add coordinates to make the data self-describing
        self.dataset[f'potential_R_{mode_idx}_{frequency_idx}'] = xr.DataArray(
            potential_data["R"],
            dims=['z_coord', 'r_coord']
        )
        self.dataset[f'potential_Z_{mode_idx}_{frequency_idx}'] = xr.DataArray(
            potential_data["Z"],
            dims=['z_coord', 'r_coord']
        )
        self.dataset[f'potential_phi_real_{mode_idx}_{frequency_idx}'] = xr.DataArray(
            potential_data["phi"].real,
            dims=['z_coord', 'r_coord']
        )
        self.dataset[f'potential_phi_imag_{mode_idx}_{frequency_idx}'] = xr.DataArray(
            potential_data["phi"].imag,
            dims=['z_coord', 'r_coord']
        )
        print(f"Stored single potential field for mode {mode_idx} and frequency {frequency_idx}.")
    
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
        potentials_array = np.full(
            (len(self.frequencies), len(self.modes), len(domain_names), max_harmonics),
            np.nan + 1j * np.nan, # Ensure complex NaNs for fill value
            dtype=complex # Explicitly define complex dtype
        )

        # Create coordinate arrays for r and z
        r_coord_values = np.full((len(domain_names), max_harmonics), np.nan, dtype=float) 
        z_coord_values = np.full((len(domain_names), max_harmonics), np.nan, dtype=float) 


        for item in all_potentials_batch:
            freq_idx = item['frequency_idx']
            mode_idx = item['mode_idx']

            for domain_name, data in item['data'].items():
                domain_idx = domain_names.index(domain_name)
                domain_potentials = data['potentials']
                domain_r_coords = np.concatenate([np.atleast_1d(v) for _, v in sorted(data['r_coords_dict'].items())])
                domain_z_coords = np.concatenate([np.atleast_1d(v) for _, v in sorted(data['z_coords_dict'].items())])

                
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
        # Ensure 'modes' coord is present from __init__
        if 'modes' not in self.dataset.coords:
            self.dataset.coords['modes'] = self.modes

        self.dataset['potentials_real'] = xr.DataArray(
            potentials_array.real,
            dims=['frequency', 'modes', 'domain_name', 'harmonics'],
            coords={
                'frequency': self.frequencies,
                'modes': self.modes,
                'domain_name': domain_names,
                'harmonics': np.arange(max_harmonics)
            }
        )

        self.dataset['potentials_imag'] = xr.DataArray(
            potentials_array.imag,
            dims=['frequency', 'modes', 'domain_name', 'harmonics'],
            coords={
                'frequency': self.frequencies,
                'modes': self.modes,
                'domain_name': domain_names,
                'harmonics': np.arange(max_harmonics)
            }
        )

        
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


    def store_hydrodynamic_coefficients(self, frequencies: np.ndarray,
                                        added_mass_matrix: np.ndarray, damping_matrix: np.ndarray):
        """
        Store hydrodynamic coefficients (added mass and damping).

        :param frequencies: Array of frequency values.
        :param added_mass_matrix: 3D array (frequencies x modes x modes) of added mass coefficients.
        :param damping_matrix: 3D array (frequencies x modes x modes) of damping coefficients.
        """
        # Ensure dimensions match
        expected_shape = (len(frequencies), len(self.modes), len(self.modes))
        
        if added_mass_matrix.shape != expected_shape or \
           damping_matrix.shape != expected_shape:
            raise ValueError(
                f"Matrices must have shape (num_frequencies, num_modes, num_modes). "
                f"Expected {expected_shape} (based on heaving flags), but got {added_mass_matrix.shape}."
            )

        # Assign the 3D data with the correct 3D dimension names
        self.dataset['added_mass'] = (('frequency', 'mode_i', 'mode_j'), added_mass_matrix)
        self.dataset['damping'] = (('frequency', 'mode_i', 'mode_j'), damping_matrix)
        print("Hydrodynamic coefficients stored in xarray dataset.")

    def export_to_netcdf(self, file_path: str):
        """
        Exports the dataset to a NetCDF file.
        Complex values are split into real and imaginary parts for compatibility.
        """
        def _split_complex(ds):
            new_vars = {}
            for var in ds.data_vars:
                data = ds[var].data
                if np.iscomplexobj(data):
                    new_vars[var + "_real"] = (ds[var].dims, np.real(data))
                    new_vars[var + "_imag"] = (ds[var].dims, np.imag(data))
                else:
                    new_vars[var] = ds[var]
            return xr.Dataset(new_vars, attrs=ds.attrs)

        safe_ds = _split_complex(self.dataset)
        safe_ds.to_netcdf(file_path, engine="h5netcdf")

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