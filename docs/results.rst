Results Class
==============

The `Results` class is designed to store results in an `xarray` format similar to Capytaine's conventions. It provides methods to store, access, and export results to a `.nc` file, with a focus on eigenfunction data (radial and vertical) across different frequencies and modes.

Imports
-------

The following libraries are imported:

- `xarray` (as `xr`): Used for managing multi-dimensional arrays and datasets.
- `numpy` (as `np`): Provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.
- `geometry`: The `Geometry` class, which contains domain and body information for the computational model.

Class: `Results`
----------------

### __init__(self, geometry: Geometry, frequencies: np.ndarray, modes: np.ndarray)
Initializes the `Results` class, which will store the eigenfunction results in an `xarray` Dataset.

#### Parameters:
- `geometry` (`Geometry`): A `Geometry` object that contains domain and body information used for setting up the coordinate system.
- `frequencies` (`np.ndarray`): An array of frequency values at which the eigenfunctions are evaluated.
- `modes` (`np.ndarray`): An array of mode shapes or identifiers corresponding to different eigenfunction modes.

---

### store_results(self, domain_index: int, radial_data: np.ndarray, vertical_data: np.ndarray)
Stores the radial and vertical eigenfunction results for a specific domain. The results are stored in an `xarray` Dataset.

#### Parameters:
- `domain_index` (`int`): The index of the domain, corresponding to a key in the `domain_list` of the `Geometry` object.
- `radial_data` (`np.ndarray`): An array of radial eigenfunction values for the given domain.
- `vertical_data` (`np.ndarray`): An array of vertical eigenfunction values for the given domain.

#### Raises:
- `ValueError`: If the domain index is not found in the `domain_list`.

---

### export_to_netcdf(self, file_path: str)
Exports the stored results to a NetCDF (.nc) file.

#### Parameters:
- `file_path` (`str`): The path where the `.nc` file will be saved.

---

### get_results(self)
Returns the stored results as an `xarray.Dataset`.

#### Returns:
- `xarray.Dataset`: The dataset containing the stored eigenfunction results.

---

### display_results(self)
Displays the stored results in a readable format. This method returns a string representation of the results.

#### Returns:
- `str`: A string representation of the results, or a message indicating no results are stored.

---

Conclusion
----------

The `Results` class is designed to facilitate the storage, access, and export of eigenfunction data for computational models. It leverages the power of `xarray` to handle multi-dimensional data arrays, with support for storing results for multiple domains, frequencies, and modes. The class also includes functionality to export the results to NetCDF files for further analysis or sharing.
