import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def read_and_plot_hydro_results(file_path="meem_hydro_results.nc"):
    """
    Reads hydrodynamic coefficients from a NetCDF file and plots them.

    :param file_path: The path to the NetCDF file generated by meem_engine_example.py.
    """
    print(f"--- Starting NetCDF Results Reader Example ---")
    print(f"Attempting to read file: {file_path}")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Please run meem_engine_example.py first.")
        return

    try:
        # 1. Read the NetCDF file using xarray
        ds = xr.open_dataset(file_path)
        print("\nSuccessfully loaded NetCDF dataset:")
        print(ds) # Print a summary of the dataset (dimensions, coordinates, data variables)

        # 2. Access the stored data variables
        frequencies = ds['frequencies'].values
        added_mass = ds['added_mass'].values
        damping = ds['damping'].values

        # Since 'modes' dimension has size 1, we can squeeze it
        # to get 1D arrays for plotting
        added_mass_squeezed = added_mass.squeeze()
        damping_squeezed = damping.squeeze()

        print(f"\nExtracted Frequencies (first 5): {frequencies[:5]}")
        print(f"Extracted Added Mass (first 5): {added_mass_squeezed[:5]}")
        print(f"Extracted Damping (first 5): {damping_squeezed[:5]}")

        # 3. Re-plot the data to verify
        print("\n--- Re-plotting Hydrodynamic Coefficients from NetCDF Data ---")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Added Mass
        ax1.plot(frequencies, added_mass_squeezed, 'b-x', markersize=5, label='Added Mass (from .nc)')
        ax1.set_title('Hydrodynamic Coefficients for Heaving Cylinders (from NetCDF)')
        ax1.set_ylabel('Added Mass ($A_{33}$)')
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.legend()

        # Plot Damping
        ax2.plot(frequencies, damping_squeezed, 'r-x', markersize=5, label='Damping (from .nc)')
        ax2.set_xlabel(r'Angular Frequency ($\omega$, rad/s)')
        ax2.set_ylabel('Damping ($B_{33}$)')
        ax2.grid(True, linestyle=':', alpha=0.7)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred while reading or processing the NetCDF file: {e}")

    print("\n--- NetCDF Results Reader Example Finished ---")

if __name__ == "__main__":
    # Ensure this script is run from the same directory as meem_engine_example.py
    # or specify the full path to 'meem_hydro_results.nc' if it's elsewhere.
    read_results_example_path = os.path.join(os.path.dirname(__file__), "meem_hydro_results_full.nc")
    read_and_plot_hydro_results(read_results_example_path)